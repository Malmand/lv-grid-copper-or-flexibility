"""Vectorised AC power flow for radial low-voltage feeders: a whole year of 15-min steps at once.

pandapower solves one operating point per call (about 0.1 s on these feeders), far too slow for
35,000 steps x pathways x years x scenarios. Here the bus admittance matrix is taken from pandapower
once, and the fixed-point (Z-bus) iteration

    V_L <- Z_LL conj(S_L / V_L) + w,     Z_LL = inv(Y_LL),  w = -Z_LL Y_LS V_S

is applied to all time steps together (S_L is a T x n matrix of constant-power injections). The
results are checked against pandapower in tests/test_powerflow.py.
"""
from dataclasses import dataclass

import numpy as np
import pandapower as pp


@dataclass
class Feeder:
    net: object              # pandapower net (topology and types)
    lv_buses: np.ndarray     # pandapower bus indices of the LV network (in matrix order)
    Z: np.ndarray            # Z_LL (pu)
    w: np.ndarray            # no-load voltages for V_S = 1 (pu)
    Y_SL: np.ndarray         # coupling of the slack bus to the LV buses
    Y_SS: complex
    base_mva: float
    line_from: np.ndarray    # matrix positions of line ends
    line_to: np.ndarray
    line_y: np.ndarray       # series admittance of each line (pu, with parallel cables)
    line_imax_pu: np.ndarray # thermal limit (pu current)
    trafo_sn_mva: float
    busbar: int              # matrix position of the transformer LV terminal
    pos: dict                # pandapower bus -> matrix position (fused buses share one)


def build(net):
    """Matrices of the feeder from pandapower (one power flow to set up the internal model)."""
    pp.runpp(net, numba=True)
    ppc = net._ppc
    Y = ppc["internal"]["Ybus"].toarray()
    lookup = net._pd2ppc_lookups["bus"]
    slack_pd = int(net.ext_grid.bus.iloc[0])
    s = lookup[slack_pd]
    lv_pd = net.bus.index[(net.bus.vn_kv < 1.0) & net.bus.in_service].to_numpy()
    lv_ppc = lookup[lv_pd]
    # buses fused by bus-bus switches share one internal bus: keep one pandapower bus per internal bus
    uniq, first = np.unique(lv_ppc, return_index=True)
    order = np.argsort(first)
    pos_ppc = uniq[order]
    lv_buses = lv_pd[np.sort(first)]
    Y_LL = Y[np.ix_(pos_ppc, pos_ppc)]
    Y_LS = Y[pos_ppc, s]
    Z = np.linalg.inv(Y_LL)
    w = -Z @ Y_LS
    base = ppc["baseMVA"]
    where = {b: i for i, b in enumerate(pos_ppc)}
    vn = net.bus.vn_kv
    lf, lt, ly, lim = [], [], [], []
    for _, ln in net.line[net.line.in_service].iterrows():
        f, t = where[lookup[ln.from_bus]], where[lookup[ln.to_bus]]
        zb = vn[ln.from_bus] ** 2 / base
        z = (ln.r_ohm_per_km + 1j * ln.x_ohm_per_km) * ln.length_km / ln.parallel / zb
        ib = base / (np.sqrt(3) * vn[ln.from_bus])          # kA
        lf.append(f)
        lt.append(t)
        ly.append(1 / z)
        lim.append(ln.max_i_ka * ln.parallel * ln.df / ib)
    tr = net.trafo.iloc[0]
    busbar = where[lookup[tr.lv_bus]]
    pos = {int(b): where[lookup[b]] for b in lv_pd}
    return Feeder(net, lv_buses, Z, w, Y[s, pos_ppc], Y[s, s], base, np.array(lf), np.array(lt),
                  np.array(ly), np.array(lim), float(tr.sn_mva), busbar, pos)


def solve(fd: Feeder, S, v_slack=1.0, tol=1e-8, max_iter=60):
    """S: (T, n) complex power injections in MW/Mvar at the LV buses (generation positive).
    v_slack: scalar or (T,) slack voltage magnitude (an on-load tap changer is modelled through it).
    Returns V (T, n) in pu."""
    Spu = np.asarray(S) / fd.base_mva
    vs = np.broadcast_to(np.asarray(v_slack, float), (Spu.shape[0],))[:, None]
    w = fd.w[None, :] * vs
    V = w.copy()
    ZT = fd.Z.T
    for _ in range(max_iter):
        Vn = np.conj(Spu / V) @ ZT + w
        if np.max(np.abs(Vn - V)) < tol:
            return Vn
        V = Vn
    raise RuntimeError("power flow did not converge")


def flows(fd: Feeder, V, v_slack=1.0):
    """Line loading (T, n_lines, as a share of the limit), line losses (T,) MW, transformer
    loading (T,) as a share of its rating, and total losses incl. the transformer (T,) MW."""
    I = (V[:, fd.line_from] - V[:, fd.line_to]) * fd.line_y[None, :]
    line_loading = np.abs(I) / fd.line_imax_pu[None, :]
    line_loss = np.sum(np.abs(I) ** 2 * np.real(1 / fd.line_y)[None, :], axis=1) * fd.base_mva
    vs = np.broadcast_to(np.asarray(v_slack, float), (V.shape[0],))
    # complex power drawn from the slack bus: S = V_s conj(Y_SS V_s + Y_SL V)
    S_hv = vs * np.conj(fd.Y_SS * vs + V @ fd.Y_SL)
    trafo_loading = np.abs(S_hv) * fd.base_mva / fd.trafo_sn_mva / vs
    return line_loading, line_loss, trafo_loading, S_hv * fd.base_mva
