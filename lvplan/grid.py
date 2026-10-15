"""Operating a feeder for one year: injections, power flow, limits, flexibility, battery, tap changer."""
from dataclasses import dataclass

import numpy as np
from numba import njit

from . import powerflow as pf

DT_H = 0.25
TRAFO_SIZES = [0.16, 0.25, 0.4, 0.63, 0.8, 1.0, 1.26]     # MVA, distribution transformer series


@dataclass
class Limits:
    vmin: float
    vmax: float
    thermal: float = 1.0


@dataclass
class YearResult:
    V: np.ndarray            # (T, n) |V| pu
    line_loading: np.ndarray
    trafo_loading: np.ndarray
    losses_mw: np.ndarray    # (T,)
    viol_steps: np.ndarray   # (T,) bool: any limit violated
    curtailed_mwh: float = 0.0
    battery_throughput_mwh: float = 0.0
    tap: np.ndarray = None
    S: np.ndarray = None            # injections actually used (after curtailment and battery)
    v_slack: object = 1.0

    def summary(self):
        return dict(viol_hours=float(self.viol_steps.sum() * DT_H),
                    v_min=float(self.V.min()), v_max=float(self.V.max()),
                    line_max=float(self.line_loading.max()), trafo_max=float(self.trafo_loading.max()),
                    losses_mwh=float(self.losses_mw.sum() * DT_H), curtailed_mwh=self.curtailed_mwh,
                    battery_throughput_mwh=self.battery_throughput_mwh)


def injection_matrix(fd, P, devices=None, device_P=None):
    """S (T, n) in MW/Mvar, generation positive. P: bus -> (P_load, Q_load, P_gen) from scenarios.
    device_P optionally replaces the active power of the controllable devices (curtailment)."""
    T = max(len(np.atleast_1d(v)) for vals in P.values() for v in vals)  # entries may be scalars or series
    n = len(fd.lv_buses)
    S = np.zeros((T, n), complex)
    for b, (pl, ql, pg) in P.items():
        S[:, fd.pos[int(b)]] += pg - pl - 1j * ql  # fused buses share one position
    if device_P is not None:
        for (tech, b, d), p_new in zip(devices, device_P):
            S[:, fd.pos[int(b)]] += d.P - p_new          # put back the original, take the new
    return S


def check(fd, V, limits, v_slack=1.0):
    ll, loss_lines, tl, S_hv = pf.flows(fd, V, v_slack)
    Vm = np.abs(V)
    viol = ((Vm < limits.vmin) | (Vm > limits.vmax)).any(axis=1) | (ll > limits.thermal).any(axis=1) \
        | (tl > limits.thermal)
    return Vm, ll, tl, S_hv, viol


def run_year(fd, P, limits, devices=(), flex=None, battery=None, ront=None):
    """One year of operation.
    flex: dict(floor_mw) -> paragraph 14a: in every step with a violation, each controllable device is
          limited to floor_mw; the energy it could not take is delivered in the next steps without a
          violation, at up to its own rating (EV charging and heat pumps shift, they do not lose energy).
    battery: dict(e_mwh, p_mw) at the transformer busbar, shaving transformer loading to 95 %.
    ront: dict(step, n) on-load tap changer at the transformer: per step the tap that keeps all
          voltages inside the band with the largest margin."""
    S = injection_matrix(fd, P)
    v_slack = 1.0
    tap = None
    V = pf.solve(fd, S)
    if ront is not None:
        v_slack, tap = _choose_tap(fd, S, V, limits, ront)
        V = pf.solve(fd, S, v_slack)
    Vm, ll, tl, S_hv, viol = check(fd, V, limits, v_slack)
    curtailed = 0.0
    if flex is not None and len(devices) and viol.any():
        for _ in range(3):                       # curtail, re-solve, repeat if new violations appear
            dev_P, curtailed = _curtail(devices, viol, flex["floor_mw"])
            S = injection_matrix(fd, P, devices, dev_P)
            if ront is not None:                 # the tap changer follows the curtailed flows
                v_slack, tap = _choose_tap(fd, S, pf.solve(fd, S), limits, ront)
            V = pf.solve(fd, S, v_slack)
            Vm, ll, tl, S_hv, viol_new = check(fd, V, limits, v_slack)
            if not (viol_new & ~viol).any():  # rebound created no new violations
                viol = viol_new
                break
            viol = viol | viol_new
        Vm, ll, tl, S_hv, viol = check(fd, V, limits, v_slack)
    throughput = 0.0
    if battery is not None and battery["p_mw"] > 0:
        p_bat, throughput = _battery(S_hv, fd.trafo_sn_mva, battery)
        S[:, fd.busbar] += p_bat                 # discharge positive = generation at the busbar
        V = pf.solve(fd, S, v_slack)
        Vm, ll, tl, S_hv, viol = check(fd, V, limits, v_slack)
    losses = np.real(S_hv) + np.real(S).sum(axis=1)  # import from MV + net LV injection
    return YearResult(Vm, ll, tl, losses, viol, curtailed, throughput, tap, S, v_slack)


def _choose_tap(fd, S, V0, limits, ront):
    """Pick per step the slack voltage (tap) that best centres the LV voltages in the band; the
    tap position is chosen from the no-tap solution, which is accurate because the LV voltages shift
    almost one-to-one with the secondary voltage."""
    Vm = np.abs(V0)
    taps = np.arange(-ront["n"], ront["n"] + 1) * ront["step"]
    lo, hi = Vm.min(axis=1), Vm.max(axis=1)
    # worst-side margin for each tap; pick the max-min
    margin = np.stack([np.minimum(lo + t - limits.vmin, limits.vmax - hi - t) for t in taps], axis=1)
    best = taps[np.argmax(margin, axis=1)]
    return 1.0 + best, best


@njit(cache=True)
def _curtail_kernel(P, rating, viol, floor_mw, dt):
    out = P.copy()
    cut = 0.0
    for i in range(P.shape[0]):
        debt = 0.0  # energy owed to device i (MWh)
        for k in range(P.shape[1]):
            if viol[k]:
                if out[i, k] > floor_mw:
                    e = (out[i, k] - floor_mw) * dt
                    debt += e
                    cut += e
                    out[i, k] = floor_mw
            elif debt > 0:
                extra = min(rating[i] - out[i, k], debt / dt)  # repay up to the device rating
                if extra > 0:
                    out[i, k] += extra
                    debt -= extra * dt
    return out, cut


def _curtail(devices, viol, floor_mw):
    """Paragraph 14a: in violated steps each device is limited to floor_mw; the energy it could not
    take is delivered in the next steps without a violation, at up to its rating."""
    P = np.array([d.P for _, _, d in devices])
    rating = np.array([max(d.size_mw, d.P.max()) for _, _, d in devices])  # some profiles exceed nominal size
    out, cut = _curtail_kernel(P, rating, np.asarray(viol, np.bool_), float(floor_mw), DT_H)
    return list(out), cut


def _battery(S_hv, sn_mva, bat):
    """Greedy peak shaving at the busbar on apparent power: the active power the transformer may carry
    in a step is sqrt((0.95 S_n)^2 - Q^2). Discharge above it in import, charge above it in export
    (PV); otherwise recharge towards full with the headroom left (the feeders here peak on import). Returns the battery injection
    (T,) in MW and the throughput in MWh."""
    P, Q = np.real(S_hv), np.imag(S_hv)
    lim = np.sqrt(np.maximum((0.95 * sn_mva) ** 2 - Q ** 2, 0.0))
    e_max, p_max = bat["e_mwh"], bat["p_mw"]
    soc = e_max  # start full, recharge between peaks
    out = np.zeros(len(P))
    thr = 0.0
    for k, p in enumerate(P):
        if p > lim[k]:                                       # import peak: discharge
            q = min(p - lim[k], p_max, soc / DT_H)
        elif p < -lim[k]:                                    # export peak: charge
            q = -min(-lim[k] - p, p_max, (e_max - soc) / DT_H)
        elif soc < e_max:                                    # recharge with the headroom left
            q = -min(0.5 * (lim[k] - p), p_max, (e_max - soc) / DT_H / 0.92)
        else:
            q = 0.0
        soc -= q * DT_H if q > 0 else q * DT_H * 0.92        # 92 % charging efficiency
        out[k] = q
        thr += abs(q) * DT_H
    return out, thr
