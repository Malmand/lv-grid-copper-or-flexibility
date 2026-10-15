"""Planning a feeder from 2025 to 2045 along one pathway.

At every planning year Y the operator makes the feeder compliant for the load expected at Y + 5 (or at
2045), using the pathway's means: copper (parallel cables, bigger transformer), paragraph-14a
flexibility, a battery at the substation, an on-load tap-changing transformer, or combinations. What
the pathway's own means cannot solve is solved with copper. Each year is then operated with the
actual load of that year, which gives the losses, curtailment and battery use.

The output is a ledger of physical quantities (assets built, energy lost, energy curtailed), so that
costs and emissions can be recomputed for any set of prices in the uncertainty analysis without
running the power flows again.
"""
import copy
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandapower.topology as top

from . import grid as G
from . import powerflow as pf

PATHWAYS = {
    "A0": "Do nothing",
    "A1": "Copper, when needed",
    "A2": "Copper, built for 2045 at once",
    "A3": "Flexibility (14a) + copper",
    "A4": "Substation battery + copper",
    "A5": "Tap-changing transformer + flexibility + copper",
}
YEARS = [2025, 2030, 2035, 2040, 2045]


@dataclass
class Ledger:
    grid: str
    scenario: str
    pathway: str
    assets: list = field(default_factory=list)      # dicts: year, kind, qty, unit, mm2 (cables)
    years: list = field(default_factory=list)       # dicts: year, losses_mwh, curtailed_mwh, devices, viol_hours ...
    unresolved: list = field(default_factory=list)


def _limits(P, ront):
    b = P["planning"]["voltage_band"]["with_ront" if ront else "without_ront"]
    return G.Limits(b["min"], b["max"], P["planning"]["thermal_limit"])


def _section_mm2(std_type):
    s = str(std_type)
    return float(s.split("x")[1].split("S")[0].split(" ")[0]) if "x" in s else 150.0


def _upgrade_trafo(net, ledger, year, ront=False):
    """Next transformer size. With a tap changer in place the replacement is again a tap-changing
    transformer; if one was installed earlier in the same year, the two become one unit."""
    sn = float(net.trafo.sn_mva.iloc[0])
    bigger = [s for s in G.TRAFO_SIZES if s > sn + 1e-6]
    if not bigger:
        return False
    new = bigger[0]
    t = net.trafo.index[0]
    net.trafo.at[t, "sn_mva"] = new
    net.trafo.at[t, "pfe_kw"] = 1.65 * (new / 0.63) ** 0.75      # no-load losses scale with size
    kind = "ront" if ront else "transformer"
    if ront:
        ledger.assets[:] = [a for a in ledger.assets if not (a["kind"] == "ront" and a["year"] == year)]
    ledger.assets.append(dict(year=year, kind=kind, qty=1, unit="unit", mva=new))
    return True


def _choose_action(net, fd, graph, Vm, ll, tl, limits):
    """The copper measure for the worst problem in a set of steps: a bigger transformer if it is
    overloaded more than any cable, else a parallel cable on the most loaded line, else (voltage) a
    parallel cable on the line with the largest voltage drop on the path to the worst bus.
    Returns ("trafo", None) or ("line", position) or (None, reason)."""
    if tl.max() > limits.thermal and tl.max() >= ll.max():
        return "trafo", None
    if ll.max() > limits.thermal:
        return "line", int(np.unravel_index(np.argmax(ll), ll.shape)[1])
    dev = np.maximum(limits.vmin - Vm, Vm - limits.vmax)
    ks, bpos = np.unravel_index(np.argmax(dev), dev.shape)
    bus = int(fd.lv_buses[bpos])
    busbar = int(net.trafo.lv_bus.iloc[0])
    path = nx.shortest_path(graph, busbar, bus)
    on_path = set(zip(path[:-1], path[1:])) | set(zip(path[1:], path[:-1]))
    cand = [i for i, ln in enumerate(net.line.itertuples()) if (ln.from_bus, ln.to_bus) in on_path]
    if not cand:
        return None, "voltage at busbar"
    return "line", cand[int(np.argmax([ll[ks, i] * np.abs(1 / fd.line_y[i]) for i in cand]))]


def _reinforce(ledger, net, inj, devices, year, limits, opts, max_rounds=10, max_steps=400):
    """Add copper until the year's load is served within limits. The search works on the critical
    steps of the last full-year run (violations and near-violations, with curtailment, battery and
    taps as they were) and the whole year is re-run after each round to confirm."""
    graph = top.create_nxgraph(net, respect_switches=True)
    fd = pf.build(net)
    r = G.run_year(fd, inj, limits, devices, **opts)
    for _ in range(max_rounds):
        if not r.viol_steps.any():
            return fd, r
        near = r.viol_steps | (r.trafo_loading > 0.9 * limits.thermal) | (r.line_loading > 0.9 * limits.thermal).any(axis=1) \
            | (r.V < limits.vmin + 0.01).any(axis=1) | (r.V > limits.vmax - 0.005).any(axis=1)
        crit = np.flatnonzero(near)
        S_c = r.S[crit]
        acted = False
        vs_c = np.broadcast_to(np.asarray(r.v_slack, float), (len(r.viol_steps),))[crit]
        for _ in range(max_steps):
            V = pf.solve(fd, S_c, vs_c)
            if "ront" in opts:
                vs_c, _ = G._choose_tap(fd, S_c, pf.solve(fd, S_c), limits, opts["ront"])
                V = pf.solve(fd, S_c, vs_c)
            Vm, ll, tl, _, viol = G.check(fd, V, limits, vs_c)
            if not viol.any():
                break
            if not _apply(net, ledger, year, opts, *_choose_action(net, fd, graph, Vm[viol], ll[viol], tl[viol], limits)):
                return fd, r
            acted = True
            fd = pf.build(net)
        if not acted:
            # the critical steps pass on their own but the full year does not (the operating
            # measures, curtailment and its rebound or the tap choice, play out differently over
            # the whole year): act on the violations of the full-year run
            v = r.viol_steps
            if not _apply(net, ledger, year, opts, *_choose_action(net, fd, graph, r.V[v], r.line_loading[v],
                                                                   r.trafo_loading[v], limits)):
                return fd, r
            fd = pf.build(net)
        r = G.run_year(fd, inj, limits, devices, **opts)
    if r.viol_steps.any():
        ledger.unresolved.append((year, "rounds limit"))
    return fd, r


def _apply(net, ledger, year, opts, act, arg):
    """Carry out one copper measure; False (and a note in the ledger) if it is not possible."""
    if act == "trafo":
        if not _upgrade_trafo(net, ledger, year, ront="ront" in opts):
            ledger.unresolved.append((year, "transformer at largest size"))
            return False
    elif act == "line":
        idx = net.line.index[arg]
        net.line.at[idx, "parallel"] += 1
        ledger.assets.append(dict(year=year, kind="cable", qty=float(net.line.at[idx, "length_km"]),
                                  unit="km", mm2=_section_mm2(net.line.at[idx, "std_type"])))
    else:
        ledger.unresolved.append((year, arg))
        return False
    return True


def _battery_size(fd, inj, limits, devices, opts):
    """Smallest battery that keeps the transformer at 95 % of its rating: power = largest excess,
    energy = largest daily excess energy, both with a 20 % margin."""
    S = G.injection_matrix(fd, inj)
    V = pf.solve(fd, S)
    _, _, _, S_hv = pf.flows(fd, V)
    p_lim = np.sqrt(np.maximum((0.95 * fd.trafo_sn_mva) ** 2 - np.imag(S_hv) ** 2, 0))
    excess = np.maximum(np.abs(np.real(S_hv)) - p_lim, 0)
    if excess.max() <= 0:
        return 0.0, 0.0
    daily = excess.reshape(-1, 96).sum(axis=1) * G.DT_H
    return 1.2 * excess.max(), 1.2 * daily.max()


def plan(feeder, P, scenario, pathway, central=None):
    """Plan one feeder for one adoption scenario along one pathway. Returns the ledger."""
    adoption = P["adoption"][scenario]
    net = copy.deepcopy(feeder.net)
    led = Ledger(feeder.grid, scenario, pathway)
    use_flex = pathway in ("A3", "A5")
    use_ront = False
    battery = {"e_mwh": 0.0, "p_mw": 0.0}
    floor = P["cost"]["flexibility_floor_kw"]["value"] / 1000
    for i, year in enumerate(YEARS):
        target = min(year + 5, YEARS[-1])
        inj, devices, _, _ = feeder.injections(target, adoption)
        if pathway == "A2" and year == YEARS[0]:
            inj_c, dev_c, _, _ = feeder.injections(YEARS[-1], P["adoption"]["central"])
            _reinforce(led, net, inj_c, dev_c, year, _limits(P, False), {})
        if pathway == "A5" and not use_ront:
            fd = pf.build(net)
            r = G.run_year(fd, inj, _limits(P, False), devices, flex={"floor_mw": floor})
            if r.viol_steps.any():
                use_ront = True
                led.assets.append(dict(year=year, kind="ront", qty=1, unit="unit", mva=float(net.trafo.sn_mva.iloc[0])))
        limits = _limits(P, use_ront)
        opts = {}
        if use_flex:
            opts["flex"] = {"floor_mw": floor}
        if use_ront:
            opts["ront"] = {"step": P["planning"]["voltage_band"]["with_ront"]["tap_step"],
                            "n": P["planning"]["voltage_band"]["with_ront"]["tap_range"]}
        if pathway == "A4":
            fd = pf.build(net)
            p_need, e_need = _battery_size(fd, inj, limits, devices, opts)
            if p_need > battery["p_mw"] or e_need > battery["e_mwh"]:
                add_p, add_e = max(p_need - battery["p_mw"], 0), max(e_need - battery["e_mwh"], 0)
                battery = {"p_mw": max(p_need, battery["p_mw"]), "e_mwh": max(e_need, battery["e_mwh"])}
                led.assets.append(dict(year=year, kind="battery", qty=add_e * 1000, unit="kWh", kw=add_p * 1000))
            opts["battery"] = battery
        if pathway != "A0":
            _reinforce(led, net, inj, devices, year, limits, opts)
        # operate the actual year
        inj_y, dev_y, sh, counts = feeder.injections(year, adoption)
        fd = pf.build(net)
        r = G.run_year(fd, inj_y, limits, dev_y, **opts)
        s = r.summary()
        led.years.append(dict(year=year, devices=len(dev_y) if use_flex else 0, n_hp=counts["hp"], n_ev=counts["ev"],
                              n_pv=counts["pv"], trafo_mva=float(net.trafo.sn_mva.iloc[0]), **s))
    return led
