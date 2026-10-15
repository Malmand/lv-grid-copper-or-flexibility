"""Power flow against pandapower, paragraph-14a energy shifting, and reinforcement."""
import copy
from pathlib import Path

import numpy as np
import pandapower as pp
import pytest
import yaml

from lvplan import grid as G
from lvplan import plan as PL
from lvplan import powerflow as pf
from lvplan.scenarios import Device, Feeder

P = yaml.safe_load(open(Path(__file__).parents[1] / "params.yaml"))


@pytest.fixture(scope="module")
def rural():
    return Feeder("rural")


def test_powerflow_matches_pandapower(rural):
    net = copy.deepcopy(rural.net)
    fd = pf.build(net)
    inj, _, _, _ = rural.injections(2045, P["adoption"]["central"])
    S = G.injection_matrix(fd, inj)
    k = int(np.argmin(np.real(S).sum(axis=1)))                 # the step with the highest load
    V = pf.solve(fd, S[k:k + 1])
    ll, _, tl, _ = pf.flows(fd, V)
    net.load.drop(net.load.index, inplace=True)
    net.sgen.drop(net.sgen.index, inplace=True)
    for i, b in enumerate(fd.lv_buses):
        s = S[k, i]
        if abs(s) > 0:
            pp.create_load(net, int(b), p_mw=-s.real, q_mvar=-s.imag)
    pp.runpp(net, tolerance_mva=1e-10)
    vm = net.res_bus.vm_pu.loc[fd.lv_buses].to_numpy()
    assert np.max(np.abs(np.abs(V[0]) - vm)) < 1e-6
    lines = net.line[net.line.in_service].index
    assert np.max(np.abs(ll[0] * 100 - net.res_line.loading_percent.loc[lines].to_numpy())) < 0.01
    assert abs(tl[0] * 100 - net.res_trafo.loading_percent.iloc[0]) < 0.5
    assert np.abs(V[0]).min() < 0.9                          # this step is a real stress case


def test_curtailment_shifts_energy():
    T = 96
    p = np.full(T, 0.011)                                     # 11 kW charger
    devs = [("ev", 0, Device("ev", p.copy(), np.zeros(T), 0.011)),
            ("hp", 0, Device("hp", np.full(T, 0.003), np.zeros(T), 0.003))]
    viol = np.zeros(T, bool)
    viol[10:20] = True
    out, cut = G._curtail(devs, viol, 0.0042)
    assert np.all(out[0][viol] <= 0.0042 + 1e-12)
    assert np.allclose(out[1], 0.003)                         # below the floor: untouched
    assert np.isclose(cut, (0.011 - 0.0042) * 10 * G.DT_H)
    # the charger is already at its rating, so nothing can be recovered; with a 22 kW rating it can
    devs[0] = ("ev", 0, Device("ev", p.copy(), np.zeros(T), 0.022))
    out, cut = G._curtail(devs, viol, 0.0042)
    assert np.isclose(out[0].sum(), p.sum())                  # energy conserved
    assert np.all(out[0] <= 0.022 + 1e-12)


def test_reinforcement_clears_violations():
    f = Feeder("urban")
    net = copy.deepcopy(f.net)
    inj, dev, _, _ = f.injections(2045, P["adoption"]["high"])
    lim = PL._limits(P, False)
    r0 = G.run_year(pf.build(net), inj, lim)
    assert r0.viol_steps.any()
    led = PL.Ledger("urban", "high", "A1")
    _, r = PL._reinforce(led, net, inj, dev, 2045, lim, {})
    assert not r.viol_steps.any() and not led.unresolved and led.assets
