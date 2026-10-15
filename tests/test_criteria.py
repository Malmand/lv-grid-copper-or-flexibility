"""Hand checks of the cost and life-cycle bookkeeping."""
from pathlib import Path

import numpy as np
import yaml

from lvplan import criteria as C

P = yaml.safe_load(open(Path(__file__).parents[1] / "params.yaml"))


def _ledger(assets, losses=0.0, devices=0, curtailed=0.0, thr=0.0):
    years = [dict(year=y, losses_mwh=losses, curtailed_mwh=curtailed, battery_throughput_mwh=thr,
                  devices=devices, viol_hours=0.0) for y in (2025, 2030, 2035, 2040, 2045)]
    return dict(grid="rural", scenario="central", pathway="A1", assets=assets, years=years, unresolved=[])


def test_cable_and_transformer_by_hand():
    led = _ledger([dict(year=2025, kind="cable", qty=1.0, unit="km", mm2=150.0),
                   dict(year=2025, kind="transformer", qty=1, unit="unit", mva=0.63)])
    p = C.central(P)
    out = {k: v[0] for k, v in C.evaluate(led, P, p).items()}
    share = 21 / 40                                                  # 21 of 40 years inside 2025-2045
    co2_cable = 27000 * share / 1000
    co2_trafo = (2182 * 5.5 + 500) * share / 1000
    assert np.isclose(out["co2_embodied"], co2_cable + co2_trafo)
    assert np.isclose(out["metals_t"], 1.620 + (332 + 6.2) / 1000)
    assert out["trench_km"] == 1.0
    # annuity over 40 years at 5 %, 21 annuities discounted to 2025
    r, L = 0.05, 40
    crf = r / (1 - (1 + r) ** -L)
    a21 = sum((1 + r) ** -t for t in range(1, 22))
    assert np.isclose(out["capex_cable"], 80000 * 1.88 * crf * a21 / 1000)
    assert np.isclose(out["capex_transformer"], 15000 * 1.88 * crf * a21 / 1000)
    # an asset that lasts exactly the horizon costs its full price (annuities of its whole life)
    p2 = dict(p, life=np.array([21.0]))
    out2 = C.evaluate(led, P, p2)
    assert np.isclose(out2["capex_cable"][0], 80000 * 1.88 / 1000)


def test_losses_and_flex_payments():
    led = _ledger([], losses=10.0, devices=20, curtailed=1.0)
    p = C.central(P)
    dso = {k: v[0] for k, v in C.evaluate(led, P, p, "dso").items()}
    sysv = {k: v[0] for k, v in C.evaluate(led, P, p, "system").items()}
    d = sum(1.05 ** -t for t in range(1, 22))
    assert np.isclose(dso["cost_losses"], 10 * 100 * d / 1000)
    assert np.isclose(dso["cost_flex"], 20 * 150 * d / 1000)
    assert sysv["cost_flex"] == 0
    # grid factor falls linearly from 0.363 t/MWh (2024) to 0 in 2045
    f = [0.363 * (2045 - y) / 21 for y in range(2025, 2046)]
    assert np.isclose(dso["co2_losses"], 10 * sum(f))
    assert np.isclose(dso["shifted_mwh"], 21.0)


def test_sampling_within_ranges():
    s = C.sample(P, 5000, seed=3)
    for k, (sec, key) in C.UNCERTAIN.items():
        d = P[sec][key]
        assert s[k].min() >= d["low"] - 1e-12 and s[k].max() <= d["high"] + 1e-12
    assert np.array_equal(C.sample(P, 10, seed=3)["esc"], C.sample(P, 10, seed=3)["esc"])
