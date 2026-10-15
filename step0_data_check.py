"""Step 0: check the inputs before any planning.

Every parameter in params.yaml must have a source and a consistent range; the SimBench feeders and
device profiles must load completely. Writes results/grids.csv (the three feeders today and in 2045).

    python step0_data_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lvplan import grid as G
from lvplan import powerflow as pf
from lvplan.scenarios import Feeder, library

ROOT = Path(__file__).parent


def check_params(P):
    n = 0
    for sect in ("cost", "lca"):
        for key, v in P[sect].items():
            if not isinstance(v, dict):
                continue
            if "value" in v or "value_2016" in v or "rural" in v:
                assert "source" in v or all("source" in x for x in v.values() if isinstance(x, dict)), f"{key}: no source"
                if "low" in v:
                    assert v["low"] <= v["value"] <= v["high"], f"{key}: value outside its range"
                n += 1
    e = P["escalation"]["from_2016"]
    assert e["low"] <= e["value"] <= e["high"]
    return n


def main():
    P = yaml.safe_load(open(ROOT / "params.yaml"))
    n = check_params(P)
    lib = library()
    assert len(lib.hp) and len(lib.ev) and len(lib.pv), "device library empty"
    rows = []
    for grid in P["planning"]["grids"]:
        f = Feeder(grid)
        assert f.T == 35136 and all(np.isfinite(v).all() for v in f.base_P.values())  # 2016: 366 days x 96
        fd = pf.build(f.net)
        band = P["planning"]["voltage_band"]["without_ront"]
        lim = G.Limits(band["min"], band["max"])
        for year, sc in ((2025, "central"), (2045, "low"), (2045, "central"), (2045, "high")):
            inj, dev, sh, cnt = f.injections(year, P["adoption"][sc])
            r = G.run_year(fd, inj, lim, dev).summary()
            rows.append(dict(grid=grid, code=P["planning"]["grids"][grid], year=year, scenario=sc,
                             households=len(f.households), transformer_mva=float(f.net.trafo.sn_mva.iloc[0]),
                             cable_km=round(float(f.net.line.length_km.sum()), 3),
                             cable_type=str(f.net.line.std_type.iloc[0]), **cnt,
                             **{k: round(v, 3) for k, v in r.items()}))
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "grids.csv", index=False)
    print(f"{n} parameters with sources; device library: {len(lib.hp)} heat pumps, {len(lib.ev)} EV chargers, {len(lib.pv)} PV")
    print(df.to_string())


if __name__ == "__main__":
    main()
