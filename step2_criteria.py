"""Step 2: criteria of every planned case at central parameter values.

Reads results/ledgers.json and params.yaml; writes results/criteria_central.csv with, per grid,
adoption scenario and pathway, the assets built, the cost split, emissions, metals, trenching,
shifted energy, losses, feasibility and cost regret across the adoption scenarios (in two views of
the paragraph-14a payments: "dso" counts them, "system" treats them as a transfer).

    python step2_criteria.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from lvplan import criteria as C
from lvplan.plan import PATHWAYS

ROOT = Path(__file__).parent
SCEN = ("low", "central", "high")


def assets_summary(led):
    a = led["assets"]
    return dict(cable_km=sum(x["qty"] for x in a if x["kind"] == "cable"),
                transformer_swaps=sum(1 for x in a if x["kind"] == "transformer"),
                ront_units=sum(1 for x in a if x["kind"] == "ront"),
                battery_kwh=sum(x["qty"] for x in a if x["kind"] == "battery"),
                battery_kw=sum(x.get("kw", 0) for x in a if x["kind"] == "battery"),
                final_trafo_mva=led["years"][-1]["trafo_mva"],
                max_viol_hours=max(d["viol_hours"] for d in led["years"]),
                hp_2045=led["years"][-1]["n_hp"], ev_2045=led["years"][-1]["n_ev"], pv_2045=led["years"][-1]["n_pv"])


def add_regret(df):
    """Cost regret of each pathway: in every scenario, cost minus the cheapest feasible pathway's
    cost; the criterion is the largest regret over the scenarios (minimax regret)."""
    out = []
    for (g, view), d in df.groupby(["grid", "view"]):
        ok = d[d.feasible]
        best = ok.groupby("scenario")["cost"].min()
        d = d.copy()
        d["regret"] = (d["cost"] - d["scenario"].map(best)).where(d.feasible)
        mx = d.groupby("pathway")["regret"].max().where(d.groupby("pathway")["feasible"].all())
        d["max_regret"] = d["pathway"].map(mx)
        out.append(d)
    return pd.concat(out)


def main():
    P = yaml.safe_load(open(ROOT / "params.yaml"))
    leds = json.loads((ROOT / "results" / "ledgers.json").read_text())
    p = C.central(P)
    rows = []
    for led in leds:
        for view in ("dso", "system"):
            crit = {k: float(v[0]) for k, v in C.evaluate(led, P, p, view).items()}
            rows.append(dict(grid=led["grid"], scenario=led["scenario"], pathway=led["pathway"],
                             name=PATHWAYS[led["pathway"]], view=view, feasible=C.feasible(led),
                             unresolved=";".join(map(str, led["unresolved"])), **assets_summary(led), **crit))
    df = add_regret(pd.DataFrame(rows))
    order = {s: i for i, s in enumerate(SCEN)}
    df = df.sort_values(["grid", "view", "scenario", "pathway"], key=lambda s: s.map(order) if s.name == "scenario" else s)
    df.to_csv(ROOT / "results" / "criteria_central.csv", index=False, float_format="%.4g")
    show = df[(df.view == "dso") & (df.scenario == "central")]
    cols = ["grid", "pathway", "feasible", "cable_km", "transformer_swaps", "ront_units", "battery_kwh", "cost",
            "co2", "metals_t", "trench_km", "shifted_mwh", "max_regret"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
