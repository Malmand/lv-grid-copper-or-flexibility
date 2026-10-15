"""Step 3: multi-criteria ranking of the pathways, with uncertainty (SMAA-2).

For each feeder the alternatives are the pathways that meet the limits in every adoption scenario
(A0, do nothing, is a reference only). Six criteria, all minimised:

    cost        net present cost 2025-2045 (kEUR)                         central adoption scenario
    co2         life-cycle CO2: embodied (spread over life) + losses (t)  central adoption scenario
    metals      aluminium and copper in the new assets (t)                central adoption scenario
    trench      new cable trench (km), a proxy for disruption             central adoption scenario
    shifted     energy shifted by paragraph-14a control (MWh)             central adoption scenario
    regret      largest cost regret over the low/central/high scenarios (kEUR)

Uncertainty: N samples of all uncertain parameters (triangular over their ranges, incl. the price
escalation of the 2012/2016 cost data) and, independently, N weight vectors uniform on the simplex.

Robustness: the SMAA is repeated without the metals criterion.

Outputs in results/: smaa.csv, named_weights.csv, weight_map.npz, criteria_uncertainty.csv.

    python step3_mcda.py [--n 10000]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lvplan import criteria as C
from lvplan import mcda

ROOT = Path(__file__).parent
SCEN = ("low", "central", "high")
CRIT = ["cost", "co2", "metals_t", "trench_km", "shifted_mwh", "regret"]
NAMED = {   # illustrative stakeholder profiles, same order as CRIT
    "equal": [1, 1, 1, 1, 1, 1],
    "cost-led operator": [0.40, 0.10, 0.05, 0.15, 0.10, 0.20],
    "climate-led": [0.15, 0.40, 0.15, 0.10, 0.05, 0.15],
    "resource-led": [0.15, 0.20, 0.35, 0.15, 0.05, 0.10],
}


def matrices(leds, P, p, grid, view):
    """Criteria per sample: X (N, m, 6) for the feasible pathways of a grid, and the per-sample
    criteria dicts of the central scenario (for the uncertainty table)."""
    by = {(l["scenario"], l["pathway"]): l for l in leds if l["grid"] == grid}
    paths = sorted({pw for (_, pw) in by if pw != "A0"})
    paths = [pw for pw in paths if all(C.feasible(by[(s, pw)]) for s in SCEN)]
    ev = {(s, pw): C.evaluate(by[(s, pw)], P, p, view) for s in SCEN for pw in paths}
    cost = np.stack([np.stack([ev[(s, pw)]["cost"] for pw in paths], axis=1) for s in SCEN], axis=2)  # (N, m, 3)
    regret = (cost - cost.min(axis=1, keepdims=True)).max(axis=2)
    cols = [np.stack([ev[("central", pw)][c] for pw in paths], axis=1) for c in CRIT[:-1]] + [regret]
    return paths, np.stack(cols, axis=2), {pw: ev[("central", pw)] for pw in paths}, regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    P = yaml.safe_load(open(ROOT / "params.yaml"))
    leds = json.loads((ROOT / "results" / "ledgers.json").read_text())
    grids = list(P["planning"]["grids"])
    ps = C.sample(P, a.n, seed=0)
    pc = C.central(P)
    W = mcda.dirichlet_weights(a.n, len(CRIT), seed=1)
    W5 = mcda.dirichlet_weights(a.n, len(CRIT) - 1, seed=2)
    smaa_rows, named_rows, unc_rows, wmap = [], [], [], {}
    step = 0.02
    grid_w = [(wc, wo) for wc in np.arange(0, 1 + 1e-9, step) for wo in np.arange(0, 1 - wc + 1e-9, step)]
    Wm = np.array([[wc, wo] + [(1 - wc - wo) / 4] * 4 for wc, wo in grid_w])
    for g in grids:
        for view in ("dso", "system"):
            paths, X, ev, regret = matrices(leds, P, ps, g, view)
            # robustness: the same analysis without the metals criterion (the battery's metal use is a
            # lower bound, cells only, and the transformer swap counts the whole new unit)
            keep = [j for j, c in enumerate(CRIT) if c != "metals_t"]
            for crit_set, Xs, Ws, names in (("all", X, W, CRIT),
                                             ("no_metals", X[:, :, keep], W5, [CRIT[j] for j in keep])):
                for method in mcda.METHODS:
                    acc, cw, conf = mcda.smaa(Xs, Ws, method)
                    for i, pw in enumerate(paths):
                        smaa_rows.append(dict(grid=g, view=view, criteria=crit_set, method=method, pathway=pw,
                                              confidence=conf[i], **{f"rank{r + 1}": acc[i, r] for r in range(len(paths))},
                                              **{f"cw_{c}": cw[i, j] for j, c in enumerate(names)}))
            paths_c, Xc, _, _ = matrices(leds, P, pc, g, view)
            for wname, w in NAMED.items():
                w = np.array(w, float) / sum(w)
                for method, f in mcda.METHODS.items():
                    sc = f(Xc, w[None, :])
                    R = mcda.ranks(sc)[0]
                    # fixed weights, uncertain parameters: how often each pathway ranks first
                    Rp = mcda.ranks(f(X, np.broadcast_to(w, (a.n, len(CRIT)))))
                    for i, pw in enumerate(paths_c):
                        named_rows.append(dict(grid=g, view=view, weights=wname, method=method, pathway=pw,
                                               score=sc[0, i], rank=int(R[i]), p_first=(Rp[:, i] == 1).mean()))
            if view == "dso":
                wmap[g] = dict(paths=paths_c, w=Wm[:, :2],
                               winner=np.array([mcda.ranks(mcda.promethee(Xc, Wm[j:j + 1]))[0].argmin()
                                                for j in range(len(Wm))]))
            for i, pw in enumerate(paths):
                for c in list(ev[pw]):
                    v = ev[pw][c]
                    unc_rows.append(dict(grid=g, view=view, pathway=pw, criterion=c, p05=np.percentile(v, 5),
                                         p50=np.percentile(v, 50), p95=np.percentile(v, 95)))
                v = regret[:, i]
                unc_rows.append(dict(grid=g, view=view, pathway=pw, criterion="regret", p05=np.percentile(v, 5),
                                     p50=np.percentile(v, 50), p95=np.percentile(v, 95)))
    out = ROOT / "results"
    pd.DataFrame(smaa_rows).to_csv(out / "smaa.csv", index=False, float_format="%.4g")
    pd.DataFrame(named_rows).to_csv(out / "named_weights.csv", index=False, float_format="%.4g")
    pd.DataFrame(unc_rows).to_csv(out / "criteria_uncertainty.csv", index=False, float_format="%.4g")
    np.savez(out / "weight_map.npz", **{f"{g}_{k}": np.asarray(v) for g, d in wmap.items() for k, v in d.items()})
    s = pd.DataFrame(smaa_rows)
    print(s.pivot_table(index=["grid", "pathway"], columns=["view", "criteria", "method"], values="rank1").round(2).to_string())


if __name__ == "__main__":
    main()
