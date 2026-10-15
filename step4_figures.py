"""Step 4: figures (journal style: L-frame axes, outward ticks, no grid, Okabe-Ito colours, 300 dpi).

    python step4_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
GRIDS = {"rural": "Rural", "semi_urban": "Semi-urban", "urban": "Urban"}
PCOL = {"A0": "#000000", "A1": "#0072B2", "A2": "#56B4E9", "A3": "#009E73", "A4": "#E69F00", "A5": "#CC79A7"}
PLAB = {"A1": "A1 Copper when needed", "A2": "A2 Copper for 2045", "A3": "A3 Flexibility + copper",
        "A4": "A4 Battery + copper", "A5": "A5 rONT + flex + copper"}
SCOL = {"low": "#009E73", "central": "#0072B2", "high": "#D55E00"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def label(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def fig_need(leds):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (g, title), L in zip(axs, GRIDS.items(), "ABC"):
        for j, s in enumerate(SCOL):
            led = next(l for l in leds if l["grid"] == g and l["scenario"] == s and l["pathway"] == "A0")
            y = [d["year"] for d in led["years"]]
            h = [d["viol_hours"] for d in led["years"]]
            ax.bar(np.arange(len(y)) + (j - 1) * 0.27, h, 0.27, color=SCOL[s], label=s)
        ax.set_xticks(range(len(y)), y)
        ax.set_title(title)
        ax.set_ylabel("Hours outside limits per year" if L == "A" else "")
        label(ax, L)
    axs[0].legend(title="Adoption", loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_need_without_action.png")
    plt.close(fig)


def fig_cost(crit, unc):
    parts = [("capex_cable", "Cables", "#0072B2"), ("capex_transformer", "Transformer swaps", "#56B4E9"),
             ("capex_ront", "rONT", "#CC79A7"), ("capex_battery", "Battery", "#E69F00"),
             ("cost_losses", "Losses", "#999999"), ("cost_flex", "14a payments", "#009E73")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (g, title), L in zip(axs, GRIDS.items(), "ABC"):
        d = crit[(crit.grid == g) & (crit.scenario == "central") & (crit.view == "dso") & (crit.pathway != "A0")]
        d = d.set_index("pathway").sort_index()
        bottom = np.zeros(len(d))
        for col, name, c in parts:
            ax.bar(range(len(d)), d[col], 0.65, bottom=bottom, color=c, label=name)
            bottom += d[col].to_numpy()
        u = unc[(unc.grid == g) & (unc.view == "dso") & (unc.criterion == "cost")].set_index("pathway").loc[d.index]
        ax.errorbar(range(len(d)), d["cost"], yerr=[np.maximum(d["cost"] - u.p05, 0), np.maximum(u.p95 - d["cost"], 0)], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_xticks(range(len(d)), d.index)
        ax.set_title(title)
        ax.set_ylabel("Net present cost 2025-2045 (kEUR)" if L == "A" else "")
        label(ax, L)
    axs[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_cost_breakdown.png")
    plt.close(fig)


def fig_tradeoff(unc):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (g, title), L in zip(axs, GRIDS.items(), "ABC"):
        u = unc[(unc.grid == g) & (unc.view == "dso")]
        for pw in sorted(u.pathway.unique()):
            c = u[(u.pathway == pw) & (u.criterion == "cost")].iloc[0]
            e = u[(u.pathway == pw) & (u.criterion == "co2")].iloc[0]
            ax.errorbar(c.p50, e.p50, xerr=[[c.p50 - c.p05], [c.p95 - c.p50]], yerr=[[e.p50 - e.p05], [e.p95 - e.p50]],
                        fmt="o", ms=4, color=PCOL[pw], elinewidth=0.8, capsize=1.5, label=PLAB[pw])
        ax.set_title(title)
        ax.set_xlabel("Net present cost (kEUR)")
        ax.set_ylabel("Life-cycle CO$_2$e 2025-2045 (t)" if L == "A" else "")
        label(ax, L)
    axs[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_cost_vs_co2.png")
    plt.close(fig)


def fig_smaa(smaa):
    methods = ["WSM", "TOPSIS", "PROMETHEE II"]
    mcol = {"WSM": "#56B4E9", "TOPSIS": "#E69F00", "PROMETHEE II": "#0072B2"}
    views = {"dso": "14a payments counted", "system": "14a payments as transfer"}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True)
    letters = iter("ABCDEF")
    for row, (view, vname) in zip(axs, views.items()):
        for ax, (g, title) in zip(row, GRIDS.items()):
            d = smaa[(smaa.grid == g) & (smaa.view == view) & (smaa.criteria == "all")]
            paths = sorted(d.pathway.unique())
            for j, m in enumerate(methods):
                v = d[d.method == m].set_index("pathway").loc[paths, "rank1"]
                ax.bar(np.arange(len(paths)) + (j - 1) * 0.27, v, 0.27, color=mcol[m], label=m)
            ax.set_xticks(range(len(paths)), paths)
            ax.set_ylim(0, 1)
            ax.set_title(f"{title}\n{vname}")
            ax.set_ylabel("Rank-1 acceptability" if g == "rural" else "")
            label(ax, next(letters))
    axs[0, 0].legend(loc="upper left", bbox_to_anchor=(0.25, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig4_smaa_rank1.png")
    plt.close(fig)


def fig_weight_map(wm):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (g, title), L in zip(axs, GRIDS.items(), "ABC"):
        paths = list(wm[f"{g}_paths"])
        w, win = wm[f"{g}_w"], wm[f"{g}_winner"]
        cols = [PCOL[p] for p in paths]
        ax.scatter(w[:, 0], w[:, 1], c=[cols[i] for i in win], s=9, marker="s", linewidths=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Weight on cost")
        ax.set_ylabel("Weight on CO$_2$" if L == "A" else "")
        ax.set_title(title)
        label(ax, L)
        present = sorted(set(paths[i] for i in win))
        ax.legend(handles=[Patch(color=PCOL[p], label=p) for p in present], loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_weight_map.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    leds = json.loads((RES / "ledgers.json").read_text())
    crit = pd.read_csv(RES / "criteria_central.csv")
    unc = pd.read_csv(RES / "criteria_uncertainty.csv")
    smaa = pd.read_csv(RES / "smaa.csv")
    wm = np.load(RES / "weight_map.npz", allow_pickle=False)
    fig_need(leds)
    fig_cost(crit, unc)
    fig_tradeoff(unc)
    fig_smaa(smaa)
    fig_weight_map(wm)


if __name__ == "__main__":
    main()
