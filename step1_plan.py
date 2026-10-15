"""Step 1: plan every feeder, adoption scenario and pathway from 2025 to 2045.

Writes results/ledgers.json: for each case the assets built (year, kind, quantity) and the yearly
operation (losses, curtailed energy, battery throughput, limit violations). Resumable; with
--pathways only those pathways are (re)planned and replaced in the file.

    python step1_plan.py [--workers 2] [--pathways A5 ...]
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from lvplan.plan import PATHWAYS, plan
from lvplan.scenarios import Feeder

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ledgers.json"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_case(args):
    grid, scenario, pathways = args
    P = yaml.safe_load(open(ROOT / "params.yaml"))
    f = Feeder(grid)
    out = []
    for pw in pathways:
        t0 = time.time()
        led = plan(f, P, scenario, pw)
        out.append(_clean(asdict(led)))
        logging.info("%s %s %s: %d assets, unresolved %s (%.0f s)", grid, scenario, pw, len(led.assets),
                     led.unresolved, time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pathways", nargs="*", default=None)
    a = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "results" / "step1.log"), logging.StreamHandler(sys.stdout)])
    P = yaml.safe_load(open(ROOT / "params.yaml"))
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    cases = [(g, s) for g in P["planning"]["grids"] for s in ("low", "central", "high")]
    if a.pathways:
        todo = [(g, s, a.pathways) for g, s in cases]
    else:
        have = {(d["grid"], d["scenario"]) for d in done}
        todo = [(g, s, list(PATHWAYS)) for g, s in cases if (g, s) not in have]
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(run_case, todo):
            new = {(d["grid"], d["scenario"], d["pathway"]) for d in res}
            done = [d for d in done if (d["grid"], d["scenario"], d["pathway"]) not in new] + res
            OUT.write_text(json.dumps(done, indent=0))


if __name__ == "__main__":
    main()
