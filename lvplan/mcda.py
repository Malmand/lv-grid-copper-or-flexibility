"""Multi-criteria ranking, vectorised over samples, and SMAA-2.

All criteria here are to be minimised. X has shape (N, m, k): N samples (weights and/or parameters),
m alternatives, k criteria; W has shape (N, k) with rows summing to 1. The three methods follow the
definitions in pymcdm (checked in tests/test_mcda.py):

- WSM: min-max normalisation per criterion (best = 1, worst = 0), weighted sum.
- TOPSIS: min-max normalisation, distances to the ideal and anti-ideal weighted points.
- PROMETHEE II: linear preference function with an indifference threshold (pymcdm "vshape_2"):
  differences below q = 10 % of the criterion's range count as no preference, above p = 50 % as
  full preference, linear in between; net flow. Note: with q = 0 and p = the range, PROMETHEE II
  ranks exactly like WSM with min-max normalisation (the net flow is then an affine function of the
  weighted sum), which is why thresholds are used (tested in tests/test_mcda.py).

A criterion on which all alternatives are equal does not discriminate: min-max normalisation then
gives 1 to all (as in pymcdm) and PROMETHEE gives no preference.

SMAA-2 (Lahdelma and Salminen, 2001): weights drawn uniformly from the simplex (no preference
information) and uncertain parameters drawn from their ranges. Results per alternative:
rank acceptability b_i^r (share of samples where i has rank r), central weight vector (mean weights
of the samples where i ranks first) and confidence factor (share of parameter samples in which i
ranks first when the central weights of i are used).
"""
import numpy as np


def _minmax(X):
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    rng = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        # cost criteria: best 1, worst 0; ties 1 as in pymcdm
        n = np.where(rng > 0, (hi - X) / np.where(rng > 0, rng, 1), 1.0)
    return n


def wsm(X, W):
    return (_minmax(X) * W[:, None, :]).sum(axis=2)


def topsis(X, W):
    v = _minmax(X) * W[:, None, :]
    pis, nis = v.max(axis=1, keepdims=True), v.min(axis=1, keepdims=True)
    dp = np.sqrt(((v - pis) ** 2).sum(axis=2))
    dm = np.sqrt(((v - nis) ** 2).sum(axis=2))
    with np.errstate(invalid="ignore"):
        s = dm / (dm + dp)
    return np.nan_to_num(s, nan=1.0)            # all alternatives equal on every criterion


def promethee(X, W, q_share=0.1, p_share=0.5):
    m = X.shape[1]
    rng = (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True))[:, None]   # (N, 1, 1, k)
    q, p = q_share * rng, p_share * rng
    d = X[:, None, :, :] - X[:, :, None, :]           # d[n, i, j, c] = x_j - x_i: > 0 means i better
    with np.errstate(invalid="ignore", divide="ignore"):
        # linear between q and p (pymcdm vshape_2)
        pref = np.where(rng > 0, np.clip((d - q) / np.where(rng > 0, p - q, 1), 0, 1), 0.0)
    pi = (pref * W[:, None, None, :]).sum(axis=3)     # (N, m, m)
    return (pi.sum(axis=2) - pi.sum(axis=1)) / (m - 1)


METHODS = {"WSM": wsm, "TOPSIS": topsis, "PROMETHEE II": promethee}


def ranks(scores, decimals=9):
    """Rank 1 = best (highest score). Ties (equal scores to the given decimals) share the best rank."""
    s = np.round(scores, decimals)
    return 1 + (s[:, None, :] > s[:, :, None]).sum(axis=2)


def dirichlet_weights(n, k, seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(k), n)


def smaa(X, W, method):
    """X (N, m, k) criteria per sample, W (N, k) weights per sample (drawn together with X).
    Returns rank acceptability (m, m), central weights (m, k), confidence factors (m,)."""
    f = METHODS[method]
    R = ranks(f(X, W))
    N, m, k = X.shape
    acc = np.stack([(R == r + 1).mean(axis=0) for r in range(m)], axis=1)
    first = R == 1
    # central weights: mean weights where i ranks first
    cw = np.array([W[first[:, i]].mean(axis=0) if first[:, i].any() else np.full(k, np.nan) for i in range(m)])
    conf = np.full(m, np.nan)
    for i in range(m):
        if np.isfinite(cw[i]).all():
            # confidence: i first under its own central weights
            Ri = ranks(f(X, np.broadcast_to(cw[i], (N, k))))
            conf[i] = (Ri[:, i] == 1).mean()
    return acc, cw, conf
