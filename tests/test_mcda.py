"""The vectorised MCDA methods against pymcdm, and SMAA properties."""
import numpy as np
from pymcdm import methods
from pymcdm.normalizations import minmax_normalization

from lvplan import mcda


def _cases(n=20, m=5, k=6, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1, 100, (n, m, k))
    X[0, :, 2] = 7.0                                    # one criterion that does not discriminate
    W = rng.dirichlet(np.ones(k), n)
    return X, W


def test_wsm_topsis_promethee_match_pymcdm():
    X, W = _cases()
    k = X.shape[2]
    types = -np.ones(k)
    ours = {name: f(X, W) for name, f in mcda.METHODS.items()}
    for n in range(X.shape[0]):
        rng = X[n].max(axis=0) - X[n].min(axis=0)
        ref = {
            "WSM": methods.WSM(minmax_normalization)(X[n], W[n], types),
            "TOPSIS": methods.TOPSIS(minmax_normalization)(X[n], W[n], types),
            "PROMETHEE II": methods.PROMETHEE_II("vshape_2", p=list(0.5 * rng), q=list(0.1 * rng))(X[n], W[n], types)
                            if (rng > 0).all() else None,
        }
        for name, r in ref.items():
            if r is not None:
                assert np.allclose(ours[name][n], r, atol=1e-10), (name, n)


def test_promethee_worked_example():
    # a = (1, 6), b = (1.5, 4), c = (3, 2), both criteria minimised, weights 0.7 / 0.3.
    # Criterion 1: range 2, q = 0.2, p = 1. a over b: d = 0.5 -> 0.3 / 0.8 = 0.375; a over c, b over c: 1.
    # Criterion 2: range 4, q = 0.4, p = 2. b over a: d = 2 -> 1; c over a, c over b: 1.
    # pi(a,b) = 0.2625, pi(a,c) = 0.7, pi(b,a) = 0.3, pi(b,c) = 0.7, pi(c,a) = 0.3, pi(c,b) = 0.3
    X = np.array([[[1.0, 6.0], [1.5, 4.0], [3.0, 2.0]]])
    W = np.array([[0.7, 0.3]])
    phi = mcda.promethee(X, W)[0]
    assert np.allclose(phi, [(0.9625 - 0.6) / 2, (1.0 - 0.5625) / 2, (0.6 - 1.4) / 2])


def test_promethee_without_thresholds_ranks_like_wsm():
    X, W = _cases(n=200)
    assert np.array_equal(mcda.ranks(mcda.promethee(X, W, 0.0, 1.0)), mcda.ranks(mcda.wsm(X, W)))


def test_ranks_ties_share_best():
    s = np.array([[0.3, 0.3, 0.1]])
    assert mcda.ranks(s).tolist() == [[1, 1, 3]]


def test_smaa_reproducible_and_dominance():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 10, (4, 3))
    base[0] = base.min(axis=0) * 0.5                       # alternative 0 dominates
    X = np.broadcast_to(base, (2000, 4, 3)).copy()
    W1 = mcda.dirichlet_weights(2000, 3, seed=5)
    W2 = mcda.dirichlet_weights(2000, 3, seed=5)
    assert np.array_equal(W1, W2)
    for m in mcda.METHODS:
        acc, cw, conf = mcda.smaa(X, W1, m)
        assert acc[0, 0] == 1.0 and conf[0] == 1.0
        assert np.allclose(acc.sum(axis=1), 1)
