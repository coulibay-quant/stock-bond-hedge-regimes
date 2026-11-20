"""Tests of the results of the methodological note (propositions 1 to 4)."""
import numpy as np
import pandas as pd

from p8regimes.backtest import Strategy, run_backtest
from p8regimes.config import Params
from p8regimes.indicator import shock_stats
from p8regimes.simulate import B0
from p8regimes.structural import RotationBank, draw_rotations, identified_set, identify


def test_rotations_orthonormal_and_reproduce_covariance():           # proposition 1
    rng = np.random.default_rng(0)
    Q = draw_rotations(4, 1000, rng)
    assert np.allclose(Q @ np.transpose(Q, (0, 2, 1)), np.eye(4), atol=1e-10)
    S = B0 @ B0.T
    P = np.linalg.cholesky(S)
    B = P @ Q
    assert np.allclose(B @ np.transpose(B, (0, 2, 1)), S, atol=1e-8)


def test_identify_recognises_permuted_and_flipped_matrix():
    assert len(identify(B0[None])) == 1
    perm = [2, 0, 3, 1]
    C = B0[:, perm] * np.array([-1, 1, -1, 1])
    B = identify(C[None])
    assert len(B) == 1
    assert np.allclose(B[0], B0)


def test_identified_set_contains_only_admissible_matrices():
    S = B0 @ B0.T
    B_all, rate, used = identified_set(S, RotationBank(chunk=50_000, seed=1), n_accept=300)
    assert len(B_all) == 300 and 0 < rate < 1
    assert np.allclose(B_all @ np.transpose(B_all, (0, 2, 1)), S, atol=1e-6)
    assert len(identify(B_all)) == len(B_all)


def _dim2_set(rho, n=200_000):
    """Sign-only identified set of HB with two assets (equity, bond), proposition 2."""
    P = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    th = np.random.default_rng(1).uniform(0, 2 * np.pi, n)
    c1 = np.stack([np.cos(th), np.sin(th)])
    c2 = np.stack([-np.sin(th), np.cos(th)])
    vals = []
    for h, k in ((P @ c1, P @ c2), (P @ c2, P @ c1)):
        h_ok = np.sign(h[0]) == -np.sign(h[1])          # H: opposite moves of equity and bond
        k_ok = np.sign(k[0]) == np.sign(k[1])           # K: same moves
        vals.append(k[0][h_ok & k_ok] ** 2)
    return np.concatenate(vals)


def test_proposition_2_two_assets():
    for rho in (-0.6, -0.3, 0.3, 0.6):
        v = _dim2_set(rho)
        lo, hi = (rho ** 2, 1.0) if rho >= 0 else (0.0, 1 - rho ** 2)
        assert abs(v.min() - lo) < 0.01 and abs(v.max() - hi) < 0.01
        assert abs(np.median(v) - (1 + rho) / 2) < 0.01


def test_proposition_3_beta_is_weighted_hedge_ratios():
    rng = np.random.default_rng(3)
    eps = rng.standard_normal((500, 4)) * np.array([1.0, 1.5, 0.8, 1.2])
    u = eps @ B0.T
    u = u - u.mean(axis=0)
    S = np.cov(u.T)
    B_all, _, _ = identified_set(S, RotationBank(chunk=50_000, seed=2), n_accept=200)
    st = shock_stats(B_all, u, duration=7.0)          # recent window = whole window
    rb = -7.0 * (0.6 * u[:, 1] + 0.4 * u[:, 2]) / 100   # bond pocket return (%) implied by the yields
    re = u[:, 3]
    beta = np.cov(rb, re)[0, 1] / np.var(re, ddof=1)
    assert np.allclose(st["beta"], beta, atol=1e-10)


def test_proposition_4_vt_identity():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2000-01-31", periods=121, freq="ME")
    monthly = pd.DataFrame({"r_e": rng.normal(0.006, 0.045, 121), "r_b": rng.normal(0.003, 0.02, 121),
                            "r_f": 0.002}, index=idx)
    sig = pd.DataFrame({"C": rng.integers(0, 2, 121).astype(float)}, index=idx)
    risk = pd.DataFrame({"sigma_ref": 0.10, "sigma_6040": rng.uniform(0.07, 0.16, 121),
                         "sigma_e": rng.uniform(0.12, 0.3, 121)}, index=idx)
    risk["k"] = np.minimum(1, risk["sigma_ref"] / risk["sigma_6040"])
    p = Params(cost_bp=0.0)
    res = run_backtest(monthly, sig, risk, [Strategy("60/40", None, "A", ""), Strategy("VC", None, "VT", ""),
                                            Strategy("P8", "C", "A", "")], p)
    a, vt, base = res["P8"], res["VC"], res["60/40"]
    x6040 = base["R"] - base["R_f"]
    assert np.allclose(a["x"] - vt["x"], (1 - a["C"]) * (1 - a["k"]) * x6040, atol=1e-12)
    assert np.allclose(a["x"] - base["x"], -a["C"] * (1 - a["k"]) * x6040, atol=1e-12)
