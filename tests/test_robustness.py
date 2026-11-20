"""Deflated Sharpe ratio, regime switches and costs, HMM stability."""
import numpy as np
import pandas as pd
from scipy import stats

from p8regimes import robustness as rb
from p8regimes.config import Params
from p8regimes.signals import hmm_run


def test_psr_matches_closed_form_for_normal_returns():
    rng = np.random.default_rng(0)
    x = rng.normal(0.006, 0.04, 600)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    expected = stats.norm.cdf(sr * np.sqrt(599) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert np.isclose(rb.psr(x, 0.0), expected)


def test_dsr_equals_psr_with_one_trial_and_falls_with_more_trials():
    rng = np.random.default_rng(1)
    srs = rng.normal(0.05, 0.03, 20)
    assert rb.expected_max_sharpe(srs, 1) == 0.0
    assert rb.expected_max_sharpe(srs, 5) < rb.expected_max_sharpe(srs, 20) < rb.expected_max_sharpe(srs, 100)
    x = rng.normal(0.01, 0.04, 400)
    assert rb.psr(x, rb.expected_max_sharpe(srs, 20)) < rb.psr(x, 0.0)


def test_effective_trials():
    rng = np.random.default_rng(2)
    base = rng.normal(size=500)
    same = pd.DataFrame({f"s{i}": base + 1e-6 * rng.normal(size=500) for i in range(5)})
    indep = pd.DataFrame({f"s{i}": rng.normal(size=500) for i in range(5)})
    assert abs(rb.effective_trials(same) - 1) < 0.01
    assert abs(rb.effective_trials(indep) - 5) < 0.3


def test_switch_stats_counts_flips_and_whipsaws():
    C = [0, 0, 1, 0, 0, 1, 1, 1, 0]
    idx = pd.date_range("2020-01-31", periods=len(C), freq="ME")
    df = pd.DataFrame({"C": np.array(C, float), "turnover": [0.0, 0.01, 0.3, 0.3, 0.01, 0.3, 0.01, 0.01, 0.3],
                       "x": 0.0}, index=idx)
    st = rb.switch_stats({"S": df}).loc["S"]
    assert np.isclose(st["bascules par an"], 12 * 4 / 9)
    assert np.isclose(st["allers-retours en 2 mois (%)"], 50.0)
    assert np.isclose(st["part de la rotation due aux bascules (%)"], 100 * 1.2 / 1.24)


def test_break_even_cost_interpolation():
    grid = pd.DataFrame({"0 pb": [0.50, 0.55, 0.49], "10 pb": [0.50, 0.53, 0.48], "20 pb": [0.50, 0.49, 0.47]},
                        index=["60/40", "S", "T"])
    be = rb.break_even_costs(grid, refs=("60/40",))
    assert be.loc["S", "seuil contre 60/40"] == "17.5 pb"
    assert be.loc["T", "seuil contre 60/40"] == "aucun avantage sans coûts"


def test_hmm_stability_log_on_two_regimes():
    rng = np.random.default_rng(3)
    T = 240
    states = (np.arange(T) // 60) % 2
    covs = [np.array([[16.0, -3.0], [-3.0, 4.0]]), np.array([[25.0, 6.0], [6.0, 4.0]])]
    xy = np.array([rng.multivariate_normal([0.5, 0.2], covs[s]) for s in states]) / 100
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    mon = pd.DataFrame({"r_e": xy[:, 0], "r_b": xy[:, 1], "r_f": 0.0}, index=idx)
    p = Params(hmm_min_months=120, hmm_refit_every=12)
    r = hmm_run(mon, p, track=True)
    st = r["stability"]
    assert len(st) == (T - 120 - 1) // 12
    assert st["accord passé"].mean() > 0.8
    reg = (r["p"].dropna() > 0.5).astype(int)
    assert ((reg == states[120:]).mean()) > 0.8
