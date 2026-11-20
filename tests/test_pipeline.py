"""Real-time properties: no look-ahead, Monte Carlo accuracy, HMM filter, backtest arithmetic."""
import numpy as np
import pandas as pd

from p8regimes.backtest import Strategy, run_backtest
from p8regimes.config import Params
from p8regimes.dataset import build_ea
from p8regimes.indicator import run_indicator
from p8regimes.signals import hmm_filter, hmm_fit
from p8regimes.simulate import simulate_raw, true_hb


def _panel(years=12, seed=7):
    raw, sig = simulate_raw(years, seed)
    panel = build_ea(raw, Params())
    panel.market = "TEST"
    return panel, sig


def test_no_lookahead(tmp_path):
    panel, _ = _panel(6)
    p = Params(n_accept=200, batch=50_000)
    d = panel.month_ends[30]
    a = run_indicator(panel, p, paths=None, use_cache=False, progress=False, dates=[d])
    noisy = panel.daily.copy()
    after = noisy.index > d
    noisy.loc[after, ["dy2", "dy5", "dy10", "re_log"]] *= 3.0          # change the future only
    panel.daily = noisy
    b = run_indicator(panel, p, paths=None, use_cache=False, progress=False, dates=[d])
    cols = ["hb", "hb_p16", "hb_p84", "beta_now"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_monte_carlo_short():
    panel, sig = _panel(12)
    p = Params(n_accept=300, batch=50_000)
    dates = panel.month_ends[24::2]
    ind = run_indicator(panel, p, use_cache=False, progress=False, dates=dates)
    tr = true_hb(sig.to_numpy(), panel.daily.index, ind.index)
    assert ind["hb"].corr(tr) > 0.85
    assert abs((ind["hb"] - tr).mean()) < 0.1


def test_hmm_filter_uses_no_future_data():
    rng = np.random.default_rng(0)
    c1 = np.array([[1.0, 0.6], [0.6, 1.0]])
    c2 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    states = np.repeat([0, 1, 0, 1], 60)
    Y = np.array([rng.multivariate_normal([0, 0], c1 if s == 0 else c2) for s in states])
    model = hmm_fit(Y, seed=1)
    full = hmm_filter(Y, model)
    part = hmm_filter(Y[:100], model)
    assert np.allclose(full[:100], part)
    assert np.allclose(full.sum(axis=1), 1)
    corr = [m[0, 1] / np.sqrt(m[0, 0] * m[1, 1]) for m in model["cov"]]
    pos = int(np.argmax(corr))
    acc = ((full[:, pos] > 0.5) == (states == 0)).mean()
    assert acc > 0.85


def test_backtest_6040_arithmetic():
    idx = pd.date_range("2010-01-31", periods=13, freq="ME")
    rng = np.random.default_rng(1)
    mon = pd.DataFrame({"r_e": rng.normal(0, 0.04, 13), "r_b": rng.normal(0, 0.02, 13), "r_f": 0.001}, index=idx)
    risk = pd.DataFrame({"sigma_ref": 0.1, "sigma_6040": 0.1, "sigma_e": 0.15, "k": 1.0}, index=idx)
    sig = pd.DataFrame(index=idx)
    res = run_backtest(mon, sig, risk, [Strategy("60/40", None, "A", "")], Params(cost_bp=0.0))["60/40"]
    expected = 0.6 * mon["r_e"].iloc[1:] + 0.4 * mon["r_b"].iloc[1:]
    assert np.allclose(res["R"].to_numpy(), expected.to_numpy())
