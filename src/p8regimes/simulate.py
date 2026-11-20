"""Synthetic data with known shocks (section 3.8 and annex C of the note).

* ``simulate_raw`` builds files with the same structure as the real euro-area inputs
  (zero-coupon grid, equity index, cash rate, inflation), so that the *whole* pipeline can run
  without internet: this is the DEMO mode of the working notebook.
* ``true_hb`` gives the true hedge-breaking share, to validate the estimator (Monte Carlo).
Nothing here is real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEDGE_BREAKING

B0 = np.array([[4.0, 4.0, -1.2, 0.8],         # dy2  (bp)
               [3.6, 2.8, -2.4, 2.0],         # dy5  (bp)
               [2.4, 1.6, -3.6, 2.8],         # dy10 (bp)
               [0.45, -0.35, -0.55, -0.40]])  # equity (log %)

REGIMES = [  # (start year, end year, std multipliers for g, m, p+, p-)
    (0, 8, (0.8, 1.6, 0.7, 1.5)),     # inflationary: hedge-breaking shocks dominate
    (8, 13, (1.0, 0.7, 1.1, 0.7)),    # transition
    (13, 14, (1.2, 0.6, 3.0, 0.8)),   # crisis: flight to quality
    (14, 22, (1.0, 0.6, 1.2, 0.6)),   # great moderation: hedge
    (22, 26, (0.9, 1.9, 0.8, 1.2)),   # monetary tightening
    (26, 30, (0.9, 0.9, 0.9, 1.6)),   # term-premium shock
]


def simulate_shocks(years: int = 30, seed: int = 2026, start: str = "1990-01-01",
                    regimes=REGIMES, a_coef: float = 0.03):
    """Daily VAR(1) data driven by four structural shocks with regime-dependent intensities."""
    rng = np.random.default_rng(seed)
    T = 252 * years
    mult = np.ones((T, 4))
    for a, b, m in regimes:
        mult[a * 252:min(b, years) * 252] = m
    logn = np.zeros((T, 4))
    for t in range(1, T):
        logn[t] = 0.98 * logn[t - 1] + 0.04 * rng.standard_normal(4)
    sig = mult * np.exp(logn)
    eps = sig * rng.standard_normal((T, 4))
    u = eps @ B0.T
    A = a_coef * np.eye(4)
    z = np.zeros((T, 4))
    for t in range(1, T):
        z[t] = A @ z[t - 1] + u[t]
    dates = pd.bdate_range(start, periods=T)
    return dates, z, sig


def true_hb(sig: np.ndarray, dates: pd.DatetimeIndex, month_ends: pd.DatetimeIndex, h: int = 63) -> pd.Series:
    pos = dates.get_indexer(month_ends)
    out = {}
    for d, p in zip(month_ends, pos):
        if p + 1 < h:
            continue
        v = (sig[p - h + 1:p + 1] ** 2).mean(axis=0)
        c = B0[3] ** 2 * v
        out[d] = c[list(HEDGE_BREAKING)].sum() / c.sum()
    return pd.Series(out, name="hb_true")


def simulate_raw(years: int = 30, seed: int = 2026, start: str = "1990-01-01") -> tuple[dict, pd.Series]:
    """Raw inputs shaped like the euro-area files + the true daily shock intensities."""
    dates, z, sig = simulate_shocks(years, seed, start)
    y2 = 5.0 + np.cumsum(z[:, 0]) / 100
    y5 = 5.5 + np.cumsum(z[:, 1]) / 100
    y10 = 6.0 + np.cumsum(z[:, 2]) / 100
    grid = pd.DataFrame({"y01": y2 - 0.3, "y02": y2, "y05": y5, "y06": 0.8 * y5 + 0.2 * y10,
                         "y07": 0.6 * y5 + 0.4 * y10, "y08": 0.4 * y5 + 0.6 * y10, "y10": y10},
                        index=dates)
    eq = 1000 * np.exp(np.cumsum(z[:, 3] / 100 + 0.0003))
    dax = pd.Series(eq, index=dates, name="close")
    cash = pd.Series(np.clip(y2 - 0.5, 0, None), index=dates, name="rate")
    # inflation: higher when monetary and common-premium shocks are strong
    m = pd.Series(sig[:, 1] + sig[:, 3], index=dates).resample("ME").mean()
    rng = np.random.default_rng(seed + 1)
    infl = 1.0 + 2.2 * (m - 1.2) + 0.4 * rng.standard_normal(len(m))
    hicp = pd.Series(infl.to_numpy(), index=m.index.to_period("M"), name="yoy")
    raw = {"curve": grid, "dax": dax, "cash": cash, "hicp": hicp.to_timestamp(how="end")}
    return raw, pd.DataFrame(sig, index=dates, columns=["g", "m", "pp", "pc"])
