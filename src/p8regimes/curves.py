"""Zero-coupon curves and the constant-maturity bond pocket (section 6.3 of the note).

Yields are in percent, treated as continuously compounded. The bond pocket is a zero-coupon
bond of constant maturity n (7 years by default), bought each day and sold the next day once
it has aged by Delta = calendar days / 365.25:

    r_t = n * y_{t-1}(n) - (n - Delta_t) * y_t(n - Delta_t)          (log return, decimals)

On a flat, unchanged curve at level y this gives the carry y * Delta_t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- Svensson
def svensson_yield(n, b0, b1, b2, b3, tau1, tau2):
    """Svensson (1994) zero-coupon yield for maturity n (years); same unit as the betas.

    Missing b3 / tau2 (early Fed data, Nelson-Siegel only) are treated as zero.
    All arguments broadcast (scalars or arrays of the same length).
    """
    n = np.asarray(n, dtype=float)
    tau1 = np.asarray(tau1, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    b3 = np.asarray(b3, dtype=float)
    x1 = n / tau1
    f1 = (1.0 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    ok3 = np.isfinite(b3) & np.isfinite(tau2) & (tau2 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x2 = n / np.where(ok3, tau2, 1.0)
        f3 = np.where(ok3, (1.0 - np.exp(-x2)) / x2 - np.exp(-x2), 0.0)
    return b0 + b1 * f1 + b2 * f2 + np.where(ok3, b3, 0.0) * f3


class SvenssonCurve:
    """Daily Svensson parameters (Federal Reserve GSW file)."""

    def __init__(self, params: pd.DataFrame):
        need = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
        p = params.copy()
        for c in ("BETA3", "TAU2"):
            if c not in p:
                p[c] = np.nan
        self.params = p[need].astype(float)
        self.index = self.params.index

    def yields(self, n, rows=None) -> np.ndarray:
        """Yields (%) at maturity n for each date (or for the selected integer rows)."""
        p = self.params.to_numpy() if rows is None else self.params.to_numpy()[rows]
        return svensson_yield(n, p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 5])

    def maturities(self, mats=(2, 5, 7, 10)) -> pd.DataFrame:
        return pd.DataFrame({f"y{m:02d}": self.yields(float(m)) for m in mats}, index=self.index)


class GridCurve:
    """Yields published on a grid of maturities (Bundesbank, ECB); linear interpolation."""

    def __init__(self, grid: pd.DataFrame):
        cols = sorted(grid.columns, key=lambda c: float(c.lstrip("y")))
        self.mats = np.array([float(c.lstrip("y")) for c in cols])
        self.values = grid[cols].astype(float).to_numpy()
        self.index = grid.index

    def yields(self, n, rows=None) -> np.ndarray:
        v = self.values if rows is None else self.values[rows]
        n = np.broadcast_to(np.asarray(n, dtype=float), (v.shape[0],))
        out = np.empty(v.shape[0])
        for i in range(v.shape[0]):           # np.interp is fast; loop over dates is fine
            out[i] = np.interp(n[i], self.mats, v[i])
        return out

    def maturities(self, mats=(2, 5, 7, 10)) -> pd.DataFrame:
        out = {}
        for m in mats:
            out[f"y{m:02d}"] = self.yields(float(m))
        return pd.DataFrame(out, index=self.index)


# ----------------------------------------------------------------------------- bond pocket
def bond_returns(curve, dates: pd.DatetimeIndex, n: float = 7.0) -> pd.Series:
    """Daily log return of the constant-maturity zero-coupon pocket between consecutive dates.

    ``dates`` must be a subset of the curve's index (the common trading calendar).
    Returns a Series indexed by dates[1:].
    """
    pos = curve.index.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("dates not all in the curve index")
    dt = np.diff(dates.values).astype("timedelta64[D]").astype(float) / 365.25
    y_prev = curve.yields(n, rows=pos[:-1]) / 100.0
    y_now = curve.yields(n - dt, rows=pos[1:]) / 100.0
    return pd.Series(n * y_prev - (n - dt) * y_now, index=dates[1:], name="r_b_log")


def cash_from_rate(rate_pct: pd.Series, dates: pd.DatetimeIndex, basis: float = 360.0) -> pd.Series:
    """Simple daily cash return between consecutive dates from an annual money-market rate
    (% per year, ACT/basis), using the rate known on the previous date."""
    r = rate_pct.reindex(rate_pct.index.union(dates)).sort_index().ffill().reindex(dates)
    days = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    out = r.to_numpy()[:-1] / 100.0 * days / basis
    return pd.Series(out, index=dates[1:], name="r_f")
