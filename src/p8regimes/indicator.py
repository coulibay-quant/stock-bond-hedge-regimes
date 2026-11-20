"""Real-time indicator (sections 3.6-3.7 of the note).

At each month-end t, using only data up to t:
    1. VAR(1) and residual covariance on the last W days                (slow speed: structure)
    2. identified set of impact matrices B (sign + profile restrictions)
    3. shocks of the last h days, eps = B^-1 u, and their variances      (fast speed: intensities)
    4. HB^(j) = share of equity variance from m and p- for each admissible B; HB_t = median
    5. structural bond beta now and forecast for the next quarter (P8-beta)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import HEDGE_BREAKING, SHOCKS, Params, Paths, get_paths
from .dataset import Panel
from .structural import RotationBank, identified_set_fallback, median_target, var1


def bond_loadings(B: np.ndarray, duration: float) -> np.ndarray:
    """Effect of each shock on the bond pocket (%), from the yield rows of B (basis points).

    The pocket's yield is interpolated linearly between the 2/5/10-year rows."""
    D = duration
    if D <= 5:
        w = np.array([(5 - D) / 3, (D - 2) / 3, 0.0])
    else:
        w = np.array([0.0, (10 - D) / 5, (D - 5) / 5])
    dy = np.einsum("r,mrk->mk", w, B[:, :3, :])
    return -D * dy / 100.0


def shock_stats(B_all: np.ndarray, u_recent: np.ndarray, duration: float) -> dict:
    """Per-rotation quantities for the recent window."""
    Binv = np.linalg.inv(B_all)
    eps = np.einsum("mkn,tn->mtk", Binv, u_recent)
    var_k = eps.var(axis=1, ddof=1)                    # (M, 4) intensities (window mean = 1)
    b = B_all[:, 3, :]                                 # equity impacts
    c = bond_loadings(B_all, duration)                 # bond-pocket impacts
    contrib = b ** 2 * var_k
    s = contrib / contrib.sum(axis=1, keepdims=True)   # variance shares of equities
    h = c / b                                          # shock-specific hedge ratios
    return {"var": var_k, "b": b, "c": c, "s": s, "h": h,
            "hb": s[:, list(HEDGE_BREAKING)].sum(axis=1), "beta": (s * h).sum(axis=1)}


def fit_persistence(hist: pd.DataFrame, horizon: int, min_obs: int):
    """Direct h-step regression x_{s+h} = a + phi x_s for each shock, on past months only.

    Returns (a, phi) arrays of length 4, or None if there are not enough pairs."""
    x = hist.to_numpy()
    if len(x) - horizon < min_obs:
        return None
    X, Y = x[:-horizon], x[horizon:]
    a, phi = np.empty(4), np.empty(4)
    for k in range(4):
        xk, yk = X[:, k], Y[:, k]
        ok = np.isfinite(xk) & np.isfinite(yk)
        if ok.sum() < min_obs:
            return None
        xk, yk = xk[ok], yk[ok]
        vx = xk.var()
        p = np.cov(xk, yk, ddof=0)[0, 1] / vx if vx > 0 else 0.0
        phi[k] = min(max(p, 0.0), 1.0)
        a[k] = yk.mean() - phi[k] * xk.mean()
    return a, phi


def _row_from(st: dict, B_all: np.ndarray, rate: float, used: int) -> dict:
    row = {"hb": np.median(st["hb"]), "hb_p16": np.percentile(st["hb"], 16),
           "hb_p84": np.percentile(st["hb"], 84), "beta_now": np.median(st["beta"]),
           "n_adm": len(B_all), "acceptance": rate, "draws": used}
    mt = median_target(B_all)
    st_mt_idx = np.argmin(np.abs(B_all - mt).sum(axis=(1, 2)))
    row["hb_mt"] = st["hb"][st_mt_idx]
    for k, name in enumerate(SHOCKS):
        row[f"s_{name}"] = np.median(st["s"][:, k])
        row[f"var_{name}"] = np.median(st["var"][:, k])
        row[f"h_{name}"] = np.median(st["h"][:, k])
        row[f"b_{name}"] = np.median(st["b"][:, k])
    return row


def _cache_paths(paths: Paths, market: str, params: Params):
    key = params.indicator_key()
    return (paths.processed / f"indicator_{market}_{key}.csv",
            paths.processed / f"indicator_{market}_{key}.json")


def run_indicator(panel: Panel, params: Params, paths: Paths | None = None, use_cache: bool = True,
                  progress: bool = True, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Monthly real-time indicator. Cached in data/processed (incremental: only new month-ends
    are computed). ``dates`` restricts the month-ends (tests, quick runs)."""
    paths = paths or get_paths()
    daily = panel.daily
    Z = panel.z
    W, h = params.window, params.recent
    month_ends = panel.month_ends if dates is None else pd.DatetimeIndex(dates)
    pos_all = daily.index.get_indexer(month_ends)
    keep = (pos_all >= W) & (pos_all >= 0)
    month_ends, pos_all = month_ends[keep], pos_all[keep]

    csv_path, meta_path = _cache_paths(paths, panel.market, params)
    cached = pd.DataFrame()
    first_day = str(daily.index[0].date())
    if use_cache and dates is None and csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("first_day") == first_day:
            cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            # the last cached month may have been incomplete: recompute it
            if len(cached):
                cached = cached.iloc[:-1]
    todo = [(d, p) for d, p in zip(month_ends, pos_all) if d not in cached.index]

    bank = RotationBank(chunk=params.batch, seed=params.seed)
    rows, t0 = [], time.time()
    hist = cached[[f"var_{s}" for s in SHOCKS]].copy() if len(cached) else pd.DataFrame(
        columns=[f"var_{s}" for s in SHOCKS], dtype=float)
    for i, (d, p) in enumerate(todo):
        win = Z[p - W + 1:p + 1]
        _, _, u = var1(win)
        S = np.cov(u.T)
        B_all, rate, used, level = identified_set_fallback(S, bank, params.n_accept, params.max_draws,
                                                           params.share_restrictions, params.min_accept)
        if len(B_all) < params.min_accept:
            row = {"n_adm": len(B_all), "acceptance": rate, "draws": used, "level": level,
                   "flag": "aucun niveau de restriction admissible"}
            rows.append(pd.Series(row, name=d))
            hist.loc[d] = np.nan
            continue
        st = shock_stats(B_all, u[-h:], params.duration)
        row = _row_from(st, B_all, rate, used)
        row["level"] = level
        row["flag"] = "" if level <= (0 if params.share_restrictions else 1) else f"restrictions allégées (niveau {level})"
        # --- P8-beta: forecast of next-quarter shock intensities, shock by shock
        fit = fit_persistence(hist, params.beta_horizon, params.phi_min_obs)
        if fit is not None:
            a, phi = fit
            x_hat = np.maximum(a + phi * st["var"], 1e-4)
            contrib = st["b"] ** 2 * x_hat
            s_hat = contrib / contrib.sum(axis=1, keepdims=True)
            bf = (s_hat * st["h"]).sum(axis=1)
            row["beta_fcst"] = np.median(bf)
            row["beta_fcst_p16"], row["beta_fcst_p84"] = np.percentile(bf, [16, 84])
            for k, name in enumerate(SHOCKS):
                row[f"phi_{name}"] = phi[k]
        rows.append(pd.Series(row, name=d))
        hist.loc[d] = [row[f"var_{s}"] for s in SHOCKS]
        if progress and (i + 1) % 24 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"  {i + 1}/{len(todo)} dates  ({d.date()})  écoulé {el / 60:.1f} min, "
                  f"reste ~{eta / 60:.1f} min", flush=True)
    new = pd.DataFrame(rows)
    out = pd.concat([cached, new]).sort_index() if len(cached) else new.sort_index()
    out.index.name = "date"
    if dates is None and len(out):
        out.to_csv(csv_path, encoding="utf-8")
        meta_path.write_text(json.dumps({"first_day": first_day, "params": params.to_dict(),
                                         "computed": time.strftime("%Y-%m-%d %H:%M")},
                                        indent=2, default=str), encoding="utf-8")
    if progress:
        print(f"  indicateur : {len(out)} fins de mois ({len(new)} calculées, "
              f"{len(cached)} lues dans le cache), {(time.time() - t0) / 60:.1f} min")
    return out


def realized_stats(daily: pd.DataFrame, month_ends: pd.DatetimeIndex, window: int = 63) -> pd.DataFrame:
    """Realized correlation and beta of the bond pocket on equities over the last ``window``
    days (at each month-end) and over the next ``window`` days (forecast target)."""
    re, rb = daily["r_e"], daily["r_b"]
    pos = daily.index.get_indexer(month_ends)
    out = []
    n = len(daily)
    for d, p in zip(month_ends, pos):
        rec = {}
        if p + 1 >= window:
            e, b = re.iloc[p - window + 1:p + 1], rb.iloc[p - window + 1:p + 1]
            rec["corr"] = np.corrcoef(e, b)[0, 1]
            rec["beta"] = np.cov(b, e)[0, 1] / e.var()
            rec["vol_e"] = e.std() * np.sqrt(252)
        if p + window < n:
            e, b = re.iloc[p + 1:p + window + 1], rb.iloc[p + 1:p + window + 1]
            rec["corr_next"] = np.corrcoef(e, b)[0, 1]
            rec["beta_next"] = np.cov(b, e)[0, 1] / e.var()
        out.append(pd.Series(rec, name=d))
    return pd.DataFrame(out)
