"""Performance measures, statistical tests and diagnostics (section 5 of the note)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import EPISODES, HEDGE_BREAKING, SHOCKS, Params


# ============================================================================ measures
def max_drawdown(R: pd.Series) -> float:
    nav = (1 + R).cumprod()
    return float((1 - nav / nav.cummax()).max())


def metrics(df: pd.DataFrame, crisis_months: pd.Index | None = None) -> dict:
    R, x = df["R"], df["x"]
    n = len(R)
    ann_ret = (1 + R).prod() ** (12 / n) - 1 if n else np.nan
    mdd = max_drawdown(R)
    q = R.quantile(0.05)
    out = {
        "rendement (%/an)": 100 * ann_ret,
        "volatilité (%/an)": 100 * R.std() * np.sqrt(12),
        "Sharpe": x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan,
        "perte max (%)": 100 * mdd,
        "CVaR 5 % (%/mois)": -100 * R[R <= q].mean(),
        "Calmar": ann_ret / mdd if mdd > 0 else np.nan,
        "rotation (/an)": 12 * df["turnover"].mean(),
        "temps en cash (%)": 100 * (df["C"] > 0).mean() if df["C"].notna().any() else np.nan,
        "mois": n,
    }
    if crisis_months is not None:
        cm = R.index.intersection(crisis_months)
        out["pires mois actions (%/mois)"] = 100 * R.loc[cm].mean() if len(cm) else np.nan
    return out


def crisis_index(res: dict, q: float = 0.10) -> pd.Index:
    """Months in the worst q of equity excess returns (same months for every strategy)."""
    ref = next(iter(res.values()))
    xe = ref["R_e"] - ref["R_f"]
    return xe[xe <= xe.quantile(q)].index


def perf_table(res: dict, start=None, end=None) -> pd.DataFrame:
    sub = {k: v.loc[start:end] if (start or end) else v for k, v in res.items()}
    cm = crisis_index(sub)
    return pd.DataFrame({k: metrics(v, cm) for k, v in sub.items()}).T


# ============================================================================ tests
def _circular_blocks(T: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
    return idx % T


def sharpe_diff_test(x1: pd.Series, x2: pd.Series, block: int = 6, reps: int = 2000, seed: int = 0) -> dict:
    """Difference of annualized Sharpe ratios with a paired circular block bootstrap.

    p-value: two-sided, from the bootstrap distribution of the difference re-centred at zero."""
    x = pd.concat([x1, x2], axis=1).dropna().to_numpy()
    T = len(x)
    sr = lambda a: a.mean(0) / a.std(0, ddof=1) * np.sqrt(12)
    d0 = sr(x)[0] - sr(x)[1]
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for r in range(reps):
        s = sr(x[_circular_blocks(T, block, rng)])
        boot[r] = s[0] - s[1]
    se = boot.std(ddof=1)
    p = float(np.mean(np.abs(boot - boot.mean()) >= abs(d0)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"écart de Sharpe": d0, "IC 95 % bas": lo, "IC 95 % haut": hi, "écart-type": se, "p-valeur": p}


def tests_table(res: dict, refs=("60/40", "VC"), params: Params | None = None, start=None, end=None) -> pd.DataFrame:
    params = params or Params()
    rows = []
    for name, df in res.items():
        for ref in refs:
            if name == ref or ref not in res:
                continue
            a = df["x"].loc[start:end] if (start or end) else df["x"]
            b = res[ref]["x"].loc[start:end] if (start or end) else res[ref]["x"]
            t = sharpe_diff_test(a, b, params.block, params.bootstrap_reps)
            rows.append({"stratégie": name, "contre": ref, **t})
    return pd.DataFrame(rows).set_index(["stratégie", "contre"])


def vt_decomposition(res: dict, p8: str = "P8-HB-A") -> pd.DataFrame:
    """Proposition 4: P8-A minus the volatility target is (1-C)(1-k) x_6040, P8-A minus 60/40 is
    -C (1-k) x_6040 (before costs). Reports the terms in % per year and the months involved."""
    if p8 not in res or "VC" not in res or "60/40" not in res:
        return pd.DataFrame()
    a, vt, base = res[p8], res["VC"], res["60/40"]
    x6040 = base["R"] - base["R_f"]
    C, k = a["C"].fillna(0), a["k"].fillna(1)
    t_vt = (1 - C) * (1 - k) * x6040
    t_6040 = -C * (1 - k) * x6040
    rows = {
        "écart observé P8-A moins VC (%/an)": 12 * 100 * (a["x"] - vt["x"]).mean(),
        "terme de la proposition 4 (%/an)": 12 * 100 * t_vt.mean(),
        "mois « couverture, vol. haute » (C=0, k<1)": int(((C == 0) & (k < 1)).sum()),
        "rendement moyen du 60/40 ces mois-là (%/mois)": 100 * x6040[(C == 0) & (k < 1)].mean(),
        "écart observé P8-A moins 60/40 (%/an)": 12 * 100 * (a["x"] - base["x"]).mean(),
        "terme de la proposition 4 face au 60/40 (%/an)": 12 * 100 * t_6040.mean(),
        "mois « cash, vol. haute » (C=1, k<1)": int(((C == 1) & (k < 1)).sum()),
        "rendement moyen du 60/40 ces mois-là (%/mois) ": 100 * x6040[(C == 1) & (k < 1)].mean(),
    }
    return pd.Series(rows, name=p8).to_frame()


# ============================================================================ forecasting test (criterion 3)
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def clark_west(y: np.ndarray, f1: np.ndarray, f2: np.ndarray, lags: int = 2) -> dict:
    """Clark-West (2007) test that the larger model (f2) does not improve on the nested one (f1).
    One-sided, Newey-West standard error with ``lags`` lags."""
    f = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    T = len(f)
    fm = f - f.mean()
    v = fm @ fm / T
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (fm[L:] @ fm[:-L]) / T
    t = f.mean() / np.sqrt(v / T)
    return {"stat CW": t, "p-valeur": 1 - stats.norm.cdf(t)}


def forecast_paths(ind: pd.DataFrame, rs: pd.DataFrame, params: Params,
                   target: str = "corr_next") -> dict[str, pd.DataFrame]:
    """Out-of-sample forecasts, one DataFrame (index = decision date; columns y, f1, f2) per model 2.

    Model 1: y = a + b * (last-quarter value).  Model 2: adds HB_t, or the structural beta forecast.
    Expanding OLS from the start of the sample; a pair (t, y_t) enters the estimation only once y_t
    is observed (3 months later)."""
    base = "corr" if target == "corr_next" else "beta"
    df = pd.concat([rs[[base, target]], ind[["hb", "beta_fcst"]] if "beta_fcst" in ind else ind[["hb"]]],
                   axis=1).dropna(subset=[base, "hb"])
    h = params.beta_horizon
    out = {}
    for extra in ["hb", "beta_fcst"]:
        if extra not in df:
            continue
        d = df.dropna(subset=[extra])
        y = d[target].to_numpy()
        X1 = np.column_stack([np.ones(len(d)), d[base]])
        X2 = np.column_stack([X1, d[extra]])
        f1, f2, yy, dates = [], [], [], []
        for t in range(params.forecast_min_obs + h, len(d)):
            tr = slice(0, t - h + 1)                     # targets known by month t
            ok = np.isfinite(y[tr])
            if ok.sum() < params.forecast_min_obs or not np.isfinite(y[t]):
                continue
            b1 = _ols(X1[tr][ok], y[tr][ok]); b2 = _ols(X2[tr][ok], y[tr][ok])
            f1.append(X1[t] @ b1); f2.append(X2[t] @ b2); yy.append(y[t]); dates.append(d.index[t])
        if yy:
            out[f"+ {extra}"] = pd.DataFrame({"y": yy, "f1": f1, "f2": f2}, index=pd.DatetimeIndex(dates))
    return out


def _forecast_row(p: pd.DataFrame, lags: int) -> dict:
    yy, f1, f2 = p["y"].to_numpy(), p["f1"].to_numpy(), p["f2"].to_numpy()
    return {"prévisions": len(yy), "début": p.index[0].date(), "fin": p.index[-1].date(),
            "RMSE modèle 1": np.sqrt(np.mean((yy - f1) ** 2)),
            "RMSE modèle 2": np.sqrt(np.mean((yy - f2) ** 2)),
            "R² hors échantillon": 1 - np.sum((yy - f2) ** 2) / np.sum((yy - f1) ** 2),
            **clark_west(yy, f1, f2, lags=lags)}


def forecast_test(ind: pd.DataFrame, rs: pd.DataFrame, params: Params, target: str = "corr_next") -> pd.DataFrame:
    """Criterion 3 on the whole out-of-sample period: RMSE, out-of-sample R², Clark-West test."""
    results = {}
    for name, p in forecast_paths(ind, rs, params, target).items():
        if len(p) >= 24:
            row = _forecast_row(p, params.beta_horizon - 1)
            row.pop("fin")
            results[name] = row
    return pd.DataFrame(results).T


def default_forecast_periods(dates: pd.DatetimeIndex) -> list[tuple[str, str]]:
    """Two halves of the forecast sample (used when the market has no periods, e.g. demo mode)."""
    if len(dates) < 2:
        return []
    mid = dates[len(dates) // 2]
    return [(str(dates[0].date()), str(mid.date())), (str((mid + pd.Timedelta(days=1)).date()), "2100-01-01")]


def forecast_by_period(ind: pd.DataFrame, rs: pd.DataFrame, params: Params, periods=None,
                       target: str = "corr_next", min_obs: int = 24) -> pd.DataFrame:
    """Criterion 3 evaluated on sub-periods (robustness, not a pre-registered criterion).

    The forecasts are the same as in ``forecast_test`` (expanding estimation from the start of the
    sample); only the evaluation window changes. Periods with fewer than ``min_obs`` forecasts are
    reported with their count only."""
    paths = forecast_paths(ind, rs, params, target)
    if periods is None:
        first = next(iter(paths.values()), pd.DataFrame())
        periods = default_forecast_periods(first.index)
    rows = []
    for a, b in periods:
        label = f"{a[:4]}-{b[:4] if b < '2100' else 'fin'}"
        for name, p in paths.items():
            sub = p.loc[a:b]
            if len(sub) >= min_obs:
                row = _forecast_row(sub, params.beta_horizon - 1)
            else:
                row = {"prévisions": len(sub)}
            rows.append({"période": label, "modèle": name, **row})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["période", "modèle"])


# ============================================================================ episodes
def episodes_table(ind: pd.DataFrame, rs: pd.DataFrame, market: str, res: dict | None = None) -> pd.DataFrame:
    rows = []
    names = {"g": "croissance", "m": "monétaire", "pp": "prime de couverture", "pc": "prime commune"}
    for label, a, b, expected, mkt in EPISODES:
        if mkt != market:
            continue
        sl = ind.loc[a:b]
        if sl.empty or sl["hb"].isna().all():
            continue
        shares = sl[[f"s_{s}" for s in SHOCKS]].mean()
        dom = SHOCKS[int(np.argmax(shares.to_numpy()))]
        row = {"épisode": label, "lecture attendue": expected, "HB moyen": sl["hb"].mean(),
               "choc dominant": names[dom],
               **{f"part {names[s]}": shares[f"s_{s}"] for s in SHOCKS},
               "corr. réalisée": rs["corr"].reindex(sl.index).mean()}
        if res:
            for k in ("60/40", "VC", "P8-HB-A", "P8-BETA-A"):
                if k in res:
                    rr = res[k]["R"].loc[a:pd.Timestamp(b) + pd.offsets.MonthEnd(1)]
                    row[f"{k} (%)"] = 100 * ((1 + rr).prod() - 1) if len(rr) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("épisode") if rows else pd.DataFrame()


# ============================================================================ success criteria
def success_criteria(perf_full: pd.DataFrame, tests: pd.DataFrame, fc: pd.DataFrame,
                     p8_names=("P8-HB-A", "P8-HB-B", "P8-BETA-A", "P8-BETA-B")) -> pd.DataFrame:
    rows = []
    for n in p8_names:
        if n not in perf_full.index:
            continue
        p, b, v = perf_full.loc[n], perf_full.loc["60/40"], perf_full.loc["VC"]
        c1 = (p["Sharpe"] > b["Sharpe"]) and (p["perte max (%)"] < b["perte max (%)"])
        c2 = (p["Sharpe"] >= v["Sharpe"]) or (p["Calmar"] >= v["Calmar"])
        rows.append({"stratégie": n, "1. bat le 60/40 (Sharpe et perte max)": c1,
                     "2. au moins la vol. cible (Sharpe ou Calmar)": c2})
    out = pd.DataFrame(rows).set_index("stratégie") if rows else pd.DataFrame()
    if len(fc):
        c3 = {k: bool(r["p-valeur"] < 0.10) for k, r in fc.iterrows()}   # Clark-West, 10 %
        out.attrs["critère 3"] = c3
    return out
