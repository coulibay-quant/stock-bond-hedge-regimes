"""Robustness and discipline checks.

1. Deflated Sharpe ratio (Bailey and López de Prado, 2014): is a Sharpe ratio (or an information
   ratio against a benchmark) still significant once we account for the number of strategies
   tried, the sample length, skewness and fat tails?
2. Regime switches and costs: how often each signal flips, how much turnover the flips create,
   and the transaction cost at which the advantage over the benchmark disappears.
3. Stability of the HMM: sensitivity to the number of states, the estimation window and the
   initialisation, and stability of the regimes between successive re-estimations.
4. Stability of P8: sensitivity of the regime to the structure and intensity windows (W, h).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import Strategy, run_backtest
from .config import Params

EULER = 0.5772156649015329
NOT_TRIALS = ("60/40", "ORACLE-A", "ORACLE-B")      # benchmark and infeasible strategies


# ============================================================================ 1. deflated Sharpe
def psr(x: np.ndarray, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true Sharpe > sr_star), per-period (monthly) Sharpe ratios."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)               # normal = 3
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def expected_max_sharpe(sr_trials: np.ndarray, n_trials: float) -> float:
    """SR0: expected maximum of n_trials Sharpe ratios under the null of no skill."""
    v = np.var(sr_trials, ddof=1) if len(sr_trials) > 1 else 0.0
    if n_trials <= 1 or v <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(v) * ((1 - EULER) * z1 + EULER * z2))


def effective_trials(series: pd.DataFrame) -> float:
    """Approximate number of independent trials from the average pairwise correlation:
    N_eff = rho_bar + (1 - rho_bar) N."""
    c = series.corr().to_numpy()
    n = c.shape[0]
    if n < 2:
        return float(n)
    rho = (c.sum() - n) / (n * (n - 1))
    return float(rho + (1 - rho) * n)


def deflated_sharpe_table(res: dict, benchmark: str | None = None) -> pd.DataFrame:
    """DSR for every candidate strategy.

    benchmark=None: Sharpe ratio of the strategy's excess return over cash.
    benchmark='60/40' or 'VC': information ratio of the active return (strategy minus benchmark).
    The trials are all candidate strategies (benchmark and oracle excluded)."""
    names = [n for n in res if n not in NOT_TRIALS and n != benchmark]
    series = {}
    for n in names:
        x = res[n]["x"]
        series[n] = x - res[benchmark]["x"] if benchmark else x
    df = pd.DataFrame(series).dropna()
    sr = df.mean() / df.std(ddof=1)
    N = len(names)
    N_eff = effective_trials(df)
    sr0, sr0_eff = expected_max_sharpe(sr.to_numpy(), N), expected_max_sharpe(sr.to_numpy(), N_eff)
    rows = []
    for n in names:
        x = df[n].to_numpy()
        rows.append({"stratégie": n,
                     ("ratio d'information annualisé" if benchmark else "Sharpe annualisé"): sr[n] * np.sqrt(12),
                     "PSR (seuil 0)": psr(x, 0.0),
                     f"DSR (N = {N})": psr(x, sr0),
                     f"DSR (N effectif = {N_eff:.1f})": psr(x, sr0_eff),
                     "asymétrie": stats.skew(x), "kurtosis": stats.kurtosis(x, fisher=False)})
    out = pd.DataFrame(rows).set_index("stratégie")
    out.attrs.update({"N": N, "N_eff": N_eff, "SR0 mensuel": sr0, "SR0 mensuel (N eff)": sr0_eff,
                      "mois": len(df), "référence": benchmark or "cash"})
    return out


# ============================================================================ 2. switches and costs
def switch_stats(res: dict, cost_bp: float = 2.0) -> pd.DataFrame:
    """Regime flips, whipsaws and the turnover they create, for every strategy with a regime."""
    rows = []
    for n, df in res.items():
        C = df["C"]
        if C.isna().all():
            continue
        C = C.fillna(0)
        ch = C.diff().abs().fillna(0) > 0
        months = len(C)
        n_sw = int(ch.sum())
        # whipsaw: the previous regime comes back within two months
        back = 0
        vals = C.to_numpy()
        for i in np.flatnonzero(ch.to_numpy()):
            prev = vals[i - 1]
            if any(vals[j] == prev for j in range(i + 1, min(i + 3, months))):
                back += 1
        to = df["turnover"]
        rows.append({
            "stratégie": n, "bascules par an": 12 * n_sw / months,
            "durée moyenne d'un régime (mois)": months / (n_sw + 1),
            "allers-retours en 2 mois (%)": 100 * back / n_sw if n_sw else 0.0,
            "rotation annuelle": 12 * to.mean(),
            "part de la rotation due aux bascules (%)": 100 * to[ch].sum() / to.sum() if to.sum() > 0 else 0.0,
            f"coût annuel à {cost_bp:g} pb (pb/an)": 12 * to.mean() * cost_bp,
        })
    return pd.DataFrame(rows).set_index("stratégie")


def cost_grid(monthly: pd.DataFrame, signals: pd.DataFrame, risk: pd.DataFrame, strategies: list[Strategy],
              params: Params, costs=(0, 2, 5, 10, 20, 50)) -> pd.DataFrame:
    """Annualized Sharpe ratio of every strategy for several costs per unit of turnover (bp)."""
    out = {}
    for c in costs:
        res = run_backtest(monthly, signals, risk, strategies, replace(params, cost_bp=float(c)))
        out[c] = {n: d["x"].mean() / d["x"].std() * np.sqrt(12) for n, d in res.items()}
    df = pd.DataFrame(out)
    df.columns = [f"{c} pb" for c in costs]
    return df


def break_even_costs(grid: pd.DataFrame, refs=("60/40", "VC")) -> pd.DataFrame:
    """Cost (bp per unit of turnover) at which each strategy's Sharpe falls to the reference's,
    by linear interpolation on the cost grid."""
    costs = np.array([float(c.split()[0]) for c in grid.columns])
    rows = {}
    for n in grid.index:
        row = {}
        for ref in refs:
            if ref not in grid.index or n == ref:
                continue
            d = (grid.loc[n] - grid.loc[ref]).to_numpy()
            if d[0] <= 0:
                row[f"seuil contre {ref}"] = "aucun avantage sans coûts"
            elif (d > 0).all():
                row[f"seuil contre {ref}"] = f"> {costs[-1]:g} pb"
            else:
                k = int(np.argmax(d <= 0))
                c = costs[k - 1] + d[k - 1] * (costs[k] - costs[k - 1]) / (d[k - 1] - d[k])
                row[f"seuil contre {ref}"] = f"{c:.1f} pb"
        rows[n] = row
    return pd.DataFrame(rows).T


# ============================================================================ 3. HMM stability
DEFAULT_HMM_CONFIGS = [
    ("base : 2 états, fenêtre croissante", dict()),
    ("3 états", dict(n_states=3)),
    ("fenêtre glissante 10 ans", dict(window=120)),
    ("fenêtre glissante 15 ans", dict(window=180)),
    ("initialisation unique, graine 1", dict(n_init=1, seed_offset=1000)),
    ("initialisation unique, graine 2", dict(n_init=1, seed_offset=2000)),
    ("initialisation unique, graine 3", dict(n_init=1, seed_offset=3000)),
]


def hmm_stability(monthly: pd.DataFrame, risk: pd.DataFrame, params: Params, decision_index: pd.DatetimeIndex,
                  configs=DEFAULT_HMM_CONFIGS, verbose: bool = True) -> dict:
    """Run the HMM signal under several configurations and compare them.

    Returns {"table": summary per configuration, "stability": re-estimation log of the base run,
    "signals": regime series per configuration}."""
    from .signals import hmm_run
    regimes, logs, rows = {}, {}, []
    for label, kw in configs:
        if verbose:
            print(f"  HMM : {label}", flush=True)
        r = hmm_run(monthly, params, track=True, **kw)
        p = r["p"].reindex(decision_index)
        C = (p > 0.5).astype(float)
        C[p.isna()] = np.nan
        regimes[label], logs[label] = C, r["stability"]
    sig = pd.DataFrame(regimes)
    base = sig.columns[0]
    strategies = [Strategy(label, label, "A", label) for label in sig.columns]
    res = run_backtest(monthly, sig, risk, strategies, params)
    for label in sig.columns:
        both = pd.concat([sig[base].rename("a"), sig[label].rename("b")], axis=1).dropna()
        st = logs[label]
        c = sig[label].dropna()
        rows.append({
            "configuration": label,
            "accord avec la base (%)": 100 * (both["a"] == both["b"]).mean() if len(both) else np.nan,
            "ré-estimations": len(st),
            "temps en cash (%)": 100 * c.mean() if len(c) else np.nan,
            "bascules par an": 12 * (c.diff().abs() > 0).sum() / len(c) if len(c) else np.nan,
            "accord entre ré-estimations (%)": 100 * st["accord passé"].mean() if len(st) else np.nan,
            "pire accord entre ré-estimations (%)": 100 * st["accord passé"].min() if len(st) else np.nan,
            "ré-estimations qui changent le régime courant (%)":
                100 * st["régime courant change"].mean() if len(st) else np.nan,
            "Sharpe HMM-A": res[label]["x"].mean() / res[label]["x"].std() * np.sqrt(12),
        })
    table = pd.DataFrame(rows).set_index("configuration")
    return {"table": table, "stability": logs[base], "signals": sig}


# ============================================================================ 4. P8 window sensitivity
DEFAULT_P8_GRID = [(500, 63), (250, 63), (750, 63), (500, 42), (500, 126)]


def indicator_sensitivity(panel, params: Params, monthly: pd.DataFrame, risk: pd.DataFrame, paths=None,
                          grid=DEFAULT_P8_GRID, n_accept: int = 300, verbose: bool = True) -> pd.DataFrame:
    """Recompute HB for several (W, h) and compare the P8-HB regimes. Each run is cached."""
    from .indicator import run_indicator
    hbs, regimes = {}, {}
    for W, h in grid:
        p = replace(params, window=W, recent=h, n_accept=n_accept)
        if verbose:
            print(f"  P8 : W = {W} jours, h = {h} jours", flush=True)
        ind = run_indicator(panel, p, paths, use_cache=True, progress=False)
        label = f"W = {W}, h = {h}"
        hbs[label] = ind["hb"]
        C = (ind["hb"].ffill() >= params.hb_threshold).astype(float)
        C[ind["hb"].ffill().isna()] = np.nan
        regimes[label] = C
    hb, sig = pd.DataFrame(hbs), pd.DataFrame(regimes)
    common = sig.dropna().index
    sig = sig.loc[common]
    base = sig.columns[0]
    res = run_backtest(monthly, sig, risk, [Strategy(c, c, "A", c) for c in sig.columns], params)
    rows = []
    for c in sig.columns:
        rows.append({"fenêtres": c,
                     "corrélation de HB avec la base": hb[c].loc[common].corr(hb[base].loc[common]),
                     "accord des régimes avec la base (%)": 100 * (sig[c] == sig[base]).mean(),
                     "temps en cash (%)": 100 * sig[c].mean(),
                     "bascules par an": 12 * (sig[c].diff().abs() > 0).sum() / len(sig),
                     "Sharpe P8-HB-A": res[c]["x"].mean() / res[c]["x"].std() * np.sqrt(12)})
    out = pd.DataFrame(rows).set_index("fenêtres")
    out.attrs["début commun"] = str(common[0].date()) if len(common) else ""
    return out
