"""Regime signals of all competitors (section 5.2 of the note).

Every signal is indexed by month-end t and uses only information available at t.
Value 1 = "cash" regime, 0 = "hedge" regime, 0.5 = uncertain (prudent variant only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params
from .dataset import monthly_returns


# ============================================================================ HMM
def _gauss_logpdf(Y: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = Y.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (Y - mu).T)
    return -0.5 * (z ** 2).sum(axis=0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


def hmm_fit(Y: np.ndarray, n_states: int = 2, n_init: int = 6, n_iter: int = 300,
            tol: float = 1e-6, seed: int = 0) -> dict:
    """Gaussian HMM fitted by EM (Baum-Welch) with scaled forward-backward."""
    T, d = Y.shape
    rng = np.random.default_rng(seed)
    best, best_ll = None, -np.inf
    reg = 1e-6 * np.eye(d) * Y.var(axis=0).mean()
    for _ in range(n_init):
        gamma = rng.dirichlet(np.ones(n_states), size=T)
        pi = gamma[0]
        A = np.full((n_states, n_states), 0.1 / (n_states - 1))
        np.fill_diagonal(A, 0.9)
        mu = np.array([(gamma[:, k:k + 1] * Y).sum(0) / gamma[:, k].sum() for k in range(n_states)])
        cov = np.array([np.cov(Y.T, aweights=gamma[:, k]) + reg for k in range(n_states)])
        ll_old = -np.inf
        for _ in range(n_iter):
            logb = np.column_stack([_gauss_logpdf(Y, mu[k], cov[k]) for k in range(n_states)])
            m = logb.max(axis=1, keepdims=True)
            b = np.exp(logb - m)
            alpha = np.empty((T, n_states)); c = np.empty(T)
            a = pi * b[0]; c[0] = a.sum(); alpha[0] = a / c[0]
            for t in range(1, T):
                a = (alpha[t - 1] @ A) * b[t]; c[t] = a.sum(); alpha[t] = a / c[t]
            beta = np.ones((T, n_states))
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
            ll = np.log(c).sum() + m.sum()
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (alpha[:-1, :, None] * A[None] * (b[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
            pi = gamma[0]
            A = xi.sum(0) / xi.sum(0).sum(axis=1, keepdims=True)
            for k in range(n_states):
                w = gamma[:, k]
                mu[k] = (w[:, None] * Y).sum(0) / w.sum()
                diff = Y - mu[k]
                cov[k] = (w[:, None, None] * np.einsum("ti,tj->tij", diff, diff)).sum(0) / w.sum() + reg
            if ll - ll_old < tol * abs(ll):
                break
            ll_old = ll
        if ll > best_ll:
            best_ll, best = ll, {"pi": pi.copy(), "A": A.copy(), "mu": mu.copy(), "cov": cov.copy(), "ll": ll}
    return best


def hmm_filter(Y: np.ndarray, model: dict) -> np.ndarray:
    """Filtered probabilities P(s_t | y_1..y_t) (Hamilton filter). Never uses later data."""
    n = len(model["pi"])
    logb = np.column_stack([_gauss_logpdf(Y, model["mu"][k], model["cov"][k]) for k in range(n)])
    b = np.exp(logb - logb.max(axis=1, keepdims=True))
    xi = np.empty((len(Y), n))
    prior = model["pi"]
    for t in range(len(Y)):
        a = prior * b[t]
        xi[t] = a / a.sum()
        prior = xi[t] @ model["A"]
    return xi


def _cash_state(model: dict) -> int:
    """State with the highest stock-bond correlation (the 'cash' regime)."""
    corr = [m[0, 1] / np.sqrt(m[0, 0] * m[1, 1]) for m in model["cov"]]
    return int(np.argmax(corr))


def hmm_run(monthly: pd.DataFrame, params: Params, n_states: int = 2, window: int | None = None,
            n_init: int = 6, seed_offset: int = 0, track: bool = False) -> dict:
    """Real-time HMM signal with optional stability tracking.

    Refitted every ``hmm_refit_every`` months on the data available at that date (expanding window,
    or the last ``window`` months). Each month, the probability of the high-correlation state is
    filtered with the current parameters. With ``track=True``, every refit is compared with the
    previous parameters on the same data: share of past months whose regime is unchanged, and
    whether the current regime flips only because the model was re-estimated."""
    Y_all = np.column_stack([monthly["r_e"] - monthly["r_f"], monthly["r_b"] - monthly["r_f"]]) * 100
    idx = monthly.index
    out = pd.Series(np.nan, index=idx, name="p_hmm")
    model, cash, rows = None, None, []
    for t in range(params.hmm_min_months, len(idx)):
        lo = 0 if window is None else max(0, t + 1 - window)
        if model is None or (t - params.hmm_min_months) % params.hmm_refit_every == 0:
            new = hmm_fit(Y_all[lo:t + 1], n_states=n_states, n_init=n_init, seed=t + seed_offset)
            new_cash = _cash_state(new)
            if track and model is not None:
                old_reg = hmm_filter(Y_all[lo:t + 1], model)[:, cash] > 0.5
                new_reg = hmm_filter(Y_all[lo:t + 1], new)[:, new_cash] > 0.5
                rows.append({"date": idx[t], "accord passé": float((old_reg == new_reg).mean()),
                             "mois comparés": int(len(old_reg)),
                             "régime courant change": bool(old_reg[-1] != new_reg[-1])})
            model, cash = new, new_cash
        out.iloc[t] = hmm_filter(Y_all[lo:t + 1], model)[-1, cash]
    stab = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return {"p": out, "stability": stab}


def signal_hmm(monthly: pd.DataFrame, params: Params) -> pd.Series:
    """Probability of the high-correlation state, filtered, refitted every ``hmm_refit_every``
    months on an expanding window (data up to the refit date only). Base configuration."""
    return hmm_run(monthly, params)["p"]


# ============================================================================ all signals
def build_signals(panel, ind: pd.DataFrame, rs: pd.DataFrame, params: Params,
                  with_hmm: bool = True) -> pd.DataFrame:
    """Signals at month-ends. ``ind`` = run_indicator output, ``rs`` = realized_stats output."""
    idx = ind.index
    sig = pd.DataFrame(index=idx)
    sig["C_corr"] = (rs["corr"].reindex(idx) > 0).astype(float)

    if panel.inflation is not None:
        infl = panel.inflation.copy()
        lagged = infl.shift(1)                              # month t-1, published during month t
        per = idx.to_period("M")
        sig["inflation_t-1"] = lagged.reindex(per).ffill(limit=2).to_numpy()
        sig["C_macro"] = (sig["inflation_t-1"] > params.macro_threshold).astype(float)
        sig.loc[sig["inflation_t-1"].isna(), "C_macro"] = np.nan

    hb = ind["hb"].ffill()
    sig["C_hb"] = (hb >= params.hb_threshold).astype(float)
    sig.loc[hb.isna(), "C_hb"] = np.nan
    lo, hi = ind["hb_p16"].ffill(), ind["hb_p84"].ffill()
    pr = sig["C_hb"].copy()
    pr[(lo < params.hb_threshold) & (hi >= params.hb_threshold)] = 0.5
    sig["C_hb_prudent"] = pr

    if "beta_fcst" in ind:
        bf = ind["beta_fcst"]
        sig["C_beta"] = np.where(bf.notna(), (bf > 0).astype(float), sig["C_hb"])
        sig["beta_actif"] = bf.notna()
    else:
        sig["C_beta"] = sig["C_hb"]
        sig["beta_actif"] = False

    # oracle: realized correlation over the next calendar month
    d = panel.daily
    nxt = d[["r_e", "r_b"]].groupby(d.index.to_period("M")).apply(lambda x: x["r_e"].corr(x["r_b"]))
    per = idx.to_period("M")
    sig["C_oracle"] = (nxt.reindex(per + 1).to_numpy() > 0).astype(float)
    sig.loc[nxt.reindex(per + 1).isna().to_numpy(), "C_oracle"] = np.nan

    if with_hmm:
        mon = monthly_returns(d)
        p = signal_hmm(mon, params)
        sig["p_hmm"] = p.reindex(idx).to_numpy()
        sig["C_hmm"] = (sig["p_hmm"] > 0.5).astype(float)
        sig.loc[sig["p_hmm"].isna(), "C_hmm"] = np.nan
    return sig
