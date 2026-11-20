"""Allocation rules (section 4 of the note).

Hedge regime (C = 0)          : 60 % equities, 40 % bonds
Cash regime, variant A (C = 1) : k_t x (60/40), rest in cash       k_t = min(1, sigma_ref / sigma_hat_6040)
Cash regime, variant B (C = 1) : equities min(60 %, sigma_ref / sigma_hat_e), no bonds, rest in cash
Volatility target              : k_t x (60/40) at every date
Intermediate C (0.5, prudent variant) mixes the two weight vectors linearly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params


def risk_inputs(daily: pd.DataFrame, month_ends: pd.DatetimeIndex, params: Params) -> pd.DataFrame:
    """sigma_ref (expanding, up to t), 63-day ex-ante vols of the 60/40 and of equities, k_t."""
    we, wb = params.w_equity, params.w_bond
    r6040 = we * daily["r_e"] + wb * daily["r_b"]
    exp_std = r6040.expanding(min_periods=params.min_ref_days).std() * np.sqrt(252)
    cov = daily[["r_e", "r_b"]].rolling(params.recent).cov()
    rows = []
    for d in month_ends:
        if d not in daily.index:
            continue
        c = cov.loc[d].to_numpy()
        s6040 = np.sqrt(max(we ** 2 * c[0, 0] + wb ** 2 * c[1, 1] + 2 * we * wb * c[0, 1], 0) * 252)
        se = np.sqrt(c[0, 0] * 252)
        rows.append(pd.Series({"sigma_ref": exp_std.loc[d], "sigma_6040": s6040, "sigma_e": se}, name=d))
    out = pd.DataFrame(rows)
    out["k"] = np.minimum(1.0, out["sigma_ref"] / out["sigma_6040"])
    return out


def target_weights(C: float, risk: pd.Series, variant: str, params: Params) -> tuple[float, float]:
    """(w_e, w_b) for a regime value C in [0, 1] and a variant 'A', 'B' or 'VT'."""
    we, wb = params.w_equity, params.w_bond
    k = risk["k"]
    if not np.isfinite(k):
        k = 1.0
    if variant == "VT":
        return k * we, k * wb
    if variant == "A":
        cash_w = (k * we, k * wb)
    elif variant == "B":
        eq = min(we, risk["sigma_ref"] / risk["sigma_e"]) if np.isfinite(risk["sigma_ref"]) else we
        cash_w = (eq, 0.0)
    else:
        raise ValueError(variant)
    C = 0.0 if not np.isfinite(C) else float(C)
    return (1 - C) * we + C * cash_w[0], (1 - C) * wb + C * cash_w[1]
