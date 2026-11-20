"""One monthly engine for every strategy (section 4.4 and 5 of the note).

Weights decided at month-end t are held (buy and hold) over month t+1. At the next month-end,
the drifted weights are rebalanced to the new targets; the cost is cost_bp per unit of
turnover on the equity and bond legs (futures overlay).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .allocation import target_weights
from .config import Params


@dataclass
class Strategy:
    name: str
    signal: str | None      # column of the signals table, None = always hedge regime
    variant: str            # 'A', 'B' or 'VT'
    label: str


def default_strategies(signals: pd.DataFrame) -> list[Strategy]:
    s = [Strategy("60/40", None, "A", "60/40 statique"),
         Strategy("VC", None, "VT", "60/40 à volatilité cible")]
    cand = [("CORR", "C_corr", "corrélation glissante"), ("MACRO", "C_macro", "règle macro"),
            ("HMM", "C_hmm", "HMM filtré"), ("P8-HB", "C_hb", "P8-HB"),
            ("P8-BETA", "C_beta", "P8-bêta"), ("P8-HB-PRUDENT", "C_hb_prudent", "P8-HB prudent"),
            ("ORACLE", "C_oracle", "oracle")]
    for name, col, lab in cand:
        if col in signals:
            for v in ("A", "B"):
                s.append(Strategy(f"{name}-{v}", col, v, f"{lab}, variante {v}"))
    return s


def run_backtest(monthly: pd.DataFrame, signals: pd.DataFrame, risk: pd.DataFrame,
                 strategies: list[Strategy], params: Params, start=None, end=None) -> dict:
    """monthly: returns per month (r_e, r_b, r_f) indexed by month-end.
    Returns {name: DataFrame indexed by the month-end at which the holding period ends}."""
    dec = signals.index.intersection(risk.index)
    dec = dec[risk.loc[dec, "sigma_ref"].notna().to_numpy()]
    if start is not None:
        dec = dec[dec >= pd.Timestamp(start)]
    if end is not None:
        dec = dec[dec <= pd.Timestamp(end)]
    mi = monthly.index
    pos = mi.get_indexer(dec)
    ok = (pos >= 0) & (pos + 1 < len(mi))
    dec, pos = dec[ok], pos[ok]
    hold = mi[pos + 1]
    R = monthly.loc[hold, ["r_e", "r_b", "r_f"]].to_numpy()
    cost = params.cost_bp / 1e4
    out = {}
    for st in strategies:
        rows = []
        w_drift = None
        for i, (d, h) in enumerate(zip(dec, hold)):
            C = 0.0 if st.signal is None else signals.at[d, st.signal]
            C = 0.0 if not np.isfinite(C) else C
            we, wb = target_weights(C, risk.loc[d], st.variant, params)
            if st.variant == "VT":
                C = np.nan
            if w_drift is None:
                turnover = 0.0
            else:
                turnover = abs(we - w_drift[0]) + abs(wb - w_drift[1])
            re, rb, rf = R[i]
            gross = we * re + wb * rb + (1 - we - wb) * rf
            net = gross - cost * turnover
            w_drift = (we * (1 + re) / (1 + gross), wb * (1 + rb) / (1 + gross))
            rows.append((h, d, C, risk.at[d, "k"], we, wb, turnover, net, rf, net - rf, re, rb))
        df = pd.DataFrame(rows, columns=["date", "decision", "C", "k", "w_e", "w_b", "turnover",
                                         "R", "R_f", "x", "R_e", "R_b"]).set_index("date")
        df.attrs["label"] = st.label
        out[st.name] = df
    return out
