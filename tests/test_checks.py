"""Check of the 7-year zero-coupon pocket against a traded bond fund (IEF)."""
import sys
import types

import numpy as np
import pandas as pd

from p8regimes.checks import compare_bond_pocket
from p8regimes.config import Params
from p8regimes.data import yahoo_adjusted
from p8regimes.dataset import build_ea
from p8regimes.simulate import simulate_raw


def _daily():
    raw, _ = simulate_raw(20, seed=3, start="1995-01-02")
    return build_ea(raw, Params()).daily


def test_pocket_check_recognises_a_close_fund():
    d = _daily()
    rng = np.random.default_rng(0)
    rb = d["r_b"].loc["2002-07-01":]
    fund = (1 + 1.05 * rb + rng.normal(0, 0.0003, len(rb)) - 0.0015 / 252).cumprod()
    c = compare_bond_pocket(d, fund)
    st = c["stats"]
    assert c["ok"] and st["corrélation mensuelle"] > 0.95
    assert abs(st["bêta de IEF sur la poche"] - 1.05) < 0.05
    assert st["écart moyen IEF - poche (%/an)"] < 0          # fees and lower duration


def test_pocket_check_flags_an_unrelated_series():
    d = _daily()
    rng = np.random.default_rng(1)
    idx = d.loc["2002-07-01":].index
    fund = pd.Series((1 + rng.normal(0, 0.004, len(idx))).cumprod(), index=idx)
    assert not compare_bond_pocket(d, fund)["ok"]


def test_yahoo_adjusted_handles_multiindex(monkeypatch):
    idx = pd.bdate_range("2002-07-26", periods=300)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["IEF"]], names=["Price", "Ticker"])
    df = pd.DataFrame(np.column_stack([np.linspace(80, 90, 300), np.ones(300)]), index=idx, columns=cols)
    fake = types.SimpleNamespace(download=lambda *a, **k: df)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    s = yahoo_adjusted("IEF", "2002-07-01")
    assert len(s) == 300 and s.iloc[0] == 80
