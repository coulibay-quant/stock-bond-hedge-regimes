"""Daily and monthly panels used by the model.

Daily panel columns
    y02, y05, y10       zero-coupon yields (%)
    dy2, dy5, dy10      daily changes (basis points)            -> VAR variables 1-3
    re_log              equity log return (%)                    -> VAR variable 4
    r_e, r_b, r_f       simple daily returns (decimals): equities, 7y bond pocket, cash
All series are aligned on the common trading calendar of the curve and the equity index;
returns between two common dates compound whatever happened in between.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params, Paths
from .curves import GridCurve, SvenssonCurve, bond_returns, cash_from_rate
from .data import load_raw

VAR_COLS = ["dy2", "dy5", "dy10", "re_log"]


@dataclass
class Panel:
    market: str
    daily: pd.DataFrame
    inflation: pd.Series | None      # monthly yoy %, PeriodIndex('M')
    notes: list
    bond_benchmark: pd.Series | None = None   # traded bond index/ETF price (total return), for the check
    bond_benchmark_name: str = "IEF"

    @property
    def z(self) -> np.ndarray:
        return self.daily[VAR_COLS].to_numpy()

    @property
    def month_ends(self) -> pd.DatetimeIndex:
        d = self.daily.index
        s = pd.Series(d, index=d)
        return pd.DatetimeIndex(s.groupby(d.to_period("M")).max().values)

    def summary(self) -> pd.DataFrame:
        d = self.daily
        rows = {
            "début": d.index.min().date(), "fin": d.index.max().date(), "jours": len(d),
            "vol. actions (%/an)": 100 * d["r_e"].std() * np.sqrt(252),
            "vol. poche taux (%/an)": 100 * d["r_b"].std() * np.sqrt(252),
            "corr. actions-taux": d[["r_e", "r_b"]].corr().iloc[0, 1],
            "cash moyen (%/an)": 100 * d["r_f"].mean() * 252,
            "écart-type dy2 (pb/j)": d["dy2"].std(), "écart-type dy10 (pb/j)": d["dy10"].std(),
            "jours avec trou > 5 j": int((d.index.to_series().diff().dt.days > 5).sum()),
        }
        return pd.Series(rows, name=self.market).to_frame()


def _assemble(curve, equity_index: pd.Series, cash_index: pd.Series | None, cash_rate: pd.Series | None,
              dates: pd.DatetimeIndex, duration: float) -> pd.DataFrame:
    mats = curve.maturities((2, 5, 10))
    pos = curve.index.get_indexer(dates)
    y = mats.iloc[pos].set_index(dates)
    out = pd.DataFrame(index=dates)
    out["y02"], out["y05"], out["y10"] = y["y02"], y["y05"], y["y10"]
    out["dy2"] = 100 * out["y02"].diff()
    out["dy5"] = 100 * out["y05"].diff()
    out["dy10"] = 100 * out["y10"].diff()
    eq = equity_index.reindex(dates)
    out["re_log"] = 100 * np.log(eq / eq.shift(1))
    out["r_e"] = eq / eq.shift(1) - 1
    out["r_b"] = np.expm1(bond_returns(curve, dates, duration)).reindex(dates)
    if cash_index is not None:
        ci = cash_index.reindex(dates)
        out["r_f"] = ci / ci.shift(1) - 1
    else:
        out["r_f"] = cash_from_rate(cash_rate, dates).reindex(dates)
    return out.iloc[1:].dropna()


def build_us(raw: dict, params: Params) -> Panel:
    gsw = raw["curve"]
    if "SVENY10" in gsw:
        gsw = gsw[gsw["SVENY10"].notna()]
    curve = SvenssonCurve(gsw)
    ff = raw["ff"].dropna()
    eq_index = (1 + (ff["mkt_rf"] + ff["rf"]) / 100).cumprod()
    cash_index = (1 + ff["rf"] / 100).cumprod()
    dates = curve.index.intersection(ff.index).sort_values()
    daily = _assemble(curve, eq_index, cash_index, None, dates, params.duration)
    cpi = raw["cpi"]
    cpi.index = pd.to_datetime(cpi.index).to_period("M")
    infl = (100 * (cpi / cpi.shift(12) - 1)).dropna().rename("inflation")
    notes = [f"courbe : {len(curve.index)} dates ; actions : {len(ff)} dates ; communes : {len(dates)}"]
    return Panel("US", daily, infl, notes, raw.get("ief"))


def build_ea(raw: dict, params: Params) -> Panel:
    grid = raw["curve"].dropna(how="any")
    curve = GridCurve(grid)
    dax = raw["dax"].dropna()
    dates = curve.index.intersection(dax.index).sort_values()
    notes = [f"courbe : {len(curve.index)} dates ; DAX : {len(dax)} dates ; communes : {len(dates)}"]
    cash = raw.get("cash")
    y1 = grid["y01"] if "y01" in grid else grid.iloc[:, 0]
    if cash is None or len(cash) == 0:
        rate = y1
        notes.append("cash : taux 1 an de la courbe (pas de série monétaire)")
    else:
        before = y1[y1.index < cash.index.min()]
        rate = pd.concat([before, cash]).sort_index()
        if len(before):
            notes.append(f"cash : taux 1 an de la courbe avant {cash.index.min().date()}, puis EONIA/€STR")
    daily = _assemble(curve, dax, None, rate, dates, params.duration)
    infl = None
    if raw.get("hicp") is not None:
        h = raw["hicp"].copy()
        h.index = pd.to_datetime(h.index).to_period("M")
        infl = h.rename("inflation")
    return Panel("EA", daily, infl, notes)


def build_panel(market: str, params: Params, paths: Paths | None = None) -> Panel:
    raw = load_raw(market, paths)
    return build_us(raw, params) if market == "US" else build_ea(raw, params)


def monthly_returns(daily: pd.DataFrame, cols=("r_e", "r_b", "r_f")) -> pd.DataFrame:
    """Compounded simple returns per calendar month, indexed by the month's last trading date."""
    g = daily[list(cols)].groupby(daily.index.to_period("M"))
    out = g.apply(lambda x: (1 + x).prod() - 1)
    last = daily.index.to_series().groupby(daily.index.to_period("M")).max()
    out.index = pd.DatetimeIndex(last.loc[out.index].values)
    return out


def month_in_progress(date, last_data_date) -> bool:
    """True if ``date`` is the last day of data and business days remain in its calendar month:
    the 'month-end' reading is then provisional (the month is not over)."""
    date, last = pd.Timestamp(date), pd.Timestamp(last_data_date)
    return bool(date == last and (date + pd.offsets.BDay(1)).month == date.month)
