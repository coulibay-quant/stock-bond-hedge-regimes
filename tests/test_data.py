"""Parsers (on samples shaped like the real files), curves and bond-pocket returns."""
import io
import zipfile

import numpy as np
import pandas as pd

from p8regimes.curves import GridCurve, SvenssonCurve, bond_returns, cash_from_rate, svensson_yield
from p8regimes.data import (parse_any_series, parse_dated_lines, parse_french_daily, parse_gsw,
                            parse_sdmx_csv, parse_two_columns, splice_cash)

GSW = """The U.S. Treasury Yield Curve: 1961 to the Present
Refet S. Gurkaynak, Brian Sack, and Jonathan H. Wright
Federal Reserve Board

Series Compounding Description

Date,BETA0,BETA1,BETA2,BETA3,SVEN1F01,SVENY01,SVENY02,SVENY05,SVENY07,SVENY10,TAU1,TAU2
1971-08-16,6.2,-1.1,0.5,NA,5.9,5.5,5.7,6.0,6.1,6.2,1.5,NA
1971-08-17,6.21,-1.12,0.52,NA,5.9,5.5,5.7,6.0,6.1,6.21,1.5,NA
2026-10-23,4.1,-0.2,-1.5,2.0,4.0,3.7,3.6,3.7,3.9,4.2,1.2,9.5
"""

FRENCH = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01
20260831,   -0.52,    0.11,    0.20,    0.017

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

BBK_CLASSIC = """,BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A,BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A_FLAGS
,Term structure of interest rates on listed Federal securities (method by Svensson) / residual maturity of 10.0 years / Daily data,
unit,% per annum,
unit multiplier,one,
last update,2026-10-30 08:12:01,
1997-08-07,5.73,
1997-08-08,5.70,
1997-08-09,.,No value available
2026-10-29,2.61,
"""

ECB = """KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE,OBS_STATUS
FM.D.U2.EUR.4F.MM.EONIA.HSTA,D,U2,EUR,4F,MM,EONIA,HSTA,1999-01-04,3.2,A
FM.D.U2.EUR.4F.MM.EONIA.HSTA,D,U2,EUR,4F,MM,EONIA,HSTA,1999-01-05,3.22,A
"""

FRED = """observation_date,CPIAUCSL
1947-01-01,21.48
1947-02-01,21.62
"""

STOOQ = """Date,Open,High,Low,Close,Volume
1987-12-30,1000.0,1005.0,995.0,1000.0,0
1987-12-31,1001.0,1006.0,996.0,1003.5,0
"""


def test_parse_gsw():
    df = parse_gsw(GSW)
    assert list(df.index.strftime("%Y-%m-%d")) == ["1971-08-16", "1971-08-17", "2026-10-23"]
    assert np.isnan(df.loc["1971-08-16", "BETA3"]) and df.loc["2026-10-23", "TAU2"] == 9.5


def test_parse_french_text_and_zip():
    a = parse_french_daily(FRENCH)
    assert len(a) == 3 and a.iloc[0]["mkt_rf"] == 0.09 and a.iloc[-1]["rf"] == 0.017
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.csv", FRENCH)
    b = parse_french_daily(buf.getvalue())
    pd.testing.assert_frame_equal(a, b)


def test_parse_bundesbank_classic():
    s = parse_dated_lines(BBK_CLASSIC)
    assert len(s) == 3 and s.iloc[0] == 5.73 and s.index[-1] == pd.Timestamp("2026-10-29")
    s2 = parse_any_series(BBK_CLASSIC)
    assert len(s2) == 3


def test_parse_ecb_fred_stooq():
    e = parse_sdmx_csv(ECB)
    assert e.iloc[1] == 3.22
    f = parse_two_columns(FRED)
    assert f.iloc[0] == 21.48 and f.index[1] == pd.Timestamp("1947-02-01")
    s = parse_two_columns(STOOQ)
    assert s.iloc[-1] == 1003.5


def test_splice_cash():
    eon = pd.Series([3.0, 3.0, 3.0], index=pd.to_datetime(["2019-09-30", "2019-10-01", "2019-10-02"]), name="eonia")
    est = pd.Series([2.9, 2.9], index=pd.to_datetime(["2019-10-01", "2019-10-02"]), name="estr")
    s = splice_cash(eon, est)
    assert list(s.values) == [3.0, 2.9, 2.9]


def test_svensson_reduces_to_nelson_siegel_when_b3_missing():
    y = svensson_yield(7.0, 4.0, -1.0, 0.5, np.nan, 1.5, np.nan)
    y_ns = svensson_yield(7.0, 4.0, -1.0, 0.5, 0.0, 1.5, 5.0)
    assert np.isclose(y, y_ns)


def test_bond_pocket_carry_and_duration():
    dates = pd.to_datetime(["2026-10-23", "2026-10-26", "2026-10-27"])     # Friday, Monday, Tuesday
    params = pd.DataFrame({"BETA0": [4.0, 4.0, 4.01], "BETA1": 0.0, "BETA2": 0.0, "BETA3": np.nan,
                           "TAU1": 1.5, "TAU2": np.nan}, index=dates)
    r = bond_returns(SvenssonCurve(params), dates, 7.0)
    assert np.isclose(r.iloc[0], 0.04 * 3 / 365.25)                       # weekend carry
    d = 1 / 365.25
    assert np.isclose(r.iloc[1], 0.04 * d - (7 - d) * 0.0001)              # +1 bp costs ~7 bp
    grid = pd.DataFrame({"y02": [4.0, 4.0, 4.01], "y05": [4.0, 4.0, 4.01], "y10": [4.0, 4.0, 4.01]}, index=dates)
    r2 = bond_returns(GridCurve(grid), dates, 7.0)
    assert np.allclose(r, r2, atol=1e-12)


def test_cash_from_rate_act360():
    dates = pd.to_datetime(["2026-10-23", "2026-10-26"])
    r = cash_from_rate(pd.Series([3.6], index=pd.to_datetime(["2026-10-01"])), dates)
    assert np.isclose(r.iloc[0], 0.036 * 3 / 360)
