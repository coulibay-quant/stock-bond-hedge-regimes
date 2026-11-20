"""Version 1.2: criterion 3 by sub-period, data sources (EONIA keys, HICP splice), provisional month,
guard on the HTML copy of the notebook."""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from p8regimes import data, evaluation as ev
from p8regimes.config import MARKETS, Params
from p8regimes.dataset import month_in_progress
from p8regimes.journal import NotebookNotSaved, check_saved, saved_market


def _forecast_inputs(T=320, switch=160, seed=4):
    """HB carries information on next-quarter correlation only after ``switch``."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    hb = rng.uniform(0, 1, T)
    corr = np.zeros(T)
    nxt = np.zeros(T)
    for t in range(T):
        corr[t] = 0.5 * (corr[t - 1] if t else 0) + rng.normal(0, 0.2)
        signal = 0.8 * (hb[t] - 0.5) if t >= switch else 0.0
        nxt[t] = 0.4 * corr[t] + signal + rng.normal(0, 0.15)
    ind = pd.DataFrame({"hb": hb, "beta_fcst": np.nan}, index=idx)
    rs = pd.DataFrame({"corr": corr, "corr_next": nxt, "beta": corr / 3, "beta_next": nxt / 3}, index=idx)
    return ind, rs


def test_forecast_by_period_finds_information_where_it_is():
    ind, rs = _forecast_inputs()
    p = Params()
    full = ev.forecast_test(ind, rs, p)
    assert list(full.columns) == ["prévisions", "début", "RMSE modèle 1", "RMSE modèle 2",
                                  "R² hors échantillon", "stat CW", "p-valeur"]
    by = ev.forecast_by_period(ind, rs, p, [("1990-01-01", "2003-04-30"), ("2003-05-01", "2100-01-01")])
    early, late = by.loc[("1990-2003", "+ hb")], by.loc[("2003-fin", "+ hb")]
    assert late["R² hors échantillon"] > 0.1 and late["p-valeur"] < 0.01
    assert early["R² hors échantillon"] < late["R² hors échantillon"]
    # same forecasts as the full-sample test: counts add up
    assert early["prévisions"] + late["prévisions"] == full.loc["+ hb", "prévisions"]


def test_forecast_by_period_default_halves_and_short_periods():
    ind, rs = _forecast_inputs()
    by = ev.forecast_by_period(ind, rs, Params())
    assert len(by) == 2
    short = ev.forecast_by_period(ind, rs, Params(), [("2015-01-01", "2015-06-30")])
    assert short["prévisions"].iloc[0] == 6
    assert "R² hors échantillon" not in short.columns          # too few forecasts: count only


def test_markets_have_forecast_periods():
    assert ("2000-01-01", "2100-01-01") in MARKETS["US"].forecast_periods
    assert len(MARKETS["EA"].forecast_periods) == 2


def test_splice_after_keeps_old_then_new():
    old = pd.Series([2.0, 2.1, 2.2], index=pd.to_datetime(["2025-10-01", "2025-11-01", "2025-12-01"]))
    new = pd.Series([2.05, 2.15, 2.25, 1.9, 1.8],
                    index=pd.to_datetime(["2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"]))
    s = data.splice_after(old, new)
    assert list(s.values) == [2.0, 2.1, 2.2, 1.9, 1.8]
    assert data.splice_after(None, new).equals(new)
    assert data.splice_after(old, None).equals(old)


def test_series_key_and_first_working_url(monkeypatch):
    assert data._series_key(data.URLS["hicp_new"]) == "HICP/M.U2.N.000000.4D0.ANR"
    calls = []

    class R:
        content = b"x" * 100

    def fake_get(url, retries=3, timeout=90):
        calls.append(url)
        if "EON/" in url:
            raise RuntimeError("HTTP 404")
        return R()

    monkeypatch.setattr(data, "_get", fake_get)
    r, url = data._get_first(data.URLS["eonia"])
    assert "FM/" in url and len(calls) == 2
    monkeypatch.setattr(data, "_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        data._get_first(data.URLS["eonia"])


def test_month_in_progress():
    assert month_in_progress("2026-11-19", "2026-11-19")          # Thursday, month not over
    assert not month_in_progress("2026-09-30", "2026-09-30")      # last business day of September
    assert not month_in_progress("2026-10-30", "2026-10-30")      # Friday 30 October, 31st is a Saturday
    assert not month_in_progress("2026-10-30", "2026-11-19")      # not the last data date


def _write_nb(path, market):
    nb = {"cells": [{"cell_type": "code", "source": [f'MARCHE = "{market}"          # comment\n',
                                                      "DEMO = False\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb), encoding="utf-8")


def test_html_copy_guard(tmp_path):
    nb = tmp_path / "02.ipynb"
    _write_nb(nb, "US")
    assert saved_market(nb) == "US"
    old = time.time() - 3600
    os.utime(nb, (old, old))
    with pytest.raises(NotebookNotSaved):
        check_saved(nb, after=datetime.now())                      # not saved since the session
    with pytest.raises(NotebookNotSaved):
        check_saved(nb, market="EA")                               # saved notebook is the US run
    check_saved(nb, after=datetime.fromtimestamp(old - 60), market="US")
