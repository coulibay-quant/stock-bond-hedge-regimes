"""End-to-end session: data -> indicator -> signals -> backtest -> evaluation -> figures.

Used by the working notebook (02) and by the command line (``p8 run``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import report
from . import robustness as rb
from .allocation import risk_inputs
from .backtest import default_strategies, run_backtest
from .checks import compare_bond_pocket
from .config import MARKETS, Params, Paths, get_paths
from .dataset import Panel, build_ea, build_panel, month_in_progress, monthly_returns
from .indicator import realized_stats, run_indicator
from .signals import build_signals
from .simulate import simulate_raw, true_hb


@dataclass
class Session:
    market: str
    params: Params
    panel: Panel
    ind: pd.DataFrame
    rs: pd.DataFrame
    signals: pd.DataFrame
    risk: pd.DataFrame
    res: dict
    perf: dict = field(default_factory=dict)       # {"période": DataFrame}
    tests: pd.DataFrame | None = None
    vt: pd.DataFrame | None = None
    forecast: pd.DataFrame | None = None
    forecast_periods: pd.DataFrame | None = None   # criterion 3 by sub-period (robustness)
    episodes: pd.DataFrame | None = None
    criteria: pd.DataFrame | None = None
    figures: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    hb_true: pd.Series | None = None
    pocket_check: dict | None = None
    robust: dict = field(default_factory=dict)

    def regime_now(self) -> dict:
        last = self.ind.dropna(subset=["hb"]).iloc[-1]
        d = last.name
        sig = self.signals.loc[d]
        return {
            "date": d.date(), "provisoire": month_in_progress(d, self.panel.daily.index[-1]),
            "HB": last["hb"], "HB p16": last["hb_p16"], "HB p84": last["hb_p84"],
            "bêta structurel": last.get("beta_now"), "bêta prévu": last.get("beta_fcst"),
            "régime P8-HB": "cash" if sig.get("C_hb") == 1 else "couverture",
            "régime P8-bêta": "cash" if sig.get("C_beta") == 1 else "couverture",
            "corrélation 63 j": self.rs.loc[d, "corr"] if d in self.rs.index else None,
            "facteur k": self.risk.loc[d, "k"] if d in self.risk.index else None,
        }


def demo_panel(params: Params, years: int = 30, seed: int = 2026):
    raw, sig = simulate_raw(years, seed)
    panel = build_ea(raw, params)
    panel.market = "DEMO"
    # simulated traded fund for the pocket check: 0.95 x pocket, tracking noise, 0.15 % fees
    rng = np.random.default_rng(seed + 2)
    rb = panel.daily["r_b"]
    fund = (1 + 0.95 * rb + rng.normal(0, 0.0004, len(rb)) - 0.0015 / 252).cumprod()
    panel.bond_benchmark, panel.bond_benchmark_name = fund.loc["2002-07-01":], "fonds simulé"
    panel.notes.append("DONNÉES SIMULÉES (mode démo) : aucun résultat n'est réel")
    return panel, sig


def run_session(market: str = "US", params: Params | None = None, paths: Paths | None = None,
                demo: bool = False, use_cache: bool = True, with_hmm: bool = True,
                figures: bool = True, out_dir: Path | None = None, verbose: bool = True) -> Session:
    params = params or Params()
    paths = paths or get_paths()
    t = {}
    say = print if verbose else (lambda *a, **k: None)

    t0 = time.time()
    if demo:
        panel, sig_true = demo_panel(params)
    else:
        panel, sig_true = build_panel(market, params, paths), None
    t["données"] = time.time() - t0
    say(f"[1/5] panel {panel.market} : {panel.daily.index[0].date()} -> {panel.daily.index[-1].date()}, "
        f"{len(panel.daily)} jours")

    pocket_check = None
    if panel.bond_benchmark is not None and len(panel.bond_benchmark) > 250:
        pocket_check = compare_bond_pocket(panel.daily, panel.bond_benchmark, panel.bond_benchmark_name)
        say("      contrôle de la poche taux : " + pocket_check["verdict"])

    t0 = time.time()
    say("[2/5] indicateur en temps réel (peut prendre plusieurs minutes la première fois)")
    ind = run_indicator(panel, params, paths, use_cache=use_cache, progress=verbose)
    t["indicateur"] = time.time() - t0

    t0 = time.time()
    say("[3/5] signaux des concurrents (HMM compris)")
    rs = realized_stats(panel.daily, ind.index, params.recent)
    sg = build_signals(panel, ind, rs, params, with_hmm=with_hmm)
    risk = risk_inputs(panel.daily, ind.index, params)
    t["signaux"] = time.time() - t0

    t0 = time.time()
    say("[4/5] backtest")
    mon = monthly_returns(panel.daily)
    res = run_backtest(mon, sg, risk, default_strategies(sg), params)
    t["backtest"] = time.time() - t0

    t0 = time.time()
    say("[5/5] évaluation")
    s = Session(panel.market, params, panel, ind, rs, sg, risk, res, timings=t, pocket_check=pocket_check)
    s.perf["tout l'échantillon"] = ev.perf_table(res)
    sub = MARKETS[market].subperiods if (market in MARKETS and not demo) else []
    for a, b in sub:
        label = f"{a[:4]}-{b[:4] if b < '2100' else 'fin'}"
        if len(res["60/40"].loc[a:b]) >= 24:
            s.perf[label] = ev.perf_table(res, a, b)
    s.tests = ev.tests_table(res, params=params)
    s.vt = pd.concat([ev.vt_decomposition(res, n) for n in ("P8-HB-A", "P8-BETA-A")], axis=1)
    s.forecast = ev.forecast_test(ind, rs, params)
    periods = MARKETS[market].forecast_periods if (market in MARKETS and not demo) else None
    s.forecast_periods = ev.forecast_by_period(ind, rs, params, periods)
    s.episodes = ev.episodes_table(ind, rs, market if not demo else "XX", res)
    s.criteria = ev.success_criteria(s.perf["tout l'échantillon"], s.tests, s.forecast)
    s.robust = robustness_checks(res, mon, sg, risk, params, ind.index, with_hmm=with_hmm, verbose=verbose)
    if sig_true is not None:
        s.hb_true = true_hb(sig_true.to_numpy(), panel.daily.index, ind.index, params.recent)
    t["évaluation"] = time.time() - t0

    if figures:
        out_dir = out_dir or (paths.outputs / datetime.now().strftime("%Y-%m-%d"))
        s.figures = ([report.fig_pocket_check(pocket_check, panel.market, out_dir)] if pocket_check else []) + [
            report.fig_data(panel, out_dir),
            report.fig_hb(ind, rs, panel.market, params.hb_threshold, out_dir),
            report.fig_shares(ind, panel.market, out_dir),
            report.fig_performance(res, panel.market, out_dir=out_dir),
            report.fig_scatter(ind, rs, panel.market, out_dir),
        ]
    say("terminé : " + ", ".join(f"{k} {v:.0f} s" for k, v in t.items()))
    return s


def robustness_checks(res: dict, monthly, signals, risk, params: Params, decision_index,
                      with_hmm: bool = True, verbose: bool = True) -> dict:
    """Deflated Sharpe ratios, regime switches and costs, and (optionally) HMM stability."""
    out = {"dsr": rb.deflated_sharpe_table(res),
           "dsr_6040": rb.deflated_sharpe_table(res, "60/40"),
           "dsr_vc": rb.deflated_sharpe_table(res, "VC"),
           "switches": rb.switch_stats(res, params.cost_bp)}
    out["costs"] = rb.cost_grid(monthly, signals, risk, default_strategies(signals), params)
    out["break_even"] = rb.break_even_costs(out["costs"])
    if with_hmm:
        out["hmm"] = rb.hmm_stability(monthly, risk, params, decision_index, verbose=verbose)
    return out


def save_tables(s: Session, out_dir: Path) -> list[Path]:
    """CSV copies of the main tables (outputs/YYYY-MM-DD/)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    m = s.market
    items = {f"{m}_indicateur.csv": s.ind, f"{m}_signaux.csv": s.signals,
             f"{m}_tests_sharpe.csv": s.tests, f"{m}_prevision_correlation.csv": s.forecast,
             f"{m}_episodes.csv": s.episodes, f"{m}_decomposition_vc.csv": s.vt,
             f"{m}_prevision_par_periode.csv": s.forecast_periods}
    if s.pocket_check:
        items[f"{m}_controle_poche_taux.csv"] = s.pocket_check["stats"].to_frame()
    r = s.robust or {}
    for key, name in (("dsr", "sharpe_deflate"), ("dsr_6040", "sharpe_deflate_vs_6040"),
                      ("dsr_vc", "sharpe_deflate_vs_vc"), ("switches", "bascules_de_regime"),
                      ("costs", "sharpe_selon_couts"), ("break_even", "couts_seuils")):
        if key in r:
            items[f"{m}_{name}.csv"] = r[key]
    if "hmm" in r:
        items[f"{m}_stabilite_hmm.csv"] = r["hmm"]["table"]
        items[f"{m}_stabilite_hmm_reestimations.csv"] = r["hmm"]["stability"]
    if "p8_windows" in r:
        items[f"{m}_sensibilite_fenetres_p8.csv"] = r["p8_windows"]
    for k, df in s.perf.items():
        safe = k.replace("tout l'échantillon", "tout_echantillon").replace(" ", "_")
        items[f"{m}_performance_{safe}.csv"] = df
    for name, df in items.items():
        if df is not None and len(df):
            p = out_dir / name
            df.to_csv(p, encoding="utf-8-sig")
            files.append(p)
    return files
