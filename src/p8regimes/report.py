"""Figures of a working session (saved as PNG in outputs/YYYY-MM-DD/)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHOCK_COLORS = {"g": BLUE, "pp": AQUA, "m": ORANGE, "pc": MAGENTA}
SHOCK_NAMES = {"g": "croissance", "pp": "prime de couverture", "m": "monétaire", "pc": "prime commune"}
STRAT_COLORS = {"60/40": INK2, "VC": AQUA, "P8-HB-A": BLUE, "P8-BETA-A": VIOLET, "CORR-A": ORANGE,
                "P8-HB-B": BLUE, "P8-BETA-B": VIOLET, "CORR-B": ORANGE, "HMM-A": YELLOW, "MACRO-A": MAGENTA}

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _save(fig, out_dir: Path | None, name: str):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name)
    return fig


def fig_data(panel, out_dir=None):
    d = panel.daily
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(d.index, d["y02"], color=BLUE, lw=1, label="2 ans")
    ax[0].plot(d.index, d["y10"], color=ORANGE, lw=1, label="10 ans")
    ax[0].set_ylabel("taux zéro-coupon (%)"); ax[0].legend(loc="upper right", ncol=2)
    nav = (1 + d[["r_e", "r_b", "r_f"]]).cumprod()
    for c, col, lab in (("r_e", BLUE, "actions"), ("r_b", ORANGE, "poche taux 7 ans"), ("r_f", INK2, "cash")):
        ax[1].plot(nav.index, nav[c], color=col, lw=1, label=lab)
    ax[1].set_yscale("log"); ax[1].set_ylabel("valeur (base 1, log)"); ax[1].legend(loc="upper left", ncol=3)
    corr = d["r_e"].rolling(63).corr(d["r_b"])
    ax[2].plot(corr.index, corr, color=INK, lw=0.8)
    ax[2].axhline(0, color=INK2, lw=0.8); ax[2].set_ylabel("corrélation 63 jours")
    ax[0].set_title(f"{panel.market} : données quotidiennes", loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, f"{panel.market}_01_donnees.png")


def fig_hb(ind, rs, market, threshold=0.5, out_dir=None):
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax[0].fill_between(ind.index, 100 * ind["hb_p16"], 100 * ind["hb_p84"], color=BLUE, alpha=0.15, lw=0,
                       label="bande 16-84 % (rotations)")
    ax[0].plot(ind.index, 100 * ind["hb"], color=BLUE, lw=1.3, label="HB (médiane)")
    ax[0].axhline(100 * threshold, color=INK2, lw=0.8, ls=":")
    ax[0].set_ylim(0, 100); ax[0].set_ylabel("part « cassure » (%)")
    ax[0].legend(loc="upper left", ncol=2)
    ax[0].set_title(f"{market} : part du risque actions venant des chocs qui cassent la couverture",
                    loc="left", fontsize=10)
    ax[1].plot(rs.index, rs["corr"], color=INK, lw=1, label="corrélation réalisée 63 j")
    if "beta_fcst" in ind:
        ax[1].plot(ind.index, ind["beta_now"], color=ORANGE, lw=1, label="bêta structurel")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_ylabel("corrélation / bêta")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    return _save(fig, out_dir, f"{market}_02_indicateur.png")


def fig_shares(ind, market, out_dir=None):
    order = ["g", "pp", "m", "pc"]
    s = ind[[f"s_{k}" for k in order]].copy()
    s = s.div(s.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.stackplot(s.index, *[100 * s[f"s_{k}"] for k in order], colors=[SHOCK_COLORS[k] for k in order],
                 labels=[SHOCK_NAMES[k] for k in order], alpha=0.85, lw=0)
    ax.axhline(50, color="white", lw=0.8, ls=":")
    ax.set_ylim(0, 100); ax.set_ylabel("part de la variance des actions (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.set_title(f"{market} : décomposition du risque actions par choc (médianes)", loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, f"{market}_03_parts_des_chocs.png")


def fig_performance(res, market, names=("60/40", "VC", "CORR-A", "P8-HB-A", "P8-BETA-A"), out_dir=None):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    for n in names:
        if n not in res:
            continue
        R = res[n]["R"]
        nav = (1 + R).cumprod()
        col = STRAT_COLORS.get(n, INK2)
        ax[0].plot(nav.index, nav, color=col, lw=1.6 if n.startswith("P8") else 1.1, label=n)
        ax[1].plot(nav.index, -100 * (1 - nav / nav.cummax()), color=col, lw=1)
    ax[0].set_yscale("log"); ax[0].set_ylabel("valeur (base 1, log)"); ax[0].legend(loc="upper left", ncol=3)
    ax[0].set_title(f"{market} : performance cumulée nette de coûts", loc="left", fontsize=10)
    ax[1].set_ylabel("perte depuis le plus haut (%)")
    fig.tight_layout()
    return _save(fig, out_dir, f"{market}_04_performance.png")


def fig_scatter(ind, rs, market, out_dir=None):
    d = pd.concat([ind["hb"], rs["corr"], rs.get("corr_next")], axis=1).dropna(subset=["hb", "corr"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].scatter(100 * d["hb"], d["corr"], s=8, color=BLUE, alpha=0.6, lw=0)
    ax[0].set_xlabel("HB (%)"); ax[0].set_ylabel("corrélation du trimestre écoulé")
    if "corr_next" in d:
        ax[1].scatter(100 * d["hb"], d["corr_next"], s=8, color=ORANGE, alpha=0.6, lw=0)
        ax[1].set_xlabel("HB (%)"); ax[1].set_ylabel("corrélation du trimestre suivant")
    for a in ax:
        a.axhline(0, color=INK2, lw=0.8); a.axvline(50, color=INK2, lw=0.8, ls=":")
    fig.suptitle(f"{market} : indicateur et corrélation réalisée", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, f"{market}_05_hb_et_correlation.png")


def fig_pocket_check(check: dict, market: str = "US", out_dir=None):
    """Cumulative returns and monthly scatter: 7-year zero-coupon pocket vs a traded bond fund."""
    m, name = check["monthly"], check["name"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.7, 1]})
    nav = (1 + m).cumprod()
    t = nav.index.to_timestamp(how="end")
    ax[0].plot(t, nav["poche"], color=BLUE, lw=1.4, label="poche zéro-coupon 7 ans")
    ax[0].plot(t, nav[name], color=ORANGE, lw=1.2, label=f"{name} (rendement total)")
    ax[0].set_ylabel("valeur (base 1)"); ax[0].legend(loc="upper left")
    ax[1].scatter(100 * m["poche"], 100 * m[name], s=8, color=BLUE, alpha=0.6, lw=0)
    lim = 100 * np.nanmax(np.abs(m.to_numpy())) * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls=":")
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel("poche 7 ans (%/mois)"); ax[1].set_ylabel(f"{name} (%/mois)")
    fig.suptitle(f"{market} : contrôle de la poche taux contre {name}", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, f"{market}_00_controle_poche_taux.png")
