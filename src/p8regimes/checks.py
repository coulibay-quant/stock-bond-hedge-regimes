"""Data checks: is the synthetic 7-year zero-coupon pocket close to a traded bond fund?

United States: iShares 7-10 Year Treasury Bond ETF (IEF), price adjusted for distributions
(total return), daily since mid-2002. The pocket is rebuilt from the fitted zero-coupon curve,
so differences come from coupons (an index duration a bit different from 7), the curve fit,
fund fees (0.15 % a year) and the ETF's own trading noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# reference values used to read the check (rules of thumb, not statistical tests)
REPERES = {"corrélation mensuelle min": 0.95, "bêta min": 0.8, "bêta max": 1.2}


def _monthly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def compare_bond_pocket(daily: pd.DataFrame, benchmark: pd.Series, name: str = "IEF") -> dict:
    """Monthly and daily comparison of the pocket (daily['r_b']) with a traded fund's price."""
    bench = benchmark.dropna().sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    start = bench.index.min()
    pocket = daily["r_b"].loc[start:]

    # daily: returns between dates common to both calendars
    common = pocket.index.intersection(bench.index)
    p_idx = (1 + pocket).cumprod().reindex(common)
    d = pd.DataFrame({"poche": p_idx.pct_change(), name: bench.reindex(common).pct_change()}).dropna()

    # monthly: calendar months, first (partial) and last (possibly incomplete) months dropped
    m_pocket = _monthly(pocket)
    m_bench = bench.groupby(bench.index.to_period("M")).last().pct_change()
    m = pd.DataFrame({"poche": m_pocket, name: m_bench}).dropna().iloc[:-1]
    m = m.iloc[1:] if len(m) > 1 else m

    x, y = m["poche"], m[name]
    beta = np.cov(y, x)[0, 1] / x.var()
    ann = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
    diff = y - x
    stats = pd.Series({
        "période": f"{m.index[0]} -> {m.index[-1]}",
        "mois": len(m),
        "rendement poche 7 ans (%/an)": 100 * ann(x),
        f"rendement {name} (%/an)": 100 * ann(y),
        "volatilité poche (%/an)": 100 * x.std() * np.sqrt(12),
        f"volatilité {name} (%/an)": 100 * y.std() * np.sqrt(12),
        "corrélation mensuelle": x.corr(y),
        "corrélation quotidienne": d["poche"].corr(d[name]),
        f"bêta de {name} sur la poche": beta,
        "écart de suivi (%/an)": 100 * diff.std() * np.sqrt(12),
        f"écart moyen {name} - poche (%/an)": 1200 * diff.mean(),
        "plus grand écart mensuel (%)": 100 * diff.abs().max(),
    }, name=f"poche 7 ans vs {name}")
    ok = (stats["corrélation mensuelle"] >= REPERES["corrélation mensuelle min"]
          and REPERES["bêta min"] <= beta <= REPERES["bêta max"])
    verdict = (f"La poche zéro-coupon 7 ans suit {name} de près (corrélation mensuelle "
               f"{stats['corrélation mensuelle']:.3f}, bêta {beta:.2f}, écart de suivi "
               f"{stats['écart de suivi (%/an)']:.1f} %/an)." if ok else
               f"Écart notable entre la poche 7 ans et {name} (corrélation mensuelle "
               f"{stats['corrélation mensuelle']:.3f}, bêta {beta:.2f}) : à examiner avant de conclure.")
    return {"stats": stats, "monthly": m, "daily": d, "ok": ok, "verdict": verdict, "name": name}
