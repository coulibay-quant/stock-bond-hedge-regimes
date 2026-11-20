"""Paths, parameters and market definitions.

Every parameter that the methodological note fixes *before* any backtest lives here.
Changing one of them changes the cache key of the real-time indicator.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------------- paths
def find_root(start: Path | None = None) -> Path:
    """Repository root: $P8_ROOT if set, else the first parent holding pyproject.toml."""
    env = os.environ.get("P8_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "p8regimes").exists():
            return p
    # installed package used outside the repo: fall back to the package location
    pkg = Path(__file__).resolve()
    for p in pkg.parents:
        if (p / "pyproject.toml").exists():
            return p
    return here


@dataclass
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def manual(self) -> Path:
        return self.root / "data" / "raw" / "manual"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def journal(self) -> Path:
        return self.root / "journal"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.manual, self.processed, self.outputs, self.journal):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: Path | None = None) -> Paths:
    return Paths(Path(root) if root else find_root()).ensure()


# ---------------------------------------------------------------------------------- parameters
@dataclass
class Params:
    # --- identification (section 3 of the note)
    window: int = 500            # W: days used for the VAR and the identified set
    recent: int = 63             # h: days used for the recent shock intensities
    n_accept: int = 1000         # admissible rotations kept per date
    batch: int = 100_000         # rotations drawn per batch
    max_draws: int = 2_000_000   # give up after this many draws at one date
    min_accept: int = 100        # fewer admissible rotations than this -> date flagged (NaN)
    share_restrictions: bool = True   # variance-share restrictions of Cieslak-Pang
    seed: int = 2026
    # --- bond pocket
    duration: float = 7.0        # constant maturity of the zero-coupon bond pocket (years)
    # --- regimes and allocation (section 4)
    hb_threshold: float = 0.5
    w_equity: float = 0.6
    w_bond: float = 0.4
    cost_bp: float = 2.0         # cost per unit of turnover, in basis points
    min_ref_days: int = 252      # history needed before sigma_ref is defined
    # --- P8-beta (structural beta forecast)
    beta_horizon: int = 3        # months ahead (next quarter)
    phi_min_obs: int = 36        # monthly pairs needed to estimate shock-specific persistence
    # --- competitors
    macro_threshold: float = 3.0     # % yoy inflation (Robeco rule)
    hmm_min_months: int = 120
    hmm_refit_every: int = 12
    # --- evaluation (section 5)
    bootstrap_reps: int = 2000
    block: int = 6               # months, circular block bootstrap
    forecast_min_obs: int = 60   # months before the first out-of-sample forecast

    def indicator_key(self) -> str:
        """Hash of the parameters that change the real-time indicator."""
        keys = ["window", "recent", "n_accept", "share_restrictions", "seed", "duration",
                "beta_horizon", "phi_min_obs", "min_accept"]
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------- markets
@dataclass
class Market:
    code: str
    name: str
    equity: str
    curve: str
    cash: str
    inflation: str
    subperiods: list = field(default_factory=list)
    forecast_periods: list = field(default_factory=list)   # evaluation windows for criterion 3 (robustness)


MARKETS = {
    "US": Market(
        code="US", name="United States", equity="CRSP value-weighted market (Kenneth French)",
        curve="Treasury zero-coupon curve (Gurkaynak-Sack-Wright, Federal Reserve)",
        cash="1-month T-bill (Kenneth French)", inflation="CPI-U, all items (FRED CPIAUCSL)",
        subperiods=[("1973-01-01", "1999-12-31"), ("2000-01-01", "2020-12-31"),
                    ("2021-01-01", "2100-01-01"), ("1983-01-01", "2100-01-01")],
        forecast_periods=[("1978-01-01", "1999-12-31"), ("2000-01-01", "2100-01-01"),
                          ("2000-01-01", "2020-12-31"), ("2021-01-01", "2100-01-01")],
    ),
    "EA": Market(
        code="EA", name="Euro area (German curve)", equity="DAX performance index",
        curve="German Federal securities zero-coupon curve (Bundesbank)",
        cash="EONIA then euro short-term rate (ECB)", inflation="Euro area HICP, annual rate (ECB)",
        subperiods=[("1999-01-01", "2020-12-31"), ("2021-01-01", "2100-01-01")],
        forecast_periods=[("2004-01-01", "2020-12-31"), ("2021-01-01", "2100-01-01")],
    ),
}

# Episodes of section 5.5 of the note: (label, start, end, expected reading, market)
EPISODES = [
    ("Krach d'octobre 1987", "1987-10-01", "1987-11-30", "p+", "US"),
    ("Hausses de la Fed 1994", "1994-02-01", "1994-12-31", "m", "US"),
    ("Bulle internet 2000-2002", "2000-09-01", "2002-10-31", "g / p+", "US"),
    ("Crise financière 2008", "2008-09-01", "2009-03-31", "p+", "US"),
    ("Taper tantrum 2013", "2013-05-01", "2013-07-31", "m / p-", "US"),
    ("Mars 2020", "2020-02-20", "2020-04-30", "p- puis p+", "US"),
    ("Inflation et resserrement 2022", "2022-01-01", "2022-12-31", "m", "US"),
    ("Droits de douane avril 2025", "2025-04-01", "2025-05-31", "p-", "US"),
    ("Dettes souveraines 2011-2012", "2011-07-01", "2012-07-31", "p+", "EA"),
    ("Resserrement BCE 2022", "2022-01-01", "2022-12-31", "m", "EA"),
    ("Crise financière 2008 (zone euro)", "2008-09-01", "2009-03-31", "p+", "EA"),
]

SHOCKS = ("g", "m", "pp", "pc")             # growth, monetary, hedging premium (p+), common premium (p-)
SHOCK_LABELS = {"g": "croissance", "m": "monétaire", "pp": "prime de couverture",
                "pc": "prime commune"}
HEDGE_BREAKING = (1, 3)                      # indices of m and pc
