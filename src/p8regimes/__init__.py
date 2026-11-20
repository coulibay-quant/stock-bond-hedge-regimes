"""P8 – when bonds stop hedging equities.

Real-time stock-bond regimes from sign-identified market shocks (Cieslak-Pang restrictions),
and a regime-aware de-risking rule evaluated against the 60/40 and a volatility target.
"""
__version__ = "1.2.1"

from .config import Params, get_paths  # noqa: F401
