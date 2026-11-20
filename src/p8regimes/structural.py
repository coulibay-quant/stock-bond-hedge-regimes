"""Sign-identified structural VAR (section 3 of the note).

Variables (in this order): dy2, dy5, dy10 (basis points), r_act (log %).
Shocks (in this order):    g (growth), m (monetary), pp (hedging premium p+), pc (common premium p-).

Restrictions are those of Cieslak and Pang (equation 14 of CEPR DP14708):
    g  : equities +, yields +, A2 > A10 and A5 > A10
    m  : equities -, yields +, A2 > A5 > A10
    p+ : equities -, yields -, |A2| < |A5| < |A10|
    p- : equities -, yields +, A2 < A5 < A10
plus, optionally, the variance-share restrictions: short-rate news (g, m) explain more than
half of the 2-year variance and less than half of the 10-year variance.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

PERMS = np.array(list(permutations(range(4))))     # 24 assignments column -> shock


# ----------------------------------------------------------------------------- reduced form
def var1(z: np.ndarray):
    """OLS VAR(1) with constant. z: (T, n). Returns (A, c, residuals (T-1, n))."""
    y = z[1:]
    x = np.column_stack([np.ones(len(z) - 1), z[:-1]])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    u = y - x @ coef
    return coef[1:].T, coef[0], u


# ----------------------------------------------------------------------------- rotations
def draw_rotations(n: int, ndraws: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform orthonormal matrices via QR of Gaussian matrices (Rubio-Ramirez et al. 2010)."""
    x = rng.standard_normal((ndraws, n, n))
    q, r = np.linalg.qr(x)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    d[d == 0] = 1.0
    return q * d[:, None, :]


class RotationBank:
    """Deterministic stream of rotation chunks, shared by all dates (common random numbers).

    Using the same rotations at every date removes Monte Carlo noise from month-to-month
    changes of the indicator. The first ``keep`` chunks are cached (float32).
    """

    def __init__(self, chunk: int = 100_000, seed: int = 2026, keep: int = 20):
        self.chunk, self.seed, self.keep = chunk, seed, keep
        self._cache: dict[int, np.ndarray] = {}

    def get(self, i: int) -> np.ndarray:
        if i in self._cache:
            return self._cache[i].astype(np.float64)
        q = draw_rotations(4, self.chunk, np.random.default_rng([self.seed, i]))
        if i < self.keep:
            self._cache[i] = q.astype(np.float32)
        return q


# ----------------------------------------------------------------------------- restrictions
LEVELS = {
    0: "signes + profils + parts de variance (Cieslak-Pang complet)",
    1: "signes + profils",
    2: "profils sans ordre sur le 2 ans (signe du 2 ans conservé)",
    3: "2 ans ignoré : signes et ordre 5 ans / 10 ans",
    4: "2 ans ignoré : signes seuls sur 5 ans, 10 ans et actions",
}
N_LEVELS = len(LEVELS)


def column_ok(col: np.ndarray, shock: str, level: int = 0) -> np.ndarray:
    """col: (..., 4) impact column (dy2, dy5, dy10, r_act). True if it has the shock's profile.

    Levels 0-1 are the restrictions of Cieslak and Pang. Levels 2-4 relax what involves the
    2-year yield, for periods where the short end is pinned at the lower bound (2009-2015 in the
    United States, 2014-2021 in Germany); the date is then flagged in the output."""
    a2, a5, a10, s = col[..., 0], col[..., 1], col[..., 2], col[..., 3]
    if level == 4:
        pos, neg = (a5 > 0) & (a10 > 0), (a5 < 0) & (a10 < 0)
        return {"g": (s > 0) & pos, "m": (s < 0) & pos, "pp": (s < 0) & neg, "pc": (s < 0) & pos}[shock]
    if level in (2, 3):
        two_pos = (a2 > 0) if level == 2 else True
        two_neg = (a2 < 0) if level == 2 else True
        if shock == "g":
            return (s > 0) & two_pos & (a10 > 0) & (a5 > a10)
        if shock == "m":
            return (s < 0) & two_pos & (a10 > 0) & (a5 > a10)
        if shock == "pp":
            return (s < 0) & two_neg & (a5 < 0) & (a10 < a5)
        if shock == "pc":
            return (s < 0) & two_pos & (a5 > 0) & (a10 > a5)
    if shock == "g":
        return (s > 0) & (a10 > 0) & (a2 > a10) & (a5 > a10)
    if shock == "m":
        return (s < 0) & (a10 > 0) & (a2 > a5) & (a5 > a10)
    if shock == "pp":
        return (s < 0) & (a2 < 0) & (a5 < a2) & (a10 < a5)
    if shock == "pc":
        return (s < 0) & (a2 > 0) & (a5 > a2) & (a10 > a5)
    raise ValueError(shock)


SHOCK_NAMES = ("g", "m", "pp", "pc")


def identify(C: np.ndarray, share_restrictions: bool = True, level: int = 0) -> np.ndarray:
    """C: (N, 4, 4) candidate impact matrices P @ Q. Returns the admissible ones, with columns
    re-ordered as (g, m, pp, pc) and oriented so that each column has its shock's signs.
    The variance-share restrictions apply at level 0 only."""
    N = C.shape[0]
    ok = np.empty((N, 4, 4, 2), dtype=bool)          # draw, column, shock, sign
    for j in range(4):
        col = C[:, :, j]
        for si, sgn in enumerate((1.0, -1.0)):
            c = sgn * col
            for k, name in enumerate(SHOCK_NAMES):
                ok[:, j, k, si] = column_ok(c, name, level)
    any_sign = ok.any(axis=3)
    chosen = np.full(N, -1)
    ar = np.arange(4)
    for p_idx, perm in enumerate(PERMS):             # perm[k] = column assigned to shock k
        good = (chosen < 0) & any_sign[:, perm, ar].all(axis=1)
        chosen[good] = p_idx
    idx = np.flatnonzero(chosen >= 0)
    if idx.size == 0:
        return np.empty((0, 4, 4))
    cols = PERMS[chosen[idx]]                         # (M, 4)
    B = np.take_along_axis(C[idx], cols[:, None, :], axis=2)
    pos = ok[idx][np.arange(idx.size)[:, None], cols, ar[None, :], 0]
    B = B * np.where(pos, 1.0, -1.0)[:, None, :]
    if share_restrictions and level == 0:
        sq = B ** 2
        short2 = (sq[:, 0, 0] + sq[:, 0, 1]) / (sq[:, 0, 2] + sq[:, 0, 3])
        long10 = (sq[:, 2, 0] + sq[:, 2, 1]) / (sq[:, 2, 2] + sq[:, 2, 3])
        B = B[(short2 > 1) & (long10 < 1)]
    return B


def identified_set(sigma: np.ndarray, bank: RotationBank, n_accept: int = 1000,
                   max_draws: int = 2_000_000, share_restrictions: bool = True, level: int = 0,
                   min_accept: int = 1, probe: int = 300_000):
    """Admissible impact matrices B = P Q for a covariance matrix, at one restriction level.

    Stops early when, after ``probe`` draws, the acceptance rate cannot deliver ``min_accept``
    matrices within ``max_draws``. Returns (B (M,4,4), acceptance rate, draws used)."""
    P = np.linalg.cholesky(sigma)
    kept, n_kept, used, i = [], 0, 0, 0
    while n_kept < n_accept and used < max_draws:
        Q = bank.get(i)
        B = identify(P @ Q, share_restrictions, level)
        kept.append(B)
        n_kept += len(B)
        used += len(Q)
        i += 1
        if used >= probe and n_kept / used * max_draws < min_accept:
            break
    B_all = np.concatenate(kept)[:n_accept] if kept else np.empty((0, 4, 4))
    rate = n_kept / used if used else 0.0
    return B_all, rate, used


def identified_set_fallback(sigma: np.ndarray, bank: RotationBank, n_accept: int, max_draws: int,
                            share_restrictions: bool, min_accept: int):
    """Try the restriction levels in order (0 = full Cieslak-Pang) until ``min_accept`` admissible
    matrices are found. Returns (B, acceptance rate, draws, level used)."""
    start = 0 if share_restrictions else 1
    total = 0
    for level in range(start, N_LEVELS):
        B, rate, used = identified_set(sigma, bank, n_accept, max_draws, share_restrictions, level,
                                       min_accept=min_accept)
        total += used
        if len(B) >= min_accept:
            return B, rate, total, level
    return B, rate, total, N_LEVELS


def median_target(B_all: np.ndarray) -> np.ndarray:
    """Fry-Pagan median-target solution: the admissible matrix closest to the coefficient-wise
    medians, in standardized distance."""
    flat = B_all.reshape(len(B_all), -1)
    med, sd = np.median(flat, axis=0), flat.std(axis=0) + 1e-12
    return B_all[np.argmin((((flat - med) / sd) ** 2).sum(axis=1))]
