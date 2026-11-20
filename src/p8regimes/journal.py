"""Trace of each working session: journal/journal.md and an HTML copy of the notebook."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Paths, get_paths


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n.d."
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def write_journal(session, note: str = "", paths: Paths | None = None) -> Path:
    paths = paths or get_paths()
    path = paths.journal / "journal.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    reg = session.regime_now()
    perf = session.perf.get("tout l'échantillon")
    lines = [f"\n## {now} — {session.market}\n"]
    lines.append(f"- Données jusqu'au {session.panel.daily.index[-1].date()} ; "
                 f"indicateur sur {session.ind['hb'].notna().sum()} fins de mois.")
    when = f"{reg['date']}, mois en cours : lecture provisoire" if reg.get("provisoire") else f"{reg['date']}"
    lines.append(f"- Dernier mois ({when}) : HB {_fmt(reg['HB'], 2)} "
                 f"[{_fmt(reg['HB p16'], 2)} ; {_fmt(reg['HB p84'], 2)}], "
                 f"bêta prévu {_fmt(reg['bêta prévu'])}, régime P8-HB : **{reg['régime P8-HB']}**, "
                 f"P8-bêta : **{reg['régime P8-bêta']}**.")
    if perf is not None:
        keep = [k for k in ("60/40", "VC", "CORR-A", "P8-HB-A", "P8-BETA-A", "P8-HB-B") if k in perf.index]
        cols = ["Sharpe", "perte max (%)", "rendement (%/an)", "volatilité (%/an)"]
        lines.append("- Performance (tout l'échantillon) : " + " ; ".join(
            f"{k} Sharpe {perf.loc[k, 'Sharpe']:.2f}, perte max {perf.loc[k, 'perte max (%)']:.1f} %"
            for k in keep))
    if session.criteria is not None and len(session.criteria):
        crit = session.criteria
        lines.append("- Critères 1-2 : " + " ; ".join(
            f"{i} : {'oui' if r.iloc[0] else 'non'}/{'oui' if r.iloc[1] else 'non'}" for i, r in crit.iterrows()))
        c3 = crit.attrs.get("critère 3", {})
        if c3:
            lines.append("- Critère 3 (Clark-West, 10 %) : " + " ; ".join(
                f"{k} {'oui' if v else 'non'}" for k, v in c3.items()))
    fp = getattr(session, "forecast_periods", None)
    if fp is not None and len(fp) and "R² hors échantillon" in fp:
        items = []
        for (per, mod), row in fp.iterrows():
            if np.isfinite(row.get("R² hors échantillon", np.nan)):
                items.append(f"{per} {mod} R² {100 * row['R² hors échantillon']:+.1f} %, p {row['p-valeur']:.2f}")
        if items:
            lines.append("- Critère 3 par sous-période (robustesse) : " + " ; ".join(items))
    r = getattr(session, "robust", None) or {}
    if "dsr_6040" in r:
        d = r["dsr_6040"]
        col = [c for c in d.columns if c.startswith("DSR (N effectif")][0]
        keep = [k for k in ("P8-HB-A", "P8-BETA-A", "CORR-A") if k in d.index]
        lines.append("- Sharpe déflaté du rendement actif contre le 60/40 (" + col + ") : " + " ; ".join(
            f"{k} {d.loc[k, col]:.2f}" for k in keep))
    if "break_even" in r:
        be = r["break_even"]
        keep = [k for k in ("P8-HB-A", "P8-BETA-A") if k in be.index]
        lines.append("- Coût seuil contre la vol. cible : " + " ; ".join(
            f"{k} {be.loc[k].get('seuil contre VC', 'n.d.')}" for k in keep))
    if "hmm" in r:
        t = r["hmm"]["table"]
        lines.append(f"- Stabilité HMM : accord avec la base de {t['accord avec la base (%)'].iloc[1:].min():.0f} "
                     f"à {t['accord avec la base (%)'].iloc[1:].max():.0f} % selon la configuration ; "
                     f"accord moyen entre ré-estimations {t['accord entre ré-estimations (%)'].iloc[0]:.0f} %.")
    if getattr(session, "pocket_check", None):
        lines.append(f"- Contrôle poche taux : {session.pocket_check['verdict']}")
    if note.strip():
        lines.append(f"- Note : {note.strip()}")
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Journal des séances P8\n")
        f.write("\n".join(lines) + "\n")
    return path


class NotebookNotSaved(RuntimeError):
    """The notebook file on disk is older than the end of the session: the copy would show an older run."""


def saved_market(notebook: Path) -> str | None:
    """Value of MARCHE in the parameter cell of the notebook as saved on disk."""
    import json
    import re
    nb = json.loads(Path(notebook).read_text(encoding="utf-8"))
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            m = re.search(r'^MARCHE\s*=\s*["\'](\w+)["\']', src, flags=re.M)
            if m:
                return m.group(1)
    return None


def check_saved(notebook: Path, after: datetime | None = None, market: str | None = None) -> None:
    """Refuse to copy a notebook that was not saved after the session (``after``) or whose parameter
    cell shows another market than the one just run."""
    notebook = Path(notebook)
    if after is not None and datetime.fromtimestamp(notebook.stat().st_mtime) < after:
        raise NotebookNotSaved("le notebook n'a pas été enregistré depuis la fin de la séance : "
                               "fais Ctrl+S, puis relance uniquement cette cellule.")
    if market is not None:
        on_disk = saved_market(notebook)
        if on_disk is not None and on_disk != market:
            raise NotebookNotSaved(f"le notebook enregistré est une séance {on_disk}, pas {market} : "
                                   "fais Ctrl+S, puis relance uniquement cette cellule.")


def export_notebook_html(notebook: Path, paths: Paths | None = None, after: datetime | None = None,
                         market: str | None = None) -> Path:
    """HTML copy of the (saved) notebook in outputs/YYYY-MM-DD/seance_YYYY-MM-DD_HHhMM.html.

    With ``after`` (end of the session) and ``market``, the copy is refused when the file on disk is
    older than the session or belongs to another market (the notebook was not saved)."""
    paths = paths or get_paths()
    check_saved(notebook, after, market)
    now = datetime.now()
    out_dir = paths.outputs / now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"seance_{now.strftime('%Y-%m-%d_%Hh%M')}"
    cmd = [sys.executable, "-m", "nbconvert", "--to", "html", str(notebook),
           "--output-dir", str(out_dir), "--output", name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return out_dir / f"{name}.html"
