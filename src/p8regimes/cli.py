"""Command line:  p8 fetch [--market US|EA|all]   |   p8 run --market US [--demo] [--n-accept N]"""
from __future__ import annotations

import argparse
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from .config import Params, get_paths  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="p8", description="P8 – régimes actions-taux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="télécharger les données publiques")
    f.add_argument("--market", default="all", choices=["US", "EA", "all"])
    r = sub.add_parser("run", help="séance complète : indicateur, backtest, évaluation, figures")
    r.add_argument("--market", default="US", choices=["US", "EA"])
    r.add_argument("--demo", action="store_true", help="données simulées (sans internet)")
    r.add_argument("--n-accept", type=int, default=None)
    r.add_argument("--no-hmm", action="store_true")
    sub.add_parser("status", help="état des données sur le disque")
    a = ap.parse_args(argv)

    paths = get_paths()
    pd.set_option("display.width", 200)
    if a.cmd == "fetch":
        from .data import fetch
        fetch(a.market, paths)
    elif a.cmd == "status":
        from .data import data_status
        print(data_status(paths))
    elif a.cmd == "run":
        from .pipeline import run_session, save_tables
        p = Params()
        if a.n_accept:
            p.n_accept = a.n_accept
        s = run_session(a.market, p, paths, demo=a.demo, with_hmm=not a.no_hmm)
        out = paths.outputs / datetime.now().strftime("%Y-%m-%d")
        save_tables(s, out)
        print(pd.Series(s.regime_now()).to_string())
        print(s.perf["tout l'échantillon"][["Sharpe", "perte max (%)", "rendement (%/an)",
                                             "volatilité (%/an)"]].round(2).to_string())
        print(f"sorties : {out}")


if __name__ == "__main__":
    main()
