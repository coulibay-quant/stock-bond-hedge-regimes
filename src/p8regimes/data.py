"""Download, parse and cache the public data (all free, no API key).

United States
    * Treasury zero-coupon curve, Gurkaynak-Sack-Wright (Federal Reserve), daily since 1961
    * CRSP value-weighted market return and 1-month T-bill, Kenneth French data library, daily
    * CPI-U (FRED, CPIAUCSL), monthly
Euro area
    * German Federal securities zero-coupon curve (Bundesbank, BBSIS), daily
      fallback: euro area AAA curve of the ECB (YC), daily since 2004-09
    * DAX performance index (yfinance), fallback Stooq
    * EONIA then euro short-term rate (ECB data portal; several EONIA keys tried)
    * Euro area HICP, annual rate of change (ECB data portal): ICP dataset up to Dec 2025, then the
      HICP dataset that replaced it on 4 Feb 2026

Every source can be replaced by a file dropped in ``data/raw/manual/`` (see LISEZMOI.txt):
the manual file wins over the download. Parsers are written to be tolerant: they look for
dates and numbers rather than relying on fixed layouts.
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Paths, get_paths

HEADERS = {"User-Agent": "Mozilla/5.0 (p8regimes research script, public data)"}

URLS = {
    "gsw": "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
    "ff": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "cpi": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
    "bbk_rest": "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R{m}XX.R.A.A._Z._Z.A?format=csv&lang=en",
    "bbk_classic": "https://www.bundesbank.de/statistic-rmi/StatisticDownload?tsId=BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R{m}XX.R.A.A._Z._Z.A&its_csvFormat=en&its_fileFormat=csv&mode=its",
    "ecb_yc": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{n}Y?format=csvdata",
    # EONIA: several keys tried in order (the FM key returned HTTP 404 in November 2026)
    "eonia": ["https://data-api.ecb.europa.eu/service/data/EON/D.EONIA_TO.RATE?format=csvdata",
              "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.MM.EONIA.HSTA?format=csvdata"],
    "estr": "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata",
    # HICP: the ICP dataset was discontinued on 4 Feb 2026 (last month: Dec 2025) and replaced by the
    # HICP dataset (ECOICOP 2, 2025 = 100, Bulgaria included). Old series up to its last month, new after.
    "hicp_old": "https://data-api.ecb.europa.eu/service/data/ICP/M.U2.N.000000.4.ANR?format=csvdata",
    "hicp_new": "https://data-api.ecb.europa.eu/service/data/HICP/M.U2.N.000000.4D0.ANR?format=csvdata",
    "stooq_dax": "https://stooq.com/q/d/l/?s=^dax&i=d",
}

EA_MATURITIES = (1, 2, 5, 6, 7, 8, 10)     # years, German curve
US_CURVE_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2",
                 "SVENY01", "SVENY02", "SVENY05", "SVENY07", "SVENY10"]

# files expected in data/raw/manual/ (any of them overrides the download)
MANUAL_FILES = {
    "us_gsw": "feds200628.csv",
    "us_ff": "F-F_Research_Data_Factors_daily.csv",
    "us_cpi": "CPIAUCSL.csv",
    "ea_bbk": "bbk_R{m}XX.csv",          # one file per maturity, m = 01, 02, 05, 06, 07, 08, 10
    "ea_dax": "dax.csv",
    "ea_eonia": "eonia.csv",
    "ea_estr": "estr.csv",
    "ea_hicp": "hicp.csv",
    "us_ief": "ief.csv",                 # colonnes Date et Close AJUSTÉ des dividendes
}


# ======================================================================== HTTP
def _get(url: str, retries: int = 3, timeout: int = 90) -> requests.Response:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 50:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:   # network error: retry
            last = e
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"download failed: {url} ({last})")


def _get_first(urls) -> tuple[requests.Response, str]:
    """First URL of the list that answers; returns (response, url)."""
    urls = [urls] if isinstance(urls, str) else list(urls)
    errors = []
    for u in urls:
        try:
            return _get(u, retries=2), u
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(" ; ".join(errors))


def _text(content: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


# ======================================================================== parsers
_ISO = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")
_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def _to_float(tok: str) -> float:
    tok = tok.strip().strip('"').strip()
    if not tok or not _NUM.match(tok):
        return np.nan
    return float(tok.replace(",", "."))


def parse_gsw(text: str) -> pd.DataFrame:
    """Federal Reserve feds200628.csv: free-text header, then 'Date,BETA0,...'."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().strip('"').startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["NA", "", " "])
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    keep = [c for c in US_CURVE_COLS if c in df.columns]
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    return out.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"], how="any")


def parse_french_daily(content: bytes | str) -> pd.DataFrame:
    """Kenneth French daily factors (zip or csv text). Returns Mkt-RF and RF in % per day."""
    if isinstance(content, bytes) and content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            text = _text(z.read(name))
    else:
        text = content if isinstance(content, str) else _text(content)
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and re.fullmatch(r"\d{8}", parts[0]):
            rows.append((parts[0], _to_float(parts[1]), _to_float(parts[4])))
    if not rows:
        raise ValueError("no daily rows found in the French file")
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


def parse_two_columns(text: str) -> pd.Series:
    """CSV with a date column and a value column (FRED, Stooq, simple manual files)."""
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in
                     ("date", "observation_date", "time_period", "datum")), df.columns[0])
    val_candidates = [c for c in df.columns if c != date_col]
    pref = [c for c in val_candidates if c.lower() in ("close", "adj close", "obs_value", "value")]
    val_col = pref[0] if pref else val_candidates[0]
    s = pd.Series(pd.to_numeric(df[val_col].astype(str).str.replace(",", ".", regex=False),
                                errors="coerce").values,
                  index=pd.to_datetime(df[date_col], errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def parse_sdmx_csv(text: str) -> pd.Series:
    """ECB 'csvdata' or SDMX-CSV: columns TIME_PERIOD and OBS_VALUE."""
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    cols = {c.upper().strip(): c for c in df.columns}
    if "TIME_PERIOD" not in cols or "OBS_VALUE" not in cols:
        raise ValueError("TIME_PERIOD / OBS_VALUE columns not found")
    idx = pd.to_datetime(df[cols["TIME_PERIOD"]].astype(str), errors="coerce")
    s = pd.Series(pd.to_numeric(df[cols["OBS_VALUE"]], errors="coerce").values, index=idx)
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def parse_dated_lines(text: str) -> pd.Series:
    """Tolerant parser: on each line, the first ISO date and the first number after it.

    Works for the Bundesbank 'StatisticDownload' files (header block, then 'date,value,flag')
    and for most single-series CSV exports. Missing values ('.', 'No value available') skip.
    """
    dates, vals = [], []
    for line in text.splitlines():
        m = _ISO.search(line)
        if not m:
            continue
        rest = line[m.end():]
        toks = re.split(r"[;,\t]", rest)
        val = np.nan
        for t in toks:
            v = _to_float(t)
            if np.isfinite(v):
                val = v
                break
        if not np.isfinite(val):
            continue
        y, mo, d = m.group(1), m.group(2), m.group(3) or "01"
        try:
            dates.append(datetime(int(y), int(mo), int(d)))
            vals.append(val)
        except ValueError:
            continue
    if not dates:
        raise ValueError("no dated values found")
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def parse_any_series(text: str) -> pd.Series:
    """Try SDMX columns first, then the tolerant line parser."""
    try:
        s = parse_sdmx_csv(text)
        if len(s):
            return s
    except Exception:
        pass
    return parse_dated_lines(text)


# ======================================================================== fetchers
class Fetcher:
    def __init__(self, paths: Paths | None = None, verbose: bool = True):
        self.paths = paths or get_paths()
        self.verbose = verbose
        self.log: list[str] = []

    def say(self, msg: str) -> None:
        self.log.append(msg)
        if self.verbose:
            print(msg)

    def _manual(self, name: str) -> Path | None:
        p = self.paths.manual / name
        return p if p.exists() else None

    def _save(self, obj, name: str, source: str) -> None:
        path = self.paths.raw / name
        obj.to_csv(path, encoding="utf-8")
        meta_path = self.paths.raw / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        first, last = obj.index.min(), obj.index.max()
        meta[name] = {"source": source, "downloaded": datetime.now().isoformat(timespec="seconds"),
                      "rows": int(len(obj)), "first": str(first.date()), "last": str(last.date())}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self.say(f"  {name:<14} {len(obj):>6} lignes  {first.date()} -> {last.date()}  ({source})")

    # ------------------------------------------------------------------ United States
    def us(self) -> None:
        self.say("Etats-Unis")
        m = self._manual(MANUAL_FILES["us_gsw"])
        text = m.read_text(encoding="utf-8", errors="replace") if m else _text(_get(URLS["gsw"]).content)
        self._save(parse_gsw(text), "us_gsw.csv", "manuel" if m else URLS["gsw"])

        m = self._manual(MANUAL_FILES["us_ff"])
        content = m.read_bytes() if m else _get(URLS["ff"]).content
        self._save(parse_french_daily(content), "us_ff.csv", "manuel" if m else URLS["ff"])

        m = self._manual(MANUAL_FILES["us_cpi"])
        text = m.read_text(encoding="utf-8", errors="replace") if m else _text(_get(URLS["cpi"]).content)
        cpi = parse_two_columns(text).rename("cpi")
        self._save(cpi.to_frame(), "us_cpi.csv", "manuel" if m else URLS["cpi"])

        # contrôle de la poche taux : ETF iShares 7-10 Year Treasury (IEF), rendement total
        m = self._manual(MANUAL_FILES["us_ief"])
        try:
            if m:
                ief, src = parse_two_columns(m.read_text(encoding="utf-8", errors="replace")), "manuel"
            else:
                ief, src = yahoo_adjusted("IEF", "2002-07-01"), "yfinance IEF (cours ajusté des dividendes)"
            self._save(ief.rename("close").to_frame(), "us_ief.csv", src)
        except Exception as e:
            self.say(f"  IEF indisponible ({e}) : le contrôle de la poche taux sera sauté")

    # ------------------------------------------------------------------ euro area
    def _bbk_one(self, mat: int) -> tuple[pd.Series, str]:
        code = f"{mat:02d}"
        m = self._manual(MANUAL_FILES["ea_bbk"].format(m=code))
        if m:
            return parse_any_series(m.read_text(encoding="utf-8", errors="replace")), "manuel"
        errors = []
        for key in ("bbk_rest", "bbk_classic"):
            url = URLS[key].format(m=code)
            try:
                s = parse_any_series(_text(_get(url, retries=2, timeout=45).content))
                if len(s) > 1000:
                    return s, url
                errors.append(f"{key}: only {len(s)} rows")
            except Exception as e:
                errors.append(f"{key}: {e}")
        raise RuntimeError("; ".join(errors))

    def ea(self) -> None:
        self.say("Zone euro")
        cols, sources = {}, set()
        try:
            for mat in EA_MATURITIES:
                s, src = self._bbk_one(mat)
                cols[f"y{mat:02d}"] = s
                sources.add("Bundesbank" if src != "manuel" else "manuel")
            curve = pd.DataFrame(cols).dropna(how="any")
            label = " + ".join(sorted(sources))
        except Exception as e:
            self.say(f"  Bundesbank indisponible ({e}) -> repli sur la courbe AAA de la BCE (depuis 2004)")
            for mat in EA_MATURITIES:
                url = URLS["ecb_yc"].format(n=mat)
                cols[f"y{mat:02d}"] = parse_sdmx_csv(_text(_get(url).content))
            curve = pd.DataFrame(cols).dropna(how="any")
            label = "BCE courbe AAA (repli)"
        self._save(curve, "ea_curve.csv", label)

        m = self._manual(MANUAL_FILES["ea_dax"])
        if m:
            dax, src = parse_two_columns(m.read_text(encoding="utf-8", errors="replace")), "manuel"
        else:
            dax, src = self._dax_download()
        self._save(dax.rename("close").to_frame(), "ea_dax.csv", src)

        parts, labels = [], []
        for key, fname in (("eonia", "ea_eonia"), ("estr", "ea_estr")):
            m = self._manual(MANUAL_FILES[fname])
            try:
                if m:
                    text, src = m.read_text(encoding="utf-8", errors="replace"), "manuel"
                else:
                    r, src = _get_first(URLS[key])
                    text, src = _text(r.content), _series_key(src)
                parts.append(parse_any_series(text).rename(key))
                labels.append(f"{key.upper().replace('ESTR', '€STR')} {src}")
            except Exception as e:
                self.say(f"  {key} indisponible ({e})")
        if parts:
            cash = splice_cash(*parts)
            self._save(cash.rename("rate").to_frame(), "ea_cash.csv", " puis ".join(labels))

        m = self._manual(MANUAL_FILES["ea_hicp"])
        if m:
            hicp = parse_any_series(m.read_text(encoding="utf-8", errors="replace")).rename("yoy")
            self._save(hicp.to_frame(), "ea_hicp.csv", "manuel")
        else:
            got = {}
            for key in ("hicp_old", "hicp_new"):
                try:
                    got[key] = parse_any_series(_text(_get(URLS[key], retries=2).content))
                except Exception as e:
                    self.say(f"  {key} indisponible ({e})")
            if got:
                hicp = splice_after(got.get("hicp_old"), got.get("hicp_new")).rename("yoy")
                label = " puis ".join(_series_key(URLS[k]) for k in ("hicp_old", "hicp_new") if k in got)
                self._save(hicp.to_frame(), "ea_hicp.csv", label)
                if hicp.index.max() < pd.Timestamp.today() - pd.DateOffset(months=4):
                    self.say(f"  attention : l'IPCH s'arrête en {hicp.index.max():%Y-%m} "
                             "(déposer hicp.csv dans data/raw/manual si la série BCE n'est plus à jour)")
            else:
                self.say("  IPCH indisponible : la règle macro sera désactivée en zone euro")

    def _dax_download(self) -> tuple[pd.Series, str]:
        try:
            import yfinance as yf
            df = yf.download("^GDAXI", start="1987-01-01", auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            if len(s) > 1000:
                return s, "yfinance ^GDAXI"
        except Exception as e:
            self.say(f"  yfinance indisponible ({e}) -> Stooq")
        s = parse_two_columns(_text(_get(URLS["stooq_dax"]).content))
        return s, URLS["stooq_dax"]


def yahoo_adjusted(ticker: str, start: str) -> pd.Series:
    """Daily closing price adjusted for dividends and splits (total-return proxy), via yfinance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    if len(s) < 250:
        raise RuntimeError(f"{ticker}: only {len(s)} rows")
    return s


def _series_key(url: str) -> str:
    """'ICP/M.U2.N.000000.4.ANR' from an ECB data API URL (for the source column)."""
    m = re.search(r"/service/data/([^?]+)", url)
    return m.group(1) if m else url


def splice_after(old: pd.Series | None, new: pd.Series | None) -> pd.Series:
    """Old series up to its last date, new series strictly after it."""
    if old is None or old.dropna().empty:
        return new.dropna()
    old = old.dropna()
    if new is None or new.dropna().empty:
        return old
    new = new.dropna()
    return pd.concat([old, new[new.index > old.index.max()]]).sort_index()


def splice_cash(*series: pd.Series) -> pd.Series:
    """EONIA up to the first €STR date, €STR afterwards (both in % per year)."""
    named = {s.name: s.dropna() for s in series}
    if "eonia" in named and "estr" in named:
        cut = named["estr"].index.min()
        return pd.concat([named["eonia"][named["eonia"].index < cut], named["estr"]]).sort_index()
    return next(iter(named.values()))


def fetch(market: str = "all", paths: Paths | None = None, verbose: bool = True) -> list[str]:
    f = Fetcher(paths, verbose)
    if market in ("US", "all"):
        f.us()
    if market in ("EA", "all"):
        f.ea()
    return f.log


# ======================================================================== loading
def _read(paths: Paths, name: str) -> pd.DataFrame:
    p = paths.raw / name
    if not p.exists():
        raise FileNotFoundError(f"{p} absent : lancer d'abord le téléchargement (TELECHARGER = True)")
    df = pd.read_csv(p, index_col=0, parse_dates=True, encoding="utf-8")
    return df.sort_index()


def load_raw(market: str, paths: Paths | None = None) -> dict:
    paths = paths or get_paths()
    if market == "US":
        out = {"curve": _read(paths, "us_gsw.csv"), "ff": _read(paths, "us_ff.csv"),
               "cpi": _read(paths, "us_cpi.csv")["cpi"]}
        out["ief"] = _read(paths, "us_ief.csv")["close"] if (paths.raw / "us_ief.csv").exists() else None
        return out
    if market == "EA":
        out = {"curve": _read(paths, "ea_curve.csv"), "dax": _read(paths, "ea_dax.csv")["close"]}
        out["cash"] = _read(paths, "ea_cash.csv")["rate"] if (paths.raw / "ea_cash.csv").exists() else None
        out["hicp"] = _read(paths, "ea_hicp.csv")["yoy"] if (paths.raw / "ea_hicp.csv").exists() else None
        return out
    raise ValueError(market)


def data_status(paths: Paths | None = None) -> pd.DataFrame:
    """Table of what is on disk (rows, first and last date, source, download time)."""
    paths = paths or get_paths()
    meta_path = paths.raw / "meta.json"
    if not meta_path.exists():
        return pd.DataFrame(columns=["rows", "first", "last", "source", "downloaded"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return pd.DataFrame(meta).T[["rows", "first", "last", "source", "downloaded"]]
