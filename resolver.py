"""Central ticker + currency resolver.

Every tool that accepts a user-supplied ticker funnels it through `resolve()`
to obtain a (yf_symbol, currency, unit_scale, exchange) triple before calling
yfinance. GBX→GBP scaling is applied here once, not in each tool.

Accepted input forms:
- Plain yFinance symbols:  AAPL, COPX.L, SAP.DE, RARE.L
- Bare symbols (no suffix): COPX  → probes US, .L, .DE, .F, … until one works
- MIC-style prefix:        LSE:COPX, XETRA:SAP, NASDAQ:AAPL
- Trading 212 raw:         COPX_L_EQ, AAPL_US_EQ, RAREl (compact)
- ISIN:                    IE00B4L5Y983  → OpenFIGI / yfinance lookup
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedTicker:
    yf_symbol: str              # e.g. "COPX.L"
    display: str                # original user input or preferred display form
    currency: str               # ISO-4217 like "GBP", "USD", "EUR"
    unit_scale: float = 1.0     # 0.01 when yfinance returns GBX but we want GBP
    exchange: str = ""          # "LSE", "XETRA", "NASDAQ", …
    isin: Optional[str] = None

    @property
    def needs_scale(self) -> bool:
        return self.unit_scale != 1.0


# ---------------------------------------------------------------------------
# Static maps
# ---------------------------------------------------------------------------
# Exchange prefix (MIC / common abbrev) → yFinance suffix
_MIC_MAP: dict[str, str] = {
    "LSE": ".L",   "XLON": ".L",    "LON": ".L",
    "XETRA": ".DE", "XETR": ".DE",  "ETR": ".DE",   "GER": ".DE",
    "XFRA": ".F",  "FRA": ".F",     "FWB": ".F",
    "EPA": ".PA",  "XPAR": ".PA",   "PAR": ".PA",
    "AMS": ".AS",  "XAMS": ".AS",
    "MIL": ".MI",  "BIT": ".MI",    "XMIL": ".MI",  "MTA": ".MI",
    "SWX": ".SW",  "SIX": ".SW",    "EBS": ".SW",
    "MAD": ".MC",  "BME": ".MC",    "XMAD": ".MC",
    "TSE": ".TO",  "TSX": ".TO",    "XTSE": ".TO",
    "HKEX": ".HK", "HKG": ".HK",    "XHKG": ".HK",
    "TKY": ".T",   "TYO": ".T",     "XTKS": ".T",   "JPX": ".T",
    "ASX": ".AX",  "XASX": ".AX",
    "STO": ".ST",  "XSTO": ".ST",
    "CPH": ".CO",  "XCSE": ".CO",
    "HEL": ".HE",  "XHEL": ".HE",
    "OSL": ".OL",  "XOSL": ".OL",
    "WAR": ".WA",  "XWAR": ".WA",
    "VIE": ".VI",  "XWBO": ".VI",
    "B3":  ".SA",  "BVMF": ".SA",   "BOV": ".SA",
    "SGX": ".SI",  "XSES": ".SI",
    # US — no suffix
    "NYSE": "",    "NASDAQ": "",   "NAS": "",  "NYS": "",
    "ARCA": "",    "BATS": "",     "AMEX": "", "US": "",
}

# yFinance suffix → ISO currency (best-guess default; probed afterwards for GBX)
_SUFFIX_CURRENCY: dict[str, str] = {
    ".L":  "GBP",  ".DE": "EUR",   ".F":  "EUR",   ".X":  "EUR",   ".BE": "EUR",
    ".PA": "EUR",  ".AS": "EUR",   ".MI": "EUR",   ".SW": "CHF",   ".MC": "EUR",
    ".TO": "CAD",  ".HK": "HKD",   ".T":  "JPY",   ".AX": "AUD",
    ".ST": "SEK",  ".CO": "DKK",   ".HE": "EUR",   ".OL": "NOK",
    ".WA": "PLN",  ".VI": "EUR",   ".SA": "BRL",   ".SI": "SGD",
}

_SUFFIX_EXCHANGE: dict[str, str] = {
    ".L":  "LSE",  ".DE": "XETRA", ".F":  "FRA",   ".X":  "FRA",   ".BE": "BER",
    ".PA": "EPA",  ".AS": "AMS",   ".MI": "MIL",   ".SW": "SWX",   ".MC": "MAD",
    ".TO": "TSX",  ".HK": "HKEX",  ".T":  "TSE",   ".AX": "ASX",
    ".ST": "STO",  ".CO": "CPH",   ".HE": "HEL",   ".OL": "OSL",
    ".WA": "WAR",  ".VI": "VIE",   ".SA": "B3",    ".SI": "SGX",
}

# Expanded fallback tried for bare symbols (order matters — US first, then LSE/DE)
_BARE_FALLBACK_SUFFIXES: list[str] = [
    "", ".L", ".DE", ".F", ".X", ".BE",
    ".PA", ".AS", ".MI", ".SW", ".MC", ".TO", ".HK",
]

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
_resolution_cache: dict[str, ResolvedTicker] = {}
_probe_cache: TTLCache = TTLCache(maxsize=2048, ttl=3600)   # 1h
_unit_scale_cache: TTLCache = TTLCache(maxsize=1024, ttl=21600)  # 6h


# ---------------------------------------------------------------------------
# Seed cache from known_tickers.json (portfolio-known aliases)
# ---------------------------------------------------------------------------
_SEED_PATH = Path(__file__).parent / "known_tickers.json"


def _load_seed_cache() -> None:
    if not _SEED_PATH.exists():
        return
    try:
        data = json.loads(_SEED_PATH.read_text())
    except Exception as e:
        log.warning("Failed to parse %s: %s", _SEED_PATH, e)
        return
    for entry in data.get("entries", []):
        try:
            rt = ResolvedTicker(
                yf_symbol=entry["yf_symbol"],
                display=entry.get("display", entry["yf_symbol"]),
                currency=entry.get("currency", "USD"),
                unit_scale=float(entry.get("unit_scale", 1.0)),
                exchange=entry.get("exchange", ""),
                isin=entry.get("isin"),
            )
        except KeyError:
            continue
        aliases = list(entry.get("aliases", [])) + [entry["yf_symbol"]]
        if entry.get("isin"):
            aliases.append(entry["isin"])
        for alias in aliases:
            _resolution_cache[alias.upper()] = rt


_load_seed_cache()


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def _candidates(raw: str) -> list[str]:
    """Ordered list of yFinance-format symbol candidates for the given input."""
    r = raw.strip()
    up = r.upper()

    # 1. ISIN → OpenFIGI / yfinance
    if _ISIN_RE.match(up):
        cands = _isin_to_tickers(up)
        return cands or [up]  # fallback to ISIN itself so caller gets clear "no data"

    # 2. EXCHANGE:SYMBOL
    if ":" in up:
        prefix, sym = (p.strip() for p in up.split(":", 1))
        suffix = _MIC_MAP.get(prefix)
        if suffix is not None:
            base = sym.replace(" ", "")
            return [f"{base}{suffix}"] if suffix else [base]
        # Unknown prefix — try the RHS as a bare symbol
        r = sym
        up = sym

    # 3. Trading 212 format (underscores) or compact lowercase-suffix
    if "_" in up or (len(up) > 1 and r[-1].islower() and r[:-1].isupper()):
        # Lazy import to avoid circular dep with helpers
        from helpers import strip_t212_ticker
        converted = strip_t212_ticker(r)
        cands = [converted]
        if "." not in converted:
            base = converted
            cands += [f"{base}{sfx}" for sfx in _BARE_FALLBACK_SUFFIXES
                      if sfx and f"{base}{sfx}" not in cands]
        return cands

    # 4. Already has a yFinance-style suffix
    if "." in up:
        base, suffix = up.split(".", 1)
        suffix = "." + suffix
        cands = [up]
        if suffix == ".DE":
            # .DE ETFs/small-caps often missing — try Frankfurt variants
            for alt in (".F", ".X", ".BE"):
                alt_sym = f"{base}{alt}"
                if alt_sym not in cands:
                    cands.append(alt_sym)
        elif suffix in (".F", ".X", ".BE"):
            for alt in (".DE", ".F", ".X", ".BE"):
                alt_sym = f"{base}{alt}"
                if alt_sym not in cands:
                    cands.append(alt_sym)
        return cands

    # 5. Bare symbol — try US, then major exchanges
    return [f"{up}{sfx}" for sfx in _BARE_FALLBACK_SUFFIXES]


def _isin_to_tickers(isin: str) -> list[str]:
    """Resolve an ISIN to candidate yFinance tickers via OpenFIGI, then yfinance."""
    cands: list[str] = []

    # OpenFIGI — anonymous: 25 req/6s; with key (OPENFIGI_API_KEY): 250 req/6s
    try:
        import httpx
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("OPENFIGI_API_KEY", "")
        if key:
            headers["X-OPENFIGI-APIKEY"] = key
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                "https://api.openfigi.com/v3/mapping",
                headers=headers,
                json=[{"idType": "ID_ISIN", "idValue": isin}],
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload and isinstance(payload, list) and "data" in payload[0]:
                    for hit in payload[0]["data"]:
                        tkr = hit.get("ticker")
                        mic = (hit.get("exchCode") or hit.get("micCode") or "").upper()
                        if not tkr:
                            continue
                        suffix = _MIC_MAP.get(mic)
                        if suffix is None:
                            continue
                        cand = f"{tkr}{suffix}" if suffix else tkr
                        if cand not in cands:
                            cands.append(cand)
    except Exception as e:
        log.debug("OpenFIGI lookup failed for %s: %s", isin, e)

    # yfinance fallback
    try:
        if hasattr(yf.utils, "get_ticker_by_isin"):
            t = yf.utils.get_ticker_by_isin(isin)
            if t and t not in cands:
                cands.append(t)
    except Exception:
        pass

    return cands


# ---------------------------------------------------------------------------
# Probing + GBX detection
# ---------------------------------------------------------------------------
def _silent_history(symbol: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    import logging
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    try:
        return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        return pd.DataFrame()


def _probe_sync(symbol: str) -> bool:
    if symbol in _probe_cache:
        return _probe_cache[symbol]
    df = _silent_history(symbol, period="5d", interval="1d")
    ok = (not df.empty) and len(df) >= 2
    _probe_cache[symbol] = ok
    return ok


def _detect_unit_scale(symbol: str) -> tuple[float, str]:
    """Probe yfinance fast_info for the actual listing currency and unit scale.

    For LSE (.L) tickers also applies GBX→GBP pence detection.
    Returns (scale, currency_iso4217). Falls back to _SUFFIX_CURRENCY hint on error.

    yfinance pence variants: "GBX" (uppercase) or "GBp" (mixed-case) — both handled.
    T212 pence variant: "GBX" (handled separately in helpers.position_value).
    """
    if symbol in _unit_scale_cache:
        return _unit_scale_cache[symbol]

    suffix = "." + symbol.split(".", 1)[1] if "." in symbol else ""
    hint_ccy = _SUFFIX_CURRENCY.get(suffix, "USD")

    try:
        fi = yf.Ticker(symbol).fast_info
        cur_raw = getattr(fi, "currency", "") or ""
        cur_u = str(cur_raw).upper()
        last = getattr(fi, "last_price", None)
        last_f = float(last) if last is not None else 0.0

        if suffix == ".L":
            # LSE: detect GBX (pence) pricing — yfinance uses "GBX" or "GBp"
            if cur_u == "GBX" or cur_raw == "GBp":
                result = (0.01, "GBP")
            elif cur_u == "GBP" and last_f > 1000:
                # Yahoo mislabels some GBX-quoted ETFs as GBP (SGLN.L etc).
                # Genuine GBP equities trade < £500; values > £1000 are pence.
                result = (0.01, "GBP")
            elif cur_u and cur_u not in ("GBX", "GBP"):
                # LSE-listed but priced in a foreign currency (e.g. COPX.L = USD)
                result = (1.0, cur_u)
            else:
                result = (1.0, "GBP")
        else:
            # Non-LSE: trust yfinance currency when available; suffix map is the fallback
            if cur_u and cur_u != "GBX":
                result = (1.0, cur_u)
            else:
                result = (1.0, hint_ccy)
    except Exception:
        result = (1.0, hint_ccy)

    _unit_scale_cache[symbol] = result
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve(raw: str, *, probe: bool = True) -> ResolvedTicker:
    """Resolve arbitrary user-supplied ticker into a ResolvedTicker.

    BLOCKING: probes yfinance synchronously on cache miss. Use `aresolve`
    from async contexts.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError(f"resolve() got invalid input: {raw!r}")
    key = raw.strip().upper()
    cached = _resolution_cache.get(key)
    if cached is not None:
        return cached

    cands = _candidates(raw)
    chosen: Optional[str] = None
    if probe:
        for c in cands:
            if _probe_sync(c):
                chosen = c
                break
    if chosen is None:
        chosen = cands[0] if cands else raw.strip().upper()

    suffix = "." + chosen.split(".", 1)[1] if "." in chosen else ""
    currency, scale = (_SUFFIX_CURRENCY.get(suffix, "USD"), 1.0)
    if suffix and probe:
        # Probe actual currency for all non-US tickers; handles LSE foreign-currency
        # stocks (e.g. COPX.L = USD) and alternative-ticker resolution (LYI.DE→LYI.F)
        scale, currency = _detect_unit_scale(chosen)
    exchange = _SUFFIX_EXCHANGE.get(suffix, "NYSE/NASDAQ" if not suffix else "")

    rt = ResolvedTicker(
        yf_symbol=chosen,
        display=raw.strip(),
        currency=currency,
        unit_scale=scale,
        exchange=exchange,
        isin=key if _ISIN_RE.match(key) else None,
    )
    _resolution_cache[key] = rt
    _resolution_cache[chosen.upper()] = rt
    return rt


async def aresolve(raw: str, *, probe: bool = True) -> ResolvedTicker:
    return await asyncio.to_thread(resolve, raw, probe=probe)


async def bulk_resolve(raws: list[str]) -> dict[str, ResolvedTicker]:
    """Resolve many tickers in parallel. Returns original-input → ResolvedTicker."""
    async def _one(r: str) -> tuple[str, ResolvedTicker]:
        try:
            return r, await aresolve(r)
        except Exception as e:
            log.warning("Resolve failed for %s: %s", r, e)
            return r, ResolvedTicker(
                yf_symbol=r.strip().upper(), display=r, currency="USD", unit_scale=1.0,
            )
    pairs = await asyncio.gather(*[_one(r) for r in raws])
    return dict(pairs)


async def fetch_history(
    raw: str, period: str = "1mo", interval: str = "1d",
) -> tuple[ResolvedTicker, pd.DataFrame]:
    """Fetch OHLCV history via yfinance, unit-scaled to the resolved currency."""
    rt = await aresolve(raw)
    df = await asyncio.to_thread(_silent_history, rt.yf_symbol, period, interval)
    if df.empty:
        return rt, df
    if rt.unit_scale != 1.0:
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = df[col] * rt.unit_scale
    return rt, df


async def fetch_fast_price(
    raw: str,
) -> tuple[ResolvedTicker, Optional[float], Optional[float]]:
    """Return (resolved, last_price, previous_close) — unit-scaled."""
    rt = await aresolve(raw)

    def _fetch():
        try:
            fi = yf.Ticker(rt.yf_symbol).fast_info
            last = float(fi.last_price) if fi.last_price is not None else None
            prev = float(fi.previous_close) if fi.previous_close is not None else None
            return last, prev
        except Exception:
            return None, None

    last, prev = await asyncio.to_thread(_fetch)
    if rt.unit_scale != 1.0:
        if last is not None: last *= rt.unit_scale
        if prev is not None: prev *= rt.unit_scale
    return rt, last, prev


async def fetch_fundamental_dict(raw: str) -> tuple[ResolvedTicker, dict]:
    """Fetch essential fundamental data merging .info and .fast_info.
    This provides resilience against yfinance returning empty .info dicts."""
    rt = await aresolve(raw)
    
    def _fetch():
        t = yf.Ticker(rt.yf_symbol)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass
            
        # Merge key metrics from fast_info if missing in info
        try:
            fi = t.fast_info
            if not info.get("marketCap") and getattr(fi, "market_cap", None):
                info["marketCap"] = fi.market_cap
            if not info.get("currentPrice") and getattr(fi, "last_price", None):
                info["currentPrice"] = fi.last_price
            if not info.get("currency") and getattr(fi, "currency", None):
                info["currency"] = fi.currency
            if not info.get("sharesOutstanding") and getattr(fi, "shares", None):
                info["sharesOutstanding"] = fi.shares
        except Exception:
            pass
            
        return info

    from helpers import YF_INFO_SEM
    async with YF_INFO_SEM:
        info = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=12.0)
    return rt, info


async def fetch_historic_prices_scaled(
    raws: list[str], period: str = "1y", interval: str = "1d",
) -> tuple[dict[str, ResolvedTicker], pd.DataFrame]:
    """Unit-scaled close-price frame for many tickers.

    Returns (resolution_map, closes_df). `closes_df.columns` = resolved `yf_symbol`s.
    """
    resolutions = await bulk_resolve(raws)
    unique_syms = list(dict.fromkeys(rt.yf_symbol for rt in resolutions.values()))
    if not unique_syms:
        return resolutions, pd.DataFrame()

    def _download(syms):
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        try:
            return yf.download(syms, period=period, interval=interval,
                               auto_adjust=True, progress=False)
        except Exception:
            return pd.DataFrame()

    data = await asyncio.to_thread(_download, unique_syms)
    if data is None or data.empty:
        return resolutions, pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = data[["Close"]].rename(columns={"Close": unique_syms[0]})
    closes = closes.dropna(axis=1, how="all")

    for rt in resolutions.values():
        if rt.unit_scale != 1.0 and rt.yf_symbol in closes.columns:
            closes[rt.yf_symbol] = closes[rt.yf_symbol] * rt.unit_scale

    return resolutions, closes


# ---------------------------------------------------------------------------
# Testing hooks — used by tests/test_resolver.py
# ---------------------------------------------------------------------------
def _clear_caches_for_tests() -> None:
    _resolution_cache.clear()
    _probe_cache.clear()
    _unit_scale_cache.clear()
