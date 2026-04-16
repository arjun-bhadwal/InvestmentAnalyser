"""
Shared helpers: caching, formatting, ticker conversion.
"""
import hashlib
from functools import wraps

from cachetools import TTLCache

# ---------------------------------------------------------------------------
# Caching — avoids repeated expensive API calls within TTL windows
# ---------------------------------------------------------------------------

cache_fundamentals = TTLCache(maxsize=128, ttl=900)    # 15 min
cache_prices       = TTLCache(maxsize=64,  ttl=300)    # 5 min
cache_macro        = TTLCache(maxsize=16,  ttl=3600)   # 1 hour


def finnhub_retry(func):
    """Synchronous backoff decorator for Finnhub API 429 rate limits."""
    import time
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(4):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    if attempt == 3:
                        raise RuntimeError(f"Finnhub rate limit exhausted after retries: {e}")
                    time.sleep(1.0 + (1.5 ** attempt))  # 2.0s, 2.5s, 3.25s
                else:
                    raise e
    return wrapper


def cached(cache: TTLCache):
    """Decorator to cache async function results by args hash."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            key = hashlib.md5(f"{fn.__name__}:{args}:{kwargs}".encode()).hexdigest()
            if key in cache:
                return cache[key]
            result = await fn(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

# T212 exchange code → yFinance suffix
_T212_EXCHANGE_MAP: dict[str, str] = {
    "US":  "",       # NYSE / NASDAQ — no suffix
    "L":   ".L",     # London Stock Exchange
    "DE":  ".DE",    # XETRA (Germany)
    "AS":  ".AS",    # Euronext Amsterdam
    "PA":  ".PA",    # Euronext Paris
    "MI":  ".MI",    # Borsa Italiana (Milan)
    "MC":  ".MC",    # Bolsa de Madrid
    "SW":  ".SW",    # SIX Swiss Exchange
    "TO":  ".TO",    # Toronto Stock Exchange
    "HK":  ".HK",    # Hong Kong
    "T":   ".T",     # Tokyo Stock Exchange
    "AX":  ".AX",    # ASX (Australia)
    "ST":  ".ST",    # Stockholm (Nasdaq Nordic)
    "CO":  ".CO",    # Copenhagen
    "HE":  ".HE",    # Helsinki
    "OL":  ".OL",    # Oslo
    "WA":  ".WA",    # Warsaw
    "VI":  ".VI",    # Vienna
    "BR":  ".SA",    # B3 (Brazil) — yFinance uses .SA
    "SG":  ".SI",    # Singapore — yFinance uses .SI
}


def strip_t212_ticker(raw: str) -> str:
    """Convert T212 instrument codes to yFinance tickers.

    Examples:
        'AAPL_US_EQ'  → 'AAPL'
        'VWRP_L_EQ'   → 'VWRP.L'
        'BP_L_EQ'     → 'BP.L'
        'SAP_DE_EQ'   → 'SAP.DE'
        'RARE_L_EQ'   → 'RARE.L'
        'RAREl'       → 'RARE.L'
        'EUDFd'       → 'EUDF.DE'
    """
    parts = raw.split("_")
    symbol = parts[0]

    # Middle exchange mapping (e.g. 'SAP_DE_EQ')
    if len(parts) >= 3:
        exchange = parts[-2]
        suffix = _T212_EXCHANGE_MAP.get(exchange, "")
        return f"{symbol}{suffix}"

    # If it reached here, we just have a target symbol (e.g., AAPL_EQ -> AAPL, RAREl_EQ -> RAREl).
    # Handle compact T212 tickers where the last char is lowercase exchange (e.g., 'RAREl', 'EUDFd')
    if len(symbol) > 1 and symbol[-1].islower() and symbol[:-1].isupper():
        compact_map = {
            'l': '.L',   'd': '.DE',  'a': '.AS',  'p': '.PA',
            'm': '.MI',  'e': '.MC',  's': '.SW',  't': '.TO',
            'h': '.HK',  'j': '.T',   'x': '.AX',  'c': '.CO',
            'i': '.HE',  'o': '.OL',  'w': '.WA',  'v': '.VI',
            'b': '.SA',  'g': '.SI',
        }
        suffix = compact_map.get(symbol[-1], "")
        return f"{symbol[:-1]}{suffix}"

    # US or Raw ticker with no recognizable exchange format
    return symbol


def safe_float(val, fallback=0.0) -> float:
    """Safely convert strings, NaNs, and messy API values to a float."""
    if val is None:
        return fallback
    if isinstance(val, (int, float)):
        import math
        return fallback if math.isnan(val) else float(val)
        
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "none", "n/a", "null"):
        return fallback
        
    # Strip everything except digits, comma, period, minus sign
    res = []
    for char in val_str:
        if char.isdigit() or char in ",.-":
            res.append(char)
    val_str = "".join(res)
    
    if not val_str or val_str == "-":
        return fallback

    last_comma = val_str.rfind(',')
    last_dot = val_str.rfind('.')
    
    if last_comma > -1 and last_dot > -1:
        if last_comma > last_dot:
            # European: 1.234,56
            val_str = val_str.replace('.', '').replace(',', '.')
        else:
            # US: 1,234.56
            val_str = val_str.replace(',', '')
    elif last_comma > -1:
        # Check if comma acts as decimal or thousands separator
        # If exactly 3 digits follow the *last* comma, assume thousands (e.g., 123,456)
        # Note: 12,345,678 will also have 3 digits following the last comma.
        if len(val_str) - last_comma - 1 == 3:
            val_str = val_str.replace(',', '')
        else:
            # Otherwise assume decimal (e.g., 12,50)
            val_str = val_str.replace(',', '.')
            
    try:
        return float(val_str)
    except Exception:
        return fallback


def fmt_float(val, decimals: int = 2) -> str:
    """Format a messy value beautifully, returning 'N/A' if unparseable."""
    if val is None or str(val).lower() in ("nan", "none", "n/a", ""):
        return "N/A"
    try:
        f = safe_float(val, fallback=None)
        if f is None:
            return "N/A"
        return f"{f:,.{decimals}f}"
    except Exception:
        return "N/A"


def position_value(pos: dict) -> float:
    """Calculate absolute live position value securely in Account Base Currency.
    Extracts the Live FX factor using the differential identity:
    ppl - fxPpl = (currentPrice - averagePrice) * qty * Current_FX
    """
    qty = safe_float(pos.get("quantity"))
    cur = safe_float(pos.get("currentPrice"))
    avg = safe_float(pos.get("averagePrice"))
    ppl = safe_float(pos.get("ppl"))
    fx_ppl = safe_float(pos.get("fxPpl")) # often None
    
    if qty == 0 or cur == 0:
        return 0.0
        
    pure_diff = (cur - avg) * qty
    base_margin = ppl - fx_ppl
    
    # 1. Evaluate Live FX Conversion mathematically
    if abs(pure_diff) > 0.005:
        fx_conversion = base_margin / pure_diff
        # Guard rails for math anomalies
        if 0.0001 < fx_conversion < 5000:
            return cur * qty * fx_conversion
            
    # 2. Hard Fallback if price hasn't moved (pure_diff == 0)
    total = qty * cur
    ccy = str(pos.get("currencyCode", "") or pos.get("currency", "")).upper()
    if ccy in ("GBX",):
        return total / 100
        
    # Empirical Pence fallback (LSE stocks usually trade > 500 pence)
    if total > 50000 and cur > 1000 and abs(fx_ppl) < 0.2:
        return total / 100

    return total


import asyncio
import pandas as pd

async def fetch_historic_prices(tickers: list[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch price histories via yfinance. If any ticker 404s, fall back to alternative global exchanges."""
    import yfinance as yf
    
    unique_tickers = list(dict.fromkeys(tickers))
    def _fetch(syms):
        return yf.download(syms, period=period, interval=interval, auto_adjust=True, progress=False)
        
    data = await asyncio.to_thread(_fetch, unique_tickers)
    
    if data.empty:
        closes = pd.DataFrame()
    else:
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": unique_tickers[0]})
        
    # Drop entirely absent columns
    closes = closes.dropna(axis=1, how="all")
    missing = set(unique_tickers) - set(closes.columns)
    
    # Fallback Heuristic
    for m in missing:
        base = m.split(".")[0]
        alts = [base, f"{base}.L", f"{base}.PA", f"{base}.MI", f"{base}.AS", f"{base}.F", f"{base}.DE"]
        for alt in alts:
            if alt == m: continue
            alt_data = await asyncio.to_thread(_fetch, [alt])
            if not alt_data.empty:
                alt_closes = alt_data["Close"] if isinstance(alt_data.columns, pd.MultiIndex) else alt_data[["Close"]].rename(columns={"Close": alt})
                alt_closes = alt_closes.dropna(axis=1, how="all")
                if alt in alt_closes.columns and len(alt_closes[alt].dropna()) > 30:
                    # Successful recovery! Inject back to the dataframe
                    closes[m] = alt_closes[alt]
                    break
                    
    return closes


def fmt_billions(val) -> str:
    """Format large numbers as billions/millions."""
    import numpy as np
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    v = float(val)
    if abs(v) >= 1e9:
        return f"{v/1e9:,.1f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:,.0f}M"
    return f"{v:,.0f}"
