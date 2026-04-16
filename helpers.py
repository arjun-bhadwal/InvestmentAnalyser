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
    """
    parts = raw.split("_")
    symbol = parts[0]

    # Look for an exchange code in the middle segments (skip first=symbol, last=EQ)
    if len(parts) >= 3:
        exchange = parts[-2]  # e.g. 'L', 'US', 'DE'
        suffix = _T212_EXCHANGE_MAP.get(exchange, "")
        return f"{symbol}{suffix}"

    # Fallback: 2 parts like 'AAPL_EQ' — assume US
    if len(parts) == 2 and parts[1] == "EQ":
        return symbol

    # Raw ticker with no underscore — return as-is
    return raw


def fmt_float(value, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def position_value(pos: dict) -> float:
    """Calculate position value, converting GBX (pence) → GBP when needed."""
    qty = float(pos.get("quantity", 0) or 0)
    price = float(pos.get("currentPrice", 0) or 0)
    ccy = (pos.get("currencyCode") or pos.get("currency") or "").upper()
    if ccy in ("GBX", "GBP"):  # GBX = pence, divide by 100
        if ccy == "GBX":
            price = price / 100
    return qty * price


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
