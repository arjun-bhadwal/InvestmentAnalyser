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

def strip_t212_ticker(raw: str) -> str:
    """Convert T212 instrument codes like 'AAPL_US_EQ' → 'AAPL'."""
    return raw.split("_")[0]


def fmt_float(value, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


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
