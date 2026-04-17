"""Macro economic dashboard and fear/greed index."""
import asyncio
from datetime import datetime, timedelta

import yfinance as yf

import app
from helpers import cached, cache_macro, safe_float

mcp = app.mcp


@cached(cache_macro)
async def _get_macro_dashboard() -> str:
    """Return key macroeconomic indicators: interest rates, yield curve, inflation,
    GDP growth, unemployment, VIX fear index, and USD strength.
    Use this for macro context before making investment decisions."""

    lines = [f"**Macro Economic Dashboard — {datetime.today().strftime('%d %b %Y')}**\n"]

    market_syms = {
        "10Y Treasury (^TNX)": "^TNX",
        "2Y Treasury (^IRX proxy)": "^TWO",
        "VIX Fear Index (^VIX)": "^VIX",
        "US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
        "Gold (GC=F)": "GC=F",
        "Crude Oil (CL=F)": "CL=F",
    }

    def _fetch_market():
        results = {}
        for label, sym in market_syms.items():
            try:
                fi = yf.Ticker(sym).fast_info
                price, prev = fi.last_price, fi.previous_close
                chg = ((price - prev) / prev * 100) if prev else 0
                results[label] = (price, chg)
            except Exception:
                results[label] = (None, None)
        return results

    try:
        market = await asyncio.to_thread(_fetch_market)
    except Exception as e:
        market = {}
        lines.append(f"⚠ Market data error: {e}\n")

    if market:
        lines.append("**Market Indicators**")
        for label, (price, chg) in market.items():
            if price is not None:
                lines.append(f"- {label}: {price:,.2f}  ({'+' if chg >= 0 else ''}{chg:.2f}%)")
            else:
                lines.append(f"- {label}: N/A")

        tnx = market.get("10Y Treasury (^TNX)", (None,))[0]
        if tnx is not None:
            lines.append(f"\n**Yield Curve**\n- 10Y yield: {tnx:.2f}%")

        vix = market.get("VIX Fear Index (^VIX)", (None,))[0]
        if vix is not None:
            if vix < 15: sig = f"VIX {vix:.1f} (below 15)"
            elif vix < 20: sig = f"VIX {vix:.1f} (15–20)"
            elif vix < 30: sig = f"VIX {vix:.1f} (20–30)"
            else: sig = f"VIX {vix:.1f} (above 30)"
            lines.append(f"\n**VIX:** {sig}")

    if app.FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=app.FRED_API_KEY)
            indicators = {
                "Fed Funds Rate": ("FEDFUNDS", False),
                "CPI YoY %": ("CPIAUCSL", True),       # index — compute YoY%
                "Core CPI YoY %": ("CPILFESL", True),   # index — compute YoY%
                "Unemployment Rate %": ("UNRATE", False),
                "Real GDP Growth % (QoQ)": ("A191RL1Q225SBEA", False),
                "Consumer Confidence": ("UMCSENT", False),
                "10Y-2Y Spread (bp)": ("T10Y2Y", False),
            }
            lines.append("\n**Economic Indicators (FRED)**")
            for label, (sid, compute_yoy) in indicators.items():
                try:
                    data = fred.get_series(sid, observation_start=(datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d"))
                    if data is not None and len(data) > 0:
                        data = data.dropna()
                        if compute_yoy and len(data) >= 13:
                            # Compute YoY% from index: (current - 12mo ago) / 12mo ago * 100
                            current_val = float(data.iloc[-1])
                            year_ago_val = float(data.iloc[-13])  # ~12 monthly observations ago
                            yoy_pct = (current_val - year_ago_val) / year_ago_val * 100
                            lines.append(f"- {label}: {yoy_pct:.2f}%")
                        else:
                            lines.append(f"- {label}: {float(data.iloc[-1]):.2f}")
                    else:
                        lines.append(f"- {label}: N/A")
                except Exception:
                    lines.append(f"- {label}: N/A")

            try:
                spread = fred.get_series("T10Y2Y")
                if spread is not None and len(spread) > 0:
                    ls = float(spread.dropna().iloc[-1])
                    lines.append(f"- 10Y-2Y Spread: {ls:.2f}%")
            except Exception:
                pass
        except ImportError:
            lines.append("\n⚠ Install fredapi: `pip install fredapi`")
        except Exception as e:
            lines.append(f"\n⚠ FRED error: {e}")
    else:
        lines.append("\n💡 Set FRED_API_KEY in .env for economic indicators.")

    return "\n".join(lines)


@cached(cache_macro)
async def _get_fear_greed_index() -> str:
    """Return a composite Fear & Greed score (0-100) based on VIX, momentum,
    market breadth, and safe haven demand. 0=Extreme Fear, 100=Extreme Greed."""

    def _fetch():
        d = {}
        try: d["vix"] = safe_float(yf.Ticker("^VIX").fast_info.last_price, fallback=None)
        except Exception: d["vix"] = None

        try:
            spy = yf.Ticker("SPY").history(period="6mo", interval="1d", auto_adjust=True)["Close"]
            d["spy_price"] = safe_float(spy.iloc[-1], fallback=None)
            d["spy_ma125"] = safe_float(spy.rolling(125).mean().iloc[-1], fallback=None)
        except Exception:
            d["spy_price"] = d["spy_ma125"] = None

        breadth_syms = ["XLK","XLF","XLV","XLE","XLI","XLB","XLRE","XLU","XLP","XLY","XLC"]
        above, total = 0, 0
        for sym in breadth_syms:
            try:
                h = yf.Ticker(sym).history(period="3mo", interval="1d", auto_adjust=True)["Close"]
                if len(h) >= 50:
                    total += 1
                    if safe_float(h.iloc[-1]) > safe_float(h.rolling(50).mean().iloc[-1]):
                        above += 1
            except Exception: pass
        d["breadth_pct"] = (above / total * 100) if total > 0 else None

        try:
            gold = yf.Ticker("GLD").history(period="1mo", interval="1d", auto_adjust=True)["Close"]
            spy_h = yf.Ticker("SPY").history(period="1mo", interval="1d", auto_adjust=True)["Close"]
            
            g0, g1 = safe_float(gold.iloc[0]), safe_float(gold.iloc[-1])
            s0, s1 = safe_float(spy_h.iloc[0]), safe_float(spy_h.iloc[-1])
            
            if g0 != 0 and s0 != 0:
                d["safe_haven_diff"] = ((g1 - g0) / g0 * 100) - ((s1 - s0) / s0 * 100)
            else:
                d["safe_haven_diff"] = None
        except Exception:
            d["safe_haven_diff"] = None
        return d

    try:
        d = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error computing Fear & Greed Index: {e}"

    scores, components = [], []

    if d["vix"] is not None:
        vix = d["vix"]
        s = max(0, min(100, 100 - (vix - 12) * (100 / 28)))
        scores.append(s)
        components.append(f"- VIX ({vix:.1f}): score {s:.0f}")

    if d["spy_price"] is not None and d["spy_ma125"] is not None:
        pct = (d["spy_price"] / d["spy_ma125"] - 1) * 100
        s = max(0, min(100, 50 + pct * 5))
        scores.append(s)
        components.append(f"- Momentum (SPY {pct:+.1f}% vs MA125): score {s:.0f}")

    if d["breadth_pct"] is not None:
        scores.append(d["breadth_pct"])
        components.append(f"- Breadth ({d['breadth_pct']:.0f}% of tracked stocks above MA50): score {d['breadth_pct']:.0f}")

    if d["safe_haven_diff"] is not None:
        sh = d["safe_haven_diff"]
        s = max(0, min(100, 50 - sh * 5))
        scores.append(s)
        components.append(f"- Safe Haven (Gold vs SPY 1M: {sh:+.1f}%): score {s:.0f}")

    if not scores:
        return "Insufficient data."

    composite = sum(scores) / len(scores)

    return "\n".join([
        f"**Fear & Greed Index — {datetime.today().strftime('%d %b %Y')}**\n",
        f"## Composite Score: {composite:.0f}/100\n",
        "**Components:**",
    ] + components)


# ===========================================================================
# CONSOLIDATED MACRO BUNDLE
# ===========================================================================

@mcp.tool()
async def get_macro_summary(include: str = "snapshot,macro,fear_greed,sectors") -> str:
    """Return a composite macro view in one call — market indices, economic indicators,
    fear/greed score, and sector rotation.
    include: comma-separated subset of 'snapshot', 'macro', 'fear_greed', 'sectors'"""

    from tools.market_data import _get_market_snapshot
    from tools.analysis import _get_sector_rotation

    parts = [p.strip().lower() for p in include.split(",")]

    # Build tasks for the requested sections (all run concurrently)
    task_keys, task_coros = [], []
    if "snapshot" in parts:
        task_keys.append("snapshot")
        task_coros.append(_get_market_snapshot())
    if "macro" in parts:
        task_keys.append("macro")
        task_coros.append(_get_macro_dashboard())
    if "fear_greed" in parts:
        task_keys.append("fear_greed")
        task_coros.append(_get_fear_greed_index())
    if "sectors" in parts:
        task_keys.append("sectors")
        task_coros.append(_get_sector_rotation())

    if not task_coros:
        return "No valid sections specified. Choose from: snapshot, macro, fear_greed, sectors"

    results = await asyncio.gather(*task_coros, return_exceptions=True)

    ordered = ["snapshot", "macro", "fear_greed", "sectors"]
    section_map = {k: r for k, r in zip(task_keys, results)}

    out_parts = []
    for key in ordered:
        if key not in section_map:
            continue
        val = section_map[key]
        out_parts.append(str(val) if isinstance(val, Exception) else val)

    return "\n\n---\n\n".join(out_parts)
