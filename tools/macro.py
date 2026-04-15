"""Macro economic dashboard and fear/greed index."""
import asyncio
from datetime import datetime, timedelta

import yfinance as yf

import app
from helpers import cached, cache_macro

mcp = app.mcp


@mcp.tool()
@cached(cache_macro)
async def get_macro_dashboard() -> str:
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
            if vix < 15: sig = "LOW FEAR — complacency, potential for surprise vol"
            elif vix < 20: sig = "NORMAL — market calm"
            elif vix < 30: sig = "ELEVATED — uncertainty, caution warranted"
            else: sig = "HIGH FEAR — panic selling, contrarian buy zone"
            lines.append(f"\n**VIX Signal:** {sig}")

    if app.FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=app.FRED_API_KEY)
            indicators = {
                "Fed Funds Rate": "FEDFUNDS", "CPI YoY %": "CPIAUCSL",
                "Core CPI YoY %": "CPILFESL", "Unemployment Rate %": "UNRATE",
                "Real GDP Growth % (QoQ)": "A191RL1Q225SBEA",
                "Consumer Confidence": "UMCSENT", "10Y-2Y Spread (bp)": "T10Y2Y",
            }
            lines.append("\n**Economic Indicators (FRED)**")
            for label, sid in indicators.items():
                try:
                    data = fred.get_series(sid, observation_start=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"))
                    if data is not None and len(data) > 0:
                        lines.append(f"- {label}: {float(data.dropna().iloc[-1]):.2f}")
                except Exception:
                    lines.append(f"- {label}: N/A")

            try:
                spread = fred.get_series("T10Y2Y")
                if spread is not None and len(spread) > 0:
                    ls = float(spread.dropna().iloc[-1])
                    if ls < 0:
                        lines.append(f"\n> ⚠️ **YIELD CURVE INVERTED** ({ls:.2f}%) — recession signal")
                    elif ls < 0.5:
                        lines.append(f"\n> ⚡ Yield curve flattening ({ls:.2f}%)")
            except Exception:
                pass
        except ImportError:
            lines.append("\n⚠ Install fredapi: `pip install fredapi`")
        except Exception as e:
            lines.append(f"\n⚠ FRED error: {e}")
    else:
        lines.append("\n💡 Set FRED_API_KEY in .env for economic indicators.")

    return "\n".join(lines)


@mcp.tool()
@cached(cache_macro)
async def get_fear_greed_index() -> str:
    """Return a composite Fear & Greed score (0-100) based on VIX, momentum,
    market breadth, and safe haven demand. 0=Extreme Fear, 100=Extreme Greed."""

    def _fetch():
        d = {}
        try: d["vix"] = float(yf.Ticker("^VIX").fast_info.last_price)
        except Exception: d["vix"] = None

        try:
            spy = yf.Ticker("SPY").history(period="6mo", interval="1d", auto_adjust=True)["Close"]
            d["spy_price"] = float(spy.iloc[-1])
            d["spy_ma125"] = float(spy.rolling(125).mean().iloc[-1])
        except Exception:
            d["spy_price"] = d["spy_ma125"] = None

        breadth_syms = ["XLK","XLF","XLV","XLE","XLI","XLB","XLRE","XLU","XLP","XLY","XLC"]
        above, total = 0, 0
        for sym in breadth_syms:
            try:
                h = yf.Ticker(sym).history(period="3mo", interval="1d", auto_adjust=True)["Close"]
                if len(h) >= 50:
                    total += 1
                    if float(h.iloc[-1]) > float(h.rolling(50).mean().iloc[-1]):
                        above += 1
            except Exception: pass
        d["breadth_pct"] = (above / total * 100) if total > 0 else None

        try:
            gold = yf.Ticker("GLD").history(period="1mo", interval="1d", auto_adjust=True)["Close"]
            spy_h = yf.Ticker("SPY").history(period="1mo", interval="1d", auto_adjust=True)["Close"]
            d["safe_haven_diff"] = ((float(gold.iloc[-1]) - float(gold.iloc[0])) / float(gold.iloc[0]) * 100 -
                                    (float(spy_h.iloc[-1]) - float(spy_h.iloc[0])) / float(spy_h.iloc[0]) * 100)
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
        lbl = "Extreme Fear" if vix > 30 else "Fear" if vix > 20 else "Neutral" if vix > 15 else "Greed" if vix > 12 else "Extreme Greed"
        components.append(f"- VIX ({vix:.1f}): **{lbl}** → score {s:.0f}")

    if d["spy_price"] is not None and d["spy_ma125"] is not None:
        pct = (d["spy_price"] / d["spy_ma125"] - 1) * 100
        s = max(0, min(100, 50 + pct * 5))
        scores.append(s)
        components.append(f"- Momentum (SPY {pct:+.1f}% vs MA125): **{'Greedy' if pct > 5 else 'Fearful' if pct < -5 else 'Neutral'}** → score {s:.0f}")

    if d["breadth_pct"] is not None:
        scores.append(d["breadth_pct"])
        components.append(f"- Breadth ({d['breadth_pct']:.0f}% above MA50): **{'Strong' if d['breadth_pct'] > 70 else 'Weak' if d['breadth_pct'] < 30 else 'Mixed'}** → score {d['breadth_pct']:.0f}")

    if d["safe_haven_diff"] is not None:
        sh = d["safe_haven_diff"]
        s = max(0, min(100, 50 - sh * 5))
        scores.append(s)
        components.append(f"- Safe Haven (Gold vs SPY 1M: {sh:+.1f}%): **{'Fear' if sh > 2 else 'Greed' if sh < -2 else 'Neutral'}** → score {s:.0f}")

    if not scores:
        return "Insufficient data."

    composite = sum(scores) / len(scores)
    if composite <= 20: reading = "🔴 EXTREME FEAR"
    elif composite <= 40: reading = "🟠 FEAR"
    elif composite <= 60: reading = "🟡 NEUTRAL"
    elif composite <= 80: reading = "🟢 GREED"
    else: reading = "🟢 EXTREME GREED"

    return "\n".join([
        f"**Fear & Greed Index — {datetime.today().strftime('%d %b %Y')}**\n",
        f"## {reading} — Score: {composite:.0f}/100\n",
        "**Components:**",
    ] + components + [
        "", "**Interpretation:**",
        "- 0-25: Extreme Fear — contrarian buy zone",
        "- 25-45: Fear — cautious",
        "- 45-55: Neutral",
        "- 55-75: Greed — momentum favours bulls",
        "- 75-100: Extreme Greed — caution, potential top",
    ])
