"""Macro economic dashboard and fear/greed index."""
import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
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
            lines.append(f"\n**VIX:** {vix:.2f}")

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
    """Market sentiment components expressed as **historical percentiles over
    a 5-year rolling window**. No magic coefficients. Reports raw values + rank.

    Components (each independently rank-transformed, 0 = extreme fear, 100 = extreme greed):
      1. VIX level (inverted — high VIX → low percentile)
      2. SPY price distance from its 125-day SMA (%)
      3. Breadth: % of 11 SPDR sector ETFs above their 50-day SMA (snapshot)
      4. 20-day realised SPY vol (inverted — high vol → low percentile)
      5. Safe-haven demand: 1-month GLD return − 1-month SPY return (inverted)
      6. Put/call proxy: 10-day return of TLT vs SPY (inverted — bonds outperform = fear)

    The composite is a simple unweighted average of available component percentiles.
    """
    import quant

    def _fetch_hist(sym, period="5y"):
        try:
            return yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)["Close"]
        except Exception:
            return pd.Series(dtype=float)

    def _fetch():
        vix = _fetch_hist("^VIX", period="5y")
        spy = _fetch_hist("SPY", period="5y")
        gld = _fetch_hist("GLD", period="5y")
        tlt = _fetch_hist("TLT", period="5y")
        sector_closes = {}
        for sym in ["XLK","XLF","XLV","XLE","XLI","XLB","XLRE","XLU","XLP","XLY","XLC"]:
            s = _fetch_hist(sym, period="1y")
            if not s.empty:
                sector_closes[sym] = s
        return vix, spy, gld, tlt, sector_closes

    try:
        vix, spy, gld, tlt, sector_closes = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error computing Fear & Greed: {e}"

    components = []
    percentiles = []

    # 1. VIX level — inverted (low VIX = greed)
    if not vix.empty and len(vix) > 30:
        last_vix = float(vix.iloc[-1])
        raw_pct = quant.historical_percentile(last_vix, vix)
        greed_pct = 100 - raw_pct
        percentiles.append(greed_pct)
        components.append(
            f"- VIX:                {last_vix:>6.2f}   raw pctile {raw_pct:>5.1f}   greed score {greed_pct:>5.1f}"
        )

    # 2. SPY distance from 125d SMA (%)
    if not spy.empty and len(spy) > 150:
        ma = spy.rolling(125).mean()
        dist = (spy / ma - 1) * 100
        last = float(dist.iloc[-1])
        pct = quant.historical_percentile(last, dist.dropna())
        percentiles.append(pct)
        components.append(
            f"- SPY vs 125d SMA:    {last:>+6.2f}%  pctile {pct:>5.1f}   greed score {pct:>5.1f}"
        )

    # 3. Breadth — % of sectors above MA50 (snapshot, not historical percentile)
    if sector_closes:
        df = pd.DataFrame(sector_closes)
        above50 = quant.pct_above_ma(df, 50)
        if not np.isnan(above50):
            score = above50 * 100
            percentiles.append(score)
            components.append(
                f"- Breadth (sec>MA50): {score:>6.1f}%  (greed score = breadth%)"
            )

    # 4. 20-day realised SPY vol — inverted
    if not spy.empty and len(spy) > 100:
        spy_ret = spy.pct_change()
        rv = spy_ret.rolling(20).std(ddof=1) * np.sqrt(252) * 100
        last_rv = float(rv.iloc[-1])
        raw_pct = quant.historical_percentile(last_rv, rv.dropna())
        greed_pct = 100 - raw_pct
        percentiles.append(greed_pct)
        components.append(
            f"- 20d realised vol:   {last_rv:>6.2f}%  raw pctile {raw_pct:>5.1f}   greed score {greed_pct:>5.1f}"
        )

    # 5. Safe-haven demand — GLD 1M return minus SPY 1M return, inverted
    if not gld.empty and not spy.empty and len(gld) > 250 and len(spy) > 250:
        def _roll_1m(s):
            return s.pct_change(21)
        diff = (_roll_1m(gld) - _roll_1m(spy)).dropna() * 100
        if len(diff) > 30:
            last = float(diff.iloc[-1])
            raw_pct = quant.historical_percentile(last, diff)
            greed_pct = 100 - raw_pct
            percentiles.append(greed_pct)
            components.append(
                f"- GLD-SPY 1M spread: {last:>+6.2f}%  raw pctile {raw_pct:>5.1f}   greed score {greed_pct:>5.1f}"
            )

    # 6. TLT vs SPY 10d return — bonds outperforming = risk-off
    if not tlt.empty and not spy.empty and len(tlt) > 250 and len(spy) > 250:
        diff10 = (tlt.pct_change(10) - spy.pct_change(10)).dropna() * 100
        if len(diff10) > 30:
            last = float(diff10.iloc[-1])
            raw_pct = quant.historical_percentile(last, diff10)
            greed_pct = 100 - raw_pct
            percentiles.append(greed_pct)
            components.append(
                f"- TLT-SPY 10d spread:{last:>+6.2f}%  raw pctile {raw_pct:>5.1f}   greed score {greed_pct:>5.1f}"
            )

    if not percentiles:
        return "Insufficient data for Fear & Greed components."

    composite = sum(percentiles) / len(percentiles)

    return "\n".join([
        f"**Sentiment Components — {datetime.today().strftime('%d %b %Y')}**",
        f"_All percentiles computed vs a rolling 5-year history per component._",
        f"_Greed score = 0 (extreme fear) → 100 (extreme greed)._\n",
        f"Composite (unweighted mean of components): {composite:.1f}/100",
        "",
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
