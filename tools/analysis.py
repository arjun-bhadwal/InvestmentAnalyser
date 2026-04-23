"""Technical indicators, stock screener, sector rotation (RRG), earnings calendar.

Data-only outputs. Indicator labels are factual (values + thresholds crossed);
no buy/sell or bullish/bearish language.
"""
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import ta as ta_lib
import yfinance as yf

import app
import quant
from helpers import cached, cache_prices, cache_fundamentals, strip_t212_ticker, fmt_float, safe_float

mcp = app.mcp


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------

async def _get_technical_indicators_core(ticker: str) -> str:
    """Return key technical indicators for a stock: RSI, MACD, Bollinger Bands, moving averages,
    ATR, and a plain-English signal summary."""

    from resolver import fetch_history
    rt, raw_df = await fetch_history(ticker, period="6mo", interval="1d")

    def _prep(df):
        if df is None or df.empty:
            return None
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        df["ma20"]  = ta_lib.trend.sma_indicator(close, window=20)
        df["ma50"]  = ta_lib.trend.sma_indicator(close, window=50)
        df["ma200"] = ta_lib.trend.sma_indicator(close, window=200)
        df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()

        macd_obj = ta_lib.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        df["macd"]      = macd_obj.macd()
        df["macd_sig"]  = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()

        bb_obj = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb_obj.bollinger_hband()
        df["bb_mid"]   = bb_obj.bollinger_mavg()
        df["bb_lower"] = bb_obj.bollinger_lband()

        df["atr"] = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        return df

    try:
        df = await asyncio.to_thread(_prep, raw_df)
    except Exception as e:
        return f"Error computing indicators for {ticker}: {e}"

    if df is None or df.empty:
        return f"No data found for '{ticker}' (resolved to '{rt.yf_symbol}')."

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row
    price = row["close"]

    def _g(col):
        v = row.get(col)
        sf = safe_float(v, fallback=None)
        return sf

    rsi    = _g("rsi")
    ma20   = _g("ma20")
    ma50   = _g("ma50")
    ma200  = _g("ma200")
    atr    = _g("atr")
    macd_val  = _g("macd")
    macd_sig  = _g("macd_sig")
    macd_hist = _g("macd_hist")
    bb_upper = _g("bb_upper")
    bb_mid   = _g("bb_mid")
    bb_lower = _g("bb_lower")

    def _fmt(v, d=2):
        return f"{v:,.{d}f}" if v is not None else "N/A"

    # Compute historical percentile rank of current RSI over the 6-month window
    rsi_hist = df["rsi"].dropna() if "rsi" in df.columns else pd.Series(dtype=float)
    rsi_pct = quant.historical_percentile(rsi, rsi_hist) if rsi is not None else None

    signals = []
    if rsi is not None:
        extra = f" (pctile {rsi_pct:.0f} over lookback)" if rsi_pct is not None else ""
        signals.append(f"RSI: {rsi:.1f}{extra}")

    if ma20 and ma50:
        signals.append(f"MA20 {'>' if ma20 > ma50 else '<'} MA50")
    if ma50 and ma200:
        signals.append(f"MA50 {'>' if ma50 > ma200 else '<'} MA200")

    if macd_val is not None and macd_sig is not None and macd_hist is not None:
        prev_hist = float(prev.get("macd_hist") or 0)
        if macd_hist > 0 and prev_hist <= 0:
            signals.append("MACD histogram crossed above zero")
        elif macd_hist < 0 and prev_hist >= 0:
            signals.append("MACD histogram crossed below zero")
        else:
            signals.append(f"MACD histogram sign: {'+' if macd_hist > 0 else '-'}")

    if bb_upper and bb_lower and bb_mid:
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        signals.append(f"%B (position in BB range): {bb_pct:.0f}%")

    lines = [
        f"**{rt.yf_symbol} — Technical Indicators** ({rt.currency})\n",
        f"- Price:         {_fmt(price)}",
        "",
        f"**Moving Averages**",
        f"- MA20:          {_fmt(ma20)}   {'▲ above' if ma20 and price > ma20 else '▼ below' if ma20 else ''}",
        f"- MA50:          {_fmt(ma50)}   {'▲ above' if ma50 and price > ma50 else '▼ below' if ma50 else ''}",
        f"- MA200:         {_fmt(ma200)}   {'▲ above' if ma200 and price > ma200 else '▼ below' if ma200 else ''}",
        "",
        f"**Momentum**",
        f"- RSI (14):      {_fmt(rsi, 1)}",
        f"- MACD:          {_fmt(macd_val)}  Signal: {_fmt(macd_sig)}  Hist: {_fmt(macd_hist)}",
        "",
        f"**Volatility**",
        f"- ATR (14):      {_fmt(atr)}  ({_fmt(atr/price*100, 1) if atr and price else 'N/A'}% of price)",
        f"- BB Upper:      {_fmt(bb_upper)}",
        f"- BB Mid:        {_fmt(bb_mid)}",
        f"- BB Lower:      {_fmt(bb_lower)}",
        "",
        f"**Signal Summary**",
    ]
    for s in signals:
        lines.append(f"  • {s}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sector Rotation
# ---------------------------------------------------------------------------

SECTOR_ETFS = {
    "Technology":        "XLK",
    "Financials":        "XLF",
    "Healthcare":        "XLV",
    "Energy":            "XLE",
    "Industrials":       "XLI",
    "Materials":         "XLB",
    "Real Estate":       "XLRE",
    "Utilities":         "XLU",
    "Consumer Staples":  "XLP",
    "Consumer Discret.": "XLY",
    "Communication":     "XLC",
}


@cached(cache_prices)
async def _get_sector_rotation() -> str:
    """Sector rotation — period returns and RRG (Relative Rotation Graph) reading
    for each SPDR sector ETF vs SPY.

    Methodology (Julius de Kempenaer):
      JdK RS-Ratio_t  = 100 + z-score of (sector/benchmark) over 63-day window
      JdK RS-Momentum = 100 + (RS-Ratio - SMA(RS-Ratio, 21))
    Quadrants (factual, no editorial spin):
      Leading    (RS≥100, Mom≥100)
      Weakening  (RS≥100, Mom<100)
      Lagging    (RS<100, Mom<100)
      Improving  (RS<100, Mom≥100)
    Also reports market breadth: % of sectors above their 50/200-day SMA.
    """
    symbols = list(SECTOR_ETFS.values()) + ["SPY"]

    def _fetch():
        return yf.download(symbols, period="1y", interval="1d", auto_adjust=True, progress=False)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching sector data: {e}"

    closes = data["Close"].dropna(how="all")
    if closes.empty or "SPY" not in closes.columns:
        return "Could not retrieve sector data."

    def _ret(col, days):
        c = col.dropna()
        if len(c) < 2:
            return None
        idx = min(days, len(c) - 1)
        return (float(c.iloc[-1]) / float(c.iloc[-(idx + 1)]) - 1) * 100

    periods = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    spy_rets = {lbl: _ret(closes["SPY"], d) for lbl, d in periods.items()}

    rows = []
    for name, sym in SECTOR_ETFS.items():
        if sym not in closes.columns:
            continue
        s = closes[sym]
        rets = {lbl: _ret(s, d) for lbl, d in periods.items()}
        rrg = quant.relative_strength_snapshot(s, closes["SPY"], rs_window=63, mom_window=21)
        rows.append((name, sym, rets, rrg))

    # Sort by quadrant then RS-Ratio descending
    quadrant_order = {"leading": 0, "weakening": 1, "improving": 2, "lagging": 3, "n/a": 4}
    rows.sort(key=lambda x: (quadrant_order.get(x[3]["quadrant"], 9),
                             -(x[3]["rs_ratio"] if not np.isnan(x[3]["rs_ratio"]) else -1e9)))

    def _fs(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A"
        return f"{'+' if v >= 0 else ''}{v:.1f}%"

    def _fn(v, d=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.{d}f}"

    lines = [
        f"**Sector Rotation — {datetime.today().strftime('%d %b %Y')}**",
        f"_Returns (vs SPY) + JdK RS-Ratio (63d z-score) & RS-Momentum (21d). Parity = 100._\n",
        f"{'Sector':<22} {'ETF':<6} {'1D':>7} {'1W':>7} {'1M':>7} {'3M':>7} {'1Y':>7} "
        f"{'RS-Rat':>7} {'RS-Mom':>7} {'Quadrant':<10}",
        "-" * 98,
    ]

    for name, sym, rets, rrg in rows:
        lines.append(
            f"{name:<22} {sym:<6} {_fs(rets['1D']):>7} {_fs(rets['1W']):>7} "
            f"{_fs(rets['1M']):>7} {_fs(rets['3M']):>7} {_fs(rets['1Y']):>7} "
            f"{_fn(rrg['rs_ratio'],2):>7} {_fn(rrg['rs_momentum'],2):>7} {rrg['quadrant']:<10}"
        )

    lines.append("-" * 98)
    lines.append(
        f"{'S&P 500 (SPY)':<22} {'SPY':<6} {_fs(spy_rets['1D']):>7} {_fs(spy_rets['1W']):>7} "
        f"{_fs(spy_rets['1M']):>7} {_fs(spy_rets['3M']):>7} {_fs(spy_rets['1Y']):>7} "
        f"{'100.00':>7} {'100.00':>7} {'benchmark':<10}"
    )

    # Market breadth on the 11-sector universe
    sector_closes = closes[[s for s in SECTOR_ETFS.values() if s in closes.columns]]
    above50 = quant.pct_above_ma(sector_closes, 50)
    above200 = quant.pct_above_ma(sector_closes, 200)
    nh = quant.new_highs_lows(sector_closes, window=252)

    lines.append("")
    lines.append("**Breadth (11 SPDR sectors)**")
    if not np.isnan(above50):
        lines.append(f"- % above 50-day SMA:  {above50*100:.0f}%")
    if not np.isnan(above200):
        lines.append(f"- % above 200-day SMA: {above200*100:.0f}%")
    if nh["universe_size"] > 0:
        lines.append(f"- 52w highs / lows:    {nh['new_highs']} / {nh['new_lows']} (of {nh['universe_size']})")

    # Quadrant counts
    from collections import Counter
    counts = Counter(r[3]["quadrant"] for r in rows)
    lines.append("")
    lines.append("**RRG Quadrant distribution**")
    for q in ("leading", "weakening", "lagging", "improving"):
        lines.append(f"- {q.capitalize():<10} {counts.get(q, 0)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Earnings Calendar
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_earnings_calendar() -> str:
    """Return upcoming earnings dates and recent EPS results for all stocks in your portfolio.
    Use this to manage event risk."""

    try:
        positions = await app.t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions."

    tickers = [strip_t212_ticker(p["ticker"]) for p in positions]

    async def _fetch_earnings(ticker):
        def _f():
            t = yf.Ticker(ticker)
            cal = t.calendar
            info = t.info
            return cal, info
        try:
            cal, info = await asyncio.wait_for(asyncio.to_thread(_f), timeout=8.0)
            return ticker, cal, info
        except Exception:
            return ticker, None, {}

    results = await asyncio.gather(*[_fetch_earnings(t) for t in tickers])

    # Filter out ETFs, ETCs, and funds — they have no earnings data
    equity_results = [
        (ticker, cal, info) for ticker, cal, info in results
        if info.get("quoteType", "EQUITY") in ("EQUITY", "")
    ]

    if not equity_results:
        return "**Earnings Calendar — Portfolio**\n\nNo equity positions with earnings data found."

    lines = [
        f"**Earnings Calendar — Portfolio**\n",
        f"{'Ticker':<10} {'Next Earnings':<20} {'EPS (TTM)':>12} {'EPS (Fwd)':>12} {'Fwd vs TTM':>12}",
        "-" * 70,
    ]

    def _sort_key(row):
        _, cal, _ = row
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                return str(dates[0])[:10]
        return "9999"

    for ticker, cal, info in sorted(equity_results, key=_sort_key):
        eps_ttm = info.get("trailingEps")
        eps_fwd = info.get("forwardEps")
        eps_ttm_str = f"{float(eps_ttm):,.2f}" if eps_ttm is not None else "N/A"
        eps_fwd_str = f"{float(eps_fwd):,.2f}" if eps_fwd is not None else "N/A"

        next_date = "N/A"
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                next_date = str(dates[0])[:10] if hasattr(dates[0], "strftime") else str(dates[0])[:10]

        fwd_vs_ttm = ""
        if eps_ttm is not None and eps_fwd is not None and float(eps_ttm) != 0:
            diff = (float(eps_fwd) - float(eps_ttm)) / abs(float(eps_ttm)) * 100
            fwd_vs_ttm = f"{diff:+.1f}%"

        lines.append(f"{ticker:<10} {next_date:<20} {eps_ttm_str:>12} {eps_fwd_str:>12} {fwd_vs_ttm:>12}")

    skipped = [ticker for ticker, _, info in results if info.get("quoteType", "EQUITY") not in ("EQUITY", "")]
    if skipped:
        lines.append(f"\n_Skipped (ETF/ETC/Fund): {', '.join(skipped)}_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stock Screener
# ---------------------------------------------------------------------------

async def _screen_stocks_core(
    universe: str = "watchlist",
    max_rsi: float = 40.0,
    min_rsi: float = 0.0,
    require_above_ma200: bool = True,
    require_above_ma50: bool = False,
    min_analyst_upside_pct: float = 10.0,
    max_pe: float = 0.0,
    min_momentum_1m_pct: float = 0.0,
    max_results: int = 20,
) -> str:
    """Scan a set of tickers for entry setups using technical and fundamental filters.

    universe: "watchlist" (your T212 holdings) | comma-separated tickers
              Use _search_web to discover candidates first, then pass them here.
              Example: "CCJ,UEC,DNN.TO,SPUT.TO" or "SGLN.L,PHAU.L,GLD,IAU"
    max_rsi: only include tickers with RSI below this value
    min_rsi: only include tickers with RSI above this value
    require_above_ma200: only include tickers trading above their 200-day MA
    require_above_ma50: only include tickers trading above their 50-day MA
    min_analyst_upside_pct: minimum analyst mean target upside %
    max_pe: maximum trailing P/E ratio — 0 = no filter
    min_momentum_1m_pct: minimum 1-month return % — 0 = no filter
    max_results: cap on results"""

    if universe == "watchlist":
        try:
            positions = await app.t212.get_portfolio()
            tickers = [strip_t212_ticker(p["ticker"]) for p in positions]
        except Exception:
            tickers = []
    else:
        tickers = [t.strip().upper() for t in universe.split(",") if t.strip()]

    if not tickers:
        return "No tickers to screen."

    async def _batch_parallel(syms):
        """Download in sub-batches if the main download yields nothing."""
        # Try full batch first
        try:
            df = await asyncio.to_thread(yf.download, syms, period="1y", interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    return df["Close"]
                return df[["Close"]].rename(columns={"Close": syms[0]})
        except Exception:
            pass

        # If failed, try smaller chunks of 10
        chunk_size = 10
        chunks = [syms[i:i + chunk_size] for i in range(0, len(syms), chunk_size)]
        
        async def _get_chunk(c):
            try:
                d = await asyncio.to_thread(yf.download, c, period="1y", interval="1d", auto_adjust=True, progress=False)
                if d.empty: return pd.DataFrame()
                if isinstance(d.columns, pd.MultiIndex):
                    return d["Close"]
                return d[["Close"]].rename(columns={"Close": c[0]})
            except Exception:
                return pd.DataFrame()
        
        results = await asyncio.gather(*[_get_chunk(c) for c in chunks])
        valid_dfs = [r for r in results if not r.empty]
        if not valid_dfs:
            return pd.DataFrame()
        return pd.concat(valid_dfs, axis=1)

    try:
        closes = await _batch_parallel(tickers)
    except Exception as e:
        return f"Error downloading price data: {e}"

    if closes.empty:
        return "Insufficient price data found for this universe."
    
    closes = closes.dropna(how="all")
    available = [c for c in closes.columns if closes[c].notna().sum() >= 60]
    if not available:
        return "Insufficient price data (too many NaNs in history)."

    candidates = []
    for sym in available:
        col = closes[sym].dropna()
        if len(col) < 60:
            continue
        price = safe_float(col.iloc[-1])
        ma50  = safe_float(col.rolling(50).mean().iloc[-1], fallback=None) if len(col) >= 50 else None
        ma200 = safe_float(col.rolling(200).mean().iloc[-1], fallback=None) if len(col) >= 200 else None

        delta = col.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        
        last_gain = safe_float(gain.iloc[-1])
        last_loss = safe_float(loss.iloc[-1])
        rs    = last_gain / last_loss if last_loss != 0 else 0
        rsi   = 100 - (100 / (1 + rs)) if last_loss != 0 else 100

        mom_1m = (price - safe_float(col.iloc[-22])) / safe_float(col.iloc[-22]) * 100 if len(col) >= 22 and safe_float(col.iloc[-22]) != 0 else None

        if rsi < min_rsi or rsi > max_rsi:
            continue
        if require_above_ma200 and (ma200 is None or price < ma200):
            continue
        if require_above_ma50 and (ma50 is None or price < ma50):
            continue
        if min_momentum_1m_pct and (mom_1m is None or mom_1m < min_momentum_1m_pct):
            continue

        candidates.append({"ticker": sym, "price": price, "rsi": rsi, "ma50": ma50, "ma200": ma200, "mom_1m": mom_1m})

    if not candidates:
        return f"No stocks passed technical filters in '{universe}' universe. Try relaxing criteria."

    async def _fund(sym):
        def _f():
            info = yf.Ticker(sym).info
            return {
                "pe": info.get("trailingPE"), "target": info.get("targetMeanPrice"),
                "cur_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "rec": info.get("recommendationKey", ""), "name": info.get("shortName") or sym,
                "sector": info.get("sector", ""),
            }
        try:
            return await asyncio.to_thread(_f)
        except Exception:
            return {}

    fund_results = await asyncio.gather(*[_fund(c["ticker"]) for c in candidates])

    final = []
    for c, f in zip(candidates, fund_results):
        pe = f.get("pe")
        target = f.get("target")
        cur = f.get("cur_price") or c["price"]
        upside = (float(target) - float(cur)) / float(cur) * 100 if target and cur else None
        if max_pe and pe and pe > max_pe:
            continue
        if min_analyst_upside_pct and (upside is None or upside < min_analyst_upside_pct):
            continue
        final.append({**c, **f, "upside": upside})

    if not final:
        return f"Technical candidates ({len(candidates)}) but none passed fundamental filters."

    final.sort(key=lambda x: x["rsi"])
    final = final[:max_results]

    lines = [
        f"**Stock Screener — {universe.upper()}**\n",
        f"{'Ticker':<8} {'Name':<24} {'Price':>9} {'RSI':>6} {'1M%':>7} {'Upside%':>9} {'P/E':>7} {'Rec':<10} {'Sector'}",
        "-" * 100,
    ]
    for s in final:
        pe_str  = f"{s['pe']:.1f}" if s.get("pe") else "N/A"
        up_str  = f"+{s['upside']:.1f}%" if s.get("upside") else "N/A"
        mom_str = f"{s['mom_1m']:+.1f}%" if s.get("mom_1m") is not None else "N/A"
        lines.append(
            f"{s['ticker']:<8} {(s.get('name',''))[:23]:<24} {s['price']:>9,.2f} "
            f"{s['rsi']:>6.1f} {mom_str:>7} {up_str:>9} {pe_str:>7} {(s.get('rec','') or '').upper()[:9]:<10} {(s.get('sector','') or '')[:18]}"
        )

    lines.append(f"\n{len(final)} candidates — sorted by RSI ascending")
    return "\n".join(lines)
