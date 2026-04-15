"""Market data tools: prices, fundamentals, analyst ratings, market snapshot.
Includes Polygon.io integration for real-time quotes and market status."""
import asyncio
from datetime import datetime

import numpy as np
import yfinance as yf

import app
from helpers import cached, cache_fundamentals, cache_prices, fmt_float

mcp = app.mcp


# ---------------------------------------------------------------------------
# Polygon.io helpers
# ---------------------------------------------------------------------------

def _polygon_client():
    """Lazy Polygon REST client — returns None if no key configured."""
    if not app.POLYGON_API_KEY:
        return None
    from polygon import RESTClient
    return RESTClient(api_key=app.POLYGON_API_KEY)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_price(ticker: str) -> str:
    """Return the current price and day change for a stock ticker (e.g. AAPL, TSLA, LLOY.L)."""
    def _fetch():
        info = yf.Ticker(ticker).fast_info
        return info.last_price, info.previous_close

    try:
        last, prev = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching price for {ticker}: {e}"

    if last is None or prev is None:
        return f"Could not retrieve price data for '{ticker}'. Check the ticker symbol."

    change = last - prev
    change_pct = (change / prev) * 100 if prev else 0
    arrow = "▲" if change >= 0 else "▼"
    sign = "+" if change >= 0 else ""

    return (
        f"**{ticker.upper()}**\n"
        f"- Price:      {last:,.4f}\n"
        f"- Day change: {arrow} {sign}{change:,.4f} ({sign}{change_pct:.2f}%)\n"
        f"- Prev close: {prev:,.4f}"
    )


@mcp.tool()
async def get_price_history(ticker: str, period: str = "1mo") -> str:
    """Return OHLCV price history for a ticker.
    period options: 1wk, 1mo, 3mo, 1y
    Use this to analyse performance over time and identify trends."""

    period_map = {"1wk": ("7d", "1d"), "1mo": ("1mo", "1d"), "3mo": ("3mo", "1wk"), "1y": ("1y", "1mo")}
    if period not in period_map:
        return f"Invalid period '{period}'. Use: 1wk, 1mo, 3mo, 1y"

    yf_period, interval = period_map[period]

    def _fetch():
        return yf.Ticker(ticker).history(period=yf_period, interval=interval, auto_adjust=True)

    try:
        df = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching history for {ticker}: {e}"

    if df.empty:
        return f"No historical data found for '{ticker}'."

    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    total_change = last_close - first_close
    total_pct = (total_change / first_close) * 100 if first_close else 0
    high = float(df["High"].max())
    low = float(df["Low"].min())
    avg_vol = int(df["Volume"].mean()) if "Volume" in df else 0

    lines = [
        f"**{ticker.upper()} — Price History ({period})**\n",
        f"- Period return: {'+' if total_pct >= 0 else ''}{total_pct:.2f}% "
        f"({'+' if total_change >= 0 else ''}{total_change:,.4f})",
        f"- Period high:   {high:,.4f}",
        f"- Period low:    {low:,.4f}",
        f"- Avg volume:    {avg_vol:,}",
        "",
        f"{'Date':<14} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Chg%':>7}",
        "-" * 65,
    ]

    prev_close = None
    for date, row in df.iterrows():
        date_str = date.strftime("%d %b %Y") if hasattr(date, "strftime") else str(date)[:10]
        close = float(row["Close"])
        chg_str = ""
        if prev_close:
            chg = (close - prev_close) / prev_close * 100
            chg_str = f"{'+' if chg >= 0 else ''}{chg:.2f}%"
        lines.append(
            f"{date_str:<14} {float(row['Open']):>10,.2f} {float(row['High']):>10,.2f} "
            f"{float(row['Low']):>10,.2f} {close:>10,.2f} {chg_str:>7}"
        )
        prev_close = close

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fundamentals & Analyst Ratings
# ---------------------------------------------------------------------------

@mcp.tool()
@cached(cache_fundamentals)
async def get_stock_fundamentals(ticker: str) -> str:
    """Return key fundamental data for a stock: P/E, EPS, market cap, revenue, margins, 52w range, beta, dividend.
    Use this for fundamental analysis and valuation."""

    def _fetch():
        return yf.Ticker(ticker).info

    try:
        info = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {e}"

    if not info or info.get("trailingPe") is None and info.get("marketCap") is None:
        return f"No fundamental data found for '{ticker}'."

    def _val(key, fmt=",", decimals=2, scale=1, suffix=""):
        v = info.get(key)
        if v is None:
            return "N/A"
        v = float(v) / scale
        if fmt == ",":
            return f"{v:,.{decimals}f}{suffix}"
        if fmt == "B":
            return f"${v/1e9:,.2f}B"
        if fmt == "M":
            return f"${v/1e6:,.0f}M"
        return str(v)

    name = info.get("longName") or ticker.upper()
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    currency = info.get("currency", "")

    return (
        f"**{name} ({ticker.upper()}) — Fundamentals**\n\n"
        f"**Valuation**\n"
        f"- Market cap:      {_val('marketCap', 'B')}\n"
        f"- P/E (trailing):  {_val('trailingPE')}\n"
        f"- P/E (forward):   {_val('forwardPE')}\n"
        f"- Price/Book:      {_val('priceToBook')}\n"
        f"- EV/EBITDA:       {_val('enterpriseToEbitda')}\n\n"
        f"**Earnings & Revenue**\n"
        f"- EPS (TTM):       {currency} {_val('trailingEps')}\n"
        f"- EPS (forward):   {currency} {_val('forwardEps')}\n"
        f"- Revenue (TTM):   {_val('totalRevenue', 'B')}\n"
        f"- Revenue growth:  {_val('revenueGrowth', ',', 1, 0.01, '%')}\n"
        f"- Gross margin:    {_val('grossMargins', ',', 1, 0.01, '%')}\n"
        f"- Profit margin:   {_val('profitMargins', ',', 1, 0.01, '%')}\n\n"
        f"**Price & Risk**\n"
        f"- 52w high:        {currency} {_val('fiftyTwoWeekHigh')}\n"
        f"- 52w low:         {currency} {_val('fiftyTwoWeekLow')}\n"
        f"- 50d MA:          {currency} {_val('fiftyDayAverage')}\n"
        f"- 200d MA:         {currency} {_val('twoHundredDayAverage')}\n"
        f"- Beta:            {_val('beta')}\n"
        f"- Dividend yield:  {_val('dividendYield', ',', 2, 0.01, '%')}\n\n"
        f"**Company**\n"
        f"- Sector:   {sector}\n"
        f"- Industry: {industry}\n"
        f"- Employees: {_val('fullTimeEmployees', ',', 0)}"
    )


@mcp.tool()
@cached(cache_fundamentals)
async def get_analyst_ratings(ticker: str) -> str:
    """Return analyst consensus ratings, price targets, and recommendation trends for a stock.
    Use this to understand what professional analysts think."""

    def _fetch():
        t = yf.Ticker(ticker)
        info = t.info
        try:
            recs = t.recommendations
        except Exception:
            recs = None
        return info, recs

    try:
        info, recs = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching analyst data for {ticker}: {e}"

    name = info.get("longName") or ticker.upper()
    currency = info.get("currency", "")
    lines = [f"**Analyst Ratings — {name} ({ticker.upper()})**\n"]

    n = info.get("numberOfAnalystOpinions", "?")
    mean_t  = info.get("targetMeanPrice")
    high_t  = info.get("targetHighPrice")
    low_t   = info.get("targetLowPrice")
    med_t   = info.get("targetMedianPrice")
    rec_key = info.get("recommendationKey", "N/A").upper()
    current = info.get("currentPrice") or info.get("regularMarketPrice")

    lines.append(f"**Consensus: {rec_key}** ({n} analysts)")
    lines.append("")
    lines.append(f"**Price Targets ({currency})**")
    lines.append(f"- Current price: {fmt_float(current)}")
    lines.append(f"- Mean target:   {fmt_float(mean_t)}")
    lines.append(f"- Median target: {fmt_float(med_t)}")
    lines.append(f"- High target:   {fmt_float(high_t)}")
    lines.append(f"- Low target:    {fmt_float(low_t)}")

    if mean_t and current:
        upside = (float(mean_t) - float(current)) / float(current) * 100
        sign = "+" if upside >= 0 else ""
        lines.append(f"- Implied upside: {sign}{upside:.1f}%")

    if recs is not None and not recs.empty:
        lines.append("")
        lines.append("**Recent Analyst Recommendations**")
        lines.append(f"{'Date':<14} {'Firm':<28} {'To Grade':<20} {'Action'}")
        lines.append("-" * 74)
        recent = recs.tail(8).iloc[::-1]
        for date, row in recent.iterrows():
            date_str = date.strftime("%d %b %Y") if hasattr(date, "strftime") else str(date)[:10]
            firm  = str(row.get("Firm", ""))[:27]
            grade = str(row.get("To Grade", row.get("toGrade", "")))[:19]
            action = str(row.get("Action", row.get("action", "")))
            lines.append(f"{date_str:<14} {firm:<28} {grade:<20} {action}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Market Snapshot
# ---------------------------------------------------------------------------

@mcp.tool()
@cached(cache_prices)
async def get_market_snapshot() -> str:
    """Return today's price moves for FTSE 100, S&P 500, and NASDAQ Composite."""
    indices = {
        "FTSE 100": "^FTSE",
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
    }

    def _fetch():
        return yf.download(
            list(indices.values()),
            period="2d",
            auto_adjust=True,
            progress=False,
        )

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching market data: {e}"

    lines = [f"**Market Snapshot — {datetime.today().strftime('%d %b %Y')}**\n",
             f"{'Index':<14} {'Price':>12} {'Change':>10} {'Change %':>10}",
             "-" * 50]

    close = data["Close"]
    for name, symbol in indices.items():
        try:
            prices = close[symbol].dropna()
            if len(prices) < 2:
                lines.append(f"{name:<14} {'N/A':>12}")
                continue
            prev = float(prices.iloc[-2])
            last = float(prices.iloc[-1])
            change = last - prev
            change_pct = (change / prev) * 100 if prev else 0
            arrow = "▲" if change >= 0 else "▼"
            sign = "+" if change >= 0 else ""
            lines.append(
                f"{name:<14} {last:>12,.2f} {arrow}{sign}{change:>8,.2f} {sign}{change_pct:>8.2f}%"
            )
        except Exception:
            lines.append(f"{name:<14} {'N/A':>12}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Financial Statements
# ---------------------------------------------------------------------------

@mcp.tool()
@cached(cache_fundamentals)
async def get_financial_statements(ticker: str) -> str:
    """Return income statement, balance sheet, and cash flow highlights for a stock.
    Shows last 4 annual periods with key ratios: ROE, ROA, current ratio, D/E, FCF yield.
    Use this for deep fundamental analysis beyond the basics."""
    from helpers import fmt_billions

    def _fetch():
        t = yf.Ticker(ticker)
        return {"info": t.info, "income": t.income_stmt, "balance": t.balance_sheet, "cashflow": t.cashflow}

    try:
        d = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching financial statements for {ticker}: {e}"

    info = d["info"]
    name = info.get("longName") or ticker.upper()
    currency = info.get("currency", "USD")

    lines = [f"**{name} ({ticker.upper()}) — Financial Statements ({currency})**\n"]

    inc = d["income"]
    if inc is not None and not inc.empty:
        lines.append("**Income Statement (Annual)**")
        rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA"]
        header = f"{'Metric':<22}" + "".join(f"{c.strftime('%Y'):>12}" for c in inc.columns[:4])
        lines.append(header)
        lines.append("-" * (22 + 12 * min(4, len(inc.columns))))
        for row in rows:
            if row in inc.index:
                vals = "".join(f"{fmt_billions(inc.loc[row, c]):>12}" for c in inc.columns[:4])
                lines.append(f"{row:<22}{vals}")
        lines.append("")

    bal = d["balance"]
    if bal is not None and not bal.empty:
        lines.append("**Balance Sheet (Annual)**")
        rows = ["Total Assets", "Total Liabilities Net Minority Interest",
                "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents"]
        header = f"{'Metric':<40}" + "".join(f"{c.strftime('%Y'):>12}" for c in bal.columns[:4])
        lines.append(header)
        lines.append("-" * (40 + 12 * min(4, len(bal.columns))))
        for row in rows:
            if row in bal.index:
                vals = "".join(f"{fmt_billions(bal.loc[row, c]):>12}" for c in bal.columns[:4])
                lines.append(f"{row[:39]:<40}{vals}")
        lines.append("")

    cf = d["cashflow"]
    if cf is not None and not cf.empty:
        lines.append("**Cash Flow (Annual)**")
        rows = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
                "Repurchase Of Capital Stock", "Cash Dividends Paid"]
        header = f"{'Metric':<30}" + "".join(f"{c.strftime('%Y'):>12}" for c in cf.columns[:4])
        lines.append(header)
        lines.append("-" * (30 + 12 * min(4, len(cf.columns))))
        for row in rows:
            if row in cf.index:
                vals = "".join(f"{fmt_billions(cf.loc[row, c]):>12}" for c in cf.columns[:4])
                lines.append(f"{row:<30}{vals}")
        lines.append("")

    lines.append("**Key Ratios**")
    try:
        if bal is not None and not bal.empty and inc is not None and not inc.empty:
            lb, li = bal.iloc[:, 0], inc.iloc[:, 0]
            equity = float(lb.get("Stockholders Equity", 0) or 0)
            assets = float(lb.get("Total Assets", 0) or 0)
            ni = float(li.get("Net Income", 0) or 0)
            td = float(lb.get("Total Debt", 0) or 0)
            ca = float(lb.get("Current Assets", 0) or 0)
            cl = float(lb.get("Current Liabilities", 0) or 0)
            lines.append(f"- ROE: {ni/equity*100:.1f}%" if equity else "- ROE: N/A")
            lines.append(f"- ROA: {ni/assets*100:.1f}%" if assets else "- ROA: N/A")
            lines.append(f"- Debt/Equity: {td/equity:.2f}" if equity else "- Debt/Equity: N/A")
            lines.append(f"- Current Ratio: {ca/cl:.2f}" if cl else "- Current Ratio: N/A")
        if cf is not None and not cf.empty:
            fcf = float(cf.iloc[:, 0].get("Free Cash Flow", 0) or 0)
            mcap = float(info.get("marketCap", 0) or 0)
            lines.append(f"- FCF Yield: {fcf/mcap*100:.2f}%" if mcap else "- FCF Yield: N/A")
    except Exception:
        lines.append("- Could not compute ratios")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DCF Valuation
# ---------------------------------------------------------------------------

@mcp.tool()
@cached(cache_fundamentals)
async def get_dcf_valuation(ticker: str, growth_rate_pct: float = 0.0, discount_rate_pct: float = 10.0) -> str:
    """Estimate intrinsic value using a Discounted Cash Flow (DCF) model.
    growth_rate_pct: override FCF growth rate (0 = auto-detect from analyst estimates)
    discount_rate_pct: discount rate / WACC (default 10%)
    Returns intrinsic value per share and margin of safety vs current price."""
    from helpers import fmt_billions as _b

    def _fetch():
        t = yf.Ticker(ticker)
        return t.info, t.cashflow

    try:
        info, cf = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching data for DCF: {e}"

    name = info.get("longName") or ticker.upper()
    currency = info.get("currency", "USD")
    current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    shares = float(info.get("sharesOutstanding", 0) or 0)

    if not shares or not current_price:
        return f"Insufficient data for DCF on {ticker}."

    if cf is None or cf.empty or "Free Cash Flow" not in cf.index:
        return f"No free cash flow data available for {ticker}."

    fcf_values = [float(v) for v in cf.loc["Free Cash Flow"].dropna().values[:4] if not np.isnan(float(v))]
    if not fcf_values:
        return f"No valid FCF data for {ticker}."

    latest_fcf = fcf_values[0]
    if latest_fcf <= 0:
        return f"{ticker} has negative FCF ({currency} {latest_fcf/1e9:.2f}B) — DCF not applicable."

    if growth_rate_pct == 0:
        ag = info.get("earningsGrowth") or info.get("revenueGrowth")
        growth_rate = float(ag) if ag else (
            (fcf_values[0] / fcf_values[-1]) ** (1 / (len(fcf_values) - 1)) - 1
            if len(fcf_values) >= 2 and fcf_values[-1] > 0 else 0.05
        )
    else:
        growth_rate = growth_rate_pct / 100

    discount_rate = discount_rate_pct / 100
    terminal_growth = 0.025
    years = 10

    projected = []
    fcf = latest_fcf
    for yr in range(1, years + 1):
        fcf *= (1 + growth_rate)
        pv = fcf / (1 + discount_rate) ** yr
        projected.append((yr, fcf, pv))

    tv_fcf = projected[-1][1] * (1 + terminal_growth)
    tv = tv_fcf / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** years

    total_pv = sum(pv for _, _, pv in projected)
    ev = total_pv + pv_tv
    cash = float(info.get("totalCash", 0) or 0)
    debt = float(info.get("totalDebt", 0) or 0)
    equity_val = ev + cash - debt
    intrinsic = equity_val / shares
    mos = (intrinsic - current_price) / intrinsic * 100

    verdict = "UNDERVALUED ✅" if mos > 15 else "FAIRLY VALUED ⚖️" if mos > -10 else "OVERVALUED ⚠️"

    lines = [
        f"**{name} ({ticker.upper()}) — DCF Valuation**\n",
        f"**Inputs**",
        f"- Latest FCF: {currency} {_b(latest_fcf)}",
        f"- Growth rate: {growth_rate*100:.1f}% | Discount: {discount_rate*100:.1f}% | Terminal: {terminal_growth*100:.1f}%",
        "",
        f"**Valuation**",
        f"- PV of FCFs:    {currency} {_b(total_pv)}",
        f"- PV terminal:   {currency} {_b(pv_tv)}",
        f"- Enterprise:    {currency} {_b(ev)}",
        f"- + Cash / - Debt: {currency} {_b(cash)} / {_b(debt)}",
        f"- **Equity:      {currency} {_b(equity_val)}**",
        "",
        f"**Result**",
        f"- Intrinsic value/share: **{currency} {intrinsic:,.2f}**",
        f"- Current price:         {currency} {current_price:,.2f}",
        f"- Margin of safety:      {mos:+.1f}%",
        f"- Verdict:               **{verdict}**",
        "",
        f"⚠️ DCF is sensitive to growth/discount assumptions. Use as one input among many.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Peer Comparison
# ---------------------------------------------------------------------------

@mcp.tool()
@cached(cache_fundamentals)
async def compare_peers(tickers: str) -> str:
    """Compare 2-6 stocks side-by-side on fundamentals, technicals, and analyst views.
    tickers: comma-separated list, e.g. 'AAPL,MSFT,GOOGL'"""

    import pandas as pd
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(symbols) < 2:
        return "Provide at least 2 tickers separated by commas."
    symbols = symbols[:6]

    async def _get(sym):
        def _f():
            t = yf.Ticker(sym)
            return t.info, t.history(period="1y", interval="1d", auto_adjust=True)
        try:
            return sym, *(await asyncio.to_thread(_f))
        except Exception:
            return sym, {}, pd.DataFrame()

    results = await asyncio.gather(*[_get(s) for s in symbols])

    metrics = [
        ("Name", lambda i, h: (i.get("shortName") or "")[:20]),
        ("Sector", lambda i, h: (i.get("sector") or "")[:15]),
        ("Mkt Cap", lambda i, h: f"{float(i.get('marketCap',0) or 0)/1e9:.0f}B" if i.get("marketCap") else "N/A"),
        ("P/E (TTM)", lambda i, h: f"{float(i['trailingPE']):.1f}" if i.get("trailingPE") else "N/A"),
        ("P/E (Fwd)", lambda i, h: f"{float(i['forwardPE']):.1f}" if i.get("forwardPE") else "N/A"),
        ("EV/EBITDA", lambda i, h: f"{float(i['enterpriseToEbitda']):.1f}" if i.get("enterpriseToEbitda") else "N/A"),
        ("Rev Growth", lambda i, h: f"{float(i['revenueGrowth'])*100:.1f}%" if i.get("revenueGrowth") else "N/A"),
        ("Profit Mgn", lambda i, h: f"{float(i['profitMargins'])*100:.1f}%" if i.get("profitMargins") else "N/A"),
        ("ROE", lambda i, h: f"{float(i['returnOnEquity'])*100:.1f}%" if i.get("returnOnEquity") else "N/A"),
        ("Beta", lambda i, h: f"{float(i['beta']):.2f}" if i.get("beta") else "N/A"),
        ("52w Chg", lambda i, h: f"{float(i['52WeekChange'])*100:+.1f}%" if i.get("52WeekChange") else "N/A"),
        ("Analyst", lambda i, h: (i.get("recommendationKey") or "N/A").upper()),
        ("Target Upside", lambda i, h: (
            f"{(float(i['targetMeanPrice']) - float(i.get('currentPrice', i.get('regularMarketPrice',0)) or 1)) / float(i.get('currentPrice', i.get('regularMarketPrice',0)) or 1) * 100:+.1f}%"
            if i.get("targetMeanPrice") and (i.get("currentPrice") or i.get("regularMarketPrice")) else "N/A"
        )),
    ]

    cw = 14
    lines = ["**Peer Comparison**\n"]
    lines.append(f"{'Metric':<16}" + "".join(f"{sym:>{cw}}" for sym, _, _ in results))
    lines.append("-" * (16 + cw * len(results)))

    for mname, ext in metrics:
        row = f"{mname:<16}"
        for sym, info, hist in results:
            try:
                row += f"{ext(info, hist):>{cw}}"
            except Exception:
                row += f"{'N/A':>{cw}}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Polygon.io — Real-time Quote & Market Status
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_realtime_quote(ticker: str) -> str:
    """Return a real-time snapshot quote for a stock via Polygon.io.
    Faster and more accurate than yfinance for intraday data.
    Includes last trade, bid/ask spread, and today's OHLCV."""

    client = _polygon_client()
    if not client:
        return await get_price(ticker)  # fallback to yfinance

    def _fetch():
        snapshot = client.get_snapshot_ticker("stocks", ticker.upper())
        return snapshot

    try:
        snap = await asyncio.to_thread(_fetch)
    except Exception as e:
        # Fallback to yfinance on error
        return await get_price(ticker)

    if not snap:
        return await get_price(ticker)

    try:
        day = snap.day
        prev = snap.prev_day
        last = snap.last_trade

        price = float(last.price) if last else float(day.close) if day else 0
        prev_close = float(prev.close) if prev else 0
        change = price - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        arrow = "▲" if change >= 0 else "▼"
        sign = "+" if change >= 0 else ""

        lines = [
            f"**{ticker.upper()} — Real-time (Polygon.io)**\n",
            f"- Last price:    {price:,.4f}",
            f"- Day change:    {arrow} {sign}{change:,.4f} ({sign}{change_pct:.2f}%)",
            f"- Prev close:    {prev_close:,.4f}",
        ]

        if day:
            lines += [
                "",
                f"**Today's Session**",
                f"- Open:    {float(day.open):,.4f}",
                f"- High:    {float(day.high):,.4f}",
                f"- Low:     {float(day.low):,.4f}",
                f"- Volume:  {int(day.volume):,}" if day.volume else "",
                f"- VWAP:    {float(day.vwap):,.4f}" if day.vwap else "",
            ]

        if last:
            lines.append(f"\n- Last trade size: {int(last.size):,}" if last.size else "")

        return "\n".join(l for l in lines if l)
    except Exception:
        return await get_price(ticker)


@mcp.tool()
async def get_market_status() -> str:
    """Return current market status (open/closed) and upcoming holidays via Polygon.io.
    Use this to know if markets are trading right now."""

    client = _polygon_client()
    if not client:
        return "POLYGON_API_KEY not set. Add it to .env for market status data."

    def _fetch():
        status = client.get_market_status()
        return status

    try:
        status = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching market status: {e}"

    if not status:
        return "Could not retrieve market status."

    try:
        lines = [
            f"**Market Status — {datetime.today().strftime('%d %b %Y %H:%M')}**\n",
        ]

        # Market status
        market = getattr(status, 'market', None) or 'unknown'
        lines.append(f"- Overall market: **{str(market).upper()}**")

        exchanges = getattr(status, 'exchanges', None)
        if exchanges:
            lines.append("\n**Exchanges**")
            for name in ['nyse', 'nasdaq', 'otc']:
                val = getattr(exchanges, name, None)
                if val:
                    lines.append(f"- {name.upper()}: {val}")

        currencies = getattr(status, 'currencies', None)
        if currencies:
            fx = getattr(currencies, 'fx', None)
            crypto = getattr(currencies, 'crypto', None)
            if fx:
                lines.append(f"- Forex: {fx}")
            if crypto:
                lines.append(f"- Crypto: {crypto}")

        # Server time
        server_time = getattr(status, 'serverTime', None)
        if server_time:
            lines.append(f"\n- Server time: {server_time}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error parsing market status: {e}"
