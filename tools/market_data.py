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



# ---------------------------------------------------------------------------
# Fundamentals & Analyst Ratings
# ---------------------------------------------------------------------------

def _div_yield(info: dict) -> str:
    """Return dividend yield as a clean percentage string.
    yfinance returns dividendYield as a decimal fraction (0.0047 = 0.47%).
    Guard against miscaled values: if result exceeds 25% it's likely already
    in pct form, so use the raw value directly."""
    dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    if not dy:
        return "N/A"
    pct = float(dy) * 100          # expect: 0.0047 → 0.47%
    if pct > 25:                    # sanity cap — almost certainly bad scale
        pct = float(dy)             # treat raw value as already a percentage
    return f"{pct:.2f}%"


@cached(cache_fundamentals)
async def _get_stock_fundamentals(ticker: str) -> str:
    """Return key fundamental data for a stock: P/E, EPS, market cap, revenue, margins, 52w range, beta, dividend.
    Use this for fundamental analysis and valuation."""

    from resolver import fetch_fundamental_dict
    try:
        rt, info = await asyncio.wait_for(fetch_fundamental_dict(ticker), timeout=15.0)
    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {e}"

    if not info or (not info.get("marketCap") and not info.get("trailingPE")):
        # If yfinance is still empty, attempt Finnhub fallback for basic profile
        try:
            from helpers import finnhub_retry
            import finnhub
            
            @finnhub_retry
            def _fh_profile():
                client = finnhub.Client(api_key=app.FINNHUB_API_KEY)
                # Finnhub: LSE:AZN for UK tickers
                fh_sym = f"LSE:{ticker.split('.')[0]}" if ticker.endswith(".L") else ticker
                return client.company_profile2(symbol=fh_sym)
                
            profile = await asyncio.to_thread(_fh_profile)
            if profile:
                info["marketCap"] = profile.get("marketCapitalization", 0) * 1e6 # Finnhub is in Millions
                info["longName"] = profile.get("name")
                info["sector"] = profile.get("finnhubIndustry")
                info["currency"] = profile.get("currency")
        except Exception:
            pass

    if not info or (not info.get("marketCap") and not info.get("trailingPE") and not info.get("longName")):
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
        f"- Dividend yield:  {_div_yield(info)}\n\n"
        f"**Company**\n"
        f"- Sector:   {sector}\n"
        f"- Industry: {industry}\n"
        f"- Employees: {_val('fullTimeEmployees', ',', 0)}"
    )


@cached(cache_fundamentals)
async def _get_analyst_ratings(ticker: str) -> str:
    """Return analyst consensus ratings, price targets, and recommendation trends for a stock.
    Use this to understand what professional analysts think."""

    def _fetch():
        t = yf.Ticker(ticker)
        info = t.info
        # yfinance ≥0.2.x: per-analyst history is in upgrades_downgrades
        # .recommendations now returns aggregate period counts — not useful here
        upgrades = None
        try:
            upgrades = t.upgrades_downgrades
        except Exception:
            pass
        return info, upgrades

    try:
        info, upgrades = await asyncio.to_thread(_fetch)
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

    # Recent upgrades/downgrades from yfinance upgrades_downgrades
    if upgrades is not None and not upgrades.empty:
        lines.append("")
        lines.append("**Recent Analyst Actions**")
        lines.append(f"{'Date':<14} {'Firm':<28} {'To Grade':<22} {'Action'}")
        lines.append("-" * 76)
        # Most recent 8, newest first
        recent = upgrades.head(8)
        for date, row in recent.iterrows():
            date_str = date.strftime("%d %b %Y") if hasattr(date, "strftime") else str(date)[:10]
            firm     = str(row.get("Firm", ""))[:27]
            to_grade = str(row.get("ToGrade", row.get("To Grade", "")))[:21]
            action   = str(row.get("Action", ""))
            lines.append(f"{date_str:<14} {firm:<28} {to_grade:<22} {action}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Market Snapshot
# ---------------------------------------------------------------------------

@cached(cache_prices)
async def _get_market_snapshot() -> str:
    """Return today's price moves for FTSE 100, S&P 500, and NASDAQ Composite."""
    indices = {
        "FTSE 100": "^FTSE",
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
    }

    def _fetch():
        return yf.download(
            list(indices.values()),
            period="5d",
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

@cached(cache_fundamentals)
async def _get_financial_statements(ticker: str) -> str:
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

@cached(cache_fundamentals)
async def _get_dcf_valuation(ticker: str, growth_rate_pct: float = 0.0, discount_rate_pct: float = 10.0) -> str:
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
        "",
        f"Note: DCF is sensitive to growth/discount assumptions. Use as one input among many.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Peer Comparison
# ---------------------------------------------------------------------------

@cached(cache_fundamentals)
async def _compare_peers(tickers: str) -> str:
    """Compare 2-6 stocks side-by-side on fundamentals, technicals, and analyst views.
    tickers: comma-separated list, e.g. 'AAPL,MSFT,GOOGL'"""

    import pandas as pd
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(symbols) < 2:
        return "Provide at least 2 tickers separated by commas."
    symbols = symbols[:6]

    async def _get(sym):
        from resolver import fetch_fundamental_dict
        try:
            rt, info = await asyncio.wait_for(fetch_fundamental_dict(sym), timeout=15.0)
            yf_sym = rt.yf_symbol
        except Exception:
            info, yf_sym = {}, sym

        def _hist():
            return yf.Ticker(yf_sym).history(period="1y", interval="1d", auto_adjust=True)
        try:
            hist = await asyncio.wait_for(asyncio.to_thread(_hist), timeout=10.0)
        except Exception:
            hist = pd.DataFrame()
        return sym, info, hist

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
async def get_market_status() -> str:
    """Return current market status (open/closed) and upcoming holidays via Polygon.io.
    Falls back to a timezone-based estimate if Polygon is not configured.
    Use this to know if markets are trading right now."""

    client = _polygon_client()
    if not client:
        # Fallback: estimate from time zones
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        now_uk = datetime.now(ZoneInfo("Europe/London"))
        weekday = now_et.weekday()  # 0=Mon, 6=Sun
        is_weekend = weekday >= 5
        us_open = not is_weekend and 9 <= now_et.hour < 16 and not (now_et.hour == 9 and now_et.minute < 30)
        lse_open = not is_weekend and 8 <= now_uk.hour < 16 and not (now_uk.hour == 16 and now_uk.minute > 30)

        lines = [
            f"**Market Status — {now_et.strftime('%d %b %Y %H:%M ET')}**\n",
            f"- NYSE/NASDAQ: **{'OPEN' if us_open else 'CLOSED'}** (9:30–16:00 ET)",
            f"- LSE:         **{'OPEN' if lse_open else 'CLOSED'}** (08:00–16:30 GMT)",
            f"- Weekend:     {'Yes ⛔' if is_weekend else 'No'}",
            "",
            "_⚠️ Estimate only — does not account for holidays. Set POLYGON_API_KEY for precise status._",
        ]
        return "\n".join(lines)

    def _fetch():
        status = client.get_market_status()
        return status

    try:
        status = await asyncio.to_thread(_fetch)
    except Exception as e:
        if "Unknown API Key" in str(e) or "401" in str(e):
            return f"❌ Polygon API Key Error: Your key is invalid. Please check .env."
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


# ===========================================================================
# NEW BATCH-CAPABLE API ENDPOINTS
# ===========================================================================

async def _get_prices_core(tickers: str, period: str = "current") -> str:
    """Return prices and historical data for one or multiple tickers.
    tickers: comma-separated list of symbols (e.g., 'AAPL, GOOG.O, LSE:COPX')
    period: 'current' (latest quote), '1wk', '1mo', '3mo', '1y'"""
    
    from resolver import bulk_resolve, fetch_fast_price
    from helpers import fetch_historic_prices
    
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not syms:
        return "Please provide at least one ticker."
        
    lines = []
    if period == "current":
        lines.append(f"**Current Prices ({len(syms)} assets)**\n")
        
        async def _get(s):
            rt, last, prev = await fetch_fast_price(s)
            return s, rt, last, prev
            
        results = await asyncio.gather(*[_get(s) for s in syms])
        
        for s, rt, last, prev in results:
            if last is None:
                lines.append(f"- **{s}**: Data unavailable.")
                continue
            
            change = last - (prev or last)
            change_pct = (change / prev) * 100 if prev else 0
            arrow = "▲" if change >= 0 else "▼"
            sign = "+" if change >= 0 else ""
            lines.append(f"- **{rt.yf_symbol}** ({rt.currency}): {last:,.4f} | Day change: {arrow} {sign}{change:,.4f} ({sign}{change_pct:.2f}%)")
        
        return "\n".join(lines)
        
    else:
        # History batching
        df = await fetch_historic_prices(syms, period=period)
        if df.empty:
            return f"No historical data found for '{tickers}'."
            
        lines.append(f"**Historical Prices ({period})**\n")
        # For multi-asset, compute period return summary
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        lines.append(f"{'Ticker':<12} {'Return':>10} {'Start':>12} {'Current':>12}")
        lines.append("-" * 50)
        
        for col in df.columns:
            import pandas as pd
            start_price = first_row[col]
            end_price = last_row[col]
            if pd.isna(start_price) or pd.isna(end_price):
                continue
            ret = (end_price - start_price) / start_price * 100
            sign = "+" if ret >= 0 else ""
            lines.append(f"{col:<12} {sign}{ret:>9.2f}% {start_price:>12,.2f} {end_price:>12,.2f}")
            
        return "\n".join(lines)

@mcp.tool()
async def get_fundamentals(tickers: str, section: str = "overview") -> str:
    """Return fundamental data for one or multiple tickers.
    tickers: comma-separated list of symbols (e.g. 'AAPL, GOOG').
    section: 'overview' (default), 'ratings', 'statements', 'dcf', 'peers'"""
    
    # For peer comparison, route back to the legacy comparative engine natively
    if section == "peers" or len(tickers.split(",")) > 1:
        # Route multi-ticker to peer logic
        return await _compare_peers(tickers)
        
    # For single ticker routing:
    t = tickers.split(",")[0].strip()
    if section == "ratings":
        return await _get_analyst_ratings(t)
    elif section == "statements":
        return await _get_financial_statements(t)
    elif section == "dcf":
        return await _get_dcf_valuation(t)
    else:
        return await _get_stock_fundamentals(t)

