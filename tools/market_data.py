"""Market data tools: prices, fundamentals, analyst ratings, market snapshot."""
import asyncio
from datetime import datetime

import numpy as np
import yfinance as yf

import app
import finnhub_data as fd
from helpers import cached, cache_fundamentals, cache_prices, fmt_float

mcp = app.mcp


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

    def _val(key, fmt=",", decimals=2, scale=1, suffix="", is_price=False):
        v = info.get(key)
        if v is None:
            return "N/A"
        val = float(v) / scale
        if is_price or key == "marketCap":
            val *= rt.unit_scale
        if fmt == ",":
            return f"{val:,.{decimals}f}{suffix}"
        if fmt == "B":
            return f"{val/1e9:,.2f}B"
        if fmt == "M":
            return f"${val/1e6:,.0f}M"
        return str(val)

    name = info.get("longName") or ticker.upper()
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    currency = info.get("currency", "")

    # Check how many critical fields are populated
    critical = ["trailingPE", "forwardPE", "trailingEps", "forwardEps",
                "grossMargins", "profitMargins", "beta", "sector"]
    missing = [f for f in critical if not info.get(f)]
    data_warning = ""
    if len(missing) >= 5:
        data_warning = (
            f"⚠️  DATA UNRELIABLE — {len(missing)}/{len(critical)} critical fields missing "
            f"({', '.join(missing)}). yFinance may be throttling or this instrument "
            f"has no coverage. Cross-check before trading.\n\n"
        )

    return (
        f"**{name} ({ticker.upper()}) — Fundamentals**\n\n"
        f"{data_warning}"
        f"**Valuation**\n"
        f"- Market cap:      {currency} {_val('marketCap', 'B')}\n"
        f"- P/E (trailing):  {_val('trailingPE')}\n"
        f"- P/E (forward):   {_val('forwardPE')}\n"
        f"- Price/Book:      {_val('priceToBook')}\n"
        f"- EV/EBITDA:       {_val('enterpriseToEbitda')}\n\n"
        f"**Earnings & Revenue**\n"
        f"- EPS (TTM):       {currency} {_val('trailingEps')}\n"
        f"- EPS (forward):   {currency} {_val('forwardEps')}\n"
        f"- Revenue (TTM):   {currency} {_val('totalRevenue', 'B')}\n"
        f"- Revenue growth:  {_val('revenueGrowth', ',', 1, 0.01, '%')}\n"
        f"- Gross margin:    {_val('grossMargins', ',', 1, 0.01, '%')}\n"
        f"- Profit margin:   {_val('profitMargins', ',', 1, 0.01, '%')}\n\n"
        f"**Price & Risk**\n"
        f"- 52w high:        {currency} {_val('fiftyTwoWeekHigh', is_price=True)}\n"
        f"- 52w low:         {currency} {_val('fiftyTwoWeekLow', is_price=True)}\n"
        f"- 50d MA:          {currency} {_val('fiftyDayAverage', is_price=True)}\n"
        f"- 200d MA:         {currency} {_val('twoHundredDayAverage', is_price=True)}\n"
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
        upgrades = None
        try:
            upgrades = t.upgrades_downgrades
        except Exception:
            pass
        return info, upgrades

    from resolver import aresolve
    try:
        from helpers import YF_INFO_SEM
        rt = await aresolve(ticker)
        async with YF_INFO_SEM:
            info, upgrades = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=12.0)
    except Exception as e:
        return f"Error fetching analyst data for {ticker}: {e}"

    name = info.get("longName") or ticker.upper()
    currency = info.get("currency", "")
    lines = [f"**Analyst Ratings — {name} ({ticker.upper()})**\n"]

    n = info.get("numberOfAnalystOpinions", "?")
    rec_key = info.get("recommendationKey", "N/A").upper()
    # Apply scaling
    def _s(v):
        if v is None: return None
        return float(v) * rt.unit_scale

    current = _s(info.get("currentPrice") or info.get("regularMarketPrice"))
    mean_t  = _s(info.get("targetMeanPrice"))
    high_t  = _s(info.get("targetHighPrice"))
    low_t   = _s(info.get("targetLowPrice"))
    med_t   = _s(info.get("targetMedianPrice"))

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

    # Finnhub recommendation-trend history (US names) — clean buy/hold/sell counts
    rec_trend = await fd.recommendation_trends(rt.yf_symbol)
    if rec_trend:
        lines.append("")
        lines.append("**Recommendation Trend (Finnhub, by month)**")
        lines.append(f"{'Month':<10} {'StrBuy':>7} {'Buy':>5} {'Hold':>6} {'Sell':>6} {'StrSell':>8}")
        for r in rec_trend[:6]:
            lines.append(
                f"{str(r.get('period', ''))[:7]:<10} {r.get('strongBuy', 0):>7} "
                f"{r.get('buy', 0):>5} {r.get('hold', 0):>6} {r.get('sell', 0):>6} "
                f"{r.get('strongSell', 0):>8}"
            )

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
        data = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=20.0)
    except asyncio.TimeoutError:
        return "Market snapshot timed out. Try again later."
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

def _format_finnhub_statements(symbol: str, periods: list[dict]) -> str:
    """Render as-reported Finnhub statements (US names) in the standard layout."""
    from helpers import fmt_billions

    cols = periods[:4]
    yrs = [p["period"][:4] or "?" for p in cols]
    lines = [f"**{symbol} — Financial Statements (as-reported, via Finnhub/SEC filings)**\n"]

    def _section(title: str, rows: list[tuple[str, str]], block: str):
        lines.append(f"**{title}**")
        lines.append(f"{'Metric':<24}" + "".join(f"{y:>14}" for y in yrs))
        lines.append("-" * (24 + 14 * len(yrs)))
        for label, key in rows:
            vals = "".join(f"{fmt_billions(p[block].get(key)):>14}" for p in cols)
            lines.append(f"{label:<24}{vals}")
        lines.append("")

    _section("Income Statement (Annual)", [
        ("Revenue", "revenue"), ("Gross Profit", "gross_profit"),
        ("Operating Income", "operating_income"), ("Pre-tax Income", "pretax_income"),
        ("Net Income", "net_income")], "income")
    _section("Balance Sheet (Annual)", [
        ("Total Assets", "total_assets"), ("Total Liabilities", "total_liabilities"),
        ("Shareholders Equity", "equity"), ("Cash & Equivalents", "cash"),
        ("Long-term Debt", "long_term_debt")], "balance")
    _section("Cash Flow (Annual)", [
        ("Operating Cash Flow", "operating_cf"), ("Capex", "capex"),
        ("Dividends Paid", "dividends_paid"), ("Buybacks", "buybacks")], "cashflow")

    latest = cols[0]
    inc, bal = latest["income"], latest["balance"]
    lines.append("**Key Ratios (latest period)**")

    def _ratio(label: str, num, den, pct: bool = True, dp: int = 1) -> str:
        if num is None or not den:
            return f"- {label}: N/A"
        v = num / den * (100 if pct else 1)
        return f"- {label}: {v:,.{dp}f}{'%' if pct else ''}"

    lines.append(_ratio("ROE", inc.get("net_income"), bal.get("equity")))
    lines.append(_ratio("ROA", inc.get("net_income"), bal.get("total_assets")))
    lines.append(_ratio("Gross margin", inc.get("gross_profit"), inc.get("revenue")))
    lines.append(_ratio("Net margin", inc.get("net_income"), inc.get("revenue")))
    lines.append(_ratio("Current ratio", bal.get("current_assets"),
                        bal.get("current_liabilities"), pct=False, dp=2))
    lines.append(_ratio("Debt/Equity (LT)", bal.get("long_term_debt"),
                        bal.get("equity"), pct=False, dp=2))
    lines.append(f"- Free cash flow: {fmt_billions(latest.get('fcf'))}")
    lines.append(f"\n_As-reported from SEC filings ({cols[0].get('form', '')} basis). Source: Finnhub._")
    return "\n".join(lines)


@cached(cache_fundamentals)
async def _get_financial_statements(ticker: str) -> str:
    """Return income statement, balance sheet, and cash flow highlights for a stock.
    Shows last 4 annual periods with key ratios: ROE, ROA, current ratio, D/E, FCF yield.

    For US names, uses Finnhub as-reported figures (parsed from SEC filings);
    falls back to yfinance for non-US names or when Finnhub has no data."""
    from helpers import fmt_billions
    from resolver import aresolve

    try:
        rt = await aresolve(ticker)
        yf_symbol = rt.yf_symbol
    except Exception:
        yf_symbol = ticker.upper()

    fh = await fd.reported_financials(yf_symbol)
    if fh:
        return _format_finnhub_statements(yf_symbol, fh)

    def _fetch():
        t = yf.Ticker(ticker)
        return {"info": t.info, "income": t.income_stmt, "balance": t.balance_sheet, "cashflow": t.cashflow}

    try:
        d = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=25.0)
    except asyncio.TimeoutError:
        return f"Financial statements request timed out for {ticker}."
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
        info, cf = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=20.0)
    except asyncio.TimeoutError:
        return f"DCF valuation data request timed out for {ticker}."
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
# Market Status — Finnhub (holiday-aware), timezone heuristic as last resort
# ---------------------------------------------------------------------------

def _market_status_heuristic() -> str:
    """Timezone-based open/closed estimate — used only if Finnhub is unreachable."""
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    now_uk = datetime.now(ZoneInfo("Europe/London"))
    is_weekend = now_et.weekday() >= 5
    us_open = not is_weekend and (
        (now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 30)) and now_et.hour < 16)
    lse_open = not is_weekend and 8 <= now_uk.hour < 16 and not (
        now_uk.hour == 16 and now_uk.minute > 30)
    return "\n".join([
        f"**Market Status — {now_et.strftime('%d %b %Y %H:%M ET')}**\n",
        f"- NYSE/NASDAQ: **{'OPEN' if us_open else 'CLOSED'}** (9:30–16:00 ET)",
        f"- London (LSE): **{'OPEN' if lse_open else 'CLOSED'}** (08:00–16:30 GMT)",
        "",
        "_⚠️ Timezone estimate — Finnhub unreachable, holidays not accounted for._",
    ])


@mcp.tool()
async def get_market_status() -> str:
    """Return current market status (open/closed, session) and upcoming US holidays
    for the NYSE/NASDAQ and London exchanges. Holiday-aware, via Finnhub.
    Use this to know if markets are trading right now."""

    # Finnhub market-status is US-only on the free tier; LSE uses a timezone estimate.
    us, us_hols = await asyncio.gather(fd.market_status("US"), fd.market_holidays("US"))

    if not us:
        return _market_status_heuristic()

    def _us_row(st: dict) -> str:
        state = "OPEN" if st.get("isOpen") else "CLOSED"
        session = st.get("session") or ""
        extra = f" — {session}" if session else ""
        if st.get("holiday"):
            extra += f"  (holiday: {st['holiday']})"
        return f"- US (NYSE/NASDAQ): **{state}**{extra}"

    from zoneinfo import ZoneInfo
    now_uk = datetime.now(ZoneInfo("Europe/London"))
    lse_open = now_uk.weekday() < 5 and 8 <= now_uk.hour < 16 and not (
        now_uk.hour == 16 and now_uk.minute > 30)

    lines = [
        f"**Market Status — {datetime.now().strftime('%d %b %Y %H:%M')}**\n",
        _us_row(us),
        f"- London (LSE): **{'OPEN' if lse_open else 'CLOSED'}** "
        f"(08:00–16:30 GMT — timezone estimate, holidays not checked)",
    ]
    if us_hols:
        lines.append("\n**Upcoming US holidays**")
        for h in us_hols:
            th = h.get("tradingHour") or ""
            lines.append(f"- {h.get('atDate', '')}: {h.get('eventName', '')}"
                         f"{f' (partial: {th})' if th else ' (closed)'}")
    lines.append("\n_Source: Finnhub_")
    return "\n".join(lines)


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
    
    # For peer comparison, route to the comparative engine.
    if section == "peers" or len(tickers.split(",")) > 1:
        syms = [t.strip() for t in tickers.split(",") if t.strip()]
        # Single ticker + peers → auto-expand the peer set via Finnhub (US names)
        if section == "peers" and len(syms) == 1:
            from resolver import aresolve
            try:
                rt = await aresolve(syms[0])
                peers = await fd.company_peers(rt.yf_symbol)
            except Exception:
                peers = None
            if peers:
                return await _compare_peers(",".join(peers[:6]))
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



@mcp.tool()
async def get_price_history(tickers: str, period: str = "5y", interval: str = "1mo") -> str:
    """Return OHLCV price history for one or more tickers.

    Use for regime analysis, scenario modelling, and drawdown mapping.

    tickers:  comma-separated symbols (e.g. 'AAPL, SPY'). Max 5.
    period:   '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | '10y' | 'max'
    interval: '1d' | '1wk' | '1mo'  (use '1mo' for long regimes, '1d' for short-term)
    """
    from resolver import aresolve
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()][:5]
    if not syms:
        return "Provide at least one ticker."

    lines = [f"**Price History | Period: {period} | Interval: {interval}**\n"]

    for raw in syms:
        try:
            rt = await aresolve(raw)
            def _fetch(sym=rt.yf_symbol, p=period, iv=interval):
                import logging
                logging.getLogger("yfinance").setLevel(logging.CRITICAL)
                return yf.Ticker(sym).history(period=p, interval=iv, auto_adjust=True)

            df = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=15.0)
            if df.empty:
                lines.append(f"**{raw}**: no data returned.\n")
                continue

            close = df["Close"] * rt.unit_scale
            ret_total = (close.iloc[-1] / close.iloc[0] - 1) * 100
            peak = close.max()
            trough_after_peak = close[close.index >= close.idxmax()].min()
            max_dd = (trough_after_peak / peak - 1) * 100

            lines.append(f"**{raw}** ({rt.yf_symbol}) | {rt.currency} | {len(close)} bars")
            lines.append(f"- Range:        {close.index[0].date()} → {close.index[-1].date()}")
            lines.append(f"- Total return: {ret_total:+.1f}%")
            lines.append(f"- Max drawdown: {max_dd:.1f}%")
            lines.append(f"- First close:  {close.iloc[0]:,.4f}")
            lines.append(f"- Last close:   {close.iloc[-1]:,.4f}")
            lines.append(f"- Peak:         {peak:,.4f}  |  Trough post-peak: {trough_after_peak:,.4f}")
            lines.append("")

            # OHLCV table (capped at 60 rows to keep response manageable)
            df_out = df.copy()
            df_out["Close"] = df_out["Close"] * rt.unit_scale
            df_out["Open"]  = df_out["Open"]  * rt.unit_scale
            df_out["High"]  = df_out["High"]  * rt.unit_scale
            df_out["Low"]   = df_out["Low"]   * rt.unit_scale
            sample = df_out.tail(60) if len(df_out) > 60 else df_out
            if len(df_out) > 60:
                lines.append(f"_(showing last 60 of {len(df_out)} bars)_")
            lines.append(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>14}")
            lines.append("-" * 60)
            for dt, row in sample.iterrows():
                date_s = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
                lines.append(
                    f"{date_s:<12} {row['Open']:>10.4f} {row['High']:>10.4f} "
                    f"{row['Low']:>10.4f} {row['Close']:>10.4f} {int(row.get('Volume', 0)):>14,}"
                )
            lines.append("")

        except Exception as e:
            lines.append(f"**{raw}**: error — {e}\n")

    return "\n".join(lines)
