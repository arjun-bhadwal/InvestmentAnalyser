import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps

import finnhub
import numpy as np
import pandas as pd
import ta as ta_lib
import yfinance as yf
from cachetools import TTLCache
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from fastmcp import FastMCP
from scipy.stats import norm

from t212_client import T212Client

load_dotenv()

T212_API_KEY = os.environ["T212_API_KEY"]
T212_API_SECRET = os.environ["T212_API_SECRET"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
T212_MODE = os.environ.get("T212_MODE", "demo")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

t212: T212Client

# ---------------------------------------------------------------------------
# Caching — avoids repeated expensive API calls within TTL windows
# ---------------------------------------------------------------------------

_cache_fundamentals = TTLCache(maxsize=128, ttl=900)    # 15 min
_cache_prices       = TTLCache(maxsize=64,  ttl=300)    # 5 min
_cache_macro        = TTLCache(maxsize=16,  ttl=3600)   # 1 hour


def _cached(cache: TTLCache):
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


@asynccontextmanager
async def lifespan(server):
    global t212
    t212 = T212Client(api_key=T212_API_KEY, api_secret=T212_API_SECRET, mode=T212_MODE)
    yield
    await t212.aclose()


mcp = FastMCP(
    "Investment Analyser",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Trader Protocol Prompt
# ---------------------------------------------------------------------------

TRADER_PROTOCOL = """You are a trader and financial investor with 30 years of experience. You reason market dynamics over long and short periods. You understand geopolitical plays. You only take calculated risks when necessary. You always prioritise growth to meet the target whilst keeping in mind safety — but ensuring the target is met.

Your job is to be my lead trader and investment analyst. You advise me on trades, investments and my portfolio. My risk appetite is medium. I am looking to grow my wealth. I like to back plays that are geopolitically guaranteed.

You keep track of real-time stock updates, stocks on my watchlist, Domino effects, and news that affects my stocks. You look out for better plays about to pop up and suggest investments.

You back conclusions with thorough financial analysis, news, and chatter on social media and from expert analysts.

STRATEGY: Take calculated risks when needed. No risks on assumptions or predictions alone — calculated and precise. Always analyse current events, geopolitical dominos, and chatter from legitimate voices and expert analysts.

Be realistic. First grow and protect the wealth. Hitting a target is an ambition, not the priority. Wealth must be grown aggressively only by taking calculated risks, with priority towards protecting it.

When asked for updates: tell what has happened with the stock in the past few hours using latest prices and data. Analyse chatter from legitimate voices, experts, historical trends, ALL geopolitical news, Domino effects — present the current view and outlook for near future and long term. Analyse past performance for 1 week, 1 month, 1 quarter and 1 year. Based on fundamentals and all available data, make an outlook for the next week, month, quarter and year.

Decisions should be based on how long we plan to hold a bought asset. Always use the latest market data. Do not hallucinate. If you don't have data from within the past hour, say so."""


@mcp.prompt()
def trader_protocol() -> str:
    """Load the Investment Analyser trader protocol and persona."""
    return TRADER_PROTOCOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_t212_ticker(raw: str) -> str:
    """Convert T212 instrument codes like 'AAPL_US_EQ' → 'AAPL'."""
    return raw.split("_")[0]


def _fmt_float(value, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Portfolio & Account Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio() -> str:
    """Return all open positions from your Trading 212 account with quantity, average price, current price, and P&L."""
    try:
        positions = await t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return f"No open positions in your T212 {T212_MODE} account."

    lines = [
        f"**Trading 212 Portfolio ({T212_MODE.upper()})**\n",
        f"{'Ticker':<10} {'Qty':>10} {'Avg Price':>12} {'Current':>12} {'P&L':>12}",
        "-" * 58,
    ]
    for pos in positions:
        ticker = _strip_t212_ticker(pos.get("ticker", "?"))
        qty = _fmt_float(pos.get("quantity"), 4)
        avg = _fmt_float(pos.get("averagePrice"))
        cur = _fmt_float(pos.get("currentPrice"))
        ppl = pos.get("ppl", 0)
        ppl_str = f"{float(ppl):+,.2f}" if ppl is not None else "N/A"
        lines.append(f"{ticker:<10} {qty:>10} {avg:>12} {cur:>12} {ppl_str:>12}")

    total_ppl = sum(float(p.get("ppl", 0) or 0) for p in positions)
    lines.append("-" * 58)
    lines.append(f"{'Total P&L':<46} {total_ppl:>+,.2f}")
    return "\n".join(lines)


@mcp.tool()
async def get_account_summary() -> str:
    """Return a summary of your Trading 212 account: total value, free cash, and total invested."""
    try:
        data = await t212.get_account_summary()
    except Exception as e:
        return f"Error fetching account summary: {e}"

    currency = data.get("currency", "")
    total = _fmt_float(data.get("totalValue"))
    free_cash = _fmt_float(data.get("cash", {}).get("availableToTrade"))
    invested = _fmt_float(data.get("investments", {}).get("totalCost"))
    ppl = data.get("investments", {}).get("unrealizedProfitLoss")
    ppl_str = f"{float(ppl):+,.2f}" if ppl is not None else "N/A"

    return (
        f"**Trading 212 Account Summary ({T212_MODE.upper()})**\n\n"
        f"- Total value:    {currency} {total}\n"
        f"- Available cash: {currency} {free_cash}\n"
        f"- Invested:       {currency} {invested}\n"
        f"- Unrealised P&L: {currency} {ppl_str}"
    )


@mcp.tool()
async def get_trade_history(limit: int = 20) -> str:
    """Return your recent Trading 212 order/trade history (buys and sells)."""
    try:
        orders = await t212.get_order_history(limit=limit)
    except Exception as e:
        return f"Error fetching trade history: {e}"

    if not orders:
        return "No trade history found."

    lines = [f"**Trade History — last {limit} orders ({T212_MODE.upper()})**\n",
             f"{'Date':<14} {'Name':<28} {'Side':<5} {'Qty':>10} {'Price':>10} {'Value':>10} {'CCY':<5}",
             "-" * 86]

    for o in orders:
        # Live API wraps in {order: {...}, fill: {...}}; demo is flat
        order = o.get("order", o)
        fill  = o.get("fill", o)

        raw_ticker = order.get("ticker", "?")
        instrument = order.get("instrument", {}) or {}
        name = instrument.get("name") or _strip_t212_ticker(raw_ticker)
        name = name[:27]

        side = order.get("side", "")
        if not side:
            otype = order.get("type", "")
            side = "BUY" if "BUY" in otype.upper() else "SELL" if "SELL" in otype.upper() else otype[:4]

        qty = float(fill.get("quantity") or order.get("filledQuantity") or order.get("quantity") or 0)

        # Use walletImpact.netValue as the true GBP value — avoids GBX/GBP confusion
        # where some LSE instruments return prices in pence but currency shows GBP
        wallet = (fill.get("walletImpact") or {})
        value  = abs(float(wallet.get("netValue") or order.get("filledValue") or order.get("value") or 0))
        ccy    = wallet.get("currency") or order.get("currency", "")

        # Back-calculate price in GBP from the true value
        price = (value / qty) if qty else 0

        raw_date = order.get("createdAt") or order.get("dateCreated") or fill.get("filledAt") or ""
        try:
            date_str = datetime.fromisoformat(raw_date[:19].replace("Z", "")).strftime("%d %b %Y")
        except Exception:
            date_str = raw_date[:10]

        lines.append(
            f"{date_str:<14} {name:<28} {side:<5} {_fmt_float(qty, 4):>10} "
            f"{_fmt_float(price):>10} {_fmt_float(value):>10} {ccy:<5}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Market Data Tools
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


@mcp.tool()
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

    # Price targets & consensus
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
    lines.append(f"- Current price: {_fmt_float(current)}")
    lines.append(f"- Mean target:   {_fmt_float(mean_t)}")
    lines.append(f"- Median target: {_fmt_float(med_t)}")
    lines.append(f"- High target:   {_fmt_float(high_t)}")
    lines.append(f"- Low target:    {_fmt_float(low_t)}")

    if mean_t and current:
        upside = (float(mean_t) - float(current)) / float(current) * 100
        sign = "+" if upside >= 0 else ""
        lines.append(f"- Implied upside: {sign}{upside:.1f}%")

    # Recent recommendation history from yfinance
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


@mcp.tool()
async def get_news(ticker: str) -> str:
    """Return the 5 most recent news headlines for a stock ticker via Finnhub.
    Use standard tickers for US stocks (e.g. AAPL). For LSE stocks use EXCHANGE:TICKER format (e.g. LSE:LLOY).
    """
    today = datetime.today().date()
    week_ago = today - timedelta(days=7)

    def _fetch():
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        return client.company_news(
            ticker.upper(),
            _from=str(week_ago),
            to=str(today),
        )

    try:
        articles = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching news for {ticker}: {e}"

    if not articles:
        return f"No news found for '{ticker}' in the past 7 days."

    lines = [f"**Recent news: {ticker.upper()}**\n"]
    for article in articles[:5]:
        headline = article.get("headline", "No headline")
        source = article.get("source", "")
        url = article.get("url", "")
        ts = article.get("datetime", 0)
        date_str = datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else ""
        lines.append(f"- [{headline}]({url}) — {source} ({date_str})")

    return "\n".join(lines)


@mcp.tool()
async def search_web(query: str) -> str:
    """Search the web for financial news, analyst commentary, market chatter, geopolitical events, or any topic.
    Use this to find latest news on stocks, sectors, macro events, or X/social media sentiment.
    Examples: 'TSLA Elon Musk latest news', 'oil price geopolitical risk 2025', 'CVX analyst outlook'"""

    def _fetch():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=8))

    try:
        results = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error searching web for '{query}': {e}"

    if not results:
        return f"No results found for '{query}'."

    lines = [f"**Web Search: {query}**\n"]
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")[:200]
        url = r.get("href", "")
        lines.append(f"**{title}**")
        lines.append(f"{body}...")
        lines.append(f"[{url}]({url})\n")

    return "\n".join(lines)


@mcp.tool()
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
# History — Dividends & Transactions
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_dividend_history(limit: int = 20) -> str:
    """Return dividend payments received in your Trading 212 account.
    Use this to track income from dividend-paying stocks."""
    try:
        dividends = await t212.get_dividend_history(limit=limit)
    except Exception as e:
        return f"Error fetching dividend history: {e}"

    if not dividends:
        return f"No dividend history found in your T212 {T212_MODE} account."

    lines = [
        f"**Dividend History ({T212_MODE.upper()}) — last {limit}**\n",
        f"{'Date':<14} {'Ticker':<10} {'Shares':>10} {'Amount':>12} {'Tax':>10}",
        "-" * 60,
    ]
    total = 0.0
    for d in dividends:
        ticker = _strip_t212_ticker(d.get("ticker", "?"))
        raw_date = d.get("paidOn") or d.get("date") or ""
        try:
            date_str = datetime.fromisoformat(raw_date[:10]).strftime("%d %b %Y")
        except Exception:
            date_str = raw_date[:10]
        amount = float(d.get("amount", 0) or 0)
        tax = float(d.get("grossAmountPerShare", 0) or 0)
        quantity = float(d.get("quantity", 0) or 0)
        total += amount
        lines.append(
            f"{date_str:<14} {ticker:<10} {_fmt_float(quantity, 4):>10} "
            f"{_fmt_float(amount):>12} {_fmt_float(tax):>10}"
        )

    lines.append("-" * 60)
    lines.append(f"{'Total dividends received':<46} {_fmt_float(total):>12}")
    return "\n".join(lines)


@mcp.tool()
async def get_transaction_history(limit: int = 20) -> str:
    """Return cash transaction history: deposits and withdrawals on your Trading 212 account."""
    try:
        transactions = await t212.get_transaction_history(limit=limit)
    except Exception as e:
        return f"Error fetching transaction history: {e}"

    if not transactions:
        return f"No transaction history found in your T212 {T212_MODE} account."

    lines = [
        f"**Transaction History ({T212_MODE.upper()}) — last {limit}**\n",
        f"{'Date':<14} {'Type':<16} {'Amount':>14} {'Currency':<10}",
        "-" * 58,
    ]
    for t in transactions:
        raw_date = t.get("dateTime") or t.get("date") or ""
        try:
            date_str = datetime.fromisoformat(raw_date[:10]).strftime("%d %b %Y")
        except Exception:
            date_str = raw_date[:10]
        tx_type = t.get("type", "UNKNOWN")
        amount = float(t.get("amount", 0) or 0)
        currency = t.get("currency", "")
        sign = "+" if amount >= 0 else ""
        lines.append(f"{date_str:<14} {tx_type:<16} {sign}{_fmt_float(amount):>14} {currency:<10}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orders — Open / Pending
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_open_orders() -> str:
    """Return any currently open or pending orders on your Trading 212 account.
    Use this to check what limit or pending orders are waiting to be filled."""
    try:
        orders = await t212.get_open_orders()
    except Exception as e:
        return f"Error fetching open orders: {e}"

    if not orders:
        return f"No open or pending orders in your T212 {T212_MODE} account."

    lines = [
        f"**Open Orders ({T212_MODE.upper()})**\n",
        f"{'Ticker':<10} {'Type':<12} {'Side':<6} {'Qty':>10} {'Limit':>10} {'Value':>10} {'Status':<12}",
        "-" * 74,
    ]
    for o in orders:
        ticker = _strip_t212_ticker(o.get("ticker", "?"))
        order_type = o.get("type", "")
        side = "BUY" if "BUY" in order_type.upper() else "SELL" if "SELL" in order_type.upper() else "?"
        qty = o.get("quantity", 0)
        limit = o.get("limitPrice") or o.get("stopPrice") or 0
        value = o.get("value", 0) or (float(qty or 0) * float(limit or 0))
        status = o.get("status", "?")
        lines.append(
            f"{ticker:<10} {order_type:<12} {side:<6} {_fmt_float(qty, 4):>10} "
            f"{_fmt_float(limit):>10} {_fmt_float(value):>10} {status:<12}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pies
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_pies() -> str:
    """Return all Pies (automated portfolios) in your Trading 212 account with their value and performance.
    Use this to see how your pie investments are performing."""
    try:
        pies = await t212.get_pies()
    except Exception as e:
        return f"Error fetching pies: {e}"

    if not pies:
        return f"No pies found in your T212 {T212_MODE} account."

    # Fetch names for each pie in parallel
    async def _get_name(pie_id: int) -> str:
        try:
            detail = await t212.get_pie(pie_id)
            return (detail.get("settings", {}) or {}).get("name", "") or f"Pie {pie_id}"
        except Exception:
            return f"Pie {pie_id}"

    names = await asyncio.gather(*[_get_name(p["id"]) for p in pies])

    lines = [
        f"**Pies ({T212_MODE.upper()})**\n",
        f"{'ID':<10} {'Name':<28} {'Invested':>12} {'Value':>12} {'Return':>10} {'Return%':>8}",
        "-" * 84,
    ]
    for pie, name in zip(pies, names):
        pie_id = str(pie.get("id", "?"))
        result = pie.get("result", {}) or {}
        invested = float(result.get("priceAvgInvestedValue", 0) or 0)
        value    = float(result.get("priceAvgValue", 0) or 0)
        ret      = float(result.get("priceAvgResult", 0) or 0)
        ret_coef = float(result.get("priceAvgResultCoef", 0) or 0)
        ret_pct  = ret_coef * 100
        sign = "+" if ret >= 0 else ""
        lines.append(
            f"{pie_id:<10} {name[:27]:<28} {_fmt_float(invested):>12} {_fmt_float(value):>12} "
            f"{sign}{_fmt_float(ret):>10} {sign}{ret_pct:.2f}%"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_technical_indicators(ticker: str) -> str:
    """Return key technical indicators for a stock: RSI, MACD, Bollinger Bands, moving averages,
    ATR, and a plain-English signal summary. Use this for entry/exit timing and trend confirmation."""

    def _fetch():
        df = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # Moving averages
        df["ma20"]  = ta_lib.trend.sma_indicator(close, window=20)
        df["ma50"]  = ta_lib.trend.sma_indicator(close, window=50)
        df["ma200"] = ta_lib.trend.sma_indicator(close, window=200)

        # RSI
        df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()

        # MACD
        macd_obj = ta_lib.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        df["macd"]      = macd_obj.macd()
        df["macd_sig"]  = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()

        # Bollinger Bands
        bb_obj = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb_obj.bollinger_hband()
        df["bb_mid"]   = bb_obj.bollinger_mavg()
        df["bb_lower"] = bb_obj.bollinger_lband()

        # ATR
        df["atr"] = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        return df

    try:
        df = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error computing indicators for {ticker}: {e}"

    if df is None or df.empty:
        return f"No data found for '{ticker}'."

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row
    price = row["close"]

    def _g(col):
        v = row.get(col)
        return float(v) if v is not None and not np.isnan(float(v) if v is not None else float("nan")) else None

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

    # --- Signal interpretation ---
    signals = []

    if rsi is not None:
        if rsi < 30:
            signals.append("RSI OVERSOLD — potential reversal / buy zone")
        elif rsi > 70:
            signals.append("RSI OVERBOUGHT — caution, potential pullback")
        else:
            signals.append(f"RSI neutral ({rsi:.1f})")

    if ma20 and ma50:
        if ma20 > ma50:
            signals.append("MA20 > MA50 — short-term uptrend")
        else:
            signals.append("MA20 < MA50 — short-term downtrend")

    if ma50 and ma200:
        if ma50 > ma200:
            signals.append("Golden Cross: MA50 > MA200 — bullish long-term")
        else:
            signals.append("Death Cross: MA50 < MA200 — bearish long-term")

    if macd_val is not None and macd_sig is not None:
        prev_hist = float(prev.get("macd_hist") or 0)
        if macd_hist is not None:
            if macd_hist > 0 and prev_hist <= 0:
                signals.append("MACD bullish crossover — momentum turning positive")
            elif macd_hist < 0 and prev_hist >= 0:
                signals.append("MACD bearish crossover — momentum turning negative")
            elif macd_hist > 0:
                signals.append("MACD positive — bullish momentum")
            else:
                signals.append("MACD negative — bearish momentum")

    if bb_upper and bb_lower and bb_mid:
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        if bb_pct > 90:
            signals.append(f"Price near upper Bollinger Band ({bb_pct:.0f}%) — extended, watch for rejection")
        elif bb_pct < 10:
            signals.append(f"Price near lower Bollinger Band ({bb_pct:.0f}%) — oversold zone")
        else:
            signals.append(f"Price within Bollinger Bands ({bb_pct:.0f}% of range)")

    lines = [
        f"**{ticker.upper()} — Technical Indicators**\n",
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
# Portfolio Risk Analytics
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio_risk() -> str:
    """Compute portfolio-level risk metrics: Sharpe ratio, Sortino ratio, max drawdown,
    annualised return/volatility, and pairwise correlation between positions.
    Use this for portfolio health checks and position sizing decisions."""

    # Step 1: get current positions from T212
    try:
        positions = await t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions to analyse."

    tickers = [_strip_t212_ticker(p["ticker"]) for p in positions]

    # Step 2: fetch 1-year daily closes for all tickers
    def _fetch_history(syms):
        raw = yf.download(syms, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"]
        return raw[["Close"]].rename(columns={"Close": syms[0]}) if len(syms) == 1 else raw

    try:
        closes = await asyncio.to_thread(_fetch_history, tickers)
    except Exception as e:
        return f"Error fetching price history: {e}"

    closes = closes.dropna(how="all")
    if closes.empty:
        return "Could not retrieve sufficient price history for portfolio risk analysis."

    returns = closes.pct_change().dropna()

    TRADING_DAYS = 252

    # Dynamic risk-free rate from 10-Year Treasury yield
    try:
        _rf_info = yf.Ticker("^TNX").fast_info
        RISK_FREE = float(_rf_info.last_price) / 100  # ^TNX quotes in %, e.g. 4.5 → 0.045
    except Exception:
        RISK_FREE = 0.045  # fallback

    lines = ["**Portfolio Risk Analytics — 1-Year Lookback**\n"]

    # --- Per-position stats ---
    lines.append(f"{'Ticker':<10} {'Ann.Ret%':>10} {'Ann.Vol%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Beta':>7}")
    lines.append("-" * 56)

    sharpes = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 20:
            lines.append(f"{col:<10} {'insufficient data':>44}")
            continue

        ann_ret  = float(r.mean() * TRADING_DAYS)
        ann_vol  = float(r.std() * np.sqrt(TRADING_DAYS))
        sharpe   = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd   = float(drawdown.min())

        # Beta vs S&P 500
        try:
            spy = yf.Ticker("^GSPC").history(period="1y", interval="1d", auto_adjust=True)["Close"].pct_change().dropna()
            aligned = pd.concat([r, spy], axis=1).dropna()
            if len(aligned) > 20:
                cov = aligned.cov().iloc[0, 1]
                var = aligned.iloc[:, 1].var()
                beta = cov / var if var > 0 else 1.0
            else:
                beta = float("nan")
        except Exception:
            beta = float("nan")

        sharpes.append(sharpe)
        beta_str = f"{beta:.2f}" if not np.isnan(beta) else "N/A"
        lines.append(
            f"{col:<10} {ann_ret*100:>+9.1f}% {ann_vol*100:>9.1f}% "
            f"{sharpe:>8.2f} {max_dd*100:>7.1f}% {beta_str:>7}"
        )

    # --- Portfolio-level stats (equal-weight approximation) ---
    port_returns = returns[list(returns.columns)].mean(axis=1)
    p_ann_ret  = float(port_returns.mean() * TRADING_DAYS)
    p_ann_vol  = float(port_returns.std() * np.sqrt(TRADING_DAYS))
    p_sharpe   = (p_ann_ret - RISK_FREE) / p_ann_vol if p_ann_vol > 0 else 0

    p_cum = (1 + port_returns).cumprod()
    p_max_dd = float(((p_cum - p_cum.cummax()) / p_cum.cummax()).min())

    # Sortino (downside deviation)
    downside = port_returns[port_returns < 0]
    sortino_denom = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else 0
    sortino = (p_ann_ret - RISK_FREE) / sortino_denom if sortino_denom > 0 else 0

    # Value at Risk (95% 1-day)
    var_95 = float(norm.ppf(0.05, port_returns.mean(), port_returns.std()))

    lines.append("-" * 56)
    lines.append(f"\n**Portfolio Summary (equal-weight)**")
    lines.append(f"- Annualised return:   {p_ann_ret*100:+.1f}%")
    lines.append(f"- Annualised vol:      {p_ann_vol*100:.1f}%")
    lines.append(f"- Sharpe ratio:        {p_sharpe:.2f}  (>1 = good, >2 = strong)")
    lines.append(f"- Sortino ratio:       {sortino:.2f}  (penalises downside only)")
    lines.append(f"- Max drawdown:        {p_max_dd*100:.1f}%")
    lines.append(f"- 1-day 95% VaR:       {var_95*100:.2f}%  (expected worst-day loss 1-in-20)")

    # --- Correlation matrix ---
    if len(returns.columns) > 1:
        corr = returns.corr().round(2)
        lines.append(f"\n**Correlation Matrix (1Y daily returns)**")
        header = f"{'':>10}" + "".join(f"{c:>8}" for c in corr.columns)
        lines.append(header)
        for row_name in corr.index:
            row_vals = "".join(f"{corr.loc[row_name, c]:>8.2f}" for c in corr.columns)
            lines.append(f"{row_name:>10}{row_vals}")
        lines.append("\n  (Values near 1.0 = highly correlated = less diversification benefit)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Earnings Calendar
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_earnings_calendar() -> str:
    """Return upcoming earnings dates and recent EPS results for all stocks in your portfolio.
    Use this to manage event risk — avoid being caught offside by surprise earnings moves."""

    try:
        positions = await t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions found."

    tickers = [_strip_t212_ticker(p["ticker"]) for p in positions]

    def _fetch_earnings(ticker):
        t = yf.Ticker(ticker)
        info = t.info
        try:
            cal = t.calendar
        except Exception:
            cal = None
        try:
            history = t.earnings_history
        except Exception:
            history = None
        return info, cal, history

    lines = ["**Earnings Calendar — Current Portfolio**\n"]

    upcoming = []   # (date, ticker, eps_est)
    results  = []   # (ticker, recent rows)

    for ticker in tickers:
        try:
            info, cal, hist = await asyncio.to_thread(_fetch_earnings, ticker)
        except Exception:
            continue

        name = (info.get("longName") or ticker)[:28]

        # Next earnings date
        next_date = None
        if cal is not None:
            if isinstance(cal, dict):
                next_date = cal.get("Earnings Date") or cal.get("earningsDate")
                if isinstance(next_date, list) and next_date:
                    next_date = next_date[0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                try:
                    nd = cal.loc["Earnings Date"]
                    next_date = nd.iloc[0] if hasattr(nd, "iloc") else nd
                except Exception:
                    pass

        eps_est = info.get("epsForward") or info.get("forwardEps")

        if next_date:
            try:
                dt = pd.Timestamp(next_date).date()
                days_away = (dt - datetime.today().date()).days
                upcoming.append((dt, days_away, ticker, name, eps_est))
            except Exception:
                pass

        # Recent earnings history (last 4 quarters)
        if hist is not None and not hist.empty:
            results.append((ticker, name, hist.tail(4)))

    # --- Upcoming earnings ---
    if upcoming:
        upcoming.sort(key=lambda x: x[0])
        lines.append("**Upcoming Earnings**")
        lines.append(f"{'Date':<14} {'Ticker':<8} {'Company':<30} {'Days':<7} {'EPS Est':>8}")
        lines.append("-" * 72)
        for dt, days_away, ticker, name, eps_est in upcoming:
            urgency = "⚠️ " if 0 <= days_away <= 14 else ""
            eps_str = f"{float(eps_est):+.2f}" if eps_est else "N/A"
            lines.append(
                f"{urgency}{str(dt):<14} {ticker:<8} {name:<30} "
                f"{days_away:>5}d   {eps_str:>8}"
            )
    else:
        lines.append("No upcoming earnings dates found for current positions.")

    # --- Recent results ---
    if results:
        lines.append("\n**Recent EPS Results (last 4 quarters)**")
        for ticker, name, hist in results:
            lines.append(f"\n  {ticker} — {name}")
            lines.append(f"  {'Quarter':<12} {'EPS Est':>10} {'EPS Actual':>12} {'Surprise':>10}")
            lines.append(f"  {'-'*48}")
            for date, row in hist.iterrows():
                date_str = str(date)[:10] if date else ""
                est    = row.get("epsEstimate")
                actual = row.get("epsActual")
                surp   = row.get("epsSurprise") or row.get("surprisePercent")
                est_s    = f"{float(est):+.3f}"    if est    is not None else "N/A"
                actual_s = f"{float(actual):+.3f}" if actual is not None else "N/A"
                surp_s   = f"{float(surp):+.2f}%"  if surp   is not None else "N/A"
                lines.append(f"  {date_str:<12} {est_s:>10} {actual_s:>12} {surp_s:>10}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sector Rotation Tracker
# ---------------------------------------------------------------------------

SECTOR_ETFS = {
    "Technology":        "XLK",
    "Energy":            "XLE",
    "Financials":        "XLF",
    "Health Care":       "XLV",
    "Industrials":       "XLI",
    "Materials":         "XLB",
    "Real Estate":       "XLRE",
    "Utilities":         "XLU",
    "Consumer Staples":  "XLP",
    "Consumer Discret.": "XLY",
    "Communication":     "XLC",
}


@mcp.tool()
async def get_sector_rotation() -> str:
    """Return performance and momentum for all 11 S&P 500 sectors (SPDR ETFs) vs the S&P 500.
    Shows 1-day, 1-week, 1-month, 3-month, and 1-year returns plus relative strength vs SPY.
    Use this to identify where institutional money is flowing and which sectors to favour or avoid."""

    symbols = list(SECTOR_ETFS.values()) + ["SPY"]

    def _fetch():
        return yf.download(symbols, period="1y", interval="1d", auto_adjust=True, progress=False)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching sector data: {e}"

    closes = data["Close"].dropna(how="all")
    if closes.empty:
        return "Could not retrieve sector data."

    def _ret(sym, days):
        col = closes[sym].dropna()
        if len(col) < days + 1:
            return None
        start = float(col.iloc[-(days + 1)])
        end   = float(col.iloc[-1])
        return (end - start) / start * 100 if start else None

    # Approximate trading days
    periods = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252}

    spy_rets = {label: _ret("SPY", days) for label, days in periods.items()}

    rows = []
    for name, sym in SECTOR_ETFS.items():
        rets = {label: _ret(sym, days) for label, days in periods.items()}
        # Relative strength vs SPY (1M momentum diff)
        rs_1m = (rets["1M"] - spy_rets["1M"]) if rets["1M"] is not None and spy_rets["1M"] is not None else None
        rows.append((name, sym, rets, rs_1m))

    # Sort by 1-month return descending (momentum ranking)
    rows.sort(key=lambda x: x[2]["1M"] if x[2]["1M"] is not None else -999, reverse=True)

    def _fs(v):
        if v is None:
            return "  N/A "
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}%"

    lines = [
        f"**Sector Rotation — {datetime.today().strftime('%d %b %Y')}**\n",
        f"{'Sector':<22} {'ETF':<6} {'1D':>7} {'1W':>7} {'1M':>7} {'3M':>7} {'1Y':>7} {'vs SPY 1M':>10}",
        "-" * 76,
    ]

    for name, sym, rets, rs_1m in rows:
        rs_str = _fs(rs_1m)
        trend = "▲" if rs_1m and rs_1m > 0 else "▼" if rs_1m and rs_1m < 0 else " "
        lines.append(
            f"{name:<22} {sym:<6} {_fs(rets['1D']):>7} {_fs(rets['1W']):>7} "
            f"{_fs(rets['1M']):>7} {_fs(rets['3M']):>7} {_fs(rets['1Y']):>7} "
            f"{trend}{rs_str:>9}"
        )

    # S&P 500 benchmark row
    lines.append("-" * 76)
    lines.append(
        f"{'S&P 500 (SPY)':<22} {'SPY':<6} {_fs(spy_rets['1D']):>7} {_fs(spy_rets['1W']):>7} "
        f"{_fs(spy_rets['1M']):>7} {_fs(spy_rets['3M']):>7} {_fs(spy_rets['1Y']):>7} {'benchmark':>10}"
    )

    # Flow interpretation
    leaders  = [r[0] for r in rows[:3] if r[3] and r[3] > 0]
    laggards = [r[0] for r in rows[-3:] if r[3] and r[3] < 0]
    lines.append("")
    if leaders:
        lines.append(f"**Inflow leaders (outperforming SPY):** {', '.join(leaders)}")
    if laggards:
        lines.append(f"**Outflow / underperforming:**           {', '.join(laggards)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stock Screener
# ---------------------------------------------------------------------------

# Curated liquid universe — top S&P 500 names + FTSE 100 blue chips
_SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","AVGO","JPM",
    "LLY","V","UNH","XOM","MA","JNJ","PG","COST","HD","MRK","ABBV","CVX",
    "KO","PEP","ADBE","WMT","BAC","CRM","ACN","MCD","TMO","CSCO","ABT","LIN",
    "NFLX","AMD","TXN","DHR","NKE","PM","NEE","INTC","ORCL","UPS","QCOM",
    "HON","LOW","UNP","RTX","AMGN","CAT","GS","SPGI","IBM","DE","AXP","BLK",
    "ISRG","NOW","INTU","SYK","GILD","LMT","PLD","ADI","REGN","MDLZ","CI",
    "MMC","ZTS","MO","DUK","SO","TGT","ETN","ITW","PGR","AON","CME","CL",
    "APD","SHW","EW","WM","NSC","GD","F","GM","PYPL","SBUX","PANW","SNOW",
    "CRWD","UBER","ABNB","COIN","PLTR","ARM",
]

_FTSE100_TICKERS = [
    "SHEL.L","AZN.L","HSBA.L","BP.L","ULVR.L","RIO.L","GSK.L","REL.L",
    "DGE.L","NG.L","LLOY.L","BARC.L","BT-A.L","VOD.L","STAN.L","RR.L",
    "EXPN.L","LSEG.L","CPG.L","IMB.L","MKS.L","TSCO.L","BATS.L","AAL.L",
    "ANTO.L","CCH.L","CNA.L","ENT.L","FRES.L","HLN.L","JMAT.L","KGF.L",
    "MNDI.L","NWG.L","PHNX.L","PRU.L","SGE.L","SMDS.L","SSE.L","WPP.L",
]

_WATCHLIST: list[str] = []  # populated at runtime from T212 positions


@mcp.tool()
async def screen_stocks(
    universe: str = "sp500",
    max_rsi: float = 40.0,
    min_rsi: float = 0.0,
    require_above_ma200: bool = True,
    require_above_ma50: bool = False,
    min_analyst_upside_pct: float = 10.0,
    max_pe: float = 0.0,
    min_momentum_1m_pct: float = 0.0,
    max_results: int = 20,
) -> str:
    """Scan a universe of stocks for high-conviction entry setups.

    universe: "sp500" | "ftse100" | "both" | "watchlist" | comma-separated tickers
    max_rsi: filter stocks with RSI below this (default 40 = oversold zone)
    min_rsi: filter stocks with RSI above this (default 0 = no floor)
    require_above_ma200: only include stocks trading above their 200-day MA
    require_above_ma50: only include stocks trading above their 50-day MA
    min_analyst_upside_pct: minimum analyst mean price target upside % (default 10)
    max_pe: maximum trailing P/E ratio — 0 means no filter
    min_momentum_1m_pct: minimum 1-month price return % — 0 means no filter
    max_results: cap on number of results returned (default 20)

    Example: screen_stocks(universe="sp500", max_rsi=35, require_above_ma200=True)
    = find oversold S&P 500 names still in long-term uptrends — classic mean-reversion entry.
    """

    # Build ticker list
    global _WATCHLIST
    if universe == "sp500":
        tickers = list(_SP500_TICKERS)
    elif universe == "ftse100":
        tickers = list(_FTSE100_TICKERS)
    elif universe == "both":
        tickers = list(_SP500_TICKERS) + list(_FTSE100_TICKERS)
    elif universe == "watchlist":
        try:
            positions = await t212.get_portfolio()
            tickers = [_strip_t212_ticker(p["ticker"]) for p in positions]
        except Exception:
            tickers = list(_WATCHLIST)
    else:
        tickers = [t.strip().upper() for t in universe.split(",") if t.strip()]

    if not tickers:
        return "No tickers to screen."

    # --- Step 1: batch price download (1 year daily) ---
    def _batch_download(syms):
        raw = yf.download(syms, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"]
        # Single ticker returns flat df
        return raw[["Close"]].rename(columns={"Close": syms[0]})

    try:
        closes = await asyncio.to_thread(_batch_download, tickers)
    except Exception as e:
        return f"Error downloading price data: {e}"

    closes = closes.dropna(how="all")
    available = [c for c in closes.columns if closes[c].notna().sum() >= 60]
    if not available:
        return "Insufficient price data for the requested universe."

    # --- Step 2: compute technicals per ticker in pandas (fast, no API calls) ---
    candidates = []

    for sym in available:
        col = closes[sym].dropna()
        if len(col) < 60:
            continue

        price = float(col.iloc[-1])

        # MAs
        ma50  = float(col.rolling(50).mean().iloc[-1])  if len(col) >= 50  else None
        ma200 = float(col.rolling(200).mean().iloc[-1]) if len(col) >= 200 else None

        # RSI (14)
        delta = col.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi   = 100 - (100 / (1 + rs)) if loss.iloc[-1] != 0 else 100

        # 1-month momentum
        mom_1m = (price - float(col.iloc[-22])) / float(col.iloc[-22]) * 100 if len(col) >= 22 else None

        # --- Apply technical filters ---
        if rsi < min_rsi or rsi > max_rsi:
            continue
        if require_above_ma200 and (ma200 is None or price < ma200):
            continue
        if require_above_ma50 and (ma50 is None or price < ma50):
            continue
        if min_momentum_1m_pct and (mom_1m is None or mom_1m < min_momentum_1m_pct):
            continue

        candidates.append({
            "ticker": sym,
            "price":  price,
            "rsi":    rsi,
            "ma50":   ma50,
            "ma200":  ma200,
            "mom_1m": mom_1m,
        })

    if not candidates:
        return (
            f"No stocks passed technical filters in '{universe}' universe.\n"
            f"Filters: RSI {min_rsi:.0f}–{max_rsi:.0f}, "
            f"{'above MA200, ' if require_above_ma200 else ''}"
            f"{'above MA50, ' if require_above_ma50 else ''}"
            f"{'momentum ≥ ' + str(min_momentum_1m_pct) + '%, ' if min_momentum_1m_pct else ''}"
            f"\nTry relaxing the criteria."
        )

    # --- Step 3: fetch fundamentals + analyst targets for technical candidates only ---
    async def _get_fundamentals(sym):
        def _fetch():
            info = yf.Ticker(sym).info
            return {
                "pe":        info.get("trailingPE"),
                "target":    info.get("targetMeanPrice"),
                "cur_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "rec":       info.get("recommendationKey", ""),
                "name":      info.get("shortName") or sym,
                "sector":    info.get("sector", ""),
            }
        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            return {}

    fund_results = await asyncio.gather(*[_get_fundamentals(c["ticker"]) for c in candidates])

    # Merge and apply fundamental filters
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
        return (
            f"Technical candidates found ({len(candidates)}) but none passed fundamental filters "
            f"(analyst upside ≥ {min_analyst_upside_pct}%, "
            f"{'P/E ≤ ' + str(max_pe) if max_pe else ''}).\n"
            f"Try lowering min_analyst_upside_pct or removing max_pe."
        )

    # Sort by RSI ascending (most oversold first)
    final.sort(key=lambda x: x["rsi"])
    final = final[:max_results]

    lines = [
        f"**Stock Screener — {universe.upper()} universe**\n",
        f"Filters: RSI {min_rsi:.0f}–{max_rsi:.0f}"
        + (", above MA200" if require_above_ma200 else "")
        + (", above MA50"  if require_above_ma50  else "")
        + (f", analyst upside ≥ {min_analyst_upside_pct}%" if min_analyst_upside_pct else "")
        + (f", P/E ≤ {max_pe}" if max_pe else "")
        + "\n",
        f"{'Ticker':<8} {'Name':<24} {'Price':>9} {'RSI':>6} {'1M%':>7} {'Upside%':>9} {'P/E':>7} {'Rec':<10} {'Sector'}",
        "-" * 100,
    ]

    for s in final:
        pe_str  = f"{s['pe']:.1f}" if s.get("pe") else "N/A"
        up_str  = f"+{s['upside']:.1f}%" if s.get("upside") else "N/A"
        mom_str = f"{s['mom_1m']:+.1f}%" if s.get("mom_1m") is not None else "N/A"
        rec_str = (s.get("rec") or "").upper()[:9]
        sector  = (s.get("sector") or "")[:18]
        lines.append(
            f"{s['ticker']:<8} {(s.get('name',''))[:23]:<24} {s['price']:>9,.2f} "
            f"{s['rsi']:>6.1f} {mom_str:>7} {up_str:>9} {pe_str:>7} {rec_str:<10} {sector}"
        )

    lines.append(f"\n{len(final)} candidates — sorted by RSI ascending (most oversold first)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deep Research via Perplexity
# ---------------------------------------------------------------------------

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


@mcp.tool()
async def research(query: str) -> str:
    """Deep research synthesis using Perplexity AI — returns a cited, multi-source answer.
    Better than web search for: earnings analysis, geopolitical context, sector deep-dives,
    analyst report synthesis, macro themes, or any question needing reasoned synthesis.
    Requires PERPLEXITY_API_KEY in .env.
    Examples: 'Why is the energy sector outperforming in Q2 2025?'
              'NVDA competitive moat vs AMD in AI inference chips'
              'Impact of US-China tariffs on semiconductor supply chain'"""

    if not PERPLEXITY_API_KEY:
        return (
            "PERPLEXITY_API_KEY not set. Add it to .env:\n"
            "  PERPLEXITY_API_KEY=pplx-xxxx\n"
            "Get a key at https://www.perplexity.ai/settings/api"
        )

    import httpx

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial research analyst. Provide concise, factual, "
                    "well-sourced answers. Include relevant data points, dates, and cite sources."
                ),
            },
            {"role": "user", "content": query},
        ],
        "max_tokens": 1024,
        "return_citations": True,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return f"Error calling Perplexity API: {e}"

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    citations = data.get("citations", [])

    lines = [f"**Research: {query}**\n", content]
    if citations:
        lines.append("\n**Sources**")
        for i, url in enumerate(citations[:6], 1):
            lines.append(f"{i}. {url}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Macro Economic Dashboard
# ---------------------------------------------------------------------------

@mcp.tool()
@_cached(_cache_macro)
async def get_macro_dashboard() -> str:
    """Return key macroeconomic indicators: interest rates, yield curve, inflation,
    GDP growth, unemployment, VIX fear index, and USD strength.
    Use this for macro context before making investment decisions."""

    lines = [f"**Macro Economic Dashboard — {datetime.today().strftime('%d %b %Y')}**\n"]

    # --- Market-based indicators via yfinance (always available) ---
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
                t = yf.Ticker(sym)
                fi = t.fast_info
                price = fi.last_price
                prev = fi.previous_close
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
                sign = "+" if chg >= 0 else ""
                lines.append(f"- {label}: {price:,.2f}  ({sign}{chg:.2f}%)")
            else:
                lines.append(f"- {label}: N/A")

        # Yield curve interpretation
        tnx = market.get("10Y Treasury (^TNX)", (None,))[0]
        if tnx is not None:
            lines.append(f"\n**Yield Curve**")
            lines.append(f"- 10Y yield: {tnx:.2f}%")

        vix = market.get("VIX Fear Index (^VIX)", (None,))[0]
        if vix is not None:
            if vix < 15:
                vix_signal = "LOW FEAR — complacency, potential for surprise vol"
            elif vix < 20:
                vix_signal = "NORMAL — market calm"
            elif vix < 30:
                vix_signal = "ELEVATED — uncertainty, caution warranted"
            else:
                vix_signal = "HIGH FEAR — panic selling, contrarian buy zone"
            lines.append(f"\n**VIX Signal:** {vix_signal}")

    # --- FRED economic data (if API key available) ---
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)

            indicators = {
                "Fed Funds Rate": "FEDFUNDS",
                "CPI YoY %": "CPIAUCSL",
                "Core CPI YoY %": "CPILFESL",
                "Unemployment Rate %": "UNRATE",
                "Real GDP Growth % (QoQ)": "A191RL1Q225SBEA",
                "Consumer Confidence": "UMCSENT",
                "10Y-2Y Spread (bp)": "T10Y2Y",
            }

            lines.append("\n**Economic Indicators (FRED)**")
            for label, series_id in indicators.items():
                try:
                    data = fred.get_series(series_id, observation_start=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"))
                    if data is not None and len(data) > 0:
                        val = float(data.dropna().iloc[-1])
                        lines.append(f"- {label}: {val:.2f}")
                except Exception:
                    lines.append(f"- {label}: N/A")

            # Yield curve inversion warning
            try:
                spread = fred.get_series("T10Y2Y")
                if spread is not None and len(spread) > 0:
                    latest_spread = float(spread.dropna().iloc[-1])
                    if latest_spread < 0:
                        lines.append(f"\n> ⚠️ **YIELD CURVE INVERTED** ({latest_spread:.2f}%) — historically precedes recession by 6-18 months")
                    elif latest_spread < 0.5:
                        lines.append(f"\n> ⚡ Yield curve flattening ({latest_spread:.2f}%) — watch for inversion")
            except Exception:
                pass

        except ImportError:
            lines.append("\n⚠ Install fredapi: `pip install fredapi`")
        except Exception as e:
            lines.append(f"\n⚠ FRED error: {e}")
    else:
        lines.append("\n💡 Set FRED_API_KEY in .env for economic indicators (CPI, GDP, unemployment). Free at https://fred.stlouisfed.org/docs/api/api_key.html")

    return "\n".join(lines)


@mcp.tool()
@_cached(_cache_prices)
async def get_fear_greed_index() -> str:
    """Return a composite Fear & Greed score (0-100) based on market breadth, momentum,
    VIX, and safe haven demand. 0=Extreme Fear, 100=Extreme Greed.
    Use this to gauge overall market sentiment before making buy/sell decisions."""

    def _fetch():
        data = {}
        # VIX
        try:
            data["vix"] = float(yf.Ticker("^VIX").fast_info.last_price)
        except Exception:
            data["vix"] = None

        # SPY momentum vs 125-day MA
        try:
            spy = yf.Ticker("SPY").history(period="6mo", interval="1d", auto_adjust=True)["Close"]
            data["spy_price"] = float(spy.iloc[-1])
            data["spy_ma125"] = float(spy.rolling(125).mean().iloc[-1])
        except Exception:
            data["spy_price"] = data["spy_ma125"] = None

        # Market breadth: % of SPY-correlated ETFs above their 50-day MA
        breadth_syms = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLP", "XLY", "XLC"]
        above_ma = 0
        total = 0
        for sym in breadth_syms:
            try:
                h = yf.Ticker(sym).history(period="3mo", interval="1d", auto_adjust=True)["Close"]
                if len(h) >= 50:
                    total += 1
                    if float(h.iloc[-1]) > float(h.rolling(50).mean().iloc[-1]):
                        above_ma += 1
            except Exception:
                pass
        data["breadth_pct"] = (above_ma / total * 100) if total > 0 else None

        # Safe haven demand: gold vs SPY 20-day relative performance
        try:
            gold = yf.Ticker("GLD").history(period="1mo", interval="1d", auto_adjust=True)["Close"]
            spy_h = yf.Ticker("SPY").history(period="1mo", interval="1d", auto_adjust=True)["Close"]
            gold_ret = (float(gold.iloc[-1]) - float(gold.iloc[0])) / float(gold.iloc[0]) * 100
            spy_ret = (float(spy_h.iloc[-1]) - float(spy_h.iloc[0])) / float(spy_h.iloc[0]) * 100
            data["safe_haven_diff"] = gold_ret - spy_ret
        except Exception:
            data["safe_haven_diff"] = None

        return data

    try:
        d = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error computing Fear & Greed Index: {e}"

    scores = []
    components = []

    # 1. VIX score (inverted: low VIX = greed)
    if d["vix"] is not None:
        vix = d["vix"]
        vix_score = max(0, min(100, 100 - (vix - 12) * (100 / 28)))
        scores.append(vix_score)
        label = "Extreme Fear" if vix > 30 else "Fear" if vix > 20 else "Neutral" if vix > 15 else "Greed" if vix > 12 else "Extreme Greed"
        components.append(f"- VIX ({vix:.1f}): **{label}** → score {vix_score:.0f}")

    # 2. Momentum: SPY vs 125-day MA
    if d["spy_price"] is not None and d["spy_ma125"] is not None:
        pct_above = (d["spy_price"] / d["spy_ma125"] - 1) * 100
        mom_score = max(0, min(100, 50 + pct_above * 5))
        scores.append(mom_score)
        label = "Greedy" if pct_above > 5 else "Fearful" if pct_above < -5 else "Neutral"
        components.append(f"- Momentum (SPY {pct_above:+.1f}% vs MA125): **{label}** → score {mom_score:.0f}")

    # 3. Market breadth
    if d["breadth_pct"] is not None:
        breadth_score = d["breadth_pct"]
        scores.append(breadth_score)
        label = "Strong" if breadth_score > 70 else "Weak" if breadth_score < 30 else "Mixed"
        components.append(f"- Breadth ({breadth_score:.0f}% sectors above MA50): **{label}** → score {breadth_score:.0f}")

    # 4. Safe haven demand (inverted: gold outperforming = fear)
    if d["safe_haven_diff"] is not None:
        sh = d["safe_haven_diff"]
        sh_score = max(0, min(100, 50 - sh * 5))
        scores.append(sh_score)
        label = "Fear (gold leading)" if sh > 2 else "Greed (risk-on)" if sh < -2 else "Neutral"
        components.append(f"- Safe Haven (Gold vs SPY 1M: {sh:+.1f}%): **{label}** → score {sh_score:.0f}")

    if not scores:
        return "Could not compute Fear & Greed Index — insufficient data."

    composite = sum(scores) / len(scores)

    if composite <= 20:
        reading = "🔴 EXTREME FEAR"
    elif composite <= 40:
        reading = "🟠 FEAR"
    elif composite <= 60:
        reading = "🟡 NEUTRAL"
    elif composite <= 80:
        reading = "🟢 GREED"
    else:
        reading = "🟢 EXTREME GREED"

    lines = [
        f"**Fear & Greed Index — {datetime.today().strftime('%d %b %Y')}**\n",
        f"## {reading} — Score: {composite:.0f}/100\n",
        "**Components:**",
    ] + components + [
        "",
        "**Interpretation:**",
        "- 0-25: Extreme Fear — contrarian buy opportunities",
        "- 25-45: Fear — cautious positioning",
        "- 45-55: Neutral",
        "- 55-75: Greed — momentum favours bulls",
        "- 75-100: Extreme Greed — caution, potential top",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Financial Statements
# ---------------------------------------------------------------------------

@mcp.tool()
@_cached(_cache_fundamentals)
async def get_financial_statements(ticker: str) -> str:
    """Return income statement, balance sheet, and cash flow highlights for a stock.
    Shows last 4 annual periods with key ratios: ROE, ROA, current ratio, D/E, FCF yield.
    Use this for deep fundamental analysis beyond the basics."""

    def _fetch():
        t = yf.Ticker(ticker)
        return {
            "info": t.info,
            "income": t.income_stmt,
            "balance": t.balance_sheet,
            "cashflow": t.cashflow,
        }

    try:
        d = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching financial statements for {ticker}: {e}"

    info = d["info"]
    name = info.get("longName") or ticker.upper()
    currency = info.get("currency", "USD")

    def _b(val):
        """Format large numbers as billions/millions."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        v = float(val)
        if abs(v) >= 1e9:
            return f"{v/1e9:,.1f}B"
        if abs(v) >= 1e6:
            return f"{v/1e6:,.0f}M"
        return f"{v:,.0f}"

    lines = [f"**{name} ({ticker.upper()}) — Financial Statements ({currency})**\n"]

    # --- Income Statement ---
    inc = d["income"]
    if inc is not None and not inc.empty:
        lines.append("**Income Statement (Annual)**")
        rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA"]
        header = f"{'Metric':<22}" + "".join(f"{c.strftime('%Y'):>12}" for c in inc.columns[:4])
        lines.append(header)
        lines.append("-" * (22 + 12 * min(4, len(inc.columns))))
        for row in rows:
            if row in inc.index:
                vals = "".join(f"{_b(inc.loc[row, c]):>12}" for c in inc.columns[:4])
                lines.append(f"{row:<22}{vals}")
        lines.append("")

    # --- Balance Sheet ---
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
                vals = "".join(f"{_b(bal.loc[row, c]):>12}" for c in bal.columns[:4])
                display = row[:39]
                lines.append(f"{display:<40}{vals}")
        lines.append("")

    # --- Cash Flow ---
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
                vals = "".join(f"{_b(cf.loc[row, c]):>12}" for c in cf.columns[:4])
                lines.append(f"{row:<30}{vals}")
        lines.append("")

    # --- Key Ratios ---
    lines.append("**Key Ratios**")
    try:
        if bal is not None and not bal.empty and inc is not None and not inc.empty:
            latest_bal = bal.iloc[:, 0]
            latest_inc = inc.iloc[:, 0]
            equity = float(latest_bal.get("Stockholders Equity", 0) or 0)
            assets = float(latest_bal.get("Total Assets", 0) or 0)
            net_income = float(latest_inc.get("Net Income", 0) or 0)
            total_debt = float(latest_bal.get("Total Debt", 0) or 0)
            current_assets = float(latest_bal.get("Current Assets", 0) or 0)
            current_liab = float(latest_bal.get("Current Liabilities", 0) or 0)

            roe = (net_income / equity * 100) if equity else None
            roa = (net_income / assets * 100) if assets else None
            de = (total_debt / equity) if equity else None
            current_ratio = (current_assets / current_liab) if current_liab else None

            lines.append(f"- ROE: {roe:.1f}%" if roe else "- ROE: N/A")
            lines.append(f"- ROA: {roa:.1f}%" if roa else "- ROA: N/A")
            lines.append(f"- Debt/Equity: {de:.2f}" if de else "- Debt/Equity: N/A")
            lines.append(f"- Current Ratio: {current_ratio:.2f}" if current_ratio else "- Current Ratio: N/A")

        if cf is not None and not cf.empty:
            fcf = float(cf.iloc[:, 0].get("Free Cash Flow", 0) or 0)
            mcap = float(info.get("marketCap", 0) or 0)
            fcf_yield = (fcf / mcap * 100) if mcap else None
            lines.append(f"- FCF Yield: {fcf_yield:.2f}%" if fcf_yield else "- FCF Yield: N/A")
    except Exception:
        lines.append("- Could not compute ratios")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DCF Valuation
# ---------------------------------------------------------------------------

@mcp.tool()
@_cached(_cache_fundamentals)
async def get_dcf_valuation(ticker: str, growth_rate_pct: float = 0.0, discount_rate_pct: float = 10.0) -> str:
    """Estimate intrinsic value using a Discounted Cash Flow (DCF) model.
    Uses free cash flow from financials, analyst growth estimates, and WACC.
    growth_rate_pct: override FCF growth rate (0 = auto-detect from analyst estimates)
    discount_rate_pct: discount rate / WACC (default 10%)
    Returns intrinsic value per share and margin of safety vs current price."""

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
        return f"Insufficient data for DCF on {ticker} (missing price or shares outstanding)."

    # Get FCF
    if cf is None or cf.empty or "Free Cash Flow" not in cf.index:
        return f"No free cash flow data available for {ticker}."

    fcf_values = [float(v) for v in cf.loc["Free Cash Flow"].dropna().values[:4] if not np.isnan(float(v))]
    if not fcf_values:
        return f"No valid FCF data for {ticker}."

    latest_fcf = fcf_values[0]
    if latest_fcf <= 0:
        return f"{ticker} has negative FCF ({currency} {latest_fcf/1e9:.2f}B) — DCF not applicable for negative cash flow companies."

    # Growth rate
    if growth_rate_pct == 0:
        analyst_growth = info.get("earningsGrowth") or info.get("revenueGrowth")
        if analyst_growth:
            growth_rate = float(analyst_growth)
        elif len(fcf_values) >= 2 and fcf_values[-1] > 0:
            growth_rate = (fcf_values[0] / fcf_values[-1]) ** (1 / (len(fcf_values) - 1)) - 1
        else:
            growth_rate = 0.05
    else:
        growth_rate = growth_rate_pct / 100

    discount_rate = discount_rate_pct / 100
    terminal_growth = 0.025  # 2.5% long-term growth
    projection_years = 10

    # Project FCFs
    projected_fcfs = []
    fcf = latest_fcf
    for year in range(1, projection_years + 1):
        fcf *= (1 + growth_rate)
        pv = fcf / (1 + discount_rate) ** year
        projected_fcfs.append((year, fcf, pv))

    # Terminal value
    terminal_fcf = projected_fcfs[-1][1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

    # Enterprise value and equity value
    total_pv_fcf = sum(pv for _, _, pv in projected_fcfs)
    enterprise_value = total_pv_fcf + pv_terminal

    # Adjust for cash and debt
    cash = float(info.get("totalCash", 0) or 0)
    debt = float(info.get("totalDebt", 0) or 0)
    equity_value = enterprise_value + cash - debt
    intrinsic_per_share = equity_value / shares

    margin_of_safety = (intrinsic_per_share - current_price) / intrinsic_per_share * 100

    def _b(v):
        return f"{v/1e9:,.2f}B" if abs(v) >= 1e9 else f"{v/1e6:,.0f}M"

    lines = [
        f"**{name} ({ticker.upper()}) — DCF Valuation**\n",
        f"**Inputs**",
        f"- Latest FCF: {currency} {_b(latest_fcf)}",
        f"- Growth rate: {growth_rate*100:.1f}% (years 1-{projection_years})",
        f"- Discount rate (WACC): {discount_rate*100:.1f}%",
        f"- Terminal growth: {terminal_growth*100:.1f}%",
        f"- Projection period: {projection_years} years",
        "",
        f"**Projected FCF**",
    ]

    for yr, fcf_val, pv_val in projected_fcfs[:5]:
        lines.append(f"  Year {yr}: FCF {_b(fcf_val)}  →  PV {_b(pv_val)}")
    if projection_years > 5:
        lines.append(f"  ... (years 6-{projection_years} omitted)")

    verdict = "UNDERVALUED ✅" if margin_of_safety > 15 else "FAIRLY VALUED ⚖️" if margin_of_safety > -10 else "OVERVALUED ⚠️"

    lines += [
        "",
        f"**Valuation**",
        f"- PV of projected FCFs: {currency} {_b(total_pv_fcf)}",
        f"- PV of terminal value: {currency} {_b(pv_terminal)}",
        f"- Enterprise value:     {currency} {_b(enterprise_value)}",
        f"- + Cash:               {currency} {_b(cash)}",
        f"- - Debt:               {currency} {_b(debt)}",
        f"- **Equity value:       {currency} {_b(equity_value)}**",
        "",
        f"**Result**",
        f"- Intrinsic value/share: **{currency} {intrinsic_per_share:,.2f}**",
        f"- Current price:         {currency} {current_price:,.2f}",
        f"- Margin of safety:      {margin_of_safety:+.1f}%",
        f"- Verdict:               **{verdict}**",
        "",
        f"⚠️ DCF is highly sensitive to growth and discount rate assumptions. Use as one input among many.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Peer Comparison
# ---------------------------------------------------------------------------

@mcp.tool()
@_cached(_cache_fundamentals)
async def compare_peers(tickers: str) -> str:
    """Compare 2-6 stocks side-by-side on fundamentals, technicals, and analyst views.
    tickers: comma-separated list, e.g. 'AAPL,MSFT,GOOGL'
    Returns valuation, growth, margins, momentum, and analyst consensus comparison."""

    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(symbols) < 2:
        return "Provide at least 2 tickers separated by commas, e.g. 'AAPL,MSFT,GOOGL'"
    symbols = symbols[:6]

    async def _get_info(sym):
        def _f():
            t = yf.Ticker(sym)
            info = t.info
            hist = t.history(period="1y", interval="1d", auto_adjust=True)
            return info, hist
        try:
            return sym, *(await asyncio.to_thread(_f))
        except Exception:
            return sym, {}, pd.DataFrame()

    results = await asyncio.gather(*[_get_info(s) for s in symbols])

    # Build comparison table
    metrics = [
        ("Name", lambda i, h: (i.get("shortName") or "")[:20]),
        ("Sector", lambda i, h: (i.get("sector") or "")[:15]),
        ("Mkt Cap", lambda i, h: f"{float(i.get('marketCap',0) or 0)/1e9:.0f}B" if i.get("marketCap") else "N/A"),
        ("P/E (TTM)", lambda i, h: f"{float(i['trailingPE']):.1f}" if i.get("trailingPE") else "N/A"),
        ("P/E (Fwd)", lambda i, h: f"{float(i['forwardPE']):.1f}" if i.get("forwardPE") else "N/A"),
        ("EV/EBITDA", lambda i, h: f"{float(i['enterpriseToEbitda']):.1f}" if i.get("enterpriseToEbitda") else "N/A"),
        ("P/B", lambda i, h: f"{float(i['priceToBook']):.1f}" if i.get("priceToBook") else "N/A"),
        ("Rev Growth", lambda i, h: f"{float(i['revenueGrowth'])*100:.1f}%" if i.get("revenueGrowth") else "N/A"),
        ("Profit Margin", lambda i, h: f"{float(i['profitMargins'])*100:.1f}%" if i.get("profitMargins") else "N/A"),
        ("ROE", lambda i, h: f"{float(i['returnOnEquity'])*100:.1f}%" if i.get("returnOnEquity") else "N/A"),
        ("Div Yield", lambda i, h: f"{float(i['dividendYield'])*100:.2f}%" if i.get("dividendYield") else "N/A"),
        ("Beta", lambda i, h: f"{float(i['beta']):.2f}" if i.get("beta") else "N/A"),
        ("52w Chg", lambda i, h: f"{float(i['52WeekChange'])*100:+.1f}%" if i.get("52WeekChange") else "N/A"),
        ("Analyst", lambda i, h: (i.get("recommendationKey") or "N/A").upper()),
        ("Target Upside", lambda i, h: (
            f"{(float(i['targetMeanPrice']) - float(i.get('currentPrice', i.get('regularMarketPrice', 0)) or 1)) / float(i.get('currentPrice', i.get('regularMarketPrice', 0)) or 1) * 100:+.1f}%"
            if i.get("targetMeanPrice") and (i.get("currentPrice") or i.get("regularMarketPrice")) else "N/A"
        )),
    ]

    col_w = 14
    lines = [f"**Peer Comparison**\n"]
    header = f"{'Metric':<16}" + "".join(f"{sym:>{col_w}}" for sym, _, _ in results)
    lines.append(header)
    lines.append("-" * (16 + col_w * len(results)))

    for metric_name, extractor in metrics:
        row = f"{metric_name:<16}"
        for sym, info, hist in results:
            try:
                val = extractor(info, hist)
            except Exception:
                val = "N/A"
            row += f"{val:>{col_w}}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

@mcp.tool()
async def calculate_position_size(
    ticker: str,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 2.0,
) -> str:
    """Calculate optimal position size for a trade based on risk management rules.
    ticker: stock ticker
    entry_price: planned entry price
    stop_loss_price: stop-loss price level
    risk_pct: max portfolio risk per trade (default 2%)
    Returns position size, number of shares, Kelly Criterion estimate, and concentration impact."""

    try:
        acct = await t212.get_account_summary()
    except Exception as e:
        return f"Error fetching account: {e}"

    total_value = float(acct.get("totalValue", 0) or 0)
    if total_value <= 0:
        return "Could not determine portfolio value."

    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0:
        return "Stop loss must be different from entry price."

    max_risk_amount = total_value * (risk_pct / 100)
    shares = int(max_risk_amount / risk_per_share)
    position_value = shares * entry_price
    position_pct = (position_value / total_value) * 100

    # Kelly Criterion estimate using analyst win rate proxy
    def _fetch_info():
        return yf.Ticker(ticker).info

    try:
        info = await asyncio.to_thread(_fetch_info)
        target = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice") or entry_price
        if target and current:
            win_amount = abs(float(target) - entry_price)
            loss_amount = risk_per_share
            # Estimate win probability from analyst consensus
            rec = (info.get("recommendationKey") or "").lower()
            win_prob = {"strong_buy": 0.65, "buy": 0.6, "hold": 0.5, "sell": 0.4, "strong_sell": 0.35}.get(rec, 0.5)
            b = win_amount / loss_amount if loss_amount > 0 else 1
            kelly = win_prob - ((1 - win_prob) / b) if b > 0 else 0
            kelly = max(0, kelly)
        else:
            kelly = None
            win_prob = None
    except Exception:
        kelly = None
        win_prob = None

    currency = acct.get("currency", "")

    lines = [
        f"**Position Size Calculator — {ticker.upper()}**\n",
        f"**Trade Setup**",
        f"- Entry price:      {currency} {entry_price:,.2f}",
        f"- Stop loss:        {currency} {stop_loss_price:,.2f}",
        f"- Risk per share:   {currency} {risk_per_share:,.2f}",
        f"- Direction:        {'LONG' if entry_price > stop_loss_price else 'SHORT'}",
        "",
        f"**Risk Management**",
        f"- Portfolio value:  {currency} {total_value:,.2f}",
        f"- Max risk ({risk_pct}%):  {currency} {max_risk_amount:,.2f}",
        f"- **Shares to buy:  {shares:,}**",
        f"- Position value:   {currency} {position_value:,.2f}",
        f"- Portfolio weight:  {position_pct:.1f}%",
    ]

    if kelly is not None:
        kelly_shares = int(kelly * total_value / entry_price)
        lines += [
            "",
            f"**Kelly Criterion**",
            f"- Win probability: {win_prob*100:.0f}% (from analyst consensus)",
            f"- Kelly fraction:  {kelly*100:.1f}%",
            f"- Kelly position:  {kelly_shares:,} shares ({currency} {kelly_shares * entry_price:,.2f})",
            f"- Recommended:     Use **half-Kelly** ({kelly_shares // 2:,} shares) for safety",
        ]

    # Concentration warning
    if position_pct > 20:
        lines.append(f"\n⚠️ **CONCENTRATION WARNING**: {position_pct:.1f}% is very high. Consider reducing to <10%.")
    elif position_pct > 10:
        lines.append(f"\n⚡ Position is {position_pct:.1f}% of portfolio — moderate concentration.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Stress Test
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio_stress_test(simulations: int = 1000) -> str:
    """Run stress tests on your portfolio: historical crisis scenarios and Monte Carlo simulation.
    simulations: number of Monte Carlo runs (default 1000)
    Returns expected drawdowns under 2008 crisis, COVID crash, 2022 rate hikes, and probability distribution."""

    try:
        positions = await t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions to stress test."

    tickers = [_strip_t212_ticker(p["ticker"]) for p in positions]
    weights_raw = {_strip_t212_ticker(p["ticker"]): float(p.get("currentPrice", 1) or 1) * float(p.get("quantity", 0) or 0) for p in positions}
    total_val = sum(weights_raw.values()) or 1
    weights = {k: v / total_val for k, v in weights_raw.items()}

    def _fetch():
        return yf.download(tickers, period="5y", interval="1d", auto_adjust=True, progress=False)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching price history: {e}"

    closes = data["Close"].dropna(how="all") if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
    returns = closes.pct_change().dropna()

    lines = ["**Portfolio Stress Test**\n"]

    # --- Historical scenarios ---
    scenarios = {
        "2008 Financial Crisis": ("2008-09-01", "2009-03-09"),
        "COVID Crash (2020)": ("2020-02-19", "2020-03-23"),
        "2022 Rate Hike Selloff": ("2022-01-03", "2022-10-12"),
        "2024 Aug VIX Spike": ("2024-07-16", "2024-08-05"),
    }

    lines.append("**Historical Scenario Analysis**")
    lines.append(f"{'Scenario':<28} {'Est. Portfolio Impact':>22}")
    lines.append("-" * 52)

    for scenario_name, (start, end) in scenarios.items():
        try:
            period_returns = returns.loc[start:end]
            if period_returns.empty:
                lines.append(f"{scenario_name:<28} {'no data':>22}")
                continue
            port_ret = sum(period_returns.get(t, pd.Series([0])).sum() * weights.get(t, 0) for t in tickers)
            lines.append(f"{scenario_name:<28} {port_ret*100:>+20.1f}%")
        except Exception:
            lines.append(f"{scenario_name:<28} {'N/A':>22}")

    # --- Monte Carlo simulation ---
    lines.append(f"\n**Monte Carlo Simulation ({simulations:,} runs, 1-year horizon)**")

    available_tickers = [t for t in tickers if t in returns.columns]
    if not available_tickers:
        lines.append("Insufficient data for Monte Carlo simulation.")
        return "\n".join(lines)

    port_returns = sum(returns[t] * weights.get(t, 0) for t in available_tickers).dropna()

    if len(port_returns) < 20:
        lines.append("Insufficient return history for Monte Carlo.")
        return "\n".join(lines)

    mean_ret = float(port_returns.mean())
    std_ret = float(port_returns.std())

    np.random.seed(42)
    sim_results = []
    for _ in range(simulations):
        daily_returns = np.random.normal(mean_ret, std_ret, 252)
        cumulative = np.cumprod(1 + daily_returns)
        sim_results.append(float(cumulative[-1] - 1))

    sim_results.sort()
    p5 = sim_results[int(0.05 * simulations)]
    p25 = sim_results[int(0.25 * simulations)]
    p50 = sim_results[int(0.50 * simulations)]
    p75 = sim_results[int(0.75 * simulations)]
    p95 = sim_results[int(0.95 * simulations)]

    lines += [
        f"- 5th percentile (worst case):  {p5*100:+.1f}%",
        f"- 25th percentile:              {p25*100:+.1f}%",
        f"- **Median outcome:             {p50*100:+.1f}%**",
        f"- 75th percentile:              {p75*100:+.1f}%",
        f"- 95th percentile (best case):  {p95*100:+.1f}%",
        "",
        f"- Probability of loss: {sum(1 for r in sim_results if r < 0) / simulations * 100:.1f}%",
        f"- Probability of >20% gain: {sum(1 for r in sim_results if r > 0.2) / simulations * 100:.1f}%",
        f"- Probability of >20% loss: {sum(1 for r in sim_results if r < -0.2) / simulations * 100:.1f}%",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Insider Trades
# ---------------------------------------------------------------------------

@mcp.tool()
@_cached(_cache_fundamentals)
async def get_insider_trades(ticker: str) -> str:
    """Return recent insider buy/sell activity for a stock via Finnhub.
    Insider buying is one of the most reliable bullish signals.
    Use this to check whether company insiders are putting their own money in."""

    def _fetch():
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        today = datetime.today().strftime("%Y-%m-%d")
        year_ago = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        return client.stock_insider_transactions(ticker.upper(), year_ago, today)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching insider trades for {ticker}: {e}"

    transactions = data.get("data", []) if isinstance(data, dict) else data
    if not transactions:
        return f"No insider transactions found for {ticker} in the past year."

    lines = [
        f"**Insider Trades — {ticker.upper()} (last 12 months)**\n",
        f"{'Date':<14} {'Name':<24} {'Type':<8} {'Shares':>12} {'Price':>10} {'Value':>14}",
        "-" * 86,
    ]

    total_buy_value = 0
    total_sell_value = 0

    for t in transactions[:20]:
        name = (t.get("name") or "Unknown")[:23]
        tx_type = t.get("transactionType", "")
        shares = float(t.get("share", 0) or 0)
        price = float(t.get("transactionPrice", 0) or 0)
        value = abs(shares * price)
        date_str = t.get("filingDate", "")[:10]

        if "purchase" in tx_type.lower() or "buy" in tx_type.lower() or "acquisition" in tx_type.lower():
            side = "BUY ✅"
            total_buy_value += value
        elif "sale" in tx_type.lower() or "sell" in tx_type.lower() or "disposition" in tx_type.lower():
            side = "SELL"
            total_sell_value += value
        else:
            side = tx_type[:7]

        lines.append(
            f"{date_str:<14} {name:<24} {side:<8} {shares:>12,.0f} {price:>10,.2f} {value:>14,.0f}"
        )

    lines.append("-" * 86)
    lines.append(f"Total insider buying:  ${total_buy_value:>14,.0f}")
    lines.append(f"Total insider selling: ${total_sell_value:>14,.0f}")

    ratio = total_buy_value / total_sell_value if total_sell_value > 0 else float("inf")
    if ratio > 2:
        lines.append(f"\n✅ **STRONG INSIDER BUYING** — buy/sell ratio: {ratio:.1f}x")
    elif ratio > 1:
        lines.append(f"\n🟢 Net insider buying — ratio: {ratio:.1f}x")
    elif ratio > 0.5:
        lines.append(f"\n🟡 Mixed insider activity — ratio: {ratio:.1f}x")
    else:
        lines.append(f"\n🔴 Heavy insider selling — ratio: {ratio:.2f}x")

    return "\n".join(lines)




# ---------------------------------------------------------------------------
# Portfolio Allocation
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio_allocation() -> str:
    """Analyse your portfolio allocation by sector, geography, and market cap.
    Shows concentration metrics and diversification score.
    Use this for portfolio health checks and rebalancing decisions."""

    try:
        positions = await t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions to analyse."

    # Calculate position values
    holdings = []
    for p in positions:
        ticker = _strip_t212_ticker(p["ticker"])
        qty = float(p.get("quantity", 0) or 0)
        price = float(p.get("currentPrice", 0) or 0)
        value = qty * price
        holdings.append({"ticker": ticker, "value": value})

    total_value = sum(h["value"] for h in holdings) or 1

    # Fetch sector/country/cap info for each position
    async def _get_meta(ticker):
        def _f():
            return yf.Ticker(ticker).info
        try:
            info = await asyncio.to_thread(_f)
            return {
                "sector": info.get("sector", "Unknown"),
                "country": info.get("country", "Unknown"),
                "marketCap": float(info.get("marketCap", 0) or 0),
                "name": info.get("shortName") or ticker,
            }
        except Exception:
            return {"sector": "Unknown", "country": "Unknown", "marketCap": 0, "name": ticker}

    metas = await asyncio.gather(*[_get_meta(h["ticker"]) for h in holdings])

    # Merge
    for h, m in zip(holdings, metas):
        h.update(m)

    lines = ["**Portfolio Allocation Analysis**\n"]

    # --- Top holdings ---
    holdings.sort(key=lambda x: x["value"], reverse=True)
    lines.append("**Top Holdings**")
    lines.append(f"{'#':<4} {'Ticker':<10} {'Name':<24} {'Value':>12} {'Weight':>8}")
    lines.append("-" * 62)
    for i, h in enumerate(holdings[:10], 1):
        w = h["value"] / total_value * 100
        lines.append(f"{i:<4} {h['ticker']:<10} {h['name'][:23]:<24} {h['value']:>12,.2f} {w:>7.1f}%")

    # --- Sector allocation ---
    sectors = {}
    for h in holdings:
        s = h.get("sector", "Unknown")
        sectors[s] = sectors.get(s, 0) + h["value"]
    sectors_sorted = sorted(sectors.items(), key=lambda x: x[1], reverse=True)

    lines.append(f"\n**Sector Allocation**")
    for sector, val in sectors_sorted:
        w = val / total_value * 100
        bar = "█" * int(w / 2)
        lines.append(f"  {sector:<22} {w:>5.1f}%  {bar}")

    # --- Geographic allocation ---
    countries = {}
    for h in holdings:
        c = h.get("country", "Unknown")
        countries[c] = countries.get(c, 0) + h["value"]
    countries_sorted = sorted(countries.items(), key=lambda x: x[1], reverse=True)

    lines.append(f"\n**Geographic Allocation**")
    for country, val in countries_sorted:
        w = val / total_value * 100
        lines.append(f"  {country:<22} {w:>5.1f}%")

    # --- Market cap allocation ---
    cap_buckets = {"Mega (>$200B)": 0, "Large ($10-200B)": 0, "Mid ($2-10B)": 0, "Small (<$2B)": 0}
    for h in holdings:
        mc = h.get("marketCap", 0)
        if mc >= 200e9:
            cap_buckets["Mega (>$200B)"] += h["value"]
        elif mc >= 10e9:
            cap_buckets["Large ($10-200B)"] += h["value"]
        elif mc >= 2e9:
            cap_buckets["Mid ($2-10B)"] += h["value"]
        else:
            cap_buckets["Small (<$2B)"] += h["value"]

    lines.append(f"\n**Market Cap Allocation**")
    for bucket, val in cap_buckets.items():
        w = val / total_value * 100
        if w > 0:
            lines.append(f"  {bucket:<22} {w:>5.1f}%")

    # --- Concentration metrics ---
    weights = [h["value"] / total_value for h in holdings]
    hhi = sum(w ** 2 for w in weights)  # Herfindahl-Hirschman Index
    top5_weight = sum(w for w in sorted(weights, reverse=True)[:5]) * 100

    lines.append(f"\n**Concentration Metrics**")
    lines.append(f"- Total positions: {len(holdings)}")
    lines.append(f"- Top 5 weight: {top5_weight:.1f}%")
    lines.append(f"- HHI (concentration): {hhi:.4f}  ({'concentrated' if hhi > 0.15 else 'moderate' if hhi > 0.08 else 'well diversified'})")
    lines.append(f"- Effective positions: {1/hhi:.1f}  (HHI-equivalent equal-weight count)")

    if top5_weight > 70:
        lines.append(f"\n⚠️ **Heavy concentration** — top 5 positions are {top5_weight:.0f}% of portfolio")
    if len(set(h.get("sector") for h in holdings)) <= 2:
        lines.append(f"\n⚠️ **Sector concentration** — only {len(set(h.get('sector') for h in holdings))} sectors represented")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        path="/mcp",
        stateless_http=True,
    )
