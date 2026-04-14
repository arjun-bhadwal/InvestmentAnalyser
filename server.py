import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import finnhub
import yfinance as yf
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from fastmcp import FastMCP

from t212_client import T212Client

load_dotenv()

T212_API_KEY = os.environ["T212_API_KEY"]
T212_API_SECRET = os.environ["T212_API_SECRET"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
T212_MODE = os.environ.get("T212_MODE", "demo")

t212: T212Client


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
             f"{'Date':<14} {'Ticker':<10} {'Side':<6} {'Qty':>10} {'Price':>10} {'Value':>10}",
             "-" * 64]

    for o in orders:
        ticker = _strip_t212_ticker(o.get("ticker", "?"))
        qty = o.get("filledQuantity") or o.get("quantity", 0)
        price = o.get("filledPrice") or o.get("limitPrice") or 0
        value = float(qty or 0) * float(price or 0)
        order_type = o.get("type", "")
        side = "BUY" if "BUY" in order_type.upper() else "SELL" if "SELL" in order_type.upper() else order_type[:4]
        raw_date = o.get("dateCreated") or o.get("dateModified") or ""
        try:
            date_str = datetime.fromisoformat(raw_date[:19]).strftime("%d %b %Y")
        except Exception:
            date_str = raw_date[:10]
        lines.append(
            f"{date_str:<14} {ticker:<10} {side:<6} {_fmt_float(qty, 4):>10} "
            f"{_fmt_float(price):>10} {_fmt_float(value):>10}"
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
        client = finnhub.Client(api_key=FINNHUB_API_KEY)
        recs = client.recommendation_trends(ticker.upper())
        target = client.price_target(ticker.upper())
        return recs, target

    try:
        recs, target = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching analyst data for {ticker}: {e}"

    lines = [f"**Analyst Ratings — {ticker.upper()}**\n"]

    # Price target
    if target:
        mean_t = target.get("targetMean")
        high_t = target.get("targetHigh")
        low_t = target.get("targetLow")
        n_analysts = target.get("targetNumberOfAnalysts") or target.get("numberOfAnalysts", "?")
        lines.append(f"**Price Targets ({n_analysts} analysts)**")
        lines.append(f"- Mean target:  {_fmt_float(mean_t)}")
        lines.append(f"- High target:  {_fmt_float(high_t)}")
        lines.append(f"- Low target:   {_fmt_float(low_t)}")
        lines.append("")

    # Recommendation trends (most recent 3 periods)
    if recs:
        lines.append(f"**Recommendation Trends**")
        lines.append(f"{'Period':<12} {'Strong Buy':>11} {'Buy':>6} {'Hold':>6} {'Sell':>6} {'Strong Sell':>12}")
        lines.append("-" * 57)
        for r in recs[:3]:
            period = r.get("period", "")[:7]
            lines.append(
                f"{period:<12} {r.get('strongBuy', 0):>11} {r.get('buy', 0):>6} "
                f"{r.get('hold', 0):>6} {r.get('sell', 0):>6} {r.get('strongSell', 0):>12}"
            )

        # Summarise latest
        if recs:
            latest = recs[0]
            total = sum(latest.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
            bullish = latest.get("strongBuy", 0) + latest.get("buy", 0)
            bearish = latest.get("sell", 0) + latest.get("strongSell", 0)
            if total:
                lines.append(f"\nLatest consensus: {bullish/total*100:.0f}% bullish, "
                             f"{latest.get('hold',0)/total*100:.0f}% hold, "
                             f"{bearish/total*100:.0f}% bearish ({total} analysts)")

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
# Order Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def place_order(ticker: str, side: str, quantity: float | None = None, value: float | None = None) -> str:
    """Place a market buy or sell order on your Trading 212 account.
    - ticker: stock symbol, e.g. AAPL, TSLA, LLOY
    - side: 'buy' or 'sell'
    - quantity: number of shares (fractional ok), OR
    - value: amount in account currency (e.g. 100 = £100 worth)
    Only provide one of quantity or value.
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        return "Error: side must be 'buy' or 'sell'."
    if quantity is None and value is None:
        return "Error: provide either quantity or value."
    if quantity is not None and value is not None:
        return "Error: provide either quantity or value, not both."

    try:
        instrument = await t212.find_instrument(ticker)
    except Exception as e:
        return f"Error looking up instrument '{ticker}': {e}"

    if instrument is None:
        return f"Instrument '{ticker}' not found on Trading 212."

    t212_ticker = instrument["ticker"]
    name = instrument.get("name", ticker)

    if side == "sell":
        if quantity is not None:
            quantity = -abs(quantity)
        elif value is not None:
            value = -abs(value)

    try:
        order = await t212.place_market_order(t212_ticker, quantity=quantity, value=value)
    except Exception as e:
        return f"Error placing order: {e}"

    order_id = order.get("id", "?")
    status = order.get("status", "?")
    filled_qty = order.get("filledQuantity") or order.get("quantity", "?")
    filled_val = order.get("filledValue") or order.get("value", "?")

    return (
        f"**Order placed — {side.upper()} {name} ({ticker})**\n\n"
        f"- T212 ticker: {t212_ticker}\n"
        f"- Order ID:    {order_id}\n"
        f"- Status:      {status}\n"
        f"- Quantity:    {filled_qty}\n"
        f"- Value:       {filled_val}\n"
        f"- Account:     {T212_MODE.upper()}"
    )


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
