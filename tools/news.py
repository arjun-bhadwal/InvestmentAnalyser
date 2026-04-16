"""News, web search, and Perplexity deep research tools."""
import asyncio
from datetime import datetime, timedelta

import finnhub
try:
    from ddgs import DDGS  # renamed package (v7+)
except ImportError:
    from duckduckgo_search import DDGS

import app
from helpers import finnhub_retry

mcp = app.mcp


@mcp.tool()
async def get_news(ticker: str) -> str:
    """Return the 5 most recent news headlines for a stock ticker via Finnhub.
    Use standard tickers for US stocks (e.g. AAPL). For LSE stocks use EXCHANGE:TICKER (e.g. LSE:LLOY)."""
    today = datetime.today().date()
    week_ago = today - timedelta(days=7)

    @finnhub_retry
    def _fetch():
        client = finnhub.Client(api_key=app.FINNHUB_API_KEY)
        return client.company_news(ticker.upper(), _from=str(week_ago), to=str(today))

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
    """Search the web for financial news, analyst commentary, market chatter, geopolitical events.
    Examples: 'TSLA latest news', 'oil price geopolitical risk 2025'"""

    def _fetch():
        # Retry up to 3 times — DuckDuckGo occasionally rate-limits single attempts
        last_err = None
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=8, region="wt-wt", timelimit="m"))
                if results:
                    return results, None
                # No results with time limit — retry without it
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=8, region="wt-wt"))
                if results:
                    return results, None
            except Exception as e:
                last_err = e
        return [], last_err

    try:
        results, err = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=20)
    except asyncio.TimeoutError:
        return f"Search timed out for '{query}'. Try a shorter query."
    except Exception as e:
        return f"Error searching: {e}"

    if err and not results:
        return f"Search failed for '{query}': {err}"

    if not results:
        return f"No results found for '{query}'."

    lines = [f"**Web Search: {query}**\n"]
    for r in results:
        lines.append(f"**{r.get('title', '')}**")
        lines.append(f"{r.get('body', '')[:200]}...")
        lines.append(f"[{r.get('href', '')}]({r.get('href', '')})\n")

    return "\n".join(lines)


@mcp.tool()
async def research(query: str) -> str:
    """Deep research synthesis using Perplexity AI — returns a cited, multi-source answer.
    Better than web search for earnings analysis, geopolitical context, sector deep-dives.
    Requires PERPLEXITY_API_KEY in .env."""

    if not app.PERPLEXITY_API_KEY:
        return "PERPLEXITY_API_KEY not set. Add it to .env. Get a key at https://www.perplexity.ai/settings/api"

    import httpx

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a financial research analyst. Provide concise, factual, well-sourced answers."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 1024,
        "return_citations": True,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {app.PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
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
