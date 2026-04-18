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


async def _get_news_core(ticker: str, max_headlines: int = 5) -> str:
    """Return recent news headlines for a stock ticker.

    Tries Finnhub first (good for US equities). On empty results — including
    LSE-listed names and ETFs where Finnhub coverage is thin or absent — falls
    back to a web search using the fully-qualified yf_symbol (e.g. 'RARE.L')
    so the query can never cross-contaminate with a same-name US ticker.
    """
    from resolver import aresolve

    try:
        rt = await aresolve(ticker)
        base = rt.yf_symbol.split(".")[0]
        # Finnhub uses LSE:SYMBOL for London-listed stocks; bare symbol for US
        finnhub_sym = f"LSE:{base}" if rt.exchange == "LSE" else base
        display = rt.yf_symbol
    except Exception:
        finnhub_sym = ticker.upper().split(".")[0]
        display = ticker.upper()

    today = datetime.today().date()
    week_ago = today - timedelta(days=7)

    @finnhub_retry
    def _fetch():
        client = finnhub.Client(api_key=app.FINNHUB_API_KEY)
        return client.company_news(finnhub_sym, _from=str(week_ago), to=str(today))

    try:
        articles = await asyncio.to_thread(_fetch)
    except Exception as e:
        articles = []

    if articles:
        lines = [f"**Recent news: {display}** _(via Finnhub)_\n"]
        for article in articles[:max_headlines]:
            headline = article.get("headline", "No headline")
            source = article.get("source", "")
            url = article.get("url", "")
            ts = article.get("datetime", 0)
            date_str = datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else ""
            lines.append(f"- [{headline}]({url}) — {source} ({date_str})")
        return "\n".join(lines)

    # Finnhub returned nothing — don't fall back to a bare symbol (risk of wrong
    # instrument, e.g. LSE:RARE → bare RARE → Ultragenyx biotech). Use web search
    # with the fully-qualified symbol so the query is unambiguous.
    try:
        exchange_hint = f" {rt.exchange}" if rt.exchange else ""
    except NameError:
        exchange_hint = ""
    web_query = f"{display}{exchange_hint} news {today.year}"
    web_result = await _search_web(web_query)
    return f"⚠️ Finnhub: no coverage for '{finnhub_sym}' — using web search instead.\n\n{web_result}"


async def _search_web(query: str) -> str:
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


async def _research_perplexity(query: str) -> str:
    """Deep research via Perplexity AI (requires PERPLEXITY_API_KEY in .env).
    Not exposed — use _search_web instead, which requires no API key."""

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
