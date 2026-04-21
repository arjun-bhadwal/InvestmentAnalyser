"""Insider trading activity via Finnhub."""
import asyncio
from datetime import datetime, timedelta

import finnhub

import app
from helpers import cached, cache_fundamentals, finnhub_retry

mcp = app.mcp


@cached(cache_fundamentals)
async def _get_insider_trades_core(ticker: str) -> str:
    """Recent insider transactions for a stock via Finnhub — raw purchase/sale data."""
    from resolver import aresolve

    # Resolve ticker to get correct Finnhub format
    try:
        rt = await aresolve(ticker)
        base = rt.yf_symbol.split(".")[0]
        finnhub_sym = f"LSE:{base}" if rt.exchange == "LSE" else base
        display = rt.yf_symbol
    except Exception:
        finnhub_sym = ticker.upper().split(".")[0]
        display = ticker.upper()

    @finnhub_retry
    def _fetch():
        client = finnhub.Client(api_key=app.FINNHUB_API_KEY)
        today = datetime.today().strftime("%Y-%m-%d")
        year_ago = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        return client.stock_insider_transactions(finnhub_sym, year_ago, today)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching insider trades for {display}: {e}"

    transactions = data.get("data", []) if isinstance(data, dict) else data
    if not transactions:
        return f"No insider transactions found for {display} in the past year."

    lines = [
        f"**Insider Trades — {display} (last 12 months)**\n",
        f"{'Date':<14} {'Name':<24} {'Type':<8} {'Shares':>12} {'Price':>10} {'Value':>14}",
        "-" * 86,
    ]

    total_buy, total_sell = 0, 0

    for t in transactions[:20]:
        name = (t.get("name") or "Unknown")[:23]
        tx_type = t.get("transactionType", "")
        shares = float(t.get("share", 0) or 0)
        price = float(t.get("transactionPrice", 0) or 0)
        value = abs(shares * price)
        date_str = t.get("filingDate", "")[:10]

        if any(kw in tx_type.lower() for kw in ("purchase", "buy", "acquisition")):
            side = "BUY"
            total_buy += value
        elif any(kw in tx_type.lower() for kw in ("sale", "sell", "disposition")):
            side = "SELL"
            total_sell += value
        else:
            side = tx_type[:7]

        lines.append(f"{date_str:<14} {name:<24} {side:<8} {shares:>12,.0f} {price:>10,.2f} {value:>14,.0f}")

    lines.append("-" * 86)
    lines.append(f"Total insider buying:  ${total_buy:>14,.0f}")
    lines.append(f"Total insider selling: ${total_sell:>14,.0f}")

    ratio = total_buy / total_sell if total_sell > 0 else float("inf")
    ratio_str = f"{ratio:.2f}x" if ratio != float("inf") else "∞ (no sales)"
    lines.append(f"\nInsider buy/sell ratio (90d): {ratio_str}  (buys: ${total_buy:,.0f} | sells: ${total_sell:,.0f})")

    return "\n".join(lines)
