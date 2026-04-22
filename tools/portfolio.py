"""T212 portfolio, account, history, orders, and pies tools."""
import asyncio
from datetime import datetime

import app
from helpers import strip_t212_ticker, fmt_float

mcp = app.mcp


def _t212():
    return app.t212


# ---------------------------------------------------------------------------
# Portfolio & Account
# ---------------------------------------------------------------------------

async def _get_portfolio_core() -> str:
    """Return all open positions from your Trading 212 account with quantity, average price, current price, and P&L."""
    try:
        positions = await _t212().get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return f"No open positions in your T212 {app.T212_MODE} account."

    lines = [
        f"**Trading 212 Portfolio ({app.T212_MODE.upper()})**\n",
        f"{'Ticker':<10} {'Qty':>10} {'Avg Price':>12} {'Current':>12} {'P&L':>12} {'Ccy':>4}",
        "-" * 64,
    ]
    for pos in positions:
        ticker = strip_t212_ticker(pos.get("ticker", "?"))
        qty = fmt_float(pos.get("quantity"), 4)
        avg = fmt_float(pos.get("averagePrice"))
        cur = fmt_float(pos.get("currentPrice"))
        ppl = pos.get("ppl", 0)
        ppl_str = f"{float(ppl):+,.2f}" if ppl is not None else "N/A"
        ccy = str(pos.get("currencyCode", "") or pos.get("currency", "") or "").upper()
        lines.append(f"{ticker:<10} {qty:>10} {avg:>12} {cur:>12} {ppl_str:>12} {ccy:>4}")

    total_ppl = sum(float(p.get("ppl", 0) or 0) for p in positions)
    lines.append("-" * 64)
    lines.append(f"{'Total P&L':<52} {total_ppl:>+,.2f}")
    return "\n".join(lines)


async def _get_account_summary_core() -> str:
    """Return a summary of your Trading 212 account: total value, free cash, and total invested."""
    try:
        data = await _t212().get_account_summary()
    except Exception as e:
        return f"Error fetching account summary: {e}"

    currency = data.get("currency", "")
    total = fmt_float(data.get("totalValue"))
    free_cash = fmt_float(data.get("cash", {}).get("availableToTrade"))
    invested = fmt_float(data.get("investments", {}).get("totalCost"))
    ppl = data.get("investments", {}).get("unrealizedProfitLoss")
    ppl_str = f"{float(ppl):+,.2f}" if ppl is not None else "N/A"

    return (
        f"**Trading 212 Account Summary ({app.T212_MODE.upper()})**\n\n"
        f"- Total value:    {currency} {total}\n"
        f"- Available cash: {currency} {free_cash}\n"
        f"- Invested:       {currency} {invested}\n"
        f"- Unrealised P&L: {currency} {ppl_str}"
    )


async def _get_trade_history(limit: int = 20) -> str:
    """Return your recent Trading 212 order/trade history."""
    try:
        orders = await _t212().get_order_history(limit=limit)
    except Exception as e:
        return f"Error fetching trade history: {e}"

    if not orders:
        return "No trade history found."

    lines = [f"**Trade History — last {limit} orders ({app.T212_MODE.upper()})**\n",
             f"{'Date':<14} {'Name':<28} {'Side':<5} {'Qty':>10} {'Price':>10} {'Value':>10} {'CCY':<5}",
             "-" * 86]

    for o in orders:
        order = o.get("order", o)
        fill  = o.get("fill", o)

        raw_ticker = order.get("ticker", "?")
        instrument = order.get("instrument", {}) or {}
        name = instrument.get("name") or strip_t212_ticker(raw_ticker)
        name = name[:27]

        side = order.get("side", "")
        if not side:
            otype = order.get("type", "")
            side = "BUY" if "BUY" in otype.upper() else "SELL" if "SELL" in otype.upper() else otype[:4]

        qty = float(fill.get("quantity") or order.get("filledQuantity") or order.get("quantity") or 0)
        wallet = (fill.get("walletImpact") or {})
        value  = abs(float(wallet.get("netValue") or order.get("filledValue") or order.get("value") or 0))
        ccy    = wallet.get("currency") or order.get("currency", "")
        price = (value / qty) if qty else 0

        raw_date = order.get("createdAt") or order.get("dateCreated") or fill.get("filledAt") or ""
        try:
            date_str = datetime.fromisoformat(raw_date[:19].replace("Z", "")).strftime("%d %b %Y")
        except Exception:
            date_str = raw_date[:10]

        lines.append(
            f"{date_str:<14} {name:<28} {side:<5} {fmt_float(qty, 4):>10} "
            f"{fmt_float(price):>10} {fmt_float(value):>10} {ccy:<5}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History — Dividends & Transactions
# ---------------------------------------------------------------------------

async def _get_dividend_history(limit: int = 20) -> str:
    """Return dividend payments received in your Trading 212 account.
    Use this to track income from dividend-paying stocks."""
    try:
        dividends = await _t212().get_dividend_history(limit=limit)
    except Exception as e:
        return f"Error fetching dividend history: {e}"

    if not dividends:
        return f"No dividend history found in your T212 {app.T212_MODE} account."

    lines = [
        f"**Dividend History ({app.T212_MODE.upper()}) — last {limit}**\n",
        f"{'Date':<14} {'Ticker':<10} {'Shares':>10} {'Amount':>12} {'Tax':>10}",
        "-" * 60,
    ]
    total = 0.0
    for d in dividends:
        ticker = strip_t212_ticker(d.get("ticker", "?"))
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
            f"{date_str:<14} {ticker:<10} {fmt_float(quantity, 4):>10} "
            f"{fmt_float(amount):>12} {fmt_float(tax):>10}"
        )

    lines.append("-" * 60)
    lines.append(f"{'Total dividends received':<46} {fmt_float(total):>12}")
    return "\n".join(lines)


async def _get_transaction_history(limit: int = 20) -> str:
    """Return cash transaction history: deposits and withdrawals on your Trading 212 account."""
    try:
        transactions = await _t212().get_transaction_history(limit=limit)
    except Exception as e:
        return f"Error fetching transaction history: {e}"

    if not transactions:
        return f"No transaction history found in your T212 {app.T212_MODE} account."

    lines = [
        f"**Transaction History ({app.T212_MODE.upper()}) — last {limit}**\n",
        f"{'Date':<14} {'Type':<16} {'Amount':>14} {'Currency':<10}",
        "-" * 58,
    ]
    for tx in transactions:
        raw_date = tx.get("dateTime") or tx.get("date") or ""
        try:
            date_str = datetime.fromisoformat(raw_date[:10]).strftime("%d %b %Y")
        except Exception:
            date_str = raw_date[:10]
        tx_type = tx.get("type", "UNKNOWN")
        amount = float(tx.get("amount", 0) or 0)
        currency = tx.get("currency", "")
        sign = "+" if amount >= 0 else ""
        lines.append(f"{date_str:<14} {tx_type:<16} {sign}{fmt_float(amount):>14} {currency:<10}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orders — Open / Pending
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_open_orders() -> str:
    """Return any currently open or pending orders on your Trading 212 account.
    Use this to check what limit or pending orders are waiting to be filled."""
    try:
        orders = await _t212().get_open_orders()
    except Exception as e:
        return f"Error fetching open orders: {e}"

    if not orders:
        return f"No open or pending orders in your T212 {app.T212_MODE} account."

    lines = [
        f"**Open Orders ({app.T212_MODE.upper()})**\n",
        f"{'Ticker':<10} {'Type':<12} {'Side':<6} {'Qty':>10} {'Limit':>10} {'Value':>10} {'Status':<12}",
        "-" * 74,
    ]
    for o in orders:
        ticker = strip_t212_ticker(o.get("ticker", "?"))
        order_type = o.get("type", "")
        side = "BUY" if "BUY" in order_type.upper() else "SELL" if "SELL" in order_type.upper() else "?"
        qty = o.get("quantity", 0)
        limit = o.get("limitPrice") or o.get("stopPrice") or 0
        value = o.get("value", 0) or (float(qty or 0) * float(limit or 0))
        status = o.get("status", "?")
        lines.append(
            f"{ticker:<10} {order_type:<12} {side:<6} {fmt_float(qty, 4):>10} "
            f"{fmt_float(limit):>10} {fmt_float(value):>10} {status:<12}"
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
        pies = await _t212().get_pies()
    except Exception as e:
        return f"Error fetching pies: {e}"

    if not pies:
        return f"No pies found in your T212 {app.T212_MODE} account."

    async def _get_name(pie_id: int) -> str:
        try:
            detail = await _t212().get_pie(pie_id)
            return (detail.get("settings", {}) or {}).get("name", "") or f"Pie {pie_id}"
        except Exception:
            return f"Pie {pie_id}"

    names = await asyncio.gather(*[_get_name(p["id"]) for p in pies])

    lines = [
        f"**Pies ({app.T212_MODE.upper()})**\n",
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
            f"{pie_id:<10} {name[:27]:<28} {fmt_float(invested):>12} {fmt_float(value):>12} "
            f"{sign}{fmt_float(ret):>10} {sign}{ret_pct:.2f}%"
        )

    return "\n".join(lines)


# ===========================================================================
# NEW CONSOLIDATED ACCOUNT HISTORY ENDPOINT
# ===========================================================================

@mcp.tool()
async def get_account_history(report_type: str = "all", limit: int = 20) -> str:
    """Return historical account activity.
    report_type: 'all' (default), 'trades' (orders), 'dividends', 'transactions' (cash)"""
    
    tasks = []
    if report_type in ("all", "trades"):
        tasks.append(("trades", _get_trade_history(limit=limit)))
    if report_type in ("all", "dividends"):
        tasks.append(("dividends", _get_dividend_history(limit=limit)))
    if report_type in ("all", "transactions"):
        tasks.append(("transactions", _get_transaction_history(limit=limit)))

    if not tasks:
        return f"Invalid report_type: {report_type}. Choose 'all', 'trades', 'dividends', or 'transactions'."

    try:
        keys, coros = zip(*tasks)
        results = await asyncio.gather(*coros, return_exceptions=True)
        
        output_parts = []
        for key, res in zip(keys, results):
            if isinstance(res, Exception):
                output_parts.append(f"Error fetching {key}: {res}")
            else:
                s_res = str(res)
                if "_[DEPRECATED" in s_res:
                    s_res = s_res.split("]_", 1)[-1].strip()
                output_parts.append(s_res)
        
        return "\n\n---\n\n".join(output_parts)
    except Exception as e:
        return f"Account history error: {e}"

