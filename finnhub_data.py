"""Finnhub data layer — structured fetchers used as best-source / fallback.

Finnhub's free tier is US-centric: filings and as-reported financials return
403 for non-US symbols. Every fetcher therefore returns structured data or
None — callers fall back to yfinance when None. A symbol containing "." (a
yfinance exchange suffix, e.g. SHEL.L) is treated as non-US and skipped.

All functions are async, pure retrieval, no formatting.
"""
from __future__ import annotations

import asyncio

import httpx

import app

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 15.0


def is_us_symbol(symbol: str) -> bool:
    """Finnhub free tier only serves US listings — a yfinance suffix means non-US."""
    return bool(symbol) and "." not in symbol


async def _get(path: str, params: dict) -> dict | list | None:
    """GET a Finnhub endpoint with light 429 backoff. Returns parsed JSON or None."""
    params = {**params, "token": app.FINNHUB_API_KEY}
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(f"{_BASE}{path}", params=params)
            if r.status_code == 429:
                await asyncio.sleep(1.0 + attempt)
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# As-reported financial statements
# ---------------------------------------------------------------------------

# us-gaap concepts (prefix stripped) → our normalised field. First match wins.
_INCOME = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
}
_BALANCE = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
_CASHFLOW = {
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
}


def _pick(items: list, concepts: list[str]) -> float | None:
    """First-matching us-gaap concept value from a statement line-item list."""
    by_concept = {}
    for it in items:
        c = (it.get("concept") or "").replace("us-gaap_", "")
        if c and c not in by_concept:
            by_concept[c] = it.get("value")
    for concept in concepts:
        v = by_concept.get(concept)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _extract(items: list, field_map: dict) -> dict:
    return {field: _pick(items, concepts) for field, concepts in field_map.items()}


async def reported_financials(symbol: str, freq: str = "annual", periods: int = 4) -> list[dict] | None:
    """As-reported financial statements (US only) — list of periods, newest first.

    Each period: {period, form, income{...}, balance{...}, cashflow{...}, fcf}.
    Returns None for non-US symbols or on any failure → caller falls back to yfinance.
    """
    if not is_us_symbol(symbol):
        return None
    data = await _get("/stock/financials-reported", {"symbol": symbol, "freq": freq})
    if not data or not isinstance(data, dict) or not data.get("data"):
        return None

    out = []
    for rep in data["data"][:periods]:
        report = rep.get("report") or {}
        income = _extract(report.get("ic") or [], _INCOME)
        cashflow = _extract(report.get("cf") or [], _CASHFLOW)
        ocf, capex = cashflow.get("operating_cf"), cashflow.get("capex")
        fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
        out.append({
            "period": str(rep.get("endDate") or rep.get("year") or "")[:10],
            "form": rep.get("form", ""),
            "income": income,
            "balance": _extract(report.get("bs") or [], _BALANCE),
            "cashflow": cashflow,
            "fcf": fcf,
        })
    return out or None


# ---------------------------------------------------------------------------
# SEC filings, earnings, ratings, insider, peers, metrics, market status
# ---------------------------------------------------------------------------

async def sec_filings(symbol: str, limit: int = 8) -> list[dict] | None:
    """Recent SEC filings (US only) — newest first, periodic reports prioritised."""
    if not is_us_symbol(symbol):
        return None
    data = await _get("/stock/filings", {"symbol": symbol})
    if not isinstance(data, list) or not data:
        return None
    priority = {"10-K": 0, "10-Q": 1, "8-K": 2, "20-F": 1, "6-K": 2}
    ranked = sorted(data, key=lambda f: (priority.get(f.get("form", ""), 5),
                                         -_date_key(f.get("filedDate", ""))))
    return [{
        "form": f.get("form", ""),
        "filed": str(f.get("filedDate", ""))[:10],
        "url": f.get("filingUrl") or f.get("reportUrl", ""),
    } for f in ranked[:limit]]


def _date_key(s: str) -> int:
    return int("".join(ch for ch in str(s)[:10] if ch.isdigit()) or 0)


async def earnings_surprises(symbol: str, limit: int = 8) -> list[dict] | None:
    """Historical EPS actual vs estimate with surprise % (US only)."""
    if not is_us_symbol(symbol):
        return None
    data = await _get("/stock/earnings", {"symbol": symbol})
    if not isinstance(data, list) or not data:
        return None
    return data[:limit]


async def recommendation_trends(symbol: str) -> list[dict] | None:
    """Analyst buy/hold/sell counts over recent months (US only), newest first."""
    if not is_us_symbol(symbol):
        return None
    data = await _get("/stock/recommendation", {"symbol": symbol})
    if not isinstance(data, list) or not data:
        return None
    return data


async def insider_sentiment(symbol: str, months: int = 12) -> list[dict] | None:
    """Monthly aggregated insider sentiment (MSPR) and net share change (US only)."""
    if not is_us_symbol(symbol):
        return None
    from datetime import date, timedelta
    today = date.today()
    frm = today - timedelta(days=months * 31)
    data = await _get("/stock/insider-sentiment",
                      {"symbol": symbol, "from": str(frm), "to": str(today)})
    if not isinstance(data, dict) or not data.get("data"):
        return None
    return data["data"]


async def company_peers(symbol: str) -> list[str] | None:
    """Finnhub's peer set for a symbol (US only)."""
    if not is_us_symbol(symbol):
        return None
    data = await _get("/stock/peers", {"symbol": symbol})
    if not isinstance(data, list) or len(data) < 2:
        return None
    return data


async def basic_metrics(symbol: str) -> dict | None:
    """Finnhub basic financial metrics (`metric` block) — US only."""
    if not is_us_symbol(symbol):
        return None
    data = await _get("/stock/metric", {"symbol": symbol, "metric": "all"})
    if not isinstance(data, dict) or not data.get("metric"):
        return None
    return data["metric"]


async def market_status(exchange: str = "US") -> dict | None:
    """Holiday-aware market status for an exchange (US, L, etc.)."""
    data = await _get("/stock/market-status", {"exchange": exchange})
    if not isinstance(data, dict) or "isOpen" not in data:
        return None
    return data


async def market_holidays(exchange: str = "US", limit: int = 6) -> list[dict] | None:
    """Upcoming exchange holidays."""
    data = await _get("/stock/market-holiday", {"exchange": exchange})
    if not isinstance(data, dict) or not data.get("data"):
        return None
    rows = sorted(data["data"], key=lambda h: str(h.get("atDate", "")))
    from datetime import date
    today = str(date.today())
    upcoming = [h for h in rows if str(h.get("atDate", "")) >= today]
    return (upcoming or rows)[:limit]
