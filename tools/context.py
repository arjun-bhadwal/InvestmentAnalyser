"""Context bundle tools — answer-shaped, single-call tools that return everything
Claude needs for a thematic question in one response.

Layer 1 (these tools) replaces the multi-call fan-out pattern.
Layer 2 (drill-downs) are the explicit deep-dives exposed individually.
Layer 3 (internal _helpers) are called from within bundles only.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import app
from helpers import (
    strip_t212_ticker, position_value, safe_float,
    cached, cache_prices, cache_fundamentals,
)

mcp = app.mcp

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _period_label(horizon: str) -> str:
    return {"1w": "1 Week", "1m": "1 Month", "1q": "1 Quarter", "1y": "1 Year"}.get(horizon, "1 Month")


def _yf_period(horizon: str) -> str:
    return {"1w": "5d", "1m": "1mo", "1q": "3mo", "1y": "1y"}.get(horizon, "1mo")


def _rsi_from_series(close: pd.Series, window: int = 14) -> float | None:
    """Compute RSI from a price series. Returns None if insufficient data."""
    if len(close) < window + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    last_loss = safe_float(loss.iloc[-1], fallback=None)
    last_gain = safe_float(gain.iloc[-1], fallback=None)
    if last_loss is None or last_gain is None:
        return None
    rs = last_gain / last_loss if last_loss != 0 else float("inf")
    return 100 - (100 / (1 + rs)) if last_loss != 0 else 100.0


def _ma_signal(price: float, ma20: float | None, ma50: float | None, ma200: float | None) -> str:
    """Compact MA alignment summary."""
    parts = []
    if ma200 is not None:
        parts.append("▲200" if price > ma200 else "▼200")
    if ma50 is not None:
        parts.append("▲50" if price > ma50 else "▼50")
    if ma20 is not None:
        parts.append("▲20" if price > ma20 else "▼20")
    return " ".join(parts) if parts else "N/A"


# ===========================================================================
# 1. PORTFOLIO CONTEXT BUNDLE
# ===========================================================================

@mcp.tool()
async def get_portfolio_context(horizon: str = "1m") -> str:
    """Full portfolio bundle — one call returns: live positions + P&L, period returns,
    portfolio risk metrics (Sharpe, vol, beta, max-DD), sector/geo/cap allocation,
    market backdrop, fear/greed, and top-5 headlines.

    Use this as the default starting point for ANY portfolio question.
    Only reach for individual drill-down tools when you need deeper data on a specific holding.

    horizon: '1w' | '1m' (default) | '1q' | '1y'
    """
    from resolver import fetch_historic_prices_scaled, bulk_resolve
    from tools.market_data import _get_market_snapshot
    from tools.macro import _get_fear_greed_index
    from tools.news import _get_news_core

    period = _yf_period(horizon)
    hl = _period_label(horizon)
    now_str = datetime.now().strftime("%d %b %Y %H:%M")

    # ── 1. Parallel: T212 data + market macro ────────────────────────────────
    try:
        (positions, acct), (snapshot_str, fg_str) = await asyncio.gather(
            asyncio.gather(app.t212.get_portfolio(), app.t212.get_account_summary()),
            asyncio.gather(_get_market_snapshot(), _get_fear_greed_index()),
        )
    except Exception as e:
        return f"Error fetching portfolio data: {e}"

    if not positions:
        return f"No open positions in T212 {app.T212_MODE} account."

    raw_tickers = [strip_t212_ticker(p["ticker"]) for p in positions]

    # ── 2. ONE batched price download (all holdings + SPY benchmark) ──────────
    try:
        resolutions, closes = await fetch_historic_prices_scaled(
            raw_tickers + ["SPY"], period=period
        )
    except Exception as e:
        resolutions, closes = {}, pd.DataFrame()

    # ── 3. Portfolio weights by position value ────────────────────────────────
    weights_raw = {strip_t212_ticker(p["ticker"]): position_value(p) for p in positions}
    total_portfolio_val = sum(weights_raw.values()) or 1.0
    weights = {k: v / total_portfolio_val for k, v in weights_raw.items()}

    # ── 4. Account summary values ─────────────────────────────────────────────
    currency = acct.get("currency", "")
    total_val = float(acct.get("totalValue", 0) or 0)
    free_cash = float((acct.get("cash") or {}).get("availableToTrade", 0) or 0)
    invested = float((acct.get("investments") or {}).get("totalCost", 0) or 0)
    unreal_pnl = float((acct.get("investments") or {}).get("unrealizedProfitLoss", 0) or 0)

    # ── 5. Per-holding stats ──────────────────────────────────────────────────
    holding_rows = []
    for pos in positions:
        raw = strip_t212_ticker(pos["ticker"])
        rt = resolutions.get(raw)
        yf_sym = rt.yf_symbol if rt else raw
        ccy = rt.currency if rt else "?"

        wgt = weights.get(raw, 0) * 100
        unit_scale = rt.unit_scale if rt else 1.0
        cur_price_raw = safe_float(pos.get("currentPrice"))
        cur_price = cur_price_raw * unit_scale if cur_price_raw is not None else None
        ppl = safe_float(pos.get("ppl"))

        # Period return from the single downloaded frame
        period_ret: float | None = None
        if not closes.empty and yf_sym in closes.columns:
            col = closes[yf_sym].dropna()
            if len(col) >= 2:
                start, end = float(col.iloc[0]), float(col.iloc[-1])
                if start != 0:
                    period_ret = (end - start) / start * 100

        contrib = (period_ret / 100 * weights.get(raw, 0) * 100) if period_ret is not None else None
        trend = ("▲" if period_ret >= 0 else "▼") if period_ret is not None else "—"

        holding_rows.append({
            "raw": raw, "display": yf_sym, "currency": ccy,
            "wgt": wgt, "cur_price": cur_price, "ppl": ppl,
            "period_ret": period_ret, "contrib": contrib, "trend": trend,
        })

    holding_rows.sort(key=lambda x: x["wgt"], reverse=True)

    # ── 6. Portfolio-level risk metrics ───────────────────────────────────────
    port_metrics: dict = {}
    available_raw = [
        h["raw"] for h in holding_rows
        if resolutions.get(h["raw"]) and
           resolutions[h["raw"]].yf_symbol in closes.columns and
           closes[resolutions[h["raw"]].yf_symbol].dropna().shape[0] >= 5
    ]

    if available_raw and not closes.empty:
        rets_frame = pd.DataFrame()
        for raw in available_raw:
            rt = resolutions[raw]
            if rt.yf_symbol in closes.columns:
                rets_frame[raw] = closes[rt.yf_symbol].pct_change()

        rets_frame = rets_frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        port_ret_series = sum(
            rets_frame[r] * weights.get(r, 0) for r in available_raw if r in rets_frame.columns
        ).dropna()

        if len(port_ret_series) >= 5:
            ann_factor = _TRADING_DAYS / len(port_ret_series) * len(port_ret_series)
            ann_ret = float(port_ret_series.mean()) * _TRADING_DAYS * 100
            ann_vol = float(port_ret_series.std()) * np.sqrt(_TRADING_DAYS) * 100
            sharpe = (ann_ret / 100 - 0.045) / (ann_vol / 100) if ann_vol > 0 else 0.0
            cum = (1 + port_ret_series).cumprod()
            max_dd = float(((cum / cum.cummax()) - 1).min()) * 100
            period_total = float(cum.iloc[-1] - 1) * 100

            beta: float | None = None
            spy_rt = resolutions.get("SPY")
            if spy_rt and spy_rt.yf_symbol in closes.columns:
                spy_rets = closes[spy_rt.yf_symbol].pct_change().replace([np.inf, -np.inf], np.nan)
                aligned = pd.concat([port_ret_series, spy_rets], axis=1, keys=["port", "SPY"]).dropna()
                if len(aligned) >= 10:
                    cov = np.cov(aligned["port"].values, aligned["SPY"].values)
                    beta = cov[0][1] / cov[1][1] if cov[1][1] != 0 else None

            port_metrics = {
                "period_total": period_total, "ann_ret": ann_ret,
                "ann_vol": ann_vol, "sharpe": sharpe,
                "max_dd": max_dd, "beta": beta,
            }

    # ── 7. Allocation breakdown (concurrent yf.info per holding) ─────────────
    async def _meta(ticker: str) -> dict:
        def _f():
            return yf.Ticker(ticker).info
        try:
            info = await asyncio.to_thread(_f)
            return {
                "sector": info.get("sector", "Unknown"),
                "country": info.get("country", "Unknown"),
                "marketCap": float(info.get("marketCap", 0) or 0),
            }
        except Exception:
            return {"sector": "Unknown", "country": "Unknown", "marketCap": 0}

    metas = await asyncio.gather(*[_meta(h["display"]) for h in holding_rows])
    for h, m in zip(holding_rows, metas):
        h.update(m)

    # ── 8. Top-5 headlines (concurrent, 2 per ticker) ─────────────────────────
    top5 = [h["raw"] for h in holding_rows[:5]]
    news_results = await asyncio.gather(
        *[_get_news_core(t, max_headlines=2) for t in top5],
        return_exceptions=True,
    )

    # ── Compose output ─────────────────────────────────────────────────────────
    lines: list[str] = [f"## Portfolio Snapshot — {now_str} ({app.T212_MODE.upper()})\n"]

    # Account summary bar
    pnl_sign = "+" if unreal_pnl >= 0 else ""
    lines.append(
        f"**{currency} {total_val:,.2f}** total  |  "
        f"Cash: {currency} {free_cash:,.2f}  |  "
        f"Invested: {currency} {invested:,.2f}  |  "
        f"Unrealised P&L: {pnl_sign}{unreal_pnl:,.2f}"
    )

    # Market + sentiment inline
    snap_inline = "  |  ".join(
        l.strip() for l in snapshot_str.split("\n")
        if any(x in l for x in ("▲", "▼", "S&P", "FTSE", "NASDAQ"))
    )[:200]
    fg_inline = next(
        (l.strip("# ") for l in fg_str.split("\n") if "Score:" in l or "FEAR" in l or "GREED" in l or "NEUTRAL" in l),
        ""
    )
    if snap_inline:
        lines.append(f"**Market:** {snap_inline}")
    if fg_inline:
        lines.append(f"**Sentiment:** {fg_inline}")
    lines.append("")

    # Holdings table
    lines.append(f"### Holdings — {hl} View\n")
    col_w = max(len(h["display"]) for h in holding_rows) + 2
    lines.append(
        f"{'Ticker':<{col_w}} {'Wgt%':>6} {'Price':>10}  {hl + ' Ret':>10}  {'Contrib%':>9}  {'Trend':>5}  {'CCY'}"
    )
    lines.append("-" * (col_w + 52))

    for h in holding_rows:
        ret_s = f"{h['period_ret']:+.1f}%" if h["period_ret"] is not None else "  N/A"
        ctr_s = f"{h['contrib']:+.2f}%" if h["contrib"] is not None else "  N/A"
        px_s = f"{h['cur_price']:,.2f}" if h["cur_price"] else "N/A"
        lines.append(
            f"{h['display']:<{col_w}} {h['wgt']:>5.1f}% {px_s:>10}  {ret_s:>10}  {ctr_s:>9}  {h['trend']:>5}  {h['currency']}"
        )

    # Portfolio summary metrics
    if port_metrics:
        lines.append("")
        lines.append(f"### Portfolio Metrics ({hl})")
        lines.append(
            f"- Period return:  {port_metrics['period_total']:+.2f}%  "
            f"|  Ann. return: {port_metrics['ann_ret']:+.2f}%  "
            f"|  Ann. vol: {port_metrics['ann_vol']:.2f}%"
        )
        beta_s = f"{port_metrics['beta']:.2f}" if port_metrics.get("beta") is not None else "N/A"
        lines.append(
            f"- Sharpe: {port_metrics['sharpe']:.2f}  "
            f"|  Max DD: {port_metrics['max_dd']:.2f}%  "
            f"|  Beta (SPY): {beta_s}"
        )

    # Allocation
    lines.append("")
    lines.append("### Allocation")

    def _bucket_table(label: str, key: str) -> list[str]:
        buckets: dict[str, float] = {}
        for h in holding_rows:
            k = h.get(key, "Unknown") or "Unknown"
            buckets[k] = buckets.get(k, 0.0) + weights_raw.get(h["raw"], 0.0)
        out = [f"**{label}**"]
        for name, val in sorted(buckets.items(), key=lambda x: x[1], reverse=True):
            w = val / total_portfolio_val * 100
            bar = "█" * max(1, int(w / 4))
            out.append(f"  {name:<22} {w:>5.1f}%  {bar}")
        return out

    lines += _bucket_table("Sector", "sector")
    lines.append("")
    lines += _bucket_table("Geography", "country")

    # Concentration
    wts_list = [weights_raw.get(h["raw"], 0) / total_portfolio_val for h in holding_rows]
    hhi = sum(w ** 2 for w in wts_list)
    top5_w = sum(sorted(wts_list, reverse=True)[:5]) * 100
    conc_label = "concentrated" if hhi > 0.15 else "moderate" if hhi > 0.08 else "diversified"
    lines.append("")
    lines.append(f"**Concentration:** {len(holding_rows)} positions  |  Top-5: {top5_w:.1f}%  |  HHI: {hhi:.4f} ({conc_label})")
    if top5_w > 70:
        lines.append(f"⚠️  Top-5 weight {top5_w:.0f}% — consider diversifying")

    # Top-5 headlines
    lines.append("")
    lines.append("### Recent Headlines (Top 5 by Weight)")
    for ticker_raw, result in zip(top5, news_results):
        display_name = resolutions.get(ticker_raw, type("", (), {"yf_symbol": ticker_raw})()).yf_symbol \
            if resolutions.get(ticker_raw) else ticker_raw
        if isinstance(result, Exception):
            lines.append(f"\n**{display_name}:** (unavailable — {result})")
            continue
        headlines = [ln for ln in result.split("\n") if ln.startswith("- [")][:2]
        if headlines:
            lines.append(f"\n**{display_name}:**")
            lines.extend(headlines)
        else:
            lines.append(f"\n**{display_name}:** No headlines in the past 7 days.")

    # Data integrity footer
    lines.append("")
    no_data = [h["display"] for h in holding_rows if h["period_ret"] is None]
    lines.append("### Data")
    lines.append(f"- Price source: yFinance  |  Period: {period}  |  As of: {now_str}")
    if no_data:
        lines.append(f"- ⚠️  No price history for: {', '.join(no_data)} (check known_tickers.json or resolver)")
    else:
        lines.append(f"- ✅ All {len(holding_rows)} holdings have price history for this period")

    return "\n".join(lines)


# ===========================================================================
# 2. TICKER CONTEXT BUNDLE
# ===========================================================================

@mcp.tool()
async def get_ticker_context(ticker: str, depth: str = "standard") -> str:
    """Full ticker bundle — one call returns: multi-horizon performance (1W/1M/3M/1Y),
    fundamentals, technical indicators (RSI, MACD, MAs, ATR, BB), analyst consensus,
    and recent headlines.

    depth: 'standard' (default) | 'deep' (adds DCF, insider trades, financial statements)

    Use this before analysing any individual ticker. Avoids the N+1 call pattern.
    """
    from resolver import aresolve, fetch_history
    from tools.market_data import _get_stock_fundamentals, _get_analyst_ratings, _get_dcf_valuation, _get_financial_statements
    from tools.news import _get_news_core
    from tools.insider import _get_insider_trades_core

    import ta as ta_lib

    # ── Resolve ticker once ───────────────────────────────────────────────────
    try:
        rt = await aresolve(ticker)
    except Exception as e:
        return f"Could not resolve '{ticker}': {e}"

    sym = rt.yf_symbol
    display = sym

    # ── Parallel fetches: 1y price history + fundamentals + news ─────────────
    async def _fetch_info():
        def _f():
            return yf.Ticker(sym).info
        try:
            return await asyncio.to_thread(_f)
        except Exception:
            return {}

    base_tasks = [
        fetch_history(ticker, period="1y", interval="1d"),
        _fetch_info(),
        _get_analyst_ratings(sym),
        _get_news_core(ticker, max_headlines=5),
    ]

    (rt_h, hist_df), info, ratings_str, news_str = await asyncio.gather(*base_tasks)

    # ── Deep drill-downs (optional) ──────────────────────────────────────────
    dcf_str = stmts_str = insider_str = None
    if depth == "deep":
        dcf_str, stmts_str, insider_str = await asyncio.gather(
            _get_dcf_valuation(sym),
            _get_financial_statements(sym),
            _get_insider_trades_core(ticker),
        )

    # ── Multi-horizon returns from single 1y frame ────────────────────────────
    def _period_ret(df: pd.DataFrame, trading_days: int) -> float | None:
        col = df["Close"].dropna() if "Close" in df.columns else (
            df["close"].dropna() if "close" in df.columns else pd.Series(dtype=float)
        )
        if len(col) < trading_days + 1:
            return None
        start, end = float(col.iloc[-(trading_days + 1)]), float(col.iloc[-1])
        return (end - start) / start * 100 if start != 0 else None

    def _fmt_ret(v: float | None) -> str:
        return f"{v:+.2f}%" if v is not None else "N/A"

    ret_1w = _period_ret(hist_df, 5)
    ret_1m = _period_ret(hist_df, 21)
    ret_3m = _period_ret(hist_df, 63)
    ret_1y = _period_ret(hist_df, 252)

    # ── Technical indicators (computed from the 6mo slice of the 1y data) ─────
    tech_lines: list[str] = []
    if not hist_df.empty:
        df6 = hist_df.tail(126).copy()  # ~6 months of trading days
        df6.columns = [c.lower() for c in df6.columns]
        if "close" in df6.columns:
            close = df6["close"]
            high = df6.get("high", close)
            low = df6.get("low", close)
            price = float(close.iloc[-1])

            ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
            ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            ma200_col = hist_df["Close"].dropna() if "Close" in hist_df.columns else close
            ma200 = float(ma200_col.rolling(200).mean().iloc[-1]) if len(ma200_col) >= 200 else None

            rsi = _rsi_from_series(close)

            try:
                macd_obj = ta_lib.trend.MACD(close)
                macd_val = float(macd_obj.macd().iloc[-1])
                macd_sig = float(macd_obj.macd_signal().iloc[-1])
                macd_hist = float(macd_obj.macd_diff().iloc[-1])
            except Exception:
                macd_val = macd_sig = macd_hist = None

            try:
                bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
                bb_upper = float(bb.bollinger_hband().iloc[-1])
                bb_lower = float(bb.bollinger_lband().iloc[-1])
                bb_pct = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
            except Exception:
                bb_upper = bb_lower = bb_pct = None

            try:
                atr = float(ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
                atr_pct = atr / price * 100 if price else None
            except Exception:
                atr = atr_pct = None

            ma_sig = _ma_signal(price, ma20, ma50, ma200)

            rsi_lbl = ""
            if rsi is not None:
                if rsi < 30: rsi_lbl = "OVERSOLD"
                elif rsi > 70: rsi_lbl = "OVERBOUGHT"
                else: rsi_lbl = "neutral"

            macd_lbl = ""
            if macd_hist is not None:
                macd_lbl = "bullish" if macd_hist > 0 else "bearish"

            tech_lines = [
                f"**Technical Indicators** ({rt.currency})",
                f"- Price: {price:,.4f}  |  MAs: {ma_sig}",
                f"- RSI(14): {rsi:.1f} — {rsi_lbl}" if rsi is not None else "- RSI(14): N/A",
                (f"- MACD: {macd_val:.4f}  Signal: {macd_sig:.4f}  Hist: {macd_hist:.4f} — {macd_lbl}"
                 if macd_val is not None else "- MACD: N/A"),
                (f"- ATR(14): {atr:.4f} ({atr_pct:.1f}% of price)" if atr is not None else "- ATR: N/A"),
                (f"- BB position: {bb_pct:.0f}% of range {'(extended)' if bb_pct and bb_pct > 90 else '(oversold zone)' if bb_pct and bb_pct < 10 else ''}"
                 if bb_pct is not None else "- BB: N/A"),
            ]

    # ── Fundamentals inline ───────────────────────────────────────────────────
    def _v(key: str, scale: float = 1, pct: bool = False, billions: bool = False) -> str:
        val = info.get(key)
        if val is None:
            return "N/A"
        v = float(val) / scale
        if billions:
            return f"${v / 1e9:,.1f}B"
        if pct:
            return f"{v * 100:.1f}%"
        return f"{v:,.2f}"

    ccy = info.get("currency", rt.currency)
    name = info.get("longName") or info.get("shortName") or sym
    sector = info.get("sector", "N/A")

    # ── Compose output ─────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%d %b %Y %H:%M")
    lines: list[str] = [f"## {name} ({display}) — {now_str}\n"]
    lines.append(f"**Sector:** {sector}  |  **Currency:** {ccy}  |  **Exchange:** {rt.exchange or 'US'}")
    lines.append("")

    # Performance table
    lines.append("### Performance")
    lines.append(f"| Horizon | Return |")
    lines.append(f"|---------|--------|")
    lines.append(f"| 1 Week  | {_fmt_ret(ret_1w)} |")
    lines.append(f"| 1 Month | {_fmt_ret(ret_1m)} |")
    lines.append(f"| 3 Month | {_fmt_ret(ret_3m)} |")
    lines.append(f"| 1 Year  | {_fmt_ret(ret_1y)} |")
    lines.append("")

    # Fundamentals
    lines.append("### Fundamentals")
    lines.append(
        f"- P/E (TTM/Fwd): {_v('trailingPE')} / {_v('forwardPE')}  |  "
        f"EV/EBITDA: {_v('enterpriseToEbitda')}  |  P/B: {_v('priceToBook')}"
    )
    lines.append(
        f"- EPS (TTM/Fwd): {ccy} {_v('trailingEps')} / {ccy} {_v('forwardEps')}  |  "
        f"Mkt Cap: {_v('marketCap', billions=True)}"
    )
    lines.append(
        f"- Rev growth: {_v('revenueGrowth', pct=True)}  |  "
        f"Profit margin: {_v('profitMargins', pct=True)}  |  "
        f"Beta: {_v('beta')}"
    )
    lines.append(
        f"- 52w High: {ccy} {_v('fiftyTwoWeekHigh')}  |  "
        f"52w Low: {ccy} {_v('fiftyTwoWeekLow')}"
    )
    lines.append("")

    # Technical indicators
    if tech_lines:
        lines.append("### Technical")
        lines.extend(tech_lines)
        lines.append("")

    # Analyst ratings (inline summary from existing helper)
    lines.append("### Analyst Consensus")
    # Extract key lines from ratings_str
    for ln in ratings_str.split("\n"):
        if any(kw in ln for kw in ("Consensus:", "Mean target:", "Implied upside:", "analysts)", "STRONG", "BUY", "HOLD", "SELL")):
            lines.append(ln)
    lines.append("")

    # Deep sections
    if depth == "deep":
        if dcf_str:
            lines.append("### DCF Valuation")
            lines.append(dcf_str)
            lines.append("")
        if stmts_str:
            lines.append("### Financial Statements")
            lines.append(stmts_str)
            lines.append("")
        if insider_str:
            lines.append("### Insider Activity")
            lines.append(insider_str)
            lines.append("")

    # News
    lines.append("### Recent Headlines")
    for ln in news_str.split("\n"):
        if ln.startswith("- ["):
            lines.append(ln)

    lines.append("")
    lines.append(f"_Data: yFinance (delayed) + Finnhub  |  {now_str}_")

    return "\n".join(lines)


# ===========================================================================
# 3. OPPORTUNITY CONTEXT BUNDLE
# ===========================================================================

@mcp.tool()
async def get_opportunity_context(
    universe: str = "sp500",
    style: str = "value_dip",
    max_results: int = 15,
) -> str:
    """Screen for high-conviction entry setups and annotate results against your current portfolio.

    universe: 'sp500' | 'ftse100' | 'both' | 'watchlist' | comma-separated tickers
    style:
      'value_dip'     — low RSI (<40), above MA200, analyst upside >10%  (default)
      'momentum'      — strong 1M momentum, RSI 50-70, above all MAs
      'deep_value'    — RSI <30, any MA alignment, upside >20%
      'quality_growth'— above MA200, low P/E (<25), revenue growth >10%
      'custom'        — no preset filters (pass individual params via the raw screen_stocks tool)
    max_results: cap results (default 15)
    """
    from tools.analysis import _screen_stocks_core
    from tools.portfolio import _get_portfolio_core

    # Style → screener knobs
    style_map = {
        "value_dip": dict(max_rsi=40, min_rsi=0, require_above_ma200=True, min_analyst_upside_pct=10),
        "momentum": dict(max_rsi=70, min_rsi=50, require_above_ma200=True, require_above_ma50=True, min_momentum_1m_pct=3),
        "deep_value": dict(max_rsi=30, min_rsi=0, require_above_ma200=False, min_analyst_upside_pct=20),
        "quality_growth": dict(max_rsi=65, min_rsi=0, require_above_ma200=True, max_pe=25, min_momentum_1m_pct=2),
        "custom": dict(max_rsi=100, min_rsi=0, require_above_ma200=False),
    }
    knobs = style_map.get(style, style_map["value_dip"])

    # Fetch current holdings for exclude/annotate logic (concurrent with screener)
    async def _get_held_tickers() -> set[str]:
        try:
            positions = await app.t212.get_portfolio()
            return {strip_t212_ticker(p["ticker"]).upper() for p in (positions or [])}
        except Exception:
            return set()

    screen_result, held = await asyncio.gather(
        _screen_stocks_core(
            universe=universe,
            max_results=max_results + 10,  # fetch extra to allow for held exclusion
            **knobs,
        ),
        _get_held_tickers(),
    )

    # Annotate candidates already in portfolio
    annotated_lines: list[str] = []
    count = 0
    for line in screen_result.split("\n"):
        if count >= max_results:
            break
        # Check if first token of the data row matches a held ticker
        stripped = line.strip()
        if stripped and not stripped.startswith("-") and not stripped.startswith("*") and not stripped.startswith("#"):
            first_token = stripped.split()[0].upper() if stripped.split() else ""
            if first_token in held:
                line = line + "  ← HELD"
            elif first_token and not any(c in first_token for c in ("-", "|", "=")):
                count += 1
        annotated_lines.append(line)

    # Prepend context header
    header = [
        f"## Opportunity Screen — {universe.upper()} | Style: {style} | {datetime.now().strftime('%d %b %Y')}",
        "",
        f"_Profile: medium risk, geopolitical-aware, wealth-preservation first._  ",
        f"_Tickers marked '← HELD' are already in your portfolio._",
        "",
    ]

    return "\n".join(header) + "\n".join(annotated_lines)
