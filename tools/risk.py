"""Portfolio risk, stress testing, position sizing, and allocation analysis."""
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

import app
from helpers import strip_t212_ticker, fmt_float, position_value

mcp = app.mcp


# ---------------------------------------------------------------------------
# Portfolio Risk Analytics
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio_risk() -> str:
    """Run a comprehensive risk analysis on your portfolio: annualised return & volatility,
    Sharpe ratio, Sortino ratio, Value at Risk, max drawdown, beta to SPY, and a correlation matrix.
    Uses 1 year of daily price data."""

    try:
        positions = await app.t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions to analyse."

    tickers = list(dict.fromkeys(strip_t212_ticker(p["ticker"]) for p in positions))  # unique, ordered

    def _fetch():
        return yf.download(tickers + ["SPY"], period="1y", interval="1d", auto_adjust=True, progress=False)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error fetching price history: {e}"

    if data.empty:
        return "No price data returned. Check that ticker symbols are valid."

    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
    closes = closes.dropna(how="all")
    if closes.empty:
        return "Insufficient price data after filtering."

    returns = closes.pct_change().dropna()
    TRADING_DAYS = 252

    try:
        _rf = yf.Ticker("^TNX").fast_info
        RISK_FREE = float(_rf.last_price) / 100
    except Exception:
        RISK_FREE = 0.045

    lines = ["**Portfolio Risk Analytics — 1-Year Lookback**\n"]

    # Filter to tickers that actually have data (>= 30 days)
    available = [t for t in tickers if t in returns.columns and returns[t].dropna().shape[0] >= 30]
    missing = [t for t in tickers if t not in available]
    if missing:
        lines.append(f"⚠️ Insufficient data for: {', '.join(missing)}\n")

    if not available:
        return "No tickers had sufficient price history (need 30+ trading days)."

    lines.append(f"{'Ticker':<10} {'Ann.Ret%':>10} {'Ann.Vol%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Beta':>7}")
    lines.append("-" * 58)

    spy_ret = returns.get("SPY")
    spy_clean = spy_ret.dropna() if spy_ret is not None else pd.Series(dtype=float)

    for t in available:
        r = returns[t].dropna()
        if r.empty:
            continue
        ann_ret = float(r.mean()) * TRADING_DAYS * 100
        ann_vol = float(r.std()) * np.sqrt(TRADING_DAYS) * 100
        sharpe = (ann_ret / 100 - RISK_FREE) / (ann_vol / 100) if ann_vol > 0 else 0
        cum = (1 + r).cumprod()
        max_dd = float(((cum / cum.cummax()) - 1).min()) * 100

        beta = 0.0
        if len(spy_clean) > 0:
            # Align on overlapping dates
            aligned = pd.concat([returns[t], spy_ret], axis=1, keys=[t, "SPY"]).dropna()
            if len(aligned) >= 30:
                cov = np.cov(aligned[t].values, aligned["SPY"].values)
                beta = cov[0][1] / cov[1][1] if cov[1][1] != 0 else 0.0

        lines.append(f"{t:<10} {ann_ret:>+10.2f} {ann_vol:>10.2f} {sharpe:>8.2f} {max_dd:>+8.2f} {beta:>7.2f}")

    weights_raw = {strip_t212_ticker(p["ticker"]): position_value(p) for p in positions}
    total_val = sum(weights_raw.values()) or 1
    weights = {k: v / total_val for k, v in weights_raw.items()}

    if available:
        port_ret = sum(returns[t] * weights.get(t, 0) for t in available).dropna()
        if len(port_ret) > 5:
            ann_ret = float(port_ret.mean()) * TRADING_DAYS * 100
            ann_vol = float(port_ret.std()) * np.sqrt(TRADING_DAYS) * 100
            sharpe = (ann_ret / 100 - RISK_FREE) / (ann_vol / 100) if ann_vol > 0 else 0
            neg = port_ret[port_ret < 0]
            downside = float(neg.std()) * np.sqrt(TRADING_DAYS) if len(neg) > 1 else 0
            sortino = (ann_ret / 100 - RISK_FREE) / downside if downside > 0 else 0
            cum = (1 + port_ret).cumprod()
            max_dd = float(((cum / cum.cummax()) - 1).min()) * 100
            var95 = float(port_ret.quantile(0.05)) * 100
            tail = port_ret[port_ret <= port_ret.quantile(0.05)]
            cvar95 = float(tail.mean()) * 100 if len(tail) > 0 else var95

            lines.append("-" * 58)
            lines.append(f"\n**Portfolio Summary**")
            lines.append(f"- Annualised return:   {ann_ret:+.2f}%")
            lines.append(f"- Annualised vol:      {ann_vol:.2f}%")
            lines.append(f"- Sharpe ratio:        {sharpe:.2f}")
            lines.append(f"- Sortino ratio:       {sortino:.2f}")
            lines.append(f"- Max drawdown:        {max_dd:.2f}%")
            lines.append(f"- VaR (95%, daily):    {var95:.2f}%")
            lines.append(f"- CVaR (95%, daily):   {cvar95:.2f}%")
            lines.append(f"- Risk-free rate used: {RISK_FREE*100:.2f}% (10Y Treasury)")

    if len(available) >= 2 and len(available) <= 10:
        corr = returns[available].corr()
        lines.append(f"\n**Correlation Matrix**")
        header = f"{'':>8}" + "".join(f"{t[:6]:>8}" for t in available)
        lines.append(header)
        for t1 in available:
            row_str = f"{t1[:6]:>8}"
            for t2 in available:
                val = corr.loc[t1, t2]
                row_str += f"{float(val):>8.2f}" if not np.isnan(val) else f"{'N/A':>8}"
            lines.append(row_str)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

@mcp.tool()
async def calculate_position_size(
    ticker: str, entry_price: float, stop_loss_price: float, risk_pct: float = 2.0,
) -> str:
    """Calculate optimal position size based on risk management rules.
    ticker: stock symbol, entry_price: planned entry, stop_loss_price: stop-loss level,
    risk_pct: max portfolio risk per trade (default 2%).
    Returns position size, shares, Kelly Criterion, and concentration impact."""

    try:
        acct = await app.t212.get_account_summary()
    except Exception as e:
        return f"Error fetching account: {e}"

    total_value = float(acct.get("totalValue", 0) or 0)
    if total_value <= 0:
        return "Could not determine portfolio value."

    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0:
        return "Stop loss must differ from entry."

    max_risk = total_value * (risk_pct / 100)
    shares = int(max_risk / risk_per_share)
    pos_val = shares * entry_price
    pos_pct = (pos_val / total_value) * 100

    def _info():
        return yf.Ticker(ticker).info

    kelly = win_prob = None
    try:
        info = await asyncio.to_thread(_info)
        target = info.get("targetMeanPrice")
        cur = info.get("currentPrice") or info.get("regularMarketPrice") or entry_price
        if target and cur:
            win_amount = abs(float(target) - entry_price)
            rec = (info.get("recommendationKey") or "").lower()
            win_prob = {"strong_buy": .65, "buy": .6, "hold": .5, "sell": .4, "strong_sell": .35}.get(rec, .5)
            b = win_amount / risk_per_share if risk_per_share > 0 else 1
            kelly = max(0, win_prob - ((1 - win_prob) / b)) if b > 0 else 0
    except Exception:
        pass

    ccy = acct.get("currency", "")
    lines = [
        f"**Position Size — {ticker.upper()}**\n",
        f"- Entry: {ccy} {entry_price:,.2f}  |  Stop: {ccy} {stop_loss_price:,.2f}  |  Risk/share: {ccy} {risk_per_share:,.2f}",
        f"- Direction: {'LONG' if entry_price > stop_loss_price else 'SHORT'}",
        "",
        f"**Risk Management**",
        f"- Portfolio: {ccy} {total_value:,.2f}  |  Max risk ({risk_pct}%): {ccy} {max_risk:,.2f}",
        f"- **Shares: {shares:,}**  |  Value: {ccy} {pos_val:,.2f}  |  Weight: {pos_pct:.1f}%",
    ]

    if kelly is not None:
        ks = int(kelly * total_value / entry_price)
        lines += ["", f"**Kelly Criterion** — win prob {win_prob*100:.0f}%, fraction {kelly*100:.1f}%",
                   f"- Full Kelly: {ks:,} shares  |  Half-Kelly (safer): {ks//2:,} shares"]

    if pos_pct > 20:
        lines.append(f"\n⚠️ **CONCENTRATION**: {pos_pct:.1f}% — consider reducing to <10%.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Stress Test
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio_stress_test(simulations: int = 1000) -> str:
    """Run stress tests: historical crisis scenarios and Monte Carlo simulation.
    simulations: number of Monte Carlo runs (default 1000).
    Returns drawdowns under 2008 crisis, COVID crash, 2022 rate hikes, and probability distribution."""

    try:
        positions = await app.t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"

    if not positions:
        return "No open positions."

    tickers = list(dict.fromkeys(strip_t212_ticker(p["ticker"]) for p in positions))
    weights_raw = {strip_t212_ticker(p["ticker"]): position_value(p) for p in positions}
    total = sum(weights_raw.values()) or 1
    weights = {k: v / total for k, v in weights_raw.items()}

    # Use max available history — try to go back as far as possible for scenarios
    def _fetch():
        return yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False)

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"Error: {e}"

    if data.empty:
        return "No price data returned. Check that ticker symbols are valid."

    closes = data["Close"].dropna(how="all") if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
    returns = closes.pct_change().dropna()

    avail = [t for t in tickers if t in returns.columns and returns[t].dropna().shape[0] >= 30]
    missing = [t for t in tickers if t not in avail]

    lines = ["**Portfolio Stress Test**\n"]
    if missing:
        lines.append(f"⚠️ Insufficient data for: {', '.join(missing)}\n")

    if not avail:
        return "No tickers had sufficient price history for stress testing."

    # Determine earliest date available
    earliest = returns.index.min().strftime("%Y-%m-%d")
    lines.append(f"_Data available from: {earliest}_\n")

    scenarios = {
        "2008 Financial Crisis": ("2008-09-01", "2009-03-09"),
        "COVID Crash (2020)": ("2020-02-19", "2020-03-23"),
        "2022 Rate Hike Selloff": ("2022-01-03", "2022-10-12"),
        "2024 Aug VIX Spike": ("2024-07-16", "2024-08-05"),
    }

    lines.append("**Historical Scenarios**")
    lines.append(f"{'Scenario':<28} {'Impact':>12}")
    lines.append("-" * 42)
    for name, (s, e) in scenarios.items():
        try:
            pr = returns.loc[s:e]
            if pr.empty or len(pr) < 2:
                lines.append(f"{name:<28} {'no data':>12}")
                continue
            port_ret = sum(
                pr[t].sum() * weights.get(t, 0)
                for t in avail if t in pr.columns
            )
            lines.append(f"{name:<28} {port_ret*100:>+10.1f}%")
        except Exception:
            lines.append(f"{name:<28} {'N/A':>12}")

    if avail:
        port_ret = sum(returns[t] * weights.get(t, 0) for t in avail).dropna()
        if len(port_ret) >= 20:
            mu = float(port_ret.mean())
            sig = float(port_ret.std())
            if sig > 0:  # guard against zero volatility
                np.random.seed(42)
                sims = sorted([float(np.cumprod(1 + np.random.normal(mu, sig, 252))[-1] - 1) for _ in range(simulations)])
                p5, p25, p50, p75, p95 = [sims[int(p * simulations)] for p in [.05, .25, .5, .75, .95]]

                lines += [
                    f"\n**Monte Carlo ({simulations:,} runs, 1yr)**",
                    f"- 5th pctile:  {p5*100:+.1f}%", f"- 25th:        {p25*100:+.1f}%",
                    f"- **Median:    {p50*100:+.1f}%**", f"- 75th:        {p75*100:+.1f}%",
                    f"- 95th pctile: {p95*100:+.1f}%", "",
                    f"- P(loss): {sum(1 for r in sims if r < 0)/simulations*100:.1f}%",
                    f"- P(>20% gain): {sum(1 for r in sims if r > .2)/simulations*100:.1f}%",
                    f"- P(>20% loss): {sum(1 for r in sims if r < -.2)/simulations*100:.1f}%",
                ]
            else:
                lines.append("\n⚠️ Portfolio volatility is zero — cannot run Monte Carlo.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Allocation
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_portfolio_allocation() -> str:
    """Analyse portfolio allocation by sector, geography, and market cap.
    Shows concentration metrics and diversification score."""

    try:
        positions = await app.t212.get_portfolio()
    except Exception as e:
        return f"Error: {e}"

    if not positions:
        return "No open positions."

    holdings = []
    for p in positions:
        ticker = strip_t212_ticker(p["ticker"])
        holdings.append({"ticker": ticker, "value": position_value(p)})

    total_val = sum(h["value"] for h in holdings) or 1

    async def _meta(ticker):
        def _f():
            return yf.Ticker(ticker).info
        try:
            info = await asyncio.to_thread(_f)
            return {"sector": info.get("sector", "Unknown"), "country": info.get("country", "Unknown"),
                    "marketCap": float(info.get("marketCap", 0) or 0), "name": info.get("shortName") or ticker}
        except Exception:
            return {"sector": "Unknown", "country": "Unknown", "marketCap": 0, "name": ticker}

    metas = await asyncio.gather(*[_meta(h["ticker"]) for h in holdings])
    for h, m in zip(holdings, metas):
        h.update(m)

    lines = ["**Portfolio Allocation**\n"]

    holdings.sort(key=lambda x: x["value"], reverse=True)
    lines.append("**Top Holdings**")
    lines.append(f"{'#':<4} {'Ticker':<10} {'Name':<24} {'Value':>12} {'Wgt':>7}")
    lines.append("-" * 60)
    for i, h in enumerate(holdings[:10], 1):
        lines.append(f"{i:<4} {h['ticker']:<10} {h['name'][:23]:<24} {h['value']:>12,.2f} {h['value']/total_val*100:>6.1f}%")

    for label, key in [("Sector", "sector"), ("Geography", "country")]:
        buckets = {}
        for h in holdings:
            buckets[h.get(key, "Unknown")] = buckets.get(h.get(key, "Unknown"), 0) + h["value"]
        lines.append(f"\n**{label} Allocation**")
        for name, val in sorted(buckets.items(), key=lambda x: x[1], reverse=True):
            w = val / total_val * 100
            lines.append(f"  {name:<22} {w:>5.1f}%  {'█' * int(w / 2)}")

    cap_buckets = {"Mega (>$200B)": 0, "Large ($10-200B)": 0, "Mid ($2-10B)": 0, "Small (<$2B)": 0}
    for h in holdings:
        mc = h.get("marketCap", 0)
        if mc >= 200e9: cap_buckets["Mega (>$200B)"] += h["value"]
        elif mc >= 10e9: cap_buckets["Large ($10-200B)"] += h["value"]
        elif mc >= 2e9: cap_buckets["Mid ($2-10B)"] += h["value"]
        else: cap_buckets["Small (<$2B)"] += h["value"]

    lines.append(f"\n**Market Cap**")
    for b, val in cap_buckets.items():
        w = val / total_val * 100
        if w > 0:
            lines.append(f"  {b:<22} {w:>5.1f}%")

    wts = [h["value"] / total_val for h in holdings]
    hhi = sum(w ** 2 for w in wts)
    top5 = sum(w for w in sorted(wts, reverse=True)[:5]) * 100
    lines += [
        f"\n**Concentration**",
        f"- Positions: {len(holdings)}  |  Top 5: {top5:.1f}%",
        f"- HHI: {hhi:.4f} ({'concentrated' if hhi > .15 else 'moderate' if hhi > .08 else 'diversified'})  |  Effective positions: {1/hhi:.1f}",
    ]
    if top5 > 70:
        lines.append(f"\n⚠️ Top 5 = {top5:.0f}% — heavy concentration")

    return "\n".join(lines)
