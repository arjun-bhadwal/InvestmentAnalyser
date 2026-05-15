"""Quant-layer MCP tools — strategy backtesting, signal evaluation, portfolio optimization.

These are the analytical core that makes the server a quant engine rather than a
data feed. They run pure numerical work (backtest.py / signals.py / optimize.py)
on price history pulled through the resolver.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

import app
import backtest as bt
import optimize as opt
import signals as sig

mcp = app.mcp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _close_series(df: pd.DataFrame) -> pd.Series:
    """Extract a clean close-price Series from a yfinance OHLCV frame."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in ("Close", "close"):
        if col in df.columns:
            return df[col].dropna().astype(float)
    return pd.Series(dtype=float)


def _fmt(v, suffix: str = "", dp: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    return f"{v:,.{dp}f}{suffix}"


def _build_signal(prices: pd.Series, name: str) -> pd.Series | None:
    """Named signal templates — higher value is stated to be 'more bullish'."""
    if name == "rsi":                        # low RSI = oversold; invert so high = bullish
        return 100.0 - bt._rsi(prices, 14)
    if name == "momentum":                   # 63-day rate of change
        return prices.pct_change(63)
    if name == "ma_distance":                # % distance above the 50-day MA
        return prices / prices.rolling(50).mean() - 1.0
    if name == "volatility":                 # negative realised vol (low vol = bullish)
        return -prices.pct_change().rolling(20).std()
    return None


_SIGNALS = ("rsi", "momentum", "ma_distance", "volatility")


# ---------------------------------------------------------------------------
# 1. Backtesting
# ---------------------------------------------------------------------------

@mcp.tool()
async def backtest_strategy(ticker: str, strategy: str = "sma_crossover",
                            params: dict | None = None, period: str = "5y",
                            fee_bps: float = 5.0, oos_split: float = 0.7) -> str:
    """Backtest a named strategy template on one ticker and compare it to buy-and-hold.

    strategy:  sma_crossover | momentum | rsi_meanrev | breakout | vol_target
    params:    optional strategy overrides, e.g. {"fast": 10, "slow": 50} for
               sma_crossover, {"lookback": 126} for momentum, {"window": 55} for
               breakout, {"window": 14, "entry": 30, "exit": 55} for rsi_meanrev,
               {"target_vol": 0.15, "window": 20} for vol_target.
    period:    price history window — '2y', '5y' (default), '10y'.
    fee_bps:   transaction cost per unit of turnover, in basis points (default 5).
    oos_split: in-sample fraction for the out-of-sample check (0 disables).

    Reports equity growth, Sharpe, max drawdown, trade stats, and the strategy's
    edge over simply holding the asset. A flat/zero strategy ≈ buy-and-hold.
    """
    from resolver import fetch_history

    rt, df = await fetch_history(ticker, period=period, interval="1d")
    prices = _close_series(df)
    if len(prices) < 30:
        return f"Insufficient price history for '{ticker}' (resolved → {rt.yf_symbol})."

    oos = oos_split if 0 < oos_split < 1 else None
    res = await asyncio.to_thread(
        bt.backtest_strategy, prices, strategy, params or {}, fee_bps, 0.045, oos
    )
    if "error" in res:
        return f"Backtest error: {res['error']}"

    s, b, t = res["strategy"], res["buy_and_hold"], res["trade_stats"]
    edge = res["edge"]
    lines = [
        f"## Backtest — {rt.yf_symbol} · {strategy}",
        f"_Params: {res['params'] or 'defaults'}  |  Period: {period}  |  "
        f"Fee: {fee_bps:g}bps  |  {s['observations']} bars_\n",
        f"| Metric | Strategy | Buy & Hold |",
        f"|--------|---------:|-----------:|",
        f"| Total return | {_fmt(s['total_return_pct'],'%')} | {_fmt(b['total_return_pct'],'%')} |",
        f"| Annual return | {_fmt(s['annual_return_pct'],'%')} | {_fmt(b['annual_return_pct'],'%')} |",
        f"| Annual vol | {_fmt(s['annual_vol_pct'],'%')} | {_fmt(b['annual_vol_pct'],'%')} |",
        f"| Sharpe | {_fmt(s['sharpe'])} | {_fmt(b['sharpe'])} |",
        f"| Sortino | {_fmt(s['sortino'])} | {_fmt(b['sortino'])} |",
        f"| Calmar | {_fmt(s['calmar'])} | {_fmt(b['calmar'])} |",
        f"| Max drawdown | {_fmt(s['max_drawdown_pct'],'%')} | {_fmt(b['max_drawdown_pct'],'%')} |",
        "",
        f"**Trades:** {t['trades']}  |  Win rate: {_fmt(t['win_rate_pct'],'%',1)}  |  "
        f"Avg win: {_fmt(t['avg_win_pct'],'%')}  |  Avg loss: {_fmt(t['avg_loss_pct'],'%')}  |  "
        f"Exposure: {_fmt(t['exposure_pct'],'%',0)}",
        f"**Edge vs buy-and-hold:** {_fmt(edge['excess_return_pct'],'%')} annualised  |  "
        f"Sharpe Δ {_fmt(edge['sharpe_delta'])}",
    ]

    if "out_of_sample" in res:
        is_, oos_ = res["in_sample"], res["out_of_sample"]
        lines += [
            "",
            f"**Robustness (split {oos_split:g}):** "
            f"in-sample Sharpe {_fmt(is_['sharpe'])} / return {_fmt(is_['annual_return_pct'],'%')}  →  "
            f"out-of-sample Sharpe {_fmt(oos_['sharpe'])} / return {_fmt(oos_['annual_return_pct'],'%')}",
        ]

    lines.append("\n_Source: yFinance daily closes. Past performance is not predictive._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Signal evaluation
# ---------------------------------------------------------------------------

@mcp.tool()
async def evaluate_signal(ticker: str, signal: str = "momentum",
                          horizon: int = 21, period: str = "5y") -> str:
    """Test whether a signal predicts forward returns — information coefficient,
    quantile spread, hit rate, and decay across horizons.

    signal:  rsi | momentum | ma_distance | volatility  (each oriented so a
             higher value is the 'more bullish' reading).
    horizon: forward-return horizon in trading days (default 21 ≈ 1 month).
    period:  price history window — '2y', '5y' (default), '10y'.

    IC near 0 = no edge; |IC| > ~0.05 with a low p-value is economically
    meaningful. A monotonic quantile spread is the cleaner confirmation.
    """
    from resolver import fetch_history

    if signal not in _SIGNALS:
        return f"Unknown signal '{signal}'. Available: {', '.join(_SIGNALS)}"

    rt, df = await fetch_history(ticker, period=period, interval="1d")
    prices = _close_series(df)
    if len(prices) < 120:
        return f"Insufficient price history for '{ticker}' (resolved → {rt.yf_symbol})."

    sig_series = _build_signal(prices, signal)
    res = await asyncio.to_thread(sig.evaluate_signal, sig_series, prices, horizon, 5)
    if "error" in res:
        return f"Signal evaluation error: {res['error']}"

    ic = res["ic"]
    lines = [
        f"## Signal Evaluation — {rt.yf_symbol} · {signal}",
        f"_Forward horizon: {horizon} trading days  |  Period: {period}_\n",
        f"**Information coefficient:** {_fmt(ic['ic'],'',3)}  "
        f"(p={_fmt(ic['p_value'],'',3)}, n={ic['n']})",
        f"**Hit rate:** {_fmt(res['hit_rate_pct'],'%',1)}  |  "
        f"**Top−bottom quantile spread:** {_fmt(res['top_minus_bottom_pct'],'%')}",
        "",
        "**Forward return by signal quantile** (Q1 = lowest signal):",
    ]
    qr = res["quantile_returns"]
    if isinstance(qr, pd.DataFrame) and not qr.empty:
        lines += ["", "| Quantile | Mean fwd return | Hit rate | N |",
                  "|----------|----------------:|---------:|--:|"]
        for q, row in qr.iterrows():
            lines.append(f"| {q} | {_fmt(row['mean_fwd_return_pct'],'%')} | "
                         f"{_fmt(row['hit_rate_pct'],'%',0)} | {int(row['count'])} |")
    else:
        lines.append("_Not enough data for quantile buckets._")

    decay = res["decay"]
    if isinstance(decay, pd.DataFrame) and not decay.empty:
        lines += ["", "**IC decay across horizons:**",
                  "  " + "  ".join(f"{int(h)}d: {_fmt(r['ic'],'',3)}"
                                    for h, r in decay.iterrows())]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Portfolio optimization
# ---------------------------------------------------------------------------

@mcp.tool()
async def optimize_portfolio(tickers: str, objective: str = "max_sharpe",
                             period: str = "2y", max_weight: float = 1.0) -> str:
    """Compute optimal portfolio weights for a set of tickers.

    tickers:   comma-separated symbols (2+), e.g. 'AAPL, MSFT, GLD, TLT'.
    objective: max_sharpe (default) | min_variance | risk_parity.
    period:    return history window — '1y', '2y' (default), '5y'.
    max_weight: per-asset weight cap in [0, 1] — lower it to force diversification.

    Returns optimal weights, expected return / volatility / Sharpe, and each
    asset's contribution to portfolio risk.
    """
    from resolver import fetch_historic_prices_scaled

    syms = [t.strip() for t in tickers.split(",") if t.strip()]
    if len(syms) < 2:
        return "Provide at least 2 tickers to optimize."

    resolutions, closes = await fetch_historic_prices_scaled(syms, period=period)
    if closes is None or closes.shape[1] < 2:
        return "Insufficient price data — need at least 2 tickers with history."

    returns = closes.sort_index().ffill().pct_change().dropna(how="all")
    res = await asyncio.to_thread(
        opt.optimize_portfolio, returns, objective, 0.045, (0.0, float(max_weight))
    )
    if "error" in res:
        return f"Optimization error: {res['error']}"

    p = res["portfolio"]
    missing = [s for s in syms if not any(
        (resolutions.get(s) and resolutions[s].yf_symbol == c) for c in res["weights"])]

    lines = [
        f"## Portfolio Optimization — {objective}",
        f"_Universe: {len(res['weights'])} assets  |  Period: {period}  |  "
        f"Max weight: {max_weight:g}_\n",
        f"**Expected return:** {_fmt(p['expected_return_pct'],'%')} annual  |  "
        f"**Volatility:** {_fmt(p['volatility_pct'],'%')}  |  "
        f"**Sharpe:** {_fmt(p['sharpe'])}",
        "",
        "| Asset | Weight | Risk contribution |",
        "|-------|-------:|------------------:|",
    ]
    for sym, w in sorted(res["weights"].items(), key=lambda kv: -kv[1]):
        rc = res["risk_contribution_pct"].get(sym, float("nan"))
        lines.append(f"| {sym} | {_fmt(w * 100,'%',1)} | {_fmt(rc,'%',1)} |")

    if missing:
        lines.append(f"\n_⚠️  Excluded (no price history): {', '.join(missing)}_")
    return "\n".join(lines)
