"""Vectorised strategy backtesting.

Design principles (mirrors `quant.py`)
--------------------------------------
* Pure functions over pandas/numpy — no I/O, no side effects.
* Strategies are *named templates*: a function `prices -> positions`, where
  `positions` is a daily target weight in [0, 1] (long/flat only).
* No look-ahead: the position decided on day *t* earns the return of day *t+1*
  (positions are shifted by one bar inside `run_backtest`).
* Transaction costs are charged on turnover (|Δposition|) in basis points.
* All performance statistics are delegated to `quant.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import quant

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Indicators used by strategy templates
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI on a close-price series."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(loss != 0, 100.0)


# ---------------------------------------------------------------------------
# Strategy templates — each returns a daily target weight in [0, 1]
# ---------------------------------------------------------------------------

def strat_sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long while SMA(fast) > SMA(slow), flat otherwise."""
    sma_f = prices.rolling(int(fast)).mean()
    sma_s = prices.rolling(int(slow)).mean()
    return (sma_f > sma_s).astype(float)


def strat_momentum(prices: pd.Series, lookback: int = 126) -> pd.Series:
    """Long while price is above its level `lookback` days ago (time-series momentum)."""
    return (prices > prices.shift(int(lookback))).astype(float)


def strat_rsi_meanrev(prices: pd.Series, window: int = 14,
                      entry: float = 30.0, exit: float = 55.0) -> pd.Series:
    """Enter long when RSI < entry, exit when RSI > exit; hold in between."""
    rsi = _rsi(prices, int(window))
    raw = pd.Series(np.nan, index=prices.index)
    raw[rsi < entry] = 1.0
    raw[rsi > exit] = 0.0
    return raw.ffill().fillna(0.0)


def strat_breakout(prices: pd.Series, window: int = 55) -> pd.Series:
    """Donchian channel breakout — long on a new `window`-day high, flat on a new low."""
    upper = prices.shift(1).rolling(int(window)).max()
    lower = prices.shift(1).rolling(int(window)).min()
    raw = pd.Series(np.nan, index=prices.index)
    raw[prices >= upper] = 1.0
    raw[prices <= lower] = 0.0
    return raw.ffill().fillna(0.0)


def strat_vol_target(prices: pd.Series, target_vol: float = 0.15,
                      window: int = 20) -> pd.Series:
    """Scale a long position so realised vol tracks `target_vol` (annualised)."""
    rets = prices.pct_change()
    rv = rets.rolling(int(window)).std() * np.sqrt(TRADING_DAYS)
    return (target_vol / rv).clip(0.0, 1.0).fillna(0.0)


STRATEGIES = {
    "sma_crossover": strat_sma_crossover,
    "momentum": strat_momentum,
    "rsi_meanrev": strat_rsi_meanrev,
    "breakout": strat_breakout,
    "vol_target": strat_vol_target,
}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def _metrics(returns: pd.Series, rf: float = 0.0) -> dict:
    """Performance summary for a daily return series — delegates to quant.py."""
    r = returns.dropna()
    dd = quant.max_drawdown(r)
    return {
        "total_return_pct": quant.total_return(r) * 100,
        "annual_return_pct": quant.annualised_return(r) * 100,
        "annual_vol_pct": quant.annualised_volatility(r) * 100,
        "sharpe": quant.sharpe_ratio(r, rf=rf),
        "sortino": quant.sortino_ratio(r, mar=rf),
        "calmar": quant.calmar_ratio(r),
        "max_drawdown_pct": dd["max_drawdown"] * 100,
        "observations": int(len(r)),
    }


def _trade_stats(held: pd.Series, strat_ret: pd.Series) -> dict:
    """Round-trip trade stats. A 'trade' is a contiguous block of non-zero exposure."""
    in_pos = held > 0
    if not in_pos.any():
        return {"trades": 0, "win_rate_pct": float("nan"),
                "avg_win_pct": float("nan"), "avg_loss_pct": float("nan"),
                "exposure_pct": 0.0}
    block = (in_pos != in_pos.shift()).cumsum()
    trade_rets = []
    for _, grp in strat_ret[in_pos].groupby(block[in_pos]):
        trade_rets.append(float((1 + grp).prod() - 1))
    wins = [t for t in trade_rets if t > 0]
    losses = [t for t in trade_rets if t <= 0]
    return {
        "trades": len(trade_rets),
        "win_rate_pct": (len(wins) / len(trade_rets) * 100) if trade_rets else float("nan"),
        "avg_win_pct": (float(np.mean(wins)) * 100) if wins else float("nan"),
        "avg_loss_pct": (float(np.mean(losses)) * 100) if losses else float("nan"),
        "exposure_pct": float(in_pos.mean() * 100),
    }


def run_backtest(prices: pd.Series, positions: pd.Series,
                 fee_bps: float = 5.0, rf: float = 0.0) -> dict:
    """Backtest a daily target-weight series against buy-and-hold.

    prices:    close-price Series indexed by date.
    positions: daily target weight in [0, 1], same index (reindexed/filled here).
    fee_bps:   per-unit-turnover transaction cost in basis points.
    rf:        annual risk-free rate (decimal) for risk-adjusted ratios.

    Returns a dict with the strategy & benchmark return series, equity curves,
    performance metrics, trade stats, and the strategy's edge over buy-and-hold.
    """
    prices = prices.dropna().astype(float)
    if len(prices) < 2:
        return {"error": "need at least 2 price observations"}

    positions = positions.reindex(prices.index).ffill().fillna(0.0).clip(0.0, 1.0)
    asset_ret = prices.pct_change().fillna(0.0)

    held = positions.shift(1).fillna(0.0)            # no look-ahead
    turnover = positions.diff().abs().fillna(positions.abs())
    cost = turnover * (fee_bps / 1e4)
    strat_ret = held * asset_ret - cost

    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + asset_ret).cumprod()

    strat_m = _metrics(strat_ret, rf=rf)
    bh_m = _metrics(asset_ret, rf=rf)

    return {
        "strategy_returns": strat_ret,
        "benchmark_returns": asset_ret,
        "equity_curve": equity,
        "benchmark_equity": bh_equity,
        "strategy": strat_m,
        "buy_and_hold": bh_m,
        "trade_stats": _trade_stats(held, strat_ret),
        "edge": {
            "excess_return_pct": strat_m["annual_return_pct"] - bh_m["annual_return_pct"],
            "sharpe_delta": strat_m["sharpe"] - bh_m["sharpe"],
        },
    }


def backtest_strategy(prices: pd.Series, strategy: str, params: dict | None = None,
                      fee_bps: float = 5.0, rf: float = 0.0,
                      oos_split: float | None = None) -> dict:
    """Generate positions from a named strategy template and backtest them.

    strategy:  one of `STRATEGIES`.
    params:    keyword args for the strategy template (defaults used if omitted).
    oos_split: if set in (0, 1), also report in-sample / out-of-sample sub-periods
               split at that fraction of the timeline (e.g. 0.7 = first 70% IS).
    """
    if strategy not in STRATEGIES:
        return {"error": f"unknown strategy '{strategy}'. "
                         f"Available: {', '.join(sorted(STRATEGIES))}"}
    prices = prices.dropna().astype(float)
    if len(prices) < 30:
        return {"error": "need at least 30 price observations to backtest"}

    params = params or {}
    try:
        positions = STRATEGIES[strategy](prices, **params)
    except TypeError as e:
        return {"error": f"bad params for '{strategy}': {e}"}

    result = run_backtest(prices, positions, fee_bps=fee_bps, rf=rf)
    result["strategy_name"] = strategy
    result["params"] = params

    if oos_split is not None and 0 < oos_split < 1:
        cut = int(len(prices) * oos_split)
        if 30 <= cut <= len(prices) - 30:
            is_bt = run_backtest(prices.iloc[:cut], positions.iloc[:cut], fee_bps, rf)
            oos_bt = run_backtest(prices.iloc[cut:], positions.iloc[cut:], fee_bps, rf)
            result["in_sample"] = is_bt["strategy"]
            result["out_of_sample"] = oos_bt["strategy"]

    return result
