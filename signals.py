"""Signal evaluation — does an indicator actually predict forward returns?

Design principles (mirrors `quant.py`)
--------------------------------------
* Pure functions over pandas/numpy — no I/O, no side effects.
* A `signal` is a Series indexed by date — higher value = more bullish.
* `forward_returns` are returns realised *after* the signal is observed, so
  every function aligns by shifting price returns backward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def forward_returns(prices: pd.Series, horizon: int = 21) -> pd.Series:
    """Return realised over the next `horizon` days, indexed at the decision date."""
    prices = prices.dropna().astype(float)
    fwd = prices.shift(-horizon) / prices - 1
    return fwd


def information_coefficient(signal: pd.Series, prices: pd.Series,
                            horizon: int = 21) -> dict:
    """Rank correlation (Spearman) between a signal and forward returns.

    IC is the standard quant measure of predictive power: ~0 = no edge,
    >0.05 = economically meaningful for a daily signal.
    """
    fwd = forward_returns(prices, horizon)
    aligned = pd.concat([signal, fwd], axis=1, sort=True).dropna()
    if len(aligned) < 10:
        return {"ic": float("nan"), "p_value": float("nan"), "n": int(len(aligned))}
    ic, p = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return {"ic": float(ic), "p_value": float(p), "n": int(len(aligned)), "horizon": horizon}


def quantile_returns(signal: pd.Series, prices: pd.Series, horizon: int = 21,
                     n_quantiles: int = 5) -> pd.DataFrame:
    """Mean forward return per signal quantile.

    A predictive signal shows a monotonic spread — top quantile clearly above
    bottom quantile. Returns a DataFrame indexed by quantile (1 = lowest signal).
    """
    fwd = forward_returns(prices, horizon)
    aligned = pd.concat([signal, fwd], axis=1, sort=True).dropna()
    aligned.columns = ["signal", "fwd"]
    if len(aligned) < n_quantiles * 2:
        return pd.DataFrame()
    try:
        buckets = pd.qcut(aligned["signal"], n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    grp = aligned.groupby(buckets)["fwd"]
    out = pd.DataFrame({
        "mean_fwd_return_pct": grp.mean() * 100,
        "hit_rate_pct": grp.apply(lambda s: (s > 0).mean() * 100),
        "count": grp.size(),
    })
    out.index = [f"Q{int(i) + 1}" for i in out.index]
    return out


def hit_rate(signal: pd.Series, prices: pd.Series, horizon: int = 21) -> float:
    """Fraction of observations where sign(signal) matches sign(forward return)."""
    fwd = forward_returns(prices, horizon)
    aligned = pd.concat([signal, fwd], axis=1, sort=True).dropna()
    if aligned.empty:
        return float("nan")
    s, f = np.sign(aligned.iloc[:, 0]), np.sign(aligned.iloc[:, 1])
    return float((s == f).mean() * 100)


def signal_decay(signal: pd.Series, prices: pd.Series,
                 horizons: tuple[int, ...] = (1, 5, 21, 63)) -> pd.DataFrame:
    """IC measured at several horizons — shows how fast the edge decays."""
    rows = {h: information_coefficient(signal, prices, h) for h in horizons}
    return pd.DataFrame(rows).T[["ic", "p_value", "n"]]


def evaluate_signal(signal: pd.Series, prices: pd.Series, horizon: int = 21,
                    n_quantiles: int = 5) -> dict:
    """Full signal evaluation: IC, quantile spread, hit rate, decay profile."""
    signal = signal.dropna()
    if len(signal) < 30:
        return {"error": "need at least 30 signal observations"}
    qr = quantile_returns(signal, prices, horizon, n_quantiles)
    spread = float("nan")
    if not qr.empty and len(qr) >= 2:
        spread = float(qr["mean_fwd_return_pct"].iloc[-1] - qr["mean_fwd_return_pct"].iloc[0])
    return {
        "ic": information_coefficient(signal, prices, horizon),
        "quantile_returns": qr,
        "top_minus_bottom_pct": spread,
        "hit_rate_pct": hit_rate(signal, prices, horizon),
        "decay": signal_decay(signal, prices),
        "horizon": horizon,
    }
