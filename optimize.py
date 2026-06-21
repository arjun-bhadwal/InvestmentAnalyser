"""Portfolio optimization — mean-variance, minimum-variance, risk parity.

Design principles (mirrors `quant.py`)
--------------------------------------
* Pure functions over pandas/numpy/scipy — no I/O, no side effects.
* Input is a returns DataFrame: columns = tickers, rows = periodic (daily) returns.
* Long-only by default (weights bounded [0, 1]); pass `bounds` to change.
* Statistics annualised to 252 trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _annualised(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Annualised mean-return vector and covariance matrix."""
    clean = returns_df.dropna(how="all").fillna(0.0)
    cols = list(clean.columns)
    mu = clean.mean().values * TRADING_DAYS
    cov = clean.cov().values * TRADING_DAYS
    return mu, cov, cols


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     rf: float = 0.0) -> dict:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return {"expected_return_pct": ret * 100, "volatility_pct": vol * 100,
            "sharpe": sharpe}


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio variance."""
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.full_like(w, np.nan)
    return w * (cov @ w) / port_var


def _solve(objective, n: int, bounds, extra_constraints=None) -> np.ndarray | None:
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        cons.extend(extra_constraints)
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x if res.success else None


def optimize_portfolio(returns_df: pd.DataFrame, objective: str = "max_sharpe",
                       rf: float = 0.045, weight_bounds: tuple[float, float] = (0.0, 1.0)) -> dict:
    """Optimal weights for a returns DataFrame.

    objective: 'max_sharpe' | 'min_variance' | 'risk_parity'
    rf:        annual risk-free rate (decimal) — used by max_sharpe.
    weight_bounds: per-asset (min, max) weight.

    Returns a dict with weights, portfolio stats, and risk contributions.
    """
    if returns_df is None or returns_df.shape[1] < 2:
        return {"error": "need at least 2 assets"}
    mu, cov, cols = _annualised(returns_df)
    n = len(cols)
    if len(returns_df.dropna(how="all")) < 20:
        return {"error": "need at least 20 return observations"}
    bounds = [weight_bounds] * n

    if objective == "min_variance":
        w = _solve(lambda w: w @ cov @ w, n, bounds)
    elif objective == "risk_parity":
        def rp_obj(w):
            rc = _risk_contributions(w, cov)
            return float(np.sum((rc - rc.mean()) ** 2))
        w = _solve(rp_obj, n, bounds)
    elif objective == "max_sharpe":
        def neg_sharpe(w):
            vol = np.sqrt(w @ cov @ w)
            return -((w @ mu - rf) / vol) if vol > 0 else 1e6
        w = _solve(neg_sharpe, n, bounds)
    else:
        return {"error": f"unknown objective '{objective}'. "
                         f"Use max_sharpe | min_variance | risk_parity"}

    if w is None:
        return {"error": f"optimizer failed to converge for '{objective}'"}

    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else w
    rc = _risk_contributions(w, cov)
    return {
        "objective": objective,
        "weights": {c: float(round(wi, 6)) for c, wi in zip(cols, w)},
        "portfolio": _portfolio_stats(w, mu, cov, rf=rf),
        "risk_contribution_pct": {c: float(r * 100) for c, r in zip(cols, rc)},
    }


def efficient_frontier(returns_df: pd.DataFrame, n_points: int = 15,
                       weight_bounds: tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
    """Minimum-variance portfolio at each of `n_points` target returns.

    Returns a DataFrame with columns: target_return_pct, volatility_pct, sharpe.
    """
    if returns_df is None or returns_df.shape[1] < 2:
        return pd.DataFrame()
    mu, cov, cols = _annualised(returns_df)
    n = len(cols)
    bounds = [weight_bounds] * n
    targets = np.linspace(mu.min(), mu.max(), n_points)

    rows = []
    for t in targets:
        cons = [{"type": "eq", "fun": lambda w, t=t: w @ mu - t}]
        w = _solve(lambda w: w @ cov @ w, n, bounds, extra_constraints=cons)
        if w is None:
            continue
        vol = float(np.sqrt(w @ cov @ w))
        rows.append({"target_return_pct": float(t * 100),
                     "volatility_pct": vol * 100,
                     "sharpe": (float(t) / vol) if vol > 0 else float("nan")})
    return pd.DataFrame(rows)
