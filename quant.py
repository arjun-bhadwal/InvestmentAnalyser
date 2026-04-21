"""Quantitative primitives for portfolio & market analytics.

Design principles
-----------------
* Pure functions over pandas/numpy — no I/O, no side effects.
* Every function returns numbers or structured dicts — never interpretive labels.
* Conventions:
    - `returns` is a pandas Series of simple periodic returns (pct_change).
    - Daily frequency assumed → 252 trading days / year unless explicit.
    - Risk-free rate `rf` is annualised as a decimal (e.g. 0.045 = 4.5%).
    - MAR (Minimum Acceptable Return) is annualised as a decimal.
* Sample statistics use Bessel correction (ddof=1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return statistics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Geometric cumulative return over the observed window."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() - 1)


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR — geometric annualised return.

    `(1 + total_return) ** (periods_per_year / n_obs) - 1`
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return float("nan")
    gross = float((1 + r).prod())
    if gross <= 0:
        return -1.0
    return gross ** (periods_per_year / n) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sample standard deviation × √periods_per_year."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised semi-deviation below MAR.

    Correct Sortino denominator: sqrt(mean(min(r - mar_daily, 0)^2)) × √py.
    Divides by N (not by count of negatives) — this is the canonical form.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    mar_per = mar / periods_per_year
    shortfall = np.minimum(r - mar_per, 0.0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods_per_year))


def skewness(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(stats.skew(r, bias=False)) if len(r) >= 3 else float("nan")


def excess_kurtosis(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(stats.kurtosis(r, fisher=True, bias=False)) if len(r) >= 4 else float("nan")


# ---------------------------------------------------------------------------
# Risk-adjusted ratios
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe = (mean_excess / std) × √py, on daily excess returns."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = r - rf_per
    sd = excess.std(ddof=1)
    if not sd or sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sortino ratio to MAR."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    mar_per = mar / periods_per_year
    excess_mean = (r - mar_per).mean()
    dd = downside_deviation(r, mar=mar, periods_per_year=periods_per_year)
    if not dd or dd <= 0:
        return float("nan")
    return float(excess_mean * periods_per_year / dd)


def calmar_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised return / |max drawdown|."""
    ann = annualised_return(returns, periods_per_year)
    dd = max_drawdown(returns)["max_drawdown"]
    if not dd or dd == 0 or np.isnan(dd):
        return float("nan")
    return float(ann / abs(dd))


def omega_ratio(returns: pd.Series, mar: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Omega = E[max(r - threshold, 0)] / E[max(threshold - r, 0)]."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    thresh = mar / periods_per_year
    gains = np.maximum(r - thresh, 0).sum()
    losses = np.maximum(thresh - r, 0).sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS
) -> dict:
    """Active return vs benchmark, tracking error, IR = active / TE.

    Returns dict with tracking_error_ann, active_return_ann, information_ratio.
    """
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 2:
        return {"tracking_error": float("nan"), "active_return": float("nan"), "information_ratio": float("nan")}
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    act_ann = float(active.mean() * periods_per_year)
    ir = act_ann / te if te > 0 else float("nan")
    return {"tracking_error": te, "active_return": act_ann, "information_ratio": ir}


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------

def max_drawdown(returns: pd.Series) -> dict:
    """Max drawdown magnitude, peak/trough dates, duration, recovery days.

    Returns dict with keys: max_drawdown (negative decimal), peak, trough,
    duration_days (peak→trough), recovery_days (trough→full recovery, or None if not recovered).
    """
    r = returns.dropna()
    if r.empty:
        return {"max_drawdown": float("nan"), "peak": None, "trough": None,
                "duration_days": None, "recovery_days": None}
    wealth = (1 + r).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])
    peak = wealth.loc[:trough].idxmax()
    duration = (trough - peak).days if hasattr(trough - peak, "days") else None
    # recovery: first index after trough where wealth >= wealth at peak
    peak_level = wealth.loc[peak]
    post = wealth.loc[trough:]
    recovered = post[post >= peak_level]
    if len(recovered) > 0:
        rec_date = recovered.index[0]
        recovery = (rec_date - trough).days if hasattr(rec_date - trough, "days") else None
    else:
        recovery = None
    return {"max_drawdown": max_dd, "peak": peak, "trough": trough,
            "duration_days": duration, "recovery_days": recovery}


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square of drawdowns — sensitive to depth AND duration."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    wealth = (1 + r).cumprod()
    dd = (wealth / wealth.cummax() - 1) * 100
    return float(np.sqrt((dd ** 2).mean()))


def pain_index(returns: pd.Series) -> float:
    """Mean absolute drawdown (%). Linear in depth; shallow+long > deep+short can differ from Ulcer."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    wealth = (1 + r).cumprod()
    dd = (wealth / wealth.cummax() - 1) * 100
    return float(dd.abs().mean())


# ---------------------------------------------------------------------------
# Value-at-Risk / Expected Shortfall
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Empirical quantile VaR. Reported as a negative number (loss)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(np.quantile(r, alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall — mean of returns in the left tail."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) > 0 else float(q)


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Gaussian VaR = μ + σ·Φ⁻¹(α)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    mu, sd = r.mean(), r.std(ddof=1)
    return float(mu + sd * stats.norm.ppf(alpha))


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Cornish-Fisher VaR — Gaussian VaR adjusted for skew & excess kurtosis.

    More accurate than parametric VaR when returns are non-normal.
    """
    r = returns.dropna()
    if len(r) < 4:
        return float("nan")
    mu, sd = r.mean(), r.std(ddof=1)
    s = stats.skew(r, bias=False)
    k = stats.kurtosis(r, fisher=True, bias=False)
    z = stats.norm.ppf(alpha)
    z_cf = (z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36)
    return float(mu + sd * z_cf)


def bootstrap_var(
    returns: pd.Series, alpha: float = 0.05, horizon: int = 1,
    n_sims: int = 10_000, seed: int = 42,
) -> dict:
    """Block-free bootstrap VaR & CVaR over an H-day horizon.

    Samples H-day returns with replacement from the empirical distribution —
    no parametric assumption. Returns {var, cvar, n_sims}.
    """
    r = returns.dropna().values
    if len(r) < 2:
        return {"var": float("nan"), "cvar": float("nan"), "n_sims": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(r, size=(n_sims, horizon), replace=True)
    cumulative = (1 + samples).prod(axis=1) - 1
    q = np.quantile(cumulative, alpha)
    tail = cumulative[cumulative <= q]
    return {"var": float(q), "cvar": float(tail.mean()) if len(tail) else float(q), "n_sims": n_sims}


# ---------------------------------------------------------------------------
# Factor metrics (market model: r_i = α + β·r_m + ε)
# ---------------------------------------------------------------------------

@dataclass
class FactorFit:
    alpha_ann: float       # Jensen's alpha (annualised)
    beta: float
    r_squared: float        # fraction of variance explained by benchmark
    idio_vol_ann: float     # residual (idiosyncratic) annualised volatility
    n_obs: int


def market_model(
    returns: pd.Series, benchmark: pd.Series,
    rf: float = 0.0, periods_per_year: int = TRADING_DAYS,
) -> Optional[FactorFit]:
    """OLS fit of excess returns on excess benchmark returns.

    α, β, R², and residual (idiosyncratic) volatility.
    """
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 20:
        return None
    rf_per = rf / periods_per_year
    y = (aligned.iloc[:, 0] - rf_per).values
    x = (aligned.iloc[:, 1] - rf_per).values
    x_mean, y_mean = x.mean(), y.mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).sum() / (len(x) - 1)
    var_x = ((x - x_mean) ** 2).sum() / (len(x) - 1)
    if var_x <= 0:
        return None
    beta = cov_xy / var_x
    alpha_daily = y_mean - beta * x_mean
    alpha_ann = (1 + alpha_daily) ** periods_per_year - 1
    resid = y - (alpha_daily + beta * x)
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    idio_vol = float(resid.std(ddof=1) * np.sqrt(periods_per_year))
    return FactorFit(alpha_ann=float(alpha_ann), beta=float(beta),
                     r_squared=float(r2), idio_vol_ann=idio_vol, n_obs=len(aligned))


# ---------------------------------------------------------------------------
# Portfolio-level risk decomposition
# ---------------------------------------------------------------------------

def portfolio_risk_decomposition(
    returns_df: pd.DataFrame, weights: dict, periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Marginal + component contribution to total portfolio volatility.

    For portfolio vol σ_p = √(wᵀ Σ w):
        MCTR_i = (Σ w)_i / σ_p            (marginal contribution to risk)
        CCTR_i = w_i · MCTR_i              (component contribution — sums to σ_p)
        %CTR_i = CCTR_i / σ_p              (fraction of portfolio risk)

    Returns DataFrame indexed by ticker with columns:
    weight, marginal_risk, component_risk_ann, pct_risk_contribution.
    """
    cols = [c for c in returns_df.columns if c in weights]
    if not cols:
        return pd.DataFrame()
    rets = returns_df[cols].dropna(how="all").fillna(0.0)
    if len(rets) < 2:
        return pd.DataFrame()
    w = np.array([weights[c] for c in cols], dtype=float)
    cov = rets.cov().values * periods_per_year  # annualised covariance
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.DataFrame()
    port_vol = np.sqrt(port_var)
    mctr = (cov @ w) / port_vol
    cctr = w * mctr
    pct = cctr / port_vol
    return pd.DataFrame(
        {"weight": w, "marginal_risk": mctr, "component_risk_ann": cctr, "pct_risk_contribution": pct},
        index=cols,
    ).sort_values("pct_risk_contribution", ascending=False)


def portfolio_returns_from_weights(returns_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted portfolio return series. Weights are renormalised over available cols."""
    cols = [c for c in returns_df.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)
    w = np.array([weights[c] for c in cols], dtype=float)
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    return returns_df[cols].fillna(0.0).dot(w)


# ---------------------------------------------------------------------------
# Breadth & momentum (used by sentiment / rotation)
# ---------------------------------------------------------------------------

def pct_above_ma(closes: pd.DataFrame, window: int) -> float:
    """Fraction of columns whose latest close is above their `window`-day SMA."""
    if closes.empty:
        return float("nan")
    above = 0
    total = 0
    for col in closes.columns:
        s = closes[col].dropna()
        if len(s) < window + 1:
            continue
        total += 1
        if float(s.iloc[-1]) > float(s.rolling(window).mean().iloc[-1]):
            above += 1
    if total == 0:
        return float("nan")
    return above / total


def new_highs_lows(closes: pd.DataFrame, window: int = 252) -> dict:
    """Count of names at a new `window`-day high / low on the latest bar."""
    highs = lows = total = 0
    for col in closes.columns:
        s = closes[col].dropna()
        if len(s) < window:
            continue
        total += 1
        last = float(s.iloc[-1])
        if last >= float(s.tail(window).max()):
            highs += 1
        if last <= float(s.tail(window).min()):
            lows += 1
    return {"new_highs": highs, "new_lows": lows, "universe_size": total}


def historical_percentile(value: float, series: pd.Series) -> float:
    """Percentile rank (0-100) of `value` within a historical series."""
    s = series.dropna()
    if s.empty or value is None or np.isnan(value):
        return float("nan")
    return float((s <= value).mean() * 100)


# ---------------------------------------------------------------------------
# Relative Rotation Graph (RRG) — JdK RS-Ratio & RS-Momentum
# ---------------------------------------------------------------------------

def jdk_rs_ratio(security: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    """JdK RS-Ratio — normalised relative strength, centred on 100.

    Standard Bloomberg / Stockcharts formulation:
        rs_raw = security / benchmark
        rs_ratio = 100 + ((rs_raw - SMA(rs_raw, window)) / STD(rs_raw, window)) × 10 + ???

    We use the canonical Julius-de-Kempenaer smoothed form:
        ratio = (security / benchmark)
        rs = 100 * ratio / SMA(ratio, window)         (% of trailing mean)
        rs_ratio = 100 + z-score(rs, window) * sigma_scale

    A simpler robust variant (used here) — expresses current RS as a Z-score of
    the security/benchmark ratio over the window, rescaled so ≈100 is parity:

        z_t = (rs_t - mean(rs, window)) / std(rs, window)
        rs_ratio_t = 100 + z_t

    Values > 100 → outperforming benchmark (in z-units over window).
    """
    aligned = pd.concat([security, benchmark], axis=1).dropna()
    if len(aligned) < window + 1:
        return pd.Series(dtype=float)
    ratio = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    roll_mean = ratio.rolling(window).mean()
    roll_std = ratio.rolling(window).std(ddof=1)
    z = (ratio - roll_mean) / roll_std.replace(0, np.nan)
    return 100 + z


def jdk_rs_momentum(rs_ratio: pd.Series, window: int = 21) -> pd.Series:
    """JdK RS-Momentum — rate of change of the RS-Ratio, centred on 100.

    rs_mom_t = 100 + (rs_ratio_t - SMA(rs_ratio, window))
    """
    if rs_ratio.empty:
        return rs_ratio
    sma = rs_ratio.rolling(window).mean()
    return 100 + (rs_ratio - sma)


def rrg_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Classify into canonical RRG quadrant. Factual — no editorial spin."""
    if np.isnan(rs_ratio) or np.isnan(rs_momentum):
        return "n/a"
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "leading"       # strong RS, positive momentum
    if rs_ratio >= 100 and rs_momentum < 100:
        return "weakening"     # strong RS but decelerating
    if rs_ratio < 100 and rs_momentum < 100:
        return "lagging"       # weak RS, negative momentum
    return "improving"         # weak RS but accelerating


def relative_strength_snapshot(
    security: pd.Series, benchmark: pd.Series,
    rs_window: int = 63, mom_window: int = 21,
) -> dict:
    """Single-point RRG reading for one (security, benchmark) pair."""
    rs = jdk_rs_ratio(security, benchmark, window=rs_window)
    if rs.empty:
        return {"rs_ratio": float("nan"), "rs_momentum": float("nan"),
                "quadrant": "n/a", "rs_window": rs_window, "mom_window": mom_window}
    mom = jdk_rs_momentum(rs, window=mom_window)
    rs_last = float(rs.iloc[-1]) if not rs.empty else float("nan")
    mom_last = float(mom.iloc[-1]) if not mom.empty else float("nan")
    return {
        "rs_ratio": rs_last,
        "rs_momentum": mom_last,
        "quadrant": rrg_quadrant(rs_last, mom_last),
        "rs_window": rs_window,
        "mom_window": mom_window,
    }


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

def hhi(weights: list[float]) -> float:
    """Herfindahl–Hirschman Index = Σ w²."""
    if not weights:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    return float((w ** 2).sum())


def effective_n(weights: list[float]) -> float:
    """1 / HHI — effective number of equally-weighted positions."""
    h = hhi(weights)
    return float(1 / h) if h and h > 0 else float("nan")
