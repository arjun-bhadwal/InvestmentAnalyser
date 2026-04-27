"""Portfolio risk, stress testing, position sizing, and allocation.

Data-only: returns numbers, not opinions.
All statistics computed via `quant.py` primitives.
"""
import asyncio

import numpy as np
import pandas as pd
import yfinance as yf

import app
import quant
from helpers import strip_t212_ticker, position_value

mcp = app.mcp


def _risk_free_annual(default: float = 0.045) -> float:
    """10Y Treasury yield (^TNX) expressed as annual decimal. Falls back to default."""
    import sys
    from contextlib import redirect_stdout
    try:
        with redirect_stdout(sys.stderr):
            val = yf.Ticker("^TNX").fast_info.last_price
        return float(val) / 100 if val is not None else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Portfolio Risk Analytics
# ---------------------------------------------------------------------------

async def _compute_portfolio_risk(weights_raw: dict[str, float]) -> str:
    """Core portfolio risk computation engine taking explicit weights."""
    tickers = list(weights_raw.keys())
    if not tickers:
        return "No tickers to analyse."

    from helpers import fetch_historic_prices
    try:
        closes = await fetch_historic_prices(tickers + ["SPY"], period="1y", interval="1d")
    except Exception as e:
        return f"Error fetching price history: {e}"

    if closes.empty:
        return "No price data returned."

    closes = closes.dropna(axis=1, how="all")
    returns = closes.pct_change().iloc[1:].replace([np.inf, -np.inf], np.nan)

    rf = _risk_free_annual()

    available = [t for t in tickers if t in returns.columns and returns[t].dropna().shape[0] >= 30]
    missing = [t for t in tickers if t not in available]

    lines = [
        f"**Portfolio Risk Analytics — 1Y daily, rf={rf*100:.2f}% (^TNX)**\n",
    ]
    if missing:
        lines.append(f"Insufficient data (<30 bars): {', '.join(missing)}\n")

    if not available:
        return "No tickers had sufficient price history (need 30+ trading days)."

    spy_ret = returns["SPY"] if "SPY" in returns.columns else pd.Series(dtype=float)

    # Per-holding table
    lines.append(
        f"{'Ticker':<10} {'CAGR%':>8} {'Vol%':>7} {'Shrp':>6} {'Sort':>6} "
        f"{'MaxDD%':>8} {'Beta':>6} {'R²':>6} {'Alpha%':>8} {'Skew':>6} {'ExKurt':>7}"
    )
    lines.append("-" * 90)

    for t in available:
        r = returns[t].dropna()
        cagr = quant.annualised_return(r) * 100
        vol = quant.annualised_volatility(r) * 100
        sh = quant.sharpe_ratio(r, rf=rf)
        so = quant.sortino_ratio(r, mar=rf)
        dd = quant.max_drawdown(r)["max_drawdown"] * 100
        skew = quant.skewness(r)
        exkurt = quant.excess_kurtosis(r)

        fit = quant.market_model(r, spy_ret, rf=rf) if not spy_ret.empty else None
        beta = fit.beta if fit else float("nan")
        r2 = fit.r_squared if fit else float("nan")
        alpha = fit.alpha_ann * 100 if fit else float("nan")

        def _f(v, d=2):
            return f"{v:>.{d}f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"

        lines.append(
            f"{t:<10} {_f(cagr,2):>8} {_f(vol,2):>7} {_f(sh,2):>6} {_f(so,2):>6} "
            f"{_f(dd,2):>8} {_f(beta,2):>6} {_f(r2,2):>6} {_f(alpha,2):>8} {_f(skew,2):>6} {_f(exkurt,2):>7}"
        )

    # Portfolio-level
    total_val = sum(weights_raw.values()) or 1
    weights = {k: v / total_val for k, v in weights_raw.items() if k in available}
    w_sum = sum(weights.values()) or 1
    weights = {k: v / w_sum for k, v in weights.items()}

    port_ret = quant.portfolio_returns_from_weights(returns[available], weights).dropna()
    if len(port_ret) > 5:
        cagr = quant.annualised_return(port_ret) * 100
        vol = quant.annualised_volatility(port_ret) * 100
        sh = quant.sharpe_ratio(port_ret, rf=rf)
        so = quant.sortino_ratio(port_ret, mar=rf)
        calmar = quant.calmar_ratio(port_ret)
        omega = quant.omega_ratio(port_ret, mar=rf)
        dd_info = quant.max_drawdown(port_ret)
        dd_pct = dd_info["max_drawdown"] * 100
        ulcer = quant.ulcer_index(port_ret)
        pain = quant.pain_index(port_ret)

        var_h = quant.historical_var(port_ret, alpha=0.05) * 100
        cvar_h = quant.historical_cvar(port_ret, alpha=0.05) * 100
        var_cf = quant.cornish_fisher_var(port_ret, alpha=0.05) * 100
        skew = quant.skewness(port_ret)
        exkurt = quant.excess_kurtosis(port_ret)

        fit = quant.market_model(port_ret, spy_ret, rf=rf) if not spy_ret.empty else None
        ir = quant.information_ratio(port_ret, spy_ret) if not spy_ret.empty else None

        lines.append("-" * 90)
        lines.append("\n**Portfolio Summary**")
        lines.append(f"- CAGR:                 {cagr:+.2f}%")
        lines.append(f"- Annualised vol:       {vol:.2f}%")
        lines.append(f"- Sharpe:               {sh:.2f}")
        lines.append(f"- Sortino (MAR=rf):     {so:.2f}")
        lines.append(f"- Calmar:               {calmar:.2f}")
        lines.append(f"- Omega (threshold=rf): {omega:.2f}")
        lines.append(f"- Max drawdown:         {dd_pct:+.2f}%"
                     + (f" (duration {dd_info['duration_days']}d, "
                        f"{'recovered in ' + str(dd_info['recovery_days']) + 'd' if dd_info['recovery_days'] is not None else 'not recovered'})"
                        if dd_info.get('duration_days') is not None else ""))
        lines.append(f"- Ulcer Index:          {ulcer:.2f}")
        lines.append(f"- Pain Index:           {pain:.2f}%")
        lines.append(f"- Skew / Excess kurt:   {skew:+.2f} / {exkurt:+.2f}")
        lines.append(f"- VaR 95% (hist):       {var_h:.2f}%  daily")
        lines.append(f"- CVaR 95% (hist):      {cvar_h:.2f}%  daily")
        lines.append(f"- VaR 95% (C-F):        {var_cf:.2f}%  daily  (skew/kurt adjusted)")
        if fit:
            lines.append(
                f"- β / α / R² vs SPY:    {fit.beta:.2f} / {fit.alpha_ann*100:+.2f}% / {fit.r_squared:.2f}"
            )
            lines.append(f"- Idiosyncratic vol:    {fit.idio_vol_ann*100:.2f}%")
        if ir:
            lines.append(
                f"- Tracking error / IR:  {ir['tracking_error']*100:.2f}% / {ir['information_ratio']:.2f}"
            )

        # Risk decomposition
        decomp = quant.portfolio_risk_decomposition(returns[available], weights)
        if not decomp.empty:
            lines.append("\n**Risk Decomposition (annualised)**")
            lines.append(f"{'Ticker':<10} {'Weight%':>8} {'MCTR%':>8} {'CCTR%':>8} {'%Risk':>7}")
            lines.append("-" * 46)
            for idx, row in decomp.iterrows():
                lines.append(
                    f"{idx:<10} {row['weight']*100:>7.1f}% "
                    f"{row['marginal_risk']*100:>7.2f}% {row['component_risk_ann']*100:>7.2f}% "
                    f"{row['pct_risk_contribution']*100:>6.1f}%"
                )

    if 2 <= len(available) <= 10:
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
    ticker: str, entry_price: float, stop_loss_price: float,
    risk_pct: float = 2.0, target_price: float | None = None,
    win_probability: float | None = None,
) -> str:
    """Calculate position size from explicit risk-per-trade rules. Returns raw numbers only.

    ticker: symbol (used only for label).
    entry_price / stop_loss_price: trade levels.
    risk_pct: percent of portfolio to risk on this trade (default 2%).
    target_price: optional — if supplied with win_probability, Kelly fraction is computed.
    win_probability: optional decimal in (0,1). MUST be user-supplied — no heuristic estimate.
    """
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
    r_multiple = ((target_price - entry_price) / risk_per_share) if target_price else None

    ccy = acct.get("currency", "")
    lines = [
        f"**Position Size — {ticker.upper()}**\n",
        f"- Entry: {ccy} {entry_price:,.4f}  Stop: {ccy} {stop_loss_price:,.4f}  Risk/share: {ccy} {risk_per_share:,.4f}",
        f"- Direction: {'LONG' if entry_price > stop_loss_price else 'SHORT'}",
        "",
        f"**Fixed-fractional sizing**",
        f"- Portfolio: {ccy} {total_value:,.2f}   Max risk @ {risk_pct}%: {ccy} {max_risk:,.2f}",
        f"- Shares: {shares:,}   Position value: {ccy} {pos_val:,.2f}   Weight: {pos_pct:.2f}%",
    ]

    if target_price is not None and r_multiple is not None:
        reward_per_share = abs(target_price - entry_price)
        lines.append("")
        lines.append(f"**Reward / risk**")
        lines.append(f"- Target: {ccy} {target_price:,.4f}  Reward/share: {ccy} {reward_per_share:,.4f}")
        lines.append(f"- R-multiple: {r_multiple:.2f}R")

    if win_probability is not None and target_price is not None:
        p = max(0.0, min(1.0, float(win_probability)))
        b = abs(target_price - entry_price) / risk_per_share
        kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
        k_shares = int(kelly * total_value / entry_price) if entry_price > 0 else 0
        lines += [
            "",
            f"**Kelly (user-supplied p={p:.2f})**",
            f"- Full Kelly fraction: {kelly*100:.2f}%  ({k_shares:,} shares)",
            f"- Half Kelly:          {kelly*50:.2f}%  ({k_shares//2:,} shares)",
        ]
    elif target_price is None or win_probability is None:
        lines.append("")
        lines.append("_Kelly requires both `target_price` and `win_probability` — not auto-estimated._")

    lines.append("")
    lines.append(f"Position weight: {pos_pct:.2f}% of portfolio.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Stress Test
# ---------------------------------------------------------------------------

async def _compute_portfolio_stress_test(weights_raw: dict[str, float], simulations: int = 10_000) -> str:
    """Core portfolio stress test engine taking explicit weights.
    Bootstrap resamples the empirical return distribution — no normality assumption.
    """
    tickers = list(weights_raw.keys())
    if not tickers:
        return "No tickers."
    total = sum(weights_raw.values()) or 1
    weights = {k: v / total for k, v in weights_raw.items()}

    from helpers import fetch_historic_prices
    chosen_period = "max"
    closes = pd.DataFrame()
    for p in ("max", "10y", "5y", "2y", "1y", "6mo"):
        closes = await fetch_historic_prices(tickers, period=p, interval="1d")
        if not closes.empty:
            avail = [t for t in tickers if t in closes.columns and closes[t].dropna().shape[0] >= 30]
            if len(avail) >= max(1, len(tickers) // 2):
                chosen_period = p
                break

    if closes.empty:
        return "No price data returned."

    returns = closes.pct_change().iloc[1:].replace([np.inf, -np.inf], np.nan)
    avail = [t for t in tickers if t in returns.columns and returns[t].dropna().shape[0] >= 30]
    missing = [t for t in tickers if t not in avail]

    lines = [f"**Portfolio Stress Test — lookback {chosen_period}**\n"]
    if missing:
        lines.append(f"Insufficient data: {', '.join(missing)}\n")

    if not avail:
        return "No tickers had sufficient price history for stress testing."

    earliest = returns.index.min().strftime("%Y-%m-%d")
    lines.append(f"_Data from {earliest}_\n")

    scenarios = {
        "2008 GFC (Sep 08 – Mar 09)":  ("2008-09-01", "2009-03-09"),
        "COVID crash (Feb – Mar 20)":  ("2020-02-19", "2020-03-23"),
        "2022 rate-hike selloff":      ("2022-01-03", "2022-10-12"),
        "2024 Aug VIX spike":          ("2024-07-16", "2024-08-05"),
    }

    lines.append("**Historical scenarios** (weights held fixed, returns compounded)")
    lines.append(f"{'Scenario':<32} {'Impact':>10}")
    lines.append("-" * 44)
    for name, (s, e) in scenarios.items():
        try:
            pr = returns.loc[s:e]
            if pr.empty or len(pr) < 2:
                lines.append(f"{name:<32} {'no data':>10}")
                continue
            # Proper compounding of weighted daily returns
            w_ret = quant.portfolio_returns_from_weights(pr[[t for t in avail if t in pr.columns]],
                                                         {t: weights.get(t, 0) for t in avail})
            port_ret = float((1 + w_ret).prod() - 1)
            lines.append(f"{name:<32} {port_ret*100:>+9.1f}%")
        except Exception:
            lines.append(f"{name:<32} {'N/A':>10}")

    port_ret = quant.portfolio_returns_from_weights(returns[avail],
                                                    {t: weights.get(t, 0) for t in avail}).dropna()

    if len(port_ret) >= 20:
        mu = float(port_ret.mean())
        sig = float(port_ret.std(ddof=1))

        # Parametric (normal) 1-year MC
        lines.append(f"\n**Monte Carlo — parametric normal ({simulations:,} paths × 252d)**")
        if sig > 0:
            rng = np.random.default_rng(42)
            sims = (1 + rng.normal(mu, sig, size=(simulations, 252))).prod(axis=1) - 1
            qs = np.quantile(sims, [0.05, 0.25, 0.50, 0.75, 0.95])
            lines += [
                f"- 5th pctile:   {qs[0]*100:+.1f}%",
                f"- 25th pctile:  {qs[1]*100:+.1f}%",
                f"- Median:       {qs[2]*100:+.1f}%",
                f"- 75th pctile:  {qs[3]*100:+.1f}%",
                f"- 95th pctile:  {qs[4]*100:+.1f}%",
                f"- P(loss):      {(sims < 0).mean()*100:.1f}%",
                f"- P(>+20%):     {(sims > 0.20).mean()*100:.1f}%",
                f"- P(<-20%):     {(sims < -0.20).mean()*100:.1f}%",
            ]
        else:
            lines.append("zero volatility — skipped")

        # Bootstrap (empirical) MC
        lines.append(f"\n**Monte Carlo — bootstrap ({simulations:,} paths × 252d, empirical dist)**")
        rng = np.random.default_rng(43)
        samples = rng.choice(port_ret.values, size=(simulations, 252), replace=True)
        b_sims = (1 + samples).prod(axis=1) - 1
        bqs = np.quantile(b_sims, [0.05, 0.25, 0.50, 0.75, 0.95])
        lines += [
            f"- 5th pctile:   {bqs[0]*100:+.1f}%",
            f"- 25th pctile:  {bqs[1]*100:+.1f}%",
            f"- Median:       {bqs[2]*100:+.1f}%",
            f"- 75th pctile:  {bqs[3]*100:+.1f}%",
            f"- 95th pctile:  {bqs[4]*100:+.1f}%",
            f"- P(loss):      {(b_sims < 0).mean()*100:.1f}%",
            f"- P(>+20%):     {(b_sims > 0.20).mean()*100:.1f}%",
            f"- P(<-20%):     {(b_sims < -0.20).mean()*100:.1f}%",
        ]

        # 1-day VaR at different alphas
        lines.append("\n**1-day VaR / CVaR from empirical distribution**")
        for a in (0.01, 0.05):
            v_h = quant.historical_var(port_ret, alpha=a) * 100
            cv_h = quant.historical_cvar(port_ret, alpha=a) * 100
            v_cf = quant.cornish_fisher_var(port_ret, alpha=a) * 100
            lines.append(f"- α={int(a*100)}%:  hist VaR {v_h:.2f}%   CVaR {cv_h:.2f}%   C-F VaR {v_cf:.2f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Allocation
# ---------------------------------------------------------------------------

async def _compute_portfolio_allocation(weights_raw: dict[str, float]) -> str:
    """Core portfolio allocation engine."""
    holdings = [{"ticker": t, "value": v} for t, v in weights_raw.items()]
    if not holdings:
        return "No positions."

    total_val = sum(h["value"] for h in holdings) or 1

    async def _meta(ticker):
        def _f():
            import sys
            from contextlib import redirect_stdout
            with redirect_stdout(sys.stderr):
                return yf.Ticker(ticker).info
        try:
            info = await asyncio.to_thread(_f)
            return {"sector": info.get("sector", "Unknown"),
                    "country": info.get("country", "Unknown"),
                    "marketCap": float(info.get("marketCap", 0) or 0),
                    "name": info.get("shortName") or ticker}
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
            lines.append(f"  {name:<22} {w:>5.1f}%")

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
    sorted_w = sorted(wts, reverse=True)
    top1 = sorted_w[0] * 100 if sorted_w else 0
    top5 = sum(sorted_w[:5]) * 100
    top10 = sum(sorted_w[:10]) * 100
    hhi = quant.hhi(wts)
    eff_n = quant.effective_n(wts)

    lines += [
        f"\n**Concentration**",
        f"- Positions: {len(holdings)}",
        f"- Top 1 / 5 / 10: {top1:.1f}% / {top5:.1f}% / {top10:.1f}%",
        f"- HHI: {hhi:.4f}",
        f"- Effective N (1/HHI): {eff_n:.1f}",
    ]

    return "\n".join(lines)


async def _get_portfolio_stress_test(simulations: int = 10_000) -> str:
    """Live T212 portfolio stress test."""
    try:
        positions = await app.t212.get_portfolio()
    except Exception as e:
        return f"Error fetching portfolio: {e}"
    if not positions:
        return "No open positions."
    weights_raw = {}
    for p in positions:
        t = strip_t212_ticker(p["ticker"])
        weights_raw[t] = weights_raw.get(t, 0) + position_value(p)
    return await _compute_portfolio_stress_test(weights_raw, simulations=simulations)

async def _get_portfolio_allocation() -> str:
    """Live T212 portfolio allocation."""
    try:
        positions = await app.t212.get_portfolio()
    except Exception as e:
        return f"Error: {e}"
    if not positions:
        return "No open positions."
    weights_raw = {}
    for p in positions:
        t = strip_t212_ticker(p["ticker"])
        weights_raw[t] = weights_raw.get(t, 0) + position_value(p)
    return await _compute_portfolio_allocation(weights_raw)

# ===========================================================================
# Consolidated analyze_portfolio & analyze_scenario endpoints
# ===========================================================================

@mcp.tool()
async def analyze_scenario(tickers: str, weights: str, metrics: str = "risk,stress,allocation", initial_value: float = 10000.0, simulations: int = 10_000) -> str:
    """Run quantitative analysis (Risk, Monte Carlo stress testing) on a custom hypothetical portfolio.
    
    tickers: comma-separated symbols (e.g. 'AAPL, MSFT, GOOG')
    weights: explicit weights per ticker (e.g. '50, 25, 25' or '0.5, 0.25, 0.25'). MUST MATCH TICKERS EXACTLY.
    metrics: 'risk', 'stress', 'allocation' (comma separated)
    """
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not syms:
        return "Error: Provide at least one ticker."

    w_strs = [w.strip().replace('%', '') for w in weights.split(",") if w.strip()]
    
    if len(syms) != len(w_strs):
        return f"Error: Mismatch between tickers count ({len(syms)}) and weights count ({len(w_strs)}). You must provide an explicit weight for every ticker."
        
    weights_raw = {}
    for s, w_str in zip(syms, w_strs):
        try:
            val = float(w_str)
            if val <= 0:
                return f"Error: Weight for {s} must be > 0 (got {val})"
            weights_raw[s] = val * initial_value
        except ValueError:
            return f"Error: Invalid weight '{w_str}' provided for {s}."

    m_list = [m.strip().lower() for m in metrics.split(",")]
    lines = [f"**Custom Scenario Analysis: {len(syms)} assets**\n"]

    if "risk" in m_list:
        lines.append(await _compute_portfolio_risk(weights_raw))
    if "allocation" in m_list:
        if lines: lines.append("\n")
        lines.append(await _compute_portfolio_allocation(weights_raw))
    if "stress" in m_list:
        if lines: lines.append("\n")
        lines.append(await _compute_portfolio_stress_test(weights_raw, simulations=simulations))

    return "\n".join(lines)

@mcp.tool()
async def analyze_portfolio(metrics: str = "risk,stress,allocation", simulations: int = 10_000) -> str:
    """Full quantitative analysis of your actual portfolio.
    metrics: 'risk' (vol, Sharpe/Sortino/Calmar/Omega, drawdowns, VaR/CVaR/C-F, β/α/R², risk decomp) |
             'stress' (historical scenarios + parametric + bootstrap MC) |
             'allocation' (sector/geo/cap, HHI, effective N)
    """
    m_list = [m.strip().lower() for m in metrics.split(",")]
    lines = []

    if "risk" in m_list:
        lines.append(await _get_portfolio_risk())
    if "allocation" in m_list:
        if lines: lines.append("\n")
        lines.append(await _get_portfolio_allocation())
    if "stress" in m_list:
        if lines: lines.append("\n")
        lines.append(await _get_portfolio_stress_test(simulations=simulations))

    return "\n".join(lines)
