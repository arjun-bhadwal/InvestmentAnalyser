"""
Tool registration — importing each module triggers @mcp.tool() decoration.

This server is a deterministic quant + data engine — price feeds, fundamentals,
trades, and quantitative analysis. News, headlines, and qualitative research are
intentionally NOT handled here; they belong to the orchestrating agent's web search.

Layer 1 — Context bundles (default path, single-call answers):
  context.get_portfolio_context, get_ticker_context, get_opportunity_context

Layer 2 — Drill-downs (explicit deep-dives):
  risk.analyze_portfolio, risk.calculate_position_size
  market_data.get_fundamentals, get_market_status
  macro.get_macro_summary
  analysis.get_earnings_calendar
  portfolio.get_account_history, get_open_orders, get_pies
  strategy.backtest_strategy, evaluate_signal, optimize_portfolio

Layer 3 — Internal helpers (prefix _):
  All demoted single-purpose tools — called only from bundles/drill-downs.
"""
from tools import context        # noqa: F401  ← bundles (register first)
from tools import portfolio      # noqa: F401
from tools import market_data    # noqa: F401
from tools import analysis       # noqa: F401
from tools import macro          # noqa: F401
from tools import risk           # noqa: F401
from tools import insider        # noqa: F401
from tools import strategy       # noqa: F401  ← quant layer (backtest/signal/optimize)
