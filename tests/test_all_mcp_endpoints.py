import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from the project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

import app
from tools.context import get_portfolio_context, get_ticker_context, get_opportunity_context
from tools.macro import get_macro_summary
from tools.portfolio import get_account_history, get_open_orders, get_pies
from tools.market_data import get_market_status, get_fundamentals, get_price_history
from tools.analysis import get_earnings_calendar
from tools.risk import analyze_portfolio, calculate_position_size

# Output file will be placed in the project root
OUT_FILE = PROJECT_ROOT / 'mcp_test_results.txt'

async def run_test(name, coro, f):
    f.write(f"\n{'='*80}\n[TEST] {name}\n{'='*80}\n")
    try:
        res = await coro
        if isinstance(res, str):
            f.write(res + "\n")
        else:
            f.write(f"Result type: {type(res)}\n")
    except Exception as e:
        f.write(f"❌ CRITICAL FAILURE: {e}\n")

async def main():
    print(f"Starting test suite. Output will be written to {OUT_FILE}")
    
    with open(OUT_FILE, 'w') as f:
        f.write("Investment Analyser MCP - Master Test Suite Results\n\n")
        
    async with app.lifespan(app.mcp):
        tasks = [
            ("Portfolio Context (1w)", get_portfolio_context(horizon="1w")),
            ("Ticker Context (AAPL, standard)", get_ticker_context("AAPL", "standard")),
            ("Ticker Context (VWRP.L, deep)", get_ticker_context("VWRP.L", "deep")),
            ("Opportunity (watchlist, momentum)", get_opportunity_context("watchlist", "momentum")),
            ("Opportunity (SPY, value_dip)", get_opportunity_context("SPY", "value_dip")),
            ("Macro Summary", get_macro_summary("snapshot,macro,fear_greed,sectors")),
            ("Account History (all)", get_account_history("all", 5)),
            ("Open Orders", get_open_orders()),
            ("Pies", get_pies()),
            ("Market Status", get_market_status()),
            ("Fundamentals (AAPL, overview)", get_fundamentals("AAPL", "overview")),
            ("Price History (MSFT, 1y, 1mo)", get_price_history("MSFT", "1y", "1mo")),
            ("Earnings Calendar", get_earnings_calendar()),
            ("Analyze Portfolio", analyze_portfolio("risk,stress,allocation", 1000)),
            ("Calculate Position Size", calculate_position_size("AAPL", 150.0, 140.0, 1.0))
        ]
        
        with open(OUT_FILE, 'a') as f:
            for name, coro in tasks:
                print(f"Running: {name}...")
                await run_test(name, coro, f)
                
    print(f"\n✅ All tests complete! Results saved to {OUT_FILE}")

if __name__ == '__main__':
    asyncio.run(main())
