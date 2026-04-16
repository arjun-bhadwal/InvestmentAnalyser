# Investment Analyser — Prompt Playbook

Copy-paste these into Claude Desktop when using your Investment Analyser project.

---

## 🌅 Daily Routine

### Morning Brief
```
Give me a full portfolio update. Check market status, show my positions, 
flag anything that moved significantly, and tell me if there's anything 
I need to act on today.
```

### End of Day Review
```
Markets are closing. Summarise today's performance across my portfolio. 
What moved and why? Any after-hours earnings or news I should know about 
before tomorrow?
```

### Weekly Review
```
Run a full weekly review: portfolio performance, risk metrics, allocation 
check, and sector rotation. Are we properly positioned for next week? 
Flag any rebalancing needed.
```

---

## 🔍 Stock Analysis

### Full Deep Dive
```
Give me a complete analysis of [TICKER]. Price history across all timeframes, 
fundamentals, DCF valuation, technicals, analyst ratings, insider activity, 
recent news, and your recommendation with entry/stop/target.
```

### Quick Pulse Check
```
Quick check on [TICKER] — current price, what's happened today, and any 
news driving the move.
```

### Should I Buy This?
```
I'm considering buying [TICKER]. Stress-test this idea. Run full 
fundamentals, technicals, DCF, check insider activity, and calculate 
position size with a stop at [PRICE]. Tell me if the data supports it 
or not.
```

### Compare Before Deciding
```
I'm choosing between [TICKER1], [TICKER2], and [TICKER3]. Compare them 
side by side on valuation, growth, risk, and analyst sentiment. 
Which one is the best entry right now and why?
```

---

## 📊 Portfolio Management

### Risk Health Check
```
Run a full risk analysis on my portfolio. Show me Sharpe, Sortino, VaR, 
max drawdown, and the correlation matrix. Am I too concentrated or 
correlated anywhere?
```

### Allocation Review
```
Show my portfolio allocation by sector, geography, and market cap. 
Is anything dangerously concentrated? Suggest rebalancing moves if needed.
```

### Stress Test
```
Stress test my portfolio. Show how it would've performed in the 2008 
crisis, COVID crash, and 2022 rate hikes. Run a Monte Carlo simulation 
and tell me my probability of loss over the next year.
```

### What Should I Sell?
```
Analyse every position in my portfolio. Identify the weakest holdings 
based on fundamentals, technicals, and outlook. Which positions should 
I consider trimming or exiting, and where should I redeploy that capital?
```

---

## 🎯 Finding Opportunities

### Oversold Scanner
```
Screen the S&P 500 for oversold stocks with RSI below 35 that are still 
above their 200-day MA and have analyst upside of at least 15%. 
Show me the top 10.
```

### FTSE Bargain Hunt
```
Screen the FTSE 100 for undervalued plays. Look for low RSI, positive 
momentum, and strong analyst upside. What's the best value on the 
London market right now?
```

### Momentum Plays
```
Screen for stocks with strong 1-month momentum, RSI between 50-65 
(not overbought yet), and analyst buy ratings. I want to ride trends 
that still have room to run.
```

### Sector Rotation Play
```
Show me sector rotation data. Which sectors are outperforming and seeing 
inflows? Which are bleeding? Suggest a sector rotation trade based on 
where institutional money is flowing.
```

### Custom Screen
```
Screen these tickers: [TICKER1, TICKER2, TICKER3, ...]. 
Compare RSI, momentum, analyst targets, and P/E. Rank them by 
conviction for a new entry.
```

---

## 🌍 Macro & Geopolitics

### Macro Landscape
```
Give me the full macro dashboard. Rates, VIX, dollar, oil, gold, 
yield curve, and fear & greed. What's the macro environment telling us 
about risk right now?
```

### Geopolitical Impact
```
Research the latest on [EVENT — e.g. "US-China tariffs", "OPEC cuts", 
"Middle East tensions"]. How does this affect my portfolio specifically? 
Map the domino effects to my holdings.
```

### Rate Decision Prep
```
The Fed/BOE is meeting [DATE]. What's the market pricing in? How are 
my positions exposed to a rate hike/cut/hold? Should I adjust anything 
before the announcement?
```

---

## 💰 Trade Execution

### Size a Position
```
I want to buy [TICKER] at [ENTRY PRICE] with a stop loss at [STOP PRICE]. 
Calculate my position size, Kelly criterion, and check if this would 
over-concentrate my portfolio.
```

### Entry Timing
```
I've decided to buy [TICKER]. Based on technicals, what's the optimal 
entry point? Should I wait for a pullback or enter now? Give me specific 
price levels.
```

### Exit Strategy
```
I'm holding [TICKER] at an average price of [PRICE]. It's now at [PRICE]. 
Based on fundamentals and technicals, should I take profit, hold, or add? 
Give me a trailing stop level.
```

---

## 📰 News & Intelligence

### What's Moving Markets?
```
Search for the biggest market-moving news today. What are analysts and 
experts saying? Any major events, earnings surprises, or geopolitical 
developments I need to know about?
```

### Deep Research
```
Do a deep research dive on [TOPIC — e.g. "AI semiconductor supply chain 
bottlenecks 2026", "UK housing market outlook"]. Give me a multi-source 
synthesis with citations.
```

### Earnings Preview
```
Show me upcoming earnings for my portfolio. Which positions have earnings 
risk in the next 2 weeks? What are the EPS expectations and historical 
surprise rates?
```

### Insider Activity
```
Check insider trading activity for [TICKER]. Are executives buying or 
selling? What does the buy/sell ratio signal?
```

---

## 🧠 Strategic Thinking

### Portfolio Construction
```
I have £[AMOUNT] to invest. Based on current macro conditions, sector 
rotation, and my existing portfolio, build me a diversified allocation 
plan. Show position sizes and rationale for each.
```

### Dividend Income Play
```
I want to add dividend income to my portfolio. Screen for stocks with 
sustainable yields above 3%, strong payout coverage, and growing 
dividends. Avoid yield traps.
```

### Defensive Pivot
```
I'm worried about a downturn. Analyse my portfolio's defensive qualities. 
Which positions would hold up in a recession? Suggest defensive swaps 
for my most cyclical holdings.
```

### Thesis Challenge
```
My thesis is: [STATE YOUR THESIS — e.g. "Oil will rally because of OPEC 
cuts"]. Challenge this. What could go wrong? What does the data actually 
say? Give me the bear case.
```

---

## ⚡ Quick One-Liners

| Prompt | What it does |
|--------|-------------|
| `Price of AAPL` | Quick price check |
| `How's my portfolio?` | Holdings + P&L summary |
| `Fear and greed?` | Market sentiment score |
| `News on TSLA` | Latest headlines |
| `Is NVDA overbought?` | Technical signal check |
| `What's the VIX at?` | Macro fear gauge |
| `Show my dividends` | Dividend income history |
| `Any open orders?` | Pending limit orders |
| `How are my pies doing?` | Pie performance |
| `Sector rotation` | Where money is flowing |
