# BTC Orderflow Backtester

Backtesting engine for the **BTCUSDT perpetual on 5-minute bars**, built on order flow (delta,
CVD, absorption) and positioning (open interest, funding, long/short ratios). Six signals are
combined into a **weighted confluence** system, and the strategy is evaluated against the rules
of a **prop firm challenge** (profit target, maximum drawdown and daily loss limit).

The focus is on backtest engineering: aligned multi-exchange data, automatically verified
zero look-ahead, walk-forward validation and Monte Carlo.

> Personal research project. Nothing here is investment advice.

## Architecture

```
config/            settings.yaml (parameters) · prop_firm_rules.yaml (rules for 3 prop firm profiles)
scripts/           historical download · backtest runner · signal and violation diagnostics
src/
  data/            Binance, Bybit and Coinalyze clients · cleaning and alignment · Parquet storage
  features/        OHLCV · volatility · order flow · open interest · funding · sentiment · sessions
  signals/         BaseSignal interface · 6 signals · confluence engine
  execution/       entry and exit · position sizing · daily limits
  backtest/        engine · portfolio · metrics · walk-forward · Monte Carlo · look-ahead checker
```

Each layer depends only on the one before it: data → features → signals → execution → backtest.
All parameters come from YAML, with no hard-coded values.

## Data

- **Binance Futures:** OHLCV bars, funding, taker buy/sell volume and **aggTrades** (every
  executed trade) downloaded from `data.binance.vision` and aggregated into 5-minute delta bars.
- **Bybit V5:** open interest and long/short ratio at 5-minute resolution. Bybit keeps history
  since the symbol's launch, while Binance only keeps 30 days.
- **Coinalyze:** open interest aggregated across exchanges, used as context.

A cleaning pipeline validates gaps, forward-fills small ones up to a bar limit, aligns every
source to an exact 5-minute grid and merges them into a single DataFrame.

## Signals

| Signal | Logic |
|---|---|
| **Liquidation sweep** | Price sweeps a zone with heavy open interest, OI drops sharply (liquidations) and delta shows absorption: enter against the sweep. |
| **Absorption** | Strong delta in one direction while price does not move: someone is absorbing the aggressive flow. Confirmed with CVD divergence. |
| **Funding reversion** | Extreme funding (z-score) means a market leveraged to one side: contrarian entry. |
| **OI divergence** | OI rises while price falls: shorts opened as a hedge while spot is being sold. |
| **OI drop** | Sharp OI drop with a strong move: the liquidation cascade is over and price tends to revert. |
| **Vol compression** | Bollinger Bands inside Keltner Channels and compressed ATR while OI builds up: enter on the breakout with delta confirmation. |

The **confluence engine** adds up the signals with their weights and only trades when the score
exceeds a minimum. Above a second threshold the trade is flagged as enhanced and can be sized up.

## Backtest engine

**Hybrid vectorized + bar-by-bar.** Features and signals are computed vectorized with
pandas/NumPy over the whole series. Position management runs bar by bar because it needs state:
partial exits, trailing stops and time stops.

**Look-ahead rules**, enforced in code and checked afterwards:
1. Every derived feature is shifted by one bar (`shift(1)`) in `builder.py`: the feature at T only uses data up to T−1.
2. Signal on bar T → confirmation is checked from T+1.
3. Entry at the **open** of the entry bar.
4. Stop and take-profit are checked against the bar's high and low, **stop before take-profit**: if both fit in the same bar, the worst case is assumed.

`LookAheadBiasChecker` reviews every trade after the run (entry timing, entry price within the
bar's range, consistent exit price) and lists any violations.

**Four management versions** over the same signals: V1 closes everything at TP1; V2 closes part
at TP1 and lets the rest run; V3 adds an ATR trailing stop; V4 adds larger size for enhanced
confluence trades.

**Risk and execution:** fixed risk per trade (0.5% of equity), stop at 1.5 × ATR, TP1 at 1.5 R
and TP2 at 3 R, 0.04% taker fee and 0.01% slippage, at most 3 trades per day and a halt after
2 consecutive losses.

## Validation

- **Walk-forward:** train on a 2-month block, test on the following month and roll forward. Every metric that counts comes from periods the system did not see while being tuned.
- **Monte Carlo:** 10,000 resamples with replacement of trade outcomes (fixed seed, reproducible) to estimate the **probability of passing the challenge** at 6% and 10% maximum drawdown.
- **Metrics:** Sharpe, Sortino, Calmar, maximum drawdown and its duration, win rate, profit factor, expectancy and average R per trade, exit reasons, and whether the challenge would have been passed.

## Engineering

- **Concurrent download:** aggTrades are downloaded and aggregated day by day with a `ThreadPoolExecutor` (4 threads by default). The work is network-I/O bound, so threads pay off in Python despite the GIL.
- **Batch processing:** work is done in 20-day batches and each day is aggregated to 5-minute bars as soon as it is downloaded, so raw ticks for at most one batch are ever held in memory.
- **Parquet with pyarrow** as the storage format, with date filtering on load.
- **Common interface** (`BaseSignal`) for every signal: adding one is a new class plus a weight in the YAML.

## Limitations and next steps

- **Incremental Parquet writes.** `append_parquet` reads and rewrites the whole file on every batch, so each batch costs more than the last. Batches are also kept in memory to be returned at the end. The fix is to write each batch as its own file in a date-partitioned Parquet dataset and stop retaining batches.
- **Monte Carlo in a Python loop.** The 10,000 simulations walk through the trades one by one. Vectorizing with NumPy (a simulations × trades matrix with `cumsum` and drawdown masks) or parallelizing with `multiprocessing` would cut the runtime by orders of magnitude.
- **No unit tests.** `pytest` is in the requirements but there are no tests yet. The exit simulator (stop and take-profit in the same bar, trailing, partials) and the look-ahead checker are the first candidates.
- **Single asset.** The engine is built for one symbol. The natural extension is a universe of perpetuals with a shared portfolio.

## Usage

```bash
pip install -r requirements.txt

python -m scripts.download_historical                   # download everything to data/raw/
python -m scripts.download_historical --skip-aggtrades  # without ticks (much faster)

python -m scripts.run_backtest                  # V1
python -m scripts.run_backtest --version v3     # V3 with trailing stop
python -m scripts.run_backtest --walk-forward   # walk-forward validation
python -m scripts.run_backtest --monte-carlo    # probability of passing the challenge
python -m scripts.run_backtest --all            # everything
```

## Stack

Python · pandas · NumPy · pyarrow (Parquet) · requests · concurrent.futures · PyYAML ·
matplotlib / Plotly
