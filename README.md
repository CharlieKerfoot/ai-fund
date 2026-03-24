# AI Fund Research Stack

An AI-native quantitative research and paper trading platform that combines traditional quantitative signals with LLM-based feature extraction to generate alpha.

## Architecture

The platform is organized into four layers:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Data Ingestion                                 │
│  yfinance · FRED · EDGAR · NewsAPI · Quiver Quant        │
│  ingest/sources/  →  ingest/store/ (DuckDB + Parquet)   │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Research & Signal Generation                   │
│  LLM Extraction · Numeric Features · Embeddings         │
│  research/signals/  ·  ingest/extractors/               │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Portfolio Construction                         │
│  Regime Detection · Mean-Variance Optimization           │
│  Risk Limits · Signal Combination                        │
│  portfolio/regime/  ·  portfolio/optimizer/  ·  portfolio/risk/  │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Execution & Reporting                          │
│  Paper Trading · PnL Tracking · Daily HTML Reports       │
│  execution/paper/  ·  execution/reporting/              │
└─────────────────────────────────────────────────────────┘
```

All layers are coordinated by `orchestrator/daily.py` and exposed via the `cli.py` entry point.

## Quick Start

### Prerequisites

- Python 3.11+
- [UV package manager](https://docs.astral.sh/uv/)

### Installation

```bash
uv sync
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables

```bash
# Required for full functionality
ANTHROPIC_API_KEY=sk-ant-...      # Claude API (LLM extraction + regime)
FRED_API_KEY=...                   # Federal Reserve data (macro series)

# Optional
DUCKDB_PATH=./data/fund.duckdb    # Database path (default: data/fund.duckdb)
NEWS_API_KEY=...                   # NewsAPI (sentiment source)
QUIVER_API_KEY=...                 # Quiver Quant (congressional trades)
LOG_LEVEL=INFO                     # Logging level (DEBUG/INFO/WARNING)
```

### First Run

```bash
# Backfill 5 years of historical data
uv run ai-fund backfill --years 5

# Run daily pipeline
uv run ai-fund run-pipeline

# Check status
uv run ai-fund status

# View today's report
uv run ai-fund report --open
```

## CLI Commands

### `run-pipeline`

Execute the full daily research pipeline through all 8 stages.

```bash
# Full pipeline for today
uv run ai-fund run-pipeline

# Dry run (no API calls)
uv run ai-fund run-pipeline --dry-run

# Run for a specific date
uv run ai-fund run-pipeline --date 2024-01-15

# Run only one stage
uv run ai-fund run-pipeline --stage signals
```

Pipeline stages: `ingest` → `extract` → `signals` → `regime` → `optimize` → `risk` → `execute` → `report`

### `backtest`

Run a historical backtest for a registered signal.

```bash
# Backtest the momentum signal over 5 years
uv run ai-fund backtest momentum --start 2019-01-01

# Custom date range and universe
uv run ai-fund backtest mean_reversion --start 2020-01-01 --end 2024-01-01
```

### `status`

Show the state of the most recent pipeline run and latest portfolio value.

```bash
uv run ai-fund status
```

### `positions`

Display current paper portfolio holdings.

```bash
uv run ai-fund positions
```

### `signals`

List all registered signals with their weights and activation status.

```bash
uv run ai-fund signals
```

### `report`

View the daily HTML report.

```bash
# View latest report (print path)
uv run ai-fund report

# View specific date
uv run ai-fund report 2024-01-15

# Open directly in browser
uv run ai-fund report --open
```

### `risk-check`

Run a risk assessment against current portfolio holdings and configured limits.

```bash
uv run ai-fund risk-check
```

### `add-signal`

Register a new signal by providing its Python class path.

```bash
# Register a custom signal with default weight
uv run ai-fund add-signal mypackage.signals.MyMomentumSignal

# Register with custom weight
uv run ai-fund add-signal mypackage.signals.MySignal --weight 0.5
```

### `backfill`

Download and store historical market and macro data.

```bash
# Default: 5 years
uv run ai-fund backfill

# Custom years
uv run ai-fund backfill --years 10

# From a specific start date
uv run ai-fund backfill --start 2015-01-01
```

## Configuration

### `config.yaml`

Non-secret settings live in `config.yaml` at the project root:

| Section | Key | Description |
|---------|-----|-------------|
| `universe.sp500_top100` | list | Trading universe (100 tickers) |
| `fred.series` | list | FRED macro series to fetch |
| `edgar.user_agent` | string | EDGAR HTTP user-agent header |
| `risk_limits.max_position_pct` | float | Max single position as % of portfolio |
| `risk_limits.max_sector_pct` | float | Max sector concentration |
| `risk_limits.max_daily_loss_pct` | float | Daily loss circuit breaker |
| `backtest.transaction_cost_bps` | int | Round-trip transaction cost in basis points |
| `backtest.slippage_bps` | int | Market impact slippage estimate |
| `execution.initial_capital` | int | Starting paper trading capital in USD |
| `llm.model` | string | Claude model ID for extractions |
| `llm.daily_budget_usd` | float | Daily LLM cost cap |

### Data source schedules (`ingest/config/sources.yaml`)

Cron schedules for each data source:

| Source | Schedule | Description |
|--------|----------|-------------|
| market | `0 18 * * 1-5` | 6pm weekdays (yfinance) |
| macro | `0 19 * * 1-5` | 7pm weekdays (FRED) |
| sec_filings | `0 20 * * 1-5` | 8pm weekdays (EDGAR) |
| earnings | `0 21 * * 1-5` | 9pm weekdays (FMP) |
| pipeline | `0 22 * * 1-5` | 10pm ET (full orchestration) |

## Architecture Details

### Layer 1: Data Ingestion (`ingest/`)

- **`sources/market.py`** — Downloads OHLCV data from Yahoo Finance via `yfinance`
- **`sources/macro.py`** — Fetches FRED macroeconomic time series
- **`sources/sec_filings.py`** — Downloads 10-K, 10-Q, 8-K filings from EDGAR
- **`sources/earnings.py`** — Earnings call transcripts
- **`sources/sentiment.py`** — News sentiment via NewsAPI (stub with sample data)
- **`sources/congress.py`** — Congressional trading disclosures via Quiver Quant (stub)
- **`store/feature_store.py`** — DuckDB-backed feature store with Parquet staging
- **`quality/`** — Data validation and quality checks

### Layer 2: Research (`research/`, `ingest/extractors/`)

- **`extractors/llm_extractor.py`** — LLM-based structured feature extraction from filings and earnings calls
- **`extractors/numeric.py`** — Numeric feature engineering (momentum, mean reversion, volatility)
- **`extractors/embedding.py`** — Text embedding generation
- **`signals/registry.py`** — Signal registry with weight management and performance tracking
- **`signals/combination.py`** — Weighted signal combination
- **`signals/library/`** — Built-in signals (momentum, mean reversion, etc.)
- **`backtest/engine.py`** — Walk-forward backtesting engine

### Layer 3: Portfolio Construction (`portfolio/`)

- **`regime/detector.py`** — Market regime detection (risk-on/off/stagflation/crisis) blending quant signals with LLM
- **`optimizer/mean_variance.py`** — Mean-variance portfolio optimization via CVXPY
- **`optimizer/constraints.py`** — Position and sector constraints
- **`risk/limits.py`** — Risk limit enforcement
- **`risk/var.py`** — VaR calculation
- **`risk/stress_test.py`** — Historical and hypothetical stress testing
- **`allocator/allocate.py`** — Combines optimizer, constraints, and risk checks

### Layer 4: Execution (`execution/`)

- **`paper/simulator.py`** — Paper trading simulator with slippage and commissions
- **`paper/slippage.py`** — Market impact / slippage model
- **`orders/manager.py`** — Order generation and persistence
- **`tracking/pnl.py`** — Daily PnL computation
- **`tracking/attribution.py`** — Signal-level return attribution
- **`tracking/feedback.py`** — Feedback loop for signal weight updates
- **`reporting/daily.py`** — Jinja2-based daily HTML report generation

### Orchestration (`orchestrator/`)

- **`daily.py`** — `DailyPipeline` — coordinates all 8 stages with idempotency, error handling, and retry logic

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_cli.py -v

# With coverage
uv run pytest tests/ --cov --cov-report=html
```

### Linting

```bash
uv run ruff check . --fix
```

### Adding a New Signal

1. Create a class in `research/signals/library/` that inherits from `research.signals.base.Signal`
2. Implement `compute(features: pd.DataFrame, as_of: date) -> pd.Series`
3. Set `name` and `version` class attributes
4. Register via CLI:

```bash
uv run ai-fund add-signal research.signals.library.my_signal.MySignal --weight 1.0
```

Or register programmatically:

```python
from research.signals.registry import SignalRegistry
from research.signals.library.my_signal import MySignal

registry = SignalRegistry(conn=conn)
registry.register(MySignal(), weight=1.0)
```

### Database Schema

The platform uses DuckDB at `$DUCKDB_PATH` (default `data/fund.duckdb`). Key tables:

| Table | Description |
|-------|-------------|
| `prices` | Daily OHLCV + adj_close per symbol |
| `macro_data` | FRED macro time series |
| `signals` | Computed signal values per symbol/date |
| `fills` | Paper trading fill records |
| `orders` | Generated order records |
| `pipeline_runs` | Stage execution log (idempotency) |
| `regime_history` | Daily regime detection results |
| `daily_pnl` | Daily portfolio PnL |

Raw data is also staged to Parquet files in `data/parquet/` before being merged into DuckDB.
