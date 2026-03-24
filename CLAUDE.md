# AI Fund Research Stack — Claude Code Conventions

## Project Overview

AI-native quantitative research and paper trading platform. Four layers:
- **Ingest** (`ingest/`): data pipelines → DuckDB feature store
- **Research** (`research/`): signal generation → backtest → validation
- **Portfolio** (`portfolio/`): regime detection → optimization → risk management
- **Execution** (`execution/`): paper trading → P&L → feedback loop

Entry point: `cli.py` (typer). Orchestrator: `orchestrator/daily.py`.

## Non-Negotiable Rules

### Point-in-Time Correctness
Every query to the feature store **must** pass `as_of: date`. No query may use data
that would not have been available as of that date.

```python
# CORRECT
df = store.query_prices(symbols, start, end, as_of=backtest_date)

# WRONG — leaks future data
df = store.query_prices(symbols, start, end)
```

### Signal Dependency Rule
Signals read raw features from the FeatureStore. Signals must NOT depend on other
signals' outputs. Cross-asset signals read raw cross-asset data from the store.

### DuckDB Single-Writer Pattern
Never write to DuckDB from parallel processes. Pattern:
1. Parallel async tasks write to source-specific Parquet files in `PARQUET_DIR`
2. Sequential merge step loads Parquet files into DuckDB

### Extraction Atomicity
LLM extraction is all-or-nothing. Mark `extraction_status = 'complete'` only after
full schema validation passes. Never persist partial records.

### Pipeline Idempotency
Every pipeline stage is idempotent. Check `pipeline_runs` table before executing a
stage — skip if `status = 'complete'` for today's date and that stage.

## Code Conventions

### Python Style
- 4 spaces for indentation (per user preference)
- `ruff` for linting — run before committing
- Type hints on all public functions
- Dataclasses for data transfer objects

### Naming
- Modules: snake_case
- Classes: PascalCase
- Constants: SCREAMING_SNAKE
- Private methods/attrs: `_leading_underscore`

### Signal Interface Contract
```python
class Signal(ABC):
    name: str
    version: str         # semver e.g. "1.0.0"
    description: str
    universe: str
    frequency: str       # "daily" | "weekly"
    lookback_days: int
    features_required: list[str]

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series: ...
    def explain(self, symbol: str, as_of: date) -> str: ...
```

### FeatureStore Interface Contract
```python
class FeatureStore:
    def query_prices(self, symbols, start, end, as_of) -> pd.DataFrame: ...
    def query_macro(self, series_ids, start, end, as_of) -> pd.DataFrame: ...
    def insert_prices(self, df, source) -> None: ...
    def get_unprocessed(self, source_type) -> list[dict]: ...
```

### Error Handling
- Let errors propagate unless you have a specific recovery strategy
- Log errors with `structlog` before re-raising in pipeline stages
- Use `tenacity` for retrying external API calls (max 3 retries, exponential backoff)

## Module Boundaries

| Module | Reads from | Writes to |
|--------|-----------|-----------|
| `ingest/sources/` | External APIs | Parquet staging |
| `ingest/store/` | Parquet staging | DuckDB |
| `research/signals/` | DuckDB (via FeatureStore) | DuckDB `signals` table |
| `portfolio/` | DuckDB `signals`, `macro` | DuckDB `regime_history` |
| `execution/` | DuckDB `positions`, `orders` | DuckDB `fills`, `daily_pnl` |
| `orchestrator/` | `pipeline_runs` | All of the above |

## Testing Requirements

- Unit tests for all public functions
- Integration tests for full pipeline stages
- Tests use a temp DuckDB in-memory (`:memory:`) — never the production database
- Mock external APIs (yfinance, FRED, EDGAR, Claude) in unit tests
- At least one integration test per pipeline stage that runs the real chain with fixtures

## Secrets

Never hardcode API keys. Load from environment:
```python
import os
api_key = os.environ["FRED_API_KEY"]
```
All keys documented in `.env.example`.
