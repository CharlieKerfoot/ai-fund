"""DuckDB table schemas for the AI Fund feature store."""

import duckdb

CREATE_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    symbol      VARCHAR NOT NULL,
    date        DATE NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE,
    volume      BIGINT,
    source      VARCHAR,
    ingested_at TIMESTAMP,
    PRIMARY KEY (symbol, date)
)
"""

CREATE_TEXT_FEATURES = """
CREATE TABLE IF NOT EXISTS text_features (
    id                VARCHAR PRIMARY KEY,
    symbol            VARCHAR,
    date              DATE,
    source_type       VARCHAR,
    source_id         VARCHAR,
    feature_type      VARCHAR,
    feature_value     DOUBLE,
    feature_json      JSON,
    raw_text_hash     VARCHAR,
    extraction_status VARCHAR DEFAULT 'pending',
    retry_count       INTEGER DEFAULT 0,
    extracted_at      TIMESTAMP
)
"""

CREATE_MACRO = """
CREATE TABLE IF NOT EXISTS macro (
    series_id    VARCHAR NOT NULL,
    date         DATE NOT NULL,
    release_date DATE NOT NULL,
    value        DOUBLE,
    PRIMARY KEY (series_id, date, release_date)
)
"""

CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id      VARCHAR NOT NULL,
    symbol         VARCHAR NOT NULL,
    date           DATE NOT NULL,
    value          DOUBLE,
    confidence     DOUBLE,
    signal_version VARCHAR,
    metadata       JSON,
    PRIMARY KEY (signal_id, symbol, date)
)
"""

CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    symbol       VARCHAR NOT NULL,
    date         DATE NOT NULL,
    quantity     DOUBLE,
    avg_cost     DOUBLE,
    market_value DOUBLE,
    PRIMARY KEY (symbol, date)
)
"""

CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id    VARCHAR PRIMARY KEY,
    symbol      VARCHAR,
    date        DATE,
    side        VARCHAR,
    quantity    DOUBLE,
    order_type  VARCHAR,
    limit_price DOUBLE,
    status      VARCHAR,
    created_at  TIMESTAMP,
    filled_at   TIMESTAMP
)
"""

CREATE_FILLS = """
CREATE TABLE IF NOT EXISTS fills (
    fill_id      VARCHAR PRIMARY KEY,
    order_id     VARCHAR,
    symbol       VARCHAR,
    date         DATE,
    quantity     DOUBLE,
    fill_price   DOUBLE,
    slippage_bps DOUBLE,
    commission   DOUBLE,
    filled_at    TIMESTAMP
)
"""

CREATE_DAILY_PNL = """
CREATE TABLE IF NOT EXISTS daily_pnl (
    date              DATE PRIMARY KEY,
    gross_pnl         DOUBLE,
    net_pnl           DOUBLE,
    transaction_costs DOUBLE,
    portfolio_value   DOUBLE,
    cash              DOUBLE,
    regime            VARCHAR
)
"""

CREATE_REGIME_HISTORY = """
CREATE TABLE IF NOT EXISTS regime_history (
    date             DATE PRIMARY KEY,
    regime_probs     JSON,
    dominant_regime  VARCHAR,
    confidence       DOUBLE
)
"""

CREATE_PIPELINE_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        VARCHAR PRIMARY KEY,
    date          DATE,
    stage         VARCHAR,
    status        VARCHAR,
    started_at    TIMESTAMP,
    completed_at  TIMESTAMP,
    error_message VARCHAR,
    metadata      JSON
)
"""

CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS embeddings (
    doc_id     VARCHAR PRIMARY KEY,
    metadata   JSON,
    embedding  BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

ALL_DDL = [
    CREATE_PRICES,
    CREATE_TEXT_FEATURES,
    CREATE_MACRO,
    CREATE_SIGNALS,
    CREATE_POSITIONS,
    CREATE_ORDERS,
    CREATE_FILLS,
    CREATE_DAILY_PNL,
    CREATE_REGIME_HISTORY,
    CREATE_PIPELINE_RUNS,
    CREATE_EMBEDDINGS,
]


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables in the DuckDB connection if they do not already exist."""
    for ddl in ALL_DDL:
        conn.execute(ddl)
