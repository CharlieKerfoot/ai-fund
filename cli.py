"""AI Fund CLI — main entry point."""

from __future__ import annotations

import asyncio
import importlib
import os
from datetime import date
from pathlib import Path

import structlog
import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from logging_config import configure_logging  # noqa: E402

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))

log = structlog.get_logger()
console = Console()

app = typer.Typer(name="ai-fund", help="AI-native quantitative research platform")

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    """Load config.yaml from the project root."""
    if not _CONFIG_PATH.exists():
        log.warning("config.yaml not found", path=str(_CONFIG_PATH))
        return {}
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f) or {}


def _require_env(name: str, warn_only: bool = False) -> str | None:
    """Return env var value; warn or abort if missing."""
    val = os.getenv(name)
    if not val:
        if warn_only:
            log.warning("Environment variable not set", var=name)
        else:
            log.error("Required environment variable missing", var=name)
            console.print(f"[red]Error:[/red] {name} is not set. Add it to your .env file.")
            raise typer.Exit(1)
    return val


def _get_db_path(config: dict) -> str:
    """Return the DuckDB path from env or default."""
    return os.getenv("DUCKDB_PATH", str(Path(__file__).parent / "data" / "fund.duckdb"))


def _flatten_symbols(config: dict) -> list[str]:
    """Flatten the nested symbol lists from config.yaml."""
    raw = config.get("universe", {}).get("sp500_top100", [])
    symbols: list[str] = []
    for item in raw:
        if isinstance(item, str):
            symbols.extend([s.strip() for s in item.split(",")])
    return [s for s in symbols if s]


@app.command("run-pipeline")
def run_pipeline(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without API calls"),
    run_date: str | None = typer.Option(None, "--date", help="Date to run (YYYY-MM-DD, default: today)"),
    stage: str | None = typer.Option(None, "--stage", help="Run only this stage"),
) -> None:
    """Execute the daily research pipeline."""
    config = _load_config()

    parsed_date: date
    if run_date:
        try:
            parsed_date = date.fromisoformat(run_date)
        except ValueError:
            console.print(f"[red]Invalid date format:[/red] {run_date}. Use YYYY-MM-DD.")
            raise typer.Exit(1)
    else:
        parsed_date = date.today()

    db_path = _get_db_path(config)
    anthropic_key = _require_env("ANTHROPIC_API_KEY", warn_only=True)
    fred_key = _require_env("FRED_API_KEY", warn_only=True)

    log.info(
        "Initializing pipeline",
        date=str(parsed_date),
        dry_run=dry_run,
        stage=stage,
        db_path=db_path,
    )

    try:
        import duckdb

        from ingest.store.feature_store import FeatureStore
        from ingest.store.schema import create_schema

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(db_path)
        create_schema(conn)

        parquet_dir = str(Path(db_path).parent / "parquet")
        feature_store = FeatureStore(db_path=db_path, parquet_dir=parquet_dir)

        symbols = _flatten_symbols(config)
        pipeline_config = {
            "symbols": symbols,
            "macro_series": config.get("fred", {}).get("series", []),
            "initial_capital": config.get("execution", {}).get("initial_capital", 1_000_000),
            "max_position_pct": config.get("risk_limits", {}).get("max_position_pct", 0.05),
            "min_positions": config.get("risk_limits", {}).get("min_positions", 20),
        }

        # Build components — degrade gracefully when API keys are missing
        market_source = None
        macro_source = None
        llm_extractor = None

        if not dry_run:
            from ingest.sources.market import MarketDataSource
            market_source = MarketDataSource(parquet_dir=parquet_dir)

            if fred_key:
                from ingest.sources.macro import MacroDataSource
                macro_source = MacroDataSource(fred_api_key=fred_key, parquet_dir=parquet_dir)

            if anthropic_key:
                from ingest.extractors.llm_extractor import LLMExtractor
                llm_extractor = LLMExtractor(api_key=anthropic_key)

        from execution.orders.manager import OrderManager
        from execution.paper.simulator import PaperTradingSimulator
        from execution.paper.slippage import SlippageModel
        from execution.reporting.daily import DailyReporter
        from execution.tracking.pnl import PnLTracker
        from portfolio.regime.detector import RegimeDetector
        from research.signals.combination import SignalCombiner
        from research.signals.registry import SignalRegistry

        signal_registry = SignalRegistry(conn=conn)
        signal_combiner = SignalCombiner(registry=signal_registry)
        regime_detector = RegimeDetector(feature_store=feature_store)

        # Build allocator with required sub-components
        allocator = None
        try:
            from portfolio.allocator.allocate import PortfolioAllocator
            from portfolio.optimizer.constraints import PortfolioConstraints
            from portfolio.optimizer.mean_variance import MeanVarianceOptimizer
            from portfolio.risk.limits import RiskLimitChecker

            optimizer = MeanVarianceOptimizer()
            constraints_engine = PortfolioConstraints(
                max_position_pct=pipeline_config.get("max_position_pct", 0.05),
                min_positions=pipeline_config.get("min_positions", 20),
            )
            risk_checker = RiskLimitChecker(limits=config.get("risk_limits", {}))
            allocator = PortfolioAllocator(
                optimizer=optimizer,
                constraint_engine=constraints_engine,
                risk_checker=risk_checker,
                regime_detector=regime_detector,
            )
        except Exception as exc:
            log.warning("Could not construct allocator; optimization stage will be skipped", error=str(exc))

        order_manager = OrderManager(conn=conn)
        slippage_model = SlippageModel(
            base_bps=config.get("backtest", {}).get("slippage_bps", 5),
        )
        simulator = PaperTradingSimulator(slippage_model=slippage_model, conn=conn)
        pnl_tracker = PnLTracker(
            conn=conn,
            initial_capital=pipeline_config.get("initial_capital", 1_000_000),
        )
        reporter = DailyReporter(conn=conn, reports_dir=str(Path(db_path).parent / "reports"))

        from orchestrator.daily import DailyPipeline

        pipeline = DailyPipeline(
            config=pipeline_config,
            feature_store=feature_store,
            monitor=None,
            market_source=market_source,
            macro_source=macro_source,
            llm_extractor=llm_extractor,
            signal_registry=signal_registry,
            signal_combiner=signal_combiner,
            regime_detector=regime_detector,
            allocator=allocator,
            order_manager=order_manager,
            simulator=simulator,
            pnl_tracker=pnl_tracker,
            feedback_loop=None,
            reporter=reporter,
        )

        if stage:
            if stage not in DailyPipeline.STAGES:
                console.print(f"[red]Unknown stage:[/red] {stage}. Valid: {', '.join(DailyPipeline.STAGES)}")
                raise typer.Exit(1)

            async def _run_single() -> None:
                await pipeline._dispatch_stage(stage, parsed_date, dry_run)

            asyncio.run(_run_single())
            console.print(f"[green]Stage '{stage}' complete.[/green]")
        else:
            summary = asyncio.run(pipeline.run(run_date=parsed_date, dry_run=dry_run))
            table = Table(title=f"Pipeline Summary — {parsed_date}")
            table.add_column("Stage", style="cyan")
            table.add_column("Status", style="bold")
            for s, status in summary.items():
                color = "green" if status == "complete" else "yellow" if status == "skipped" else "red"
                table.add_row(s, f"[{color}]{status}[/{color}]")
            console.print(table)

        feature_store.close()

    except Exception as exc:
        log.error("Pipeline failed", error=str(exc), exc_info=True)
        console.print(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("backtest")
def run_backtest(
    signal: str = typer.Argument(..., help="Signal name to backtest"),
    start: str = typer.Option("2019-01-01", "--start", help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="End date (default: today)"),
    universe: str = typer.Option("sp500_top100", "--universe"),
) -> None:
    """Run backtest for a specific signal."""
    config = _load_config()
    end_date = end or date.today().isoformat()

    log.info("Starting backtest", signal=signal, start=start, end=end_date, universe=universe)

    try:
        start_date = date.fromisoformat(start)
        end_dt = date.fromisoformat(end_date)
    except ValueError as exc:
        console.print(f"[red]Invalid date:[/red] {exc}")
        raise typer.Exit(1)

    db_path = _get_db_path(config)

    try:
        from ingest.store.feature_store import FeatureStore
        from research.backtest.engine import BacktestEngine
        from research.signals.registry import SignalRegistry

        parquet_dir = str(Path(db_path).parent / "parquet")
        feature_store = FeatureStore(db_path=db_path, parquet_dir=parquet_dir)

        registry = SignalRegistry()
        reg = registry.get(signal)
        if reg is None:
            console.print(f"[yellow]Signal '{signal}' not found in registry. Running empty backtest.[/yellow]")
            feature_store.close()
            return

        engine = BacktestEngine(
            feature_store=feature_store,
            transaction_cost_bps=config.get("backtest", {}).get("transaction_cost_bps", 10),
            slippage_bps=config.get("backtest", {}).get("slippage_bps", 5),
        )
        symbols = _flatten_symbols(config)
        result = engine.run(
            signal=reg.signal,
            symbols=symbols,
            start=start_date,
            end=end_dt,
        )

        console.print(f"\n[bold]Backtest Results: {signal}[/bold]")
        console.print(f"  Sharpe Ratio:     {result.sharpe_ratio:.4f}")
        console.print(f"  Annual Return:    {result.annual_return:.2%}")
        console.print(f"  Max Drawdown:     {result.max_drawdown:.2%}")
        console.print(f"  Total Trades:     {result.total_trades}")

        feature_store.close()

    except ImportError as exc:
        log.warning("Backtest module not available", error=str(exc))
        console.print(f"[yellow]Backtest module not fully available: {exc}[/yellow]")
    except Exception as exc:
        log.error("Backtest failed", error=str(exc), exc_info=True)
        console.print(f"[red]Backtest error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("status")
def show_status() -> None:
    """Show current pipeline and portfolio state."""
    config = _load_config()
    db_path = _get_db_path(config)

    log.info("Fetching pipeline status", db_path=db_path)

    if not Path(db_path).exists():
        console.print("[yellow]No database found. Run 'ai-fund backfill' or 'ai-fund run-pipeline' first.[/yellow]")
        return

    try:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)

        # Pipeline run status
        runs = conn.execute(
            """
            SELECT stage, status, MAX(completed_at) AS last_run
            FROM pipeline_runs
            GROUP BY stage, status
            ORDER BY stage
            """
        ).fetchall()

        table = Table(title="Pipeline Status")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Last Run")

        if runs:
            for row in runs:
                stage_name, status, last_run = row
                color = "green" if status == "complete" else "red" if status == "failed" else "yellow"
                table.add_row(stage_name, f"[{color}]{status}[/{color}]", str(last_run or "—"))
        else:
            table.add_row("—", "[yellow]No runs recorded[/yellow]", "—")

        console.print(table)

        # Portfolio summary
        pnl = conn.execute(
            "SELECT date, portfolio_value, net_pnl FROM daily_pnl ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if pnl:
            console.print(f"\n[bold]Latest Portfolio:[/bold] {pnl[0]}  Value: ${pnl[1]:,.2f}  Net PnL: ${pnl[2]:,.2f}")
        else:
            console.print("\n[dim]No portfolio data yet.[/dim]")

        conn.close()

    except Exception as exc:
        log.error("Status check failed", error=str(exc))
        console.print(f"[red]Error reading status:[/red] {exc}")
        raise typer.Exit(1)


@app.command("positions")
def show_positions() -> None:
    """Show current portfolio holdings."""
    config = _load_config()
    db_path = _get_db_path(config)

    if not Path(db_path).exists():
        console.print("[yellow]No database found.[/yellow]")
        return

    try:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)

        positions = conn.execute(
            """
            SELECT
                symbol,
                SUM(quantity) AS quantity,
                CASE
                    WHEN SUM(quantity) = 0 THEN 0.0
                    ELSE SUM(fill_price * ABS(quantity)) / SUM(ABS(quantity))
                END AS avg_cost
            FROM fills
            GROUP BY symbol
            HAVING SUM(quantity) != 0
            ORDER BY symbol
            """
        ).fetchall()

        if not positions:
            console.print("[dim]No open positions.[/dim]")
            conn.close()
            return

        table = Table(title="Current Positions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Quantity", justify="right")
        table.add_column("Avg Cost", justify="right")

        for sym, qty, avg_cost in positions:
            table.add_row(sym, f"{qty:,.2f}", f"${avg_cost:,.2f}")

        console.print(table)
        conn.close()

    except Exception as exc:
        log.error("Positions query failed", error=str(exc))
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("signals")
def list_signals() -> None:
    """List signal registry with performance metrics."""
    _load_config()

    log.info("Listing registered signals")

    try:
        from research.signals.registry import SignalRegistry

        registry = SignalRegistry()

        # Try to load built-in signals
        try:
            from research.signals.library.mean_reversion import MeanReversionSignal
            from research.signals.library.momentum import MomentumSignal
            registry.register(MomentumSignal(), weight=1.0)
            registry.register(MeanReversionSignal(), weight=1.0)
        except ImportError:
            pass

        all_signals = registry.list_all()

        if not all_signals:
            console.print("[dim]No signals registered. Use 'ai-fund add-signal' to register one.[/dim]")
            return

        table = Table(title="Signal Registry")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Active", justify="center")
        table.add_column("Weight", justify="right")
        table.add_column("Registered At")

        for reg in all_signals:
            active_str = "[green]Yes[/green]" if reg.active else "[red]No[/red]"
            table.add_row(
                reg.signal.name,
                reg.signal.version,
                active_str,
                f"{reg.weight:.2f}",
                reg.registered_at,
            )

        console.print(table)

    except Exception as exc:
        log.error("Signals listing failed", error=str(exc))
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("report")
def show_report(
    report_date: str | None = typer.Argument(None, help="Date YYYY-MM-DD (default: latest)"),
    open_browser: bool = typer.Option(False, "--open", help="Open in browser"),
) -> None:
    """View daily report."""
    config = _load_config()
    db_path = _get_db_path(config)
    reports_dir = Path(db_path).parent / "reports"

    if report_date:
        try:
            target_date = date.fromisoformat(report_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {report_date}")
            raise typer.Exit(1)
        report_path = reports_dir / f"{target_date}.html"
    else:
        # Find latest report
        reports = sorted(reports_dir.glob("*.html")) if reports_dir.exists() else []
        if not reports:
            console.print("[yellow]No reports found. Run 'ai-fund run-pipeline' first.[/yellow]")
            return
        report_path = reports[-1]

    if not report_path.exists():
        console.print(f"[yellow]Report not found:[/yellow] {report_path}")
        return

    console.print(f"[green]Report:[/green] {report_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(report_path.as_uri())
        log.info("Opened report in browser", path=str(report_path))
    else:
        console.print("[dim]Use --open to open in browser.[/dim]")


@app.command("risk-check")
def risk_check() -> None:
    """Run risk assessment on current portfolio."""
    config = _load_config()
    db_path = _get_db_path(config)

    if not Path(db_path).exists():
        console.print("[yellow]No database found.[/yellow]")
        return

    log.info("Running risk assessment", db_path=db_path)

    try:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)

        # Fetch positions
        positions = conn.execute(
            """
            SELECT symbol, SUM(quantity) AS quantity
            FROM fills
            GROUP BY symbol
            HAVING SUM(quantity) != 0
            """
        ).fetchall()

        if not positions:
            console.print("[dim]No open positions to assess.[/dim]")
            conn.close()
            return

        risk_limits = config.get("risk_limits", {})
        max_pos = risk_limits.get("max_position_pct", 0.05)
        max_sector = risk_limits.get("max_sector_pct", 0.25)
        min_pos = risk_limits.get("min_positions", 20)

        total_positions = len(positions)

        table = Table(title="Risk Assessment")
        table.add_column("Check", style="cyan")
        table.add_column("Limit")
        table.add_column("Current")
        table.add_column("Status", justify="center")

        min_status = "[green]PASS[/green]" if total_positions >= min_pos else "[red]FAIL[/red]"
        table.add_row("Min Positions", str(min_pos), str(total_positions), min_status)
        table.add_row("Max Position %", f"{max_pos:.0%}", "—", "[yellow]Needs prices[/yellow]")
        table.add_row("Max Sector %", f"{max_sector:.0%}", "—", "[yellow]Needs sector data[/yellow]")

        console.print(table)
        conn.close()

    except Exception as exc:
        log.error("Risk check failed", error=str(exc))
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("add-signal")
def add_signal(
    module_path: str = typer.Argument(..., help="Python module path e.g. mymodule.MySignal"),
    weight: float = typer.Option(1.0, "--weight"),
) -> None:
    """Register a new signal from a Python class path."""
    log.info("Adding signal", module_path=module_path, weight=weight)

    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        console.print("[red]Invalid module path.[/red] Use format: mypackage.module.ClassName")
        raise typer.Exit(1)

    module_name, class_name = parts

    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        console.print(f"[red]Could not import '{module_path}':[/red] {exc}")
        raise typer.Exit(1)

    try:
        instance = cls()
    except Exception as exc:
        console.print(f"[red]Could not instantiate '{class_name}':[/red] {exc}")
        raise typer.Exit(1)

    from research.signals.registry import SignalRegistry
    registry = SignalRegistry()
    registry.register(instance, weight=weight)

    console.print(f"[green]Registered signal:[/green] {instance.name} v{instance.version} (weight={weight})")
    log.info("Signal registered", name=instance.name, version=instance.version, weight=weight)


@app.command("backfill")
def run_backfill(
    years: int = typer.Option(5, "--years", help="Number of years to backfill"),
    start_date: str | None = typer.Option(None, "--start"),
) -> None:
    """Backfill historical data."""
    config = _load_config()
    db_path = _get_db_path(config)
    fred_key = _require_env("FRED_API_KEY", warn_only=True)

    from datetime import timedelta
    end = date.today()
    if start_date:
        try:
            start = date.fromisoformat(start_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {start_date}")
            raise typer.Exit(1)
    else:
        start = end - timedelta(days=years * 365)

    log.info("Starting backfill", start=str(start), end=str(end), db_path=db_path)
    console.print(f"Backfilling {start} → {end} ({years} years)")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    parquet_dir = str(Path(db_path).parent / "parquet")

    symbols = _flatten_symbols(config)
    if not symbols:
        console.print("[red]No symbols configured in config.yaml[/red]")
        raise typer.Exit(1)

    console.print(f"  Symbols:   {len(symbols)}")

    try:
        from ingest.sources.market import MarketDataSource
        market_source = MarketDataSource(parquet_dir=parquet_dir)

        with console.status("Fetching market data..."):
            df = market_source.fetch_daily(symbols=symbols, start=start, end=end)

        if df.empty:
            console.print("[yellow]No market data returned.[/yellow]")
        else:
            path = market_source.write_parquet(df, end)
            console.print(f"[green]Market data:[/green] {len(df):,} rows → {path}")

            # Load into DuckDB
            from ingest.store.feature_store import FeatureStore
            feature_store = FeatureStore(db_path=db_path, parquet_dir=parquet_dir)
            feature_store.insert_prices(df)
            feature_store.close()
            log.info("Market data backfill complete", rows=len(df))

    except Exception as exc:
        log.error("Market backfill failed", error=str(exc))
        console.print(f"[red]Market data error:[/red] {exc}")

    if fred_key:
        try:
            from ingest.sources.macro import MacroDataSource
            macro_source = MacroDataSource(fred_api_key=fred_key, parquet_dir=parquet_dir)
            macro_series = config.get("fred", {}).get("series", [])

            with console.status("Fetching macro data..."):
                macro_df = macro_source.fetch_series(series_ids=macro_series, start=start, end=end)

            if not macro_df.empty:
                path = macro_source.write_parquet(macro_df, end)
                console.print(f"[green]Macro data:[/green] {len(macro_df):,} rows → {path}")

                from ingest.store.feature_store import FeatureStore
                feature_store = FeatureStore(db_path=db_path, parquet_dir=parquet_dir)
                feature_store.insert_macro(macro_df)
                feature_store.close()
                log.info("Macro backfill complete", rows=len(macro_df))
            else:
                console.print("[yellow]No macro data returned.[/yellow]")

        except Exception as exc:
            log.error("Macro backfill failed", error=str(exc))
            console.print(f"[yellow]Macro data warning:[/yellow] {exc}")
    else:
        console.print("[yellow]FRED_API_KEY not set — skipping macro backfill.[/yellow]")

    console.print("\n[green]Backfill complete.[/green]")


if __name__ == "__main__":
    app()
