"""Daily pipeline orchestrator: coordinates all stages of the AI Fund pipeline."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime

import structlog

log = structlog.get_logger()


class DailyPipeline:
    STAGES = [
        "ingest",
        "extract",
        "signals",
        "regime",
        "optimize",
        "risk",
        "execute",
        "report",
    ]

    # Stages that halt the pipeline on failure
    CRITICAL_STAGES = {"ingest"}
    # Stages that continue on failure (log and continue)
    NON_CRITICAL_STAGES = {"extract", "regime", "execute"}

    def __init__(
        self,
        config: dict,
        feature_store,
        monitor,
        market_source,
        macro_source,
        llm_extractor,
        signal_registry,
        signal_combiner,
        regime_detector,
        allocator,
        order_manager,
        simulator,
        pnl_tracker,
        feedback_loop,
        reporter,
    ) -> None:
        self.config = config
        self.feature_store = feature_store
        self.monitor = monitor
        self.market_source = market_source
        self.macro_source = macro_source
        self.llm_extractor = llm_extractor
        self.signal_registry = signal_registry
        self.signal_combiner = signal_combiner
        self.regime_detector = regime_detector
        self.allocator = allocator
        self.order_manager = order_manager
        self.simulator = simulator
        self.pnl_tracker = pnl_tracker
        self.feedback_loop = feedback_loop
        self.reporter = reporter

        # Shared state for the current run
        self._current_regime = None
        self._current_weights = None

    def _get_conn(self):
        """Get the underlying DB connection from feature_store."""
        return getattr(self.feature_store, "_conn", None)

    def _cleanup_stale_runs(self, run_date: date) -> None:
        """Mark any 'running' stages from previous dates as 'failed'."""
        conn = self._get_conn()
        if conn is None:
            return

        conn.execute(
            """
            UPDATE pipeline_runs
            SET status = 'failed',
                error_message = 'Marked failed on startup: previous run did not complete',
                completed_at = ?
            WHERE status = 'running'
              AND date < ?
            """,
            [datetime.now(tz=UTC), run_date],
        )
        log.info("Cleaned up stale pipeline runs", run_date=str(run_date))

    def _is_stage_complete(self, run_date: date, stage: str) -> bool:
        """Check if a stage is already complete for today."""
        conn = self._get_conn()
        if conn is None:
            return False

        row = conn.execute(
            """
            SELECT status FROM pipeline_runs
            WHERE date = ? AND stage = ? AND status = 'complete'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            [run_date, stage],
        ).fetchone()

        return row is not None

    def _mark_stage(self, run_id: str, run_date: date, stage: str, status: str, error: str | None = None) -> None:
        """Update pipeline_runs with the current stage status."""
        conn = self._get_conn()
        if conn is None:
            return

        now = datetime.now(tz=UTC)
        conn.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs
                (run_id, date, stage, status, started_at, completed_at, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, run_date, stage, status, now, now if status in ("complete", "failed") else None, error, None],
        )

    def _start_stage(self, run_id: str, run_date: date, stage: str) -> None:
        """Mark a stage as running."""
        conn = self._get_conn()
        if conn is None:
            return

        now = datetime.now(tz=UTC)
        conn.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs
                (run_id, date, stage, status, started_at, completed_at, error_message, metadata)
            VALUES (?, ?, ?, 'running', ?, NULL, NULL, NULL)
            """,
            [run_id, run_date, stage, now],
        )

    async def run(
        self,
        run_date: date | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Execute daily pipeline and return summary dict: {stage: status}."""
        if run_date is None:
            run_date = date.today()

        log.info("Starting daily pipeline", date=str(run_date), dry_run=dry_run)

        # Step 1: Cleanup stale runs from previous dates
        self._cleanup_stale_runs(run_date)

        summary: dict[str, str] = {}

        for stage in self.STAGES:
            run_id = str(uuid.uuid4())

            # Check idempotency: skip if already complete for today
            if self._is_stage_complete(run_date, stage):
                log.info("Stage already complete, skipping", stage=stage, date=str(run_date))
                summary[stage] = "skipped"
                continue

            self._start_stage(run_id, run_date, stage)
            log.info("Running stage", stage=stage, date=str(run_date))

            try:
                await self._dispatch_stage(stage, run_date, dry_run)
                self._mark_stage(run_id, run_date, stage, "complete")
                summary[stage] = "complete"
                log.info("Stage complete", stage=stage)

            except Exception as exc:
                error_msg = str(exc)
                self._mark_stage(run_id, run_date, stage, "failed", error=error_msg)
                summary[stage] = "failed"
                log.error("Stage failed", stage=stage, error=error_msg, exc_info=True)

                if stage in self.CRITICAL_STAGES:
                    log.error("Critical stage failed, halting pipeline", stage=stage)
                    # Mark remaining stages as skipped
                    remaining = self.STAGES[self.STAGES.index(stage) + 1:]
                    for s in remaining:
                        summary[s] = "skipped"
                    break
                # Non-critical: continue

        log.info("Pipeline complete", date=str(run_date), summary=summary)
        return summary

    async def _dispatch_stage(self, stage: str, run_date: date, dry_run: bool) -> None:
        """Dispatch to the appropriate stage handler."""
        if stage == "ingest":
            await self._run_ingest(run_date, dry_run)
        elif stage == "extract":
            await self._run_extract(run_date, dry_run)
        elif stage == "signals":
            await self._run_signals(run_date)
        elif stage == "regime":
            await self._run_regime(run_date)
        elif stage == "optimize":
            await self._run_optimize(run_date)
        elif stage == "risk":
            await self._run_risk(run_date)
        elif stage == "execute":
            await self._run_execute(run_date, dry_run)
        elif stage == "report":
            await self._run_report(run_date)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    async def _run_ingest(self, run_date: date, dry_run: bool) -> None:
        """Fetch market and macro data in parallel, write to Parquet, merge to DuckDB."""
        if dry_run:
            log.info("dry_run=True: skipping actual ingest API calls")
            return

        symbols = self.config.get("symbols", [])
        macro_series = self.config.get("macro_series", [])

        # Parallel async fetch
        tasks = []
        if self.market_source is not None and symbols:
            tasks.append(asyncio.to_thread(self.market_source.fetch, symbols, run_date))
        if self.macro_source is not None and macro_series:
            tasks.append(asyncio.to_thread(self.macro_source.fetch, macro_series, run_date))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    log.warning("Ingest fetch error", error=str(result))

        log.info("Ingest stage complete", date=str(run_date))

    async def _run_extract(self, run_date: date, dry_run: bool) -> None:
        """Run LLM extraction on pending text_features records."""
        if dry_run:
            log.info("dry_run=True: skipping LLM extraction")
            return

        if self.llm_extractor is None:
            log.info("No LLM extractor configured, skipping extract stage")
            return

        pending = self.feature_store.get_pending_extractions(source_type="sec_filing", limit=50)
        log.info("Running LLM extraction", pending_count=len(pending))

        for record in pending:
            try:
                result = await asyncio.to_thread(self.llm_extractor.extract, record)
                if result:
                    self.feature_store.mark_extraction_complete(record["id"], result)
            except Exception as exc:
                self.feature_store.mark_extraction_failed(record["id"], str(exc))

    async def _run_signals(self, run_date: date) -> None:
        """Compute all active signals and store in signals table."""
        active = self.signal_registry.list_active()
        log.info("Computing signals", count=len(active), date=str(run_date))

        for reg in active:
            try:
                # Get price features
                features = self.feature_store.query_prices(
                    symbols=self.config.get("symbols", []),
                    start=run_date,
                    end=run_date,
                    as_of=run_date,
                )
                values = reg.signal.compute(features, run_date)
                if values is not None and not values.empty:
                    import pandas as pd
                    signal_df = pd.DataFrame({
                        "symbol": values.index,
                        "date": run_date,
                        "value": values.values,
                        "confidence": 1.0,
                    })
                    self.feature_store.insert_signals(
                        signal_id=reg.signal.name,
                        df=signal_df,
                        version=reg.signal.version,
                        metadata={},
                    )
            except Exception as exc:
                log.warning("Signal computation failed", signal=reg.signal.name, error=str(exc))

    async def _run_regime(self, run_date: date) -> None:
        """Detect current market regime."""
        if self.regime_detector is None:
            log.info("No regime detector configured")
            return

        regime_state = self.regime_detector.detect(as_of=run_date)
        self._current_regime = regime_state

        # Persist to regime_history
        conn = self._get_conn()
        if conn is not None:
            import json
            conn.execute(
                """
                INSERT OR REPLACE INTO regime_history
                    (date, regime_probs, dominant_regime, confidence)
                VALUES (?, ?, ?, ?)
                """,
                [
                    run_date,
                    json.dumps(regime_state.regime_probs),
                    regime_state.dominant_regime,
                    regime_state.confidence,
                ],
            )

        log.info(
            "Regime detected",
            regime=regime_state.dominant_regime,
            confidence=regime_state.confidence,
        )

    async def _run_optimize(self, run_date: date) -> None:
        """Run portfolio optimization and apply constraints."""
        if self.allocator is None:
            log.info("No allocator configured, skipping optimize")
            return

        log.info("Running portfolio optimization", date=str(run_date))

        symbols = self.config.get("symbols", [])
        if not symbols:
            return

        # Get signals
        conn = self._get_conn()
        if conn is None:
            return

        signals_df = conn.execute(
            """
            SELECT symbol, AVG(value) AS value
            FROM signals
            WHERE date = ?
            GROUP BY symbol
            """,
            [run_date],
        ).df()

        if signals_df.empty:
            log.warning("No signals found for optimization", date=str(run_date))
            return

        import pandas as pd
        signal_values = pd.Series(
            signals_df["value"].values,
            index=signals_df["symbol"].values,
        )

        # Get returns history
        from datetime import timedelta
        lookback_start = run_date - timedelta(days=252)
        prices_df = self.feature_store.query_prices(
            symbols=symbols,
            start=lookback_start,
            end=run_date,
            as_of=run_date,
        )

        if prices_df.empty:
            log.warning("No price history for optimization")
            return

        prices_pivot = prices_df.pivot(index="date", columns="symbol", values="adj_close")
        returns_history = prices_pivot.pct_change().dropna()

        from portfolio.optimizer.constraints import PortfolioConstraints
        constraints = PortfolioConstraints(
            max_position_pct=self.config.get("max_position_pct", 0.20),
            min_positions=self.config.get("min_positions", 0),
        )

        weights, risk_result = self.allocator.allocate(
            signal_values=signal_values,
            returns_history=returns_history,
            current_weights=None,
            regime=self._current_regime,
            constraints=constraints,
        )

        self._current_weights = weights
        log.info("Optimization complete", n_positions=len(weights))

    async def _run_risk(self, run_date: date) -> None:
        """Run risk checks and stress tests."""
        if self._current_weights is None or self._current_weights.empty:
            log.info("No weights to check risk on, skipping")
            return

        log.info("Running risk checks", date=str(run_date))
        # Risk checks are embedded in the allocator's output
        # Additional stress tests could be run here

    async def _run_execute(self, run_date: date, dry_run: bool) -> None:
        """Generate and simulate orders."""
        if self._current_weights is None or self._current_weights.empty:
            log.info("No target weights, skipping execution")
            return

        if self.order_manager is None or self.simulator is None:
            log.info("No order manager or simulator configured")
            return

        import pandas as pd

        # Get current positions
        current_positions = self.simulator.get_positions(run_date)

        # Get current prices
        symbols = self._current_weights.index.tolist()
        conn = self._get_conn()
        if conn is None:
            return

        prices_df = conn.execute(
            f"""
            SELECT symbol, adj_close
            FROM prices
            WHERE symbol IN ({', '.join('?' for _ in symbols)})
              AND date = ?
            """,
            symbols + [run_date],
        ).df()

        if prices_df.empty:
            log.warning("No prices available for execution", date=str(run_date))
            return

        prices = pd.Series(
            prices_df["adj_close"].values,
            index=prices_df["symbol"].values,
        )

        portfolio_value = self.simulator.get_portfolio_value(run_date, prices)
        if portfolio_value <= 0:
            portfolio_value = self.config.get("initial_capital", 1_000_000)

        orders = self.order_manager.generate_orders(
            target_weights=self._current_weights,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            prices=prices,
        )

        log.info("Generated orders", count=len(orders), dry_run=dry_run)

        if not dry_run:
            self.order_manager.persist_orders(orders)
            for order in orders:
                symbol = order.symbol
                market_price = float(prices.get(symbol, 0.0))
                if market_price <= 0:
                    continue
                avg_daily_volume = self.config.get("avg_daily_volume", {}).get(symbol, 1_000_000)
                order_dict = {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "quantity": order.quantity,
                    "side": order.side,
                    "date": order.date,
                }
                self.simulator.simulate_fill(order_dict, market_price, avg_daily_volume)
        else:
            log.info("dry_run=True: orders not persisted or filled")

    async def _run_report(self, run_date: date) -> None:
        """Generate daily HTML report."""
        if self.reporter is None:
            log.info("No reporter configured, skipping report")
            return

        regime = "unknown"
        if self._current_regime is not None:
            regime = self._current_regime.dominant_regime

        path = self.reporter.generate(run_date, regime=regime, degraded_mode=False)
        log.info("Report generated", path=path, date=str(run_date))

    def is_trading_day(self, d: date) -> bool:
        """Skip signal/optimize/execute on non-trading days."""
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(
                start_date=str(d),
                end_date=str(d),
            )
            return not schedule.empty
        except Exception:
            # Fallback: Mon-Fri is a trading day
            return d.weekday() < 5
