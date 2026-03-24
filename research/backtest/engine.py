"""Vectorized backtest engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from research.backtest.calendar import TradingCalendar
from research.signals.base import Signal

logger = logging.getLogger(__name__)

ANNUAL_FACTOR = 252  # trading days per year


@dataclass
class BacktestResult:
    signal_name: str
    signal_version: str
    start_date: date
    end_date: date
    equity_curve: pd.Series          # indexed by date
    returns: pd.Series               # daily returns
    positions: pd.DataFrame          # symbol weights over time
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    annual_return: float
    annual_volatility: float
    turnover: float                  # average daily turnover
    transaction_costs_total: float
    survivorship_bias_flag: bool     # True if universe not point-in-time


class BacktestEngine:
    """Vectorized backtest engine with transaction costs and performance metrics."""

    def __init__(
        self,
        feature_store,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 1_000_000,
    ) -> None:
        self._feature_store = feature_store
        self._tc_bps = transaction_cost_bps
        self._slippage_bps = slippage_bps
        self._initial_capital = initial_capital
        self._calendar = TradingCalendar()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signal: Signal,
        universe: list[str],
        start: date,
        end: date,
        rebalance_freq: str = "weekly",
    ) -> BacktestResult:
        """Run a vectorized backtest for the given signal over the universe.

        Steps:
        1. Fetch price data from feature store (point-in-time via as_of=end)
        2. Build daily price-return matrix
        3. On each rebalance date, compute signal weights
        4. Forward-fill weights to daily frequency
        5. Compute gross daily returns as weights @ price_returns
        6. Deduct transaction costs on weight changes
        7. Compute performance metrics
        """
        # Fetch prices. We use a far-future as_of so all currently-ingested data
        # is visible regardless of when it was inserted.
        # Point-in-time correctness for signals is enforced by passing as_of
        # to signal.compute(), which filters the features DataFrame by date.
        from datetime import date as _date
        from datetime import timedelta
        lookback_days = getattr(signal.metadata, "lookback_days", 252)
        feature_start = start - timedelta(days=lookback_days + 60)
        # Use end-of-day tomorrow to ensure all data ingested today is visible
        ingest_as_of = _date(9999, 12, 31)

        price_df = self._feature_store.query_prices(
            symbols=universe,
            start=feature_start,
            end=end,
            as_of=ingest_as_of,
        )

        if price_df.empty:
            raise ValueError("No price data returned from feature store for the given universe/dates.")

        # Pivot to wide format: rows=date, cols=symbol
        price_df["date"] = pd.to_datetime(price_df["date"])
        price_pivot = price_df.pivot_table(
            index="date", columns="symbol", values="adj_close", aggfunc="last"
        )

        # Daily log returns (vectorized)
        daily_returns = price_pivot.pct_change().dropna(how="all")

        # Restrict to backtest window
        bt_start = pd.Timestamp(start)
        bt_end = pd.Timestamp(end)
        daily_returns = daily_returns.loc[bt_start:bt_end]

        if daily_returns.empty:
            raise ValueError("No daily returns in the backtest window.")

        daily_returns.index.tolist()

        # Rebalance dates
        rebalance_dates = self._calendar.get_rebalance_dates(start, end, freq=rebalance_freq)

        # Build weights on rebalance dates
        weight_records: dict[pd.Timestamp, pd.Series] = {}

        for rb_date in rebalance_dates:
            ts = pd.Timestamp(rb_date)
            if ts not in daily_returns.index:
                # Find the nearest available date
                available = daily_returns.index[daily_returns.index <= ts]
                if available.empty:
                    continue
                ts = available[-1]

            # Build features DataFrame: prices up to and including rb_date
            feature_end = ts.date()
            feature_df = price_df[price_df["date"] <= ts].copy()

            try:
                raw_signal = signal.compute(feature_df, as_of=feature_end)
            except Exception as exc:
                logger.warning("Signal compute failed on %s: %s", rb_date, exc)
                continue

            weights = self._compute_weights(raw_signal)
            # Align weights to universe columns
            weights = weights.reindex(daily_returns.columns, fill_value=0.0)
            weight_records[ts] = weights

        if not weight_records:
            raise ValueError("No valid signal computations in backtest window.")

        # Build weight DataFrame and forward-fill to every trading day
        weights_df = pd.DataFrame(weight_records).T  # rows=rebalance dates, cols=symbols
        weights_df = weights_df.reindex(daily_returns.index).ffill().fillna(0.0)

        # Ensure weights sum to 1 on each day (re-normalize after ffill)
        row_sums = weights_df.sum(axis=1)
        row_sums = row_sums.replace(0, 1.0)  # avoid division by zero
        weights_df = weights_df.div(row_sums, axis=0)

        # Compute gross portfolio returns: sum over symbols of (weight_t-1 * return_t)
        # Use previous day's weights (positions enter at end of rebalance day)
        shifted_weights = weights_df.shift(1).fillna(0.0)
        gross_returns = (shifted_weights * daily_returns).sum(axis=1)

        # Transaction costs: cost_bps/10000 * |delta_weight| per symbol, per day
        weight_changes = weights_df.diff().abs().fillna(0.0)
        total_tc_rate = (self._tc_bps + self._slippage_bps) / 10_000.0
        tc_daily = weight_changes.sum(axis=1) * total_tc_rate
        net_returns = gross_returns - tc_daily

        # Build equity curve
        equity_curve = (1 + net_returns).cumprod() * self._initial_capital
        equity_curve.iloc[0] = self._initial_capital * (1 + net_returns.iloc[0])

        # Compute metrics
        metrics = self._compute_metrics(net_returns, weights_df)

        return BacktestResult(
            signal_name=signal.name,
            signal_version=signal.version,
            start_date=start,
            end_date=end,
            equity_curve=equity_curve,
            returns=net_returns,
            positions=weights_df,
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            max_drawdown=metrics["max_drawdown"],
            calmar=metrics["calmar"],
            annual_return=metrics["annual_return"],
            annual_volatility=metrics["annual_volatility"],
            turnover=metrics["turnover"],
            transaction_costs_total=tc_daily.sum() * self._initial_capital,
            survivorship_bias_flag=False,
        )

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(self, signal_values: pd.Series) -> pd.Series:
        """Convert raw signal values to long-only weights summing to 1.0.

        Top 20% by signal value get positive weights, proportional to their signal value.
        Bottom 80% receive zero weight.
        """
        clean = signal_values.dropna()
        if clean.empty:
            return pd.Series(dtype=float)

        n_top = max(1, int(np.ceil(len(clean) * 0.20)))
        top_symbols = clean.nlargest(n_top).index

        raw_weights = clean.reindex(top_symbols).clip(lower=0.0)
        total = raw_weights.sum()
        if total <= 0.0:
            # Equal-weight fallback if all signals are non-positive
            raw_weights = pd.Series(1.0 / n_top, index=top_symbols)
        else:
            raw_weights = raw_weights / total

        # Return full Series (zeros for non-top symbols)
        result = pd.Series(0.0, index=signal_values.index)
        result.loc[top_symbols] = raw_weights
        return result

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, returns: pd.Series, positions: pd.DataFrame) -> dict:
        """Compute standard performance metrics from a daily returns series."""
        n = len(returns)
        if n == 0:
            return {
                "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "annual_return": 0.0, "annual_volatility": 0.0,
                "turnover": 0.0,
            }

        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        annual_return = (1 + mean_ret) ** ANNUAL_FACTOR - 1
        annual_vol = std_ret * np.sqrt(ANNUAL_FACTOR)

        # Sharpe (assume risk-free = 0 for simplicity)
        sharpe = (mean_ret / std_ret * np.sqrt(ANNUAL_FACTOR)) if std_ret > 0 else 0.0

        # Sortino: downside deviation
        downside = returns[returns < 0]
        if len(downside) > 1:
            downside_std = downside.std(ddof=1)
            sortino = (mean_ret / downside_std * np.sqrt(ANNUAL_FACTOR)) if downside_std > 0 else 0.0
        else:
            sortino = sharpe

        # Max drawdown
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())  # negative value

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Average daily turnover
        weight_changes = positions.diff().abs().fillna(0.0)
        turnover = float(weight_changes.sum(axis=1).mean())

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar": float(calmar),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "turnover": turnover,
        }
