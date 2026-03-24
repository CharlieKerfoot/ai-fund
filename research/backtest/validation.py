"""Walk-forward validation for signal backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from research.backtest.engine import BacktestEngine
from research.signals.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    folds: list[dict]          # per-fold: {train_start, train_end, test_start, test_end, sharpe, return}
    mean_oos_sharpe: float
    std_oos_sharpe: float
    is_sharpe: float           # in-sample (full period)
    oos_sharpe: float          # average out-of-sample
    degradation_ratio: float   # oos_sharpe / is_sharpe (healthy: >0.5)


class WalkForwardValidator:
    """Purged k-fold walk-forward cross-validator."""

    PURGE_GAP_DAYS = 21  # days between train and test to prevent leakage

    def __init__(
        self,
        engine: BacktestEngine,
        n_folds: int = 5,
        min_train_days: int = 252,
    ) -> None:
        self._engine = engine
        self._n_folds = n_folds
        self._min_train_days = min_train_days

    def validate(
        self,
        signal: Signal,
        universe: list[str],
        start: date,
        end: date,
    ) -> WalkForwardResult:
        """Purged k-fold walk-forward validation.

        Algorithm:
        1. Split [start, end] into n_folds equal test periods
        2. Train period = all data before test period (expanding window)
        3. Purge gap = 21 days between train end and test start (prevent leakage)
        4. Compute backtest for each fold
        5. Aggregate OOS metrics

        In-sample Sharpe is computed over the full [start, end] period.
        """
        total_days = (end - start).days
        fold_days = total_days // self._n_folds

        folds: list[dict] = []

        for i in range(self._n_folds):
            # Test period: non-overlapping slices from the end of history
            test_start = start + timedelta(days=i * fold_days)
            test_end = start + timedelta(days=(i + 1) * fold_days - 1)
            if i == self._n_folds - 1:
                test_end = end  # last fold gets any remaining days

            # Train period: all data before the purge gap
            train_start = start
            train_end = test_start - timedelta(days=self.PURGE_GAP_DAYS + 1)

            if (train_end - train_start).days < self._min_train_days:
                logger.warning(
                    "Fold %d: insufficient training data (%d days < min %d). Skipping.",
                    i + 1,
                    (train_end - train_start).days,
                    self._min_train_days,
                )
                folds.append({
                    "fold": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "sharpe": float("nan"),
                    "return": float("nan"),
                    "skipped": True,
                })
                continue

            try:
                result = self._engine.run(
                    signal=signal,
                    universe=universe,
                    start=test_start,
                    end=test_end,
                )
                folds.append({
                    "fold": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "sharpe": result.sharpe,
                    "return": result.annual_return,
                    "skipped": False,
                })
            except Exception as exc:
                logger.warning("Walk-forward fold %d failed: %s", i + 1, exc)
                folds.append({
                    "fold": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "sharpe": float("nan"),
                    "return": float("nan"),
                    "skipped": True,
                    "error": str(exc),
                })

        # In-sample Sharpe (full period)
        try:
            is_result = self._engine.run(signal=signal, universe=universe, start=start, end=end)
            is_sharpe = is_result.sharpe
        except Exception as exc:
            logger.warning("In-sample full backtest failed: %s", exc)
            is_sharpe = float("nan")

        # OOS metrics — only valid (non-skipped) folds
        valid_sharpes = [f["sharpe"] for f in folds if not f.get("skipped") and not np.isnan(f["sharpe"])]
        if valid_sharpes:
            mean_oos = float(np.mean(valid_sharpes))
            std_oos = float(np.std(valid_sharpes, ddof=1)) if len(valid_sharpes) > 1 else 0.0
        else:
            mean_oos = float("nan")
            std_oos = float("nan")

        degradation = (mean_oos / is_sharpe) if (is_sharpe and not np.isnan(is_sharpe) and not np.isnan(mean_oos)) else float("nan")

        return WalkForwardResult(
            folds=folds,
            mean_oos_sharpe=mean_oos,
            std_oos_sharpe=std_oos,
            is_sharpe=is_sharpe,
            oos_sharpe=mean_oos,
            degradation_ratio=degradation,
        )
