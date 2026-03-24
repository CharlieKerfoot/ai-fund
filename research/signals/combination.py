"""Signal combination, correlation analysis, and weight estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals.registry import SignalRegistry


class SignalCombiner:
    """Combines multiple signals with weighting and diversification analysis."""

    def __init__(self, registry: SignalRegistry) -> None:
        self._registry = registry

    def combine(self, signals: list[pd.Series], weights: list[float]) -> pd.Series:
        """Weighted average of signal values. Output in [-1, 1].

        Args:
            signals: list of pd.Series indexed by symbol, values in [-1, 1].
            weights: list of floats summing to 1.0 (or will be normalized).

        Returns:
            pd.Series indexed by union of symbols, values in [-1, 1].
        """
        if not signals:
            return pd.Series(dtype=float)

        weights_arr = np.array(weights, dtype=float)
        total = weights_arr.sum()
        if total == 0:
            raise ValueError("Weights must sum to a non-zero value")
        weights_arr = weights_arr / total

        # Align all signals to a common index
        all_symbols = set()
        for s in signals:
            all_symbols.update(s.index.tolist())
        all_symbols_list = sorted(all_symbols)

        combined = pd.Series(0.0, index=all_symbols_list)
        weight_sum = pd.Series(0.0, index=all_symbols_list)

        for s, w in zip(signals, weights_arr):
            aligned = s.reindex(all_symbols_list)
            mask = aligned.notna()
            combined[mask] += aligned[mask] * w
            weight_sum[mask] += w

        # Renormalize for symbols with missing signals
        nonzero = weight_sum > 0
        combined[nonzero] = combined[nonzero] / weight_sum[nonzero]
        combined[~nonzero] = np.nan

        return combined.clip(-1.0, 1.0)

    def equal_weight(self, signal_values: dict[str, pd.Series]) -> pd.Series:
        """Equal-weight combination of named signals.

        Args:
            signal_values: dict of signal_name → pd.Series of signal values.

        Returns:
            pd.Series indexed by symbol, values in [-1, 1].
        """
        if not signal_values:
            return pd.Series(dtype=float)

        series_list = list(signal_values.values())
        n = len(series_list)
        weights = [1.0 / n] * n
        return self.combine(series_list, weights)

    def compute_correlation_matrix(
        self, signal_history: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Compute cross-signal correlation matrix from historical signal values.

        Args:
            signal_history: dict of signal_name → pd.Series (index = dates or symbols).

        Returns:
            pd.DataFrame correlation matrix (signal_name × signal_name).
        """
        if not signal_history:
            return pd.DataFrame()

        df = pd.DataFrame(signal_history)
        return df.corr()

    def analyze_diversification(
        self, signal_history: dict[str, pd.Series]
    ) -> dict:
        """Compute diversification metrics across signals.

        Returns dict with:
            mean_pairwise_corr: average off-diagonal correlation
            max_corr: maximum pairwise correlation
            min_corr: minimum pairwise correlation
            diversification_score: 1 - mean_pairwise_corr (higher = more diverse)
        """
        corr_matrix = self.compute_correlation_matrix(signal_history)

        if corr_matrix.empty or corr_matrix.shape[0] < 2:
            return {
                "mean_pairwise_corr": float("nan"),
                "max_corr": float("nan"),
                "min_corr": float("nan"),
                "diversification_score": float("nan"),
            }

        n = corr_matrix.shape[0]
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        off_diag = corr_matrix.values[mask]

        mean_corr = float(np.nanmean(off_diag))
        max_corr = float(np.nanmax(off_diag))
        min_corr = float(np.nanmin(off_diag))
        diversification_score = 1.0 - mean_corr

        return {
            "mean_pairwise_corr": mean_corr,
            "max_corr": max_corr,
            "min_corr": min_corr,
            "diversification_score": diversification_score,
        }

    def estimate_optimal_weights(
        self, signal_returns: dict[str, pd.Series]
    ) -> dict[str, float]:
        """Simple Sharpe-weighted allocation.

        weight = max(0, sharpe) / sum(max(0, sharpe))

        Args:
            signal_returns: dict of signal_name → pd.Series of returns.

        Returns:
            dict of signal_name → weight (sums to 1.0, or all zeros if all negative Sharpe).
        """
        sharpes: dict[str, float] = {}
        for name, returns in signal_returns.items():
            clean = returns.dropna()
            if len(clean) < 2:
                sharpes[name] = 0.0
                continue
            mean = clean.mean()
            std = clean.std(ddof=1)
            sharpes[name] = float(mean / std) if std > 0 else 0.0

        positive_sharpes = {n: max(0.0, s) for n, s in sharpes.items()}
        total = sum(positive_sharpes.values())

        if total == 0:
            n = len(signal_returns)
            return {name: 1.0 / n for name in signal_returns} if n > 0 else {}

        return {name: val / total for name, val in positive_sharpes.items()}
