"""Hierarchical Risk Parity optimizer (Lopez de Prado 2016)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class HierarchicalRiskParity:
    """
    Implements Hierarchical Risk Parity (HRP) portfolio optimization.

    Reference: Lopez de Prado (2016), "Building Diversified Portfolios that
    Outperform Out of Sample".
    """

    def __init__(self, lookback_days: int = 252) -> None:
        self.lookback_days = lookback_days

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: dict | None = None,
    ) -> pd.Series:
        """
        HRP algorithm:
        1. Compute correlation matrix from returns
        2. Hierarchical clustering (single-linkage on angular distance)
        3. Quasi-diagonalization: reorder assets by cluster
        4. Recursive bisection: allocate inversely proportional to cluster variance
        5. Apply constraints (position caps, sector limits)

        Returns: pd.Series of weights summing to 1.0
        """
        if returns.empty or returns.shape[1] < 2:
            n = returns.shape[1] if not returns.empty else 1
            symbols = returns.columns.tolist()
            return pd.Series(
                [1.0 / n] * n, index=symbols
            )

        # Use the most recent lookback_days
        if len(returns) > self.lookback_days:
            returns = returns.iloc[-self.lookback_days:]

        # Drop columns with all-NaN or zero variance
        returns = returns.dropna(axis=1, how="all")
        valid_cols = returns.columns[returns.std() > 1e-10]
        returns = returns[valid_cols]

        if returns.shape[1] < 2:
            n = returns.shape[1]
            return pd.Series([1.0 / n] * n, index=returns.columns)

        corr = returns.corr()
        cov = returns.cov()

        # Step 1: Cluster
        ordered_symbols = self._cluster(corr)

        # Step 2: Recursive bisection
        weights = self._recursive_bisection(cov, ordered_symbols)

        # Step 3: Apply constraints
        if constraints:
            weights = self._apply_constraints(weights, constraints)

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def _cluster(self, corr: pd.DataFrame) -> list:
        """
        Single-linkage hierarchical clustering.
        Returns ordered list of symbols using quasi-diagonalization.
        """
        # Angular distance: d = sqrt(0.5 * (1 - corr))
        # Clip correlation to [-1, 1] for numerical stability
        corr_clipped = corr.clip(-1, 1)
        dist = np.sqrt(0.5 * (1.0 - corr_clipped.values))
        np.fill_diagonal(dist, 0.0)

        # Convert to condensed form for scipy
        condensed = squareform(dist, checks=False)

        # Single-linkage hierarchical clustering
        link = linkage(condensed, method="single")

        # Get ordering via dendrogram traversal
        root, _ = to_tree(link, rd=True)
        ordered_indices = self._get_leaf_order(root)

        symbols = corr.columns.tolist()
        return [symbols[i] for i in ordered_indices]

    def _get_leaf_order(self, node) -> list[int]:
        """Recursively get leaf node indices in order."""
        if node.is_leaf():
            return [node.id]
        left = self._get_leaf_order(node.left)
        right = self._get_leaf_order(node.right)
        return left + right

    def _recursive_bisection(
        self, cov: pd.DataFrame, ordered_symbols: list
    ) -> pd.Series:
        """
        Allocate weights via recursive bisection on cluster variance.
        Each cluster receives weight inversely proportional to its variance.
        """
        weights = pd.Series(1.0, index=ordered_symbols)

        clusters = [ordered_symbols]

        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) == 1:
                    continue

                # Split cluster in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Compute cluster variance for each sub-cluster
                left_var = self._cluster_variance(cov, left, weights)
                right_var = self._cluster_variance(cov, right, weights)

                # Allocate inversely proportional to variance
                total_var = left_var + right_var
                if total_var <= 0:
                    alpha = 0.5
                else:
                    alpha = 1.0 - left_var / total_var  # weight for left cluster

                # Scale weights
                weights[left] *= alpha
                weights[right] *= (1.0 - alpha)

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)

            clusters = next_clusters

        return weights

    def _cluster_variance(
        self, cov: pd.DataFrame, symbols: list, weights: pd.Series
    ) -> float:
        """Compute the variance of a sub-cluster using current weight proportions."""
        sub_cov = cov.loc[symbols, symbols]
        weights[symbols]

        # Use inverse-vol weights within sub-cluster for variance calculation
        vols = np.sqrt(np.diag(sub_cov.values))
        inv_vol = 1.0 / np.where(vols > 1e-10, vols, 1e-10)
        inv_vol_weights = inv_vol / inv_vol.sum()

        var = inv_vol_weights @ sub_cov.values @ inv_vol_weights
        return float(var)

    def _apply_constraints(
        self, weights: pd.Series, constraints: dict
    ) -> pd.Series:
        """Apply position caps from constraints dict iteratively."""
        max_pos = constraints.get("max_position_pct", 1.0)

        if max_pos >= 1.0:
            return weights

        w = weights.copy()

        # Normalize first
        total = w.sum()
        if total > 0:
            w /= total

        # Iteratively cap and renormalize until convergence
        for _ in range(100):
            over = w > max_pos
            if not over.any():
                break
            # Redistribute excess proportionally to under-cap positions
            excess = (w[over] - max_pos).sum()
            w[over] = max_pos
            under = ~over & (w > 0)
            if under.any():
                under_total = w[under].sum()
                if under_total > 0:
                    w[under] += excess * (w[under] / under_total)
            else:
                break

        return w
