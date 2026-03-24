"""Stub for alternative data sources (satellite, shipping, patents)."""

from __future__ import annotations

import pandas as pd
import structlog

log = structlog.get_logger()


class AlternativeDataSource:
    """Stub for alternative data (satellite imagery, shipping traffic, patent filings).

    These data sources require specialized vendor contracts and are not
    implemented in the current version of the platform.
    """

    def fetch(self, data_type: str, **kwargs) -> pd.DataFrame:
        """Fetch alternative data of the specified type.

        Args:
            data_type: One of 'satellite', 'shipping', 'patents', etc.
            **kwargs: Additional parameters passed to the data provider.

        Raises:
            NotImplementedError: Always. This source is not yet implemented.
        """
        log.warning(
            "Alternative data source not implemented",
            data_type=data_type,
            kwargs=kwargs,
        )
        raise NotImplementedError(
            f"Alternative data source '{data_type}' not implemented. "
            "Contact the platform team to add support for this data type."
        )
