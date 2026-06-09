"""Analytical feature store.

Backed by DuckDB when available (matching the blueprint's "in-memory
analytical store"), otherwise by an in-memory dict of pandas DataFrames with
Parquet persistence. Both backends expose the same small API: ``put``, ``get``,
``query`` and ``tables``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wc2026.utils.logging import get_logger

logger = get_logger("data.store")


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


class FeatureStore:
    """Tiny table store with optional DuckDB acceleration.

    Parameters
    ----------
    path:
        Directory used for Parquet persistence (and the DuckDB file).
    use_duckdb:
        Force DuckDB on/off. ``None`` auto-detects.
    """

    def __init__(self, path: str | Path = "data/store", use_duckdb: bool | None = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if use_duckdb is None:
            use_duckdb = _duckdb_available()
        self._use_duckdb = use_duckdb and _duckdb_available()
        self._frames: dict[str, Any] = {}
        self._con = None
        if self._use_duckdb:
            import duckdb

            self._con = duckdb.connect(str(self.path / "wc2026.duckdb"))
            logger.debug("FeatureStore using DuckDB backend at %s", self.path)
        else:
            logger.debug("FeatureStore using pandas/Parquet backend at %s", self.path)

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return "duckdb" if self._use_duckdb else "pandas"

    def put(self, name: str, frame) -> None:
        """Register/replace a table."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        self._frames[name] = frame
        if self._use_duckdb:
            self._con.register(f"_tmp_{name}", frame)
            self._con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_{name}")
            self._con.unregister(f"_tmp_{name}")

    def get(self, name: str):
        """Return a table as a DataFrame, or raise ``KeyError``."""
        if name in self._frames:
            return self._frames[name]
        if self._use_duckdb:
            try:
                return self._con.execute(f"SELECT * FROM {name}").df()
            except Exception as exc:  # pragma: no cover - duckdb specific
                raise KeyError(name) from exc
        raise KeyError(name)

    def tables(self) -> list[str]:
        return sorted(self._frames.keys())

    def query(self, sql: str):
        """Run a SQL query against the registered tables.

        With the pandas backend this falls back to DuckDB's in-memory engine if
        installed; otherwise it raises a helpful error directing callers to use
        DataFrame operations instead.
        """
        if self._use_duckdb:
            return self._con.execute(sql).df()
        # Opportunistic in-memory query via a throwaway duckdb connection.
        if _duckdb_available():
            import duckdb

            con = duckdb.connect()
            for name, frame in self._frames.items():
                con.register(name, frame)
            try:
                return con.execute(sql).df()
            finally:
                con.close()
        raise RuntimeError(
            "SQL query requires duckdb (`pip install duckdb`); "
            "use FeatureStore.get(name) and pandas operations instead."
        )

    # --- persistence ---------------------------------------------------
    def save(self) -> None:
        """Persist all in-memory tables to Parquet (best-effort)."""
        for name, frame in self._frames.items():
            target = self.path / f"{name}.parquet"
            try:
                frame.to_parquet(target, index=False)
            except Exception:  # pyarrow missing → CSV fallback
                frame.to_csv(target.with_suffix(".csv"), index=False)

    def load(self) -> None:
        """Load any persisted tables from disk into memory."""
        import pandas as pd

        for parquet in self.path.glob("*.parquet"):
            self.put(parquet.stem, pd.read_parquet(parquet))
        for csv in self.path.glob("*.csv"):
            if csv.stem not in self._frames:
                self.put(csv.stem, pd.read_csv(csv))

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
