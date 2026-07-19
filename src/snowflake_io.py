"""Snowflake connection and I/O helpers.

Snowflake is the enterprise **source + governed store** in this project. Raw claims
and features live here; only aggregate residual JSON and a KB-sized model artifact
ever cross to GCP.

Auth resolves in this order:
  1. ``~/.snowflake/connections.toml`` (connection name from ``SNOWFLAKE_CONNECTION``,
     default ``WEB``) — **password param only, `authenticator` is deliberately dropped**
     (the toml may carry a stale authenticator that breaks PAT auth).
  2. ``SNOWFLAKE_*`` environment variables.

Defaults target the dedicated demo objects: warehouse ``CARECOST_WH``,
database ``CARECOST_DEMO``, schema ``ANALYTICS`` (all droppable — see TEARDOWN.md).
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pandas as pd

WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "CARECOST_WH")
DATABASE = os.getenv("SNOWFLAKE_DATABASE", "CARECOST_DEMO")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "ANALYTICS")
_TOML = Path.home() / ".snowflake" / "connections.toml"


def _params_from_toml() -> dict | None:
    if not _TOML.exists():
        return None
    data = tomllib.loads(_TOML.read_text())
    name = os.getenv("SNOWFLAKE_CONNECTION", data.get("default_connection_name", "WEB"))
    cfg = data.get(name)
    if not isinstance(cfg, dict) or not cfg.get("password"):
        return None
    # password-only: drop `authenticator` on purpose (PAT is passed as password).
    return {
        "account": cfg["account"], "user": cfg["user"], "password": cfg["password"],
        "role": cfg.get("role"),
    }


def _params_from_env() -> dict:
    missing = [k for k in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD") if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"No Snowflake creds: {_TOML} has no usable connection and env vars {missing} are unset."
        )
    return {
        "account": os.environ["SNOWFLAKE_ACCOUNT"], "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"], "role": os.getenv("SNOWFLAKE_ROLE"),
    }


def get_connection():
    """Open a Snowflake connection pinned to the demo warehouse/db/schema."""
    try:
        import snowflake.connector as sf
    except ImportError as exc:
        raise RuntimeError("pip install 'snowflake-connector-python[pandas]'") from exc
    params = _params_from_toml() or _params_from_env()
    try:
        return sf.connect(
            warehouse=WAREHOUSE, database=DATABASE, schema=SCHEMA,
            **{k: v for k, v in params.items() if v is not None},
        )
    except sf.errors.DatabaseError as exc:
        if "not allowed to access" in str(exc):
            raise RuntimeError(
                "Snowflake blocked this IP (network policy). Turn OFF the Tailscale exit node "
                "and connect via home/DigitalOcean VPN so you egress from an allowlisted IP."
            ) from exc
        raise


def run_sql_script(conn, path: str | Path) -> None:
    """Execute a semicolon-delimited .sql file statement by statement.

    Strips whole-line ``--`` comments first (so a leading comment header never gets
    bundled with — and silently skip — the first real statement), then splits on ``;``.
    """
    raw = Path(path).read_text()
    sql = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("--"))
    cur = conn.cursor()
    try:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                cur.execute(stmt)
    except Exception as exc:
        raise RuntimeError(f"Failed executing {path}: {exc}") from exc
    finally:
        cur.close()


def write_df(conn, df: pd.DataFrame, table: str) -> int:
    """Write a DataFrame to an existing table via write_pandas. Returns row count."""
    from snowflake.connector.pandas_tools import write_pandas
    ok, _, nrows, _ = write_pandas(conn, df, table.upper(), auto_create_table=False)
    if not ok:
        raise RuntimeError(f"write_pandas reported failure for {table}")
    return nrows


def read_table(conn, sql: str) -> pd.DataFrame:
    """Run a query and return all rows as a pandas DataFrame."""
    cur = conn.cursor()
    try:
        cur.execute(sql)
        return cur.fetch_pandas_all()
    finally:
        cur.close()


def snowflake_credits(conn, warehouse: str = WAREHOUSE, minutes: int = 60) -> float:
    """Recent credits used by the demo warehouse — logged to Vertex as a metric.

    Reads ACCOUNT_USAGE (up to ~3h latency); returns 0.0 if unavailable.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COALESCE(SUM(CREDITS_USED),0) FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY "
            "WHERE WAREHOUSE_NAME=%s AND START_TIME >= DATEADD(minute,%s,CURRENT_TIMESTAMP())",
            (warehouse, -abs(minutes)),
        )
        return float(cur.fetchone()[0])
    except Exception:
        return 0.0
    finally:
        cur.close()
