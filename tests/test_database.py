from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from src.data_loader import load_price_bars_to_dataframe, upsert_price_bars
from src.database import create_all_tables, drop_all_tables, get_engine, get_session_factory, session_scope


def test_create_and_drop_tables() -> None:
	engine = get_engine("sqlite+pysqlite:///:memory:")
	create_all_tables(engine)

	with engine.connect() as conn:
		names = [row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))]
	assert "price_bars" in names

	drop_all_tables(engine)
	with engine.connect() as conn:
		names_after = [
			row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
		]
	assert "price_bars" not in names_after


def test_upsert_and_load_price_bars_round_trip() -> None:
	engine = get_engine("sqlite+pysqlite:///:memory:")
	create_all_tables(engine)
	factory = get_session_factory(engine)

	bars = pd.DataFrame(
		{
			"date": ["2024-01-02", "2024-01-02", "2024-01-03"],
			"ticker": ["AAPL", "MSFT", "AAPL"],
			"open": [100.0, 200.0, 101.0],
			"high": [102.0, 202.0, 103.0],
			"low": [99.0, 198.0, 100.0],
			"close": [101.0, 201.0, 102.0],
			"volume": [1_000_000, 1_500_000, 1_100_000],
		}
	)

	with session_scope(factory) as session:
		upsert_price_bars(session, bars)

	with session_scope(factory) as session:
		out = load_price_bars_to_dataframe(session)

	assert len(out) == 3
	assert set(out["ticker"]) == {"AAPL", "MSFT"}


def test_upsert_price_bars_rejects_invalid_schema() -> None:
	engine = get_engine("sqlite+pysqlite:///:memory:")
	create_all_tables(engine)
	factory = get_session_factory(engine)

	bad = pd.DataFrame({"date": ["2024-01-02"], "ticker": ["AAPL"]})
	with session_scope(factory) as session:
		with pytest.raises(ValueError, match="Missing required columns"):
			upsert_price_bars(session, bad)


def test_upsert_price_bars_rejects_bad_numeric_values() -> None:
	engine = get_engine("sqlite+pysqlite:///:memory:")
	create_all_tables(engine)
	factory = get_session_factory(engine)

	bad = pd.DataFrame(
		{
			"date": ["2024-01-02"],
			"ticker": ["AAPL"],
			"open": [100.0],
			"high": [102.0],
			"low": [99.0],
			"close": [0.0],
			"volume": [1_000_000],
		}
	)

	with session_scope(factory) as session:
		with pytest.raises(ValueError, match="strictly positive"):
			upsert_price_bars(session, bad)
