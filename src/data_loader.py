from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import yfinance as yf
from sqlalchemy import delete, tuple_
from sqlalchemy.orm import Session

from src.models_db import PriceBar


_BAR_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
logger = logging.getLogger(__name__)


def _validate_bar_schema(df: pd.DataFrame) -> None:
	required = set(_BAR_COLUMNS)
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns for price bars: {sorted(missing)}")


def fetch_ohlcv_yfinance(
	tickers: Iterable[str], start_date: str, end_date: str, auto_adjust: bool = True
) -> pd.DataFrame:
	"""Fetch daily OHLCV bars for a ticker universe from yfinance."""
	ticker_list = [str(t).strip() for t in tickers if str(t).strip()]
	if not ticker_list:
		logger.warning("No tickers provided; returning empty price-bar dataframe.")
		return pd.DataFrame(columns=_BAR_COLUMNS)

	logger.info(
		"Downloading OHLCV bars for %d tickers from %s to %s (auto_adjust=%s).",
		len(ticker_list),
		start_date,
		end_date,
		auto_adjust,
	)

	df = yf.download(
		ticker_list,
		start=start_date,
		end=end_date,
		progress=False,
		auto_adjust=auto_adjust,
		group_by="ticker",
	)
	if df.empty:
		logger.warning("yfinance returned no rows for requested universe/date range.")
		return pd.DataFrame(columns=_BAR_COLUMNS)

	if isinstance(df.columns, pd.MultiIndex):
		required = {"Open", "High", "Low", "Close", "Volume"}
		if df.columns.nlevels < 2:
			return pd.DataFrame(columns=_BAR_COLUMNS)

		if df.columns.names[0] == "Ticker":
			stacked = df.stack(level=0, future_stack=True).reset_index()
		else:
			stacked = df.stack(level=1, future_stack=True).reset_index()

		date_col = stacked.columns[0]
		ticker_col = stacked.columns[1]
		stacked = stacked.rename(columns={date_col: "date", ticker_col: "ticker"})

		if not required.issubset(stacked.columns):
			return pd.DataFrame(columns=_BAR_COLUMNS)

		out = stacked.rename(
			columns={
				"Open": "open",
				"High": "high",
				"Low": "low",
				"Close": "close",
				"Volume": "volume",
			}
		)[_BAR_COLUMNS]
	else:
		required = {"Open", "High", "Low", "Close", "Volume"}
		if not required.issubset(set(df.columns)):
			return pd.DataFrame(columns=_BAR_COLUMNS)
		out = df.reset_index().rename(
			columns={
				"Date": "date",
				"Open": "open",
				"High": "high",
				"Low": "low",
				"Close": "close",
				"Volume": "volume",
			}
		)
		out["ticker"] = ticker_list[0]
		out = out[_BAR_COLUMNS]

	out["date"] = pd.to_datetime(out["date"]).dt.date
	for col in ["open", "high", "low", "close", "volume"]:
		out[col] = pd.to_numeric(out[col], errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
	out = out.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
	logger.info("Fetched %d rows across %d tickers.", len(out), out["ticker"].nunique())
	return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def upsert_price_bars(session: Session, bars_df: pd.DataFrame) -> None:
	"""Persist bars by replacing existing rows for the same (date, ticker)."""

	if bars_df.empty:
		logger.warning("No bars to upsert; skipping database write.")
		return

	_validate_bar_schema(bars_df)
	work = bars_df[_BAR_COLUMNS].copy()
	work["date"] = pd.to_datetime(work["date"]).dt.date
	for col in ["open", "high", "low", "close", "volume"]:
		work[col] = pd.to_numeric(work[col], errors="coerce")

	work = work.replace([float("inf"), float("-inf")], pd.NA)
	work = work.dropna(subset=_BAR_COLUMNS)
	if work.empty:
		logger.warning("All bars dropped after numeric sanitization; skipping database write.")
		return

	if bool((work["close"] <= 0).any()):
		raise ValueError("Column 'close' must be strictly positive.")
	if bool((work["volume"] < 0).any()):
		raise ValueError("Column 'volume' cannot be negative.")

	unique_keys = work[["date", "ticker"]].drop_duplicates()
	key_tuples = list(unique_keys.itertuples(index=False, name=None))
	logger.info(
		"Upserting %d rows for %d unique (date, ticker) keys.",
		len(work),
		len(key_tuples),
	)
	session.execute(delete(PriceBar).where(tuple_(PriceBar.date, PriceBar.ticker).in_(key_tuples)))

	payload = [PriceBar(**rec) for rec in work.to_dict(orient="records")]
	session.add_all(payload)


def load_price_bars_to_dataframe(session: Session, start_date: str | None = None) -> pd.DataFrame:
	"""Read bars from SQL into a pandas DataFrame."""
	logger.info("Loading price bars from SQL (start_date=%s).", start_date)

	query = session.query(PriceBar)
	if start_date:
		start = pd.to_datetime(start_date).date()
		query = query.filter(PriceBar.date >= start)

	rows = query.order_by(PriceBar.date, PriceBar.ticker).all()
	if not rows:
		logger.warning("No rows found in price_bars for requested filter.")
		return pd.DataFrame(columns=_BAR_COLUMNS)

	data = [
		{
			"date": r.date,
			"ticker": r.ticker,
			"open": r.open,
			"high": r.high,
			"low": r.low,
			"close": r.close,
			"volume": r.volume,
		}
		for r in rows
	]
	out = pd.DataFrame(data, columns=_BAR_COLUMNS)
	logger.info("Loaded %d rows from SQL.", len(out))
	return out
