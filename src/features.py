from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
	"ret_1d",
	"ret_5d",
	"ret_21d",
	"ma_dist_10",
	"ma_dist_21",
	"vol_21d",
	"rsi_14",
	"vol_chg_5d",
]
logger = logging.getLogger(__name__)


def _validate_price_schema(price_df: pd.DataFrame) -> None:
	required = {"date", "ticker", "close", "volume"}
	missing = required - set(price_df.columns)
	if missing:
		raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
	delta = close.diff()
	up = delta.clip(lower=0.0)
	down = -delta.clip(upper=0.0)
	avg_up = up.rolling(period, min_periods=period).mean()
	avg_down = down.rolling(period, min_periods=period).mean()
	rs = avg_up / avg_down.replace(0.0, np.nan)
	return 100.0 - (100.0 / (1.0 + rs))


def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
	"""Create per-ticker factor features from daily OHLCV bars."""

	if price_df.empty:
		logger.warning("compute_features received empty input dataframe.")
		return price_df.copy()

	_validate_price_schema(price_df)

	df = price_df.copy()
	df["date"] = pd.to_datetime(df["date"])
	df["close"] = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
	df["volume"] = pd.to_numeric(df["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan)

	if bool((df["close"] <= 0).fillna(False).any()):
		raise ValueError("Column 'close' must be strictly positive for return-based features.")

	df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
	logger.info(
		"Computing features for %d rows across %d tickers.",
		len(df),
		df["ticker"].nunique(),
	)

	g = df.groupby("ticker", group_keys=False)
	returns = g["close"].pct_change()
	df["ret_1d"] = returns
	df["ret_5d"] = g["close"].pct_change(5)
	df["ret_21d"] = g["close"].pct_change(21)

	ma10 = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
	ma21 = g["close"].transform(lambda s: s.rolling(21, min_periods=21).mean())
	df["ma_dist_10"] = (df["close"] / ma10) - 1.0
	df["ma_dist_21"] = (df["close"] / ma21) - 1.0

	df["vol_21d"] = g["ret_1d"].transform(lambda s: s.rolling(21, min_periods=21).std())
	df["rsi_14"] = g["close"].transform(lambda s: _calc_rsi(s, 14))
	df["vol_chg_5d"] = g["volume"].pct_change(5)
	df[DEFAULT_FEATURE_COLUMNS] = df[DEFAULT_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
	logger.info("Feature computation complete. Non-null feature rows: %d", int(df[DEFAULT_FEATURE_COLUMNS].notna().all(axis=1).sum()))

	return df


def add_forward_return_target(feature_df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
	"""Append forward return target used by cross-sectional models."""

	if horizon_days <= 0:
		raise ValueError("horizon_days must be positive.")

	if feature_df.empty:
		logger.warning("add_forward_return_target received empty dataframe.")
		out = feature_df.copy()
		out["target_fwd_return"] = []
		return out

	df = feature_df.copy()
	if "close" not in df.columns:
		raise ValueError("feature_df must contain 'close' to build forward return target.")
	df["close"] = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
	if bool((df["close"] <= 0).fillna(False).any()):
		raise ValueError("Column 'close' must be strictly positive for forward return targets.")

	df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
	g = df.groupby("ticker", group_keys=False)
	fwd_price = g["close"].shift(-horizon_days)
	df["target_fwd_return"] = (fwd_price / df["close"]) - 1.0
	df["target_fwd_return"] = df["target_fwd_return"].replace([np.inf, -np.inf], np.nan)
	logger.info(
		"Forward target computed with horizon_days=%d. Non-null targets: %d",
		horizon_days,
		int(df["target_fwd_return"].notna().sum()),
	)
	return df


def cross_sectional_zscore(
	df: pd.DataFrame, feature_columns: Sequence[str] | None = None
) -> pd.DataFrame:
	"""Apply date-wise z-score normalization to selected feature columns."""

	if df.empty:
		logger.warning("cross_sectional_zscore received empty dataframe.")
		return df.copy()

	if "date" not in df.columns:
		raise ValueError("Input dataframe must contain 'date' for cross-sectional z-score normalization.")

	cols = list(feature_columns) if feature_columns else list(DEFAULT_FEATURE_COLUMNS)
	out = df.copy()
	available_cols = [col for col in cols if col in out.columns]
	if not available_cols:
		logger.warning("No requested feature columns found for z-scoring.")
		return out

	values = out[available_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
	grouped = values.groupby(out["date"])
	mean = grouped.transform("mean")
	std = grouped.transform("std").replace(0.0, np.nan)
	out[available_cols] = ((values - mean) / std).replace([np.inf, -np.inf], np.nan)
	logger.info("Cross-sectional z-score complete for %d columns.", len(available_cols))

	return out


def build_model_dataset(price_df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
	"""Convenience function to produce cleaned feature + target dataset."""

	feat_df = compute_features(price_df)
	feat_df = add_forward_return_target(feat_df, horizon_days=horizon_days)
	feat_df = cross_sectional_zscore(feat_df)
	feat_df[DEFAULT_FEATURE_COLUMNS + ["target_fwd_return"]] = feat_df[
		DEFAULT_FEATURE_COLUMNS + ["target_fwd_return"]
	].replace([np.inf, -np.inf], np.nan)
	feat_df = feat_df.dropna(subset=[*DEFAULT_FEATURE_COLUMNS, "target_fwd_return"]).reset_index(drop=True)
	logger.info("Model dataset ready with %d rows after cleaning.", len(feat_df))
	return feat_df
