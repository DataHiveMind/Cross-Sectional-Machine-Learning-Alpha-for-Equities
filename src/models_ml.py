from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.features import DEFAULT_FEATURE_COLUMNS

try:
	from xgboost import XGBRegressor
except Exception:  # pragma: no cover
	XGBRegressor = None


logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
	name: str = "random_forest"
	random_state: int = 42


@dataclass
class ValidationSpec:
	n_splits: int = 5
	purge_days: int = 0
	embargo_days: int = 0
	min_train_dates: int = 30


def _validate_model_dataset(
	dataset: pd.DataFrame, feature_columns: Sequence[str], target_column: str
) -> pd.DataFrame:
	required = {"date", "ticker", target_column, *feature_columns}
	missing = required - set(dataset.columns)
	if missing:
		raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

	df = dataset.copy()
	df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
	df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
	df[feature_columns + [target_column]] = df[feature_columns + [target_column]].replace(
		[float("inf"), float("-inf")], pd.NA
	)
	return df


def build_model(spec: ModelSpec | None = None) -> RegressorMixin:
	"""Construct a supported regression model."""

	cfg = spec or ModelSpec()
	logger.info("Building model '%s' with random_state=%d.", cfg.name, cfg.random_state)
	if cfg.name == "xgboost":
		if XGBRegressor is None:
			raise ImportError("xgboost is not available. Install xgboost or choose random_forest.")
		return XGBRegressor(
			n_estimators=300,
			max_depth=5,
			learning_rate=0.05,
			subsample=0.9,
			colsample_bytree=0.9,
			objective="reg:squarederror",
			random_state=cfg.random_state,
		)

	if cfg.name == "random_forest":
		return RandomForestRegressor(
			n_estimators=250,
			max_depth=8,
			min_samples_leaf=20,
			random_state=cfg.random_state,
			n_jobs=-1,
		)

	raise ValueError(f"Unsupported model name: {cfg.name}")


def train_model(
	dataset: pd.DataFrame,
	feature_columns: Sequence[str] | None = None,
	target_column: str = "target_fwd_return",
	spec: ModelSpec | None = None,
) -> RegressorMixin:
	"""Fit a model on all provided rows and return the trained instance."""

	cols = list(feature_columns) if feature_columns else list(DEFAULT_FEATURE_COLUMNS)
	df = _validate_model_dataset(dataset, cols, target_column)
	df = df.dropna(subset=[*cols, target_column]).copy()
	if df.empty:
		raise ValueError("No valid rows available for training after NaN/Inf filtering.")
	logger.info("Training model on %d rows and %d features.", len(df), len(cols))

	model = build_model(spec)
	model.fit(df[cols], df[target_column])
	return model


def walk_forward_predictions(
	dataset: pd.DataFrame,
	feature_columns: Sequence[str] | None = None,
	target_column: str = "target_fwd_return",
	n_splits: int = 5,
	spec: ModelSpec | None = None,
	validation: ValidationSpec | None = None,
) -> pd.DataFrame:
	"""Generate out-of-sample predictions using TimeSeriesSplit on dates."""
	val_cfg = validation or ValidationSpec(n_splits=n_splits)
	if val_cfg.n_splits < 2:
		raise ValueError("n_splits must be at least 2.")
	if val_cfg.purge_days < 0 or val_cfg.embargo_days < 0:
		raise ValueError("purge_days and embargo_days must be non-negative.")
	if val_cfg.min_train_dates <= 0:
		raise ValueError("min_train_dates must be positive.")

	cols = list(feature_columns) if feature_columns else list(DEFAULT_FEATURE_COLUMNS)
	df = _validate_model_dataset(dataset, cols, target_column)
	df = df.dropna(subset=["date", "ticker", *cols, target_column]).copy()
	df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
	if df.empty:
		logger.warning("No valid rows for walk-forward prediction after filtering.")
		return pd.DataFrame(columns=["date", "ticker", target_column, "prediction"])

	dates = pd.to_datetime(df["date"])
	unique_dates = pd.Index(dates.sort_values().unique())
	if len(unique_dates) < (val_cfg.n_splits + 1):
		raise ValueError("Not enough unique dates for requested TimeSeriesSplit.")
	logger.info(
		"Running walk-forward predictions with %d rows, %d unique dates, %d splits, purge=%d, embargo=%d.",
		len(df),
		len(unique_dates),
		val_cfg.n_splits,
		val_cfg.purge_days,
		val_cfg.embargo_days,
	)

	tscv = TimeSeriesSplit(n_splits=val_cfg.n_splits)
	out_chunks: list[pd.DataFrame] = []

	for fold_num, (train_idx, test_idx) in enumerate(tscv.split(unique_dates), start=1):
		# Purge dates immediately before test start to reduce leakage via overlapping windows.
		test_start = int(test_idx.min())
		if val_cfg.purge_days > 0:
			train_idx = train_idx[train_idx < max(0, test_start - val_cfg.purge_days)]

		# Embargo retained for API completeness; with strictly past-only walk-forward it has no effect.
		if len(train_idx) < val_cfg.min_train_dates:
			logger.warning(
				"Skipping fold %d due to insufficient train dates after purging (%d < %d).",
				fold_num,
				len(train_idx),
				val_cfg.min_train_dates,
			)
			continue

		train_dates = unique_dates[train_idx]
		test_dates = unique_dates[test_idx]

		train_mask = dates.isin(train_dates)
		test_mask = dates.isin(test_dates)

		train_df = df.loc[train_mask]
		test_df = df.loc[test_mask]
		if train_df.empty or test_df.empty:
			logger.warning("Skipping fold %d because train or test set is empty.", fold_num)
			continue
		logger.info(
			"Fold %d: training rows=%d, test rows=%d.",
			fold_num,
			len(train_df),
			len(test_df),
		)

		model = build_model(spec)
		model.fit(train_df[cols], train_df[target_column])
		preds = model.predict(test_df[cols])
		preds = pd.Series(preds).replace([float("inf"), float("-inf")], pd.NA)

		fold = test_df[["date", "ticker", target_column]].copy()
		fold["prediction"] = preds.to_numpy(dtype=float, na_value=np.nan)
		fold = fold.dropna(subset=["prediction"])
		if fold.empty:
			continue
		out_chunks.append(fold)

	if not out_chunks:
		logger.warning("No predictions produced across all folds.")
		return pd.DataFrame(columns=["date", "ticker", target_column, "prediction"])

	out = pd.concat(out_chunks, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
	logger.info("Generated %d out-of-sample predictions.", len(out))
	return out
