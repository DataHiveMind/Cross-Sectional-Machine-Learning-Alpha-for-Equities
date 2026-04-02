from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import DEFAULT_FEATURE_COLUMNS
from src.models_ml import ModelSpec, ValidationSpec, train_model, walk_forward_predictions


def _make_model_dataset(days: int = 70) -> pd.DataFrame:
	dates = pd.date_range("2023-01-01", periods=days, freq="B")
	tickers = np.array(["AAA", "BBB", "CCC", "DDD"])

	date_col = np.tile(dates.to_numpy(), len(tickers))
	ticker_col = np.repeat(tickers, len(dates))
	idx = np.tile(np.arange(len(dates), dtype=float), len(tickers))
	base = np.repeat(np.linspace(0.0, 0.03, len(tickers)), len(dates))

	df = pd.DataFrame({"date": date_col, "ticker": ticker_col})
	for i, col in enumerate(DEFAULT_FEATURE_COLUMNS, start=1):
		df[col] = base + (idx * (0.0003 * i))

	df["target_fwd_return"] = 0.01 * df["ret_5d"].fillna(0.0) + 0.02 * df["ret_21d"].fillna(0.0)
	return df


def test_train_model_rejects_missing_required_columns() -> None:
	ds = _make_model_dataset().drop(columns=["target_fwd_return"])
	with pytest.raises(ValueError, match="missing required columns"):
		train_model(ds)


def test_train_model_rejects_all_non_finite_rows() -> None:
	ds = _make_model_dataset()
	ds[DEFAULT_FEATURE_COLUMNS] = np.inf
	with pytest.raises(ValueError, match="No valid rows"):
		train_model(ds, spec=ModelSpec(name="random_forest"))


def test_walk_forward_predictions_returns_finite_predictions() -> None:
	ds = _make_model_dataset()
	preds = walk_forward_predictions(ds, n_splits=5, spec=ModelSpec(name="random_forest"))

	assert not preds.empty
	assert set(["date", "ticker", "target_fwd_return", "prediction"]).issubset(preds.columns)
	assert np.isfinite(preds["prediction"]).all()


def test_walk_forward_predictions_rejects_bad_split_count() -> None:
	ds = _make_model_dataset()
	with pytest.raises(ValueError, match="n_splits"):
		walk_forward_predictions(ds, n_splits=1)


def test_walk_forward_predictions_rejects_negative_purge_days() -> None:
	ds = _make_model_dataset()
	with pytest.raises(ValueError, match="purge_days"):
		walk_forward_predictions(ds, validation=ValidationSpec(n_splits=5, purge_days=-1))


def test_walk_forward_predictions_with_validation_spec_runs() -> None:
	ds = _make_model_dataset()
	preds = walk_forward_predictions(
		ds,
		spec=ModelSpec(name="random_forest"),
		validation=ValidationSpec(n_splits=5, purge_days=2, embargo_days=1, min_train_dates=20),
	)
	assert not preds.empty
