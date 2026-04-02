from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
	DEFAULT_FEATURE_COLUMNS,
	add_forward_return_target,
	build_model_dataset,
	compute_features,
	cross_sectional_zscore,
)


def _make_price_data(days: int = 80) -> pd.DataFrame:
	dates = pd.date_range("2023-01-01", periods=days, freq="B")
	tickers = np.array(["AAA", "BBB", "CCC"])
	bases = np.array([100.0, 80.0, 120.0])
	trend = np.array([0.0008, 0.0005, 0.0010])
	osc = np.array([0.0030, 0.0025, 0.0035])
	phases = np.array([0.0, 0.7, 1.4])
	vol_bases = np.array([1_000_000.0, 1_300_000.0, 850_000.0])

	idx = np.arange(len(dates), dtype=float)
	date_col = np.tile(dates.to_numpy(), len(tickers))
	ticker_col = np.repeat(tickers, len(dates))
	base_col = np.repeat(bases, len(dates))
	trend_col = np.repeat(trend, len(dates))
	osc_col = np.repeat(osc, len(dates))
	phase_col = np.repeat(phases, len(dates))
	vol_base_col = np.repeat(vol_bases, len(dates))
	idx_col = np.tile(idx, len(tickers))

	df = pd.DataFrame(
		{
			"date": date_col,
			"ticker": ticker_col,
			"base": base_col,
			"trend": trend_col,
			"osc": osc_col,
			"phase": phase_col,
			"idx": idx_col,
		}
	)
	df["daily_ret"] = df["trend"] + df["osc"] * np.sin(df["idx"] / 3.0 + df["phase"])
	df["close"] = df["base"] * (1.0 + df["daily_ret"]).groupby(df["ticker"]).cumprod()
	df["volume"] = vol_base_col * (1.0 + 0.0003 * df["idx"] + 0.02 * np.cos(df["idx"] / 8.0 + df["phase"]))

	return pd.DataFrame(
		{
			"date": df["date"].to_numpy(),
			"ticker": df["ticker"].to_numpy(),
			"open": (df["close"] * 0.995).to_numpy(),
			"high": (df["close"] * 1.005).to_numpy(),
			"low": (df["close"] * 0.99).to_numpy(),
			"close": df["close"].to_numpy(),
			"volume": df["volume"].to_numpy(),
		}
	)


def test_compute_features_adds_expected_columns() -> None:
	prices = _make_price_data()
	out = compute_features(prices)

	for col in DEFAULT_FEATURE_COLUMNS:
		assert col in out.columns


def test_forward_target_generation() -> None:
	prices = _make_price_data()
	feat = compute_features(prices)
	out = add_forward_return_target(feat, horizon_days=5)
	assert "target_fwd_return" in out.columns
	assert out["target_fwd_return"].notna().sum() > 0


def test_cross_sectional_zscore_daily_mean_close_to_zero() -> None:
	prices = _make_price_data()
	feat = add_forward_return_target(compute_features(prices), horizon_days=5)
	z = cross_sectional_zscore(feat)

	daily_means = z.groupby("date")["ret_1d"].mean().dropna()
	assert np.isclose(daily_means.abs().mean(), 0.0, atol=1e-6)


def test_build_model_dataset_returns_non_empty_clean_table() -> None:
	prices = _make_price_data()
	ds = build_model_dataset(prices, horizon_days=5)

	assert not ds.empty
	assert ds[DEFAULT_FEATURE_COLUMNS + ["target_fwd_return"]].isna().sum().sum() == 0


def test_compute_features_rejects_non_positive_close() -> None:
	prices = _make_price_data()
	prices.loc[prices.index[0], "close"] = 0.0
	with pytest.raises(ValueError, match="strictly positive"):
		compute_features(prices)


def test_add_forward_return_target_rejects_invalid_horizon() -> None:
	with pytest.raises(ValueError, match="horizon_days"):
		add_forward_return_target(_make_price_data(), horizon_days=0)


def test_cross_sectional_zscore_requires_date_column() -> None:
	df = pd.DataFrame({"ret_1d": [1.0, 2.0]})
	with pytest.raises(ValueError, match="date"):
		cross_sectional_zscore(df, feature_columns=["ret_1d"])
