from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import assign_portfolio_weights, compute_daily_strategy_returns, summarize_performance
from src.backtester import apply_risk_constraints, compute_daily_strategy_returns_realistic


def _predictions_sample() -> pd.DataFrame:
	return pd.DataFrame(
		{
			"date": ["2024-01-02"] * 6 + ["2024-01-03"] * 6,
			"ticker": ["A", "B", "C", "D", "E", "F"] * 2,
			"prediction": [0.05, 0.04, 0.03, -0.02, -0.03, -0.04, 0.06, 0.03, 0.01, -0.01, -0.02, -0.05],
			"target_fwd_return": [0.02, 0.01, 0.005, -0.004, -0.01, -0.02, 0.03, 0.01, 0.004, -0.005, -0.01, -0.03],
		}
	)


def test_assign_portfolio_weights_is_dollar_neutral() -> None:
	df = _predictions_sample()
	weighted = assign_portfolio_weights(df, quantile=1 / 3)

	sums = weighted.groupby("date")["weight"].sum()
	assert np.allclose(sums.values, 0.0)


def test_compute_daily_strategy_returns_has_expected_shape() -> None:
	df = _predictions_sample()
	weighted = assign_portfolio_weights(df, quantile=1 / 3)
	daily = compute_daily_strategy_returns(weighted)

	assert list(daily.columns) == ["date", "strategy_return"]
	assert len(daily) == 2


def test_summarize_performance_contains_key_metrics() -> None:
	df = _predictions_sample()
	weighted = assign_portfolio_weights(df, quantile=1 / 3)
	daily = compute_daily_strategy_returns(weighted)
	summary = summarize_performance(daily)

	assert set(summary.keys()) == {
		"annual_return",
		"annual_volatility",
		"sharpe",
		"max_drawdown",
		"hit_rate",
	}


def test_assign_portfolio_weights_rejects_missing_columns() -> None:
	with pytest.raises(ValueError, match="Missing required columns"):
		assign_portfolio_weights(pd.DataFrame({"date": ["2024-01-01"]}))


def test_compute_daily_strategy_returns_sanitizes_non_finite_values() -> None:
	df = pd.DataFrame(
		{
			"date": ["2024-01-01", "2024-01-01", "2024-01-02"],
			"weight": [0.5, np.inf, -0.5],
			"target_fwd_return": [0.02, np.nan, -np.inf],
		}
	)

	daily = compute_daily_strategy_returns(df)
	assert np.isfinite(daily["strategy_return"]).all()


def test_summarize_performance_rejects_invalid_trading_days() -> None:
	with pytest.raises(ValueError, match="trading_days"):
		summarize_performance(pd.DataFrame({"strategy_return": [0.01]}), trading_days=0)


def test_apply_risk_constraints_caps_weights_and_gross() -> None:
	df = pd.DataFrame(
		{
			"date": ["2024-01-01"] * 4,
			"ticker": ["A", "B", "C", "D"],
			"weight": [0.8, 0.8, -0.8, -0.8],
		}
	)
	out = apply_risk_constraints(df, max_abs_weight=0.25, max_gross_leverage=1.0)
	assert out["weight"].abs().max() <= 0.25 + 1e-12
	assert out.groupby("date")["weight"].apply(lambda s: s.abs().sum()).iloc[0] <= 1.0 + 1e-12


def test_compute_daily_strategy_returns_realistic_includes_costs() -> None:
	df = pd.DataFrame(
		{
			"date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
			"ticker": ["A", "B", "A", "B"],
			"weight": [0.5, -0.5, 0.2, -0.2],
			"target_fwd_return": [0.01, -0.01, 0.02, -0.02],
		}
	)

	out = compute_daily_strategy_returns_realistic(df, transaction_cost_bps=10.0, slippage_bps=5.0)
	assert set(["date", "gross_return", "turnover", "cost", "strategy_return"]).issubset(out.columns)
	assert (out["cost"] >= 0.0).all()
