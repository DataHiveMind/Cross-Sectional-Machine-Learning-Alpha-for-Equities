from __future__ import annotations

import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def assign_portfolio_weights(
	prediction_df: pd.DataFrame,
	quantile: float = 0.1,
	prediction_col: str = "prediction",
) -> pd.DataFrame:
	"""Create daily dollar-neutral long/short weights from predictions."""

	if not (0.0 < quantile < 0.5):
		raise ValueError("quantile must be in (0, 0.5).")

	if prediction_df.empty:
		logger.warning("assign_portfolio_weights received empty prediction dataframe.")
		out = prediction_df.copy()
		out["weight"] = pd.Series(dtype="float64")
		out["side"] = pd.Series(dtype="string")
		return out

	required = {"date", prediction_col}
	missing = required - set(prediction_df.columns)
	if missing:
		raise ValueError(f"Missing required columns for portfolio assignment: {sorted(missing)}")

	df = prediction_df.copy()
	df["date"] = pd.to_datetime(df["date"])
	df[prediction_col] = pd.to_numeric(df[prediction_col], errors="coerce")
	df[prediction_col] = df[prediction_col].replace([np.inf, -np.inf], np.nan)

	valid_count = df[prediction_col].notna().groupby(df["date"]).transform("sum")
	k = np.floor(valid_count * quantile).astype("int64").clip(lower=1)
	eligible = (valid_count >= 2) & ((2 * k) <= valid_count)

	if not bool(eligible.any()):
		logger.warning("No eligible rows for weight assignment after quantile constraints.")
		return pd.DataFrame(columns=[*prediction_df.columns, "weight", "side"])

	df = df.loc[eligible].copy()
	k = k.loc[eligible]

	rank_desc = df.groupby("date")[prediction_col].rank(method="first", ascending=False)
	rank_asc = df.groupby("date")[prediction_col].rank(method="first", ascending=True)

	long_mask = df[prediction_col].notna() & (rank_desc <= k)
	short_mask = df[prediction_col].notna() & (rank_asc <= k)

	df["weight"] = 0.0
	df.loc[long_mask, "weight"] = 1.0 / k.loc[long_mask].astype(float)
	df.loc[short_mask, "weight"] = -1.0 / k.loc[short_mask].astype(float)

	df["side"] = "flat"
	df.loc[long_mask, "side"] = "long"
	df.loc[short_mask, "side"] = "short"

	out = df.sort_values(["date", prediction_col], ascending=[True, False]).reset_index(drop=True)
	logger.info(
		"Assigned portfolio weights for %d rows across %d dates.",
		len(out),
		out["date"].nunique(),
	)
	return out


def compute_daily_strategy_returns(
	weighted_df: pd.DataFrame, realized_return_col: str = "target_fwd_return"
) -> pd.DataFrame:
	"""Aggregate weighted realized returns to daily portfolio PnL."""

	if weighted_df.empty:
		logger.warning("compute_daily_strategy_returns received empty weighted dataframe.")
		return pd.DataFrame(columns=["date", "strategy_return"])

	required = {"date", "weight", realized_return_col}
	missing = required - set(weighted_df.columns)
	if missing:
		raise ValueError(f"Missing required columns for return calc: {sorted(missing)}")

	df = weighted_df.copy()
	df["weight"] = pd.to_numeric(df["weight"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
	df[realized_return_col] = (
		pd.to_numeric(df[realized_return_col], errors="coerce")
		.replace([np.inf, -np.inf], np.nan)
		.fillna(0.0)
	)
	df["contribution"] = df["weight"] * df[realized_return_col]
	daily = df.groupby("date", as_index=False)["contribution"].sum()
	logger.info("Computed daily strategy returns for %d dates.", len(daily))
	return daily.rename(columns={"contribution": "strategy_return"}).sort_values("date")


def summarize_performance(daily_returns: pd.DataFrame, trading_days: int = 252) -> dict[str, float]:
	"""Return common long/short performance diagnostics."""

	if daily_returns.empty:
		logger.warning("summarize_performance received empty daily_returns dataframe.")
		return {
			"annual_return": 0.0,
			"annual_volatility": 0.0,
			"sharpe": 0.0,
			"max_drawdown": 0.0,
			"hit_rate": 0.0,
		}

	if "strategy_return" not in daily_returns.columns:
		raise ValueError("daily_returns must contain 'strategy_return'.")
	if trading_days <= 0:
		raise ValueError("trading_days must be positive.")

	r = pd.to_numeric(daily_returns["strategy_return"], errors="coerce")
	r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
	mean = float(r.mean())
	std = float(r.std(ddof=1)) if len(r) > 1 else 0.0

	annual_return = mean * trading_days
	annual_vol = std * np.sqrt(trading_days) if std > 0 else 0.0
	sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

	equity_curve = (1.0 + r).cumprod()
	running_peak = equity_curve.cummax()
	drawdown = (equity_curve / running_peak) - 1.0

	metrics = {
		"annual_return": float(annual_return),
		"annual_volatility": float(annual_vol),
		"sharpe": float(sharpe),
		"max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
		"hit_rate": float((r > 0).mean()),
	}
	logger.info(
		"Performance summary computed: annual_return=%.4f, sharpe=%.4f, max_drawdown=%.4f.",
		metrics["annual_return"],
		metrics["sharpe"],
		metrics["max_drawdown"],
	)
	return metrics


def apply_risk_constraints(
	weighted_df: pd.DataFrame,
	max_abs_weight: float = 0.10,
	max_gross_leverage: float = 1.0,
) -> pd.DataFrame:
	"""Apply simple daily risk controls to portfolio weights.

	- Caps absolute position weight by `max_abs_weight`.
	- Scales weights down to satisfy gross leverage limit.
	"""

	if weighted_df.empty:
		return weighted_df.copy()
	if max_abs_weight <= 0:
		raise ValueError("max_abs_weight must be positive.")
	if max_gross_leverage <= 0:
		raise ValueError("max_gross_leverage must be positive.")

	df = weighted_df.copy()
	df["date"] = pd.to_datetime(df["date"])
	df["weight"] = pd.to_numeric(df["weight"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
	df["weight"] = df["weight"].clip(lower=-max_abs_weight, upper=max_abs_weight)

	gross = df["weight"].abs().groupby(df["date"]).transform("sum")
	scale = np.where(gross > max_gross_leverage, max_gross_leverage / gross, 1.0)
	df["weight"] = df["weight"] * scale

	logger.info(
		"Applied risk constraints (max_abs_weight=%.3f, max_gross_leverage=%.3f).",
		max_abs_weight,
		max_gross_leverage,
	)
	return df


def compute_daily_strategy_returns_realistic(
	weighted_df: pd.DataFrame,
	realized_return_col: str = "target_fwd_return",
	transaction_cost_bps: float = 5.0,
	slippage_bps: float = 2.0,
) -> pd.DataFrame:
	"""Compute daily net strategy returns including turnover-based trading costs."""

	if transaction_cost_bps < 0 or slippage_bps < 0:
		raise ValueError("transaction_cost_bps and slippage_bps must be non-negative.")
	if weighted_df.empty:
		return pd.DataFrame(columns=["date", "gross_return", "turnover", "cost", "strategy_return"])

	required = {"date", "ticker", "weight", realized_return_col}
	missing = required - set(weighted_df.columns)
	if missing:
		raise ValueError(f"Missing required columns for realistic return calc: {sorted(missing)}")

	df = weighted_df.copy()
	df["date"] = pd.to_datetime(df["date"])
	df["ticker"] = df["ticker"].astype(str)
	df["weight"] = pd.to_numeric(df["weight"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
	df[realized_return_col] = (
		pd.to_numeric(df[realized_return_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
	)

	df["contribution"] = df["weight"] * df[realized_return_col]
	gross_daily = df.groupby("date", as_index=False)["contribution"].sum().rename(columns={"contribution": "gross_return"})

	weight_matrix = (
		df.pivot_table(index="date", columns="ticker", values="weight", aggfunc="last")
		.sort_index()
		.fillna(0.0)
	)
	turnover = weight_matrix.diff().abs().sum(axis=1).fillna(0.0) / 2.0
	turnover_df = turnover.rename("turnover").reset_index()

	daily = gross_daily.merge(turnover_df, on="date", how="left")
	bps = (transaction_cost_bps + slippage_bps) / 10_000.0
	daily["cost"] = daily["turnover"] * bps
	daily["strategy_return"] = daily["gross_return"] - daily["cost"]

	logger.info(
		"Computed realistic daily returns with tc_bps=%.2f and slippage_bps=%.2f.",
		transaction_cost_bps,
		slippage_bps,
	)
	return daily.sort_values("date").reset_index(drop=True)
