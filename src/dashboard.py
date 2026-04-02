from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def build_concise_dashboard(
	daily_returns: pd.DataFrame,
	perf: Mapping[str, float],
	ic_by_day: pd.Series | None = None,
	output_path: str | Path | None = None,
):
	"""Create a compact 2x2 dashboard figure and optionally save it."""

	df = daily_returns.copy()
	df["date"] = pd.to_datetime(df["date"])
	df = df.sort_values("date").reset_index(drop=True)

	equity = (1.0 + df["strategy_return"]).cumprod()
	running_peak = equity.cummax()
	drawdown = (equity / running_peak) - 1.0

	fig, axes = plt.subplots(2, 2, figsize=(14, 9))

	axes[0, 0].plot(df["date"], equity, color="navy")
	axes[0, 0].set_title("Equity Curve")
	axes[0, 0].grid(alpha=0.3)

	axes[0, 1].plot(df["date"], drawdown, color="firebrick")
	axes[0, 1].set_title("Drawdown")
	axes[0, 1].grid(alpha=0.3)

	axes[1, 0].hist(df["strategy_return"], bins=40, color="steelblue", alpha=0.85)
	axes[1, 0].set_title("Daily Return Distribution")
	axes[1, 0].grid(alpha=0.3)

	axes[1, 1].axis("off")
	kpi_lines = [
		"Key KPIs",
		f"Annual Return: {perf.get('annual_return', 0.0):.3f}",
		f"Annual Volatility: {perf.get('annual_volatility', 0.0):.3f}",
		f"Sharpe: {perf.get('sharpe', 0.0):.3f}",
		f"Max Drawdown: {perf.get('max_drawdown', 0.0):.3f}",
		f"Hit Rate: {perf.get('hit_rate', 0.0):.3f}",
	]
	if ic_by_day is not None and not ic_by_day.empty:
		kpi_lines.append(f"Mean IC: {float(ic_by_day.mean()):.3f}")
	axes[1, 1].text(0.02, 0.98, "\n".join(kpi_lines), va="top", ha="left", fontsize=11)

	fig.suptitle("Concise Strategy Dashboard", fontsize=14)
	fig.tight_layout(rect=[0, 0, 1, 0.96])

	if output_path is not None:
		out = Path(output_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out, dpi=180, bbox_inches="tight")

	return fig
