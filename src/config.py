from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Settings:
	"""Application settings loaded from environment variables."""

	db_url: str = field(
		default_factory=lambda: os.getenv(
			"DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/quant_alpha"
		)
	)
	universe: str = field(default_factory=lambda: os.getenv("UNIVERSE", "SP500"))
	start_date: str = field(default_factory=lambda: os.getenv("START_DATE", "2015-01-01"))
	end_date: str = field(default_factory=lambda: os.getenv("END_DATE", "2025-12-31"))
	prediction_horizon_days: int = field(
		default_factory=lambda: int(os.getenv("PREDICTION_HORIZON_DAYS", "5"))
	)
	top_bottom_quantile: float = field(
		default_factory=lambda: float(os.getenv("TOP_BOTTOM_QUANTILE", "0.1"))
	)
	random_state: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))

	def __post_init__(self) -> None:
		if self.prediction_horizon_days <= 0:
			raise ValueError("PREDICTION_HORIZON_DAYS must be positive.")
		if not (0.0 < self.top_bottom_quantile < 0.5):
			raise ValueError("TOP_BOTTOM_QUANTILE must be in (0, 0.5).")
		if self.random_state < 0:
			raise ValueError("RANDOM_STATE must be non-negative.")
		# Validate date format and ordering at startup.
		start = datetime.strptime(self.start_date, "%Y-%m-%d")
		end = datetime.strptime(self.end_date, "%Y-%m-%d")
		if end <= start:
			raise ValueError("END_DATE must be later than START_DATE.")


def get_settings() -> Settings:
	"""Return immutable runtime settings."""

	return Settings()
