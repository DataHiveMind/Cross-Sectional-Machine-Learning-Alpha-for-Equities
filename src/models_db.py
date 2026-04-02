from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, Index, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
	"""Declarative base for ORM models."""


class PriceBar(Base):
	"""Daily OHLCV data for a single equity ticker."""

	__tablename__ = "price_bars"

	id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
	date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
	ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

	open: Mapped[float] = mapped_column(Float, nullable=False)
	high: Mapped[float] = mapped_column(Float, nullable=False)
	low: Mapped[float] = mapped_column(Float, nullable=False)
	close: Mapped[float] = mapped_column(Float, nullable=False)
	volume: Mapped[float] = mapped_column(Float, nullable=False)

	__table_args__ = (
		UniqueConstraint("date", "ticker", name="uq_price_bars_date_ticker"),
		Index("ix_price_bars_ticker_date", "ticker", "date"),
	)
