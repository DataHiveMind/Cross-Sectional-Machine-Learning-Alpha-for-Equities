from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.models_db import Base


def get_engine(db_url: str | None = None, *, echo: bool = False) -> Engine:
	"""Create a SQLAlchemy engine for PostgreSQL or SQLite."""

	url = db_url or get_settings().db_url
	return create_engine(url, echo=echo, future=True)


def get_session_factory(engine: Engine) -> sessionmaker:
	"""Return configured SQLAlchemy sessionmaker."""

	return sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker) -> Iterator[Session]:
	"""Provide a transactional scope around a series of operations."""

	session = session_factory()
	try:
		yield session
		session.commit()
	except Exception:
		session.rollback()
		raise
	finally:
		session.close()


def create_all_tables(engine: Engine) -> None:
	"""Create all ORM-managed database tables."""

	Base.metadata.create_all(engine)


def drop_all_tables(engine: Engine) -> None:
	"""Drop all ORM-managed database tables."""

	Base.metadata.drop_all(engine)
