from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
	"""Configure project logging once with a concise, timestamped format."""

	root = logging.getLogger()
	if not root.handlers:
		logging.basicConfig(
			level=level,
			format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		)
	else:
		root.setLevel(level)
