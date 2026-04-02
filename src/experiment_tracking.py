from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence


logger = logging.getLogger(__name__)


def log_mlflow_run(
	experiment_name: str,
	run_name: str,
	params: Mapping[str, str | int | float | bool],
	metrics: Mapping[str, float],
	artifact_paths: Sequence[str | Path] | None = None,
) -> bool:
	"""Log parameters, metrics, and artifacts to MLflow.

	Returns True when logging succeeds, False when MLflow is unavailable.
	"""

	try:
		import mlflow
	except Exception:
		logger.warning("MLflow is not available; skipping experiment tracking.")
		return False

	mlflow.set_experiment(experiment_name)
	with mlflow.start_run(run_name=run_name):
		for k, v in params.items():
			mlflow.log_param(k, v)
		for k, v in metrics.items():
			mlflow.log_metric(k, float(v))
		for artifact in artifact_paths or []:
			p = Path(artifact)
			if p.exists():
				mlflow.log_artifact(str(p))
			else:
				logger.warning("MLflow artifact path not found: %s", p)

	logger.info("MLflow run logged successfully: %s", run_name)
	return True
