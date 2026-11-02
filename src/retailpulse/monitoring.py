"""Run history + optional MLflow tracking (never a hard dependency)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def log_run(artifact_dir: Path, params: dict, metrics: dict, use_mlflow: bool | None = None) -> dict:
    """Append a run record to run_history.jsonl and (optionally) mirror it to MLflow."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "params": params, "metrics": metrics}
    with open(artifact_dir / "run_history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    if use_mlflow is None:
        use_mlflow = os.getenv("RETAILPULSE_MLFLOW", "0") == "1"
    if use_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{artifact_dir / 'mlruns'}"))
            mlflow.set_experiment("retailpulse")
            with mlflow.start_run():
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))})
            record["mlflow"] = True
        except Exception as exc:  # tracking must never break the pipeline
            record["mlflow_error"] = str(exc)
    return record


def read_history(artifact_dir: Path) -> list[dict]:
    path = Path(artifact_dir) / "run_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
