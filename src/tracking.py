"""Experiment tracking.

Uses MLflow when it is installed. If MLflow is missing (for example in a minimal
test environment) the same calls are written to reports/runs/<run_id>.json so the
pipeline still runs and nothing is lost.
"""

import json
import uuid
from pathlib import Path

from src.common import PROJECT_ROOT, project_path

try:
    import mlflow
except ImportError:          # pragma: no cover - depends on the environment
    mlflow = None


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg["tracking"]
        self.backend = "mlflow" if mlflow else "json"
        if mlflow:
            uri = self.cfg["tracking_uri"]
            if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
                uri = "sqlite:///" + str(PROJECT_ROOT / uri[len("sqlite:///"):])
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.cfg["experiment_name"])
        self.json_dir = project_path(cfg["paths"]["reports_dir"]) / "runs"

    # --- JSON fallback helpers
    def _json_path(self, run_id):
        return self.json_dir / f"{run_id}.json"

    def _json_update(self, run_id, key, values):
        path = self._json_path(run_id)
        data = json.loads(path.read_text()) if path.exists() else {"run_id": run_id}
        data.setdefault(key, {})
        data[key].update(values) if isinstance(values, dict) else data[key].update({str(values): True})
        self.json_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))

    # --- public API
    def start(self, run_name, tags=None):
        """Create a run and return its id."""
        if self.backend == "mlflow":
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                return run.info.run_id
        run_id = uuid.uuid4().hex[:12]
        self._json_update(run_id, "tags", {"run_name": run_name, **(tags or {})})
        return run_id

    def log_params(self, run_id, params):
        if self.backend == "mlflow":
            with mlflow.start_run(run_id=run_id):
                mlflow.log_params(params)
        else:
            self._json_update(run_id, "params", params)

    def log_metrics(self, run_id, metrics):
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        if self.backend == "mlflow":
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(clean)
        else:
            self._json_update(run_id, "metrics", clean)

    def log_artifact(self, run_id, path):
        if self.backend == "mlflow":
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(path))
        else:
            self._json_update(run_id, "artifacts", {Path(path).name: str(path)})
