"""Vertex AI Experiments — run tracking for the CareCost Fusion flow.

The cross-cloud ledger: records runs (baseline, challengers) with params + metrics in
the Vertex console. Requires a live Vertex project — raises if unavailable (no fallback).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _slug(name: str) -> str:
    """Vertex run/experiment names: lowercase alphanumeric + hyphens, <=128."""
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")[:128]


@dataclass
class ExperimentLogger:
    experiment_name: str
    project: str
    location: str = "us-central1"
    staging_bucket: str = ""
    runs: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.project:
            raise ValueError("ExperimentLogger requires a Vertex project.")
        self.experiment_name = _slug(self.experiment_name)
        from google.cloud import aiplatform
        aiplatform.init(
            project=self.project, location=self.location, experiment=self.experiment_name,
            staging_bucket=self.staging_bucket or None,
            # No managed TensorBoard: it's being sunset; the Experiments UI shows metrics.
            experiment_tensorboard=False,
        )
        self._ap = aiplatform
        print(f"[vertex] experiment '{self.experiment_name}' ready in {self.location}")

    def log_run(self, run_name: str, params: dict, metrics: dict) -> None:
        """Log one run to Vertex AI Experiments."""
        clean_metrics = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        self.runs.append({"run": run_name, "params": dict(params), "metrics": clean_metrics})
        with self._ap.start_run(_slug(run_name), resume=False):
            self._ap.log_params({k: str(v) for k, v in params.items() if v is not None})
            self._ap.log_metrics(clean_metrics)

    def console_url(self) -> str:
        return (f"https://console.cloud.google.com/vertex-ai/experiments/locations/"
                f"{self.location}/experiments/{self.experiment_name}/runs?project={self.project}")
