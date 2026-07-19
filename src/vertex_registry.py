"""Vertex AI Model Registry — governed catalog for the champion/challenger models.

Only a KB-sized model artifact (``model.bst``) leaves Snowflake for GCS and gets
registered here with metrics, feature list, and the accept/reject decision as
labels/description. The training *data* never leaves the warehouse — a clean
governance story: catalog the model on GCP, keep the PHI-shaped data in Snowflake.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

# Vertex prebuilt XGBoost serving container (matches pinned xgboost 2.1.x).
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


def _label(v) -> str:
    """Vertex labels: lowercase, [a-z0-9_-], <=63 chars."""
    return re.sub(r"[^a-z0-9_-]", "-", str(v).lower())[:63]


def export_booster(model, out_dir: str | Path) -> Path:
    """Save an XGBRegressor's booster as ``model.bst`` (name the prebuilt container expects).

    Clears feature names: the model trains on a DataFrame (named columns), but the prebuilt
    serving container sends bare positional float vectors — with names set, xgboost rejects
    the request ("training data did not have the following fields"). Positional is what the
    endpoint/registry contract uses, so we strip names before saving.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.bst"
    booster = model.get_booster()
    booster.feature_names = None
    booster.save_model(str(path))
    return path


def upload_dir_to_gcs(local_dir: str | Path, bucket: str, prefix: str) -> str:
    """Upload every file in local_dir to gs://bucket/prefix; return the gs:// dir URI."""
    from google.cloud import storage
    client = storage.Client()
    b = client.bucket(bucket)
    for f in Path(local_dir).iterdir():
        if f.is_file():
            b.blob(f"{prefix}/{f.name}").upload_from_filename(str(f))
    return f"gs://{bucket}/{prefix}"


def register_model(
    model,
    *,
    display_name: str,
    bucket: str,
    project: str,
    location: str = "us-central1",
    metrics: dict | None = None,
    feature_names: list[str] | None = None,
    decision: str = "",
    parent_model: str | None = None,
    version_aliases: list[str] | None = None,
):
    """Export → upload → register in Vertex Model Registry. Returns the aiplatform.Model.

    Requires a live Vertex project. ``parent_model`` (a model resource name) registers
    this as a new *version* of an existing model so champion/challenger share one entry.
    """
    if not project:
        raise ValueError("register_model requires a Vertex project.")
    metrics = metrics or {}
    version = _label(display_name)
    with tempfile.TemporaryDirectory() as tmp:
        export_booster(model, tmp)
        gcs_dir = upload_dir_to_gcs(tmp, bucket, f"models/{version}")

    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    desc = json.dumps({"metrics": metrics, "features": feature_names, "decision": decision})
    labels = {"decision": _label(decision or "none"),
              "framework": "xgboost", "project": "carecost-fusion"}
    model_obj = aiplatform.Model.upload(
        display_name="carecost-next90d-cost" if parent_model is None else None,
        parent_model=parent_model,
        artifact_uri=gcs_dir,
        serving_container_image_uri=SERVING_IMAGE,
        version_aliases=version_aliases or [version],
        version_description=desc,
        labels=labels,
    )
    return model_obj


def console_url(project: str, location: str = "us-central1") -> str:
    return f"https://console.cloud.google.com/vertex-ai/models?project={project}"


if __name__ == "__main__":
    # Smoke: export a name-stripped booster (positional serving) and confirm the file.
    import numpy as np
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=5, max_depth=2).fit(np.random.rand(50, 3), np.random.rand(50))
    with tempfile.TemporaryDirectory() as d:
        p = export_booster(m, d)
        assert p.exists() and p.stat().st_size > 0
    print("export_booster ok")
