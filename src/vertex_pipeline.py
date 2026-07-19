"""Vertex AI Pipeline (KFP) — orchestrates the CareCost Fusion flow as a DAG.

    gen_features → baseline_segment → gemini_propose → challengers_gate → register_champion

Each step is a lightweight KFP component that installs the project wheel from GCS at
runtime (the versioned-package pattern) plus its PyPI deps, then calls the same
`src` modules the app uses. Artifacts (parquet features, JSON metrics, the booster)
flow between steps.

KFP components are hermetic — they can't close over module-level variables — so the
GCS-wheel bootstrap is inlined in every component and the wheel URI is threaded as a
pipeline parameter.

Note: pipeline steps run on Google compute, whose IPs aren't in the Snowflake network
allowlist, so this pipeline uses the deterministic synthetic (pandas) feature path.
The live Snowflake source is exercised by the app / notebook. Compile with
``compile_pipeline()``; submit with ``run_pipeline()``.
"""
# NB: no `from __future__ import annotations` — KFP needs real annotation objects.
import os

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model

# Configure via env: GOOGLE_CLOUD_PROJECT and CARECOST_BUCKET (defaults derive from project).
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
BUCKET = os.environ.get("CARECOST_BUCKET", f"{PROJECT}-carecost")
WHEEL_URI = f"gs://{BUCKET}/pkg/carecost_fusion-0.1.0-py3-none-any.whl"
IMAGE = "python:3.11"
CORE = ["google-cloud-storage", "pandas", "numpy", "pyarrow", "scikit-learn", "xgboost==2.1.4", "pydantic"]


@component(base_image=IMAGE, packages_to_install=CORE)
def gen_features(wheel_uri: str, member_count: int, seed: int, features: Output[Dataset]):
    import subprocess as _sp, sys as _sys
    from google.cloud import storage as _storage
    _b, _, _blob = wheel_uri[5:].partition("/")
    _path = "/tmp/" + _blob.split("/")[-1]  # keep the real wheel filename (pip validates it)
    _storage.Client().bucket(_b).blob(_blob).download_to_filename(_path)
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", _path])
    from generate_claims import generate_claims
    from features import build_member_features, monthly_index_dates
    df = build_member_features(generate_claims(member_count, 15, "2026-01-01", seed),
                               monthly_index_dates("2026-01-01", 10))
    df.to_parquet(features.path)


@component(base_image=IMAGE, packages_to_install=CORE)
def baseline_segment(wheel_uri: str, features: Input[Dataset], model: Output[Model],
                     metrics: Output[Dataset], segment: Output[Dataset]):
    import subprocess as _sp, sys as _sys
    from google.cloud import storage as _storage
    _b, _, _blob = wheel_uri[5:].partition("/")
    _path = "/tmp/" + _blob.split("/")[-1]  # keep the real wheel filename (pip validates it)
    _storage.Client().bucket(_b).blob(_blob).download_to_filename(_path)
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", _path])
    import os, json
    import pandas as pd
    import pipeline as P
    from features import BASELINE_MODEL_FEATURES
    df = pd.read_parquet(features.path)
    base = P.train_baseline(df)
    seg = P.residual_segment(base)
    os.makedirs(model.path, exist_ok=True)
    b = base.model.get_booster(); b.feature_names = None
    b.save_model(f"{model.path}/model.bst")
    json.dump({"base": base.metrics, "median": base.median_metrics, "high_cost": base.high_cost,
               "features": BASELINE_MODEL_FEATURES}, open(metrics.path, "w"))
    json.dump(seg, open(segment.path, "w"))


@component(base_image=IMAGE, packages_to_install=CORE + ["google-genai"])
def gemini_propose(wheel_uri: str, segment: Input[Dataset], features: Input[Dataset], project: str,
                   location: str, gemini_model: str, accepted: Output[Dataset]):
    import subprocess as _sp, sys as _sys
    from google.cloud import storage as _storage
    _b, _, _blob = wheel_uri[5:].partition("/")
    _path = "/tmp/" + _blob.split("/")[-1]  # keep the real wheel filename (pip validates it)
    _storage.Client().bucket(_b).blob(_blob).download_to_filename(_path)
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", _path])
    import json
    import pandas as pd
    from gemini_hypothesis import propose_features
    from feature_catalog import validate_candidates
    seg = json.load(open(segment.path))
    cols = set(pd.read_parquet(features.path).columns)
    resp = propose_features(seg, gemini_model, project, location)
    acc, rej = validate_candidates([c.feature_name for c in resp.candidates], cols)
    hyp = {c.feature_name: c.hypothesis for c in resp.candidates}
    json.dump({"accepted": acc, "hypotheses": hyp}, open(accepted.path, "w"))


@component(base_image=IMAGE, packages_to_install=CORE)
def challengers_gate(wheel_uri: str, features: Input[Dataset], metrics: Input[Dataset],
                     accepted: Input[Dataset], champion: Output[Model], results: Output[Dataset]):
    import subprocess as _sp, sys as _sys
    from google.cloud import storage as _storage
    _b, _, _blob = wheel_uri[5:].partition("/")
    _path = "/tmp/" + _blob.split("/")[-1]  # keep the real wheel filename (pip validates it)
    _storage.Client().bucket(_b).blob(_blob).download_to_filename(_path)
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", _path])
    import os, json
    import numpy as np, pandas as pd
    import pipeline as P
    from features import BASELINE_MODEL_FEATURES
    from modeling import temporal_split, make_model, TARGET
    from feature_catalog import materialize
    df = pd.read_parquet(features.path)
    m = json.load(open(metrics.path)); acc = json.load(open(accepted.path))

    class B:  # minimal Baseline stand-in for challenger_step
        pass
    base = B(); base.metrics = m["base"]; base.high_cost = m["high_cost"]
    res = P.challenger_step(df, base, acc["accepted"], hyp_by_name=acc["hypotheses"])
    winners = [r for r in res if r["decision"] == "ACCEPT"]
    champ_feat = winners[0]["feature_name"] if winners else "baseline"
    json.dump({"champion_feature": champ_feat,
               "decisions": [{k: r[k] for k in ("feature_name", "mae_improvement_pct", "decision", "decision_reason")}
                             for r in res]}, open(results.path, "w"))

    feats = BASELINE_MODEL_FEATURES + ([champ_feat] if champ_feat != "baseline" else [])
    fdf = df.copy()
    if champ_feat != "baseline":
        fdf[champ_feat] = materialize(fdf, champ_feat)
    split = temporal_split(fdf)
    model = make_model(P.DEFAULT_MODEL_CFG)
    model.fit(split.train[feats], np.log1p(split.train[TARGET].to_numpy()))
    os.makedirs(champion.path, exist_ok=True)
    b = model.get_booster(); b.feature_names = None
    b.save_model(f"{champion.path}/model.bst")


@component(base_image=IMAGE, packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"])
def register_champion(wheel_uri: str, champion: Input[Model], results: Input[Dataset],
                      project: str, location: str) -> str:
    import subprocess as _sp, sys as _sys
    from google.cloud import storage as _storage
    _b, _, _blob = wheel_uri[5:].partition("/")
    _path = "/tmp/" + _blob.split("/")[-1]  # keep the real wheel filename (pip validates it)
    _storage.Client().bucket(_b).blob(_blob).download_to_filename(_path)
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", _path])
    import json
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    info = json.load(open(results.path))
    model = aiplatform.Model.upload(
        display_name="carecost-pipeline-champion",
        artifact_uri=champion.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest",
        version_description=json.dumps(info),
        labels={"project": "carecost-fusion", "source": "vertex-pipeline",
                "champion": info["champion_feature"].lower().replace("_", "-")},
    )
    return model.resource_name


@dsl.pipeline(name="carecost-fusion-pipeline",
              description="Synthetic features → XGBoost baseline → Gemini hypotheses → holdout gate → Model Registry")
def carecost_pipeline(wheel_uri: str = WHEEL_URI, member_count: int = 2000, seed: int = 42,
                      project: str = PROJECT, location: str = "us-central1",
                      gemini_model: str = "gemini-2.5-flash"):
    f = gen_features(wheel_uri=wheel_uri, member_count=member_count, seed=seed)
    b = baseline_segment(wheel_uri=wheel_uri, features=f.outputs["features"])
    g = gemini_propose(wheel_uri=wheel_uri, segment=b.outputs["segment"], features=f.outputs["features"],
                       project=project, location=location, gemini_model=gemini_model)
    c = challengers_gate(wheel_uri=wheel_uri, features=f.outputs["features"],
                         metrics=b.outputs["metrics"], accepted=g.outputs["accepted"])
    register_champion(wheel_uri=wheel_uri, champion=c.outputs["champion"],
                      results=c.outputs["results"], project=project, location=location)


def compile_pipeline(path: str = "pipeline.json") -> str:
    from kfp import compiler
    compiler.Compiler().compile(carecost_pipeline, path)
    return path


def run_pipeline(project: str = PROJECT, location: str = "us-central1", sync: bool = False):
    from google.cloud import aiplatform
    compile_pipeline("pipeline.json")
    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name="carecost-fusion-pipeline",
        template_path="pipeline.json",
        pipeline_root=f"gs://{BUCKET}/pipeline_root",
        enable_caching=True,
    )
    job.submit()
    print("pipeline submitted:", job.resource_name)
    print("console:", job._dashboard_uri())
    if sync:
        job.wait()
    return job


if __name__ == "__main__":
    print("compiled ->", compile_pipeline())
