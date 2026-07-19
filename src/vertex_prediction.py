"""Vertex AI Online Prediction + Explainable AI for the champion model.

Deploys the champion XGBoost to a Vertex Endpoint with **sampled-Shapley feature
attributions** (Explainable AI). Serving is managed by Vertex; the model artifact
is the KB booster in GCS. Includes explicit undeploy/delete so nothing bills once
the demo is recorded (the endpoint node is the only ~$/hr resource in the project).

⚠️ Always call ``teardown_endpoint`` when done. Logged in TEARDOWN.md.
"""
from __future__ import annotations

import tempfile

from vertex_registry import export_booster, upload_dir_to_gcs, SERVING_IMAGE

MACHINE_TYPE = "n1-standard-2"


def deploy_champion(model, feature_names: list[str], *, project: str, bucket: str,
                    location: str = "us-central1", display_name: str = "carecost-champion"):
    """Register the model WITH an Explainable AI spec, create an endpoint, deploy.

    Returns the deployed aiplatform.Endpoint. Blocks until traffic is live (~10 min).
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform.explain import ExplanationMetadata, ExplanationParameters
    aiplatform.init(project=project, location=location)

    with tempfile.TemporaryDirectory() as tmp:
        export_booster(model, tmp)
        gcs_dir = upload_dir_to_gcs(tmp, bucket, "models/champion-serving")

    # Sampled Shapley attributions over the flat feature vector (BAG_OF_FEATURES).
    explain_metadata = ExplanationMetadata(
        inputs={"features": ExplanationMetadata.InputMetadata(
            index_feature_mapping=feature_names, encoding="BAG_OF_FEATURES")},
        outputs={"cost": ExplanationMetadata.OutputMetadata()},
    )
    explain_params = ExplanationParameters(sampled_shapley_attribution={"path_count": 10})

    model_obj = aiplatform.Model.upload(
        display_name=display_name, artifact_uri=gcs_dir,
        serving_container_image_uri=SERVING_IMAGE,
        explanation_metadata=explain_metadata, explanation_parameters=explain_params,
        labels={"project": "carecost-fusion", "role": "champion-endpoint"},
    )
    endpoint = model_obj.deploy(
        machine_type=MACHINE_TYPE, min_replica_count=1, max_replica_count=1,
        traffic_percentage=100, sync=True,
    )
    return endpoint


def predict(endpoint, instances: list[list[float]]) -> list[float]:
    """Online prediction. instances = list of feature vectors (baseline feature order)."""
    return list(endpoint.predict(instances=instances).predictions)


def explain(endpoint, instances: list[list[float]], feature_names: list[str]) -> list[dict]:
    """Return per-instance {feature: attribution} from Vertex Explainable AI.

    With BAG_OF_FEATURES + index_feature_mapping, ``feature_attributions`` comes back
    keyed by feature name, each value a single-element repeated list — so we unpack the
    scalar per feature. ``feature_names`` is accepted for API symmetry.
    """
    resp = endpoint.explain(instances=instances)
    out = []
    for expl in resp.explanations:
        fa = dict(expl.attributions[0].feature_attributions)
        d = {}
        for k, v in fa.items():
            vals = list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) else [v]
            d[k] = round(float(vals[0]) if len(vals) == 1 else sum(map(float, vals)), 2)
        out.append(d)
    return out


def teardown_endpoint(endpoint) -> None:
    """Undeploy all models and delete the endpoint — stops all billing."""
    endpoint.undeploy_all(sync=True)
    endpoint.delete(sync=True)
    print("endpoint undeployed + deleted")


def find_endpoint(display_name: str, project: str, location: str = "us-central1"):
    """Return the first endpoint matching display_name, or None."""
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    eps = aiplatform.Endpoint.list(filter=f'display_name="{display_name}"')
    return eps[0] if eps else None
