"""Vertex AI Gen AI Evaluation — score Gemini's feature hypotheses.

The project's thesis is "the LLM's own confidence doesn't count." So each proposed
hypothesis is independently scored by the **Vertex Gen AI Evaluation Service** with a
custom pointwise metric. Requires a live Vertex project — raises on failure.
"""
from __future__ import annotations

import json


def evaluate_hypotheses(candidates: list, residual_summary: dict, project: str,
                        location: str = "us-central1") -> dict:
    """Score each hypothesis (plausibility, [0,1]) via the Gen AI Evaluation Service."""
    if not project:
        raise ValueError("evaluate_hypotheses requires a Vertex project.")
    import pandas as pd
    import vertexai
    from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
    vertexai.init(project=project, location=location)

    # A hypothesis extrapolates beyond the evidence, so strict GROUNDEDNESS is always
    # ~0 — we grade plausibility + relevance instead, which differentiates hypotheses.
    metric = PointwiseMetric(
        metric="hypothesis_plausibility",
        metric_prompt_template=PointwiseMetricPromptTemplate(
            criteria={
                "plausibility": "The hypothesis is a plausible mechanism for the underprediction described in the evidence.",
                "relevance": "The hypothesis is relevant to the specific error segment in the evidence.",
            },
            rating_rubric={
                "5": "Highly plausible mechanism and directly relevant to the segment.",
                "3": "Somewhat plausible or partially relevant.",
                "1": "Not plausible or not relevant to the evidence.",
            },
            input_variables=["prompt", "response"],
        ),
    )
    context = json.dumps(residual_summary)
    prompt = (f"Source evidence — aggregate residual summary of a cost model's errors:\n{context}\n\n"
              "Assess the proposed feature hypothesis as an explanation for the underprediction.")
    df = pd.DataFrame({
        "prompt": [prompt] * len(candidates),
        "response": [f"Feature {c.feature_name}: {c.hypothesis}" for c in candidates],
    })
    table = EvalTask(dataset=df, metrics=[metric]).evaluate().metrics_table
    col = next(c for c in table.columns if c.endswith("/score"))
    scores = {c.feature_name: _norm(table[col].iloc[i]) for i, c in enumerate(candidates)}
    scores["source"] = "gen_ai_evaluation"
    return scores


def _norm(v) -> float:
    """Map the 1–5 rubric score linearly to [0,1]: 1→0.0 (worst), 5→1.0 (best)."""
    return round((float(v) - 1.0) / 4.0, 3)
