"""Gemini feature-hypothesis step on Vertex — as a tool-constrained agent call.

Why this is on Vertex and not Snowflake Cortex: this is GenAI *engineering*, not a
SQL inference call. Gemini doesn't free-text feature names — it **calls a
`propose_feature` tool** whose schema is bound to the whitelist (function calling),
its proposals are then **scored by the Vertex Gen AI Evaluation Service**
(``vertex_geneval``) and **logged to Vertex AI Experiments** alongside the model
runs. That lifecycle (tools + eval + lineage) is the Vertex-native surface Cortex
doesn't offer — so the LLM step belongs with the model lifecycle it feeds.

Offline (no creds / SDK): a deterministic stub keeps the pipeline runnable.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from feature_catalog import ALLOWED_FEATURE_NAMES

FEATURE_NAMES = Literal["COST_ACCELERATION", "PROVIDER_FRAGMENTATION", "ED_ACCELERATION", "INPATIENT_COST_SHARE"]


class FeatureCandidate(BaseModel):
    feature_name: FEATURE_NAMES
    hypothesis: str
    expected_error_segment: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class HypothesisResponse(BaseModel):
    action: Literal["PROPOSE", "STOP"]
    candidates: list[FeatureCandidate] = []
    stop_reason: Optional[str] = None


PROMPT = """\
You are assisting with an offline healthcare cost-model experiment using synthetic data.

Your role is to propose testable derived features. You do NOT decide whether a proposal
is successful — a separate XGBoost holdout experiment accepts or rejects every proposal.

Review the aggregate residual evidence below. Call `propose_feature` once for each feature
(at most three) from the allowed catalog that could explain the systematic underprediction
in the identified segment. Do not invent fields, formulas, SQL, or clinical conclusions.
If the evidence is insufficient, call `stop` instead.

Residual evidence:
{evidence}
"""


def _tools():
    from google.genai import types
    propose = types.FunctionDeclaration(
        name="propose_feature",
        description="Propose one testable derived feature from the allowed whitelist.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "feature_name": {"type": "STRING", "enum": list(ALLOWED_FEATURE_NAMES)},
                "hypothesis": {"type": "STRING", "description": "Why this feature may explain the underprediction."},
                "expected_error_segment": {"type": "STRING"},
                "confidence": {"type": "NUMBER", "description": "0.0 to 1.0"},
            },
            "required": ["feature_name", "hypothesis", "confidence"],
        },
    )
    stop = types.FunctionDeclaration(
        name="stop", description="Signal that the evidence does not justify any candidate.",
        parameters={"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]},
    )
    return types.Tool(function_declarations=[propose, stop])


def propose_features(residual_summary: dict, model: str, project: str, location: str,
                     temperature: float = 0.2) -> HypothesisResponse:
    """Call Gemini on Vertex with function calling; return the parsed response.

    Requires a live Vertex project — raises on any failure (no fallback).
    """
    if not project:
        raise ValueError("propose_features requires a Vertex project (GOOGLE_CLOUD_PROJECT).")
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, project=project, location=location)
    resp = client.models.generate_content(
        model=model,
        contents=PROMPT.format(evidence=json.dumps(residual_summary, indent=2)),
        config=types.GenerateContentConfig(
            temperature=temperature, tools=[_tools()],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
        ),
    )
    return _parse_calls(resp.function_calls or [])


def _parse_calls(calls) -> HypothesisResponse:
    candidates, seen = [], set()
    for call in calls:
        if call.name == "stop":
            return HypothesisResponse(action="STOP", stop_reason=(call.args or {}).get("reason", "insufficient evidence"))
        if call.name == "propose_feature":
            a = call.args or {}
            name = a.get("feature_name")
            if name in ALLOWED_FEATURE_NAMES and name not in seen and len(candidates) < 3:
                seen.add(name)
                candidates.append(FeatureCandidate(
                    feature_name=name, hypothesis=a.get("hypothesis", ""),
                    expected_error_segment=a.get("expected_error_segment", ""),
                    confidence=float(a.get("confidence", 0.5)),
                ))
    if not candidates:
        return HypothesisResponse(action="STOP", stop_reason="model made no valid tool calls")
    return HypothesisResponse(action="PROPOSE", candidates=candidates)


if __name__ == "__main__":
    import os
    r = propose_features({"segment_description": "high ED, many providers"},
                         "gemini-2.5-flash", os.environ["GOOGLE_CLOUD_PROJECT"], "us-central1")
    print("gemini:", r.action, [c.feature_name for c in r.candidates])
