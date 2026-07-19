# Demo talk track (5 minutes)

Goal: show end-to-end Vertex AI mastery + a real Snowflake↔GCP bridge, and how the two
platforms are complementary. Record with Loom / screen capture.

## Pre-flight (before recording)
1. Tailscale exit node **OFF**; on home/DigitalOcean VPN (Snowflake IP allowlist).
2. `gcloud auth application-default print-access-token` returns a token.
3. Deploy the endpoint (~10 min) so the Explainable-AI section is live:
   `python -c "import sys; sys.path.insert(0,'src'); ..."` (or run the deploy helper).
4. `streamlit run app.py`; toggles **Use live Snowflake** and **Use live Vertex** ON.
5. Have two browser tabs ready: Vertex **Experiments** and **Model Registry**.
6. **After recording: undeploy the endpoint** (`teardown_endpoint`) — see TEARDOWN.md.

## The 5 minutes

**0:00 — Framing (30s).** "Snowflake governs the healthcare data; Vertex AI is the ML
control plane; XGBoost — not the language model — decides. Synthetic data, no PHI."

**0:30 — Snowflake data plane (45s).** Section 1. "2,000 members, ~370k claims in Snowflake;
features are built *in-warehouse* with SQL time windows — the data never leaves." Toggle
shows `Snowflake (live, in-warehouse SQL)`.

**1:15 — Baseline (30s).** Section 2. "XGBoost beats the median baseline ~2.4× on MAE. But it
systematically underpredicts a segment."

**1:45 — Residual → Gemini (75s).** Sections 3–4. "We isolate the worst-underprediction segment
with a decision tree and send **only aggregate JSON** to Gemini. On Vertex, Gemini doesn't
free-text — it **calls a `propose_feature` tool** bound to a whitelist (function calling), and
each hypothesis is scored by the **Vertex Gen AI Evaluation Service**." Click *Run Gemini*.

**3:00 — The gate (45s).** Section 5. "Each proposal becomes a challenger model on the same
holdout. The gate accepts `COST_ACCELERATION` (+1.6% MAE) and **rejects** the other two.
Gemini never sets this field — the holdout owns the truth."

**3:45 — Serving + Explainable AI (45s).** Section 6. "The champion is registered in Vertex
Model Registry and deployed to an online endpoint — live prediction on a held-out member.
Attributions (SHAP on the champion; the endpoint is also deployed with a sampled-Shapley
Explainable-AI spec) show which features drive the cost." Show Experiments + Registry tabs.

> Live-verified: endpoint `predict` returns $151,099 vs. actual $132,946 for the sample member.
> The app requires a live endpoint for section 6 — deploy it first (`vertex_prediction.deploy_champion`) so live Vertex Explainable AI is on. **Undeploy after.**

**4:30 — Close (30s).** "Snowflake governs the data and computes the features — the
analytical system of record — while Vertex runs the ML lifecycle: GenAI, evaluation,
experiments, registry, serving. They're complementary: Snowflake Cortex is a great fit for
LLM work inside a SQL/analytics workflow; here the LLM output is an ML-lifecycle artifact —
evaluated, versioned, and gated alongside the model it feeds — so it lives with the rest of
the ML lifecycle on Vertex. Two strong platforms, one clean aggregate-only boundary."

## One-liner for the resume / LinkedIn
> Built a hybrid Snowflake + Google Vertex AI MLOps pipeline: Snowflake governs synthetic
> healthcare claims and SQL features; Gemini on Vertex proposes features via function calling
> and is scored by the Gen AI Evaluation Service; runs are tracked in Vertex AI Experiments,
> the champion is versioned in Model Registry and served from an endpoint with Explainable AI;
> and a deterministic XGBoost holdout gate — not the LLM — accepts or rejects each hypothesis.
```
