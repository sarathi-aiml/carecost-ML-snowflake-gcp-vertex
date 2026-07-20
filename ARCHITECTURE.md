# Architecture

CareCost Fusion is a **hybrid MLOps pipeline**: Snowflake is the enterprise data source
and governed store; **Google Vertex AI is the ML control plane** — GenAI hypothesis,
evaluation, experiment tracking, model registry, and an explainable serving endpoint.
XGBoost's holdout gate makes every accept/reject decision.

## Division of labor

```
 SNOWFLAKE (source of truth)                 GOOGLE VERTEX AI (ML control plane)
 ─────────────────────────────              ───────────────────────────────────────
 • RAW_CLAIMS (synthetic)                    • Gemini (function-calling) → feature hypotheses
 • SQL window features  ───── features ────▶ • Gen AI Evaluation → score the hypotheses
   (MEMBER_FEATURES_BASE)                    • Vertex AI Experiments → run ledger (+ SF credits)
 • predictions + residuals                   • Model Registry → champion/challenger versions
 • EXPERIMENT_RESULTS (verdict)  ◀── gate ── • Online Endpoint + Explainable AI (Shapley)
                                             • (Vertex AI Pipeline orchestrates the above)
        ▲                                                   │
        └─────── aggregate residual JSON only ──────────────┘
                 + KB-sized model.bst artifact
```

**Data boundary:** only an *aggregate residual summary* (counts + means, no member rows,
no IDs) and a *KB-sized model artifact* ever cross to GCP. The claim-level data never
leaves Snowflake — the governance story for a healthcare-shaped workload.

## Flow
1. Snowflake generates/loads `RAW_CLAIMS` and builds `MEMBER_FEATURES_BASE` with SQL time
   windows (`history: SERVICE_DATE < INDEX_DATE`; `target: [INDEX_DATE, +90d)`).
2. Chronological 60/20/20 split; XGBoost baseline vs. median; predictions → Snowflake.
3. A depth-3 tree isolates the worst underprediction segment → **aggregate JSON only**.
4. **Gemini on Vertex** calls `propose_feature(...)` (tool-constrained to the whitelist).
5. **Vertex Gen AI Evaluation** scores each hypothesis (custom pointwise metric).
6. Python validates (whitelist + leakage + columns); one **challenger per feature**.
7. Every run → **Vertex AI Experiments**; champion → **Model Registry**; deploy →
   **Endpoint + Explainable AI**.
8. Deterministic gate writes `ACCEPT / REJECT / REVIEW` to Snowflake `EXPERIMENT_RESULTS`.

## Snowflake and Vertex — complementary, by design

Snowflake is the governed home for the data and the SQL feature engineering — the
analytical system of record. For LLM work that lives **inside** a SQL/analytics workflow,
Snowflake Cortex is an excellent fit and keeps everything data-local.

In this project the Gemini output is an **ML-lifecycle artifact** — it's tool-constrained,
independently evaluated, versioned, and gated right alongside the model it feeds — so it
naturally sits with the rest of the ML lifecycle (training, registry, serving) on Vertex.
Two strong platforms, each doing what it's best at, with a clean aggregate-only boundary
between them:

| | Snowflake (data + analytics) | Vertex AI (this project's ML lifecycle) |
|---|---|---|
| Owns | claims, SQL features, results, governance | Gemini, Gen AI Eval, Experiments, Registry, serving |
| LLM fit | in-SQL/analytics enrichment (Cortex) | tool-constrained, evaluated, versioned, gated |
| Boundary | keeps the data | receives only aggregate JSON + a KB artifact |



## What I'd productionize next
- Make Snowflake the single feature source (Dynamic Tables / Feature Store); give the Vertex Pipeline warehouse access via Cloud NAT + a reserved, allowlisted egress IP.
- Vertex **Model Monitoring** on the endpoint (training-serving skew + drift).
- Vertex **Vizier** HP tuning as a pipeline step.
- CI to compile + run the Vertex Pipeline on PRs; promote champion by alias, not redeploy.
- Replace synthetic data with Synthea / CMS DE-SynPUF via the pluggable generator.
- Secret management (Secret Manager) instead of local `connections.toml`; Workload Identity for SF↔GCS.
```
