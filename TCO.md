# Cost / TCO

Small-scale demo (2,000 members, ~370k synthetic claim rows, tiny XGBoost). Cost is
dominated by **one** thing: whether the online endpoint is left running.

## Per-run / build cost (GCP)

| Item | Unit cost | Demo usage | Cost |
|---|---|---|---|
| Gemini `2.5-flash` (function-calling) | ~$/1M tokens (tiny prompts) | a few calls | ~$0.01 |
| Gen AI Evaluation (autorater) | per request | 3 requests/run | ~$0.05–0.30 |
| Vertex AI Experiments / ML Metadata | effectively free | all runs | ~$0 |
| Model Registry | free | a few models | $0 |
| GCS (artifacts) | $0.02/GB/mo | KB–MB | ~$0 |
| **Online Endpoint** (`n1-standard-2`) | **~$0.11/hr** | **deploy only for demo, then undeploy** | **~$0.03 per 15-min cycle** |
| Batch prediction (if used) | small | occasional | ~$0.10 |

**Build + demo total: well under $15**, and **$0 out of pocket if on the GCP $300 free trial.**

## The only ongoing cost
A running **Online Endpoint** bills ~$0.11/hr (~$80/mo if left 24/7). `vertex_prediction.teardown_endpoint()`
undeploys + deletes it; `TEARDOWN.md` has the command. **Always undeploy after recording.**

## Snowflake
Source only — a ~370k-row load + a couple of SQL feature builds on an **XS** warehouse with
60s auto-suspend. A few minutes of XS credits per run (well under $1). Warehouse suspends itself.

## Cost controls in place
- XS warehouse, `AUTO_SUSPEND=60`, `INITIALLY_SUSPENDED=TRUE`.
- Managed TensorBoard disabled (it's a recurring charge and being sunset).
- Endpoint deploy is opt-in (UI works offline via local SHAP); teardown scripted.
- Recommend a **$50 GCP budget alert** as a seatbelt (Console → Billing → Budgets).

## What would change the cost at production scale
- Endpoint kept warm for real-time SLAs → size + replica count drive cost; autoscale to zero if latency allows.
- Model Monitoring → BigQuery logging + scheduled analysis jobs.
- Larger data → Snowflake warehouse size + Vertex training machine type; both scale roughly linearly.
```
