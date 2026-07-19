# Teardown & cost cleanup

Everything this project creates and how to remove it. Replace `PROJECT_ID` / `BUCKET`
with your own. **The only resource that bills continuously is a running online endpoint —
undeploy it when you're done.**

## GCP

| Resource | Remove |
|---|---|
| 💰 **Online endpoint** (`n1-standard-2`, ~$0.11/hr) | `python -c "import sys;sys.path.insert(0,'src');from vertex_prediction import find_endpoint,teardown_endpoint;e=find_endpoint('carecost-champion','PROJECT_ID');e and teardown_endpoint(e)"` |
| Registered models | `gcloud ai models list --region=us-central1` then `gcloud ai models delete <id> --region=us-central1` |
| Vertex Experiment `carecost-fusion` | `python -c "from google.cloud import aiplatform as a;a.init(project='PROJECT_ID',location='us-central1');a.Experiment('carecost-fusion').delete()"` |
| Pipeline jobs | auto-finish (no ongoing cost); artifacts live in the bucket |
| GCS bucket (artifacts, wheel, pipeline root) | `gcloud storage rm --recursive gs://BUCKET` |
| Enabled APIs | optional — leaving them enabled costs nothing |
| ADC credentials | `gcloud auth application-default revoke` |

> Set a **budget alert** (Console → Billing → Budgets) as a seatbelt. Total build cost is
> typically a few dollars, or $0 on the GCP free trial.

## Snowflake

```sql
DROP WAREHOUSE IF EXISTS CARECOST_WH;
DROP DATABASE  IF EXISTS CARECOST_DEMO;   -- drops all tables + the schema
```

> If your account uses an IP allowlist, programmatic access must egress from an allowlisted
> IP — no allowlist change is required by this project, so there's nothing to revert there.

## Local

```bash
rm -rf .venv dist pipeline.json          # virtualenv, built wheel, compiled pipeline
# (gcloud SDK itself: `brew uninstall --cask gcloud-cli` only if you no longer need it)
```

## Full sequence (when the project is done)

```bash
# 1. GCP — endpoints first (they bill), then artifacts
python -c "import sys;sys.path.insert(0,'src');from vertex_prediction import find_endpoint,teardown_endpoint;e=find_endpoint('carecost-champion','PROJECT_ID');e and teardown_endpoint(e)"
gcloud storage rm --recursive gs://BUCKET
gcloud auth application-default revoke
# 2. Snowflake
#    DROP WAREHOUSE CARECOST_WH; DROP DATABASE CARECOST_DEMO;
# 3. Local
rm -rf .venv dist pipeline.json
```
