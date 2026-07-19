# CareCost Fusion — Snowflake × Google Vertex AI

> **Gemini proposes. XGBoost proves. Snowflake governs.**

A hybrid MLOps demo where **Google Vertex AI is the ML control plane** and **Snowflake is
the governed enterprise data source**. Gemini proposes healthcare-cost features via
**function calling**; the **Vertex Gen AI Evaluation Service** scores them; runs are tracked
in **Vertex AI Experiments**; the champion is versioned in **Vertex Model Registry** and
served from an **Online Endpoint with Explainable AI**; and a deterministic **XGBoost holdout
gate — not the LLM — accepts or rejects each hypothesis. A **Vertex AI Pipeline** (KFP)
orchestrates the whole DAG.

*Synthetic data only. No PHI. Not HIPAA compliant. A financial-forecasting experiment.*
📖 The story: **[BLOG.md](BLOG.md)** · the design: **[ARCHITECTURE.md](ARCHITECTURE.md)** ·
cost: **[TCO.md](TCO.md)** · demo script: **[DEMO.md](DEMO.md)**.

---

## The problem it solves

LLMs now *suggest* features and rules, but **can't judge whether their own suggestion is
correct.** CareCost Fusion makes an LLM propose healthcare-cost features and lets a
**holdout ML gate decide** — Snowflake governs the data, Vertex runs the ML lifecycle.

## What you get (the result)

| run | feature | MAE | recall | improvement | decision |
|-----|---------|-----|--------|-------------|----------|
| median | — | $78,062 | — | — | baseline |
| xgb-baseline | 10 base features | $32,829 | 0.924 | — | baseline |
| challenger | `COST_ACCELERATION` | $32,297 | 0.924 | **+1.62%** | **ACCEPT** |
| challenger | `PROVIDER_FRAGMENTATION` | $33,022 | 0.924 | −0.59% | REJECT |
| challenger | `INPATIENT_COST_SHARE` | $32,646 | 0.925 | +0.56% | REJECT |

> This runs **live** on Snowflake + Vertex — there are no offline fallbacks. That's the point.

---

## Quick start (replicate it)

### 0. Prerequisites
- Python **3.11**, `git`
- A **GCP project** with billing (or the $300 free trial) — total cost < $15, often $0
- A **Snowflake** account (source only)
- `gcloud`: `brew install --cask gcloud-cli`

### 1. Code + environment
```bash
git clone <your-fork> carecost-fusion && cd carecost-fusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # 18 unit tests (pure ML logic — no cloud needed)
```

### 2. GCP / Vertex
```bash
gcloud auth application-default login          # ADC (no key files)
gcloud config set project <PROJECT_ID>
gcloud services enable aiplatform.googleapis.com storage.googleapis.com \
  compute.googleapis.com iam.googleapis.com cloudresourcemanager.googleapis.com
gcloud storage buckets create gs://<PROJECT_ID>-carecost --location=us-central1
export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
# confirm Gemini:
python -c "from google import genai; c=genai.Client(vertexai=True,project='<PROJECT_ID>',location='us-central1'); print(c.models.generate_content(model='gemini-2.5-flash',contents='say OK').text)"
```

### 3. Snowflake (source)
Configure `~/.snowflake/connections.toml` (connection `WEB`, **password/PAT only — no
`authenticator`**), then create the dedicated objects once:
```sql
CREATE WAREHOUSE IF NOT EXISTS CARECOST_WH WAREHOUSE_SIZE='XSMALL'
  AUTO_SUSPEND=60 AUTO_RESUME=TRUE INITIALLY_SUSPENDED=TRUE;
CREATE DATABASE IF NOT EXISTS CARECOST_DEMO;
CREATE SCHEMA  IF NOT EXISTS CARECOST_DEMO.ANALYTICS;
```
> If your account has an IP allowlist, egress from an **allowlisted IP** (turn off any VPN
> exit node that changes your public IP). Check with `curl ipinfo.io/json`.

### 4. Run the demo UI
```bash
streamlit run app.py
```
Walk sections 1→6; click **Run Gemini hypothesis step** to see function-calling + Gen AI
Eval + the gate. The **Show walkthrough notes** toggle adds a *What · Why · How* panel to
every step.

### 5. (Optional) Deploy the endpoint for live Explainable AI
`vertex_prediction.deploy_champion(...)` deploys the champion; `teardown_endpoint(...)`
removes it. **The endpoint is the only ~$0.11/hr resource — undeploy after.**

### 6. Run the Vertex AI Pipeline (KFP)
```bash
python -m build --wheel      # package src/ (once, and after code changes)
python -c "from google.cloud import storage; storage.Client().bucket('<PROJECT_ID>-carecost').blob('pkg/carecost_fusion-0.1.0-py3-none-any.whl').upload_from_filename('dist/carecost_fusion-0.1.0-py3-none-any.whl')"
python -c "import sys;sys.path.insert(0,'src');from vertex_pipeline import run_pipeline;print(run_pipeline().resource_name)"
```
Watch it in Vertex AI → **Pipelines**: `gen_features → baseline_segment → gemini_propose →
challengers_gate → register_champion`.

### 7. Tear down
Every resource + revert command is in **[TEARDOWN.md](TEARDOWN.md)**. Essentials:
```bash
python -c "import sys;sys.path.insert(0,'src');from vertex_prediction import find_endpoint,teardown_endpoint;e=find_endpoint('carecost-champion','<PROJECT_ID>');e and teardown_endpoint(e)"
gcloud storage rm --recursive gs://<PROJECT_ID>-carecost
# Snowflake:  DROP WAREHOUSE CARECOST_WH;  DROP DATABASE CARECOST_DEMO;
```

---

## Layout

```
app.py                    Streamlit demo UI (the video centerpiece)
BLOG.md · ARCHITECTURE.md · TCO.md · DEMO.md · TEARDOWN.md
src/
  generate_claims.py      deterministic synthetic claims (hidden growth trajectory)
  features.py             the feature windows (also the pipeline's synthetic source)
  modeling.py             temporal split, XGBoost, median baseline
  evaluation.py           metrics + the ACCEPT/REJECT/REVIEW gate
  residuals.py            decision-tree residual segment (aggregate only)
  feature_catalog.py      whitelist + validators
  gemini_hypothesis.py    Gemini function-calling (Vertex)
  vertex_geneval.py       Gen AI Evaluation (custom pointwise metric)
  vertex_experiments.py   Vertex AI Experiments
  vertex_registry.py      Vertex Model Registry
  vertex_prediction.py    Endpoint deploy + Explainable AI + teardown
  vertex_pipeline.py      Vertex AI Pipeline (KFP)
  pipeline.py             shared orchestration
  snowflake_io.py         Snowflake source (connections.toml, password-only)
sql/                      00_setup · 01_base_features · 02_residual_summary · 03_candidate_features
notebooks/carecost_fusion.ipynb
```

## License
MIT — see [LICENSE](LICENSE).
