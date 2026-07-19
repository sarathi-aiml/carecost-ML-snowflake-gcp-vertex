# CareCost Fusion — Snowflake + Vertex AI Weekend MVP

## AI Implementation Specification

**Project title:** CareCost Fusion  
**Public tagline:** **Gemini proposes. XGBoost proves. Snowflake governs.**  
**Repository name:** `carecost-fusion-snowflake-vertex`  
**Primary audience:** Google AI/ML recruiters, ML engineers, solution architects, and healthcare data practitioners  
**Implementation target:** A small, polished proof of concept that an AI coding agent can build end to end  
**Expected build format:** One runnable notebook plus a few support files  
**Not a production system**

---

# 1. Purpose

Build a compact healthcare cost-prediction demonstration that combines:

- **Snowflake** for governed data storage, SQL-based feature engineering, residual analysis, and result storage.
- **Vertex AI** for Gemini-based feature-hypothesis generation and experiment tracking.
- **XGBoost** for the actual healthcare cost prediction and champion–challenger validation.

The project must demonstrate that generative AI does not determine whether its own suggestion is correct.

The core principle is:

> **Gemini proposes a measurable feature hypothesis. XGBoost and a holdout experiment accept or reject it. Snowflake remains the data and analytical system of record.**

The project is designed to showcase traditional ML depth in a market where many portfolios show only chatbots, RAG, and generic agents.

---

# 2. Final Demonstration Story

The final demonstration should take three to five minutes.

1. Show synthetic healthcare claims stored in Snowflake.
2. Show Snowflake SQL creating member-level historical features.
3. Train an XGBoost baseline model for next-90-day cost.
4. Identify a segment where the model systematically underpredicts cost.
5. Send only an aggregated residual summary to Gemini through Vertex AI.
6. Gemini proposes up to three structured feature hypotheses.
7. Materialize the approved candidate features in Snowflake.
8. Train one challenger model per candidate.
9. Compare baseline and challengers on the same temporal holdout data.
10. Log all runs to Vertex AI Experiments.
11. Write predictions, metrics, and accept/reject results back to Snowflake.
12. Show that at least one Gemini idea can be rejected when it does not improve the ML model.

The closing statement should be:

> “Snowflake governed the healthcare data and computed the features. Vertex AI managed the Gemini hypothesis and experiment evidence. XGBoost—not the language model—decided whether the proposed feature improved prediction.”

---

# 3. Why Snowflake and Vertex AI Both Exist

This project must not call the same Gemini model from both Snowflake and Vertex AI.

Do **not** use Snowflake `AI_COMPLETE`, Cortex Complete, or another LLM inside Snowflake in this MVP.

The platform responsibilities must remain distinct.

## 3.1 Snowflake responsibility

Snowflake is the **data and analytical layer**.

Use it for:

- Synthetic claims storage
- Canonical healthcare tables
- SQL time-window feature engineering
- Training-view creation
- Prediction storage
- Residual calculation
- Segment aggregation
- Experiment-result storage
- Final analytical queries

## 3.2 Vertex AI responsibility

Vertex AI is the **ML experimentation and Gemini layer**.

Use it for:

- Calling Gemini with structured output
- Generating testable feature hypotheses
- Logging XGBoost experiment parameters
- Logging evaluation metrics
- Comparing baseline and challenger runs in Vertex AI Experiments

## 3.3 XGBoost responsibility

XGBoost is the **prediction and validation model**.

Use it for:

- Baseline next-90-day cost prediction
- Challenger model training
- Residual creation
- Feature importance
- Measured accept/reject decisions

## 3.4 Hybrid-enterprise justification

The architecture represents an enterprise in which:

- Governed healthcare data already resides in Snowflake.
- The data-science organization uses Vertex AI for experimentation and Gemini services.
- Raw claims do not need to be permanently copied to another warehouse.
- A small, de-identified member-level feature table is temporarily loaded into notebook memory for model training.
- Aggregated residual evidence—not raw claim rows—is sent to Gemini.

If an organization standardizes fully on Snowflake ML, Vertex AI may not be required. This POC intentionally demonstrates the common hybrid operating model.

---

# 4. Architecture

```text
Deterministic Synthetic Claims Generator
                  │
                  ▼
        Snowflake RAW_CLAIMS
                  │
                  ▼
 Snowflake SQL Member Feature View
    30d / 90d / 180d history
                  │
                  ▼
Notebook retrieves de-identified feature rows
                  │
                  ▼
       XGBoost Baseline Model
                  │
                  ▼
 Predictions written to Snowflake
                  │
                  ▼
 Snowflake Residual Segment Summary
                  │
          aggregate JSON only
                  ▼
       Gemini on Vertex AI
 proposes structured feature candidates
                  │
                  ▼
 Whitelist + leakage validation in Python
                  │
                  ▼
 Snowflake SQL materializes candidates
                  │
                  ▼
 XGBoost Challenger Models
                  │
          ┌───────┴────────┐
          ▼                ▼
Vertex AI Experiments   Snowflake
params + metrics        experiment results
          │                │
          └───────┬────────┘
                  ▼
        ACCEPT / REJECT result
```

---

# 5. Strict MVP Scope

The coding AI must implement only the following.

## Required

- One deterministic synthetic claims generator
- One Snowflake setup SQL script
- One Jupyter notebook
- One XGBoost regression baseline
- One temporal train/validation/test split
- One residual-segmentation method
- One Gemini structured-output call through Vertex AI
- Three candidate feature types
- One challenger run per valid candidate
- Vertex AI Experiments logging
- Predictions and experiment results written back to Snowflake
- README with architecture, setup, output, and screenshots
- Basic unit tests for leakage and feature validation

## Explicitly excluded

Do not add:

- Vertex AI Pipelines
- Vertex AI online endpoints
- Vertex AI Custom Training jobs
- Vertex AI Model Registry
- Vertex AI Workbench requirement
- BigQuery
- Google ADK
- Multi-agent architecture
- Snowflake Cortex or `AI_COMPLETE`
- Snowflake Model Registry
- Snowflake Feature Store
- Streamlit unless all required work is already complete
- Docker or Kubernetes
- Terraform
- CI/CD
- Real healthcare data
- PHI
- FHIR server
- Production monitoring
- Fine-tuning
- Real-time scoring
- A/B testing on real users

The notebook may run on a Mac, local Jupyter, VS Code, Google Colab, or another Python environment.

---

# 6. Repository Structure

Create exactly this initial structure:

```text
carecost-fusion-snowflake-vertex/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── config.example.yaml
├── notebooks/
│   └── carecost_fusion.ipynb
├── src/
│   ├── __init__.py
│   ├── generate_claims.py
│   ├── snowflake_io.py
│   ├── modeling.py
│   ├── residuals.py
│   ├── gemini_hypothesis.py
│   ├── feature_catalog.py
│   ├── vertex_experiments.py
│   └── evaluation.py
├── sql/
│   ├── 00_setup.sql
│   ├── 01_base_features.sql
│   ├── 02_residual_summary.sql
│   └── 03_candidate_features.sql
├── tests/
│   ├── test_data_generation.py
│   ├── test_temporal_leakage.py
│   ├── test_feature_catalog.py
│   └── test_evaluation_gate.py
└── artifacts/
    └── .gitkeep
```

Do not create unnecessary folders.

---

# 7. Runtime and Dependencies

Use Python 3.11 or later.

Minimum `requirements.txt`:

```text
pandas
numpy
pyarrow
scikit-learn
xgboost
shap
matplotlib
jupyter
python-dotenv
pyyaml
pydantic
snowflake-connector-python[pandas]
snowflake-snowpark-python
google-genai
google-cloud-aiplatform
pytest
```

Pin stable compatible versions after resolving the environment. Do not pin blindly before installation testing.

---

# 8. Configuration

Create `.env.example`:

```bash
# Snowflake
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DATABASE=CARE_COST_DEMO
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=

# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=
GEMINI_MODEL=
VERTEX_EXPERIMENT_NAME=carecost-fusion
```

Create `config.example.yaml`:

```yaml
data:
  random_seed: 42
  member_count: 2000
  history_months: 15
  index_date: "2026-01-01"

model:
  random_seed: 42
  n_estimators: 300
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.85
  colsample_bytree: 0.85

evaluation:
  minimum_mae_improvement_pct: 1.0
  maximum_high_cost_recall_drop_points: 0.02

gemini:
  max_candidates: 3
  temperature: 0.2
```

The Gemini model name must be configurable. Do not hard-code a model that may be renamed.

---

# 9. Synthetic Data Design

Do not spend time downloading or cleaning a large public dataset.

Generate a deterministic claims-style dataset that is understandable and reproducible.

## 9.1 Dataset size

Generate:

- 2,000 synthetic members
- 15 months of claim events
- Approximately 20,000 to 60,000 claim rows
- One final index date used to create the training population
- Multiple historical monthly index dates if needed for more training rows

## 9.2 Raw claims columns

Create the following columns:

```text
CLAIM_ID
MEMBER_ID
SERVICE_DATE
CLAIM_TYPE
DIAGNOSIS_GROUP
PROVIDER_ID
PAID_AMOUNT
INPATIENT_FLAG
ED_FLAG
```

Example values:

### `CLAIM_TYPE`

```text
INPATIENT
OUTPATIENT
EMERGENCY
PROFESSIONAL
PHARMACY
```

### `DIAGNOSIS_GROUP`

```text
CARDIOVASCULAR
DIABETES
RESPIRATORY
MUSCULOSKELETAL
ONCOLOGY
MENTAL_HEALTH
OTHER
```

## 9.3 Data distributions

Use plausible synthetic distributions, not uniform random values.

Recommended approach:

- Member baseline risk from a gamma distribution
- Claim count from a Poisson distribution influenced by baseline risk
- Claim cost from a lognormal distribution
- Inpatient claims rare but expensive
- Emergency claims moderately expensive
- Pharmacy and professional claims common and lower-cost
- Some members receive recent utilization escalation
- Some members use many distinct providers

## 9.4 Hidden relationship for the demo

Seed a relationship that the baseline feature set intentionally does not capture directly.

The future 90-day cost should be influenced by:

```text
cost_acceleration =
    cost_in_most_recent_30_days
    /
    normalized_cost_in_previous_90_days
```

and:

```text
provider_fragmentation =
    distinct_provider_count_90d
    /
    max(claim_count_90d, 1)
```

Include an interaction:

```text
high recent cost acceleration
AND
high provider fragmentation
→ elevated next-90-day cost
```

The baseline model must receive the underlying counts and totals but must initially omit these two derived ratios.

This creates a realistic residual pattern that Gemini can hypothesize and the challenger model can test.

Do not guarantee that every candidate wins. One candidate should ideally improve the model and at least one should be rejected.

---

# 10. Snowflake Objects

The setup script must create:

```sql
CREATE DATABASE IF NOT EXISTS CARE_COST_DEMO;
CREATE SCHEMA IF NOT EXISTS CARE_COST_DEMO.PUBLIC;
```

Create these tables:

## 10.1 `RAW_CLAIMS`

```sql
CREATE OR REPLACE TABLE RAW_CLAIMS (
    CLAIM_ID STRING,
    MEMBER_ID STRING,
    SERVICE_DATE DATE,
    CLAIM_TYPE STRING,
    DIAGNOSIS_GROUP STRING,
    PROVIDER_ID STRING,
    PAID_AMOUNT FLOAT,
    INPATIENT_FLAG INTEGER,
    ED_FLAG INTEGER
);
```

## 10.2 `MEMBER_FEATURES_BASE`

Materialized table or view containing:

```text
MEMBER_ID
INDEX_DATE
COST_30D
COST_90D
COST_180D
CLAIM_COUNT_30D
CLAIM_COUNT_90D
INPATIENT_COUNT_90D
ED_COUNT_30D
ED_COUNT_90D
DISTINCT_PROVIDER_COUNT_90D
DISTINCT_DIAGNOSIS_COUNT_90D
NEXT_90D_COST
```

## 10.3 `MODEL_PREDICTIONS`

```sql
CREATE OR REPLACE TABLE MODEL_PREDICTIONS (
    RUN_ID STRING,
    MODEL_TYPE STRING,
    MEMBER_ID STRING,
    INDEX_DATE DATE,
    ACTUAL_COST FLOAT,
    PREDICTED_COST FLOAT,
    RESIDUAL FLOAT,
    ABSOLUTE_ERROR FLOAT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
```

## 10.4 `RESIDUAL_SUMMARY`

```sql
CREATE OR REPLACE TABLE RESIDUAL_SUMMARY (
    RUN_ID STRING,
    SEGMENT_ID STRING,
    SEGMENT_DESCRIPTION STRING,
    MEMBER_COUNT INTEGER,
    MEAN_ACTUAL_COST FLOAT,
    MEAN_PREDICTED_COST FLOAT,
    MEAN_RESIDUAL FLOAT,
    MEAN_ABSOLUTE_ERROR FLOAT,
    SEGMENT_EVIDENCE VARIANT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
```

## 10.5 `EXPERIMENT_RESULTS`

```sql
CREATE OR REPLACE TABLE EXPERIMENT_RESULTS (
    RUN_ID STRING,
    PARENT_RUN_ID STRING,
    RUN_TYPE STRING,
    FEATURE_NAME STRING,
    FEATURE_SPEC VARIANT,
    MAE FLOAT,
    RMSE FLOAT,
    RMSLE FLOAT,
    HIGH_COST_RECALL FLOAT,
    MAE_IMPROVEMENT_PCT FLOAT,
    DECISION STRING,
    DECISION_REASON STRING,
    GEMINI_HYPOTHESIS STRING,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
```

---

# 11. Snowflake Feature Engineering

Use Snowflake SQL for all historical window calculations.

## 11.1 Index-date population

Create multiple index dates per member to produce enough supervised rows.

Each feature row must use only claim events before `INDEX_DATE`.

The target must use events on or after `INDEX_DATE` and before `DATEADD(day, 90, INDEX_DATE)`.

## 11.2 Temporal boundary rule

Historical features:

```sql
SERVICE_DATE < INDEX_DATE
```

Target:

```sql
SERVICE_DATE >= INDEX_DATE
AND SERVICE_DATE < DATEADD(day, 90, INDEX_DATE)
```

These rules are mandatory.

## 11.3 Baseline features

Calculate:

```sql
SUM(IFF(SERVICE_DATE >= DATEADD(day, -30, INDEX_DATE), PAID_AMOUNT, 0))
```

and analogous 90-day and 180-day windows.

Calculate counts and distinct counts using conditional aggregation.

Do not include:

- `COST_ACCELERATION`
- `PROVIDER_FRAGMENTATION`
- Any event occurring inside the prediction window
- Future claims
- Target-derived statistics

## 11.4 Data movement

Use Snowflake Connector for Python or Snowpark.

Recommended small-data approach:

```python
cursor.execute("SELECT * FROM MEMBER_FEATURES_BASE ORDER BY INDEX_DATE")
features_df = cursor.fetch_pandas_all()
```

Use `write_pandas()` or `Session.write_pandas()` to write generated claims, predictions, and experiment results back into Snowflake.

Raw claims must remain in Snowflake after loading. Only the member-level feature table is temporarily loaded into notebook memory.

---

# 12. Temporal ML Split

Do not use a random split.

Sort distinct index dates chronologically.

Use:

- Earliest 60% of index dates for training
- Next 20% for validation
- Latest 20% for final testing

All baseline and challenger models must use exactly the same splits.

Add a test confirming:

```text
max(train.index_date) < min(validation.index_date)
max(validation.index_date) < min(test.index_date)
```

---

# 13. Baseline Model

Use `XGBRegressor`.

## 13.1 Target transformation

Healthcare cost is right-skewed.

Train on:

```python
y_train_log = np.log1p(y_train)
```

Convert predictions back:

```python
predicted_cost = np.expm1(predicted_log_cost)
predicted_cost = np.clip(predicted_cost, 0, None)
```

## 13.2 Baseline features

Use:

```text
COST_30D
COST_90D
COST_180D
CLAIM_COUNT_30D
CLAIM_COUNT_90D
INPATIENT_COUNT_90D
ED_COUNT_30D
ED_COUNT_90D
DISTINCT_PROVIDER_COUNT_90D
DISTINCT_DIAGNOSIS_COUNT_90D
```

## 13.3 Simple comparison baseline

Also calculate a naïve baseline:

```text
predicted next-90-day cost = training-set median next-90-day cost
```

XGBoost should be compared against this simple baseline.

---

# 14. Evaluation Metrics

Calculate:

## Primary

```text
MAE
RMSE
RMSLE
```

## High-cost member metric

Define high-cost members using the 90th percentile of actual cost in the training set.

Calculate:

```text
HIGH_COST_RECALL
```

Meaning:

> Among actual high-cost members in the test set, how many appear in the top predicted-risk group?

Use a consistent cutoff across all runs.

## Required comparison table

```text
RUN_ID
FEATURE_NAME
MAE
RMSE
RMSLE
HIGH_COST_RECALL
MAE_IMPROVEMENT_PCT
DECISION
```

---

# 15. Residual Analysis

Calculate:

```python
residual = actual_cost - predicted_cost
absolute_error = abs(residual)
```

Positive residual means the model underpredicted cost.

## 15.1 Segment discovery

Use a shallow `DecisionTreeRegressor` or `DecisionTreeClassifier` to discover interpretable error segments.

Recommended target:

```text
large_underprediction =
residual >= 90th percentile of positive residuals
```

Features available to the error tree may include the baseline feature columns.

Limit the tree:

```text
max_depth = 3
min_samples_leaf = 30
```

Extract the highest-underprediction leaf as a human-readable segment.

## 15.2 Residual summary package

Create a compact object:

```json
{
  "segment_id": "segment_001",
  "segment_description": "Recent emergency utilization and multiple providers",
  "member_count": 150,
  "mean_actual_cost": 9200.0,
  "mean_predicted_cost": 5400.0,
  "mean_residual": 3800.0,
  "conditions": [
    "ED_COUNT_30D >= 2",
    "DISTINCT_PROVIDER_COUNT_90D >= 4"
  ],
  "existing_features": [
    "COST_30D",
    "COST_90D",
    "COST_180D",
    "CLAIM_COUNT_30D",
    "CLAIM_COUNT_90D",
    "INPATIENT_COUNT_90D",
    "ED_COUNT_30D",
    "ED_COUNT_90D",
    "DISTINCT_PROVIDER_COUNT_90D",
    "DISTINCT_DIAGNOSIS_COUNT_90D"
  ],
  "allowed_feature_families": [
    "COST_ACCELERATION",
    "PROVIDER_FRAGMENTATION",
    "ED_ACCELERATION",
    "INPATIENT_COST_SHARE"
  ]
}
```

Do not send member identifiers, claim rows, or raw diagnosis-level records to Gemini.

---

# 16. Gemini Feature-Hypothesis Step

Use the Google Gen AI SDK configured for Vertex AI.

## 16.1 Gemini role

Gemini is a feature-hypothesis assistant.

It must:

- Analyze the residual summary
- Select up to three candidates from the allowed catalog
- Explain the hypothesis
- Return structured JSON
- Allow a `STOP` recommendation when no candidate is justified

Gemini must not:

- Generate arbitrary SQL
- Generate arbitrary Python
- Invent unavailable columns
- Claim that a feature will improve the model
- Approve its own recommendation
- Receive raw claim data

## 16.2 Structured output model

Use Pydantic:

```python
from pydantic import BaseModel, Field
from typing import Literal

class FeatureCandidate(BaseModel):
    feature_name: Literal[
        "COST_ACCELERATION",
        "PROVIDER_FRAGMENTATION",
        "ED_ACCELERATION",
        "INPATIENT_COST_SHARE"
    ]
    hypothesis: str
    expected_error_segment: str
    confidence: float = Field(ge=0.0, le=1.0)

class HypothesisResponse(BaseModel):
    action: Literal["PROPOSE", "STOP"]
    candidates: list[FeatureCandidate]
    stop_reason: str | None = None
```

## 16.3 Gemini prompt

Use a prompt similar to:

```text
You are assisting with an offline healthcare cost-model experiment using
synthetic data.

Your role is to propose testable derived features. You do not decide whether
a proposal is successful. A separate XGBoost holdout experiment will accept
or reject every proposal.

Review the residual evidence. Select no more than three feature names from the
provided allowed feature catalog. Do not invent fields, formulas, SQL, or
clinical conclusions.

Prioritize candidates that could explain systematic underprediction in the
identified segment. Return STOP when the evidence is insufficient.

Residual evidence:
<INSERT AGGREGATED JSON>

Follow the provided response schema.
```

## 16.4 Vertex AI client

Implement the current supported Google Gen AI SDK pattern:

```python
from google import genai

client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location,
)
```

Use structured output with the SDK’s current response-schema support.

Keep the model ID configurable through `GEMINI_MODEL`.

---

# 17. Feature Catalog

Gemini may select only from this catalog.

## 17.1 `COST_ACCELERATION`

Formula:

```text
COST_30D / max(COST_90D / 3, 1)
```

Meaning:

> Compare the most recent 30-day cost against the prior 90-day monthly average.

## 17.2 `PROVIDER_FRAGMENTATION`

Formula:

```text
DISTINCT_PROVIDER_COUNT_90D / max(CLAIM_COUNT_90D, 1)
```

Meaning:

> Estimate how distributed recent care is across different providers.

Do not present this as a clinical diagnosis or causal statement.

## 17.3 `ED_ACCELERATION`

Formula:

```text
ED_COUNT_30D / max(ED_COUNT_90D / 3, 1)
```

Meaning:

> Compare recent emergency utilization with the recent longer-term monthly rate.

## 17.4 `INPATIENT_COST_SHARE`

Requires an additional Snowflake base feature:

```text
INPATIENT_COST_90D
```

Formula:

```text
INPATIENT_COST_90D / max(COST_90D, 1)
```

Meaning:

> Measure the share of recent cost attributable to inpatient services.

## 17.5 Validation behavior

Reject a candidate when:

- Its required base column does not exist
- It is already present
- Its formula uses future data
- More than three candidates are returned
- It is not in the whitelist
- It produces only missing or infinite values

---

# 18. Challenger Experiment Loop

For every valid Gemini candidate:

1. Materialize the feature using Snowflake SQL.
2. Retrieve the updated feature table.
3. Use the exact same temporal split.
4. Use the exact same XGBoost hyperparameters.
5. Add only one new feature.
6. Train a challenger model.
7. Evaluate on the untouched test period.
8. Log the run to Vertex AI Experiments.
9. Write the results to Snowflake.
10. Apply the automated acceptance gate.

One-feature-at-a-time experiments make the result easy to explain.

---

# 19. Vertex AI Experiments

Use Vertex AI Experiments only for run tracking.

Create one experiment:

```text
carecost-fusion
```

Recommended runs:

```text
median-baseline
xgb-baseline
challenger-cost-acceleration
challenger-provider-fragmentation
challenger-ed-acceleration
challenger-inpatient-cost-share
```

Log parameters:

```text
model_type
feature_name
feature_count
random_seed
n_estimators
max_depth
learning_rate
train_start_date
train_end_date
validation_end_date
test_end_date
gemini_model
gemini_confidence
```

Log summary metrics:

```text
mae
rmse
rmsle
high_cost_recall
mae_improvement_pct
```

Vertex AI Experiments is the Google Cloud proof layer. It is not required to train the model remotely.

The coding agent must use the currently supported `google-cloud-aiplatform` experiment APIs and verify them against the installed SDK version.

---

# 20. Acceptance Gate

A Gemini candidate is not accepted because of confidence or explanation quality.

Use this evaluation:

```python
def decide_challenger(
    baseline: dict,
    challenger: dict,
    min_mae_improvement_pct: float,
    max_recall_drop: float,
) -> tuple[str, str]:
    mae_improvement_pct = (
        (baseline["mae"] - challenger["mae"])
        / baseline["mae"]
        * 100.0
    )

    recall_drop = (
        baseline["high_cost_recall"]
        - challenger["high_cost_recall"]
    )

    if mae_improvement_pct < min_mae_improvement_pct:
        return "REJECT", "MAE improvement did not meet threshold"

    if recall_drop > max_recall_drop:
        return "REVIEW", "Overall error improved but high-cost recall declined"

    return "ACCEPT", "Holdout metrics passed configured gates"
```

Possible decisions:

```text
ACCEPT
REJECT
REVIEW
```

Gemini must never set this field.

---

# 21. Snowflake Experiment Feedback

Write every result to `EXPERIMENT_RESULTS`.

Example:

```json
{
  "run_id": "challenger-cost-acceleration",
  "parent_run_id": "xgb-baseline",
  "run_type": "CHALLENGER",
  "feature_name": "COST_ACCELERATION",
  "mae": 3125.4,
  "high_cost_recall": 0.74,
  "mae_improvement_pct": 4.8,
  "decision": "ACCEPT",
  "decision_reason": "Holdout metrics passed configured gates",
  "gemini_hypothesis": "Recent cost growth may explain underprediction."
}
```

The notebook should finish with a Snowflake query:

```sql
SELECT
    RUN_ID,
    FEATURE_NAME,
    MAE,
    HIGH_COST_RECALL,
    MAE_IMPROVEMENT_PCT,
    DECISION,
    DECISION_REASON
FROM EXPERIMENT_RESULTS
ORDER BY CREATED_AT;
```

---

# 22. Notebook Sections

The notebook must contain these sections in this order:

## 1. Project overview

Explain the division between Snowflake, Vertex AI, Gemini, and XGBoost.

## 2. Configuration and authentication

Load `.env` and YAML configuration.

## 3. Generate synthetic claims

Generate deterministic data and show basic distributions.

## 4. Load claims into Snowflake

Use `write_pandas()` or Snowpark `Session.write_pandas()`.

## 5. Build Snowflake baseline features

Execute `01_base_features.sql`.

## 6. Retrieve member features

Load the de-identified member feature table into pandas.

## 7. Create temporal split

Show train, validation, and test date boundaries.

## 8. Train median and XGBoost baselines

Log XGBoost run to Vertex AI Experiments.

## 9. Write predictions to Snowflake

Populate `MODEL_PREDICTIONS`.

## 10. Discover residual segment

Create and write `RESIDUAL_SUMMARY`.

## 11. Call Gemini on Vertex AI

Display structured candidate hypotheses.

## 12. Validate candidates

Show accepted and rejected feature specifications.

## 13. Train challengers

Run one feature at a time.

## 14. Log experiments

Log parameters and metrics to Vertex AI Experiments.

## 15. Write experiment results to Snowflake

Populate `EXPERIMENT_RESULTS`.

## 16. Final comparison

Show table and charts.

## 17. Conclusions

State which Gemini suggestions were accepted or rejected and why.

---

# 23. Required Visuals

Create only these visuals:

1. Distribution of next-90-day cost
2. Actual versus predicted cost for baseline
3. Residual distribution
4. Baseline versus challenger MAE
5. Baseline versus challenger high-cost recall

Do not spend time building a dashboard.

Save final images in `artifacts/`.

---

# 24. Required Tests

## 24.1 Data generation test

Confirm:

- Claim IDs are unique
- Paid amount is nonnegative
- Service date is populated
- Required columns exist

## 24.2 Temporal leakage test

Confirm no historical feature reads an event on or after the index date.

Confirm target events occur only in the next 90-day window.

## 24.3 Feature catalog test

Confirm:

- Only whitelist features are accepted
- Division by zero is protected
- Unknown feature names are rejected
- Candidate list is limited to three

## 24.4 Evaluation gate test

Confirm:

- A worse challenger is rejected
- A sufficiently better challenger is accepted
- A challenger with recall degradation returns `REVIEW`

---

# 25. Error Handling

The project must fail clearly.

## Snowflake failures

Show actionable errors for:

- Missing credentials
- Missing warehouse
- Missing database permissions
- Failed table creation
- Failed pandas write

## Vertex AI failures

Show actionable errors for:

- Missing Google authentication
- Vertex AI API not enabled
- Invalid project or region
- Unsupported Gemini model
- Malformed structured response

## ML failures

Show actionable errors for:

- Empty training set
- No target variation
- Infinite candidate feature values
- Missing candidate base columns

Gemini failure must not prevent the baseline ML notebook from completing. The notebook should print that the agent step was skipped.

---

# 26. Security and Data Rules

- Use only synthetic data.
- Do not include names, addresses, dates of birth, or real identifiers.
- Do not send raw claim rows to Gemini.
- Do not send member IDs to Gemini.
- Send only aggregate residual evidence.
- Do not log passwords or credentials.
- Do not commit `.env`.
- Do not call the project HIPAA compliant.
- Do not make clinical recommendations.
- Describe outputs as financial forecasting experiments.

---

# 27. README Requirements

The README must open with:

```text
CareCost Fusion demonstrates a hybrid AI/ML workflow in which Snowflake
governs synthetic healthcare claims and computes features, Vertex AI tracks
experiments and provides Gemini, and XGBoost determines whether Gemini’s
feature hypotheses improve future-cost prediction.
```

README sections:

1. Why this project
2. Architecture
3. Why Snowflake and Vertex AI both exist
4. What is intentionally excluded
5. Dataset
6. Setup
7. Run instructions
8. Experiment flow
9. Results
10. Screenshots
11. ML methodology
12. Limitations
13. Resume-ready summary

The README must explicitly say:

> The same Gemini model is not called from two platforms. Snowflake is the governed data and feature layer; Vertex AI is the experiment and Gemini layer.

---

# 28. Resume-Ready Summary

Use this after successful completion:

> Built a hybrid healthcare AI/ML proof of concept using Snowflake and Google Vertex AI. Stored synthetic claims and generated temporal cost features in Snowflake, trained an XGBoost model for next-90-day cost, analyzed systematic residual errors, and used Gemini on Vertex AI to propose controlled feature hypotheses. Logged champion–challenger runs in Vertex AI Experiments and accepted or rejected each AI recommendation using holdout MAE and high-cost-member recall.

---

# 29. Suggested Public Title

**Gemini Proposes, XGBoost Proves, Snowflake Governs**

Subtitle:

**A Hybrid Snowflake and Vertex AI Experiment for Healthcare Cost Prediction**

---

# 30. Definition of Done

The project is complete when all of these are true:

- Synthetic claims are generated with a fixed seed.
- Claims are stored in Snowflake.
- Snowflake creates the baseline member feature table.
- A chronological ML split is used.
- XGBoost beats or is meaningfully compared with the median baseline.
- Predictions are written back to Snowflake.
- A residual segment is created without using Gemini.
- Only aggregate residual evidence is sent to Gemini.
- Gemini returns structured whitelist feature candidates.
- Each valid candidate runs as a separate challenger.
- Every run is logged to Vertex AI Experiments.
- Results are written to Snowflake.
- The automated gate returns `ACCEPT`, `REJECT`, or `REVIEW`.
- At least one weak AI recommendation can be shown as rejected.
- The README explains the hybrid-platform justification.
- The notebook runs from beginning to end without manual code editing.
- Unit tests pass.
- No excluded production components were added.

---

# 31. AI Coding Agent Instructions

The implementation will be created by an advanced AI coding agent.

The coding agent must follow these rules:

1. Read this entire specification before creating files.
2. Build the smallest implementation that satisfies the definition of done.
3. Do not redesign the architecture.
4. Do not add services listed under “Explicitly excluded.”
5. Prefer readable Python and SQL over frameworks.
6. Use type hints and docstrings.
7. Keep credentials in environment variables.
8. Run tests after each major stage.
9. Use the latest official Snowflake and Google Cloud SDK documentation when an API differs from this document.
10. Preserve the platform responsibility boundary.
11. Never call Snowflake Cortex or `AI_COMPLETE`.
12. Never let Gemini generate arbitrary executable SQL or Python.
13. Never allow Gemini to accept its own feature.
14. Keep raw claims in Snowflake.
15. Send only aggregate residual evidence to Gemini.
16. Produce a notebook that tells a clear ML story.
17. Stop after the MVP is complete.
18. Do not add optional UI unless all tests and required artifacts are complete.

---

# 32. Master Prompt for an AI Coding Agent

Copy the prompt below into the coding agent after attaching this specification.

```text
Build the complete repository described in the attached
“CareCost Fusion — Snowflake + Vertex AI Weekend MVP” specification.

Act as the principal ML engineer and implementation owner.

Requirements:
- Implement the repository end to end.
- Follow the exact platform responsibility split.
- Snowflake is the data, feature-engineering, residual-analysis, and result layer.
- Vertex AI provides Gemini and Vertex AI Experiments.
- XGBoost is the predictive model and validation mechanism.
- Do not use Snowflake AI_COMPLETE or Cortex.
- Do not add pipelines, endpoints, ADK, registries, BigQuery, Docker,
  Terraform, Streamlit, or production infrastructure.
- Generate deterministic synthetic claims.
- Seed a hidden cost-acceleration and provider-fragmentation relationship.
- Build baseline features in Snowflake while intentionally omitting those
  two derived features.
- Use a chronological train/validation/test split.
- Train and evaluate a median baseline and XGBoost baseline.
- Calculate predictions and residuals.
- Discover one interpretable high-underprediction segment.
- Send only aggregate residual evidence to Gemini through Vertex AI.
- Require Gemini to select up to three candidates from a strict whitelist.
- Materialize each valid feature in Snowflake.
- Train one challenger per feature using identical data splits and parameters.
- Log every run to Vertex AI Experiments.
- Write predictions, residual summaries, and experiment results to Snowflake.
- Apply deterministic ACCEPT, REJECT, or REVIEW gates.
- Create the notebook, SQL files, source modules, tests, README,
  environment template, configuration template, and requirements file.
- Run the tests and fix failures.
- Keep all code readable and concise.
- Do not fabricate execution results when credentials are unavailable.
- In that case, provide clear setup instructions and mocked unit tests,
  while leaving integration cells ready to run.

Work in this order:
1. Scaffold the repository.
2. Implement and test synthetic data generation.
3. Implement Snowflake setup and I/O.
4. Implement baseline feature SQL.
5. Implement temporal splitting and XGBoost evaluation.
6. Implement prediction and residual persistence.
7. Implement residual-segment discovery.
8. Implement the whitelist feature catalog and validators.
9. Implement Gemini structured output through Vertex AI.
10. Implement Vertex AI Experiments logging.
11. Implement challenger experiments and acceptance gates.
12. Assemble the end-to-end notebook.
13. Write the README.
14. Run all unit tests.
15. Review the repository against every definition-of-done item.

At completion, report:
- Files created
- Tests executed and results
- Any integration step requiring user credentials
- Exact commands to run the notebook
- Any definition-of-done item that could not be completed

Do not stop after generating a plan. Create the files and implementation.
```

---

# 33. Official Documentation References

Use these current official sources when implementing:

- Snowflake Python Connector with pandas:
  `https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-pandas`

- Snowflake Connector API, including `fetch_pandas_all()`:
  `https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api`

- Snowpark `Session.write_pandas()`:
  `https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.Session.write_pandas`

- Google Gen AI SDK:
  `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview`

- Gemini structured output:
  `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output`

- Vertex AI Experiments:
  `https://docs.cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments`

- Vertex AI Python SDK:
  `https://docs.cloud.google.com/python/docs/reference/aiplatform/latest`

---

# Final Design Principle

> **Snowflake owns the governed data and analytical evidence. Vertex AI owns Gemini and experiment metadata. XGBoost owns the prediction. The holdout test owns the truth.**
