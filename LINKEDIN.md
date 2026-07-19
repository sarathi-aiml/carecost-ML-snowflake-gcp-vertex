# LinkedIn post

*(Copy-paste ready. Swap the [link] before posting. ~220 words.)*

---

Everyone's using AI to *suggest* things now — features, segments, rules, next steps.

Here's the part nobody says out loud: a language model can't tell you whether its own suggestion is actually any good. It just sounds confident.

So I built a small project to enforce the discipline — healthcare cost prediction, on a hybrid **Snowflake + Google Vertex AI** stack:

🔹 **Snowflake** stores the claims and builds the features in SQL. The data never leaves the warehouse — the AI only ever sees an anonymous summary.

🔹 **Gemini** (on Vertex AI) looks at *where the model is going wrong* and proposes new features to fix it.

🔹 But Gemini doesn't get a vote. Every idea is trained into a real model and tested on data it has never seen.

🔹 Only ideas that **measurably improve the prediction** are kept.

The result: Gemini proposed 3 confident ideas. The holdout test accepted 1 and rejected 2 — including one Gemini was 90% sure about.

**Gemini proposes. XGBoost proves. Snowflake governs.**

The interesting engineering isn't "we used an LLM." It's the guardrails around it: propose within limits, prove on held-out data, keep the data governed in the warehouse. Two great platforms, each doing what it's best at.

Full write-up + code (function calling, Gen AI Evaluation, Model Registry, a Vertex AI Pipeline): [link]

#MachineLearning #MLOps #Snowflake #VertexAI #GenAI #HealthcareAnalytics #AI

---

## Shorter variant (for the character-conscious, ~90 words)

AI can *suggest* ideas. It can't tell you if they're right — it just sounds confident.

So I built a healthcare-cost predictor where an AI proposes and a test decides:

→ **Snowflake** holds the data + builds features (data never leaves the warehouse)
→ **Gemini** on Vertex AI proposes new features to fix the model's blind spots
→ A held-out test — not the AI — accepts or rejects each one

Gemini proposed 3 ideas. The test kept 1, rejected 2 (one it was 90% sure about).

Gemini proposes. XGBoost proves. Snowflake governs. 👉 [link]

#MLOps #Snowflake #VertexAI #GenAI
