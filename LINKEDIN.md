# LinkedIn post

*(Copy-paste ready. Swap the [link] before posting.)*

---

**Healthcare cost prediction is a machine-learning problem. So why is everyone trying to solve it with a chatbot?**

*Snowflake + Google Cloud – Vertex ML (Model Registry, Experiments, AI Eval for ML, Vertex Pipelines) — a cross-cloud build.*

A health plan needs to know which members will be expensive next quarter — to budget, and to get care teams to high-cost members early. It's a real, daily actuarial problem, not a hypothetical.

You *could* paste a member's history into an AI and ask. It'll answer confidently — and you'll have no way to check it, repeat it, or measure it. A prediction you're accountable for isn't a conversation. It's a measurement.

So I built it the way it should be built: **ML does the predicting, and AI makes the ML better.**

🔹 A gradient-boosted model — trained and tested on a proper time-based holdout (no peeking into the future) — predicts each member's next-90-day cost. ~2.4× more accurate than a naive baseline.

🔹 Residual analysis finds where it's systematically wrong. Then **Gemini** (on Google Vertex AI) proposes new features to fix those misses — from a fixed menu, as structured tool calls, never free-text.

🔹 Every proposed feature has to earn its place on unseen data. The AI gets no vote.

The result: Gemini proposed 3 features, confidently. The holdout test kept 1 and rejected 2 — including one the AI was **90% sure** about.

**Gemini proposes. XGBoost proves. Snowflake governs.**

The interesting part was never "we used an LLM." It's the combination — **measured ML as the backbone, AI to surface ideas a human might miss**, and nothing trusted until the data confirms it.

Built end-to-end on **Snowflake + Google Vertex AI** — function calling, Gen AI Evaluation, Experiments, Model Registry, a serving endpoint with explainability, and a Vertex AI Pipeline.

Full write-up + code 👉 [link]

#MachineLearning #MLOps #VertexAI #Snowflake #GenAI #HealthcareAnalytics #AI

---

## Shorter variant (~110 words)

**Healthcare cost prediction is an ML problem — not a chatbot problem.**

A health plan needs to know who'll be expensive next quarter. Ask an AI directly and you get a confident number you can't check, repeat, or measure.

So I built it right: **ML predicts, AI improves the ML.**

→ A gradient-boosted model forecasts each member's next-90-day cost (2.4× better than baseline, tested on unseen data).
→ Gemini on Vertex AI proposes new features for where the model misses.
→ Each idea must prove itself on held-out data. The AI gets no vote.

3 proposed, 1 kept, 2 rejected — including one Gemini was 90% sure about.

Gemini proposes. XGBoost proves. Snowflake governs. 👉 [link]

#MachineLearning #MLOps #VertexAI #Snowflake #GenAI
