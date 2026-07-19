"""CareCost Fusion — Streamlit demo UI (live-only).

Run:  streamlit run app.py
Requires live Snowflake (source) + Vertex AI (Gemini, Gen AI Eval, endpoint). There
are no offline fallbacks — the point is that this genuinely runs on Snowflake + Vertex.
Set GOOGLE_CLOUD_PROJECT and have ADC + ~/.snowflake/connections.toml configured.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import BASELINE_MODEL_FEATURES  # noqa: E402
from feature_catalog import materialize  # noqa: E402
from modeling import temporal_split, make_model, TARGET  # noqa: E402
import pipeline as P  # noqa: E402

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")  # set your GCP project via env
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SNOW, VERTEX, INK, MUTED = "#29B5E8", "#1A73E8", "#0E1526", "#5A6478"
ACCEPT, REJECT, REVIEW = "#17935B", "#D0453B", "#C98A00"

st.set_page_config(page_title="CareCost Fusion", page_icon="🩺", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');
:root{--ink:#0E1526;--bg:#FBFCFE;--surface:#fff;--line:#E4E8F0;--muted:#5A6478;
 --snow:#29B5E8;--vertex:#1A73E8;--accept:#17935B;--reject:#D0453B;--review:#C98A00;}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,sans-serif;color:var(--ink);}
.stApp{background:var(--bg);}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif;letter-spacing:-.02em;}
.block-container{padding-top:3.6rem;max-width:1200px;}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:#0E1526;}
section[data-testid="stSidebar"] *{color:#E7ECF5 !important;}
.hero-eyebrow{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.hero-title{font-family:'Space Grotesk';font-weight:700;font-size:40px;line-height:1.05;margin:0;}
.hero-sub{color:var(--muted);font-size:15px;margin-top:10px;max-width:760px;}
.hero-tag{font-family:'Space Grotesk';font-weight:600;font-size:14px;margin-top:14px;color:var(--ink);}
.hero-tag b{color:var(--vertex);}
.ribbon{display:flex;align-items:stretch;gap:0;margin:22px 0 6px;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:var(--surface);}
.stage{flex:1;padding:14px 14px;border-right:1px solid var(--line);position:relative;}
.stage:last-child{border-right:none;}
.stage .n{font-family:'JetBrains Mono';font-size:11px;color:var(--muted);}
.stage .t{font-family:'Space Grotesk';font-weight:600;font-size:14px;margin-top:2px;}
.stage .p{display:inline-block;margin-top:8px;font-family:'JetBrains Mono';font-size:10px;letter-spacing:.05em;padding:2px 8px;border-radius:999px;font-weight:600;}
.p.snow{background:rgba(41,181,232,.12);color:#0B7FB0;}
.p.vertex{background:rgba(26,115,232,.12);color:#1558c0;}
.p.xgb{background:rgba(14,21,38,.07);color:#3a4a63;}
.stage.snowbg{background:linear-gradient(180deg,rgba(41,181,232,.05),transparent);}
.stage.vertexbg{background:linear-gradient(180deg,rgba(26,115,232,.05),transparent);}
.seam{width:3px;background:linear-gradient(180deg,var(--snow),var(--vertex));}
.seam-lbl{text-align:center;font-family:'JetBrains Mono';font-size:11px;color:var(--muted);margin:2px 0 18px;}
.sec{display:flex;align-items:baseline;gap:12px;margin:2px 0 14px;}
.sec .num{font-family:'JetBrains Mono';font-weight:600;font-size:13px;color:var(--vertex);}
.sec .title{font-family:'Space Grotesk';font-weight:600;font-size:20px;}
.sec .tag{margin-left:auto;font-family:'JetBrains Mono';font-size:11px;padding:3px 10px;border-radius:999px;}
.tag.snow{background:rgba(41,181,232,.12);color:#0B7FB0;}
.tag.vertex{background:rgba(26,115,232,.12);color:#1558c0;}
.tag.xgb{background:rgba(14,21,38,.07);color:#3a4a63;}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 4px;}
.kpi{flex:1;min-width:150px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;}
.kpi .lab{font-family:'JetBrains Mono';font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.kpi .val{font-family:'JetBrains Mono';font-weight:600;font-size:26px;margin-top:9px;line-height:1;}
.kpi .sub{font-size:12px;color:var(--muted);margin-top:6px;}
.kpi.good .val{color:var(--accept);}
.drow{display:flex;align-items:center;gap:14px;padding:12px 14px;border:1px solid var(--line);border-radius:12px;margin-bottom:8px;background:var(--surface);}
.drow .feat{font-family:'JetBrains Mono';font-weight:600;font-size:13px;flex:1;}
.drow .imp{font-family:'JetBrains Mono';font-size:14px;width:90px;text-align:right;}
.badge{font-family:'JetBrains Mono';font-size:11px;font-weight:600;padding:4px 12px;border-radius:999px;color:#fff;}
.b-ACCEPT{background:var(--accept);}.b-REJECT{background:var(--reject);}.b-REVIEW{background:var(--review);}
.foot{color:var(--muted);font-size:12.5px;margin-top:6px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

WALK = {
    "01": ("Snowflake builds member features with SQL time-windows; only aggregate evidence ever leaves.",
           "Compute goes to the data — Snowflake owns feature engineering + governance, nothing sensitive exported.",
           "30/90/180-day windows with a strict leakage rule (history < index date, target in the next 90d). The baseline omits the derived ratios by design."),
    "02": ("Trains XGBoost on log-cost and compares to a median baseline.",
           "A random split leaks the future; cost is right-skewed → log-target; median is an honest floor.",
           "Chronological 60/20/20 split on index dates. XGBoost ≈ $32.8k MAE vs $78k median (~2.4× better)."),
    "03": ("A shallow decision tree isolates the worst-underprediction segment and packages an aggregate summary.",
           "An interpretable segment is what an LLM can reason about — and aggregate-only is the privacy boundary.",
           "Depth-3 tree on the positive-residual tail. Only counts/means + conditions cross to Gemini — no member rows or IDs."),
    "04": ("Gemini calls a propose_feature tool (whitelisted) up to 3× — or stops.",
           "Tool/function calling is GenAI engineering, not a SQL inference call — so it belongs on Vertex, not Cortex.",
           "Google Gen AI SDK, vertexai=True, tool_config mode ANY. Function-call args parse into typed candidates."),
    "05": ("Vertex Gen AI Evaluation scores each hypothesis with a custom pointwise metric.",
           "The thesis is 'the LLM's confidence doesn't count' — so we evaluate the hypotheses independently.",
           "EvalTask with a hypothesis_plausibility rubric (autorater = Gemini)."),
    "05g": ("Trains one challenger per accepted feature and applies a deterministic gate.",
            "One feature at a time = explainable. Gemini never sets the decision — the holdout owns the truth.",
            "ACCEPT if MAE improves ≥1% and high-cost recall holds; else REVIEW/REJECT. COST_ACCELERATION wins at +1.62%."),
    "06": ("The champion is registered in Vertex Model Registry and served from an endpoint; attributions explain a prediction.",
           "Real-time serving + a 'why did it say $X' trust moment for a customer.",
           "Model Registry + Online Endpoint; sampled-Shapley attributions via Vertex Explainable AI."),
}


def walknote(key, show):
    if not show or key not in WALK:
        return
    what, why, how = WALK[key]
    with st.expander("ℹ️  What · Why · How"):
        st.markdown(f"**What** — {what}\n\n**Why** — {why}\n\n**How** — {how}")


def kpis(items):
    cells = "".join(
        f'<div class="kpi {i.get("cls","")}"><div class="lab">{i["lab"]}</div>'
        f'<div class="val">{i["val"]}</div>'
        + (f'<div class="sub">{i["sub"]}</div>' if i.get("sub") else "") + "</div>"
        for i in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def section(num, title, platform):
    tag = {"snowflake": ("snow", "❄ Snowflake"), "vertex": ("vertex", "▲ Vertex AI"),
           "xgboost": ("xgb", "◆ XGBoost")}[platform]
    st.markdown(f'<div class="sec"><span class="num">{num}</span>'
                f'<span class="title">{title}</span>'
                f'<span class="tag {tag[0]}">{tag[1]}</span></div>', unsafe_allow_html=True)


def bar(df, x, y, color=None, scale=None, horizontal=False):
    enc = {"x": alt.X(f"{y}:Q" if horizontal else f"{x}:N", title=None,
                      sort=None if horizontal else list(df[x])),
           "y": alt.Y(f"{x}:N" if horizontal else f"{y}:Q", title=None, sort="-x" if horizontal else None)}
    c = alt.Chart(df).mark_bar(cornerRadius=4, size=26)
    if color and scale:
        c = c.encode(color=alt.Color(f"{color}:N", scale=scale, legend=None), **enc)
    else:
        c = c.encode(color=alt.value(VERTEX), **enc)
    return c.properties(height=max(140, 34 * len(df)) if horizontal else 230).configure_view(
        stroke=None).configure_axis(labelFont="JetBrains Mono", labelColor=MUTED,
        titleColor=MUTED, grid=False, domainColor="#E4E8F0", tickColor="#E4E8F0")


# ----------------------------- cached compute (live only) -----------------------------
@st.cache_data(show_spinner="Loading claims + building features in Snowflake…")
def build(member_count: int, seed: int):
    from snowflake_io import get_connection
    conn = get_connection()
    try:
        df = P.features_from_snowflake(conn, member_count=member_count, seed=seed)
    finally:
        conn.close()
    base = P.train_baseline(df)
    return df, base, P.residual_segment(base)


@st.cache_resource(show_spinner=False)
def champion_model(member_count: int, seed: int, champ_features: tuple, _df: pd.DataFrame):
    """Train the champion on the gate-selected feature set (keyed by champ_features)."""
    df = _df.copy()
    for f in champ_features:
        if f not in df.columns:
            df[f] = materialize(df, f)
    split = temporal_split(df)
    m = make_model(P.DEFAULT_MODEL_CFG)
    m.fit(split.train[list(champ_features)], np.log1p(split.train[TARGET].to_numpy()))
    return m, split


# ----------------------------- sidebar -----------------------------
st.sidebar.markdown("### 🩺 CareCost Fusion")
st.sidebar.caption("Gemini proposes · XGBoost proves · Snowflake governs")
member_count = st.sidebar.slider("Members", 500, 2000, 2000, step=500)
seed = int(st.sidebar.number_input("Seed", value=42, step=1))
show_walk = st.sidebar.toggle("Show walkthrough notes", value=True,
                              help="Per-step What · Why · How — narrate the flow in your demo.")
st.sidebar.divider()
st.sidebar.markdown(f"**Project** `{PROJECT}`  \n**Region** `{LOCATION}`  \n**Model** `{GEMINI_MODEL}`")
st.sidebar.caption("Live only — requires Snowflake + Vertex creds.")

# ----------------------------- hero + ribbon -----------------------------
st.markdown(
    '<div class="hero-eyebrow">Hybrid MLOps · Snowflake × Google Vertex AI</div>'
    '<div class="hero-title">CareCost Fusion</div>'
    '<div class="hero-sub">A language model proposes healthcare-cost features; a holdout ML gate — '
    'not the model — decides. Snowflake governs the data; Vertex AI is the ML control plane. '
    'Synthetic data, no PHI.</div>'
    '<div class="hero-tag">Gemini <b>proposes</b>. XGBoost <b>proves</b>. Snowflake <b>governs</b>.</div>',
    unsafe_allow_html=True)

stages = [("01", "Claims + features", "snow", "snowbg"), ("02", "XGBoost baseline", "xgb", ""),
          ("03", "Residual segment", "snow", "snowbg"), ("04", "Gemini hypotheses", "vertex", "vertexbg"),
          ("05", "Holdout gate", "xgb", ""), ("06", "Endpoint + XAI", "vertex", "vertexbg")]
plat_lbl = {"snow": "❄ Snowflake", "vertex": "▲ Vertex", "xgb": "◆ XGBoost"}
chips = ""
for i, (n, t, p, bg) in enumerate(stages):
    chips += (f'<div class="stage {bg}"><div class="n">{n}</div><div class="t">{t}</div>'
              f'<span class="p {p}">{plat_lbl[p]}</span></div>')
    if i == 2:
        chips += '<div class="seam"></div>'
st.markdown(f'<div class="ribbon">{chips}</div>', unsafe_allow_html=True)
st.markdown('<div class="seam-lbl">← Snowflake governs data &nbsp;·&nbsp; only '
            '<b>aggregate residual JSON</b> crosses the seam &nbsp;·&nbsp; Vertex AI runs the ML →</div>',
            unsafe_allow_html=True)

df, base, segment = build(member_count, seed)

# ----------------------------- 1. data plane -----------------------------
with st.container(border=True):
    section("01", "Snowflake data plane", "snowflake")
    walknote("01", show_walk)
    kpis([{"lab": "Members", "val": f"{df.MEMBER_ID.nunique():,}"},
          {"lab": "Feature rows", "val": f"{len(df):,}"},
          {"lab": "Feature source", "val": "Snowflake SQL"},
          {"lab": "Target", "val": "next 90-day $"}])
    st.markdown('<div class="foot">Built <b>in-warehouse via Snowflake SQL</b> (`01_base_features.sql`). '
                'Baseline omits the derived ratios by design. Only aggregate evidence leaves the warehouse.</div>',
                unsafe_allow_html=True)
    with st.expander("Preview MEMBER_FEATURES_BASE"):
        st.dataframe(df.head(8), use_container_width=True)

# ----------------------------- 2. baseline -----------------------------
with st.container(border=True):
    section("02", "XGBoost baseline vs. median", "xgboost")
    walknote("02", show_walk)
    lift = (1 - base.metrics["mae"] / base.median_metrics["mae"]) * 100
    kpis([{"lab": "Median MAE", "val": f"${base.median_metrics['mae']:,.0f}"},
          {"lab": "XGBoost MAE", "val": f"${base.metrics['mae']:,.0f}", "sub": f"{lift:.0f}% better than median", "cls": "good"},
          {"lab": "High-cost recall", "val": f"{base.metrics['high_cost_recall']:.3f}"},
          {"lab": "RMSLE", "val": f"{base.metrics['rmsle']:.2f}"}])

# ----------------------------- 3. residual segment -----------------------------
with st.container(border=True):
    section("03", "Residual segment — decision tree, no Gemini", "snowflake")
    walknote("03", show_walk)
    c1, c2 = st.columns([1, 1])
    with c1:
        kpis([{"lab": "Segment members", "val": f"{segment['member_count']:,}"},
              {"lab": "Mean underprediction", "val": f"${segment['mean_residual']:,.0f}"}])
        st.markdown('<div class="foot">Conditions: ' + " · ".join(segment["conditions"]) + "</div>",
                    unsafe_allow_html=True)
    with c2:
        st.caption("Aggregate JSON sent to Gemini — no member rows, no IDs:")
        st.json({k: segment[k] for k in ("segment_description", "member_count", "mean_residual", "conditions")})

# ----------------------------- 4. Gemini -----------------------------
with st.container(border=True):
    section("04", "Gemini on Vertex — function calling + Gen AI Evaluation", "vertex")
    walknote("04", show_walk)
    walknote("05", show_walk)
    st.markdown('<div class="foot">Gemini doesn\'t free-text — it <b>calls a <code>propose_feature</code> tool</b> '
                'bound to the whitelist, and each hypothesis is scored by the <b>Vertex Gen AI Evaluation Service</b>.</div>',
                unsafe_allow_html=True)
    if st.button("▶  Run Gemini hypothesis step", type="primary"):
        with st.spinner("Gemini calling propose_feature(...) + Gen AI Evaluation…"):
            st.session_state["gemini"] = P.gemini_step(
                segment, project=PROJECT, location=LOCATION, model=GEMINI_MODEL, available_columns=df.columns)
    if "gemini" in st.session_state:
        g = st.session_state["gemini"]
        st.caption(f"Gen AI Evaluation: **{g['eval_scores'].get('source')}**")
        st.dataframe(pd.DataFrame([{
            "tool call": f"propose_feature({c.feature_name})", "Gemini confidence": c.confidence,
            "plausibility (Gen AI Eval)": g["eval_scores"].get(c.feature_name, "—"),
            "hypothesis": c.hypothesis} for c in g["response"].candidates]), use_container_width=True)

# ----------------------------- 5. gate -----------------------------
if "gemini" in st.session_state:
    g = st.session_state["gemini"]
    with st.container(border=True):
        section("05", "Challengers + deterministic gate", "xgboost")
        walknote("05g", show_walk)
        hyp = {c.feature_name: c.hypothesis for c in g["response"].candidates}
        with st.spinner("Training one challenger per accepted feature…"):
            results = P.challenger_step(df, base, g["accepted"], hyp_by_name=hyp)
        st.session_state["results"] = results  # section 6 serves the gate-ACCEPTED champion
        for r in results:
            st.markdown(
                f'<div class="drow"><span class="feat">{r["feature_name"]}</span>'
                f'<span class="imp">{r["mae_improvement_pct"]:+.2f}%</span>'
                f'<span class="badge b-{r["decision"]}">{r["decision"]}</span></div>',
                unsafe_allow_html=True)
        cdf = pd.DataFrame([{"feature": r["feature_name"], "imp": round(r["mae_improvement_pct"], 2),
                             "decision": r["decision"]} for r in results])
        scale = alt.Scale(domain=["ACCEPT", "REJECT", "REVIEW"], range=[ACCEPT, REJECT, REVIEW])
        st.altair_chart(bar(cdf, "feature", "imp", color="decision", scale=scale), use_container_width=True)
        st.markdown('<div class="foot">Gate: ACCEPT if MAE improves ≥1% and high-cost recall holds. '
                    '<b>Gemini never sets this field — the holdout owns the truth.</b></div>', unsafe_allow_html=True)

# ----------------------------- 6. serving + Explainable AI (live endpoint) -----------------------------
with st.container(border=True):
    section("06", "Champion serving + Explainable AI", "vertex")
    walknote("06", show_walk)
    # Champion = baseline + the feature(s) the gate ACCEPTed. Until the gate has run,
    # serve the baseline; never hardcode a winner (the holdout decides).
    accepted_extra = [r["feature_name"] for r in st.session_state.get("results", [])
                      if r["decision"] == "ACCEPT"]
    champ_features = tuple(BASELINE_MODEL_FEATURES + accepted_extra)
    st.caption("Champion features: baseline"
               + (f" + {', '.join(accepted_extra)}" if accepted_extra
                  else " (run the gate above to add ACCEPTed features)"))
    model, split = champion_model(member_count, seed, champ_features, df)
    idx = st.selectbox("Test member", range(min(20, len(split.test))),
                       format_func=lambda i: str(split.test.iloc[i]["MEMBER_ID"]))
    row = split.test.iloc[int(idx)]
    instance = [float(row[c]) for c in champ_features]
    pred = float(np.expm1(model.predict(split.test[list(champ_features)].iloc[[int(idx)]])[0]))
    c1, c2 = st.columns([1, 1.3])
    with c1:
        kpis([{"lab": "Predicted 90-day $", "val": f"${pred:,.0f}"},
              {"lab": "Actual", "val": f"${row[TARGET]:,.0f}"}])
    from vertex_prediction import find_endpoint, explain
    endpoint = find_endpoint("carecost-champion", PROJECT, LOCATION)
    with c2:
        if endpoint is None:
            st.info("No live endpoint. Deploy the champion (`vertex_prediction.deploy_champion`, see DEMO.md) "
                    "to serve predictions + Vertex Explainable AI here.")
        else:
            attrs = explain(endpoint, [instance], list(champ_features))[0]
            adf = pd.DataFrame({"feature": list(attrs), "attribution": [round(v, 3) for v in attrs.values()]})
            adf = adf.reindex(adf.attribution.abs().sort_values(ascending=False).index).head(8)
            adf["sign"] = np.where(adf.attribution >= 0, "raises cost", "lowers cost")
            st.caption("**Vertex Explainable AI** — sampled-Shapley attributions (live endpoint)")
            sc = alt.Scale(domain=["raises cost", "lowers cost"], range=[VERTEX, "#9AA6BC"])
            st.altair_chart(bar(adf, "feature", "attribution", color="sign", scale=sc, horizontal=True),
                            use_container_width=True)

st.divider()
st.markdown(
    f'<div class="foot">🔗 <a href="https://console.cloud.google.com/vertex-ai/experiments/locations/{LOCATION}/experiments/carecost-fusion/runs?project={PROJECT}">Vertex AI Experiments</a> '
    f'· <a href="https://console.cloud.google.com/vertex-ai/models?project={PROJECT}">Model Registry</a> '
    f'· Snowflake <code>CARECOST_DEMO.ANALYTICS</code></div>', unsafe_allow_html=True)
