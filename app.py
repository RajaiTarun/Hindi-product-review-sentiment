"""
app.py — Streamlit Web App for Hindi Product Review Sentiment Analyser
Phase 4 of SMAI Assignment 3 (T8.1)

Run with:  streamlit run app.py
"""

import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Page Config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Hindi Sentiment Analyser",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "saved_model"
MAX_LENGTH = 128

SENTIMENT_STYLE = {
    "Positive": {"emoji": "🟢", "hex": "#2ecc71", "bg": "#e8f8f0"},
    "Negative": {"emoji": "🔴", "hex": "#e74c3c", "bg": "#fdecea"},
}

SAMPLE_REVIEWS = [
    ("यह प्रोडक्ट बहुत अच्छा है, मुझे बहुत पसंद आया! क्वालिटी बेहतरीन है।", "Positive"),
    ("बहुत खराब क्वालिटी है, पैसे बर्बाद हो गए। बिल्कुल मत खरीदें।", "Negative"),
]

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.hero p  { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; }

/* Sentiment badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    border-radius: 50px;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

/* Metric card */
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border: 1px solid #e9ecef;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.metric-card .label { font-size: 0.85rem; color: #666; margin-top: 2px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: inherit;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e9ecef;
}

/* Confidence bar */
.conf-bar-wrap { margin: 6px 0; }
.conf-label { font-size: 0.88rem; color: #444; display: flex; justify-content: space-between; }
.conf-bar-bg { background: #e9ecef; border-radius: 6px; height: 10px; margin: 3px 0 8px; }
.conf-bar-fill { height: 10px; border-radius: 6px; transition: width 0.4s ease; }

/* Alert box */
.info-box {
    background: #eaf4ff;
    border-left: 4px solid #3498db;
    padding: 0.9rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model Loading (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading IndicBERT model…")
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load fine-tuned IndicBERT model + tokenizer once, cached for the session."""
    config_path = os.path.join(model_path, "label_config.json")
    if not os.path.exists(config_path):
        return None, None, None, None, None, None

    with open(config_path) as f:
        cfg = json.load(f)

    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    label2id = cfg["label2id"]
    num_labels = cfg["num_labels"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    return model, tokenizer, id2label, label2id, device, cfg


# ── Inference Helpers ──────────────────────────────────────────────────────
def predict_single(text: str, model, tokenizer, id2label, device) -> dict:
    if not text.strip():
        return {"label": "N/A", "confidence": 0.0, "probabilities": {}}

    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits

    probs   = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))

    return {
        "label":         id2label[pred_id],
        "confidence":    float(probs[pred_id]),
        "probabilities": {id2label[i]: float(probs[i]) for i in range(len(probs))},
    }


def predict_batch(df: pd.DataFrame, model, tokenizer, id2label, device,
                  text_col: str = "review", batch_size: int = 32) -> pd.DataFrame:
    texts   = df[text_col].fillna("").tolist()
    labels, confs, prob_rows = [], [], []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc   = tokenizer(batch, max_length=MAX_LENGTH, truncation=True,
                          padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits

        probs    = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)

        for pid, row in zip(pred_ids, probs):
            labels.append(id2label[int(pid)])
            confs.append(float(row[pid]))
            prob_rows.append({id2label[j]: float(row[j]) for j in range(len(row))})

    result             = df.copy()
    result["predicted_label"] = labels
    result["confidence"]      = confs
    prob_df = pd.DataFrame(prob_rows)
    for col in prob_df.columns:
        result[f"prob_{col}"] = prob_df[col].values

    return result


# ── UI Helpers ─────────────────────────────────────────────────────────────
def render_sentiment_badge(label: str, confidence: float):
    style = SENTIMENT_STYLE.get(label, {"emoji": "⚪", "hex": "#888", "bg": "#f0f0f0"})
    st.markdown(
        f"""<div class="badge" style="background:{style['bg']};color:{style['hex']};
            border:2px solid {style['hex']};">
            {style['emoji']}&nbsp;{label}
            &nbsp;&mdash;&nbsp;{confidence:.1%} confidence
        </div>""",
        unsafe_allow_html=True,
    )


def render_prob_bars(probabilities: dict):
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        style = SENTIMENT_STYLE.get(label, {"hex": "#888"})
        st.markdown(
            f"""<div class="conf-bar-wrap">
              <div class="conf-label"><span>{style.get('emoji','')} {label}</span><span>{prob:.1%}</span></div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill"
                     style="width:{prob*100:.1f}%;background:{style['hex']};"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_insights(result_df: pd.DataFrame):
    dist = result_df["predicted_label"].value_counts()

    col1, col2 = st.columns(2)
    total = len(result_df)
    pos = dist.get("Positive", 0)
    neg = dist.get("Negative", 0)

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#2ecc71">{pos/total:.0%}</div>
            <div class="label">🟢 Positive</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#e74c3c">{neg/total:.0%}</div>
            <div class="label">🔴 Negative</div></div>""", unsafe_allow_html=True)

    # Pie chart
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [SENTIMENT_STYLE.get(l, {}).get("hex", "#999") for l in dist.index]
    wedges, texts, autotexts = ax.pie(
        dist.values, labels=dist.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold", pad=12)
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=False)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    model, tokenizer, id2label, label2id, device, cfg = load_model(model_path)

    if model is None:
        st.error(f"❌ No model found at `{model_path}`.\n\nDownload `saved_model.zip` from your Kaggle notebook and extract it here.")
    else:
        st.success("✅ Model loaded")
        st.markdown(f"""
| Key | Value |
|-----|-------|
| **Base model** | `ai4bharat/indic-bert` |
| **Classes** | {cfg.get("num_labels", "?")} |
| **Max tokens** | {MAX_LENGTH} |
| **Test Accuracy** | {cfg.get("test_accuracy", 0):.2%} |
| **Test Macro-F1** | {cfg.get("test_macro_f1", 0):.4f} |
| **Device** | {str(device).upper()} |
""")

    st.markdown("---")
    st.markdown("""
### ℹ️ About
Built for **SMAI Assignment 3 (T8.1)**
Fine-tuned **IndicBERT** on the
[AI4Bharat IndicSentiment](https://huggingface.co/datasets/ai4bharat/IndicSentiment)
dataset for Hindi product review classification.
""")


# ── Hero Banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🇮🇳 Hindi Sentiment Analyser</h1>
  <p>Fine-tuned IndicBERT · AI4Bharat IndicSentiment Dataset · SMAI Assignment 3</p>
</div>
""", unsafe_allow_html=True)


# ── Guard: no model ────────────────────────────────────────────────────────
if model is None:
    st.warning(
        "⚠️ **Model not found.** Please ensure `saved_model/` exists in the same "
        "directory as `app.py`. Download `saved_model.zip` from your Kaggle notebook "
        "and extract it here."
    )
    st.stop()


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_insights = st.tabs([
    "🔍 Single Review", "📂 Bulk CSV Upload", "📈 Insights Dashboard"
])


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════════ #
with tab_single:
    st.markdown('<div class="section-header">Analyse a Hindi Review</div>', unsafe_allow_html=True)



    review_text = st.text_area(
        "Enter Hindi review text:",
        value=st.session_state.get("input_text", ""),
        height=130,
        placeholder="यहाँ हिंदी समीक्षा लिखें…",
        key="review_input",
    )

    col_btn, col_clear = st.columns([1, 5])
    analyse = col_btn.button("🔍 Analyse", type="primary", use_container_width=True)
    if col_clear.button("🗑️ Clear"):
        st.session_state["input_text"] = ""
        st.rerun()

    if analyse and review_text.strip():
        with st.spinner("Analysing…"):
            result = predict_single(review_text, model, tokenizer, id2label, device)

        st.markdown("---")
        
        st.markdown("#### Prediction")
        render_sentiment_badge(result["label"], result["confidence"])

        st.markdown("#### Confidence Breakdown")
        render_prob_bars(result["probabilities"])

    elif analyse:
        st.warning("Please enter some review text first.")


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 2 — Bulk CSV Upload
# ══════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.markdown('<div class="section-header">Bulk CSV Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
Upload a CSV file with a column named **`review`** containing Hindi product reviews.
The app will classify each review and let you download the results.
""")

    # Show expected format
    with st.expander("📋 Expected CSV format"):
        st.dataframe(pd.DataFrame({
            "review": [
                "यह प्रोडक्ट बहुत अच्छा है",
                "बहुत खराब क्वालिटी है",
                "ठीक-ठाक है",
            ]
        }), use_container_width=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # Auto-detect review column
        text_col = None
        for candidate in ["review", "text", "INDIC REVIEW", "indic_review", "Review"]:
            if candidate in df_upload.columns:
                text_col = candidate
                break

        if text_col is None:
            text_col = st.selectbox(
                "Which column contains the Hindi reviews?",
                options=df_upload.columns.tolist(),
            )

        st.markdown(f"**{len(df_upload):,} reviews detected** in column `{text_col}`.")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Analysis", type="primary"):
            with st.spinner(f"Classifying {len(df_upload):,} reviews…"):
                result_df = predict_batch(
                    df_upload, model, tokenizer, id2label, device, text_col=text_col
                )

            st.success("✅ Classification complete!")
            st.dataframe(
                result_df[["review" if text_col == "review" else text_col,
                            "predicted_label", "confidence"]].rename(
                    columns={text_col: "review"}
                ),
                width=True,
            )

            # ── Inline insights ──
            st.markdown("---")
            st.markdown("#### 📊 Quick Insights")
            render_insights(result_df)

            # Avg confidence
            avg_conf = result_df["confidence"].mean()
            st.metric("Average confidence", f"{avg_conf:.2%}")

            # ── Download ──
            st.markdown("---")
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="sentiment_results.csv",
                mime="text/csv",
                type="primary",
            )


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 3 — Insights Dashboard (demo with sample data)
# ══════════════════════════════════════════════════════════════════════════ #
with tab_insights:
    st.markdown('<div class="section-header">Insights Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
This tab shows a live demo dashboard using built-in sample reviews.
Upload your own CSV in the **Bulk CSV Upload** tab to see real insights.
""")

    # Build demo data
    demo_reviews = [
        ("यह प्रोडक्ट बेहतरीन है, मैंने अभी तक ऐसा नहीं देखा!", "Positive"),
        ("डिलीवरी बहुत तेज़ थी और पैकेजिंग भी अच्छी थी।", "Positive"),
        ("बहुत बुरा अनुभव रहा, दोबारा नहीं खरीदूंगा।", "Negative"),
        ("बढ़िया उत्पाद, परिवार को बहुत पसंद आया।", "Positive"),
        ("सामान खराब निकला, रिटर्न करना पड़ा।", "Negative"),
        ("बहुत अच्छी क्वालिटी, पैसे वसूल।", "Positive"),
    ]

    with st.spinner("Running demo inference…"):
        demo_df = pd.DataFrame(demo_reviews, columns=["review", "true_label"])
        demo_result = predict_batch(
            demo_df, model, tokenizer, id2label, device, text_col="review"
        )

    # Summary metrics
    st.markdown("#### Summary Metrics")
    render_insights(demo_result)

    st.markdown("---")

    # Per-review table
    st.markdown("#### Sample Predictions")
    display_df = demo_result[["review", "predicted_label", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].map("{:.1%}".format)

    def style_label(val):
        style = SENTIMENT_STYLE.get(val, {})
        return f"color: {style.get('hex', '#000')}; font-weight: bold"

    st.dataframe(
        display_df.style.map(style_label, subset=["predicted_label"]),
        width=True,
    )

    st.markdown("---")

    # Confidence distribution histogram
    st.markdown("#### Confidence Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.hist(demo_result["confidence"], bins=10, color="#3498db", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Prediction Confidence", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 1)
    fig2.patch.set_alpha(0)
    st.pyplot(fig2, use_container_width=True)
