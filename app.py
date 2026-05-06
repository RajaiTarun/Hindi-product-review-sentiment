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
    "Positive": {"emoji": "🟢", "hex": "#34d399", "bg": "rgba(52,211,153,0.12)",
                 "border": "rgba(52,211,153,0.3)", "cls": "badge-positive"},
    "Negative": {"emoji": "🔴", "hex": "#fb7185", "bg": "rgba(251,113,133,0.12)",
                 "border": "rgba(251,113,133,0.3)", "cls": "badge-negative"},
}

# ── Load External CSS ─────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found — UI may look basic.")

load_css()


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
    style = SENTIMENT_STYLE.get(label, {"emoji": "⚪", "cls": "", "hex": "#888"})
    st.markdown(
        f"""<div class="sentiment-badge {style['cls']}">
            {style['emoji']}&nbsp;{label}
            &nbsp;&mdash;&nbsp;{confidence:.1%} confidence
        </div>""",
        unsafe_allow_html=True,
    )


def render_prob_bars(probabilities: dict):
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        style = SENTIMENT_STYLE.get(label, {"hex": "#888", "emoji": ""})
        st.markdown(
            f"""<div class="prob-bar-wrap">
              <div class="prob-bar-header">
                <span>{style.get('emoji','')} {label}</span>
                <span>{prob:.1%}</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill"
                     style="width:{prob*100:.1f}%;background:linear-gradient(90deg,{style['hex']},{style['hex']}cc);"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_insights(result_df: pd.DataFrame):
    dist = result_df["predicted_label"].value_counts()
    total = len(result_df)
    pos = dist.get("Positive", 0)
    neg = dist.get("Negative", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#34d399">{pos/total:.0%}</div>
            <div class="label">🟢 Positive</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#fb7185">{neg/total:.0%}</div>
            <div class="label">🔴 Negative</div></div>""", unsafe_allow_html=True)
    with col3:
        avg_conf = result_df["confidence"].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="color:#a78bfa">{avg_conf:.1%}</div>
            <div class="label">⚡ Avg Confidence</div></div>""", unsafe_allow_html=True)

    # Pie chart — dark themed
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    colors = [SENTIMENT_STYLE.get(l, {}).get("hex", "#999") for l in dist.index]
    wedges, texts, autotexts = ax.pie(
        dist.values, labels=dist.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 12, "color": "#e2e8f0"},
        wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold",
                 pad=14, color="#e2e8f0")
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
  <div class="subtitle">Fine-tuned IndicBERT · AI4Bharat IndicSentiment · SMAI Assignment 3</div>
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

# ── Top Themes Helper ──────────────────────────────────────────────────────
THEME_KEYWORDS = {
    "📦 Delivery & Packaging": ["डिलीवरी", "पैकेजिंग", "शिपिंग", "तेज़", "समय", "डिलीवर", "लेट", "देर", "पैक", "बॉक्स"],
    "✨ Build & Quality":      ["क्वालिटी", "गुणवत्ता", "बनावट", "मजबूत", "खराब", "टूट", "बेकार", "अच्छा", "मटेरियल", "सामग्री", "प्रोडक्ट"],
    "💰 Value & Price":        ["पैसे", "कीमत", "वसूल", "महंगा", "सस्ता", "प्राइस", "मूल्य", "बजट", "ऑफर", "डिस्काउंट"],
    "📞 Service & Returns":    ["सपोर्ट", "सेवा", "कस्टमर", "रितर्न", "वापस", "सर्विस", "रिफंड", "कॉल", "मदद", "शिकायत", "बदल"],
    "🎨 Design & Looks":       ["सुंदर", "दिखने", "कलर", "रंग", "डिज़ाइन", "लुक", "स्टाइल", "खूबसूरत", "आकर्षक"],
    "🔋 Electronics & Tech":   ["कैमरा", "डिस्प्ले", "स्क्रीन", "बैटरी", "चार्जिंग", "स्लो", "फास्ट", "साउंड", "आवाज़", "फ़ोन", "लैपटॉप"],
    "👗 Fit & Comfort":        ["साइज", "फिटिंग", "आरामदायक", "कपड़ा", "पहनने", "टाइट", "लूज", "कम्फर्ट", "जूते", "साड़ी"],
    "📖 Books & Content":      ["किताब", "पेज", "कहानी", "लिखावट", "प्रिंट", "पढ़ने", "ज्ञान", "लेखक", "बुक", "कवर"],
    "😊 Ease of Use":          ["आसान", "उपयोग", "इस्तेमाल", "कठिन", "सिंपल", "सेटअप", "चलाने", "यूज़र", "काम"],
}

def extract_top_themes(df: pd.DataFrame, text_col: str = "review") -> dict:
    """Count how many reviews mention each theme keyword."""
    counts = {}
    all_text = " ".join(df[text_col].fillna("").tolist())
    for theme, keywords in THEME_KEYWORDS.items():
        counts[theme] = sum(all_text.count(kw) for kw in keywords)
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def render_top_themes(df: pd.DataFrame, text_col: str = "review", top_n: int = 5):
    theme_counts = extract_top_themes(df, text_col)
    
    # Filter out 0 counts and sort by highest
    top = {k: v for k, v in theme_counts.items() if v > 0}
    top = dict(sorted(top.items(), key=lambda x: -x[1])[:top_n])
    
    if not top:
        st.info("No theme keywords detected in the reviews.")
        return
        
    st.markdown("#### 🏷️ Top Themes Mentioned")
    max_count = max(top.values()) or 1
    for theme, count in top.items():
        pct = count / max_count
        color = "#a78bfa"
        st.markdown(
            f"""<div class="prob-bar-wrap">
              <div class="prob-bar-header"><span>{theme}</span><span>{count} mention{'s' if count!=1 else ''}</span></div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:{pct*100:.1f}%;background:linear-gradient(90deg,{color},{color}99);"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════════ #
with tab_single:
    st.markdown('<div class="section-header">✨ Analyse a Hindi Review</div>',
                unsafe_allow_html=True)

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

        # Results inside a glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Prediction")
        render_sentiment_badge(result["label"], result["confidence"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Confidence Breakdown")
        render_prob_bars(result["probabilities"])
        st.markdown("</div>", unsafe_allow_html=True)

    elif analyse:
        st.warning("Please enter some review text first.")


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 2 — Bulk CSV Upload
# ══════════════════════════════════════════════════════════════════════════ #
with tab_batch:
    st.markdown('<div class="section-header">📂 Bulk CSV Analysis</div>',
                unsafe_allow_html=True)

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

            # Store results in session state so the Dashboard tab can use them
            st.session_state["batch_result_df"] = result_df
            st.session_state["batch_text_col"] = text_col

            st.dataframe(
                result_df[[text_col, "predicted_label", "confidence"]].rename(
                    columns={text_col: "review"}
                ),
                use_container_width=True,
            )

            # ── Inline insights ──
            st.markdown("---")
            st.markdown("#### 📊 Quick Insights")
            render_insights(result_df)

            # ── Top Themes ──
            st.markdown("---")
            render_top_themes(result_df, text_col)

            # ── Go to Dashboard hint ──
            st.info("📈 Switch to the **Insights Dashboard** tab to see the full analytics for this upload!")

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
#  TAB 3 — Insights Dashboard
# ══════════════════════════════════════════════════════════════════════════ #
with tab_insights:
    st.markdown('<div class="section-header">📈 Insights Dashboard</div>',
                unsafe_allow_html=True)

    # Prefer live batch results; fall back to demo data
    if "batch_result_df" in st.session_state:
        dashboard_df = st.session_state["batch_result_df"]
        dash_text_col = st.session_state.get("batch_text_col", "review")
        st.success(f"✅ Showing analytics for your uploaded CSV ({len(dashboard_df):,} reviews)")
    else:
        st.info("📊 No CSV uploaded yet — showing demo data. Upload a CSV in **Bulk CSV Upload** to see your results here.")
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
            dashboard_df = predict_batch(
                demo_df, model, tokenizer, id2label, device, text_col="review"
            )
        dash_text_col = "review"

    # Summary metrics
    st.markdown("#### Summary Metrics")
    render_insights(dashboard_df)

    st.markdown("---")

    # Top Themes
    render_top_themes(dashboard_df, dash_text_col)

    st.markdown("---")

    # Per-review table
    st.markdown("#### Sample Predictions")
    display_df = dashboard_df[[dash_text_col, "predicted_label", "confidence"]].rename(
        columns={dash_text_col: "review"}
    ).copy()
    display_df["confidence"] = display_df["confidence"].map("{:.1%}".format)

    def style_label(val):
        style = SENTIMENT_STYLE.get(val, {})
        return f"color: {style.get('hex', '#e2e8f0')}; font-weight: bold"

    st.dataframe(
        display_df.style.map(style_label, subset=["predicted_label"]),
        use_container_width=True,
    )

    st.markdown("---")

    # Confidence distribution histogram — dark themed
    st.markdown("#### Confidence Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    fig2.patch.set_facecolor('#0f0f1a')
    ax2.set_facecolor('#0f0f1a')
    ax2.hist(dashboard_df["confidence"], bins=10, color="#a78bfa",
             edgecolor="#0f0f1a", alpha=0.85, linewidth=1.5)
    ax2.set_xlabel("Confidence", fontsize=12, color="#94a3b8")
    ax2.set_ylabel("Count", fontsize=12, color="#94a3b8")
    ax2.set_title("Distribution of Prediction Confidence", fontsize=13,
                  fontweight="bold", color="#e2e8f0")
    ax2.set_xlim(0, 1)
    ax2.tick_params(colors="#64748b")
    ax2.spines['bottom'].set_color('#334155')
    ax2.spines['left'].set_color('#334155')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    st.pyplot(fig2, use_container_width=True)
