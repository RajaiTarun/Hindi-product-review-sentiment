"""
utils.py — Inference helpers for Hindi Sentiment Analyser
Loads the fine-tuned IndicBERT model and provides single/batch prediction.
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "saved_model"
MAX_LENGTH = 128

# Emoji + color mapping for sentiment labels
SENTIMENT_STYLE = {
    "Positive": {"emoji": "🟢", "color": "#2ecc71"},
    "Negative": {"emoji": "🔴", "color": "#e74c3c"},
    "Neutral":  {"emoji": "🟡", "color": "#f39c12"},
}


# ── Model Loading ──────────────────────────────────────────────────────────

def _load(model_path: str):
    """Internal loader — called once and cached by Streamlit."""
    config_path = os.path.join(model_path, "label_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"label_config.json not found in '{model_path}'. "
            "Please download saved_model/ from your Kaggle notebook."
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    label2id: dict = cfg["label2id"]
    id2label: dict = {int(k): v for k, v in cfg["id2label"].items()}
    num_labels: int = cfg["num_labels"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer, id2label, label2id, device, cfg


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load fine-tuned model + tokenizer from disk.

    Returns
    -------
    model, tokenizer, id2label, label2id, device, cfg
    """
    return _load(model_path)


# ── Single Prediction ──────────────────────────────────────────────────────

def predict_single(
    text: str,
    model,
    tokenizer,
    id2label: dict,
    device,
    max_length: int = MAX_LENGTH,
) -> dict:
    """
    Classify a single Hindi review.

    Returns
    -------
    dict with keys:
        label       : str   — predicted sentiment label
        confidence  : float — probability of the predicted class
        probabilities : dict — {label: probability} for all classes
    """
    if not text or not text.strip():
        return {"label": "N/A", "confidence": 0.0, "probabilities": {}}

    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs   = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))

    probabilities = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "label":         id2label[pred_id],
        "confidence":    float(probs[pred_id]),
        "probabilities": probabilities,
    }


# ── Batch Prediction ───────────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame,
    model,
    tokenizer,
    id2label: dict,
    device,
    text_column: str = "review",
    batch_size: int = 32,
    max_length: int = MAX_LENGTH,
) -> pd.DataFrame:
    """
    Classify a DataFrame of Hindi reviews.

    Parameters
    ----------
    df          : DataFrame containing at least `text_column`
    text_column : name of the column with review text

    Returns
    -------
    df with extra columns: predicted_label, confidence, + one column per label class
    """
    texts = df[text_column].fillna("").tolist()

    all_labels       = []
    all_confidences  = []
    all_probs        = []  # list of dicts

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]

        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs   = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)

        for i, (pid, prob_row) in enumerate(zip(pred_ids, probs)):
            all_labels.append(id2label[int(pid)])
            all_confidences.append(float(prob_row[pid]))
            all_probs.append({id2label[j]: float(prob_row[j]) for j in range(len(prob_row))})

    result_df = df.copy()
    result_df["predicted_label"] = all_labels
    result_df["confidence"]      = all_confidences

    # Expand per-class probabilities as separate columns
    prob_df = pd.DataFrame(all_probs)
    for col in prob_df.columns:
        result_df[f"prob_{col}"] = prob_df[col].values

    return result_df
