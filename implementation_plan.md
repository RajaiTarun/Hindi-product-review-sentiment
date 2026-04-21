# Hindi Product Review Sentiment Analyser (T8.1) — Implementation Plan

## Overview

Build a Hindi Product Review Sentiment Analyser with two separate components:
1. **Training Notebook** (Kaggle/Colab) — model development, evaluation, export
2. **Streamlit App** (local) — inference-only web interface

---

## Phase 1: Training Notebook — Data & Preprocessing

> **File:** `T8_1_Hindi_Sentiment.ipynb`

### Step 1.1: Environment Setup
- Install `transformers`, `datasets`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`
- Set reproducibility seeds

### Step 1.2: Load Dataset
- Load `ai4bharat/IndicSentiment` with Hindi config (`hi`)
- Inspect train/val/test splits
- **Expected output:** Dataset shape, sample rows

### Step 1.3: EDA
- Class distribution (bar chart)
- Review length distribution (histogram)
- Sample reviews per class
- **Expected output:** 2–3 plots, printed statistics

### Step 1.4: Preprocessing
- Map labels: Positive → 0, Negative → 1, Neutral → 2
- Tokenize with `ai4bharat/indic-bert` tokenizer (max_length=128)
- Create PyTorch DataLoaders (batch_size=16)
- **Expected output:** Tokenized dataset, DataLoader samples

---

## Phase 2: Training Notebook — Fine-Tuning IndicBERT

### Step 2.1: Model Setup
- Load `ai4bharat/indic-bert` with `AutoModelForSequenceClassification` (num_labels=3)
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Scheduler: Linear warmup
- Loss: CrossEntropyLoss

### Step 2.2: Training Loop
- Train for 3 epochs
- Track train loss, train accuracy per epoch
- Validate after each epoch (val loss, val accuracy, val F1)
- **Expected output:** Training curves (loss + accuracy plots)

### Step 2.3: Evaluation
- Run on test set
- Compute: **Accuracy**, **Macro-F1**, **per-class Precision/Recall**
- Generate **Confusion Matrix** heatmap
- **Expected output:** Metrics table, confusion matrix plot

### Step 2.4: Save Model
- Save fine-tuned model + tokenizer to disk
- Export as a zip for download from Colab/Kaggle
- **Expected output:** `saved_model/` directory with model files

---

## Phase 3: Training Notebook — Zero-Shot Baseline

### Step 3.1: API Setup
- Use Gemini 1.5 Flash API (free tier) or Groq LLaMA-3
- Design structured prompt:
  ```
  Classify the following Hindi product review as Positive, Negative, or Neutral.
  Return JSON: {"label": "...", "confidence": 0.XX}
  Review: <text>
  ```

### Step 3.2: Run Zero-Shot on Test Set
- Classify a sample (100–200 reviews) from the test set
- Parse JSON responses
- **Expected output:** Predictions list

### Step 3.3: Compare Approaches
- Side-by-side metrics table (Fine-tuned vs Zero-shot)
- Accuracy, Macro-F1, per-class F1
- **Expected output:** Comparison table + bar chart

---

## Phase 4: Streamlit Web App

> **File:** `app.py` (+ `utils.py` for helper functions)

> [!IMPORTANT]
> The app does **NO training**. It only loads the saved model for inference.

### Step 4.1: Project Structure
```
Assignment 3/
├── app.py                  # Streamlit main app
├── utils.py                # Inference helpers (load model, predict)
├── requirements.txt        # App dependencies
├── saved_model/            # Fine-tuned model (from Phase 2)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── T8_1_Hindi_Sentiment.ipynb  # Training notebook
└── README.md
```

### Step 4.2: `utils.py` — Inference Module
- `load_model(path)` → loads model + tokenizer
- `predict_single(text, model, tokenizer)` → returns `{label, confidence, probabilities}`
- `predict_batch(df, model, tokenizer)` → returns DataFrame with predictions

### Step 4.3: `app.py` — Streamlit UI
**Features:**
1. **Single Review Input** — text area, "Analyze" button → shows sentiment badge + confidence bar
2. **Bulk CSV Upload** — upload CSV with a `review` column → download results CSV
3. **Insights Dashboard** — pie chart of sentiment distribution, avg confidence, sample reviews per class
4. **Model Info** — sidebar with model details, approach comparison

**UI Design:**
- Hindi-friendly fonts
- Color-coded sentiment: 🟢 Positive, 🔴 Negative, 🟡 Neutral
- Clean, professional layout

### Step 4.4: Testing the App
- Test with sample Hindi reviews
- Test CSV upload with 10–20 reviews
- Verify confidence scores sum to ~1.0

---

## Phase 5: Finalization & Deliverables

### Step 5.1: `requirements.txt`
```
torch
transformers
datasets
scikit-learn
streamlit
pandas
matplotlib
seaborn
```

### Step 5.2: `README.md`
- Project description
- How to run the notebook
- How to run the Streamlit app
- Model details

### Step 5.3: Technical Report (4–6 pages)
1. Introduction & Motivation
2. Dataset Description
3. Methodology (Fine-tuning + Zero-shot)
4. Results & Analysis
5. Web Application
6. Conclusion & Future Work

### Step 5.4: LinkedIn Pitch (1-slide)
- Problem → Approach → Results → Demo screenshot

---

## Verification Plan

### Automated Tests
- Model loads correctly from saved checkpoint
- Predictions return valid labels (0/1/2) and confidence ∈ [0, 1]
- Streamlit app runs without errors: `streamlit run app.py`

### Manual Verification
- Spot-check 10 Hindi reviews with known sentiment
- Verify confusion matrix matches reported metrics
- Test CSV upload end-to-end in browser

---

## Execution Order

| Step | What | Where |
|------|------|-------|
| 1 | Data loading + EDA | Notebook |
| 2 | Preprocessing + tokenization | Notebook |
| 3 | Fine-tune IndicBERT | Notebook |
| 4 | Evaluate + confusion matrix | Notebook |
| 5 | Save model | Notebook |
| 6 | Zero-shot baseline | Notebook |
| 7 | Compare approaches | Notebook |
| 8 | Build `utils.py` | Local |
| 9 | Build `app.py` | Local |
| 10 | Test app | Local |
| 11 | Write README + requirements | Local |
| 12 | Write report + pitch | Local |
