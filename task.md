# Task Tracker — T8.1 Hindi Sentiment Analyser

## Phase 1: Data & Preprocessing (Notebook)
- [x] Step 1.1: Environment setup
- [x] Step 1.2: Load IndicSentiment dataset (robust fallback via direct URL)
- [x] Step 1.3: EDA (class distribution, review lengths, samples)
- [x] Step 1.4: Preprocessing & tokenization (label mapping, IndicBERT tokenizer, DataLoaders)

## Phase 2: Fine-Tuning (Notebook)
- [x] Step 2.1: Model setup (IndicBERT + classification head, AdamW, LinearLR)
- [x] Step 2.2: Training loop (3 epochs with best-model checkpoint)
- [x] Step 2.3: Evaluation (Accuracy, Macro-F1, Confusion Matrix, Classification Report)
- [x] Step 2.4: Save model + tokenizer + label config + zip for download

## Phase 3: Zero-Shot Baseline (Notebook)
- [/] Step 3.1: API setup (Gemini) — code added, needs Kaggle run
- [/] Step 3.2: Run zero-shot on test set — code added, needs Kaggle run
- [/] Step 3.3: Compare approaches — code added, needs Kaggle run

## Phase 4: Streamlit App
- [x] Step 4.1: utils.py (inference module — load_model, predict_single, predict_batch)
- [x] Step 4.2: app.py (3-tab UI: Single Review, Bulk CSV, Insights Dashboard)
- [ ] Step 4.3: Test app (needs saved_model/ from Kaggle)

## Phase 5: Finalization
- [x] Step 5.1: requirements.txt
- [ ] Step 5.2: README.md
- [ ] Step 5.3: Technical report
- [ ] Step 5.4: LinkedIn pitch
