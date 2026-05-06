# 🇮🇳 Hindi Product Review Sentiment Analyser

**SMAI Assignment 3 — Task T8.1**  
Fine-tuned **IndicBERT** for binary sentiment classification (Positive / Negative) of Hindi product reviews, with a zero-shot LLM baseline and a Streamlit web application.

---

## 📊 Results at a Glance

| Metric | IndicBERT (Fine-tuned) | LLaMA 3.3 70B (Zero-shot via Groq) |
|--------|------------------------|-------------------------------------|
| **Test Accuracy** | **82.76%** | **100%** |
| **Test Macro-F1** | **0.8274** | **1.0000** |
| **F1 — Negative** | 0.8333 (P: 0.8065 / R: 0.8621) | 1.0000 |
| **F1 — Positive** | 0.8214 (P: 0.8519 / R: 0.7931) | 1.0000 |
| **Test Loss** | 0.5410 | — |
| **Approach** | Supervised fine-tuning (6 epochs) | Zero-shot prompting |
| **Dataset** | AI4Bharat IndicSentiment — Hindi split (116 test samples) | Same test split |

> **Note on Zero-Shot:** The 100% zero-shot accuracy reflects LLaMA 3.3 70B's strong multilingual capability on this specific 116-sample test split. Results may vary on a larger, more diverse set.

## 🗂️ Repository Structure

```
.
├── T8_1_training_notebook.ipynb  # Phase 1, 2 & 3: EDA + IndicBERT fine-tuning + Zero-shot baseline
├── app.py                        # Phase 4: Streamlit web application
├── utils.py                      # Inference helper functions
├── requirements.txt              # All Python dependencies
├── csv_datasets_samples/         # Sample CSV files for bulk upload testing
├── .gitignore
└── README.md

# ⚠️  NOT in this repo (too large for GitHub):
# saved_model/                   # Fine-tuned model weights (~122 MB)
# saved_model.zip                # Compressed archive of the above
```

---

## ⚠️ Important Note on the Saved Model

The fine-tuned model weights (`saved_model/`) are **~122 MB**, which exceeds GitHub's hard 100 MB per-file limit. Therefore the model is **NOT committed to this repository**.

**For the TA / Evaluator:**  
The model can be downloaded directly from the Kaggle notebook output (see Step 3 below). Once downloaded and extracted, the Streamlit app will work immediately without any code changes.

---

## 🚀 Complete Setup Guide (Step by Step)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/RajaiTarun/Hindi-product-review-sentiment.git
cd Hindi-product-review-sentiment
```

---

### Step 2 — Install Dependencies

Make sure you have **Python 3.9+** installed. Then install all requirements:

```bash
pip install -r requirements.txt
```

This installs:
- `torch`, `transformers`, `sentencepiece` — for IndicBERT inference
- `streamlit`, `matplotlib`, `seaborn` — for the web app
- `pandas`, `numpy`, `scikit-learn` — for data processing
- `openai`, `groq` — for the zero-shot notebook API calls

---

### Step 3 — Run the Fine-Tuning Notebook on Kaggle & Download the Model

> **This step is only needed if you want to reproduce the fine-tuned model yourself.  
> If you are a TA evaluating the Streamlit app, download the pre-trained model from Kaggle notebook outputs directly.**

1. Go to Kaggle and open the notebook: **`T8_1_training_notebook.ipynb`**
   *(Upload it from the cloned repository if not already on Kaggle)*

2. Enable **GPU accelerator** (Settings → Accelerator → GPU T4 x2)

3. Add the following **Kaggle Secrets** if using the zero-shot notebook:
   - Key: `GROQ_API_KEY` → Value: *(your Groq API key from [console.groq.com](https://console.groq.com))*

4. **Run All** cells (`Run → Run All`)

5. Once training is complete, the final cell saves the model. In the **Output** panel on the right, find:
   - `saved_model.zip` → click the three-dot menu → **Download**

---

### Step 4 — Extract the Model Into the Project Folder

After downloading `saved_model.zip`:

```bash
# Move the zip to the project root
mv ~/Downloads/saved_model.zip .

# Extract it — this creates a saved_model/ folder
unzip saved_model.zip
```

Verify the folder contains these files:
```
saved_model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── tokenizer.model
├── special_tokens_map.json
└── label_config.json        ← custom file with class names & metrics
```

> **One-time fix:** Due to a known bug in older `transformers` versions, after extracting you must open `saved_model/tokenizer_config.json` and **delete the `extra_special_tokens` block** (lines that look like `"extra_special_tokens": ["<pad>", ...]`). The file should contain only the standard token keys. This is a tokenizer serialization bug in the Albert model family and does not affect model accuracy.

---

### Step 5 — Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

The app features three tabs:

| Tab | What it does |
|-----|-------------|
| 🔍 **Single Review** | Type / paste any Hindi review and get real-time sentiment + confidence |
| 📂 **Bulk CSV Upload** | Upload a CSV with a `review` column → batch classify → download results |
| 📈 **Insights Dashboard** | Live sentiment distribution charts and confidence histograms |

---

### Step 6 — (Optional) Run the Zero-Shot Baseline Notebook

The notebook `smai-a3-grok.ipynb` runs the LLaMA 3.3 70B zero-shot baseline via the Groq API.

1. Get a **free Groq API key** from [console.groq.com](https://console.groq.com)
2. Add it as a Kaggle secret with key `GROQ_API_KEY`
3. Run all cells — results are saved as a CSV and printed as a classification report

---

## 🧠 Approach Summary

### Phase 1, 2 & 3 — IndicBERT Fine-Tuning + Zero-Shot Baseline
All phases are implemented in a single notebook: **`T8_1_training_notebook.ipynb`**

- **Dataset:** [AI4Bharat IndicSentiment](https://huggingface.co/datasets/ai4bharat/IndicSentiment) (Hindi split, 116 test samples)
- **Fine-tuned Model:** `ai4bharat/indic-bert` (ALBERT-based, pre-trained on 12 Indic languages)
- **Training:** 6 epochs, AdamW optimizer, linear LR schedule with warmup, batch size 32
- **Classes:** Binary — `Positive` / `Negative`
- **Zero-Shot:** Meta LLaMA 3.3 70B Versatile via [Groq](https://groq.com) LPU inference, direct prompting without fine-tuning

### Phase 4 — Streamlit Web App
- **`utils.py`:** Handles model loading (cached with `st.cache_resource`) and batch/single inference
- **`app.py`:** Full-featured 3-tab Streamlit UI with visualizations and CSV download

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language Model | `ai4bharat/indic-bert` |
| Deep Learning | PyTorch + HuggingFace Transformers |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Zero-Shot API | Groq (LLaMA 3.3 70B) |
| Tokenizer | SentencePiece |

---

## 📋 Requirements

All dependencies are listed in [`requirements.txt`](./requirements.txt). Install with:

```bash
pip install -r requirements.txt
```

**Python version:** 3.9 or higher is recommended.

---

## ✅ Reproducibility Checklist

| Item | Status |
|------|--------|
| Random seed fixed (`SEED = 42`) | ✔ |
| Stratified train/val/test split (80/10/10) | ✔ |
| Tokeniser configuration documented | ✔ |
| Model checkpoint saved (best val macro-F1) | ✔ |
| Training history saved (`training_history.json`) | ✔ |
| Label mapping saved (`label_config.json`) | ✔ |
| All hyperparameters documented in the report | ✔ |
| Requirements file (`requirements.txt`) included | ✔ |
| Model available for download (Kaggle output) | ✔ |
| Groq API key required for zero-shot (not stored in code) | ✔ |

---

## 🖥️ Environment Specification

| Component | Version / Specification |
|-----------|------------------------|
| Python | 3.9+ |
| PyTorch | ≥ 2.0.0 |
| Transformers | ≥ 4.35.0 |
| SentencePiece | ≥ 0.1.99 |
| Streamlit | ≥ 1.30.0 |
| Pandas | ≥ 1.5.0 |
| NumPy | ≥ 1.24.0 |
| scikit-learn | ≥ 1.2.0 |
| Matplotlib | ≥ 3.6.0 |
| Seaborn | ≥ 0.12.0 |
| OpenAI client | ≥ 1.0.0 |
| Groq | ≥ 0.4.0 |
| Hardware (training) | NVIDIA T4 (Kaggle free tier) |

---

## 👤 Author

**Rajai Tarun Kanaiyalal**  
**Aryan**  
**Parimal Mate**  
SMAI Assignment 3 — T8.1  
[GitHub Repository](https://github.com/RajaiTarun/Hindi-product-review-sentiment)
