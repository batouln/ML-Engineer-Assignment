# ML APPLIED ENG. ASSIGNMENT

Part 1

---

##  Project Structure

```
.
├── app.py                 # FastAPI app for inference
├── models/              # Predictor class wrapping the LLM
│   └── predictor.py
├── utils/               # Helpers for prompts and postprocessing
│   ├── prompt_builder.py
│   └── postprocessor.py
├── scripts/             # Preprocessing scripts for dataset cleaning/splitting
│   └── preprocess_datasets.py
├── data/                # Cleaned and split datasets
│   ├── arsas_cleaned.csv
│   ├── english_cleaned.csv
│   ├── eval_dev.csv
│   └── eval_test.csv
├── Candidates models/            # Model evaluation and configuration
│   ├── candidaties_models.json   # Config for candidate models and datasets
│   └── eval_model.py             # Generic script for evaluating model predictions
    └── eval_helpers.py                
├── config.py            # Model name, label list, max tokens
├── test_api.py          # CLI testing script
└── requirements.txt     # Dependencies
```

---

## Datasets Used
To evaluate and select the most effective multilingual sentiment model, I built a **balanced benchmark** from two sources: an **Arabic Twitter dataset (ArSAS)** and an **English movie review dataset (SST5)**. Both were processed into a consistent **3-class setup**: `Negative`, `Neutral`, `Positive`.

---

### 1. ArSAS (Arabic Sentiment Analysis)
- **Source**: HuggingFace [arbml/ArSAS](https://huggingface.co/datasets/arbml/ArSAS)  
- **Domain**: Arabic tweets (covering MSA and dialectal Arabic)  
- **Preprocessing**:
  - Filtered tweets with `Sentiment_label_confidence < 0.66`
  - Removed diacritics, Tatweel (ـ), punctuation, emojis, and URLs
  - Normalized common orthographic variants (e.g., "ة" → "ه")
  - Mapped labels: `{0: Negative, 1: Neutral, 2: Positive}`

**Final Test Split (ArSAS)**:  
- 200 examples per label → **600 total Arabic examples**  

---

### 2. SST5 (English Sentiment Treebank, 5-Class → 3-Class)
- **Source**: HuggingFace [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5)  
- **Domain**: English movie reviews  
- **Preprocessing**:
  - Collapsed fine-grained labels:
    - `{0,1 → Negative; 2 → Neutral; 3,4 → Positive}`
  - Removed non-English Unicode & special characters
  - Limited maximum review length to **220 characters**
  - Lowercased and stripped extra whitespace  

**Final Test Split (SST5)**:  
- 200 examples per label → **600 total English examples**

---

### 3. Evaluation Benchmark Structure
To ensure **fair and fast comparison across models**, I constructed:

- **Development Set (dev)**  
  - **100 examples per label per language**  
  - Total = **600 examples** (quick debugging & prompt design)  

- **Test Set**  
  - **200 examples per label per language**  
  - Total = **1,200 examples** (used for all reported results)  

---

### Final Distribution (Balanced)

| Language | Negative | Neutral | Positive | Total |
|----------|----------|---------|----------|-------|
| Arabic   | 200      | 200     | 200      | 600   |
| English  | 200      | 200     | 200      | 600   |
| **Total** | **400** | **400** | **400** | **1200** |

---

 This benchmark ensures that all models are tested on **equal footing**, across both **Arabic dialects/MSA** and **English reviews**, with no class imbalance.


---

## Model Evaluation
I evaluated several LLMs using a unified prompt structure, balanced evaluation sets, and consistent metrics (Accuracy, Macro-F1, Latency).

All models are publicly available on Hugging Face and/or the [Open Arabic LLM Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard).

### Evaluation Results (mini leaderboard)

Sorted by Accuracy & Macro-F1 (low → high):

| Model | Accuracy | Macro-F1 | Avg Latency (ms/example) |
|-------|----------|-----------|---------------------------|
| [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | 0.6575 | 0.6233 | 214.8 |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 0.6633 | 0.6684 | 178.1 |
| [SILMA-9B (Silma AI)](https://huggingface.co/silma-ai/SILMA-9B-Instruct-v1.0) | 0.6858 | 0.6575 | 264.0 |
| [Fanar-1-9B-Instruct](https://huggingface.co/QCRI/Fanar-1-9B-Instruct) | **0.7317** | **0.7296** | **256.7** |

---

### Why Fanar-1-9B-Instruct?
### Why Fanar-1-9B-Instruct?

Across our benchmarks, Fanar delivered the **highest accuracy (73.2%) and macro-F1 (72.9%)** on the combined Arabic + English sentiment evaluation set.  
Although not the fastest model (≈ 257 ms average latency vs Qwen2’s 178 ms), Fanar consistently produced the most reliable predictions.  

For real-time REST APIs, this trade-off favors Fanar as the **default production choice**: quality is prioritized while latency remains competitive (<300 ms).  

If ultra-low latency is the primary requirement, **Qwen2** is a strong alternative, while **Qwen3** offers a balance between speed and accuracy.


#### Strengths
- **Top accuracy and F1** across both teh collected Arabic and English datasets  
- **Handles nuance**: distinguishes descriptive text from opinionated sentiment  
- **Consistent in negatives**: reliably flags negative sentiment, including sarcasm and insults  
- **Fastest latency** among tested models (256 ms/example)  

---

#### Confusion Matrix (Fanar-1-9B-Instruct)

|               | Pred: Positive | Pred: Neutral | Pred: Negative |
|---------------|----------------|---------------|----------------|
| **True Positive** | 292 | 91  | 17 |
| **True Neutral**  | 46  | 237 | 117 |
| **True Negative** | 5   | 46  | 349 |

---

#### Example Predictions

- **English (Positive detection)**  
  *"arguably the best script that besson has written in years"* → **Positive** ✅  

- **English (Nuance handling)**  
  *"a full experience, a love story and a murder mystery"* → **Neutral** ✅ (descriptive, not opinionated)  

- **Arabic (Negative clarity)**  
  *"مع احترامي لخالد علي الكومبارس الكبير..."* → **Negative** ✅ (sarcasm caught correctly)  

- **Arabic (Neutral balance)**  
  *"الربيع العربي أقسام الأول منها قامت به الشعوب..."* → **Neutral** ✅ (did not overfit to Positive/Negative)  

---

 This balance of **top raw performance, speed, and nuanced handling** makes **Fanar-1-9B** an excellent choice for sentiment analysis in a real-world REST API.


##  How to Run the Project

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Preprocessing Scripts (optional)
```bash
python scripts/preprocess_datasets.py
```
This will save:
- `data/arsas_cleaned.csv`
- `data/english_cleaned.csv`
- `data/eval_dev.csv`
- `data/eval_test.csv`
> **Note:** You don't need to run this script unless you want to regenerate the datasets from scratch.
### 3. Start the API Server
```bash
 uvicorn app:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000/predict`

### 4. Run the CLI Test
```bash
python test_api.py "تواصل العميل مع مركز خدمة العملاء للحصول على شهادة رصيد حسابها. تم توجيهها إلى الموقع الإلكتروني للبنك لإصدار الشهادة وتوجيهها إلى جهة معينة"
```

---
> **Note:** Use another terminal to execute the sentiment detection.
##  Prediction Format
Input:
```json
{"text": "تواصل العميل مع مركز خدمة العملاء للحصول على شهادة رصيد حسابها. تم توجيهها إلى الموقع الإلكتروني للبنك لإصدار الشهادة وتوجيهها إلى جهة معينة"}
```
Output:
```json
{"sentiment": "Neutral"}
```

---

###  Hardware & Runtime Environment

- **GPU:** NVIDIA L4 (24 GB VRAM)  
- **Precision:** fp16  
- **Platform:** Local containerized setup (FastAPI + transformers)
