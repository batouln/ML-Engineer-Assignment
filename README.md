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
├── config.py            # Model name, label list, max tokens
├── test_api.py          # CLI testing script
└── requirements.txt     # Dependencies
```

---

##  Datasets Used
I combined two datasets to build a multilingual benchmark for evaluating and selecting the most effective model for sentiment analysis tasks:

### 1. ArSAS (Arabic Sentiment Analysis)
- Source: HuggingFace [arbml/ArSAS](https://huggingface.co/datasets/arbml/ArSAS)
- Preprocessing steps:
  - Removed tweets with `Sentiment_label_confidence < 0.66`
  - Cleaned and normalized Arabic text (diacritics, URLs, Tatweel, punctuation)
  - Mapped labels: `{0: Negative, 1: Neutral, 2: Positive}`

### 2. SST5 (English Sentiment Treebank 5-Class)
- Source: HuggingFace [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5)
- Preprocessing steps:
  - Merged 0 & 1 → Negative, 3 & 4 → Positive (to match ArSAS 3-class setup)
  - Cleaned text, removed non-English unicode, limited length to 220 chars
  - Mapped labels: `{0,1 → Negative; 2 → Neutral; 3,4 → Positive}`

### Final Label Distribution
I ensured a fair balance across both languages by splitting the data into dev/test:

```
✔ dev: 100 examples / label / language, I used this dataset for quick debugging for each model
✔ test: 200 examples / label / language
```

---

##  Model Evaluation
I evaluated several LLMs using the same prompt structure, the collected datasets, and metric scripts (Accuracy, Macro-F1).
All of these models are publicly available via the [Open Arabic LLM Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard), hosted on Hugging Face Spaces.

###  Evaluation Results (mini leaderboard)

| Model Name | Accuracy | Macro-F1 |
|------------|----------|-----------|
| [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | 0.3133 | 0.1687 |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 0.5700 | 0.5675 |
| [CohereLabs/c4ai-command-r7b-arabic-02-2025](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025) | 0.3350 | 0.1745 |
| [SILMA-9B (Silma AI)](https://huggingface.co/silma-ai/SILMA-9B-Instruct-v1.0) | **0.6717** | **0.6475** |
| [Fanar-1-9B](https://huggingface.co/QCRI/Fanar-1-9B) | 0.5350 | 0.5199 |



###  Why SILMA-9B?

- **Best overall performance** on both accuracy and macro-F1 across both languages.
- Robust handling of Arabic dialects and English nuances.
- Stable responses and deterministic decoding.

This makes **SILMA-9B** the most suitable LLM based on this evaluation.

---

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
