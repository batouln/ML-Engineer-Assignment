# Arabic & English Sentiment Analysis REST API

This project is a complete sentiment analysis pipeline using a **self-hosted LLM** to predict the sentiment of Arabic and English sentences. It includes preprocessing scripts, an evaluation suite, model benchmarking, and a FastAPI server for inference.

---

##  Project Structure

```
.
├── app/                 # FastAPI app for inference
│   └── main.py
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
├── config.py            # Model name, label list, max tokens
├── test_api.py          # CLI testing script
└── requirements.txt     # Dependencies
```

---

##  Datasets Used

We combined two datasets to create a multilingual benchmark:

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
We ensured a fair balance across both languages by splitting the data into dev/test:

```
✔ dev: 100 examples / label / language
✔ test: 200 examples / label / language
```

---

##  Model Evaluation
We evaluated several LLMs using the same prompt structure, dev/test sets, and metric scripts (Accuracy, Macro-F1).

###  Evaluation Results

| Model Name                                       | Accuracy | Macro-F1 |
|--------------------------------------------------|----------|-----------|
| **Qwen3-8B-Base**                                | 0.3133   | 0.1687    |
| **Qwen2.5-7B-Instruct**                          | 0.5700   | 0.5675    |
| **C4AI R7B Arabic (Feb 2025)**                   | 0.3350   | 0.1745    |
| **SILMA-9B (Silma AI)**                          | **0.6717** | **0.6475** |
| **Fanar-1-9B**                                   | 0.5350   | 0.5199    |

###  Why SILMA-9B?

- **Best overall performance** on both accuracy and macro-F1 across both languages.
- Robust handling of Arabic dialects and English nuances.
- Stable responses and deterministic decoding.

This makes **SILMA-9B** the most suitable LLM for powering our REST API.

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

### 3. Start the API Server
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000/predict`

### 4. Run the CLI Test
```bash
python test_api.py "تواصل العميل مع مركز خدمة العملاء للحصول على شهادة رصيد حسابها. تم توجيهها إلى الموقع الإلكتروني للبنك لإصدار الشهادة وتوجيهها إلى جهة معينة"
```

---

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

