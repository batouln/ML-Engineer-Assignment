
import pandas as pd
import re
import html
import random
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset

# --------------- CONFIG ---------------
AR_OUTPUT_CSV = "data/arsas_cleaned.csv"
EN_OUTPUT_CSV = "data/english_cleaned.csv"
DEV_OUTPUT_CSV = "data/eval_dev.csv"
TEST_OUTPUT_CSV = "data/eval_test.csv"

DEV_PER_LABEL = 100
TEST_PER_LABEL = 200
RANDOM_SEED = 42

# --------------- NORMALIZATION ---------------
AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = "\u0640"

LABEL_MAPS = {
    "ar": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "en": {0: "Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Positive"}
}


def normalize_arabic(text: str) -> str:
    text = html.unescape(text)
    text = text.replace(TATWEEL, "")
    text = AR_DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def basic_arabic_clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#][\w_]+", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def basic_english_clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clip_text(text: str, max_len=220) -> str:
    return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0]

# --------------- CLEANING FUNCTIONS ---------------

def clean_arsas():
    print("Loading ArSAS...")
    df = load_dataset("arbml/ArSAS")['train'].to_pandas()

    df = df[df["label"].isin(LABEL_MAPS['ar'])]
    df["Sentiment_label_confidence"] = pd.to_numeric(df["Sentiment_label_confidence"], errors="coerce")
    df = df[df["Sentiment_label_confidence"] >= 0.66].copy()

    df["text_raw"] = df["Tweet_text"].apply(basic_arabic_clean)
    df["text"] = df["text_raw"].apply(normalize_arabic)
    df["label"] = df["label"].map(LABEL_MAPS['ar'])
    df["lang"] = "ar"

    out_df = df[["text", "label", "lang"]]
    Path(AR_OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(AR_OUTPUT_CSV, index=False)

    print(f"Saved ArSAS → {AR_OUTPUT_CSV} ({len(out_df)} rows)")
    print("Label counts:", Counter(out_df["label"]))


def clean_sst5():
    print("Loading SST5...")
    df = load_dataset("SetFit/sst5")['train'].to_pandas()

    df = df[df["label"].isin(LABEL_MAPS['en'])].copy()
    df["label"] = df["label"].map(LABEL_MAPS['en'])
    df["text"] = df["text"].astype(str).apply(basic_english_clean)
    df["text"] = df["text"].apply(lambda t: clip_text(t))
    df["lang"] = "en"

    df_out = df[["text", "label", "lang"]]
    df_out.to_csv(EN_OUTPUT_CSV, index=False)

    print(f"Saved SST5 → {EN_OUTPUT_CSV} ({len(df_out)} rows)")
    print("Label counts:", Counter(df_out["label"]))

# --------------- SPLIT FUNCTION ---------------

def split_eval_sets():
    df_ar = pd.read_csv(AR_OUTPUT_CSV)
    df_en = pd.read_csv(EN_OUTPUT_CSV)
    df = pd.concat([df_ar, df_en], ignore_index=True)

    random.seed(RANDOM_SEED)
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[(row["label"], row["lang"])].append(row)

    dev_rows, test_rows = [], []
    for key, rows in grouped.items():
        random.shuffle(rows)
        dev_n = min(DEV_PER_LABEL, len(rows) // 3)
        test_n = min(TEST_PER_LABEL, len(rows) - dev_n)
        dev_rows += rows[:dev_n]
        test_rows += rows[dev_n:dev_n + test_n]

    dev_df = pd.DataFrame(dev_rows)
    test_df = pd.DataFrame(test_rows)

    dev_df.to_csv(DEV_OUTPUT_CSV, index=False)
    test_df.to_csv(TEST_OUTPUT_CSV, index=False)

    print(f"Saved dev  → {DEV_OUTPUT_CSV} ({len(dev_df)} rows)")
    print(f"Saved test → {TEST_OUTPUT_CSV} ({len(test_df)} rows)")
    print("Label counts (Dev):\n", dev_df['label'].value_counts())
    print("Label counts (Test):\n", test_df['label'].value_counts())

# --------------- MAIN ---------------

if __name__ == "__main__":
    clean_arsas()
    clean_sst5()
    split_eval_sets()
