# scripts/eval_llm_sentiment.py

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch._dynamo
from huggingface_hub import login
import json

# Disable torch dynamo optimization
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Load config
with open("models_candidaties.json") as f:
    config = json.load(f)

HF_TOKEN = config.get("hf_token")
MODEL_NAME = config.get("model_name")
MODEL_TAG = config.get("model_tag", MODEL_NAME.split("/")[-1])
RESULTS_DIR = config.get("results_dir", "results")

datasets = config.get("datasets", [
    {"path": "data/arsas_eval_test.csv", "lang": "ar", "tag": "arsas"},
    {"path": "data/english_eval_test.csv", "lang": "en", "tag": "sst"}
])

login(HF_TOKEN)

# Label mapping for Arabic (ArSAS)
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
VALID_LABELS = {"Positive", "Neutral", "Negative"}

def generate_sentiment_prompt(text: str) -> str:
    return f'''You are a helpful assistant. Your task is to classify the sentiment of the given text into one of the following categories:
["Positive", "Neutral", "Negative"]

Text: {text}

Sentiment:
'''


# --- Load model ---
def load_model(model_name=MODEL_NAME, dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
    return tokenizer, model

# --- Predict one ---
@torch.no_grad()
def predict_sentiment(prompt: str, tokenizer, model, max_new_tokens=10):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.split("Sentiment:")[-1].strip()

# --- Clean prediction ---
def clean_prediction(p):
    if isinstance(p, str):
        p = p.replace("[", "").replace("]", "").replace('"', "").replace("'", "").strip()
        p = p.split()[0].capitalize()
        if p not in VALID_LABELS:
            return "Neutral"
        return p
    return "Neutral"

if __name__ == "__main__":
    tokenizer, model = load_model()
    all_rows = []

    for d in datasets:
        df = pd.read_csv(d["path"])
        lang = d["lang"]
        tag = d["tag"]

        if df["label"].dtype != object:
            df["label"] = df["label"].map(LABEL_MAP)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {tag.upper()}"):
            text = row["text"]
            label = row["label"]
            prompt = generate_sentiment_prompt(text)
            pred = predict_sentiment(prompt, tokenizer, model)
            pred_clean = clean_prediction(pred)
            all_rows.append({
                "text": text,
                "label": label,
                "predicted": pred_clean,
                "lang": lang,
                "dataset": tag
            })

    all_df = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{MODEL_TAG}_all_results.csv")
    all_df.to_csv(results_path, index=False)

    acc = accuracy_score(all_df["label"], all_df["predicted"])
    f1 = f1_score(all_df["label"], all_df["predicted"], average="macro")

    print("\n Combined Evaluation Results:")
    print(f" Accuracy: {acc:.4f}")
    print(f" F1-macro: {f1:.4f}")

    pd.DataFrame([{"model": MODEL_TAG, "accuracy": acc, "f1_macro": f1, "total_samples": len(all_df)}]) \
        .to_csv(os.path.join(RESULTS_DIR, f"{MODEL_TAG}_metrics.csv"), index=False)

    print(f" Saved predictions to {results_path}")
