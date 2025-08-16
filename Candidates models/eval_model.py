# eval_model.py
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
from pathlib import Path

import pandas as pd
import torch
import torch._dynamo
from huggingface_hub import login as hf_login
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import DTYPE
from eval_helpers import load_model, evaluate_dataframe

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# -----------------------------
# Load candidaties_models.json .. here is we can find all the models i'm intersted in to evaluate
# -----------------------------
CFG_PATH = Path("candidaties_models.json")
if not CFG_PATH.exists():
    raise FileNotFoundError("Missing candidaties_models.json")

with CFG_PATH.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

hf_token   = cfg.get("hf_token")
model_name = cfg["model_name"] 
model_tag  = cfg.get("model_tag", Path(model_name).name)
results_dir = Path(cfg.get("results_dir", "results")).resolve()
datasets   = cfg.get("datasets", [])

if not datasets:
    raise ValueError("No datasets provided in candidaties_models.json.")

if hf_token:
    hf_login(hf_token)

# -----------------------------
# Load model once
# -----------------------------
print(f"\n=== MODEL: {model_name} (tag: {model_tag}) ===")
print(f"→ Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | dtype: {DTYPE}")
tokenizer, model = load_model(model_name, DTYPE)

# Combined accumulators
all_src, all_gold, all_pred, all_raw, all_times = [], [], [], [], []
all_wall = 0.0

# -----------------------------
# Evaluate each dataset
# -----------------------------
for ds in datasets:
    path = Path(ds["path"])
    tag  = ds.get("tag", path.stem)

    if not path.exists():
        print(f"\n[WARN] Missing dataset: {path}")
        continue

    print(f"\n=== Evaluating: {path} ({tag}) ===")
    df = pd.read_csv(path)

    y_true, y_pred, raw, times, metrics, timing = evaluate_dataframe(df, tokenizer, model, show_raw=3)

    # Print metrics
    if metrics:
        print(f"\nAccuracy   : {metrics['accuracy']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(metrics["cm"])
    else:
        print("[WARN] No valid labels for evaluation.")

    print(f"\nSpeed → total: {timing['total_time']:.2f}s | avg: {timing['avg_latency_ms']:.2f}ms | throughput: {timing['throughput_eps']:.2f}/s")

    # Save per-dataset files with tag+model_tag suffix
    prefix = path.with_suffix("")
    preds_csv   = prefix.with_name(f"{prefix.name}_{model_tag}_preds.csv")
    metrics_csv = prefix.with_name(f"{prefix.name}_{model_tag}_metrics.csv")
    cm_csv      = prefix.with_name(f"{prefix.name}_{model_tag}_cm.csv")

    pd.DataFrame({
        "text": df["text"],
        "label": df["label"],
        "gold_norm": y_true,
        "pred": y_pred,
        "raw": raw,
    }).to_csv(preds_csv, index=False, encoding="utf-8")

    if metrics:
        pd.DataFrame([metrics]).to_csv(metrics_csv, index=False, encoding="utf-8")
        pd.DataFrame(metrics["cm"],
                     index=["Positive","Neutral","Negative"],
                     columns=["Positive","Neutral","Negative"]).to_csv(cm_csv, encoding="utf-8")

    print(f"[Saved] {preds_csv}, {metrics_csv}, {cm_csv}")

    # Accumulate for combined
    all_src.extend([path.name] * len(y_true))
    all_gold.extend(y_true)
    all_pred.extend(y_pred)
    all_raw.extend(raw)
    all_times.extend(times)
    all_wall += timing["total_time"]

# -----------------------------
# Combined across all datasets
# -----------------------------
print("\n=== Combined Results Across Datasets ===")
valid_idx = [i for i,g in enumerate(all_gold) if g in ["Positive","Neutral","Negative"]]
y_true_all = [all_gold[i] for i in valid_idx]
y_pred_all = [all_pred[i] for i in valid_idx]

if y_true_all:
    acc = accuracy_score(y_true_all, y_pred_all)
    f1m = f1_score(y_true_all, y_pred_all, average="macro")
    cm  = confusion_matrix(y_true_all, y_pred_all, labels=["Positive","Neutral","Negative"])

    avg_ms = (sum(all_times)/len(all_times)*1000) if all_times else 0.0
    throughput = len(all_gold)/all_wall if all_wall > 0 else 0.0

    print(f"Accuracy   : {acc:.4f}")
    print(f"F1 (macro): {f1m:.4f}")
    print("Confusion matrix:")
    print(cm)

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dataset": all_src,
        "gold_norm": all_gold,
        "pred": all_pred,
        "raw": all_raw,
    }).to_csv(results_dir/f"{model_tag}_all_results.csv", index=False, encoding="utf-8")
    pd.DataFrame([{
        "accuracy": acc,
        "f1_macro": f1m,
        "avg_latency_ms": avg_ms,
        "throughput_eps": throughput,
    }]).to_csv(results_dir/f"{model_tag}_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(cm, index=["Positive","Neutral","Negative"], columns=["Positive","Neutral","Negative"])\
        .to_csv(results_dir/f"{model_tag}_cm.csv", encoding="utf-8")
