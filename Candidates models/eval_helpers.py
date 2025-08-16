# eval_helpers.py
import re
import time
from numbers import Number
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import GENERATION_KWARGS
from prompt_builder import build_prompt
from postprocessor import extract_label

# ---------- tiny utilities ----------

LABEL_ORDER = ["Positive", "Neutral", "Negative"]
NUM_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def normalize_gold(y) -> Optional[str]:
    """Map various gold formats to Positive/Neutral/Negative, or None if unknown."""
    if isinstance(y, Number):
        return NUM_TO_LABEL.get(int(y))
    if isinstance(y, str):
        s = y.strip()
        if s in {"إيجابي", "ايجابي"}: return "Positive"
        if s in {"محايد"}:            return "Neutral"
        if s in {"سلبي"}:             return "Negative"
        sl = s.lower()
        if sl in {"positive", "pos", "p"}:         return "Positive"
        if sl in {"neutral", "neu", "n", "neut."}: return "Neutral"
        if sl in {"negative", "neg"}:              return "Negative"
        try:
            f = float(s); i = int(f)
            if f == i: return NUM_TO_LABEL.get(i)
        except Exception:
            pass
    return None

# ---------- model I/O ----------

def load_model(model_name: str, dtype: torch.dtype) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl

@torch.no_grad()
def predict_one(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> Tuple[str, str]:
    """Return (label, raw_generated_text) for a single input."""
    messages = [
        {"role": "system", "content": "You are a precise classifier that follows output rules exactly."},
        {"role": "user", "content": build_prompt(text)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **GENERATION_KWARGS)

    # Only new tokens (after the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_label(raw), raw

# ---------- per-dataset eval ----------

def evaluate_dataframe(df: pd.DataFrame, tokenizer, model, show_raw: int = 3):
    """Evaluate a df(text,label) → (y_true, y_pred, raw_outputs, times, metrics, timing)"""
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    y_true, y_pred, raw_outputs, per_item_times = [], [], [], []
    shown = 0

    cuda_sync()
    t0_total = time.perf_counter()

    for _, row in df.iterrows():
        text = str(row["text"])
        gold = normalize_gold(row["label"])

        cuda_sync()
        t0 = time.perf_counter()
        pred, raw = predict_one(text, tokenizer, model)
        cuda_sync()
        t1 = time.perf_counter()

        y_true.append(gold)
        y_pred.append(pred)
        raw_outputs.append(raw)
        per_item_times.append(t1 - t0)

        if shown < show_raw:
            print("\nSample RAW output:")
            print("Text:", text[:200].replace("\n", " "))
            print("RAW :", repr(raw))
            print("PRED:", pred, " | GOLD(norm):", gold)
            shown += 1

    cuda_sync()
    total_time = time.perf_counter() - t0_total

    # metrics only on valid golds
    valid_idx = [i for i, g in enumerate(y_true) if g in LABEL_ORDER]
    y_true_valid = [y_true[i] for i in valid_idx]
    y_pred_valid = [y_pred[i] for i in valid_idx]

    metrics = None
    if y_true_valid:
        acc = accuracy_score(y_true_valid, y_pred_valid)
        f1m = f1_score(y_true_valid, y_pred_valid, average="macro")
        cm  = confusion_matrix(y_true_valid, y_pred_valid, labels=LABEL_ORDER)
        metrics = {"accuracy": acc, "f1_macro": f1m, "cm": cm}

    avg_latency_ms = (sum(per_item_times) / len(y_true) * 1000.0) if y_true else 0.0
    throughput = (len(y_true) / total_time) if total_time > 0 else 0.0

    timing = {
        "total_time": total_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_eps": throughput,
    }
    return y_true, y_pred, raw_outputs, per_item_times, metrics, timing
