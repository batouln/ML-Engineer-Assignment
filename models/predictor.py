# predictor.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_NAME, DTYPE, GENERATION_KWARGS
from prompt_builder import build_prompt
from postprocessor import extract_label

class SentimentPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=DTYPE,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise classifier that follows output rules exactly."},
            {"role": "user", "content": build_prompt(text)},
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(**inputs, **GENERATION_KWARGS)
        # Slice new tokens (robust if eos appears early)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return extract_label(raw)
