from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_NAME, MAX_NEW_TOKENS
from utils.prompt_builder import build_prompt
from utils.postprocessor import clean_prediction


class SentimentPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict(self, text: str) -> str:
        prompt = build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.0)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return clean_prediction(decoded.split("Sentiment:")[-1].strip())