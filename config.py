# config.py
import torch

# -----------------------------
# Model & dtype
# -----------------------------
MODEL_NAME = "silma-ai/SILMA-9B-Instruct-v1.0"  # switch here if needed
DTYPE = (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

GENERATION_KWARGS = dict(
    do_sample=False,        # greedy
    num_beams=1,
    max_new_tokens=4,
    eos_token_id=None,      # let tokenizer/model decide; we also slice by input length
)

# -----------------------------
# Universal prompt content
# -----------------------------
PROMPT_CONTENT = """You are a careful sentiment classifier.

Task:
Classify the sentiment of the given text into exactly one of these labels:
Positive, Neutral, Negative

Output rules:
- Reply with ONE WORD ONLY: Positive or Neutral or Negative.
- No punctuation, no extra words, no explanations.

Examples:
Text: The customer was experiencing an issue with the laptop that has not been resolved yet after several attempts, and an appointment was scheduled to follow up on the case next Sunday.
Answer: Negative

Text: تواصل العميل مع مركز خدمة العملاء للحصول على شهادة رصيد حسابها. تم توجيهها إلى الموقع الإلكتروني للبنك لإصدار الشهادة وتوجيهها إلى جهة معينة
Answer: Neutral

Text: كان العميل يواجه صعوبة في سماع الوكيل أثناء المكالمة بسبب انخفاض مستوى الصوت. وافق الوكيل على إرسال رسالة نصية عبر تطبيق واتساب وأبدى العميل امتنانه بينما ينتظر اتصالاً معاوداً
Answer: Positive

Now classify the new example.
Text: {TEXT}
Answer:"""

# -----------------------------
# Labels & parsing maps
# -----------------------------
VALID_LABELS = ["Positive", "Neutral", "Negative"]

# english regex with optional "answer:"
REGEX_EN = r"(?:^|\b)(?:answer\s*:\s*)?\s*(positive|neutral|negative)\b"

# arabic & english maps for fallbacks
AR_MAP = {"إيجابي": "Positive", "ايجابي": "Positive", "سلبي": "Negative", "محايد": "Neutral"}
EN_MAP = {
    "positive": "Positive", "pos": "Positive",
    "neutral":  "Neutral",  "neu": "Neutral",
    "negative": "Negative", "neg": "Negative",
}
