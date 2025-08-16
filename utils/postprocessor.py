# postprocessor.py
import re
from config import REGEX_EN, AR_MAP, EN_MAP

def extract_label(raw: str) -> str:
    s = (raw or "").strip()

    # 1) English regex (optional "answer:")
    m = re.search(REGEX_EN, s, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    # 2) Arabic explicit (optional "الإجابة:")
    m = re.search(r"(?:^|\b)(?:الإجابة\s*:\s*)?\s*(إيجابي|ايجابي|سلبي|محايد)\b", s)
    if m:
        return AR_MAP.get(m.group(1), "Neutral")

    # 3) English token fallback
    first_en = re.findall(r"[A-Za-z]+", s)
    if first_en:
        return EN_MAP.get(first_en[0].lower(), "Neutral")

    # 4) Arabic token fallback
    toks = s.split()
    if toks:
        return AR_MAP.get(toks[0], "Neutral")

    # 5) Default
    return "Neutral"
