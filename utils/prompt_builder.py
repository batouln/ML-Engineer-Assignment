# prompt_builder.py
from config import PROMPT_CONTENT

def build_prompt(text: str) -> str:
    return PROMPT_CONTENT.replace("{TEXT}", str(text))
