from config import VALID_LABELS

def clean_prediction(pred: str) -> str:
    if not isinstance(pred, str):
        return "Neutral"

    pred = pred.replace("[", "").replace("]", "").replace("\"", "").replace("'", "").strip()
    pred = pred.split("\n")[0].strip().capitalize()

    for valid in VALID_LABELS:
        if valid.lower() in pred.lower():
            return valid
    return "Neutral"