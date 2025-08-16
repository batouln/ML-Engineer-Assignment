from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.predictor import SentimentPredictor
from config import VALID_LABELS  # <-- add this

app = FastAPI()
predictor = SentimentPredictor()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")
    prediction = predictor.predict(req.text)

    # Enforce strict one-word output (safety net)
    if prediction not in VALID_LABELS:
        raise HTTPException(status_code=500, detail=f"Model returned invalid label: {prediction}")

    return {"sentiment": prediction}
