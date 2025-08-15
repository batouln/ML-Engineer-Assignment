from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.predictor import SentimentPredictor

app = FastAPI()
predictor = SentimentPredictor()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")
    prediction = predictor.predict(req.text)
    return {"sentiment": prediction}