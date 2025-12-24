from typing import List
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.perception.cnn_perception import load_model, predict

app = FastAPI(title="Vision Perception Service", version="0.1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("weights/cnn_mnist.pt", device)

class PredictRequest(BaseModel):
    image: List[List[float]] = Field(..., description="28x28 grayscale image (0-1 or  0-255)")

class PredictResponse(BaseModel):
    digit: int
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        # basic shape validation
        if len(req.image) != 28 or any(len(row) != 28 for row in req.image):
            raise ValueError("image must be 28x28")
        digit, conf = predict(model, req.image, device)
        return PredictResponse(digit=digit, confidence=conf)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    