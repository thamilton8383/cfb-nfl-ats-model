from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import json
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Vercel!"}

class GameInput(BaseModel):
    home_team: str
    away_team: str
    features: Dict[str, float]

meta_path = os.path.join(os.path.dirname(__file__), "model_metadata.txt")

with open(meta_path, "r") as f:
    metadata = json.load(f)

features_used = metadata["features_used"]
weights = np.array(metadata["weights"])
intercept = metadata.get("intercept", 0.0)

TRAP_THRESHOLD = -3.0
SHARP_THRESHOLD = 3.0

def predict_outcome(features: Dict[str, float]) -> Dict[str, Any]:
    x = np.array([features[feat] for feat in features_used]).reshape(1, -1)
    logits = np.dot(x, weights) + intercept
    prob = 1 / (1 + np.exp(-logits))

    confidence_pct = round(float(prob * 100), 1) if prob >= 0.5 else round(float((1 - prob) * 100), 1)
    model_pick = "away" if prob >= 0.5 else "home"

    edge_score = round(float(logits[0]), 2)
    trap_signal = edge_score < TRAP_THRESHOLD
    sharp_signal = edge_score > SHARP_THRESHOLD

    return {
        "model_pick": f"{model_pick} against the spread",
        "confidence_pct": confidence_pct,
        "edge_score": edge_score,
        "trap_signal": trap_signal,
        "sharp_signal": sharp_signal
    }

@app.post("/predict")
async def predict_game(input_data: GameInput):
    try:
        prediction = predict_outcome(input_data.features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

handler = app
