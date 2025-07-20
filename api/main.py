# main.py (or /api/main.py if using Vercel's API folder structure)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import json
import os
import joblib

app = FastAPI()

# --- Input schema ---
class GameInput(BaseModel):
    home_team: str
    away_team: str
    features: Dict[str, float]  # dynamic input features

# --- Load metadata and model rebuild function ---
with open("model_metadata.txt", "r") as f:
    metadata = json.load(f)

features_used = metadata["features_used"]
weights = np.array(metadata["weights"])
intercept = metadata.get("intercept", 0.0)

# Optional thresholds
TRAP_THRESHOLD = -3.0
SHARP_THRESHOLD = 3.0

# --- Model logic ---
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

# --- POST endpoint ---
@app.post("/predict")
async def predict_game(input_data: GameInput):
    try:
        prediction = predict_outcome(input_data.features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

