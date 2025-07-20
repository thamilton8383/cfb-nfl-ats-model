# api/predict.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import numpy as np
import json
import os

app = FastAPI()

class GameInput(BaseModel):
    home_team: str
    away_team: str
    features: Dict[str, float]

print("💡 Files in this folder:", os.listdir(os.path.dirname(__file__)))

# Load model metadata
meta_path = os.path.join(os.path.dirname(__file__), "model_metadata.txt")

try:
    with open(meta_path, "r") as f:
        metadata = json.load(f)
except Exception as e:
    print("❌ ERROR LOADING METADATA:", e)
    raise RuntimeError("Could not load model metadata file.")

features_used = metadata["features_used"]
weights = np.array(metadata["weights"])
intercept = metadata.get("intercept", 0.0)

@app.post("/")
async def predict_game(input_data: GameInput):
    try:
        x = np.array([input_data.features[feat] for feat in features_used])
        logits = np.dot(x, weights) + intercept
        prob = 1 / (1 + np.exp(-logits))
        model_pick = "away" if prob >= 0.5 else "home"
        confidence_pct = round(float(prob * 100), 1) if prob >= 0.5 else round(float((1 - prob) * 100), 1)
        return {
            "model_pick": model_pick,
            "confidence_pct": confidence_pct
        }
    except Exception as e:
        return {"error": str(e)}

# Required for Vercel
handler = app
