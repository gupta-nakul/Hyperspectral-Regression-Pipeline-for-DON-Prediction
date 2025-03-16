import json
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from src.data.preprocess import Preprocessor
from src.deployment.predictor import Predictor
from src.utils.logger import get_logger
from src.utils.load_config import config

logger = get_logger(__name__)

class SampleData(BaseModel):
    sample: list

app = FastAPI(
    title="DON Prediction API",
    description="API to predict DON concentration in corn samples"
)

with open("models/best_model_info.json", "r") as f:
    info = json.load(f)
best_model_type = info["best_model_type"]

if best_model_type == "xgboost":
    best_model_path = "models/best_model.pkl"
elif best_model_type in ["cnn", "fcn", "transformer"]:
    best_model_path = "models/best_model.pt"

predictor_best = Predictor(model_type=best_model_type, model_path=best_model_path)

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/predict")
def predict_don(data: SampleData):
    sample = np.array(data.sample).reshape(1, -1)
    logger.info(f"Received sample data: {sample}")
    X_sample = np.array(sample[0, :-1]).reshape(1, -1)
    _ = sample[:, -1]
    logger.info(f"Received sample data: {X_sample, _}")
    preprocessor = Preprocessor(impute_strategy=config["preprocess"]["impute_strategy"], 
                                scale_data=config["preprocess"]["scaler"])
    sample = preprocessor.fit_transform(X_sample)
    prediction = predictor_best.predict(X_sample)
    return {"DON Concentration": float(prediction[0])}
