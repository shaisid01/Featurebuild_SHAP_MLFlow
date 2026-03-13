from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow
import datetime

from utils.config import (
    MLFLOW_EXPERIMENT_NAME,
    ARTIFACT_DIR,
    MODEL_DIR
)

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Santander ML Prediction API")

# -------------------------
# Load artifacts
# -------------------------
try:
    feature_builder = joblib.load(f"{ARTIFACT_DIR}/feature_builder.pkl")
    lgb_model = joblib.load(f"{MODEL_DIR}/lgb_final.pkl")
    cat_model = joblib.load(f"{MODEL_DIR}/cat_final.pkl")
    print("✅ Models and feature builder loaded successfully")

except Exception as e:
    print("❌ Error loading artifacts:", e)
    raise e


# -------------------------
# Input schema
# -------------------------
class PredictionInput(BaseModel):
    features: dict


# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
def root():
    return {"message": "Santander ML Prediction API is running"}


# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict")
def predict(input_data: PredictionInput):

    try:

        # 1️⃣ Convert input to DataFrame
        df = pd.DataFrame([input_data.features])

        # 2️⃣ Feature transformation
        X = feature_builder.transform(df)

        # 3️⃣ Model predictions
        pred_lgb = lgb_model.predict_proba(X)[:, 1]
        pred_cat = cat_model.predict_proba(X)[:, 1]

        # 4️⃣ Ensemble prediction
        pred = 0.5 * pred_lgb + 0.5 * pred_cat
        prediction_value = float(pred[0])

        timestamp = datetime.datetime.utcnow().isoformat()

        # -------------------------
        # MLflow logging
        # -------------------------
        try:
            with mlflow.start_run(run_name=f"prediction_{timestamp}"):

                mlflow.log_dict(input_data.features, "input.json")

                mlflow.log_metric("prediction", prediction_value)

                mlflow.set_tag("model_type_lgb", type(lgb_model).__name__)
                mlflow.set_tag("model_type_cat", type(cat_model).__name__)
                mlflow.set_tag("model_version", "final")

                mlflow.log_param("timestamp", timestamp)

        except Exception as mlflow_error:
            print("⚠️ MLflow logging failed:", mlflow_error)

        # -------------------------
        # Return response
        # -------------------------
        return {
            "prediction": prediction_value,
            "timestamp": timestamp,
            "model_version": "final"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))