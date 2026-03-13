# src/preproc/predict.py
import pandas as pd
import numpy as np
import joblib
import mlflow
import datetime
import argparse

from src.features.feature_store import FeatureStore
from src.features.interactions import InteractionBuilder
from src.utils.config import ARTIFACT_DIR, MODEL_DIR, MLFLOW_EXPERIMENT_NAME

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -------------------------
# CLI / arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, required=True)
args = parser.parse_args()

# -------------------------
# Load test data
# -------------------------
test_df = pd.read_csv(args.test_path)

# -------------------------
# Load FeatureBuilder
# -------------------------
feature_builder = joblib.load(f"{ARTIFACT_DIR}/feature_builder.pkl")

# -------------------------
# Load Features from FeatureStore (transformed DF)
# -------------------------
fs = FeatureStore()
X_test = fs.load_features(return_builder=False)

# -------------------------
# Load Interaction Features
# -------------------------
interaction_builder = InteractionBuilder()
interaction_builder.interaction_pairs = joblib.load("feature_store/interactions.pkl")
X_test = interaction_builder.transform(X_test)

# -------------------------
# Align test features with training
# -------------------------
training_cols = feature_builder.features
# Add missing columns
for col in training_cols:
    if col not in X_test.columns:
        X_test[col] = 0
# Drop extra columns
X_test = X_test[training_cols]

# -------------------------
# Basic data validation
# -------------------------
def validate_data(df, fb):
    missing_cols = [col for col in fb.base_features if col not in df.columns]
    stats = {}
    anomalies = {}
    for col in fb.base_features:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        stats[col] = {"mean": mean, "std": std, "min": min_val, "max": max_val}
        outliers = ((df[col] - mean).abs() > 3 * std).sum()
        if outliers > 0:
            anomalies[col] = outliers
    return missing_cols, stats, anomalies

missing_cols, stats, anomalies = validate_data(X_test, feature_builder)
print(f"✅ Validation complete. Missing columns: {missing_cols}")
if anomalies:
    print(f"⚠️ Outliers detected in columns: {anomalies}")

# -------------------------
# Load Models
# -------------------------
lgb_models = [joblib.load(f"{MODEL_DIR}/lgb_final.pkl")]
cat_models = [joblib.load(f"{MODEL_DIR}/cat_final.pkl")]

# -------------------------
# Ensemble Predictions
# -------------------------
pred_lgb = np.mean([m.predict_proba(X_test)[:,1] for m in lgb_models], axis=0)
pred_cat = np.mean([m.predict_proba(X_test)[:,1] for m in cat_models], axis=0)
pred_ensemble = 0.5 * pred_lgb + 0.5 * pred_cat

# -------------------------
# Save Submission
# -------------------------
submission = pd.DataFrame({
    "ID_code": test_df["ID_code"],
    "target": pred_ensemble
})
submission_file = f"submission_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
submission.to_csv(submission_file, index=False)
print(f"✅ Predictions saved to {submission_file}")

# -------------------------
# Log predictions + validation to MLflow
# -------------------------
with mlflow.start_run(run_name=f"batch_prediction_{datetime.datetime.utcnow().isoformat()}"):
    mlflow.log_param("test_path", args.test_path)
    mlflow.log_param("num_rows", X_test.shape[0])
    mlflow.log_param("num_features", X_test.shape[1])
    
     # Log missing columns and validation stats
    mlflow.log_param("missing_columns", missing_cols)
    mlflow.log_dict(stats, "column_stats.json")
    mlflow.log_dict(anomalies, "outliers.json")
    
    # Log prediction summary metrics
    mlflow.log_metric("mean_prediction", float(np.mean(pred_ensemble)))
    mlflow.log_metric("median_prediction", float(np.median(pred_ensemble)))
    mlflow.log_metric("min_prediction", float(np.min(pred_ensemble)))
    mlflow.log_metric("max_prediction", float(np.max(pred_ensemble)))
    
    # Tags
    mlflow.set_tag("prediction_type", "batch")
    mlflow.set_tag("model_version", "final")
    
    mlflow.log_artifact(submission_file, artifact_path="predictions")
    mlflow.log_param("timestamp", datetime.datetime.utcnow().isoformat())
print("📊 Batch inference logged to MLflow successfully ✅")