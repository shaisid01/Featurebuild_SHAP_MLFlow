# src/preproc/data_validation.py - lightweight, practical data validation script
import pandas as pd
import numpy as np
import joblib
import mlflow
import datetime
from src.utils.config import ARTIFACT_DIR, MLFLOW_EXPERIMENT_NAME

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -------------------------
# Load data
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# -------------------------
# Load FeatureBuilder
# -------------------------
feature_builder = joblib.load(f"{ARTIFACT_DIR}/feature_builder.pkl")

# -------------------------
# Validate schema
# -------------------------
def validate_schema(df: pd.DataFrame):
    missing_cols = [col for col in feature_builder.base_features if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
    else:
        print("✅ All expected columns present.")
    return missing_cols

# -------------------------
# Validate statistics
# -------------------------
def validate_statistics(df: pd.DataFrame, threshold_std=3):
    stats = {}
    anomalies = {}
    for col in feature_builder.base_features:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        stats[col] = {"mean": mean, "std": std, "min": min_val, "max": max_val}

        # Simple outlier detection
        outliers = ((df[col] - mean).abs() > threshold_std * std).sum()
        if outliers > 0:
            anomalies[col] = outliers

    print(f"✅ Column stats computed for {len(feature_builder.base_features)} features.")
    if anomalies:
        print(f"⚠️ Columns with outliers: {anomalies}")

    return stats, anomalies

# -------------------------
# Main validation function
# -------------------------
def validate_data(path: str):
    df = load_data(path)
    missing_cols = validate_schema(df)
    stats, anomalies = validate_statistics(df)

    # -------------------------
    # Log validation results to MLflow
    # -------------------------
    with mlflow.start_run(run_name=f"data_validation_{datetime.datetime.utcnow().isoformat()}"):
        mlflow.log_param("data_path", path)
        mlflow.log_param("missing_columns", missing_cols)
        mlflow.log_dict(stats, "column_stats.json")
        mlflow.log_dict(anomalies, "outliers.json")
        mlflow.log_param("timestamp", datetime.datetime.utcnow().isoformat())

    print("✅ Validation run logged to MLflow.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    validate_data(args.data_path)