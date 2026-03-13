from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import mlflow

from src.preproc.predict import run_batch_prediction  # refactored predict.py
from src.features.feature_store import FeatureStore
from src.features.interactions import InteractionBuilder
from utils.config import ARTIFACT_DIR, MODEL_DIR, MLFLOW_EXPERIMENT_NAME

# -------------------------
# DAG default arguments
# -------------------------
default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
}

# -------------------------
# Define DAG
# -------------------------
dag = DAG(
    dag_id="santander_batch_prediction",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    description="Batch ML pipeline for Santander predictions",
)

# -------------------------
# Tasks
# -------------------------

def load_test_data(**kwargs):
    test_path = "/data/processed/test.csv"
    df = pd.read_csv(test_path)
    return df

def validate_data(df: pd.DataFrame, **kwargs):
    # Example: basic column check
    expected_cols = joblib.load(f"{ARTIFACT_DIR}/feature_builder.pkl").features
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")
    return df

def transform_features(df: pd.DataFrame, **kwargs):
    # Load FeatureBuilder and InteractionBuilder
    fb = joblib.load(f"{ARTIFACT_DIR}/feature_builder.pkl")
    ib = InteractionBuilder()
    ib.interaction_pairs = joblib.load("feature_store/interactions.pkl")

    # Transform features
    X = fb.transform(df)
    X = ib.transform(X)
    return X

def ensemble_predict(X: pd.DataFrame, **kwargs):
    # Load models
    lgb_model = joblib.load(f"{MODEL_DIR}/lgb_final.pkl")
    cat_model = joblib.load(f"{MODEL_DIR}/cat_final.pkl")

    # Predict
    pred_lgb = lgb_model.predict_proba(X)[:, 1]
    pred_cat = cat_model.predict_proba(X)[:, 1]

    pred = 0.5 * pred_lgb + 0.5 * pred_cat
    return pred

def save_submission(pred, df, **kwargs):
    submission = pd.DataFrame({
        "ID_code": df["ID_code"],
        "target": pred
    })
    output_path = "/data/submission.csv"
    submission.to_csv(output_path, index=False)

    # MLflow logging
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    timestamp = datetime.utcnow().isoformat()
    with mlflow.start_run(run_name=f"batch_prediction_{timestamp}"):
        mlflow.log_param("num_rows", len(df))
        mlflow.log_metric("mean_prediction", float(np.mean(pred)))
        mlflow.log_metric("max_prediction", float(np.max(pred)))
        mlflow.log_metric("min_prediction", float(np.min(pred)))
        mlflow.set_tag("prediction_type", "batch")
        mlflow.set_tag("model_version", "final")
        mlflow.log_artifact(output_path)

    print(f"✅ Submission saved at {output_path}")

# -------------------------
# Operators
# -------------------------
load_data_task = PythonOperator(
    task_id="load_test_data",
    python_callable=load_test_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    op_kwargs={"df": "{{ task_instance.xcom_pull(task_ids='load_test_data') }}"},
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_features",
    python_callable=transform_features,
    op_kwargs={"df": "{{ task_instance.xcom_pull(task_ids='validate_data') }}"},
    dag=dag,
)

predict_task = PythonOperator(
    task_id="ensemble_predict",
    python_callable=ensemble_predict,
    op_kwargs={"X": "{{ task_instance.xcom_pull(task_ids='transform_features') }}"},
    dag=dag,
)

save_task = PythonOperator(
    task_id="save_submission",
    python_callable=save_submission,
    op_kwargs={
        "pred": "{{ task_instance.xcom_pull(task_ids='ensemble_predict') }}",
        "df": "{{ task_instance.xcom_pull(task_ids='load_test_data') }}"
    },
    dag=dag,
)

# -------------------------
# Task dependencies
# -------------------------
load_data_task >> validate_task >> transform_task >> predict_task >> save_task