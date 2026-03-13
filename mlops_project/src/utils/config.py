import os

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

TARGET = "target"
ID_COL = "ID_code"

RANDOM_STATE = 42

TOP_SHAP_INTERACTIONS = 25

MLFLOW_EXPERIMENT_NAME = "MLOps_Santander_Trans_Pred9"

# config.py
MLFLOW_EXPERIMENT_NAME1 = "SantanderMLOps_API_Predictions"
MODEL_DIR1 = "/project_root/models"
ARTIFACT_DIR1 = "/project_root/artifacts"
FINAL_LGB_MODE1L = f"{MODEL_DIR}/lgb_final.pkl"
FINAL_CAT_MODEL1 = f"{MODEL_DIR}/cat_final.pkl"
FEATURE_BUILDER1 = f"{ARTIFACT_DIR}/feature_builder.pkl"