import os
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import mlflow
import mlflow.lightgbm
import mlflow.catboost
import collections
import argparse
import asyncio
from pathlib import Path

from src.features.feature_builder import FeatureBuilder
from src.features.interactions import InteractionBuilder
from src.features.feature_store import FeatureStore
from src.utils.config import ARTIFACT_DIR, MODEL_DIR, RANDOM_STATE
from src.utils.seed import set_seed

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -------------------------
# CLI / Config
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

# -------------------------
# Set seeds
# -------------------------
seed = config.get("seed", RANDOM_STATE)
set_seed(seed)
np.random.seed(seed)

# -------------------------
# Directories
# -------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs("feature_store", exist_ok=True)

# -------------------------
# Load data
# -------------------------
train_df = pd.read_csv(config["data"]["train_path"])
TARGET = config.get("target", "target")

object_cols = train_df.select_dtypes(include="object").columns.tolist()
if object_cols:
    print(f"Dropping object columns: {object_cols}")
    train_df = train_df.drop(columns=object_cols)

y = train_df[TARGET].astype(int)
X = train_df.drop(columns=[TARGET])

# -------------------------
# Feature engineering
# -------------------------
interaction_builder = InteractionBuilder(
    top_n=config["features"].get("top_shap_features", 10),
    max_interactions=config["features"].get("top_interaction_pairs", 25)
)

interaction_builder.fit(X, model_type="lgb")
X_interactions = interaction_builder.transform(X).add_prefix("int_")

feature_builder = FeatureBuilder()
feature_builder.fit(X)

# Frequency encoding
X_freq = pd.DataFrame({
    f"{col}_freq": X[col].map(feature_builder.freq_maps[col]).fillna(0)
    for col in feature_builder.base_features
}, index=X.index)

# Row stats
X_stats = pd.DataFrame({
    "row_mean": X[feature_builder.base_features].mean(axis=1),
    "row_std": X[feature_builder.base_features].std(axis=1),
    "row_min": X[feature_builder.base_features].min(axis=1),
    "row_max": X[feature_builder.base_features].max(axis=1),
    "row_sum": X[feature_builder.base_features].sum(axis=1)
}, index=X.index)

X = pd.concat([X, X_freq, X_stats, X_interactions], axis=1)

feature_builder.features = list(X.columns)
feature_builder.save(os.path.join(ARTIFACT_DIR, "feature_builder.pkl"))

# Save features to FeatureStore
fs = FeatureStore()
fs.save_features(X, feature_builder)

# Drop zero variance columns
zero_var_cols = X.columns[X.var() == 0].tolist()
if zero_var_cols:
    print(f"Dropping zero-variance columns: {zero_var_cols}")

X = X.loc[:, X.var() > 0]

# -------------------------
# Cross-validation setup
# -------------------------
folds = config["cv"].get("folds", 3)
shuffle = config["cv"].get("shuffle", True)

skf = StratifiedKFold(
    n_splits=folds,
    shuffle=shuffle,
    random_state=seed
)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

lgb_models = []
cat_models = []

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(config["mlflow"]["experiment_name"])

artifact_loc = os.path.abspath("mlruns")
experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])

# -------------------------
# CV Run
# -------------------------
with mlflow.start_run(run_name="CV_Run"):

    mlflow.set_tag("mlflow.experimentName", experiment.name)
    mlflow.set_tag("mlflow.note.content",
                   "Cross-validation run with fold metrics and hyperparameters")

    # Log code + config
    mlflow.log_artifacts("src")
    mlflow.log_artifact(args.config)

    root_dir = Path(__file__).parents[2]
    req_path = root_dir / "requirements.txt"

    if req_path.exists():
        mlflow.log_artifact(str(req_path))
    else:
        print(f"Warning: {req_path} not found")

    # -------------------------
    # Log parameters
    # -------------------------
    mlflow.log_params({f"lgb_{k}": v for k, v in config["model"]["lgb_params"].items()})
    mlflow.log_params({f"cat_{k}": v for k, v in config["model"]["cat_params"].items()})

    mlflow.log_param("cv_folds", folds)
    mlflow.log_param("cv_shuffle", shuffle)
    mlflow.log_param("seed", seed)

    # Enable autolog but avoid duplicate model logging
    #mlflow.lightgbm.autolog(log_models=False)

    # -------------------------
    # CV Loop
    # -------------------------
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # -------------------------
        # LightGBM
        # -------------------------
        lgb_model = lgb.LGBMClassifier(**config["model"]["lgb_params"])

        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50)]
        )

        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        lgb_models.append(lgb_model)

        fold_lgb_auc = float(roc_auc_score(y_val, oof_lgb[val_idx]))
        mlflow.log_metric(f"lgb_auc_fold_{fold}", fold_lgb_auc)

        # Save model
        lgb_fold_path = os.path.join(MODEL_DIR, f"lgb_fold_{fold}.pkl")
        joblib.dump(lgb_model, lgb_fold_path)

        # Log to MLflow
        mlflow.lightgbm.log_model(
            lgb_model,
            artifact_path=f"lgb_fold_{fold}"
        )

        mlflow.log_artifact(lgb_fold_path, artifact_path="lgb_fold_models")

        # -------------------------
        # CatBoost
        # -------------------------
        cat_model = CatBoostClassifier(**config["model"]["cat_params"])

        cat_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )

        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        cat_models.append(cat_model)

        fold_cat_auc = float(roc_auc_score(y_val, oof_cat[val_idx]))
        mlflow.log_metric(f"cat_auc_fold_{fold}", fold_cat_auc)

        # Save model
        cat_fold_path = os.path.join(MODEL_DIR, f"cat_fold_{fold}.pkl")
        joblib.dump(cat_model, cat_fold_path)

        cat_native_path = cat_fold_path.replace(".pkl", ".cbm")
        cat_model.save_model(cat_native_path)

        mlflow.catboost.log_model(
            cat_model,
            artifact_path=f"cat_fold_{fold}"
        )

        mlflow.log_artifact(cat_fold_path, artifact_path="cat_fold_models")
        mlflow.log_artifact(cat_native_path, artifact_path="cat_fold_models")

    # -------------------------
    # Log overall CV metrics
    # -------------------------
    mlflow.log_metric("lgb_auc", float(roc_auc_score(y, oof_lgb)))
    mlflow.log_metric("cat_auc", float(roc_auc_score(y, oof_cat)))

    print("✅ CV completed. LGB CV AUC:", float(roc_auc_score(y, oof_lgb)))
    print("✅ CV completed. CAT CV AUC:", float(roc_auc_score(y, oof_cat)))

    # -------------------------
    # Save feature artifacts
    # -------------------------
    features_parquet = os.path.join(ARTIFACT_DIR, "features.parquet")
    X.to_parquet(features_parquet)

    mlflow.log_artifact(features_parquet)
    mlflow.log_dict({"features": list(X.columns)}, "feature_schema.json")

    fb_path = os.path.join(ARTIFACT_DIR, "feature_builder.pkl")
    mlflow.log_artifact(fb_path)

    int_path = os.path.join("feature_store", "interactions.pkl")
    mlflow.log_artifact(int_path)

# -------------------------
# Final Model Run
# -------------------------
with mlflow.start_run(run_name="Final_Run"):

    mlflow.set_tag("mlflow.experimentName", experiment.name)
    mlflow.set_tag("mlflow.note.content",
                   "Final LightGBM & CatBoost models trained on full data with engineered features")

    mlflow.log_params({f"lgb_{k}": v for k, v in config["model"]["lgb_params"].items()})
    mlflow.log_params({f"cat_{k}": v for k, v in config["model"]["cat_params"].items()})

    # -------------------------
    # Train final models
    # -------------------------
    final_lgb = lgb.LGBMClassifier(**config["model"]["lgb_params"])
    final_lgb.fit(X, y)

    final_cat = CatBoostClassifier(**config["model"]["cat_params"])
    final_cat.fit(X, y)

    # -------------------------
    # Evaluate on full data
    # -------------------------
    oof_lgb_pred = final_lgb.predict_proba(X)[:, 1]
    oof_cat_pred = final_cat.predict_proba(X)[:, 1]

    mlflow.log_metric("lgb_full_auc", float(roc_auc_score(y, oof_lgb_pred)))
    mlflow.log_metric("cat_full_auc", float(roc_auc_score(y, oof_cat_pred)))

    # -------------------------
    # Save models locally (same as CV)
    # -------------------------
    final_lgb_path = os.path.join(MODEL_DIR, "lgb_final.pkl")
    joblib.dump(final_lgb, final_lgb_path)

    final_cat_path = os.path.join(MODEL_DIR, "cat_final.pkl")
    joblib.dump(final_cat, final_cat_path)

    final_cat_native = final_cat_path.replace(".pkl", ".cbm")
    final_cat.save_model(final_cat_native)

    # -------------------------
    # Log local artifacts
    # -------------------------
    mlflow.log_artifact(final_lgb_path, artifact_path="final_models")
    mlflow.log_artifact(final_cat_path, artifact_path="final_models")
    mlflow.log_artifact(final_cat_native, artifact_path="final_models")

    # -------------------------
    # Log MLflow models
    # -------------------------
    mlflow.lightgbm.log_model(
        final_lgb,
        artifact_path="lgb_final_model",
        registered_model_name="LGBM_Santander"
    )

    mlflow.catboost.log_model(
        final_cat,
        artifact_path="cat_final_model",
        registered_model_name="CatBoost_Santander"
    )
    
    