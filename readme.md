# 🚀 Project Overview

This repository implements a production-ready MLOps pipeline for predicting financial transactions using Santander dataset.
It demonstrates full ML lifecycle practices, including:
  * Reproducible data processing \& feature engineering
  * SHAP-based interaction feature discovery
  * Feature store for persistent feature artifacts
  * Cross-validation training with LightGBM and CatBoost
  * Experiment tracking and model registry with MLflow
  * Batch and real-time prediction pipelines via FastAPI
  * The system is designed to be reproducible, auditable, and deployable.
# 📂 Project Structure

```
project\_root/

│
├── src/
│   ├── train.py                # Training pipeline with CV + final model
│   ├── preproc/
│   │   ├── predict.py          # Batch inference pipeline
│   │   └── data\_validation.py # Data validation scripts
│   ├── features/
│   │   ├── feature\_builder.py  # Feature engineering class
│   │   ├── feature\_store.py    # Feature persistence
│   │   └── interactions.py     # SHAP interaction features
│   ├── api/
│   │   └── app.py              # FastAPI real-time API
│   └── utils/
│       ├── config.py           # Global configuration
│       └── seed.py             # Seed utility
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── artifacts/                  # Generated feature artifacts
│   └── feature\_builder.pkl
│
├── models/                     # Trained model artifacts
│   ├── lgb\_final.pkl
│   └── cat\_final.pkl
│
├── feature\_store/              # Persistent transformed features
│   ├── features.parquet
│   ├── freq\_maps.pkl
│   └── interactions.pkl
│
├── configs/
│   └── config.yaml             # YAML configuration for pipelines
│
└── README.md

```
## 🧠 System Architecture

<img width="250" height="700" alt="system_ architecture" src="https://github.com/user-attachments/assets/8914bd03-547e-41da-b70b-adb2681872f0" />

* Features → interactions → CV → final models → predictions
* All artifacts and metrics logged to MLflow

## ⚙️ Installation
1. Clone the repository:

```
git clone <repo_url>
cd project_root
```
2. Create and activate a Python environment:

```
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Ensure data files are available:

https://www.kaggle.com/c/santander-customer-transaction-prediction
```
data/train.csv
data/test.csv
```

## 📝 Configuration
* All global paths, seeds, and experiment settings are centralized in src/utils/config.py
* Ensures reproducibility, consistent artifact paths, and separation of training vs API experiments

## 🛠 Feature Engineering

**FeatureBuilder**
* Computes row statistics: mean, std, min, max, sum
* Applies frequency encoding for categorical-like numeric features
* Handles interaction features
* Save/load for reproducible transformations

**InteractionBuilder**
* Uses SHAP values to select top features and interaction pairs
* Generates multiplicative and ratio-based interaction features
* Reduces feature space via SHAP pruning

**FeatureStore**
* Saves processed features, frequency maps, and feature lists to disk
* Enables consistent feature recovery for training and inference

## ✅ Data Validation
* Validates schema and missing columns
* Computes statistics and detects outliers
* Logs results to MLflow
```
python src/preproc/data\_validation.py --data\_path data/train.csv
```
## 🎯 Model Training
* Loads config.yaml and sets seed (utils.seed.set\_seed)
* Generates features and interactions
* Saves features to FeatureStore
* Performs cross-validation (LightGBM + CatBoost)
* Logs fold metrics to MLflow
* Saves CV models locally and to MLflow
* Trains final models on full data
* Registers models in MLflow:
```
mlflow.lightgbm.log_model(final_lgb, artifact_path="lgb_final_model", registered_model_name="LGBM_Santander")
mlflow.catboost.log_model(final_cat, artifact_path="cat_final_model", registered_model_name="CatBoost_Santander")
```

## 📦 Batch Inference
* Loads FeatureBuilder, FeatureStore, and InteractionBuilder
* Aligns test features with training columns
* Performs predictions using LightGBM + CatBoost ensemble
* Saves predictions to CSV and logs metrics to MLflow
```
python src/preproc/predict.py --test_path data/test.csv
```

## ⚡ Real-Time API

FastAPI REST API serving ensemble predictions
Start API
```
uvicorn src/api/app:app --reload --host 0.0.0.0 --port 8000
```
Endpoints
GET  – Health check
POST /predict – Predict ensemble probability
Request Example:
```
{
"features": {
  "var_0": 0.12,
   "var_1": -1.34,
    ...
 }
}
```
Response Example:
```
{

"prediction": 0.8234,
"timestamp": "2026-03-12T14:35:12.123456",
"model_version": "final"
}
```
Logs inputs and predictions to MLflow for monitoring and audit.
## 🔄 Reproducibility
* utils.seed.set_seed(seed) ensures deterministic results
* FeatureBuilder + FeatureStore guarantees consistent features
* MLflow logs provide full traceability of experiments and predictions
##⚡ Quick Start
```
# Train models
python src/preproc/train.py --config configs/config.yaml
# Validate data
python src/preproc/data_validation.py --data_path data/test.csv
# Run batch predictions
python src/preproc/predict.py --test_path data/test.csv
# Start API for real-time predictions
uvicorn src/api/app:app --reload --host 0.0.0.0 --port 8000
```

