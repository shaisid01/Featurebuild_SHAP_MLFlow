import pandas as pd
import numpy as np
from itertools import combinations
import shap
import lightgbm as lgb
from catboost import CatBoostClassifier

class InteractionBuilder:
    """
    Builds SHAP-based interaction features from top features.
    """

    def __init__(self, top_n=10, max_interactions=25, random_state=42):
        self.top_n = top_n
        self.max_interactions = max_interactions
        self.random_state = random_state
        self.interaction_pairs = []

    def fit(self, X: pd.DataFrame, model_type="lgb"):
        """
        Fit the interaction builder by computing SHAP values and selecting top interactions.

        Args:
            X (pd.DataFrame): Input feature dataframe
            model_type (str): "lgb" or "cat" to determine baseline SHAP model
        """
        X_sample = X.sample(min(1000, len(X)), random_state=self.random_state)

        # Train baseline model
        if model_type == "lgb":
            model = lgb.LGBMClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1)
        elif model_type == "cat":
            model = CatBoostClassifier(iterations=200, verbose=False, random_seed=self.random_state)
        else:
            raise ValueError("model_type must be 'lgb' or 'cat'")

        model.fit(X, np.zeros(len(X)))  # dummy target, just for SHAP

        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):  # binary classification
            shap_values = shap_values[1]

        # Compute mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)

        # Select top features
        top_features = shap_df["feature"].head(self.top_n).tolist()

        # Compute interaction strengths (absolute correlation of SHAP values)
        shap_vals_top = shap_values[:, [X.columns.get_loc(f) for f in top_features]]
        pairs = []
        for i, j in combinations(range(len(top_features)), 2):
            f1, f2 = top_features[i], top_features[j]
            corr = np.abs(np.corrcoef(shap_vals_top[:, i], shap_vals_top[:, j])[0, 1])
            pairs.append((f1, f2, corr))

        # Select top max_interactions
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:self.max_interactions]
        self.interaction_pairs = [(f1, f2) for f1, f2, _ in pairs]

    def transform(self, df: pd.DataFrame):
        """
        Create interaction features based on fitted pairs.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with interaction features appended
        """
        df = df.copy()
        features = []
        for f1, f2 in self.interaction_pairs:
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
            df[f"{f1}_ratio_{f2}"] = df[f1] / (df[f2] + 1e-5)
            features.extend([f"{f1}_x_{f2}", f"{f1}_ratio_{f2}"])
        self.features = features
        return df