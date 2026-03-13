import pandas as pd
import numpy as np
import joblib


class FeatureBuilder:
    def __init__(self):
        self.freq_maps = {}
        self.base_features = []
        self.features = []

    # -------------------------
    # Fit on training data
    # -------------------------
    def fit(self, df: pd.DataFrame):
        self.base_features = [c for c in df.columns if "var" in c]

        for col in self.base_features:
            freq = df[col].value_counts(normalize=True)
            self.freq_maps[col] = freq

        return self

    # -------------------------
    # Transform for training/inference
    # -------------------------
    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # Add missing base features
        for col in self.base_features:
            if col not in df.columns:
                df[col] = 0

        # Row stats
        df["row_mean"] = df[self.base_features].mean(axis=1)
        df["row_std"] = df[self.base_features].std(axis=1)
        df["row_min"] = df[self.base_features].min(axis=1)
        df["row_max"] = df[self.base_features].max(axis=1)
        df["row_sum"] = df[self.base_features].sum(axis=1)

        stats_cols = ["row_mean", "row_std", "row_min", "row_max", "row_sum"]
        df[stats_cols] = df[stats_cols].replace([np.inf, -np.inf], 0).fillna(0)

        # Frequency encoding
        for col in self.base_features:
            df[f"{col}_freq"] = df[col].map(self.freq_maps[col]).fillna(0)

        # Include interaction features if any
        if hasattr(self, "interaction_features"):
            for col in self.interaction_features:
                if col not in df.columns:
                    df[col] = 0

        # Make final feature list
        features = self.features  # already includes base + stats + freq + interactions
        for col in features:
            if col not in df.columns:
                df[col] = 0

        return df[features]

    # -------------------------
    # Save FeatureBuilder
    # -------------------------
    def save(self, path="artifacts/feature_builder.pkl"):
        joblib.dump(self, path)

    # -------------------------
    # Load FeatureBuilder
    # -------------------------
    @staticmethod
    def load(path="artifacts/feature_builder.pkl"):
        return joblib.load(path)