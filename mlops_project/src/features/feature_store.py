import os
import joblib
import json
import pandas as pd

class FeatureStore:

    def __init__(self, store_path="feature_store"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)

    def save_features(self, df: pd.DataFrame, feature_builder):
        """
        Save features and metadata for reproducibility.

        Args:
            df (pd.DataFrame): Full feature dataframe
            feature_builder: FeatureBuilder instance (contains freq_maps and feature list)
        """
        # Save feature dataframe
        df.to_parquet(os.path.join(self.store_path, "features.parquet"), index=False)

        # Save frequency maps for reproducibility
        joblib.dump(feature_builder.freq_maps, os.path.join(self.store_path, "freq_maps.pkl"))

        # Save feature list
        with open(os.path.join(self.store_path, "features.json"), "w") as f:
            json.dump(feature_builder.features, f)

        print(f"✅ Features saved to {self.store_path}")

    def load_features(self, return_builder=False):
        """
        Load features from store.

        Args:
            return_builder (bool): If True, returns a mock FeatureBuilder with freq_maps and features
        """
        # Load dataframe
        df = pd.read_parquet(os.path.join(self.store_path, "features.parquet"))

        if return_builder:
            # Load metadata
            freq_maps = joblib.load(os.path.join(self.store_path, "freq_maps.pkl"))
            with open(os.path.join(self.store_path, "features.json"), "r") as f:
                features = json.load(f)

            # Mock feature builder
            class MockFeatureBuilder:
                pass

            fb = MockFeatureBuilder()
            fb.freq_maps = freq_maps
            fb.features = features

            return df, fb

        return df