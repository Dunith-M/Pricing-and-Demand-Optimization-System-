import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------
# 🔧 CUSTOM FEATURE ENGINEERING
# -------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Target-related feature (safe for pipeline usage)
        if "reviews per month" in X.columns and "availability 365" in X.columns:
            X["demand score"] = X["reviews per month"] / (X["availability 365"] + 1)

        # Price features
        if "price" in X.columns:
            X["price_log"] = np.log1p(X["price"])

        if "price" in X.columns and "minimum_nights" in X.columns:
            X["price_per_minimum_night"] = X["price"] / X["minimum_nights"]

        # Interaction
        if "room type" in X.columns and "neighbourhood group" in X.columns:
            X["room_type_neighbourhood"] = (
                X["room type"] + "_" + X["neighbourhood group"]
            )

        return X


# -------------------------------
# 🔧 MAIN PIPELINE
# -------------------------------
class PreprocessingPipeline:
    def __init__(self):
        self.target_column = "demand_score"

    def load_data(self):
        train_df = pd.read_csv("artifacts/data/splits/train.csv")
        test_df = pd.read_csv("artifacts/data/splits/test.csv")

        return train_df, test_df

    def build_pipeline(self, X):

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # remove target if present
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)

        # pipelines
        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        full_pipeline = Pipeline([
            ("feature_engineering", FeatureEngineering()),
            ("preprocessor", preprocessor)
        ])

        return full_pipeline

    def run(self):

        train_df, test_df = self.load_data()

        # separate target
        y_train = train_df[self.target_column]
        y_test = test_df[self.target_column]

        X_train = train_df.drop(columns=[self.target_column])
        X_test = test_df.drop(columns=[self.target_column])

        # build pipeline
        pipeline = self.build_pipeline(X_train)

        # fit + transform
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        # save preprocessor
        os.makedirs("artifacts/preprocessing", exist_ok=True)
        joblib.dump(pipeline, "artifacts/preprocessing/preprocessor.pkl")

        # save arrays
        os.makedirs("artifacts/data/processed", exist_ok=True)

        np.save("artifacts/data/processed/X_train.npy", X_train_transformed)
        np.save("artifacts/data/processed/X_test.npy", X_test_transformed)
        np.save("artifacts/data/processed/y_train.npy", y_train.values)
        np.save("artifacts/data/processed/y_test.npy", y_test.values)

        print("[INFO] Pipeline completed successfully")


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()