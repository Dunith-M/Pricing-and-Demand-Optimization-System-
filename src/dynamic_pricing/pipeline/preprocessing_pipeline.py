import os
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.dynamic_pricing.components.target_definition import TargetDefinition


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "price" in X.columns:
            X["price_log"] = np.log1p(X["price"].clip(lower=0))

        if "price" in X.columns and "minimum nights" in X.columns:
            X["price_per_minimum_night"] = X["price"] / (X["minimum nights"] + 1)

        if "room type" in X.columns and "neighbourhood group" in X.columns:
            X["room_type_neighbourhood"] = (
                X["room type"].astype(str) + "_" + X["neighbourhood group"].astype(str)
            )

        return X


def replace_inf_with_nan(X):
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan)

    X = np.asarray(X).copy()
    X[np.isinf(X)] = np.nan
    return X


def prepare_categorical_data(X):
    if isinstance(X, pd.DataFrame):
        return X.fillna("missing").astype(str)

    return pd.DataFrame(X).fillna("missing").astype(str)


class PreprocessingPipeline:
    def __init__(self):
        self.config = {
            "data_transformation": {
                "test_size": 0.2,
                "random_state": 42,
            }
        }

    def load_data(self):
        return pd.read_csv("data/processed/feature_engineered_data.csv")

    def remove_leakage_features(self, df):
        leakage_cols = ["reviews per month", "availability 365"]
        existing_cols = [col for col in leakage_cols if col in df.columns]

        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"[INFO] Removed leakage columns: {existing_cols}")

        return df

    def build_pipeline(self, X):
        categorical_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        num_pipeline = Pipeline([
            ("replace_inf", FunctionTransformer(replace_inf_with_nan, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("to_string", FunctionTransformer(prepare_categorical_data, validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ])

        return Pipeline([
            ("feature_engineering", FeatureEngineering()),
            ("preprocessor", preprocessor),
        ])

    def run(self):
        print("[INFO] Loading data...")
        df = self.load_data()

        print("[INFO] Creating demand_score target...")
        target_definer = TargetDefinition(config=self.config)
        X_train, X_test, y_train, y_test = target_definer.run(df)

        print("[INFO] Removing leakage features...")
        X_train = self.remove_leakage_features(X_train)
        X_test = self.remove_leakage_features(X_test)

        print("[INFO] Building preprocessing pipeline...")
        pipeline = self.build_pipeline(X_train)

        print("[INFO] Applying transformations...")
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        os.makedirs("artifacts/preprocessing", exist_ok=True)
        joblib.dump(pipeline, "artifacts/preprocessing/preprocessor.pkl")

        os.makedirs("artifacts/data/processed", exist_ok=True)
        np.save("artifacts/data/processed/X_train.npy", X_train_transformed)
        np.save("artifacts/data/processed/X_test.npy", X_test_transformed)
        np.save("artifacts/data/processed/y_train.npy", y_train.values)
        np.save("artifacts/data/processed/y_test.npy", y_test.values)

        print("[INFO] Preprocessing + Target pipeline completed successfully")


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
