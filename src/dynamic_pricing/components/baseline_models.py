# src/airbnb_price/components/baseline_models.py

import os
import numpy as np
import joblib
import json

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaselineModels:
    def __init__(self):
        self.model_dir = "artifacts/models/baseline"
        self.report_dir = "artifacts/reports"

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def load_data(self):
        print("[INFO] Loading processed data...")

        X_train = np.load("artifacts/data/processed/X_train.npy", allow_pickle=True)
        X_test = np.load("artifacts/data/processed/X_test.npy", allow_pickle=True)
        y_train = np.load("artifacts/data/processed/y_train.npy")
        y_test = np.load("artifacts/data/processed/y_test.npy")

        if isinstance(X_train, np.ndarray) and X_train.dtype == object and X_train.shape == ():
            X_train = X_train.item()
        if isinstance(X_test, np.ndarray) and X_test.dtype == object and X_test.shape == ():
            X_test = X_test.item()

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        else:
            X_train = np.asarray(X_train)

        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        else:
            X_test = np.asarray(X_test)

        print(f"[INFO] X_train shape: {X_train.shape}")
        print(f"[INFO] X_test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2)
        }

    def train_linear_regression(self, X_train, y_train):
        print("[INFO] Training Linear Regression...")

        model = LinearRegression()
        model.fit(X_train, y_train)

        return model

    def train_ridge(self, X_train, y_train):
        print("[INFO] Training Ridge Regression...")

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        return model

    def run(self):
        print("[INFO] Starting baseline modeling...")

        X_train, X_test, y_train, y_test = self.load_data()

        results = {}

        # -------------------------------
        # Linear Regression
        # -------------------------------
        lr_model = self.train_linear_regression(X_train, y_train)
        lr_preds = lr_model.predict(X_test)

        lr_metrics = self.evaluate(y_test, lr_preds)
        results["LinearRegression"] = lr_metrics

        joblib.dump(lr_model, os.path.join(self.model_dir, "linear_regression.pkl"))

        print("[RESULT] Linear Regression:", lr_metrics)

        # -------------------------------
        # Ridge Regression
        # -------------------------------
        ridge_model = self.train_ridge(X_train, y_train)
        ridge_preds = ridge_model.predict(X_test)

        ridge_metrics = self.evaluate(y_test, ridge_preds)
        results["Ridge"] = ridge_metrics

        joblib.dump(ridge_model, os.path.join(self.model_dir, "ridge.pkl"))

        print("[RESULT] Ridge:", ridge_metrics)

        # -------------------------------
        # Save results
        # -------------------------------
        with open(os.path.join(self.report_dir, "baseline_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        print("[INFO] Baseline models completed successfully")


if __name__ == "__main__":
    trainer = BaselineModels()
    trainer.run()
