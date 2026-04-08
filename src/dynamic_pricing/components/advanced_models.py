import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


class AdvancedModels:
    def __init__(self):
        self.model_dir = "artifacts/models/advanced"
        self.report_dir = "artifacts/reports"
        self.random_state = 42
        self.tuning_sample_size = 20000

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def _load_feature_array(self, path):
        arr = np.load(path, allow_pickle=True)

        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            arr = arr.item()

        if hasattr(arr, "toarray"):
            arr = arr.toarray()

        return np.asarray(arr, dtype=np.float32)

    def load_data(self):
        print("[INFO] Loading processed data...")

        X_train = self._load_feature_array("artifacts/data/processed/X_train.npy")
        X_test = self._load_feature_array("artifacts/data/processed/X_test.npy")
        y_train = np.load("artifacts/data/processed/y_train.npy").astype(np.float32)
        y_test = np.load("artifacts/data/processed/y_test.npy").astype(np.float32)

        print(f"[INFO] X_train shape: {X_train.shape}")
        print(f"[INFO] X_test shape: {X_test.shape}")
        print(f"[INFO] X_train dtype: {X_train.dtype}")

        return X_train, X_test, y_train, y_test

    def get_tuning_subset(self, X_train, y_train):
        if len(X_train) <= self.tuning_sample_size:
            return X_train, y_train

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X_train), size=self.tuning_sample_size, replace=False)

        return X_train[indices], y_train[indices]

    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
        }

    def train_random_forest(self, X_train, y_train):
        print("[INFO] Training Random Forest with tuning...")
        X_tune, y_tune = self.get_tuning_subset(X_train, y_train)
        print(f"[INFO] RF tuning subset shape: {X_tune.shape}")

        param_grid = {
            "n_estimators": [100, 150],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }

        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=1,
        )

        grid = GridSearchCV(
            rf,
            param_grid,
            cv=3,
            scoring="r2",
            n_jobs=1,
            pre_dispatch=1,
            verbose=1,
        )

        grid.fit(X_tune, y_tune)

        print("[INFO] Best RF params:", grid.best_params_)

        best_rf = RandomForestRegressor(
            **grid.best_params_,
            random_state=self.random_state,
            n_jobs=1,
        )
        best_rf.fit(X_train, y_train)

        return best_rf

    def train_xgboost(self, X_train, y_train):
        print("[INFO] Training XGBoost with tuning...")
        X_tune, y_tune = self.get_tuning_subset(X_train, y_train)
        print(f"[INFO] XGB tuning subset shape: {X_tune.shape}")

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }

        xgb = XGBRegressor(
            random_state=self.random_state,
            verbosity=0,
            tree_method="hist",
            n_jobs=1,
        )

        grid = GridSearchCV(
            xgb,
            param_grid,
            cv=3,
            scoring="r2",
            n_jobs=1,
            pre_dispatch=1,
            verbose=1,
        )

        grid.fit(X_tune, y_tune)

        print("[INFO] Best XGB params:", grid.best_params_)

        best_xgb = XGBRegressor(
            **grid.best_params_,
            random_state=self.random_state,
            verbosity=0,
            tree_method="hist",
            n_jobs=1,
        )
        best_xgb.fit(X_train, y_train)

        return best_xgb

    def run(self):
        print("[INFO] Starting advanced modeling...")

        X_train, X_test, y_train, y_test = self.load_data()

        results = {}

        rf_model = self.train_random_forest(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_metrics = self.evaluate(y_test, rf_preds)
        results["RandomForest"] = rf_metrics
        joblib.dump(rf_model, os.path.join(self.model_dir, "random_forest.pkl"))
        print("[RESULT] Random Forest:", rf_metrics)

        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_metrics = self.evaluate(y_test, xgb_preds)
        results["XGBoost"] = xgb_metrics
        joblib.dump(xgb_model, os.path.join(self.model_dir, "xgboost.pkl"))
        print("[RESULT] XGBoost:", xgb_metrics)

        best_model_name = max(results, key=lambda k: results[k]["R2"])
        best_model = rf_model if best_model_name == "RandomForest" else xgb_model
        joblib.dump(best_model, os.path.join(self.model_dir, "best_model.pkl"))

        print(f"[INFO] Best model selected: {best_model_name}")

        with open(os.path.join(self.report_dir, "advanced_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        print("[INFO] Advanced models completed successfully")


if __name__ == "__main__":
    trainer = AdvancedModels()
    trainer.run()
