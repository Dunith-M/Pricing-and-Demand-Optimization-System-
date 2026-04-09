import os
import time
from pathlib import Path
from urllib.parse import unquote

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


class ModelTrainer:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[3]
        self.model_dir = self.project_root / "artifacts" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._configure_mlflow()
        mlflow.set_experiment("airbnb_price_modeling")

    def _configure_mlflow(self):
        """
        Configure MLflow to use a repo-local SQLite database and artifact directory.

        This avoids Windows path issues when an inherited MLFLOW_TRACKING_URI uses
        URL-encoded paths such as %20 for spaces.
        """
        ml_root = self.project_root / "artifacts" / "ml"
        artifact_root = ml_root / "artifacts"
        tracking_db = ml_root / "mlflow.db"

        artifact_root.mkdir(parents=True, exist_ok=True)

        env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if env_tracking_uri and "%20" in env_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = unquote(env_tracking_uri)

        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tracking_db.as_posix()}"
        os.environ.setdefault("MLFLOW_ARTIFACT_URI", artifact_root.resolve().as_uri())
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    def load_data(self):
        print("[INFO] Loading processed data...")

        processed_dir = self.project_root / "artifacts" / "data" / "processed"
        X_train = np.load(processed_dir / "X_train.npy")
        X_test = np.load(processed_dir / "X_test.npy")
        y_train = np.load(processed_dir / "y_train.npy")
        y_test = np.load(processed_dir / "y_test.npy")

        return X_train, X_test, y_train, y_test

    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    def train_random_forest(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        return model

    def run(self):
        print("[INFO] Starting MLflow training...")

        X_train, X_test, y_train, y_test = self.load_data()

        models = {
            "RandomForest": self.train_random_forest,
            "XGBoost": self.train_xgboost,
        }

        best_model = None
        best_score = float("-inf")
        best_model_name = ""

        for name, train_func in models.items():
            with mlflow.start_run(run_name=name):
                print(f"[INFO] Training {name}...")

                start_time = time.time()
                model = train_func(X_train, y_train)
                preds = model.predict(X_test)

                rmse, mae, r2 = self.evaluate(y_test, preds)
                training_time = time.time() - start_time

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("training_time", training_time)

                mlflow.log_param("model_name", name)

                if name == "RandomForest":
                    mlflow.log_param("n_estimators", 200)
                    mlflow.log_param("max_depth", 20)

                if name == "XGBoost":
                    mlflow.log_param("n_estimators", 200)
                    mlflow.log_param("max_depth", 6)
                    mlflow.log_param("learning_rate", 0.1)

                mlflow.sklearn.log_model(model, "model")

                print(
                    f"[RESULT] {name} -> RMSE: {rmse:.4f}, "
                    f"MAE: {mae:.4f}, R2: {r2:.4f}"
                )

                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name

        final_model_path = self.model_dir / "final_model.pkl"
        joblib.dump(best_model, final_model_path)

        print(f"\n[INFO] Final model: {best_model_name}")
        print(f"[INFO] Saved at: {final_model_path}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
