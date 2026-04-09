import os
import numpy as np
import joblib
import json
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluation:
    def __init__(self):
        self.model_paths = {
            "LinearRegression": "artifacts/models/baseline/linear_regression.pkl",
            "Ridge": "artifacts/models/baseline/ridge.pkl",
            "RandomForest": "artifacts/models/advanced/random_forest.pkl",
            "XGBoost": "artifacts/models/advanced/xgboost.pkl",
        }

        self.report_dir = "artifacts/reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def load_data(self):
        print("[INFO] Loading processed test data...")

        X_test = np.load("artifacts/data/processed/X_test.npy")
        y_test = np.load("artifacts/data/processed/y_test.npy")

        return X_test, y_test

    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        return rmse, mae, r2

    def run(self):
        print("[INFO] Starting model evaluation...")

        X_test, y_test = self.load_data()

        results = []

        for model_name, path in self.model_paths.items():

            if not os.path.exists(path):
                print(f"[WARNING] Model not found: {model_name}")
                continue

            print(f"[INFO] Evaluating {model_name}...")

            model = joblib.load(path)

            rmse, mae, r2 = self.evaluate(model, X_test, y_test)

            results.append({
                "Model": model_name,
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4)
            })

        # -------------------------------
        # Create DataFrame
        # -------------------------------
        df = pd.DataFrame(results)

        # sort by R2 descending
        df = df.sort_values(by="R2", ascending=False)

        print("\n📊 MODEL COMPARISON:")
        print(df)

        # -------------------------------
        # Save CSV
        # -------------------------------
        csv_path = os.path.join(self.report_dir, "model_comparison.csv")
        df.to_csv(csv_path, index=False)

        # -------------------------------
        # Save JSON
        # -------------------------------
        json_path = os.path.join(self.report_dir, "model_performance.json")
        df.to_json(json_path, orient="records", indent=4)

        # -------------------------------
        # Best model
        # -------------------------------
        best_model = df.iloc[0]["Model"]
        print(f"\n🏆 BEST MODEL: {best_model}")

        print("[INFO] Evaluation completed successfully")


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.run()