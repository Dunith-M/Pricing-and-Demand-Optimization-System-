import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitting:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_column = "demand_score"

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path)
        print(f"[INFO] Data loaded. Shape: {df.shape}")
        return df

    def remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        leakage_cols = ["reviews per month", "availability 365"]

        existing_cols = [col for col in leakage_cols if col in df.columns]

        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"[INFO] Removed leakage columns: {existing_cols}")

        return df

    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        print(f"[INFO] Train shape: {X_train.shape}")
        print(f"[INFO] Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test):
        os.makedirs(self.output_dir, exist_ok=True)

        train_df = X_train.copy()
        train_df[self.target_column] = y_train

        test_df = X_test.copy()
        test_df[self.target_column] = y_test

        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"[INFO] Train saved at: {train_path}")
        print(f"[INFO] Test saved at: {test_path}")

    def run(self):
        df = self.load_data()

        df = self.remove_leakage_features(df)

        X_train, X_test, y_train, y_test = self.split_data(df)

        self.save_data(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    splitter = DataSplting = DataSplitting(
        input_path="data/processed/feature_engineered_data.csv",
        output_dir="artifacts/data/splits",
    )

    splitter.run()