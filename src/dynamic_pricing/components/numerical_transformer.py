import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


class NumericalTransformer:
    def __init__(self, train_path: str, test_path: str, output_dir: str):
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.target_column = "demand_score"

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        print(f"[INFO] Train shape: {train_df.shape}")
        print(f"[INFO] Test shape: {test_df.shape}")

        return train_df, test_df

    def get_numerical_columns(self, df):
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # remove target
        if self.target_column in num_cols:
            num_cols.remove(self.target_column)

        return num_cols

    def log_transform(self, df):
        skewed_cols = ["price", "minimum nights", "number of reviews"]

        for col in skewed_cols:
            if col in df.columns:
                # Avoid invalid log1p values from negative entries.
                df[col] = np.log1p(df[col].clip(lower=0))

        return df

    def clean_numeric_data(self, train_df, test_df, num_cols):
        train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
        test_df[num_cols] = test_df[num_cols].replace([np.inf, -np.inf], np.nan)

        fill_values = train_df[num_cols].median()

        # Guard against columns that are entirely missing in train.
        fill_values = fill_values.fillna(0)

        train_df[num_cols] = train_df[num_cols].fillna(fill_values)
        test_df[num_cols] = test_df[num_cols].fillna(fill_values)

        return train_df, test_df, fill_values

    def scale_data(self, train_df, test_df, num_cols):
        scaler = StandardScaler()

        # fit only on train
        scaler.fit(train_df[num_cols])

        train_df[num_cols] = scaler.transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])

        return train_df, test_df, scaler

    def save_scaler(self, scaler, fill_values):
        os.makedirs(self.output_dir, exist_ok=True)

        scaler_path = os.path.join(self.output_dir, "scaler.pkl")
        imputer_path = os.path.join(self.output_dir, "numeric_fill_values.pkl")
        joblib.dump(scaler, scaler_path)
        joblib.dump(fill_values.to_dict(), imputer_path)

        print(f"[INFO] Scaler saved at: {scaler_path}")
        print(f"[INFO] Numeric fill values saved at: {imputer_path}")

    def save_data(self, train_df, test_df):
        train_df.to_csv("artifacts/data/splits/train_processed.csv", index=False)
        test_df.to_csv("artifacts/data/splits/test_processed.csv", index=False)

        print("[INFO] Processed datasets saved")

    def run(self):
        train_df, test_df = self.load_data()

        # Step 1: Log transform
        train_df = self.log_transform(train_df)
        test_df = self.log_transform(test_df)

        # Step 2: Detect numerical columns dynamically
        num_cols = self.get_numerical_columns(train_df)

        print(f"[INFO] Numerical columns: {num_cols}")

        # Step 3: Clean invalid numeric values before scaling
        train_df, test_df, fill_values = self.clean_numeric_data(train_df, test_df, num_cols)

        # Step 4: Scaling
        train_df, test_df, scaler = self.scale_data(train_df, test_df, num_cols)

        # Step 5: Save scaler and fill values
        self.save_scaler(scaler, fill_values)

        # Step 6: Save processed data
        self.save_data(train_df, test_df)

        print("[INFO] Numerical transformation completed")


if __name__ == "__main__":
    transformer = NumericalTransformer(
        train_path="artifacts/data/splits/train_encoded.csv",
        test_path="artifacts/data/splits/test_encoded.csv",
        output_dir="artifacts/preprocessing/scalers",
    )

    transformer.run()
