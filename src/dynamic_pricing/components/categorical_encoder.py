import os
import inspect
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder:
    def __init__(self, train_path: str, test_path: str, output_dir: str):
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir

        self.onehot_cols = ["room type"]
        self.freq_cols = ["neighbourhood group"]

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        print(f"[INFO] Train shape: {train_df.shape}")
        print(f"[INFO] Test shape: {test_df.shape}")

        return train_df, test_df

    def one_hot_encode(self, train_df, test_df):
        encoder_kwargs = {"handle_unknown": "ignore"}

        # scikit-learn >= 1.2 uses `sparse_output`; older versions use `sparse`.
        if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
            encoder_kwargs["sparse_output"] = False
        else:
            encoder_kwargs["sparse"] = False

        encoder = OneHotEncoder(**encoder_kwargs)

        encoder.fit(train_df[self.onehot_cols])

        train_encoded = encoder.transform(train_df[self.onehot_cols])
        test_encoded = encoder.transform(test_df[self.onehot_cols])

        encoded_cols = encoder.get_feature_names_out(self.onehot_cols)

        # FIX: keep index alignment
        train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_df.index)

        train_df = train_df.drop(columns=self.onehot_cols)
        test_df = test_df.drop(columns=self.onehot_cols)

        train_df = pd.concat([train_df, train_encoded_df], axis=1)
        test_df = pd.concat([test_df, test_encoded_df], axis=1)

        return train_df, test_df, encoder

    def frequency_encode(self, train_df, test_df):
        freq_maps = {}

        for col in self.freq_cols:
            freq = train_df[col].value_counts(normalize=True)
            freq_map = freq.to_dict()
            freq_maps[col] = freq_map

            train_df[col] = train_df[col].map(freq_map)

            # FIX: handle unseen categories safely
            test_df[col] = test_df[col].map(freq_map).fillna(0)

        return train_df, test_df, freq_maps

    def save_encoders(self, onehot_encoder, freq_maps):
        os.makedirs(self.output_dir, exist_ok=True)

        joblib.dump(onehot_encoder, os.path.join(self.output_dir, "onehot_encoder.pkl"))
        joblib.dump(freq_maps, os.path.join(self.output_dir, "frequency_maps.pkl"))

        print("[INFO] Encoders saved")

    def save_data(self, train_df, test_df):
        train_df.to_csv("artifacts/data/splits/train_encoded.csv", index=False)
        test_df.to_csv("artifacts/data/splits/test_encoded.csv", index=False)

        print("[INFO] Encoded datasets saved")

    def run(self):
        train_df, test_df = self.load_data()

        # Step 1: One-hot encoding
        train_df, test_df, onehot_encoder = self.one_hot_encode(train_df, test_df)

        # Step 2: Frequency encoding
        train_df, test_df, freq_maps = self.frequency_encode(train_df, test_df)

        # Step 3: Save encoders
        self.save_encoders(onehot_encoder, freq_maps)

        # Step 4: Save datasets
        self.save_data(train_df, test_df)

        print("[INFO] Categorical encoding completed")


if __name__ == "__main__":
    encoder = CategoricalEncoder(
        train_path="artifacts/data/splits/train.csv",
        test_path="artifacts/data/splits/test.csv",
        output_dir="artifacts/preprocessing/encoders",
    )

    encoder.run()
