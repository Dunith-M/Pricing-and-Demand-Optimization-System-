import pandas as pd
import joblib
from pathlib import Path

from src.dynamic_pricing.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(
        self,
        preprocessor_path: str = "artifacts/preprocessing/preprocessor.pkl",
        model_path: str = "artifacts/models/final_model.pkl",
    ):
        """
        Initialize prediction pipeline
        """
        self.preprocessor_path = Path(preprocessor_path)
        self.model_path = Path(model_path)

        self.preprocessor = None
        self.model = None
        self.training_column_aliases = {
            "listing_id": "id",
            "host_id": "host id",
            "host_identity_verified": "host_identity_verified",
            "neighbourhood_group": "neighbourhood group",
            "neighbourhood": "neighbourhood",
            "lat": "lat",
            "long": "long",
            "country": "country",
            "instant_bookable": "instant_bookable",
            "cancellation_policy": "cancellation_policy",
            "room_type": "room type",
            "price": "price",
            "service_fee": "service fee",
            "minimum_nights": "minimum nights",
            "number_of_reviews": "number of reviews",
            "last_review": "last review",
            "reviews_per_month": "reviews per month",
            "review_rate_number": "review rate number",
            "calculated_host_listings_count": "calculated host listings count",
            "availability_365": "availability 365",
        }

    def load_artifacts(self):
        """
        Load preprocessor and trained model
        """
        try:
            logger.info("Loading preprocessor and model")

            self.preprocessor = joblib.load(self.preprocessor_path)
            self.model = joblib.load(self.model_path)

            logger.info("Artifacts loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise e

    def transform_input(self, df: pd.DataFrame):
        """
        Apply preprocessing pipeline
        """
        try:
            logger.info("Applying preprocessing pipeline")

            model_input = df.copy().rename(columns=self.training_column_aliases)
            X_transformed = self.preprocessor.transform(model_input)

            logger.info(f"Transformed input shape: {X_transformed.shape}")

            return X_transformed

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise e

    def predict_demand(self, X_transformed):
        """
        Predict demand using trained model
        """
        try:
            logger.info("Predicting demand")

            predictions = self.model.predict(X_transformed)

            logger.info("Demand prediction completed")

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e

    def build_output(self, df: pd.DataFrame, predictions):
        """
        Build final prediction output
        """
        try:
            logger.info("Building prediction output")

            result_df = pd.DataFrame()

            result_df["listing_id"] = df["listing_id"]
            result_df["current_price"] = df["price"]

            result_df["predicted_demand_before"] = predictions
            result_df["predicted_revenue_before"] = (
                result_df["current_price"] * result_df["predicted_demand_before"]
            )

            logger.info("Prediction output created")

            return result_df

        except Exception as e:
            logger.error(f"Output building failed: {e}")
            raise e

    def run(self, df: pd.DataFrame):
        """
        Full prediction pipeline
        """
        try:
            logger.info("Starting Prediction Pipeline")

            self.load_artifacts()

            X_transformed = self.transform_input(df)

            predictions = self.predict_demand(X_transformed)

            result_df = self.build_output(df, predictions)

            logger.info("Prediction Pipeline completed")

            return result_df

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            raise e
