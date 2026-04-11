import pandas as pd
from pathlib import Path

from src.dynamic_pricing.utils.common import read_yaml
from src.dynamic_pricing.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceInputPipeline:
    def __init__(self, schema_path: str = "configs/schema.yaml"):
        """
        Initialize inference input pipeline

        Args:
            schema_path (str): Path to schema.yaml
        """
        self.schema = read_yaml(Path(schema_path))
        self.required_columns = self.schema["columns"].keys()
        self.column_aliases = {
            "id": "listing_id",
            "service_fee": "service_fee",
            "room_type": "room_type",
            "neighbourhood_group": "neighbourhood_group",
            "availability_365": "availability_365",
            "number_of_reviews": "number_of_reviews",
            "reviews_per_month": "reviews_per_month",
        }

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        return str(column_name).strip().lower().replace(" ", "_")

    def load_input(self, input_data):
        """
        Load input from CSV path or DataFrame

        Args:
            input_data (str or pd.DataFrame)

        Returns:
            pd.DataFrame
        """
        try:
            logger.info("Loading inference input data")

            if isinstance(input_data, str):
                df = pd.read_csv(input_data)
            elif isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            else:
                raise ValueError("Input must be file path or pandas DataFrame")

            logger.info(f"Input data shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading input data: {e}")
            raise e

    def validate_columns(self, df: pd.DataFrame):
        """
        Check required columns exist

        Args:
            df (pd.DataFrame)

        Returns:
            pd.DataFrame
        """
        try:
            logger.info("Validating required columns")

            df = df.copy()
            df.columns = [self._normalize_column_name(col) for col in df.columns]
            df = df.rename(columns=self.column_aliases)

            missing_cols = [col for col in self.required_columns if col not in df.columns]

            if len(missing_cols) > 0:
                logger.error(f"Missing columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            logger.info("All required columns are present")
            return df

        except Exception as e:
            logger.error(f"Column validation failed: {e}")
            raise e

    def handle_missing_values(self, df: pd.DataFrame):
        """
        Handle missing values for inference

        Strategy:
        - Numerical → fill with 0
        - Categorical → fill with 'Unknown'

        Args:
            df (pd.DataFrame)

        Returns:
            pd.DataFrame
        """
        try:
            logger.info("Handling missing values")

            for col, col_type in self.schema["columns"].items():
                if col in df.columns:
                    if col_type == "numerical":
                        df[col] = df[col].fillna(0)
                    elif col_type == "categorical":
                        df[col] = df[col].fillna("Unknown")

            logger.info("Missing values handled successfully")
            return df

        except Exception as e:
            logger.error(f"Missing value handling failed: {e}")
            raise e

    def enforce_dtypes(self, df: pd.DataFrame):
        """
        Ensure column types match training schema

        Args:
            df (pd.DataFrame)

        Returns:
            pd.DataFrame
        """
        try:
            logger.info("Enforcing data types")

            for col, col_type in self.schema["columns"].items():
                if col in df.columns:
                    if col_type == "numerical":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif col_type == "categorical":
                        df[col] = df[col].astype(str)

            logger.info("Data types enforced")
            return df

        except Exception as e:
            logger.error(f"Data type enforcement failed: {e}")
            raise e

    def run(self, input_data):
        """
        Full inference input pipeline

        Args:
            input_data (str or DataFrame)

        Returns:
            pd.DataFrame
        """
        try:
            logger.info("Starting Inference Input Pipeline")

            df = self.load_input(input_data)
            df = self.validate_columns(df)
            df = self.enforce_dtypes(df)
            df = self.handle_missing_values(df)

            logger.info("Inference input pipeline completed successfully")

            return df

        except Exception as e:
            logger.error(f"Inference input pipeline failed: {e}")
            raise e
