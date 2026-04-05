from pathlib import Path

from src.dynamic_pricing.utils.common import read_yaml
from src.dynamic_pricing.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    PathsConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_config_path: str = "configs/model.yaml",
        paths_config_path: str = "configs/paths.yaml",
    ):
        """
        Load all configuration files
        """
        self.config = read_yaml(Path(config_path))
        self.model_config = read_yaml(Path(model_config_path))
        self.paths_config = read_yaml(Path(paths_config_path))

    # 🔹 DATA INGESTION CONFIG
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            raw_data_path=self.config["data_ingestion"]["raw_data_path"]
        )

    # 🔹 DATA TRANSFORMATION CONFIG (YOUR NEW METHOD)
    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            test_size=self.config["data_transformation"]["test_size"],
            random_state=self.config["data_transformation"]["random_state"],
            target_column=self.config["data_transformation"]["target_column"],

            numerical_columns=self.config["features"]["numerical_columns"],
            categorical_columns=self.config["features"]["categorical_columns"],
            boolean_columns=self.config["features"]["boolean_columns"],
            datetime_columns=self.config["features"]["datetime_columns"],

            drop_columns=self.config["drop_columns"],
        )

    # 🔹 MODEL TRAINER CONFIG
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_name = self.model_config["model_trainer"]["model_name"]
        model_params = self.model_config["model_trainer"][model_name]

        return ModelTrainerConfig(
            model_name=model_name,
            model_params=model_params,
        )

    # 🔹 PATH CONFIG
    def get_paths_config(self) -> PathsConfig:
        return PathsConfig(
            raw_data=self.paths_config["data"]["raw"],
            interim_data=self.paths_config["data"]["interim"],
            processed_data=self.paths_config["data"]["processed"],
            model_path=self.paths_config["artifacts"]["model_path"],
            preprocessor_path=self.paths_config["artifacts"]["preprocessor_path"],
            validation_report=self.paths_config["reports"]["validation_report"],
        )