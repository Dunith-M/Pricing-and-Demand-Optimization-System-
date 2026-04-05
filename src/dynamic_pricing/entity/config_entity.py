from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DataIngestionConfig:
    raw_data_path: str


@dataclass
class DataTransformationConfig:
    test_size: float
    random_state: int
    target_column: str

    numerical_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    datetime_columns: List[str]

    drop_columns: List[str]


@dataclass
class ModelTrainerConfig:
    model_name: str
    model_params: Dict


@dataclass
class PathsConfig:
    raw_data: str
    interim_data: str
    processed_data: str
    model_path: str
    preprocessor_path: str
    validation_report: str