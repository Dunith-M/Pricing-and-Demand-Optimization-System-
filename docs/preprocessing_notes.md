# Pricing and Demand Optimization System: Preprocessing Notes

## Project Focus
This part of the project prepares Airbnb listing data for a pricing and demand prediction workflow.

The work here mainly covers:
- feature engineering
- train/test splitting
- categorical encoding
- numerical transformation
- saving preprocessing artifacts for later model training

## Main Folder Structure
### Source code
- `src/dynamic_pricing/components/`
  - `data_splitting.py`
  - `categorical_encoder.py`
  - `numerical_transformer.py`
- `src/dynamic_pricing/pipeline/`
  - `preprocessing_pipeline.py`

### Config files
- `configs/paths.yaml`
- `configs/model.yaml`
- `configs/config.yaml`

### Data and artifacts
- `data/raw/airbnb.csv`
- `data/processed/feature_engineered_data.csv`
- `artifacts/data/splits/`
- `artifacts/data/processed/`
- `artifacts/preprocessing/`

## What Was Done Here
### 1. Feature engineering
The engineered dataset is stored in `data/processed/feature_engineered_data.csv`.

This dataset keeps the main Airbnb listing information and adds new predictive features for demand modeling.

### 2. Data splitting
In `data_splitting.py`, the data is:
- loaded from the feature-engineered dataset
- split into train and test sets
- saved to `artifacts/data/splits/train.csv` and `artifacts/data/splits/test.csv`

Small explanation:
- the target column is `demand_score`
- leakage columns are removed before splitting:
  - `reviews per month`
  - `availability 365`

These were removed because they are directly used to create the target and could make the model unfairly strong during training.

### 3. Categorical encoding
In `categorical_encoder.py`, categorical features are transformed into model-friendly numeric values.

What is done:
- one-hot encoding for `room type`
- frequency encoding for `neighbourhood group`

Artifacts saved:
- `artifacts/preprocessing/encoders/onehot_encoder.pkl`
- `artifacts/preprocessing/encoders/frequency_maps.pkl`

Processed split files saved:
- `artifacts/data/splits/train_encoded.csv`
- `artifacts/data/splits/test_encoded.csv`

Small explanation:
- one-hot encoding creates separate binary columns for each room type
- frequency encoding replaces each neighbourhood group with how common it is in the training data
- unseen categories in test data are safely filled with `0`

### 4. Numerical transformation
In `numerical_transformer.py`, numerical features are prepared for model training.

What is done:
- log transform on skewed features
- cleaning invalid numeric values
- filling missing numeric values using train-set medians
- standard scaling

Log-transformed columns:
- `price`
- `minimum nights`
- `number of reviews`

Artifacts saved:
- `artifacts/preprocessing/scalers/scaler.pkl`
- `artifacts/preprocessing/scalers/numeric_fill_values.pkl`

Processed files saved:
- `artifacts/data/splits/train_processed.csv`
- `artifacts/data/splits/test_processed.csv`

Small explanation:
- negative values are clipped before `log1p()` so the transformation stays valid
- missing and infinite values are replaced before scaling
- scaling helps keep numeric features on similar ranges

### 5. Combined preprocessing pipeline
In `preprocessing_pipeline.py`, there is also a full sklearn pipeline version of preprocessing.

It contains:
- a custom `FeatureEngineering` transformer
- a numerical pipeline with `StandardScaler`
- a categorical pipeline with `OneHotEncoder`
- a `ColumnTransformer` and a final sklearn `Pipeline`

Outputs saved by this pipeline:
- `artifacts/preprocessing/preprocessor.pkl`
- `artifacts/data/processed/X_train.npy`
- `artifacts/data/processed/X_test.npy`
- `artifacts/data/processed/y_train.npy`
- `artifacts/data/processed/y_test.npy`

## Features Used in This Project
### Original base features
The main source columns include:
- `id`
- `host id`
- `host_identity_verified`
- `neighbourhood group`
- `neighbourhood`
- `lat`
- `long`
- `country`
- `instant_bookable`
- `cancellation_policy`
- `room type`
- `price`
- `service fee`
- `minimum nights`
- `number of reviews`
- `last review`
- `reviews per month`
- `review rate number`
- `calculated host listings count`
- `availability 365`

### Target feature
- `demand_score`

Small explanation:
- this is the main prediction target for the model
- in your notebook work, it was created from listing activity and availability behavior

### Engineered features
The engineered dataset includes these extra features:
- `review_intensity`
- `availability_bucket`
- `price_per_minimum_night`
- `price_log`
- `neighbourhood_group_demand_avg`
- `room_type_neighbourhood`
- `price_review_interaction`
- `availability_min_nights`

Small explanation of engineered features:
- `review_intensity`: captures how active reviews are relative to listing behavior
- `availability_bucket`: groups availability into useful categories
- `price_per_minimum_night`: price adjusted by minimum stay requirement
- `price_log`: log-scaled price to reduce skewness
- `neighbourhood_group_demand_avg`: average demand signal at neighbourhood group level
- `room_type_neighbourhood`: interaction between room type and location group
- `price_review_interaction`: combines price and review behavior
- `availability_min_nights`: interaction between availability and minimum-night constraint

## Libraries and Tools Used
From `requirements.txt`, the main libraries used here are:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`

How they are used:
- `pandas` for loading and saving tabular data
- `numpy` for numeric transformations
- `scikit-learn` for train/test split, encoding, scaling, and pipeline creation
- `joblib` for saving fitted preprocessing objects
- `matplotlib` and `seaborn` for analysis and visualization in notebooks

## Files Produced by the Preprocessing Work
### Split datasets
- `artifacts/data/splits/train.csv`
- `artifacts/data/splits/test.csv`

### Encoded datasets
- `artifacts/data/splits/train_encoded.csv`
- `artifacts/data/splits/test_encoded.csv`

### Final processed datasets
- `artifacts/data/splits/train_processed.csv`
- `artifacts/data/splits/test_processed.csv`

### Saved preprocessing artifacts
- `artifacts/preprocessing/encoders/onehot_encoder.pkl`
- `artifacts/preprocessing/encoders/frequency_maps.pkl`
- `artifacts/preprocessing/scalers/scaler.pkl`
- `artifacts/preprocessing/scalers/numeric_fill_values.pkl`
- `artifacts/preprocessing/preprocessor.pkl`

### Numpy arrays for modeling
- `artifacts/data/processed/X_train.npy`
- `artifacts/data/processed/X_test.npy`
- `artifacts/data/processed/y_train.npy`
- `artifacts/data/processed/y_test.npy`

## Short Workflow Summary
The preprocessing workflow in this project is:

1. Start from the engineered dataset.
2. Remove leakage columns.
3. Split data into training and testing sets.
4. Encode categorical features.
5. clean, transform, and scale numerical features.
6. Save processed datasets and preprocessing objects.
7. Use the outputs for model training.

## Good Note for Explaining This Work
This project section builds a preprocessing system for Airbnb pricing and demand modeling. First, the cleaned and feature-engineered dataset is split into train and test sets. Then categorical variables such as room type and neighbourhood group are encoded using one-hot and frequency encoding. After that, numerical features are log-transformed, cleaned, imputed, and scaled. Finally, all important preprocessing artifacts are saved so the same transformations can be reused consistently during model training and prediction.
