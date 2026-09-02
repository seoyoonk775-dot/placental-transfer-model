#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rebuild the final CI and F/M placental-transfer regression models.

Input
-----
data/Supplementary Dataset S1.xlsx

Required sheets
---------------
1. CI_final_training_data
   - 93 retained drugs
   - columns: Drug, CI, and 30 final CI features

2. FM_final_training_data
   - 117 retained drugs
   - columns: Drug, F/M, and 91 final F/M features

Models
------
CI:
- Gradient Boosting Regression Trees (GBRT)
- fixed final hyperparameters reported in the study
- 30 features

F/M:
- Random Forest Regressor
- fixed final hyperparameters reported in the study
- 91 features

The saved objects are sklearn Pipelines containing:
    SimpleImputer(strategy="median") -> final regression model

This allows future users to supply the same feature columns with missing
cell values; missing values are automatically replaced using the medians
learned from the final training dataset.

Usage
-----
From the repository root:
    python code/01_build_final_models.py

Outputs
-------
results/final_models/
    CI_final_model.joblib
    FM_final_model.joblib
    final_model_metadata.json

Notes
-----
- This script does NOT repeat feature selection, hyperparameter optimization,
  or residual-outlier identification.
- It rebuilds the final models from the final retained/imputed training
  matrices released as Supplementary Dataset S1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# =============================================================================
# 1. Repository paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR.parent / "data").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
elif (Path.cwd() / "data").exists():
    PROJECT_ROOT = Path.cwd().resolve()
else:
    raise FileNotFoundError(
        "Repository root could not be identified. "
        "Expected a 'data' directory next to the 'code' directory "
        "or in the current working directory."
    )

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "Supplementary Dataset S1.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CI_MODEL_PATH = (
    OUTPUT_DIR
    / "CI_final_model.joblib"
)

FM_MODEL_PATH = (
    OUTPUT_DIR
    / "FM_final_model.joblib"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "final_model_metadata.json"
)


# =============================================================================
# 2. Final model settings
# =============================================================================

RANDOM_STATE = 42

CI_SHEET = "CI_final_training_data"
FM_SHEET = "FM_final_training_data"

CI_TARGET = "CI"
FM_TARGET = "F/M"

EXPECTED_CI_N = 93
EXPECTED_FM_N = 117

EXPECTED_CI_FEATURE_N = 30
EXPECTED_FM_FEATURE_N = 91


CI_HYPERPARAMETERS: dict[str, Any] = {
    "loss": "huber",
    "alpha": 0.8715280397068295,
    "learning_rate": 0.07875090571534238,
    "n_estimators": 1784,
    "max_depth": 3,
    "max_features": None,
    "min_samples_leaf": 5,
    "min_samples_split": 18,
    "subsample": 0.6254047318360005,
    "random_state": RANDOM_STATE,
}


FM_HYPERPARAMETERS: dict[str, Any] = {
    "bootstrap": True,
    "criterion": "friedman_mse",
    "n_estimators": 421,
    "max_depth": 10,
    "max_features": 0.3,
    "max_leaf_nodes": 50,
    "min_samples_leaf": 5,
    "min_samples_split": 8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =============================================================================
# 3. Data loading and validation
# =============================================================================

def load_training_sheet(
    excel_path: Path,
    sheet_name: str,
    target_column: str,
    expected_n: int,
    expected_feature_n: int,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """
    Load and validate one final training sheet.

    The first two columns must be:
        Drug | target

    Every remaining column is treated as a final model feature.
    """

    data = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        engine="openpyxl",
    )

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    if len(data) != expected_n:
        raise ValueError(
            f"[{sheet_name}] Expected {expected_n} drugs, "
            f"but found {len(data)}."
        )

    if len(data.columns) < 3:
        raise ValueError(
            f"[{sheet_name}] The sheet does not contain "
            "Drug, target, and feature columns."
        )

    if data.columns[0] != "Drug":
        raise ValueError(
            f"[{sheet_name}] First column must be 'Drug', "
            f"but found '{data.columns[0]}'."
        )

    if data.columns[1] != target_column:
        raise ValueError(
            f"[{sheet_name}] Second column must be '{target_column}', "
            f"but found '{data.columns[1]}'."
        )

    if data["Drug"].isna().any():
        raise ValueError(
            f"[{sheet_name}] Missing Drug values were found."
        )

    if data["Drug"].astype(str).duplicated().any():
        duplicated = (
            data.loc[
                data["Drug"].astype(str).duplicated(
                    keep=False
                ),
                "Drug",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"[{sheet_name}] Duplicate Drug names were found: "
            f"{duplicated[:20]}"
        )

    feature_columns = list(
        data.columns[2:]
    )

    if len(feature_columns) != expected_feature_n:
        raise ValueError(
            f"[{sheet_name}] Expected {expected_feature_n} features, "
            f"but found {len(feature_columns)}."
        )

    y = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    if y.isna().any():
        raise ValueError(
            f"[{sheet_name}] Missing or non-numeric target values "
            "were found."
        )

    X = data[
        feature_columns
    ].copy()

    # Preserve feature names and convert values to numeric.
    # Non-numeric cells become NaN and are handled by the saved imputer.
    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return (
        X,
        y.astype(float),
        feature_columns,
        data,
    )


# =============================================================================
# 4. Model construction
# =============================================================================

def build_ci_pipeline() -> Pipeline:
    """
    Final CI pipeline:
        median imputation -> tuned GBRT
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                GradientBoostingRegressor(
                    **CI_HYPERPARAMETERS
                ),
            ),
        ]
    )


def build_fm_pipeline() -> Pipeline:
    """
    Final F/M pipeline:
        median imputation -> tuned Random Forest
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    **FM_HYPERPARAMETERS
                ),
            ),
        ]
    )


# =============================================================================
# 5. Save model package
# =============================================================================

def save_model_package(
    pipeline: Pipeline,
    model_path: Path,
    endpoint: str,
    target_column: str,
    feature_columns: list[str],
    training_n: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Save the fitted pipeline together with the metadata needed for reuse.
    """

    imputer = pipeline.named_steps[
        "imputer"
    ]

    medians = {
        feature: float(value)
        for feature, value in zip(
            feature_columns,
            imputer.statistics_,
        )
    }

    package = {
        "endpoint": endpoint,
        "target_column": target_column,
        "model": pipeline,
        "selected_features": feature_columns,
        "feature_n": len(feature_columns),
        "training_n": training_n,
        "imputation": "median",
        "training_feature_medians": medians,
        "hyperparameters": hyperparameters,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        package,
        model_path,
    )

    return package


# =============================================================================
# 6. Optional helper for prediction on new drugs
# =============================================================================

def predict_new_drugs(
    model_path: str | Path,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """
    Predict CI or F/M for new drugs using a saved model package.

    Requirements
    ------------
    - new_data must contain all model feature columns.
    - Individual feature values may be missing (NaN).
    - Missing values are imputed automatically using the training-set medians.

    Example
    -------
    package = joblib.load("results/final_models/CI_final_model.joblib")
    features = package["selected_features"]

    new_drugs = pd.read_excel("new_CI_features.xlsx")
    predictions = predict_new_drugs(
        "results/final_models/CI_final_model.joblib",
        new_drugs,
    )
    """

    package = joblib.load(
        model_path
    )

    pipeline = package[
        "model"
    ]

    required_features = package[
        "selected_features"
    ]

    missing_columns = [
        feature
        for feature in required_features
        if feature not in new_data.columns
    ]

    if missing_columns:
        raise KeyError(
            "The new dataset is missing required feature columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    X_new = new_data[
        required_features
    ].copy()

    for column in required_features:
        X_new[column] = pd.to_numeric(
            X_new[column],
            errors="coerce",
        )

    X_new = X_new.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return np.asarray(
        pipeline.predict(
            X_new
        ),
        dtype=float,
    )


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Supplementary Dataset S1.xlsx was not found:\n"
            f"{DATA_PATH}"
        )

    print("=" * 80)
    print("Rebuilding final placental-transfer regression models")
    print("=" * 80)

    # -----------------------------------------------------------------
    # CI
    # -----------------------------------------------------------------

    (
        X_ci,
        y_ci,
        ci_features,
        ci_data,
    ) = load_training_sheet(
        excel_path=DATA_PATH,
        sheet_name=CI_SHEET,
        target_column=CI_TARGET,
        expected_n=EXPECTED_CI_N,
        expected_feature_n=EXPECTED_CI_FEATURE_N,
    )

    ci_pipeline = build_ci_pipeline()

    ci_pipeline.fit(
        X_ci,
        y_ci,
    )

    ci_package = save_model_package(
        pipeline=ci_pipeline,
        model_path=CI_MODEL_PATH,
        endpoint="CI",
        target_column=CI_TARGET,
        feature_columns=ci_features,
        training_n=len(ci_data),
        hyperparameters=CI_HYPERPARAMETERS,
    )

    # -----------------------------------------------------------------
    # F/M
    # -----------------------------------------------------------------

    (
        X_fm,
        y_fm,
        fm_features,
        fm_data,
    ) = load_training_sheet(
        excel_path=DATA_PATH,
        sheet_name=FM_SHEET,
        target_column=FM_TARGET,
        expected_n=EXPECTED_FM_N,
        expected_feature_n=EXPECTED_FM_FEATURE_N,
    )

    fm_pipeline = build_fm_pipeline()

    fm_pipeline.fit(
        X_fm,
        y_fm,
    )

    fm_package = save_model_package(
        pipeline=fm_pipeline,
        model_path=FM_MODEL_PATH,
        endpoint="F/M",
        target_column=FM_TARGET,
        feature_columns=fm_features,
        training_n=len(fm_data),
        hyperparameters=FM_HYPERPARAMETERS,
    )

    # -----------------------------------------------------------------
    # Save non-binary metadata
    # -----------------------------------------------------------------

    metadata = {
        "input_dataset": str(
            DATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "CI": {
            "sheet": CI_SHEET,
            "training_n": EXPECTED_CI_N,
            "feature_n": EXPECTED_CI_FEATURE_N,
            "model_type": "GradientBoostingRegressor",
            "model_file": str(
                CI_MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "selected_features": ci_features,
            "hyperparameters": CI_HYPERPARAMETERS,
        },
        "FM": {
            "sheet": FM_SHEET,
            "training_n": EXPECTED_FM_N,
            "feature_n": EXPECTED_FM_FEATURE_N,
            "model_type": "RandomForestRegressor",
            "model_file": str(
                FM_MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "selected_features": fm_features,
            "hyperparameters": FM_HYPERPARAMETERS,
        },
        "pipeline": (
            "SimpleImputer(strategy='median') -> final regression model"
        ),
        "purpose": (
            "Rebuild the final fitted CI and F/M models from "
            "Supplementary Dataset S1."
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print("\n[CI final model]")
    print(
        f"Training N: {ci_package['training_n']}"
    )
    print(
        f"Feature N: {ci_package['feature_n']}"
    )
    print(
        "Model: GradientBoostingRegressor"
    )
    print(
        f"Saved: {CI_MODEL_PATH}"
    )

    print("\n[F/M final model]")
    print(
        f"Training N: {fm_package['training_n']}"
    )
    print(
        f"Feature N: {fm_package['feature_n']}"
    )
    print(
        "Model: RandomForestRegressor"
    )
    print(
        f"Saved: {FM_MODEL_PATH}"
    )

    print(
        f"\nMetadata saved: {METADATA_PATH}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
