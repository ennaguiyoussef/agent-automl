"""Preprocessing pipeline construction.

All preprocessing is expressed as scikit-learn ``Pipeline`` / ``ColumnTransformer``
objects so that imputation, scaling and encoding are *fit on the training fold
only* — this is what prevents data leakage. The very same fitted transformer is
persisted together with the model, so inference applies identical preprocessing.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..schemas.common import InferredType
from ..schemas.dataset import DatasetInfo
from ..schemas.plan import PreprocessingConfig


def build_preprocessing_config(
    dataset_info: DatasetInfo,
    target: str,
    max_onehot_cardinality: int = 40,
) -> PreprocessingConfig:
    """Decide, deterministically, how each feature column should be handled."""
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[str] = []
    high_card: list[str] = []
    notes: list[str] = []

    for col in dataset_info.columns:
        if col.name == target:
            continue
        if col.is_constant:
            dropped.append(col.name)
            notes.append(f"Dropped constant column '{col.name}'.")
            continue
        if col.inferred_type == InferredType.datetime:
            dropped.append(col.name)
            notes.append(f"Dropped datetime column '{col.name}' (no datetime features in MVP).")
            continue
        if col.inferred_type == InferredType.text:
            dropped.append(col.name)
            notes.append(
                f"Dropped free-text / identifier column '{col.name}' "
                f"({col.unique} unique values)."
            )
            continue
        if col.inferred_type == InferredType.numeric:
            # Integer column with (almost) all-unique values is a row identifier,
            # not a feature — it carries no generalisable signal and can leak.
            is_id_like = (
                "int" in col.dtype.lower()
                and col.unique_pct > 99.0
                and dataset_info.n_rows >= 50
            )
            if is_id_like:
                dropped.append(col.name)
                notes.append(f"Dropped identifier-like numeric column '{col.name}' (all values unique).")
            else:
                numeric.append(col.name)
        else:  # categorical or boolean
            categorical.append(col.name)
            if col.is_high_cardinality:
                high_card.append(col.name)
                notes.append(
                    f"High-cardinality column '{col.name}' ({col.unique} values) — "
                    f"one-hot encoding capped at {max_onehot_cardinality} categories."
                )

    return PreprocessingConfig(
        numeric_features=numeric,
        categorical_features=categorical,
        dropped_features=dropped,
        high_cardinality_features=high_card,
        numeric_imputer="median",
        categorical_imputer="most_frequent",
        scale_numeric=True,
        onehot_encode=True,
        max_onehot_cardinality=max_onehot_cardinality,
        notes=notes,
    )


def build_column_transformer(config: PreprocessingConfig) -> ColumnTransformer:
    """Build the leakage-safe ColumnTransformer from a config."""
    transformers = []

    if config.numeric_features:
        numeric_steps = [("imputer", SimpleImputer(strategy=config.numeric_imputer))]
        if config.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), config.numeric_features))

    if config.categorical_features:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=config.categorical_imputer)),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                        max_categories=config.max_onehot_cardinality,
                        min_frequency=1,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", cat_pipeline, config.categorical_features))

    if not transformers:
        raise ValueError(
            "No usable feature columns remain after preprocessing "
            "(all columns were constant, identifiers, or datetimes)."
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(config: PreprocessingConfig, estimator) -> Pipeline:
    """Combine preprocessing + estimator into one persistable Pipeline."""
    return Pipeline(
        [
            ("preprocess", build_column_transformer(config)),
            ("model", estimator),
        ]
    )
