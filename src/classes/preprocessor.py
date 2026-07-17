from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig


class Preprocessor:
    """Build sklearn preprocessing pipelines from imputer and scaler configs."""

    def __init__(
        self,
        imputer_config: ImputerConfig,
        scaler_config: ScalerEncoderConfig,
    ) -> None:
        self.imputer_config = imputer_config
        self.scaler_config = scaler_config

    def build_pipeline(self) -> Pipeline:
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(self._build_imputer().steps + self._build_scaler().steps),
                    make_column_selector(dtype_exclude=object),
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    make_column_selector(dtype_include=object),
                ),
            ],
            verbose_feature_names_out=False,
        )

    def _build_imputer(self) -> Pipeline:
        # since only sex is categorical and we do not have missing values there (we removed the ones missing)
        # we only need imputation for numerical values
        method = self.imputer_config.imputation_method

        if method == "mean":
            return Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="mean",
                            add_indicator=self.imputer_config.flag_missing,
                        ),
                    ),
                ]
            )
        elif method == "median":
            return Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=self.imputer_config.flag_missing,
                        ),
                    ),
                ]
            )
        elif method == "knn":
            return Pipeline(
                [
                    (
                        "imputer",
                        KNNImputer(
                            n_neighbors=self.imputer_config.knn_neighbors,
                            weights="distance",
                            add_indicator=self.imputer_config.flag_missing,
                        ),
                    ),
                ]
            )
        elif method == "none":
            return Pipeline([("imputer", "passthrough")])
        else:
            raise ValueError(
                f"Tried to use unknown / wrong imputation method: {method}"
            )

    def _build_scaler(self) -> Pipeline:
        scaler_type = self.scaler_config.type
        if scaler_type == "standardization":
            return Pipeline([("scaler", StandardScaler())])
        elif scaler_type == "none":
            return Pipeline([("scaler", "passthrough")])
        else:
            raise ValueError(
                f"Tried to use unknown / wrong scaling method: {scaler_type}"
            )
