from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
from src.utils.logger import logger


class Preprocessor:
    def __init__(
        self,
        params_imputer: ImputerParams,
        params_scaler: ScalerEncoderParams,
    ) -> None:
        self.imputer = params_imputer
        self.scaler = params_scaler

    def build_pipeline(self) -> Pipeline:
        return Pipeline(self._build_imputer().steps + self._build_scaler().steps)

    def _build_imputer(self) -> Pipeline:
        # since only sex is categorical and we do not have missing values there (we removed the ones missing)
        # we only need imputation for numerical values
        method = self.imputer.imputation_method
        logger.info(f"missing data imputation via: {method}")

        if method == "mean":
            return Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="mean", add_indicator=self.imputer.flag_missing
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
                            strategy="median", add_indicator=self.imputer.flag_missing
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
                            n_neighbors=self.imputer.knn_neighbors,
                            weights="distance",
                            add_indicator=self.imputer.flag_missing,
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
        scaler_type = self.scaler.type
        logger.info(f"scaling data using: {scaler_type}")
        if scaler_type == "standardization":
            return Pipeline([("scaler", StandardScaler())])
        elif scaler_type == "none":
            return Pipeline([("scaler", "passthrough")])
        else:
            raise ValueError(
                f"Tried to use unknown / wrong scaling method: {scaler_type}"
            )
