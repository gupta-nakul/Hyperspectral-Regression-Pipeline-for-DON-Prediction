import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.load_config import config

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, impute_strategy="mean", scale_data=True, n_components=50, feature_selection=False):
        """
        :param impute_strategy: Strategy to fill missing values (e.g. 'mean', 'median', 'most_frequent')
        :param scale_data: Boolean to apply standard scaling
        """
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=n_components) if feature_selection else None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies missing value imputation and optional scaling.
        :param df: Original dataframe containing spectral columns + target
        :param target: Name of the target column
        :return: Processed dataframe
        """
        logger.info("Starting preprocessing...")

        # Impute
        logger.info(f"Imputing missing values using strategy: {self.imputer.strategy}")
        X_imputed = self.imputer.fit_transform(X)

        # Scale
        if self.scaler:
            logger.info("Applying standard scaling.")
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = X_imputed

        if self.pca:
            logger.info("Applying PCA for feature selection")
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = X_scaled

        logger.info("Preprocessing complete.")
        return X_reduced