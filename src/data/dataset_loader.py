import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
import os
from src.utils.logger import get_logger
from src.utils.load_config import config

logger = get_logger(__name__)

class DatasetLoader:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = config["data"]["dataset_path"]
        self.dataset_path = dataset_path

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found at {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        df = df.set_index("hsi_id")
        logger.info(f"Loaded dataset with shape: {df.shape}")

        logger.debug(f"Dataset head:\n{df.head()}")
        logger.debug(f"Missing values:\n{df.isna().sum()}")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    def normalize_data(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def get_data(self):
        X, y = self.load_data()
        # X = self.normalize_data(X)
        return X, y