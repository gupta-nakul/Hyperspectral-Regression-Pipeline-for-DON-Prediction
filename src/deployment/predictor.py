import joblib
import torch
import numpy as np

from src.models.fcn_model import FCNModel
from src.models.cnn_model import CNNModel
from src.models.transformer_model import TransformerModel
from src.utils.load_config import config

class Predictor:
    def __init__(self, model_type="xgboost", model_path=None):
        self.model_type = model_type.lower()
        if self.model_type not in ["xgboost", "transformer", "cnn", "fcn"]:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model_path = model_path

        if self.model_type == "xgboost":
            self.model = joblib.load(self.model_path)

        elif self.model_type == "transformer":
            self.model = TransformerModel(config["model"]["transformer"])
            self.model.load_model(self.model_path)

        elif self.model_type == "cnn":
            self.model = CNNModel(config["model"]["cnn"])
            self.model.load_model(self.model_path)

        elif self.model_type == "fcn":
            self.model = FCNModel(config["model"]["fcn"])
            self.model.load_model(self.model_path)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, X):
        """
        Generate predictions for input data X.
        :param X: NumPy array or list of shape (N, num_features) for FCN/Transformer
                  or (N, num_bands) for CNN (with a reshape to (N, 1, num_bands)),
                  or shape (N, num_features) for XGBoost.
        :return: NumPy array of predictions, shape (N,).
        """
        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if self.model_type == "xgboost":
            return self.model.predict(X)

        else:
            if self.model_type == "cnn":
                X_torch = torch.tensor(X.reshape(X.shape[0], 1, X.shape[1]), dtype=torch.float32)

            else:
                X_torch = torch.tensor(X, dtype=torch.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_torch = X_torch.to(device)
            self.model.model.eval()

            with torch.no_grad():
                preds = self.model.model(X_torch).cpu().numpy().flatten()

            return preds
