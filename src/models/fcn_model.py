import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from typing import List, Dict, Any
from .base_model import BaseModel

class FCNRegressor(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for regression on 1D spectral data.
    Expects input shape (batch_size, num_features).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2
    ):
        """
        :param input_dim: number of input features
        :param hidden_dims: list of layer sizes [128, 64, ...]
        :param dropout_rate: dropout to apply between layers
        """
        super(FCNRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, input_dim)
        returns shape: (batch_size, 1)
        """
        return self.net(x)


class FCNModel(BaseModel):
    """
    High-level model class wrapping the FCNRegressor with training, predict, save, load.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        :param params: dictionary from config.yaml -> config["model"]["fcn"], e.g.:
            {
                "hidden_dims": [128, 64],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 20
            }
        """
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        dataset = data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float().view(-1, 1)
        )
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Full training loop for the FCN regressor.
        :param X_train: shape (N_train, n_features)
        :param y_train: shape (N_train,)
        :param X_val: shape (N_val, n_features)
        :param y_val: shape (N_val,)
        """
        input_dim = X_train.shape[1]
        hidden_dims = self.params["hidden_dims"]
        dropout_rate = self.params["dropout_rate"]

        self.model = FCNRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)

        lr = self.params["learning_rate"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        batch_size = self.params["batch_size"]
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader   = self._create_dataloader(X_val,   y_val,   batch_size, shuffle=False)

        epochs = self.params["epochs"]
        for epoch in range(epochs):
            self.model.train()
            train_loss_sum = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item() * batch_X.size(0)

            # Validation
            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    preds = self.model(batch_X)
                    loss = self.criterion(preds, batch_y)
                    val_loss_sum += loss.item() * batch_X.size(0)

            avg_train_loss = train_loss_sum / len(train_loader.dataset)
            avg_val_loss   = val_loss_sum / len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            preds = self.model(X_torch).cpu().numpy().flatten()
        return preds

    def save_model(self, path: str):
        checkpoint = {
            "input_dim": self.model.net[0].in_features,
            "hidden_dims": self.params["hidden_dims"],
            "dropout_rate": self.params["dropout_rate"],
            "state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        """
        Load the saved checkpoint (which includes input_dim, hidden_dims, dropout_rate).
        Rebuild the FCN architecture accordingly, then load the state_dict.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self._build_fcn(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"]
        )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        if self.model is None:
            raise RuntimeError("FCN architecture not initialized. Can't load state dict.")
        
    def _build_fcn(self, input_dim: int, hidden_dims: list, dropout_rate: int):
        """
        Helper method to build the FCNRegressor and assign to self.model.
        """
        self.model = FCNRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
