import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from typing import Dict, Any

from .base_model import BaseModel

class CNNRegressor(nn.Module):
    """
    A 1D CNN for spectral regression.
    Expects input shape (batch_size, 1, num_bands).
    """
    def __init__(
        self,
        in_channels: int,
        num_bands: int,
        filters: list = [64, 128],
        kernel_size: int = 3,
        dropout_rate: float = 0.2
    ):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filters[0], kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(filters[1] * num_bands, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. 
        x shape: (batch_size, 1, num_bands)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class CNNModel(BaseModel):
    """
    High-level model class that implements the abstract methods 
    from BaseModel: train, predict, save_model, load_model.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        :param params: A dictionary of hyperparameters, for example:
            {
                "filters": [64, 128],
                "kernel_size": 3,
                "dropout_rate": 0.2,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001
            }
        """
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        """
        Utility function to create a PyTorch DataLoader from numpy arrays.
        """
        dataset = data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float().view(-1, 1)
        )
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train the CNN using a standard PyTorch loop.
        :param X_train: shape (N_train, num_bands)
        :param y_train: shape (N_train,)
        :param X_val: shape (N_val, num_bands)
        :param y_val: shape (N_val,)
        """
        num_bands = X_train.shape[1]
        in_channels = 1
        
        self.model = CNNRegressor(
            in_channels=in_channels,
            num_bands=num_bands,
            filters=self.params.get("filters", [64, 128]),
            kernel_size=self.params.get("kernel_size", 3),
            dropout_rate=self.params.get("dropout_rate", 0.2)
        ).to(self.device)

        lr = self.params.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        batch_size = self.params.get("batch_size", 32)
        train_loader = self._create_dataloader(X_train.reshape(-1, 1, num_bands), y_train, batch_size, shuffle=True)
        val_loader = self._create_dataloader(X_val.reshape(-1, 1, num_bands), y_val, batch_size, shuffle=False)

        # Training loop
        epochs = self.params.get("epochs", 50)
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch_X.size(0)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    preds = self.model(batch_X)
                    loss = self.criterion(preds, batch_y)
                    val_loss += loss.item() * batch_X.size(0)

            train_loss_avg = running_loss / len(train_loader.dataset)
            val_loss_avg = val_loss / len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on input X using the trained CNN.
        :param X: shape (N, num_bands)
        :return: predictions, shape (N,)
        """
        self.model.eval()
        num_bands = X.shape[1]
        X_torch = torch.from_numpy(X.reshape(-1, 1, num_bands)).float().to(self.device)

        with torch.no_grad():
            preds = self.model(X_torch).cpu().numpy().flatten()
        return preds

    def save_model(self, path: str):
        """
        Saves the PyTorch model state dict.
        """
        if self.model is None:
            raise RuntimeError("Cannot save because CNN architecture is not built or trained yet.")

        # We can inspect layers to get num_bands:
        # The final fc layer has input_features = filters[-1] * num_bands
        # But let's store the original hyperparams from self.params, 
        # plus the actual num_bands from the shape of the final linear.
        
        final_fc_in = self.model.fc.in_features
        filters = self.params.get("filters", [64, 128])
        num_bands = final_fc_in // filters[1] if filters[1] != 0 else 0
        
        checkpoint = {
            "in_channels": 1,
            "num_bands": num_bands,
            "filters": filters,
            "kernel_size": self.params.get("kernel_size", 3),
            "dropout_rate": self.params.get("dropout_rate", 0.2),
            "state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        """
        Load a checkpoint that includes architecture hyperparameters and state_dict.
        Rebuild the CNN, then load the saved weights.
        """
        checkpoint = torch.load(path, map_location=self.device)

        in_channels = checkpoint["in_channels"]
        num_bands   = checkpoint["num_bands"]
        filters     = checkpoint["filters"]
        kernel_size = checkpoint["kernel_size"]
        dropout_rate= checkpoint["dropout_rate"]

        self._build_cnn(
            in_channels=in_channels,
            num_bands=num_bands,
            filters=filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )

        if self.model is None:
            raise RuntimeError("Model architecture not initialized. Cannot load state dict.")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _build_cnn(self, in_channels: int, num_bands: int, filters: list, kernel_size: int, dropout_rate: float):
        """
        Helper to build CNNRegressor and store in self.model.
        """
        self.model = CNNRegressor(
            in_channels=in_channels,
            num_bands=num_bands,
            filters=filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        ).to(self.device)
