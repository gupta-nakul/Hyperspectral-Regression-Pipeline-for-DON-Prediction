# transformer_model.py

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import joblib
from typing import Dict, Any

from .base_model import BaseModel

class TransformerRegressor(nn.Module):
    """
    A 1D Transformer for regression over spectral data.
    Expects input shape (batch_size, seq_len) or (batch_size, 1, seq_len) 
    but we'll adapt it to (batch_size, seq_len) for simplicity.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dff: int = 512,
        dropout_rate: float = 0.1
    ):
        super(TransformerRegressor, self).__init__()
        self.input_fc = nn.Linear(seq_len, d_model)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len)
        """
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x


class TransformerModel(BaseModel):
    """
    High-level model class that wraps the TransformerRegressor with 
    training, prediction, save_model, load_model methods.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        :param params: e.g.
            {
                "d_model": 128,
                "num_heads": 8,
                "num_layers": 4,
                "dff": 512,
                "dropout_rate": 0.1,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "seq_length": 100
            }
        """
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = self.params.get("seq_length", 100)
        self.model = TransformerRegressor(
            seq_len=seq_len,
            d_model=self.params.get("d_model", 128),
            num_heads=self.params.get("num_heads", 8),
            num_layers=self.params.get("num_layers", 4),
            dff=self.params.get("dff", 512),
            dropout_rate=self.params.get("dropout_rate", 0.1),
        ).to(self.device)

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
        Train the Transformer regressor.
        :param X_train: shape (N_train, seq_len)
        :param y_train: shape (N_train,)
        :param X_val: shape (N_val, seq_len)
        :param y_val: shape (N_val,)
        """
        seq_len = X_train.shape[1]
        lr = self.params.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        batch_size = self.params.get("batch_size", 32)
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)

        epochs = self.params.get("epochs", 50)
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
            avg_val_loss = val_loss_sum / len(val_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Inference on input X.
        :param X: shape (N, seq_len)
        :return: predictions shape (N,)
        """
        self.model.eval()
        X_torch = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            preds = self.model(X_torch).cpu().numpy().flatten()
        return preds

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
