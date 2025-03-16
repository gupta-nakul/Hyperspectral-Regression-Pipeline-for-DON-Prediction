from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass
