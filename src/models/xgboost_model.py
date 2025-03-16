import xgboost as xgb
import joblib
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBRegressor(**params)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
