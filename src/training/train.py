from sklearn.model_selection import train_test_split

from src.models.xgboost_model import XGBoostModel
from src.models.cnn_model import CNNModel
from src.models.fcn_model import FCNModel
from src.models.transformer_model import TransformerModel
from src.utils.logger import get_logger
from src.utils.load_config import config

logger = get_logger(__name__)

def train_all_models(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config["training"]["validation_split"], random_state=42
    )

    logger.info("Training FCN model...")
    fcn_params = config["model"]["fcn"]
    fcn_model = FCNModel(fcn_params)
    fcn_model.train(X_train, y_train, X_val, y_val)
    fcn_model.save_model(config["deployment"]["fcn_model_path"])
    logger.info(f"FCN model saved to {config['deployment']['fcn_model_path']}")
    
    logger.info("Training XGBoost model...")
    xgb_params = config["model"]["xgboost"]
    xgb_model = XGBoostModel(xgb_params)
    xgb_model.train(X_train, y_train, X_val, y_val)
    xgb_model.save_model(config["deployment"]["xgboost_model_path"])
    logger.info(f"XGBoost model saved to {config['deployment']['xgboost_model_path']}")

    logger.info("Training CNN model...")
    cnn_params = config["model"]["cnn"]
    cnn_model = CNNModel(cnn_params)
    cnn_model.train(X_train, y_train, X_val, y_val)
    cnn_model.save_model(config["deployment"]["cnn_model_path"])
    logger.info(f"CNN model saved to {config['deployment']['cnn_model_path']}")

    # logger.info("Training Transformer model...")
    # trans_params = config["model"]["transformer"]
    # trans_model = TransformerModel(trans_params)
    # trans_model.train(X_train, y_train, X_val, y_val)
    # trans_model.save_model(config["deployment"]["transformer_model_path"])
    # logger.info(f"Transformer model saved to {config['deployment']['transformer_model_path']}")

    logger.info("Training pipeline complete.")

def main(X, y):
    train_all_models(X, y)

if __name__ == "__main__":
    pass
