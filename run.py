#!/usr/bin/env python

"""
run.py

Entry point to run the entire pipeline: data loading, preprocessing,
feature selection (optional), training, and evaluation.

Usage:
    python run.py
"""

import os

from sklearn.model_selection import train_test_split

from src.data.dataset_loader import DatasetLoader
from src.data.preprocess import Preprocessor
from src.deployment.predictor import Predictor
from src.evaluation.evaluate import evaluate_regression
from src.training.train import main as train_main
from src.utils.helper import choose_and_save_best_model
from src.utils.load_config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_evaluation(X, y):
    """
    Evaluate each model on a test split from (X, y). 
    Return a list of (model_name, model_path, metrics).
    """
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    if os.path.exists(config["deployment"]["fcn_model_path"]):
        predictor_fcn = Predictor(model_type="fcn", model_path=config["deployment"]["fcn_model_path"])
        y_pred_fcn = predictor_fcn.predict(X_test)
        metrics_fcn = evaluate_regression(y_test, y_pred_fcn, plot=False)
        results.append(("fcn", config["deployment"]["fcn_model_path"], metrics_fcn))
        logger.info(f"FCN -> MAE: {metrics_fcn['MAE']:.4f}, RMSE: {metrics_fcn['RMSE']:.4f}, R2: {metrics_fcn['R2']:.4f}")

    if os.path.exists(config["deployment"]["xgboost_model_path"]):
        predictor_xgb = Predictor(model_type="xgboost", model_path=config["deployment"]["xgboost_model_path"])
        y_pred_xgb = predictor_xgb.predict(X_test)
        metrics_xgb = evaluate_regression(y_test, y_pred_xgb, plot=False)
        results.append(("xgboost", config["deployment"]["xgboost_model_path"], metrics_xgb))
        logger.info(f"XGBoost -> MAE: {metrics_xgb['MAE']:.4f}, RMSE: {metrics_xgb['RMSE']:.4f}, R2: {metrics_xgb['R2']:.4f}")

    if os.path.exists(config["deployment"]["cnn_model_path"]):
        predictor_cnn = Predictor(model_type="cnn", model_path=config["deployment"]["cnn_model_path"])
        y_pred_cnn = predictor_cnn.predict(X_test)
        metrics_cnn = evaluate_regression(y_test, y_pred_cnn, plot=False)
        results.append(("cnn", config["deployment"]["cnn_model_path"], metrics_cnn))
        logger.info(f"CNN -> MAE: {metrics_cnn['MAE']:.4f}, RMSE: {metrics_cnn['RMSE']:.4f}, R2: {metrics_cnn['R2']:.4f}")

    # if os.path.exists(config["deployment"]["transformer_model_path"]):
    #     predictor_trans = Predictor(model_type="transformer", model_path=config["deployment"]["transformer_model_path"])
    #     y_pred_trans = predictor_trans.predict(X_test)
    #     metrics_trans = evaluate_regression(y_test, y_pred_trans, plot=False)
    #     results.append(("transformer", config["deployment"]["transformer_model_path"], metrics_trans))
    #     logger.info(f"Transformer -> MAE: {metrics_trans['MAE']:.4f}, RMSE: {metrics_trans['RMSE']:.4f}, R2: {metrics_trans['R2']:.4f}")

    return results

def main():
    """
    Main function to orchestrate the entire pipeline.
    1. Train the model(s).
    2. Evaluate the final model on the test set (optional).
    """
    logger.info("Starting run.py main pipeline...")
    loader = DatasetLoader(config["data"]["dataset_path"])
    X, y = loader.get_data()

    preprocessor = Preprocessor(impute_strategy=config["preprocess"]["impute_strategy"], 
                                scale_data=config["preprocess"]["scaler"],
                                n_components=config["preprocess"]["pca_components"]
                                # feature_selection=config["preprocess"]["feature_selection"]
                                )
    X = preprocessor.fit_transform(X)

    logger.info("Calling the training pipeline...")
    train_main(X, y)

    logger.info("Evaluating the models on the test set...")
    results = run_evaluation(X, y)

    best_model_type, best_model_path = choose_and_save_best_model(results)
    if not best_model_type:
        logger.error("No best model found. Exiting.")
        return
    
    logger.info(f"Best model is {best_model_type}")

    logger.info("Pipeline complete!")

if __name__ == "__main__":
    main()
