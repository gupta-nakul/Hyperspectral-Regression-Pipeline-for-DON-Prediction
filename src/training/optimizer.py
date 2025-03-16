import optuna
from sklearn.model_selection import train_test_split
from src.models.xgboost_model import XGBoostModel

def objective(trial, X, y):
    # Param search for XGBoost
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
    }
    model = XGBoostModel(params)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model.train(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_val)
    rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
    return rmse

def run_optimization(X, y, n_trials=20):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    return study.best_params
