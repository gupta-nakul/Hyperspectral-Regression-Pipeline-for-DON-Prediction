import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_regression(y_true, y_pred, plot=True):
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    if plot:
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.xlabel("Actual DON")
        plt.ylabel("Predicted DON")
        plt.title("Actual vs Predicted")
        plt.show()

        residuals = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
