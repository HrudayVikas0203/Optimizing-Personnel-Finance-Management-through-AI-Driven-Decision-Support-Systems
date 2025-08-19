from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return {
        "R2": r2_score(y, predictions),
        "MSE": mean_squared_error(y, predictions),
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "MAE": mean_absolute_error(y, predictions)
    }
