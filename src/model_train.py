from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import joblib
import os

os.makedirs("models", exist_ok=True)

def train_models(X, y):
    models = {
        "linear": LinearRegression(),
        "xgboost": xgb.XGBRegressor(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(),
        "adaboost": AdaBoostRegressor()
    }

    for name, model in models.items():
        model.fit(X, y)
        joblib.dump(model, f"models/{name}_model.pkl")