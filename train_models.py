import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from utils.preprocessing import load_data, preprocess_data
from utils.evaluation import evaluate_model

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load & preprocess data
df = load_data("data/finance_dataset.csv")
X, y, preprocessor = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "linear_reg": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "adaboost": AdaBoostRegressor(),
    "xgboost_model": XGBRegressor()
}

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"{name} evaluation: {metrics}")
    joblib.dump(model, f"models/{name}.pkl")

# Save preprocessor for later use in app.py
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("✅ All models and preprocessor saved successfully!")
