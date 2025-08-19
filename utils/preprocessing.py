import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path="data/finance_dataset.csv"):
    """Load dataset from CSV"""
    return pd.read_csv(path)

def clean_data(df):
    """Remove duplicates and handle missing values"""
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    return df

def feature_engineering(df):
    """Add derived features"""
    if "total_income" in df.columns and "savings" in df.columns:
        df["savings_ratio"] = df["savings"] / (df["total_income"] + 1)
    return df

def preprocess_data(df, target="savings"):
    """Preprocess dataset: scaling numeric + encoding categoricals"""
    df = clean_data(df)
    df = feature_engineering(df)

    X = df.drop(columns=[target])
    y = df[target]

    categorical = ["marital_status", "occupation"]
    numerical = [col for col in X.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ]
    )

    # Return pipeline (preprocessor only, models will be separate)
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor
