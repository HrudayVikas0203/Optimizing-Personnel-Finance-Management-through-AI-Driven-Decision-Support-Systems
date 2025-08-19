from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    y = df["Total_Expenditure"]
    X = df.drop(columns=["Total_Expenditure"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler