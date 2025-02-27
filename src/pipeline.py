from zenml import pipeline, step
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import numpy as np
import sqlite3

@step
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/input.csv", skiprows=1)

@step
def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = df[["product_id", "purchase_amount"]].values
    y = df["purchase_amount"].apply(lambda x: 1 if x > 100 else 0).values
    return X, y, df

@step
def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    with open("src/model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

@step
def make_predictions(model: RandomForestClassifier, original_data: pd.DataFrame) -> pd.DataFrame:
    X = original_data[["product_id", "purchase_amount"]]
    predictions = model.predict(X)
    original_data["prediction"] = predictions
    return original_data

@step
def save_predictions(predictions_df: pd.DataFrame) -> str:
    predictions_df.to_csv("data/predictions.csv", index=False)
    conn = sqlite3.connect("database.db")
    predictions_df.to_sql("predictions", conn, if_exists="replace", index=False)
    conn.close()
    
    return "✅ Predictions saved to database!"

@pipeline
def ml_pipeline():
    df = load_data()
    X, y, original_df = preprocess_data(df)
    model = train_model(X, y)
    predictions_df = make_predictions(model, original_df)
    result = save_predictions(predictions_df)

if __name__ == "__main__":
    ml_pipeline()
    print("✅ ZenML pipeline executed!")