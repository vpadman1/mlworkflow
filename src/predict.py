import pandas as pd
import pickle
import sqlite3
from zenml import step

@step
def load_model():
    with open("src/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@step
def make_predictions(model, data: pd.DataFrame):
    X = data[["product_id", "purchase_amount"]]
    predictions = model.predict(X)
    data["prediction"] = predictions
    return data

@step
def save_predictions(predictions_df: pd.DataFrame):
    predictions_df.to_csv("data/predictions.csv", index=False)
    conn = sqlite3.connect("database.db")
    predictions_df.to_sql("predictions", conn, if_exists="replace", index=False)
    conn.close()
    
    return "✅ Predictions saved to database!"

if __name__ == "__main__":
    df = pd.read_csv("data/input.csv", skiprows=1)
    with open("src/model.pkl", "rb") as f:
        model = pickle.load(f)
    X = df[["product_id", "purchase_amount"]]
    df["prediction"] = model.predict(X)
    df.to_csv("data/predictions.csv", index=False)
    conn = sqlite3.connect("database.db")
    df.to_sql("predictions", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ Predictions saved to database!")