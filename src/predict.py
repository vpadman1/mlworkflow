import pandas as pd
import pickle
import sqlite3

# Load model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("data/input.csv")
X = df[["product_id", "purchase_amount"]]

# Predict
df["prediction"] = model.predict(X)

# Save predictions
df.to_csv("data/predictions.csv", index=False)

# Save to SQLite database
conn = sqlite3.connect("database.db")
df.to_sql("predictions", conn, if_exists="replace", index=False)
conn.close()

print("✅ Predictions saved to database!")