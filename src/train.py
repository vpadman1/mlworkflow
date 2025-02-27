import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("data/input.csv")

# Simulating a target variable (predict if purchase_amount > 100)
df["purchase_again"] = df["purchase_amount"].apply(lambda x: 1 if x > 100 else 0)

# Features and target
X = df[["product_id", "purchase_amount"]]
y = df["purchase_again"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("src/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")