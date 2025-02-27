from zenml import pipeline, step
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

@step
def load_data():
    return pd.read_csv("data/input.csv")

@step
def train_model(data: pd.DataFrame):
    X = data[["product_id", "purchase_amount"]]
    y = data["purchase_amount"].apply(lambda x: 1 if x > 100 else 0)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    with open("src/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model

@pipeline
def ml_pipeline(data_loader, trainer):
    data = data_loader()
    model = trainer(data)

# Running the pipeline
if __name__ == "__main__":
    ml_pipeline(load_data(), train_model())
    print("✅ ZenML pipeline executed!")
