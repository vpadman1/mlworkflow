from prefect import flow, task
import subprocess

@task
def train_model():
    subprocess.run(["python", "src/train.py"])

@task
def run_predictions():
    subprocess.run(["python", "src/predict.py"])

@flow
def ml_workflow():
    train_model()
    run_predictions()

if __name__ == "__main__":
    ml_workflow()
    print("✅ Prefect workflow executed!")
