from prefect import flow, task
import subprocess
from zenml import pipeline

@task
def run_zenml_pipeline():
    subprocess.run(["python", "src/pipeline.py"])

@task
def run_predictions():
    subprocess.run(["python", "src/predict.py"])

@flow
def ml_workflow():
    run_zenml_pipeline()
    run_predictions()

if __name__ == "__main__":
    ml_workflow()
    print("✅ Prefect workflow executed!")
