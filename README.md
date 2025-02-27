# ML Workflow

A machine learning workflow using ZenML and Prefect for orchestration.

## Overview

This project demonstrates a complete ML workflow that:
1. Loads data from CSV
2. Preprocesses the data
3. Trains a Random Forest model
4. Makes predictions
5. Saves results to both CSV and SQLite database

The workflow is orchestrated using both ZenML (for pipeline steps) and Prefect (for overall workflow management).

## Project Structure

mlworkflow/
├── data/
│ └── input.csv # Training data
├── src/
│ ├── pipeline.py # ZenML pipeline
│ ├── predict.py # Prediction pipeline
│ └── train.py # Training pipeline
├── workflows/
│ └── prefect_flow.py # Prefect workflow
├── .gitignore # Git ignore file
├── README.md # Project documentation
└── run.sh # Shell script to run the workflow

## Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/mlworkflow.git
cd mlworkflow

2. Install dependencies:
bash
uv sync

3. Run the workflow:
bash
# Initialize ZenML
zenml init

# Run the Prefect workflow
python workflows/prefect_flow.py
