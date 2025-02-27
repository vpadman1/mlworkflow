#!/bin/bash

echo "🔹 Activating UV Environment..."
uv venv shell ml_env

echo "🔹 Running ZenML Pipeline..."
python src/pipeline.py

echo "🔹 Running Prefect Workflow..."
python workflows/prefect_flow.py

echo "✅ Workflow completed!"
