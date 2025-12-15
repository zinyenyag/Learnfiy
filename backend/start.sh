#!/bin/bash
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Starting FastAPI Backend Server..."
python -m uvicorn main:app --reload --port 5050

