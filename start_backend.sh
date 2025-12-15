#!/bin/bash
echo "Starting FastAPI Backend Server..."
python -m uvicorn main:app --reload --port 8000

