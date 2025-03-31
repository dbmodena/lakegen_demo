#!/bin/bash

# Exit on error
set -e

# Install dependencies using poetry
poetry install

# Activate the virtual environment
source $(poetry env info --path)/bin/activate

# Start Streamlit app in the background
streamlit run frontend/app.py &