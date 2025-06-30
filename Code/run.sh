#!/bin/bash

echo "Creating virtual environment: medxformer_venv"
python3 -m venv medxformer_env

echo "Activating virtual environment"
source medxformer_env/bin/activate

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing required packages from requirements.txt"
pip install -r requirements.txt

echo "Running inference.py"
python inference.py

echo "Deactivating virtual environment"
deactivate