#!/bin/bash

echo "==================================================="
echo "    Causal Impact Engine - Marketing Attribution"
echo "==================================================="
echo

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: Anaconda/Miniconda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda from https://www.anaconda.com/download/"
    echo
    exit 1
fi

echo "Activating Anaconda environment: causal_impact"
echo

# Try to activate the environment
if ! conda activate causal_impact 2>/dev/null; then
    echo "Environment 'causal_impact' not found. Creating it now..."
    echo "This may take a few minutes..."
    echo
    
    # Create the environment from the environment.yml file
    if [ -f environment.yml ]; then
        conda env create -f environment.yml
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create environment."
            exit 1
        fi
    else
        echo "ERROR: environment.yml file not found."
        echo "Please run this script from the project root directory."
        exit 1
    fi
    
    conda activate causal_impact
fi

echo
echo "Changing directory to causal_impact_engine..."
cd causal_impact_engine

echo
echo "Running Streamlit app: frontend/app.py"
echo
echo "NOTE: If the app doesn't start, please ensure all dependencies are installed:"
echo "      pip install -r requirements.txt"
echo

streamlit run frontend/app.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to run Streamlit app."
    echo "Please check that all dependencies are installed correctly."
fi

echo
echo "Streamlit app has exited." 