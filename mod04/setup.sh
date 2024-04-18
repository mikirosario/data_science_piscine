#!/bin/bash

# Check if the virtual environment directory exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate
# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt
echo "The virtual environment is set up and dependencies are installed."
echo "Remember to activate the virtual environment with 'source venv/bin/activate'"
