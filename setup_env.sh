#!/bin/bash

# Name of the conda environment
ENV_NAME="gwbird_env"

echo "🔧 Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.11

echo "✅ Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "📦 Installing required packages"
conda install -y numpy scipy mpmath matplotlib glob2 ipykernel
conda install -y -c conda-forge healpy

echo "📡 Installing PINT from GitHub"
pip install git+https://github.com/nanograv/PINT.git

echo "📚 Registering environment as Jupyter kernel"
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

echo "✅ Verifying installation"
python -c "import numpy, healpy, scipy, mpmath, matplotlib, pint, glob; print('✅ All packages installed correctly.')"

echo "📂 Installing GWBird"
pip install .

echo "🎉 Setup complete. To activate the environment, run: conda activate $ENV_NAME"
