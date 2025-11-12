#! /bin/bash

set -euo pipefail

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

conda env create -f environment.yml
echo "Conda Environment created."

echo "Installing em_binaries analyzer package.."
pip install .

echo "Installation complete for analyzer backend."