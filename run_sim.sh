#!/bin/bash
# run_sim.sh — Install requirements and run the Cognitive Twin local simulation in one go.

set -e

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "===================================================================="
echo "  1. Installing / Verifying Python Dependencies"
echo "===================================================================="
pip3 install -r requirements.txt

echo -e "\n===================================================================="
echo "  2. Running Cognitive Twin Local Simulation (Simulator Mode)"
echo "===================================================================="
export USE_ENV_EXAMPLE=1
python3 data_pipeline/run.py "$@"
