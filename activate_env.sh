#!/bin/bash
# Shared venv activation. Sourced by run_all.sh and all sbatch scripts.
VENV_DIR="/work/scratch/$USER/.venv"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
