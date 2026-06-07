#!/usr/bin/env bash
# Local environment setup for conecast.
# Creates a virtual environment and installs the pinned dependencies
# (including HUXt and WSA+ from their git/PyPI sources in requirements.txt).
set -euo pipefail

cd "$(dirname "$0")"

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: register a Jupyter kernel for the tutorial notebooks.
python -m ipykernel install --user --name conecast --display-name "Python (conecast)" || true

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Note: the WSA+ checkpoint data_dir/sw/wsaplus.pt and GONG/boundary data are"
echo "downloaded/generated on first run of scripts/generate_huxt_input.py - not shipped."
