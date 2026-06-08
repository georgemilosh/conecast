#!/usr/bin/env bash
# Local environment setup for conecast.
#
# Builds a dedicated conda environment: the scientific stack comes from conda-forge
# (which has macOS/Linux/arm builds for torch, sunpy, etc.), and HUXt + WSA+ are
# installed with pip since they are not on conda. This is the reliable path; the
# pinned requirements.txt is a Linux freeze and may not resolve on other platforms.
#
# Usage:  bash setup.sh [env_name]   (default env name: conecast)
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${1:-conecast}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniforge/Miniconda first." >&2
  echo "(A pip/venv install of the pinned requirements.txt is not recommended off-cluster.)" >&2
  exit 1
fi

# Prefer mamba / the libmamba solver - the classic conda solver is very slow on this stack.
if command -v mamba >/dev/null 2>&1; then
  CREATE=(mamba create)
elif conda create --help 2>/dev/null | grep -q -- "--solver"; then
  CREATE=(conda create --solver=libmamba)
else
  CREATE=(conda create)
fi

echo ">> Creating conda env '${ENV_NAME}' from conda-forge (${CREATE[*]})..."
"${CREATE[@]}" -y -n "${ENV_NAME}" -c conda-forge python=3.12 \
  numpy scipy pandas matplotlib pyyaml h5py joblib \
  astropy sunpy scikit-learn pytorch \
  jupyterlab ipykernel

echo ">> Installing HUXt and WSA+ with pip (not on conda) into '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" python -m pip install wsaplus
conda run -n "${ENV_NAME}" python -m pip install \
  "huxt @ git+https://github.com/University-of-Reading-Space-Science/HUXt"

echo ">> Registering Jupyter kernel '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" python -m ipykernel install --user \
  --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo
echo "The WSA+ checkpoint (~317 MB) and GONG/boundary data are not shipped. Fetch the"
echo "checkpoint from Zenodo with:"
echo "    conda run -n ${ENV_NAME} python scripts/fetch_wsaplus_checkpoint.py"
echo "(notebook 02 also downloads it automatically on first use)."
