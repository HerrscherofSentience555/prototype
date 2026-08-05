#!/usr/bin/env bash
set -u

echo "TorchRec Prototype WSL DLRM Environment Check"
echo "============================================="

ok=0
dlrm_root="${DLRM_ROOT:-/mnt/c/Users/han/Desktop/dlrm}"
python_env="${PYTHON_ENV:-$HOME/venvs/torchrec17}"

if [ -d "$dlrm_root" ]; then
  echo "[OK] DLRM_ROOT: $dlrm_root"
else
  echo "[MISSING] DLRM_ROOT: $dlrm_root"
  ok=1
fi

if [ -f "$python_env/bin/activate" ]; then
  echo "[OK] Python env: $python_env"
else
  echo "[MISSING] Python env: $python_env"
  ok=1
fi

if [ -f "$python_env/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$python_env/bin/activate"
  python -c "import torch, torchrec; print('[OK] torch and torchrec import')" || ok=1
fi

if [ "$ok" -eq 0 ]; then
  echo "WSL DLRM environment check passed."
else
  echo "WSL DLRM environment check failed."
fi
exit "$ok"
