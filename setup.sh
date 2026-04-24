#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

get_python_cmd() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        echo "python"
        return
    fi

    echo ""
}

check_python_version() {
    local py_cmd="$1"
    "$py_cmd" - <<'PY'
import sys
ok = (sys.version_info.major > 3) or (sys.version_info.major == 3 and sys.version_info.minor >= 10)
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
raise SystemExit(0 if ok else 1)
PY
}

echo "[setup] Checking Python installation..."
PYTHON_CMD="$(get_python_cmd)"

if [[ -z "$PYTHON_CMD" ]]; then
    echo "[setup] Python not found. Install Python 3.10+ and re-run ./setup.sh" >&2
    exit 1
fi

if ! PYTHON_VERSION="$(check_python_version "$PYTHON_CMD")"; then
    echo "[setup] Found Python version below 3.10. Python 3.10+ is required." >&2
    exit 1
fi

echo "[setup] Python $PYTHON_VERSION detected."

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[setup] Creating virtual environment (.venv)..."
    "$PYTHON_CMD" -m venv .venv
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[setup] Failed to create .venv." >&2
    exit 1
fi

echo "[setup] Installing dependencies from requirements.txt..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

echo
echo "[setup] Done. Run the dashboard with:"
echo "./.venv/bin/python -m streamlit run src/ui/dashboard.py"
