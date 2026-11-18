#!/usr/bin/env bash
# Check the Python version and install this artifact's dependencies into ./.venv.
# Uses $PYTHON if set, else python3. Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

# check the interpreter
if ! command -v "$PY" >/dev/null; then
  echo "error: $PY not found; install Python 3.9 or newer" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "error: Python 3.9 or newer is required ($("$PY" --version 2>&1) found)" >&2
  exit 1
fi

# create the venv and install dependencies
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
.venv/bin/python - <<'PYCHECK'
import importlib, sys
missing = []
for mod in ["numpy", "sklearn", "networkx"] + ([] if sys.version_info >= (3, 14) else ["zstandard"]):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit("error: could not import " + ", ".join(missing))
print("Python " + sys.version.split()[0] + "; dependencies OK")
PYCHECK

# the dataset artifact: $STATIC_ARTIFACT, else a sibling clone of static_artifacts_eurosp, else ../static
if [ -n "${STATIC_ARTIFACT:-}" ]; then
  STATIC="$STATIC_ARTIFACT"
elif [ -d ../static_artifacts_eurosp ]; then
  STATIC=../static_artifacts_eurosp
else
  STATIC=../static
fi
export STATIC_ARTIFACT="$STATIC"
if [ ! -f "$STATIC/data/manifests/manifest_v3.csv" ]; then
  echo "note: the dataset artifact was not found at $STATIC; check it out there or set STATIC_ARTIFACT" >&2
fi
echo "Done. Run ./verify.sh"
