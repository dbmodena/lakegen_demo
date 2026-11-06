#!/usr/bin/env bash
# Create the Pneuma engine's own environment, apart from LakeGen's.
#
#   pneuma_engine/install.sh
#
# PNEUMA_VENV          where to create it (default: .venv-pneuma at the repository root)
# PNEUMA_PYTHON        interpreter version uv creates it with (default: 3.12)
# PNEUMA_CONSTRAINTS   a pip constraints file to apply, e.g. pneuma_engine/constraints.txt
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${PNEUMA_VENV:-$ROOT/.venv-pneuma}
REQUIREMENTS="$ROOT/pneuma_engine/requirements.txt"
CONSTRAINTS=()
if [ -n "${PNEUMA_CONSTRAINTS:-}" ]; then
  CONSTRAINTS=(-c "$PNEUMA_CONSTRAINTS")
fi

echo "Creating the Pneuma environment in $VENV"
if command -v uv > /dev/null; then
  uv venv --python "${PNEUMA_PYTHON:-3.12}" "$VENV"
  uv pip install --python "$VENV/bin/python" -r "$REQUIREMENTS" "${CONSTRAINTS[@]}"
else
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r "$REQUIREMENTS" "${CONSTRAINTS[@]}"
fi
echo "Done. Build an index with pneuma_engine/run_bootstrap.sh <portal>," \
  "serve it with $VENV/bin/python -m pneuma_engine.server (see pneuma_engine/README.md)."
