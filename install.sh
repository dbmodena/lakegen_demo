#!/usr/bin/env bash
# Set up LakeGen: the .venv environment from uv.lock, with the tools of the gate.
#
#   ./install.sh
#
# The Pneuma engine has an environment of its own: pneuma_engine/install.sh.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv > /dev/null; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
uv sync --locked
echo "Done. Configure Solr and the model providers (README.md, docs/configuration.md);" \
  "then try: .venv/bin/lakegen --help"
