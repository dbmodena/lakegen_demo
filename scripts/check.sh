#!/usr/bin/env bash
# The quality gate: what every commit passes, locally (scripts/hooks/pre-commit)
# and in CI (.gitlab-ci.yml runs its two stages as separate jobs).
#
#   scripts/check.sh                    # lint, then tests
#   scripts/check.sh lint               # ruff (lint and format), bandit, import contracts
#   scripts/check.sh tests              # the suite against the baseline, and an import smoke
#   scripts/check.sh --update-baseline  # also rewrite tests/golden/BASELINE.txt from this run
#
# Lint covers the whole tree as configured in pyproject.toml ([tool.ruff],
# [tool.bandit], [tool.importlinter]); experiments/ is left out there.
# Tests: no failure or error, and every test id recorded in
# tests/golden/BASELINE.txt still ran, so a test cannot disappear unnoticed.
# The golden snapshots under tests/golden run as part of the suite, which is
# hermetic through tests/conftest.py.
set -euo pipefail
# Byte-order sorting everywhere (sort, comm, and the baseline file), so the
# comparison does not depend on the locale of whoever runs the gate.
export LC_ALL=C
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BIN="$ROOT/.venv/bin"
BASELINE="$ROOT/tests/golden/BASELINE.txt"
STAGE=all
UPDATE=0
REPORT=$(mktemp)
IDS=$(mktemp)
trap 'rm -f "$REPORT" "$IDS"' EXIT
for arg in "$@"; do
  case "$arg" in
    lint|tests) STAGE=$arg ;;
    --update-baseline) UPDATE=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
for tool in python ruff bandit lint-imports; do
  [ -x "$BIN/$tool" ] || { echo "$tool is missing: run ./install.sh (uv sync --group dev)"; exit 1; }
done

lint() {
  echo "== ruff"
  "$BIN/ruff" check .
  "$BIN/ruff" format --check .
  echo "== bandit"
  # The Python files git would commit, tests aside: not the local, ignored
  # helpers in scripts/, which bandit (unlike ruff) does not skip by itself.
  local files=()
  mapfile -t files < <(git ls-files --cached --others --exclude-standard -- src pneuma_engine solr_builder experiment_runner scripts \
    | grep '\.py$' | grep -v '/tests/' | while read -r file; do [ -f "$file" ] && echo "$file"; done)
  # Bandit applies a line's `# nosec` to every literal of a multi-line dict or
  # f-string and warns for each one that had nothing to report: noise, dropped.
  "$BIN/bandit" -q -c pyproject.toml "${files[@]}" 2>&1 | { grep -v "nosec encountered" || true; }
  echo "== import contracts"
  PYTHONPATH="$ROOT/src" "$BIN/lint-imports" --no-cache > /dev/null \
    || { PYTHONPATH="$ROOT/src" "$BIN/lint-imports" --no-cache; exit 1; }
  echo "lint clean"
}

tests() {
  echo "== tests"
  "$BIN/python" -m pytest -q -p no:cacheprovider --tb=short -rA > "$REPORT" || true
  # One line per test: "<node id> <OUTCOME>", sorted by id. Only pytest's own
  # summary lines qualify ("ERROR tests/..."); captured log lines such as
  # "ERROR    lakegen.retrieval..." start with the level padded by spaces.
  grep -E '^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED) (tests|pneuma_engine|solr_builder)/' "$REPORT" \
    | sed -E 's/ - .*//' | awk '{print $2, $1}' | sort > "$IDS"
  tail -1 "$REPORT"
  if ! grep -q ' PASSED$' "$IDS"; then
    tail -30 "$REPORT"
    echo "gate: pytest produced no results"
    exit 1
  fi
  if grep -qE ' (FAILED|ERROR)$' "$IDS"; then
    grep -E ' (FAILED|ERROR)$' "$IDS"
    echo "gate: failing tests"
    exit 1
  fi
  if [ "$UPDATE" = 1 ]; then
    cut -d' ' -f1 "$IDS" > "$BASELINE"
    echo "baseline rewritten: $(wc -l < "$BASELINE") tests"
  else
    local missing
    missing=$(comm -23 "$BASELINE" <(cut -d' ' -f1 "$IDS"))
    if [ -n "$missing" ]; then
      echo "$missing" | head -20
      echo "gate: tests recorded in tests/golden/BASELINE.txt did not run" \
           "(removed or renamed? rerun with --update-baseline if that is intended)"
      exit 1
    fi
    echo "baseline: all $(wc -l < "$BASELINE") recorded tests ran"
  fi

  echo "== import smoke"
  export PYTHONPATH="$ROOT/src:$ROOT"
  "$BIN/lakegen" --help > /dev/null
  "$BIN/lakegen" ask --help > /dev/null
  "$BIN/python" -m experiment_runner --help > /dev/null
  "$BIN/python" -m experiment_runner bench --help > /dev/null
  "$BIN/python" -c "import experiment_runner.batches.server"
  "$BIN/python" -c "import lakegen.entrypoints.api.app, lakegen.entrypoints.webapp.workflow"
  # The installed package imports on its own: no PYTHONPATH, from another directory.
  (cd /tmp && "$BIN/python" -I -c "import lakegen.pipeline.service")
  echo "ok"
}

case "$STAGE" in
  lint) lint ;;
  tests) tests ;;
  all) lint; tests ;;
esac
