#!/usr/bin/env bash
# The commit gate of the refactor: run it before every commit
# (docs/REFACTOR_PLAN.md, ground rule 6).
#
#   scripts/check.sh                    # run the gate
#   scripts/check.sh --update-baseline  # also rewrite tests/BASELINE.txt from this run
#
# 1. The test suite has no failure or error, and every test id recorded in
#    tests/BASELINE.txt still ran, so a test cannot disappear unnoticed. The
#    golden snapshots under tests/golden run as part of the suite. The suite
#    is hermetic through tests/conftest.py, which points the state root at a
#    temporary directory.
# 2. ruff on the Python files staged for this commit.
# 3. Import smoke: the entry points still import and parse their arguments.
set -euo pipefail
# Byte-order sorting everywhere (sort, comm, and the baseline file), so the
# comparison does not depend on the locale of whoever runs the gate.
export LC_ALL=C
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# The launchers' interpreter (Python 3.13). venv/ holds an unrelated 3.11.
PY="$ROOT/.venv/bin/python"
BASELINE="$ROOT/tests/BASELINE.txt"
UPDATE=0
[ "${1:-}" = "--update-baseline" ] && UPDATE=1
REPORT=$(mktemp)
IDS=$(mktemp)
trap 'rm -f "$REPORT" "$IDS"' EXIT

echo "== 1. tests"
"$PY" -m pytest -q -p no:cacheprovider --tb=short -rA tests > "$REPORT" || true
# One line per test: "<node id> <OUTCOME>", sorted by id. Only pytest's own
# summary lines qualify ("ERROR tests/..."); captured log lines such as
# "ERROR    lakegen.retrieval..." start with the level padded by spaces.
grep -E '^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED) tests/' "$REPORT" \
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
elif [ -f "$BASELINE" ]; then
  MISSING=$(comm -23 "$BASELINE" <(cut -d' ' -f1 "$IDS"))
  if [ -n "$MISSING" ]; then
    echo "$MISSING" | head -20
    echo "gate: tests recorded in tests/BASELINE.txt did not run" \
         "(removed or renamed? rerun with --update-baseline if that is intended)"
    exit 1
  fi
  echo "baseline: all $(wc -l < "$BASELINE") recorded tests ran"
else
  echo "no tests/BASELINE.txt yet: run with --update-baseline once the suite is hermetic"
fi

echo "== 2. ruff on staged python files"
# New and rewritten files meet the whole quality bar (pyproject [tool.ruff]).
# Files listed in scripts/ruff_legacy.txt existed before the refactor and only
# get pyflakes' real errors until they are rewritten and taken off the list.
RUFF="$ROOT/.venv/bin/ruff"
[ -x "$RUFF" ] || { echo "ruff is missing: uv pip install --python .venv/bin/python ruff"; exit 1; }
STAGED=$(git diff --cached --name-only --diff-filter=ACMR -- '*.py')
if [ -z "$STAGED" ]; then
  echo "no staged python files"
else
  NEW=$(echo "$STAGED" | grep -vFxf "$ROOT/scripts/ruff_legacy.txt" || true)
  LEGACY=$(echo "$STAGED" | grep -Fxf "$ROOT/scripts/ruff_legacy.txt" || true)
  if [ -n "$NEW" ]; then
    # shellcheck disable=SC2086
    "$RUFF" check $NEW
  fi
  if [ -n "$LEGACY" ]; then
    # shellcheck disable=SC2086
    "$RUFF" check --select F --ignore F401,F841,F541 $LEGACY
  fi
  echo "ruff: $(echo "$NEW" | grep -c .) new and $(echo "$LEGACY" | grep -c .) legacy files clean"
fi

echo "== 3. import smoke"
export PYTHONPATH="$ROOT/src:$ROOT"
"$PY" src/cli.py --help > /dev/null
"$PY" -m lakegen.retrieval.benchmark --help > /dev/null
"$PY" -c "import api, lakegen.ui.workflow"
# The installed package imports on its own: no PYTHONPATH, from another directory.
(cd /tmp && "$PY" -I -c "import lakegen.service")
echo "ok"
