#!/usr/bin/env bash
# Offline argv validation for the factory companion shell facade.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WRAPPER="$ROOT/tools/ops/factory_companion.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/venv/bin"
FAKE="$TMP/venv/bin/python"
POLICY=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_CONCURRENCY_20261023.md
MANIFEST="$TMP/approved-tests.txt"
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
FIXTURE=research/bl_labels/test_factory.py
FIXTURE_SHA=$(shasum -a 256 "$ROOT/$FIXTURE" | awk '{print $1}')
printf '%s\n' 'schema=factory-companion-test-manifest/v2' "source_commit=$COMMIT" \
  "fixture=$FIXTURE|$FIXTURE_SHA" 'node=research/bl_labels/test_factory.py::test_placeholder' > "$MANIFEST"
MANIFEST_SHA=$(shasum -a 256 "$MANIFEST" | awk '{print $1}')

printf '%s\n' '#!/usr/bin/env bash' \
  'printf "env=%s\\n" "${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-}" > "$FACTORY_COMPANION_CAPTURE"' \
  'printf "%s\\n" "$@" >> "$FACTORY_COMPANION_CAPTURE"' > "$FAKE"
chmod +x "$FAKE"
export FACTORY_COMPANION_CAPTURE="$TMP/capture"

bash -n "$WRAPPER"
bash -n "$ROOT/tools/ops/factory_companion_monitor.sh"
SNAKE_DQN_PYTHON="$FAKE" "$ROOT/tools/ops/factory_companion_monitor.sh" --repo "$ROOT" status
grep -Fx -- 'status' "$FACTORY_COMPANION_CAPTURE" >/dev/null
SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --repo "$ROOT" --lease-identity attempt-a --class light -- \
  rg -n needle tools/ops
grep -Fx -- '--class' "$FACTORY_COMPANION_CAPTURE" >/dev/null
grep -Fx -- 'light' "$FACTORY_COMPANION_CAPTURE" >/dev/null
grep -Fx -- 'rg' "$FACTORY_COMPANION_CAPTURE" >/dev/null
grep -Fx -- 'af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f' \
  "$FACTORY_COMPANION_CAPTURE" >/dev/null

SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class test \
  --approved-test-manifest "$MANIFEST" --approved-test-sha "$MANIFEST_SHA" -- \
  "$FAKE" -m pytest -q -o addopts= -p no:xdist -p research.compute.factory_companion_pytest \
  research/bl_labels/test_factory.py::test_placeholder
grep -Fx -- 'test' "$FACTORY_COMPANION_CAPTURE" >/dev/null
grep -Fx -- 'env=1' "$FACTORY_COMPANION_CAPTURE" >/dev/null

if SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class light -- python -V \
  >/dev/null 2>&1; then
  echo 'light wrapper accepted an unclassified executable' >&2; exit 1
fi
if SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class light -- \
  rg --pre /usr/bin/false needle research >/dev/null 2>&1; then
  echo 'light wrapper accepted an rg preprocessor' >&2; exit 1
fi
if SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class light -- \
  git push --message status >/dev/null 2>&1; then
  echo 'light wrapper accepted git push with a status-looking argument' >&2; exit 1
fi
if SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class test \
  --approved-test-manifest "$MANIFEST" --approved-test-sha "$MANIFEST_SHA" -- \
  "$FAKE" -m pytest -q -o addopts= -p no:xdist -p research.compute.factory_companion_pytest -n auto research/bl_labels/test_factory.py \
  >/dev/null 2>&1; then
  echo 'test wrapper accepted xdist fanout' >&2; exit 1
fi
if SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class test \
  --approved-test-manifest "$MANIFEST" --approved-test-sha "$MANIFEST_SHA" -- \
  "$FAKE" -m pytest -q -o addopts= -p no:xdist -p research.compute.factory_companion_pytest . >/dev/null 2>&1; then
  echo 'test wrapper accepted whole-suite discovery' >&2; exit 1
fi
SNAKE_DQN_PYTHON="$FAKE" "$WRAPPER" --lease-identity attempt-a --class client \
  --approved-test-manifest "$MANIFEST" --approved-test-sha "$MANIFEST_SHA" -- codex exec --help
grep -Fx -- 'client' "$FACTORY_COMPANION_CAPTURE" >/dev/null
printf 'factory companion shell wrapper checks passed\n'
