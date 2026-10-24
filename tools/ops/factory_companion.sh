#!/usr/bin/env bash
# Route bounded local companion work through the factory-wide supervisor.
#
# Usage:
#   factory_companion.sh --lease-identity ATTEMPT_ID --class light -- rg ...
#   factory_companion.sh --lease-identity ATTEMPT_ID --class client -- codex ...
#   factory_companion.sh --lease-identity ATTEMPT_ID --class test -- \
#     /absolute/venv/bin/python -m pytest -q -o addopts= -p no:xdist NODEID...
#
# This is deliberately argv-only.  It does not accept a command string, shell
# interpreter, or an arbitrary executable under the "light" classification.
set -euo pipefail

usage() {
  echo "usage: $0 [--repo REPO] --lease-identity ATTEMPT_ID --class light|client|test [--approved-test-manifest PATH --approved-test-sha SHA256] -- COMMAND [ARG ...]" >&2
  exit 2
}

HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../.." && pwd -P)"
POLICY=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_CONCURRENCY_20261023.md
POLICY_SHA=af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f
STATE_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/factory-companions
LEASE_ID=
CLASS=
TEST_MANIFEST=
TEST_MANIFEST_SHA=

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) [ $# -ge 2 ] || usage; REPO=$2; shift 2 ;;
    --lease-identity) [ $# -ge 2 ] || usage; LEASE_ID=$2; shift 2 ;;
    --class) [ $# -ge 2 ] || usage; CLASS=$2; shift 2 ;;
    --approved-test-manifest) [ $# -ge 2 ] || usage; TEST_MANIFEST=$2; shift 2 ;;
    --approved-test-sha) [ $# -ge 2 ] || usage; TEST_MANIFEST_SHA=$2; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done

[ -n "$LEASE_ID" ] && [ -n "$CLASS" ] && [ $# -gt 0 ] || usage
[ "$CLASS" = light ] || [ "$CLASS" = client ] || [ "$CLASS" = test ] || usage
[ -f "$POLICY" ] || { echo "REFUSE: concurrency ruling is unavailable: $POLICY" >&2; exit 2; }
[ "$(shasum -a 256 "$POLICY" | awk '{print $1}')" = "$POLICY_SHA" ] || {
  echo "REFUSE: concurrency ruling SHA does not match the bound policy" >&2
  exit 2
}
REPO="$(cd "$REPO" && pwd -P)"
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "REFUSE: --repo is not a Git worktree: $REPO" >&2
  exit 2
}

reject() { echo "REFUSE: $1" >&2; exit 2; }

WORKTREE=$(pwd -P)
validate_test_manifest() {
  local source_commit= fixture_count=0 source_seen=0 line value fixture_path fixture_sha
  source_commit=$(git -C "$WORKTREE" rev-parse HEAD) || reject "test worktree has no resolved commit"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      schema=factory-companion-test-manifest/v2) ;;
      source_commit=*)
        [ "${line#source_commit=}" = "$source_commit" ] || reject "test manifest source commit is not this worktree"
        source_seen=1
        ;;
      fixture=*)
        value=${line#fixture=}
        fixture_path=${value%%|*}; fixture_sha=${value#*|}
        [ "$fixture_path" != "$fixture_sha" ] || reject "test manifest fixture lacks a SHA256"
        case "$fixture_path" in ''|/*|*'..'*|*' '*|*';'*|*'$('*|*'`'*) reject "unsafe test fixture path" ;; esac
        [ "${#fixture_sha}" -eq 64 ] || reject "test manifest fixture SHA256 is invalid"
        case "$fixture_sha" in *[!0-9a-f]*) reject "test manifest fixture SHA256 is invalid" ;; esac
        [ -f "$WORKTREE/$fixture_path" ] || reject "test manifest fixture is unavailable"
        [ "$(shasum -a 256 "$WORKTREE/$fixture_path" | awk '{print $1}')" = "$fixture_sha" ] || \
          reject "test manifest fixture hash does not match"
        fixture_count=$((fixture_count + 1))
        ;;
      node=research/*/test_*.py|node=research/*/test_*.py::*|node=tests/test_*.py|node=tests/test_*.py::*|script=*) ;;
      *) reject "test manifest has an unrecognized line" ;;
    esac
  done < "$TEST_MANIFEST"
  [ "$source_seen" -eq 1 ] || reject "test manifest must pin this worktree source commit"
  [ "$fixture_count" -gt 0 ] || reject "test manifest must pin at least one fixture/source file"
}

if [ "$CLASS" = light ]; then
  command_name=$(basename "$1")
  case "$command_name" in
    rg)
      for arg in "$@"; do
        case "$arg" in --pre|--pre=*|--pre-glob|--pre-glob=*) reject "light rg must not use a preprocessor" ;; esac
      done
      ;;
    cat) ;;
    sed)
      for arg in "$@"; do
        case "$arg" in -e|--expression|--expression=*|--file|-f) reject "light sed accepts only read-only address/print argv" ;; esac
      done
      ;;
    jq)
      for arg in "$@"; do
        case "$arg" in --run-tests|--from-file|-f) reject "light jq must not execute test or program files" ;; esac
      done
      ;;
    git)
      case "${2:-}" in
        status|diff|show|log) ;;
        *) reject "light git commands are limited to status, diff, show, and log" ;;
      esac
      ;;
    bash)
      [ "$1" = /bin/bash ] || reject "light shell checks require /bin/bash"
      [ "${2:-}" = -n ] || reject "light shell is limited to /bin/bash -n FILE"
      [ $# -eq 3 ] || reject "light shell check accepts exactly one file"
      ;;
    *) reject "unclassified executable must use the test lane or a quiet window" ;;
  esac
elif [ "$CLASS" = client ]; then
  [ -n "$TEST_MANIFEST" ] && [ -n "$TEST_MANIFEST_SHA" ] || \
    reject "client lane requires a reviewed manifest path and SHA256"
  [ -f "$TEST_MANIFEST" ] || reject "approved client manifest does not exist"
  [ "$(shasum -a 256 "$TEST_MANIFEST" | awk '{print $1}')" = "$TEST_MANIFEST_SHA" ] || \
    reject "approved client manifest SHA does not match"
  validate_test_manifest
  case "$(basename "$1")" in
    codex|claude) ;;
    *) reject "client lane is limited to the reviewed Astra/Sol client executables" ;;
  esac
else
  [ -n "$TEST_MANIFEST" ] && [ -n "$TEST_MANIFEST_SHA" ] || \
    reject "test lane requires a reviewed manifest path and SHA256"
  [ -f "$TEST_MANIFEST" ] || reject "approved test manifest does not exist"
  case "$TEST_MANIFEST_SHA" in
    ????????????????????????????????????????????????????????????????) ;;
    *) reject "approved test manifest SHA256 must be 64 lowercase hex" ;;
  esac
  case "$TEST_MANIFEST_SHA" in *[!0-9a-f]*) reject "approved test manifest SHA256 must be lowercase hex" ;; esac
  [ "$(shasum -a 256 "$TEST_MANIFEST" | awk '{print $1}')" = "$TEST_MANIFEST_SHA" ] || \
    reject "approved test manifest SHA does not match"
  validate_test_manifest
  case "$1" in
    */venv/bin/python) ;;
    *) reject "test lane requires an absolute project venv Python" ;;
  esac
  [ "${2:-}" = -m ] && [ "${3:-}" = pytest ] && [ "${4:-}" = -q ] || \
    reject "test lane requires python -m pytest -q"
  [ "${5:-}" = -o ] && [ "${6:-}" = addopts= ] && [ "${7:-}" = -p ] && \
    [ "${8:-}" = no:xdist ] && [ "${9:-}" = -p ] && \
    [ "${10:-}" = research.compute.factory_companion_pytest ] || \
    reject "test lane requires -p no:xdist and the factory thread plugin"
  [ $# -gt 10 ] || reject "test lane requires an explicit node ID or small named file list"
  for arg in "${@:11}"; do
    case "$arg" in
      -n|-n*|--numprocesses*|--dist*|--cov*|--benchmark*|--looponfail*|--lf|--ff|--pdb|--capture*)
        reject "test lane forbids pytest fanout, coverage, benchmark, watch, and interactive options"
        ;;
      -*|.|/*|*'..'*|*' '*|*';'*|*'$('*|*'`'*)
        reject "test lane accepts only reviewed relative pytest files or node IDs"
        ;;
      research/*/test_*.py|research/*/test_*.py::*|tests/test_*.py|tests/test_*.py::*)
        grep -Fx -- "node=$arg" "$TEST_MANIFEST" >/dev/null || \
          reject "test node is absent from the approved manifest"
        ;;
      *) reject "test lane accepts only reviewed pytest files or node IDs" ;;
    esac
  done
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
fi

# The supervisor sets and verifies absolute nice 15/19 and the CPU/thread
# environment in the owned child process.  `exec` keeps this wrapper from
# becoming an unregistered local workload.
PYTHON=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
[ -x "$PYTHON" ] || reject "supervisor Python is not executable: $PYTHON"
if [ "$CLASS" = client ]; then
  MODE_ARGS=(client --)
else
  MODE_ARGS=(tool --class "$CLASS" --)
fi
if [ -n "$TEST_MANIFEST" ]; then
  PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON" -m research.compute.factory_companion \
    --root "$STATE_ROOT" --policy-sha "$POLICY_SHA" --lease-identity "$LEASE_ID" \
    --review-manifest "$TEST_MANIFEST" --review-sha "$TEST_MANIFEST_SHA" --worktree "$WORKTREE" \
    "${MODE_ARGS[@]}" "$@"
fi
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON" -m research.compute.factory_companion \
  --root "$STATE_ROOT" \
  --policy-sha "$POLICY_SHA" \
  --lease-identity "$LEASE_ID" \
  --worktree "$WORKTREE" "${MODE_ARGS[@]}" "$@"
