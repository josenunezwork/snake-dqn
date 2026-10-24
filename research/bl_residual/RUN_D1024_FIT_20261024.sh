#!/usr/bin/env bash
# F4 D1024 post-collection runbook: final union -> fit pack -> L4 fit -> census -> R-A packet.
# One operator-owned phase per invocation. Never refits, never opens a reserve
# look, never runs Look R-A (ra-command only prints the registered command).
# Ordered commands and decision points: big-lever/exp/f4/HANDOFF_D1024_FIT_20261024.md.
set -euo pipefail

SOURCE=${SOURCE:-/Users/josenunez/Projects/ml/snake-dqn-f4fix}
Q=${Q:-/Users/josenunez/Projects/ml/snake-dqn-f4-d1024-qualified-01.noindex}
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
F4="$ART/big-lever/exp/f4"
D1024_FACTORY_PACK="$ART/bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01"
D256_UNION="$ART/bl-labels-f0/d256-union-final-01/verified-ledger.json"
D1024_UNION="$ART/bl-labels-f0-speed/d1024-union-final-01/verified-ledger.json"
D1024_UNION_PUB=research/bl_labels/ledger/D1024_UNION_FINAL_20261024.json
D256_PACK="$F4/d256-metadata-pack-02"
PACK="$F4/d1024-pack-01"
PREFLIGHT="$F4/d1024-preflight-01.json"
CENSUS="$F4/r-a-archive-census-01"
RA="$F4/r-a-01"
FIT_PHASE=research/bl_residual/ledger/F4_D1024_FIT_PHASE_20261024.json
FIT_BINDING=research/bl_residual/ledger/F4_D1024_FIT_BINDING_20261024.json
CENSUS_PUB=research/bl_residual/ledger/F4_R_A_ARCHIVE_CENSUS_20261024.json
LOG=docs/research/final_run_20261019/CAMPAIGN_LOG.md
THREADS=(OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1)
export "${THREADS[@]}"

PHASES='publish-union|verify-union|make-q|preflight|prepare|upload-check|canary-dry-run|canary|seal-main|fit-dry-run|fit|publish-fit|verify-fit-publication|census|publish-census|verify-census|ra-prepare|ra-command'
if [ "$#" -ne 1 ] || ! printf '%s\n' "$PHASES" | tr '|' '\n' | grep -qx -- "$1"; then
  echo "usage: RUN_D1024_FIT_20261024.sh {$PHASES}" >&2
  exit 2
fi
PHASE=$1
if [ -e "$ART/ops/mac-timing-armed" ]; then
  echo 'HOLD: P1/timing window remains armed; only its owner may close it' >&2
  exit 2
fi

need() { : "${!1:?HOLD: set $1 (see HANDOFF_D1024_FIT_20261024.md)}"; }
clean_source() {
  if [ -n "$(git -C "$SOURCE" status --porcelain)" ]; then
    echo 'HOLD: publication requires a clean source worktree; preserve existing work' >&2
    exit 2
  fi
}
check_q() {
  need PIN
  if [ "$(git -C "$Q" rev-parse HEAD)" != "$PIN" ] || [ -n "$(git -C "$Q" status --porcelain)" ]; then
    echo "HOLD: $Q must be clean and detached at PIN=$PIN" >&2
    exit 2
  fi
}
push_main() {
  git -C "$SOURCE" diff --cached --check
  git -C "$SOURCE" commit -m "$1"
  git -C "$SOURCE" fetch origin
  git -C "$SOURCE" merge --no-edit origin/main
  git -C "$SOURCE" push origin HEAD:main
}
# Copy local immutable bytes into the repository as a fresh triplet named for its path.
publish_copy() {
  # Imports resolve from SOURCE explicitly, independent of the caller's directory.
  (cd "$SOURCE" && PYTHONPATH="$SOURCE" "$PY" - "$1" "$2") <<'PY'
import sys
from pathlib import Path
from research.bl_ppo.exports import export_bytes

source, target = map(Path, sys.argv[1:])
print(export_bytes(target, source.read_bytes())["sha256"])
PY
}
# Published triplet == origin/main, and payload == the local immutable source.
verify_published() {
  git -C "$SOURCE" fetch origin
  for suffix in "" .sha256 .complete; do
    if ! git -C "$SOURCE" show "origin/main:$1$suffix" | cmp -s - "$SOURCE/$1$suffix"; then
      echo "HOLD: local publication differs from origin/main: $1$suffix" >&2
      exit 2
    fi
  done
  cmp -s "$SOURCE/$1" "$2" || { echo "HOLD: published bytes differ from $2" >&2; exit 2; }
  echo "LOCAL == origin/main: $1 sha256 $(shasum -a 256 "$2" | cut -d' ' -f1)"
}
log_line() { printf '\n\n%s\n' "$1" >> "$SOURCE/$LOG"; }
dispatch_mac() {
  local name=$1 pin=$2
  shift 2
  "$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" "$name" "$pin" -- \
    /usr/bin/env "${THREADS[@]}" /usr/bin/nice -n 15 "$PY" "$@"
}

case "$PHASE" in
  publish-union)
    clean_source
    git -C "$SOURCE" fetch origin
    # Pinned publisher: the clean SOURCE commit must already be on origin/main;
    # d1024_final refuses otherwise and records the commit in its output.
    CODE_COMMIT=$(git -C "$SOURCE" rev-parse HEAD)
    cd "$SOURCE"
    nice -n 15 "$PY" -m research.bl_labels.d1024_final --pack "$D1024_FACTORY_PACK" \
      --out "$D1024_UNION" --code-commit "$CODE_COMMIT"
    sha=$(publish_copy "$D1024_UNION" "$SOURCE/$D1024_UNION_PUB")
    log_line "Outcome-blind phase publication: final D1024 F0 union. Ledger \`$D1024_UNION_PUB\`, SHA256 \`$sha\` (local \`$D1024_UNION\`), publisher code commit \`$CODE_COMMIT\`. Final COMPLETE attempt chain verified, zero pending parents; wrapper path/size/hash only; published before F4 D1024 partition preparation, census or any analysis."
    git -C "$SOURCE" add "$D1024_UNION_PUB" "$D1024_UNION_PUB.sha256" "$D1024_UNION_PUB.complete" "$LOG"
    push_main 'Publish outcome-blind final D1024 union ledger'
    verify_published "$D1024_UNION_PUB" "$D1024_UNION"
    echo "D1024_UNION_SHA=$sha  PIN candidate: $(git -C "$SOURCE" rev-parse HEAD)"
    ;;
  verify-union)
    cd "$SOURCE"
    nice -n 15 "$PY" -m research.bl_labels.d1024_final --pack "$D1024_FACTORY_PACK" --out "$D1024_UNION" --verify
    verify_published "$D1024_UNION_PUB" "$D1024_UNION"
    ;;
  make-q)
    need PIN
    git -C "$SOURCE" fetch origin
    git -C "$SOURCE" merge-base --is-ancestor "$PIN" origin/main
    git -C "$SOURCE" cat-file -e "$PIN:$D1024_UNION_PUB"
    [ ! -e "$Q" ] || { echo "HOLD: $Q exists; choose a fresh Q" >&2; exit 2; }
    git -C "$SOURCE" worktree add --detach "$Q" "$PIN"
    check_q
    "$PY" "$Q/tools/ops/qualification.py" check --repo "$Q" \
      --record "$ART/ops/colab-cli-qualified.json" --ref "$PIN"
    ;;
  preflight)
    check_q
    cd "$Q"
    F4_PREFLIGHT_OUT="$PREFLIGHT" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS='' PYTEST_PLUGINS='' \
      nice -n 15 "$PY" - <<'PY'
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from research.bl_residual.f4_d1024_package import PREFLIGHT_FIXTURES, PREFLIGHT_SCHEMA
from research.compute.factory_interim_launcher import run_test

nodes = sorted(PREFLIGHT_FIXTURES)
pin = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
lines = ["schema=factory-companion-test-manifest/v2", "source_commit=" + pin]
for node in nodes:
    lines += ["fixture=" + node + "|" + hashlib.sha256(Path(node).read_bytes()).hexdigest(),
              "node=" + node]
with tempfile.TemporaryDirectory(prefix="f4-d1024-preflight-") as directory:
    manifest = Path(directory) / "reviewed-tests.txt"
    manifest.write_text("\n".join(lines) + "\n")
    command = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:xdist",
               "-p", "research.compute.factory_companion_pytest", *nodes]
    started = time.monotonic()
    status = run_test(command, review_manifest=str(manifest),
                      review_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    if status:
        raise SystemExit(status)
    from research.bl_ppo.exports import export_json

    output = Path(os.environ["F4_PREFLIGHT_OUT"])
    export_json(output, {
        "schema": PREFLIGHT_SCHEMA, "status": "PASS", "commit": pin, "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "fixtures": {node: hashlib.sha256(Path(node).read_bytes()).hexdigest() for node in nodes},
        "fits_run": False, "study_outcomes_opened": False,
    })
    print("PREFLIGHT_PASS " + str(output))
PY
    ;;
  prepare)
    check_q
    need D1024_UNION_SHA
    dispatch_mac bl-f4-d1024-prepare-01 "$PIN" -m research.bl_residual.f4_d1024_package \
      --out "$PACK" --d256-ledger "$D256_UNION" --d1024-ledger "$D1024_UNION" \
      --d1024-union-sha256 "$D1024_UNION_SHA" --preflight-receipt "$PREFLIGHT" --commit "$PIN"
    ;;
  upload-check)
    "$PY" - "$PACK/pack.json" <<'PY'
import json
import sys

projection = json.load(open(sys.argv[1]))["upload_projection"]
print(json.dumps(projection, indent=2))
if projection["status"] != "WITHIN_CANARY_BAR":
    raise SystemExit("HOLD: projected canary transfer > 150 s; do not dispatch; return byte counts to a judge")
PY
    ;;
  canary-dry-run|canary)
    check_q
    cd "$Q"
    extra=()
    [ "$PHASE" = canary-dry-run ] && extra=(--dry-run)
    "$Q/tools/ops/dispatch.sh" bl-f4-d1024-canary-01 "$PACK/canary.json" \
      --broker "$Q" --cwd "$Q" --state-dir "$F4/d1024-canary-state-01" "${extra[@]}"
    ;;
  seal-main)
    check_q
    cd "$Q"
    "$PY" -m research.bl_residual.f4_d1024_package \
      --seal-main-template "$PACK/main-template.json" \
      --canary-receipt "$PACK/canary-output-l4-01/receipt.json" \
      --main-manifest-out "$PACK/main.json"
    ;;
  fit-dry-run|fit)
    check_q
    cd "$Q"
    extra=()
    [ "$PHASE" = fit-dry-run ] && extra=(--dry-run)
    "$Q/tools/ops/dispatch.sh" bl-f4-d1024-fit-01 "$PACK/main.json" \
      --broker "$Q" --cwd "$Q" --state-dir "$F4/d1024-fit-state-01" "${extra[@]}"
    ;;
  publish-fit)
    check_q
    clean_source
    cd "$Q"
    # Reads supervisor completion, control identity and export path/size/hash only.
    "$PY" -m research.bl_residual.f4_d1024_package --publish-fit "$PACK" --out "$SOURCE/$FIT_PHASE"
    "$PY" - "$SOURCE" "$FIT_PHASE" "$FIT_BINDING" "$PACK" "$PIN" <<'PY'
import hashlib
import sys
from pathlib import Path
from research.bl_ppo.exports import export_json

repo, phase, binding, pack, pin = sys.argv[1:]
repo, pack = Path(repo), Path(pack)
files = {}
for name, path in {
    "manifest": pack / "main.json",
    "supervisor_receipt": pack / "d1024-output-l4-01/receipt.json",
    "producer_receipt": pack / "d1024-output-l4-01/exports/fit/receipt.json",
}.items():
    files[name] = {"path": str(path), "size": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
export_json(repo / binding, {
    "schema": "bl-residual-metadata-fit-binding/v1", "fit_pin": pin, "pack": str(pack),
    "phase": phase, "phase_sha256": hashlib.sha256((repo / phase).read_bytes()).hexdigest(),
    "files": files,
})
PY
    log_line "Outcome-blind phase publication: F4 D1024 fit. Ledger \`$FIT_PHASE\`, SHA256 \`$(shasum -a 256 "$SOURCE/$FIT_PHASE" | cut -d' ' -f1)\`; binding \`$FIT_BINDING\`, SHA256 \`$(shasum -a 256 "$SOURCE/$FIT_BINDING" | cut -d' ' -f1)\`; fit pin \`$PIN\`. Published before the census consumer, R-A preparation or any look; no fit/reserve/gameplay outcome opened."
    git -C "$SOURCE" add "$FIT_PHASE" "$FIT_PHASE.sha256" "$FIT_PHASE.complete" \
      "$FIT_BINDING" "$FIT_BINDING.sha256" "$FIT_BINDING.complete" "$LOG"
    push_main 'Publish F4 D1024 fit phase ledger'
    verify_published "$FIT_PHASE" "$SOURCE/$FIT_PHASE"
    verify_published "$FIT_BINDING" "$SOURCE/$FIT_BINDING"
    ;;
  verify-fit-publication)
    verify_published "$FIT_PHASE" "$SOURCE/$FIT_PHASE"
    verify_published "$FIT_BINDING" "$SOURCE/$FIT_BINDING"
    ;;
  census)
    need D1024_UNION_SHA
    need CENSUS_PIN
    # The census opens reserve rows only after both fits are immutable and published.
    git -C "$SOURCE" fetch origin
    for path in "$FIT_PHASE" "$D1024_UNION_PUB"; do
      git -C "$SOURCE" cat-file -e "$CENSUS_PIN:$path"
    done
    dispatch_mac bl-f4-r-a-census-01 "$CENSUS_PIN" -m research.bl_residual.f4_archive_census \
      --out "$CENSUS" --d256-ledger "$D256_UNION" --d1024-ledger "$D1024_UNION" \
      --d1024-union-sha256 "$D1024_UNION_SHA" --d256-pack-inputs "$D256_PACK/inputs" \
      --d1024-pack-inputs "$PACK/inputs"
    ;;
  publish-census)
    clean_source
    sha=$(publish_copy "$CENSUS/census.json" "$SOURCE/$CENSUS_PUB")
    log_line "Outcome-blind phase publication: F4 R5 natural-archive census frozen and hashed before Look R-A. Receipt \`$CENSUS_PUB\`, SHA256 \`$sha\`; local table \`$CENSUS/archive-table.npz\` bound by its table_sha256 (Mac-only). Identities and counts only; no metric or look."
    git -C "$SOURCE" add "$CENSUS_PUB" "$CENSUS_PUB.sha256" "$CENSUS_PUB.complete" "$LOG"
    push_main 'Publish F4 R-A natural-archive census'
    verify_published "$CENSUS_PUB" "$CENSUS/census.json"
    ;;
  verify-census)
    verify_published "$CENSUS_PUB" "$CENSUS/census.json"
    ;;
  ra-prepare)
    need RA_PIN
    git -C "$SOURCE" fetch origin
    for path in "$FIT_PHASE" "$CENSUS_PUB" "$D1024_UNION_PUB"; do
      git -C "$SOURCE" cat-file -e "$RA_PIN:$path"
    done
    dispatch_mac bl-f4-r-a-prepare-01 "$RA_PIN" -m research.bl_residual.f4_ra_prepare \
      --d256-pack "$D256_PACK" --d1024-pack "$PACK" --census-dir "$CENSUS" \
      --funding-out "$RA/funding-look.json" --packet-out "$RA/command-packet.json" --pin "$RA_PIN"
    ;;
  ra-command)
    need RA_PIN
    git -C "$SOURCE" merge-base --is-ancestor "$RA_PIN" HEAD
    # Refuses if ANY F4 R-A lock exists (not only the derived path) or output exists.
    (cd "$SOURCE" && PYTHONPATH="$SOURCE" "$PY" -m research.bl_residual.f4_ra_prepare \
      --print-command "$RA/command-packet.json" --source "$SOURCE" --pin "$RA_PIN")
    ;;
esac
