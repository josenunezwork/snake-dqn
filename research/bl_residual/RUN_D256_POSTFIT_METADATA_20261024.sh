#!/usr/bin/env bash
# Post-fit continuation for the completed metadata-pack fit only.
# One operator-owned phase per invocation; never refit or open a D256 reserve look.
set -euo pipefail

SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f4fix
Q=/Users/josenunez/Projects/ml/snake-dqn-f4-metadata-qualified-02.noindex
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
PACK="$ART/big-lever/exp/f4/d256-metadata-pack-02"
F4_INPUTS="$ART/big-lever/exp/f4/d1024-metadata-staging-02/inputs"
FIT_PIN=6313fe8fce990915ef1c3702ff1687169513b624
FIT_PHASE=research/bl_residual/ledger/F4_D256_FIT_PHASE_20261023.json
FIT_BINDING=research/bl_residual/ledger/F4_D256_METADATA_FIT_BINDING_20261024.json
LOG=docs/research/final_run_20261019/CAMPAIGN_LOG.md
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

case "${1:-}" in
  publish-fit|verify-fit-publication|stage-d1024) ;;
  *)
    echo 'usage: RUN_D256_POSTFIT_METADATA_20261024.sh {publish-fit|verify-fit-publication|stage-d1024}' >&2
    exit 2
    ;;
esac
if [ "$#" -ne 1 ]; then
  echo 'HOLD: exactly one phase per invocation' >&2
  exit 2
fi
if [ -e "$ART/ops/mac-timing-armed" ]; then
  echo 'HOLD: P1/timing window remains armed; only its owner may close it' >&2
  exit 2
fi
if [ "$(git -C "$Q" rev-parse HEAD)" != "$FIT_PIN" ] || \
   [ -n "$(git -C "$Q" status --porcelain)" ]; then
  echo 'HOLD: qualified metadata source must be clean and pinned' >&2
  exit 2
fi
cd "$Q"

verify_publication() {
  git -C "$SOURCE" fetch origin
  "$PY" - "$SOURCE" "$FIT_PHASE" "$FIT_BINDING" "$PACK" "$FIT_PIN" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo, phase, binding, pack, pin = sys.argv[1:]
repo, pack = Path(repo), Path(pack)
for relative in (phase, binding):
    for suffix in ("", ".sha256", ".complete"):
        name = relative + suffix
        remote = subprocess.check_output(["git", "-C", str(repo), "show", "origin/main:" + name])
        if remote != (repo / name).read_bytes():
            raise SystemExit("HOLD: local publication differs from origin/main: " + name)
from research.bl_ppo.exports import verify_export

verify_export(repo / phase)
verify_export(repo / binding)
value = json.loads((repo / binding).read_text())
if (
    value.get("schema") != "bl-residual-metadata-fit-binding/v1"
    or value.get("fit_pin") != pin
    or value.get("pack") != str(pack)
    or value.get("phase") != phase
    or value.get("phase_sha256") != hashlib.sha256((repo / phase).read_bytes()).hexdigest()
):
    raise SystemExit("HOLD: foreign metadata-fit publication binding")
expected = {
    "manifest": pack / "main.json",
    "supervisor_receipt": pack / "d256-output-l4-01/receipt.json",
    "producer_receipt": pack / "d256-output-l4-01/exports/fit/receipt.json",
}
if set(value.get("files", {})) != set(expected):
    raise SystemExit("HOLD: incomplete metadata-fit receipt binding")
for name, path in expected.items():
    row = value["files"][name]
    if (
        row.get("path") != str(path)
        or row.get("size") != path.stat().st_size
        or row.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise SystemExit("HOLD: metadata-fit bound file changed: " + name)
print("LOCAL == origin/main; metadata-pack fit ledger and receipt binding verified")
PY
}

case "$1" in
  publish-fit)
    if [ -n "$(git -C "$SOURCE" status --porcelain)" ]; then
      echo 'HOLD: publication requires a clean source worktree; preserve existing work' >&2
      exit 2
    fi
    # The frozen helper checks terminal completion/release, then reads only
    # control, allocation identity and export path/size/hash; never torch payloads.
    "$PY" -m research.bl_residual.f4_d256_package \
      --publish-fit "$PACK" --out "$SOURCE/$FIT_PHASE"
    "$PY" - "$SOURCE" "$FIT_PHASE" "$FIT_BINDING" "$PACK" "$FIT_PIN" "$LOG" <<'PY'
import hashlib
import sys
from pathlib import Path
from research.bl_ppo.exports import export_json

repo, phase, binding, pack, pin, log = sys.argv[1:]
repo, pack = Path(repo), Path(pack)
phase_sha = hashlib.sha256((repo / phase).read_bytes()).hexdigest()
files = {}
for name, path in {
    "manifest": pack / "main.json",
    "supervisor_receipt": pack / "d256-output-l4-01/receipt.json",
    "producer_receipt": pack / "d256-output-l4-01/exports/fit/receipt.json",
}.items():
    files[name] = {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
export_json(
    repo / binding,
    {
        "schema": "bl-residual-metadata-fit-binding/v1",
        "fit_pin": pin,
        "pack": str(pack),
        "phase": phase,
        "phase_sha256": phase_sha,
        "files": files,
    },
)
binding_sha = hashlib.sha256((repo / binding).read_bytes()).hexdigest()
with (repo / log).open("a") as stream:
    stream.write(
        "\n\nOutcome-blind phase publication: F4 D256 metadata-pack fit. Ledger `"
        + phase
        + "`, SHA256 `"
        + phase_sha
        + "`. Receipt/manifest binding `"
        + binding
        + "`, SHA256 `"
        + binding_sha
        + "`, fit pin `"
        + pin
        + "`. Published before metadata staging or the deferred joint R-A look; "
        "no fit/reserve/gameplay outcomes opened by publication.\n"
    )
PY
    git -C "$SOURCE" add "$FIT_PHASE" "$FIT_PHASE.sha256" "$FIT_PHASE.complete" \
      "$FIT_BINDING" "$FIT_BINDING.sha256" "$FIT_BINDING.complete" "$LOG"
    git -C "$SOURCE" diff --cached --check
    git -C "$SOURCE" commit -m 'Publish F4 D256 metadata-pack fit phase ledger'
    git -C "$SOURCE" fetch origin
    git -C "$SOURCE" merge origin/main
    git -C "$SOURCE" push origin HEAD:main
    git -C "$SOURCE" push origin HEAD
    verify_publication
    ;;
  verify-fit-publication)
    verify_publication
    ;;
  stage-d1024)
    # Recheck the live remote before any checkpoint metadata is consumed.
    verify_publication
    "$PY" -m research.bl_residual.f4_d256_package --stage-d1024 "$PACK" \
      --fit-publication "$SOURCE/$FIT_PHASE" --out "$F4_INPUTS"
    ;;
esac
