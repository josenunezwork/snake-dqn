"""Prepare immutable, cell-resumable A1 memory-repair packs; never allocate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/colab.sh"
DEFAULT_ARTIFACT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-a1-memfix-a10040"
)
LEDGER_DEST = "inputs/verified-ledger.json"


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")


def _git_file(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def require_source_pin(commit: str) -> None:
    """Refuse an unpublished/stale source pin or an unqualified transport tree."""
    _require_commit(commit)
    # Bind the complete engine closure, including allocation fixes and resume.
    for path in sorted((ROOT / PACKAGE).rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".cu", ".cpp", ".txt"):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if _git_file(commit, relative) != path.read_bytes():
            raise ManifestRefused(f"reviewed source differs from pinned commit: {relative}")
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if "origin/" not in published:
        raise ManifestRefused("source pin must be published before pack preparation")
    from tools.ops.qualification import OPS, check

    check(ROOT, OPS / "colab-cli-qualified.json")
    check(ROOT, OPS / "colab-cli-qualified.json", ref=commit)


def empty_ledger(commit: str, hardware: str = "A100-40GB", smoke: bool = False) -> dict[str, Any]:
    """An empty, hardware-specific ledger for the frozen cell allocation."""
    return a1_units.empty_ledger(
        a1_units.namespace(commit, hardware, smoke), a1_units.allocation(smoke)
    )


def _ledger_inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    """Upload only hash-verified ledger units, using portable relative paths."""
    value = json.loads(ledger.read_text())
    paths = [ledger, Path(str(ledger) + ".sha256"), Path(str(ledger) + ".complete")]
    for row in value["units"]:
        payload = ledger.parent / row["path"]
        paths.extend((payload, Path(str(payload) + ".sha256"), Path(str(payload) + ".complete")))
    return [
        {
            "path": str(path.resolve(strict=True)),
            "sha256": sha256_file(path),
            "dest": prefix + "/" + path.relative_to(ledger.parent).as_posix(),
        }
        for path in paths
    ]


def _source_paths() -> list[str]:
    return [
        "src",
        "web/backend",
        "configs",
        PACKAGE,
        "research/colab_cli",
        "research/bl_upload",
        "research/t34_exit_b8",
        "research/speed_s0_baseline",
        "research/t34_shadow",
        "research/t33_batched_teacher",
        "research/phase0_search_20261019",
        "research/redesign_scope_20261007",
        "research/t3_rollout_labels_20261015",
        "research/t1_commitment_20261015",
        "research/apex_safety_20260926",
        "research/b8_death_census_20261014",
        "research/fixed_search.py",
        "requirements.txt",
        "pyproject.toml",
    ]


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary: bool,
    canary_receipt: Path | None = None,
    hardware: str = "A100-40GB",
    rate: float = 10.0,
    smoke_ledger: Path,
    attempt: int = 1,
    remaining_minutes: float | None = None,
    remaining_units: float | None = None,
) -> dict[str, Any]:
    """Construct one explicitly priced hardware family; never pool variants."""
    _require_commit(commit)
    output_dir = output_dir.resolve()
    if not output_dir.is_absolute():
        raise ManifestRefused("output directory must be absolute")
    if ledger.name != "verified-ledger.json" or smoke_ledger.name != "verified-ledger.json":
        raise ManifestRefused("portable ledgers must use verified-ledger.json basename")
    if not ledger.is_file():
        raise ManifestRefused("A1 verified ledger is absent")
    if hardware not in ("A100-40GB", "A100-80GB", "H100-80GB"):
        raise ManifestRefused("unsupported hardware")
    if not 0 < rate <= 10:
        raise ManifestRefused(
            "rate ceiling must be positive and <=10 within frozen 20u/2h envelope"
        )
    if type(attempt) is not int or not 1 <= attempt <= 3:
        raise ManifestRefused("one original plus at most two completion attempts")
    if attempt > 1 and (remaining_minutes is None or remaining_units is None):
        raise ManifestRefused("completion requires explicit remaining series minutes and units")
    minutes = 120.0 if remaining_minutes is None else remaining_minutes
    units = 20.0 if remaining_units is None else remaining_units
    if not 10 <= minutes <= 120 or not 0 < units <= 20:
        raise ManifestRefused("remaining envelope must be positive and within original 120min/20u")
    if minutes * rate / 60 > units:
        raise ManifestRefused("remaining whole-session timeout priced above remaining unit cap")
    a1_units.load_ledger(
        ledger, a1_units.namespace(commit, hardware, False), a1_units.allocation(False)
    )
    a1_units.load_ledger(
        smoke_ledger, a1_units.namespace(commit, hardware, True), a1_units.allocation(True)
    )
    gpu = hardware.split("-")[0]
    tag = hardware.lower().replace("-", "").replace("gb", "")
    low, high = (39000, 42000) if hardware == "A100-40GB" else (78000, 84000)
    result: dict[str, Any] = {
        "job_id": f"bl-a1-memfix-{tag}-a{attempt}-{'canary' if canary else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "A1 throughput ruler; development only; no registered or serving verdict",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_ENTRYPOINT": "a1-ruler",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_DEVICE": "cuda",
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": str(rate),
            "GPU_SIM_SPIKE_SECONDS": str(int(minutes * 60 - 450)) if not canary else "600",
            "A1_RULER_DEVICE": "cuda",
            "A1_RULER_LEDGER_PATH": "canary-inputs/verified-ledger.json" if canary else LEDGER_DEST,
            "A1_RULER_WARMUP_TICKS": "3",
            "A1_RULER_SECONDS": str(int(minutes * 60 - 450)) if not canary else "600",
            "A1_RULER_EXPECTED_GPU_FAMILY": hardware,
            "A1_RULER_MIN_USABLE_MEMORY_MIB": str(low),
            "A1_RULER_MAX_USABLE_MEMORY_MIB": str(high),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": gpu,
            "allowed_gpus": [gpu],
            "min_vcpu": 12,
            "est_minutes": 9 if canary else minutes / 2,
        },
        "timeout_minutes": 18 if canary else minutes,
        "output_dir": str(output_dir),
        "inputs": checkpoint_inputs()
        + _ledger_inputs(ledger, "inputs")
        + _ledger_inputs(smoke_ledger, "canary-inputs"),
        "budget": {
            # The authoritative exception binds a 20-u two-hour execution cap;
            # its separate 4-u reserve is held for accounting/release variance.
            "unit_cap": 3.0 if canary else units,
            "rate_units_per_hour": rate,
            "reserve_units": 0.6 if canary else 4.0,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": ["a1-ruler.json", "a0-native-oracle.json"],
        "source_paths": _source_paths(),
        "a1": {
            "canary_base_envelope": {"minutes": 9, "units": 1.5, "reserve_units": 0.3},
            "canary_effective_envelope": {"minutes": 18, "units": 3, "reserve_units": 0.6},
            "base_envelope": {"hours": 1, "units": 10, "reserve_units": 2},
            "effective_envelope": {"hours": 2, "units": 20, "reserve_units": 4},
            "attempt": attempt,
            "attempt_effective_envelope": {"minutes": minutes, "units": units},
            "hardware": {
                "family": hardware,
                "min_usable_memory_mib": low,
                "max_usable_memory_mib": high,
            },
            "shapes": [6, 256, 1024],
            "mixes": ["scripted", "frozen", "mixed"],
            "repeats": 3,
            "ticks_per_repeat": 8,
            "unrepaired_e6_control": True,
            "instrumented_pass": "separate",
        },
    }
    if canary:
        result["env"]["A1_RULER_SMOKE"] = "1"
        result["canary"] = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": SCRIPT,
        }
        result["env"]["SNAKE_COLAB_CANARY"] = "1"
        result["env"]["SNAKE_COLAB_CANARY_PACKAGE"] = PACKAGE
    else:
        if canary_receipt is not None:
            receipt = canary_receipt.resolve(strict=True)
            result["canary"] = {"receipt": {"path": str(receipt), "sha256": sha256_file(receipt)}}
    result["resume"] = {
        "unit_kind": "ruler-cell",
        "ledger_path": "canary-inputs/verified-ledger.json" if canary else LEDGER_DEST,
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }
    result["a1"]["attempt_policy"] = (
        "Mechanical OOM/transport/time-cap only, verified stop and missing units; "
        "one original + at most two manually dispatched completion attempts, fresh outputs; "
        "all attempts share remaining original series wall/unit envelope. No automatic retry."
    )
    result["a1"]["aggregate"] = "Mac-only a1_analyze; remote summary sealed diagnostics"
    result = convert_public_checkpoints(result, repo=ROOT)
    assert_upload_budget(result, repo=ROOT)
    return result


def dispatch_text(
    canary_path: Path,
    main_path: Path,
    commit: str,
    hardware: str,
    attempt: int,
    ledger: Path,
    minutes: float,
    units: float,
    rate: float,
) -> str:
    tag = hardware.lower().replace("-", "").replace("gb", "")
    return rf"""# A1 memory repair — prepared only, Claude dispatches
# Populate budget.other_assignments from fresh reviewed positive endpoint/rate inventory.
# Ceiling 10u/h is a refusal threshold, not a claim about live provider price.
# Same source pin, hardware and inputs required; different GPUs have separate namespaces.
REPO=/Users/josenunez/Projects/ml/snake-dqn-a0
ART={canary_path.parent}
PIN={commit}
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" bl-a1-memfix-{tag}-a{attempt}-canary "$ART/a1-canary.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-canary" --dispatch-txt "$ART/DISPATCH.txt"
# Wait for PASS, verified exports and backend release. Bind its immutable receipt:
"$PY" -m research.gpu_sim_spike.make_a1_pack --commit "$PIN" --artifact-dir "$ART" \
  --hardware {hardware} --attempt {attempt} --ledger "{ledger}" \
  --remaining-minutes {minutes} --remaining-units {units} --rate-units-per-hour {rate} \
  --validate --canary-receipt "$ART/canary-output/receipt.json"
"$REPO/tools/ops/dispatch.sh" bl-a1-memfix-{tag}-a{attempt}-main "$ART/a1-main.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-main" --dispatch-txt "$ART/DISPATCH.txt"
# Mac-only, once ALL frozen units are present (never on partial contents):
"$PY" -m research.gpu_sim_spike.a1_analyze --exports "$ART/main-output/exports" \
  --commit "$PIN" --hardware {hardware} --output "$ART/mac-decision.json"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument(
        "--ledger", type=Path, help="Mac-verified completion ledger; never outcomes"
    )
    parser.add_argument(
        "--hardware", choices=("A100-40GB", "A100-80GB", "H100-80GB"), default="A100-40GB"
    )
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--emit-80gb-variants", action="store_true")
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--remaining-minutes", type=float)
    parser.add_argument("--remaining-units", type=float)
    args = parser.parse_args()
    commit = args.commit or git_head()
    require_source_pin(commit)
    variants = (
        (args.hardware, "A100-80GB", "H100-80GB") if args.emit_80gb_variants else (args.hardware,)
    )
    for hardware in dict.fromkeys(variants):
        artifact_dir = args.artifact_dir.resolve()
        if hardware != args.hardware:
            artifact_dir /= hardware.lower()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ledger = artifact_dir / "verified-ledger.json"
        if args.ledger and hardware == args.hardware:
            ledger = args.ledger.resolve(strict=True)
        if not ledger.exists():
            a1_units.publish_json(ledger, empty_ledger(commit, hardware))
        canary_ledger = artifact_dir / "canary-ledger" / "verified-ledger.json"
        if not canary_ledger.exists():
            a1_units.publish_json(canary_ledger, empty_ledger(commit, hardware, True))
        canary = manifest(
            commit=commit,
            output_dir=artifact_dir / "canary-output",
            ledger=ledger,
            smoke_ledger=canary_ledger,
            attempt=args.attempt,
            remaining_minutes=args.remaining_minutes,
            remaining_units=args.remaining_units,
            canary=True,
            hardware=hardware,
            rate=args.rate_units_per_hour,
        )
        canary_receipt = args.canary_receipt if hardware == args.hardware else None
        main_job = manifest(
            commit=commit,
            output_dir=artifact_dir / "main-output",
            ledger=ledger,
            smoke_ledger=canary_ledger,
            attempt=args.attempt,
            remaining_minutes=args.remaining_minutes,
            remaining_units=args.remaining_units,
            canary=False,
            canary_receipt=canary_receipt,
            hardware=hardware,
            rate=args.rate_units_per_hour,
        )
        if args.validate:
            from research.colab_cli.supervisor import validate
            from tools.ops.dispatch import require_resume_policy

            if not canary_receipt:
                validate(canary, repo=ROOT)
            require_resume_policy(main_job)
            if canary_receipt:
                validate(main_job, repo=ROOT)
        canary_path = artifact_dir / "a1-canary.json"
        main_path = artifact_dir / ("a1-main.json" if canary_receipt else "a1-main-template.json")
        for path, value in ((canary_path, canary), (main_path, main_job)):
            content = json.dumps(value, indent=2, sort_keys=True) + "\n"
            if path.exists():
                if path.read_text() != content:
                    raise ManifestRefused(f"immutable pack differs: {path}")
            else:
                write_once(path, content)
        handoff = artifact_dir / "DISPATCH.txt"
        if not handoff.exists():
            write_once(
                handoff,
                dispatch_text(
                    canary_path,
                    main_path,
                    commit,
                    hardware,
                    args.attempt,
                    ledger,
                    main_job["timeout_minutes"],
                    main_job["budget"]["unit_cap"],
                    args.rate_units_per_hour,
                ),
            )
        print(
            json.dumps(
                {
                    "artifact_dir": str(artifact_dir),
                    "commit": commit,
                    "hardware": hardware,
                    "prepared": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
