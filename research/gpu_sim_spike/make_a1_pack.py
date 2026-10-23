"""Build the A1 A100-40 ruler transport pack without allocating Colab.

PLAN_A100 §3 explicitly names A1 an ``acceptance`` exemption.  Its effective
two-hour cap is wider than the generic ops exemption ceiling, so dispatch stays
blocked until the authoritative-plan-specific ops exception is independently
reviewed and installed; this pack never disguises that tension as resumability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
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
DEFAULT_ARTIFACT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-a1")
LEDGER_DEST = "inputs/verified-ledger.json"


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")


def _git_file(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def require_source_pin(commit: str) -> None:
    """Refuse an unpublished/stale source pin or an unqualified transport tree."""
    _require_commit(commit)
    for name in (
        "colab.sh",
        "a100_ruler.sh",
        "a100_ruler.py",
        "device_sim.py",
        "device_policy.py",
        "device_sim_frozen.py",
        "stage_timing.py",
        "event_cuda.py",
        "event_cuda.cpp",
        "event_cuda.cu",
    ):
        relative = f"{PACKAGE}/{name}"
        if _git_file(commit, relative) != (ROOT / relative).read_bytes():
            raise ManifestRefused(f"reviewed source differs from pinned commit: {relative}")
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if "origin/" not in published:
        raise ManifestRefused("source pin must be published before pack preparation")
    from tools.ops.qualification import OPS, check

    check(ROOT, OPS / "colab-cli-qualified.json")
    check(ROOT, OPS / "colab-cli-qualified.json", ref=commit)


def empty_ledger(commit: str) -> dict[str, Any]:
    """The initial resume ledger contains no outcomes and binds A1's cell namespace."""
    return {
        "schema": "gpu-sim-spike-a1-verified-ledger/v1",
        "namespace": f"a1-ruler-{commit}",
        "allocation_index": [
            {"mix": mix, "worlds": worlds, "repeat": repeat}
            for mix in ("long-scripted", "long-frozen", "long-mixed")
            for worlds in (6, 256, 1024)
            for repeat in range(3)
        ],
        "verified_units": [],
        "published_last": "verified_units",
    }


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
    *, commit: str, output_dir: Path, ledger: Path, canary: bool, canary_receipt: Path | None = None
) -> dict[str, Any]:
    """Construct exactly one A100-SXM4-40GB family manifest."""
    _require_commit(commit)
    output_dir = output_dir.resolve()
    if not output_dir.is_absolute():
        raise ManifestRefused("output directory must be absolute")
    if not ledger.is_file():
        raise ManifestRefused("A1 verified ledger is absent")
    a100_rate = 10.0  # conservative 2x-capped A100-40 planning ceiling
    result: dict[str, Any] = {
        "job_id": f"bl-a1-{'canary' if canary else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "A1 throughput ruler; development only; no registered or serving verdict",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_ENTRYPOINT": "a1-ruler",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_DEVICE": "cuda",
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": "10.0",
            "GPU_SIM_SPIKE_SECONDS": "6750" if not canary else "600",
            "A1_RULER_DEVICE": "cuda",
            "A1_RULER_WARMUP_TICKS": "3",
            "A1_RULER_SECONDS": "6750" if not canary else "600",
            "A1_RULER_EXPECTED_GPU_FAMILY": "A100-40GB",
            "A1_RULER_MIN_USABLE_MEMORY_MIB": "39000",
            "A1_RULER_MAX_USABLE_MEMORY_MIB": "42000",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": 9 if canary else 60,
        },
        "timeout_minutes": 18 if canary else 120,
        "output_dir": str(output_dir),
        "inputs": checkpoint_inputs()
        + [{"path": str(ledger.resolve()), "sha256": sha256_file(ledger), "dest": LEDGER_DEST}],
        "budget": {
            # The authoritative exception binds a 20-u two-hour execution cap;
            # its separate 4-u reserve is held for accounting/release variance.
            "unit_cap": 3.0 if canary else 20.0,
            "rate_units_per_hour": a100_rate,
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
            "hardware": {
                "family": "A100-40GB",
                "min_usable_memory_mib": 39000,
                "max_usable_memory_mib": 42000,
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
        result["resume_exemption"] = {
            "kind": "short-canary",
            "reason": "same-pin A100-40 transport and real-entrypoint proof; no ruler conclusion",
        }
        result["canary"] = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": SCRIPT,
        }
        result["env"]["SNAKE_COLAB_CANARY"] = "1"
        result["env"]["SNAKE_COLAB_CANARY_PACKAGE"] = PACKAGE
    else:
        result["resume_exemption"] = {
            "kind": "acceptance",
            "authority_sha256": "1c87898be554d8e66788da6f0a09ee7b842a514e530c1d0e1c624bcf58c3962c",
            "reason": (
                "PLAN_A100.md sha256-bound A1 full-tick ruler; authoritative effective "
                "120-minute/20-unit acceptance envelope"
            ),
        }
        if canary_receipt is not None:
            receipt = canary_receipt.resolve(strict=True)
            result["canary"] = {"receipt": {"path": str(receipt), "sha256": sha256_file(receipt)}}
    result = convert_public_checkpoints(result, repo=ROOT)
    assert_upload_budget(result, repo=ROOT)
    return result


def dispatch_text(canary_path: Path, main_path: Path, commit: str) -> str:
    return f"""# A1 repaired full-tick ruler — prepared only

Claude dispatches exclusively through tools/ops/dispatch.sh after A0's independent
oracle receipt is PASS.  The supervisor accepts generic A100, but the real A1
entrypoint refuses every non-40GB A100, accepting only 39,000–42,000 usable MiB.
Existing
qualified transport supports A100; no transport requalification is needed.

Canary: {canary_path}
Main: {main_path}

The canary is capped at 18 min / 3 u (within the 30-min ceiling) and proves
same-pin real transport. The main
manifest deliberately declares the PLAN's `resume_exemption: acceptance` at 120
minutes / 20 u (first cap wins).  The installed narrow PLAN_A100-SHA keyed ops
exception is required; ordinary acceptance exemptions remain bounded at 60 minutes.

Before either dispatch, Claude must populate budget.other_assignments from a fresh
reviewed endpoint/rate inventory, validate the exact manifest, and confirm the
qualified broker matches its registry.  A kernel-RPC stall makes this attempt
INCOMPLETE; one outcome-blind mechanical retry is allowed, then Fable judge.

REPO=/Users/josenunez/Projects/ml/snake-dqn-a0
ART={canary_path.parent}
PIN={commit}
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" bl-a1-canary "$ART/a1-canary.json" \\
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-canary" --dispatch-txt "$ART/DISPATCH.txt"
# Wait for terminal PASS, exports and release, then generate/bind main receipt.
"$REPO/../snake-dqn/venv/bin/python" -m research.gpu_sim_spike.make_a1_pack \\
  --commit "$PIN" --artifact-dir "$ART" --validate \\
  --canary-receipt "$ART/canary-output/receipt.json"
"$REPO/tools/ops/dispatch.sh" bl-a1-main "$ART/a1-main.json" \\
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-main" --dispatch-txt "$ART/DISPATCH.txt"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    commit = args.commit or git_head()
    require_source_pin(commit)
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ledger = artifact_dir / "verified-ledger.json"
    if not ledger.exists():
        write_once(ledger, json.dumps(empty_ledger(commit), indent=2, sort_keys=True) + "\n")
    canary = manifest(
        commit=commit, output_dir=artifact_dir / "canary-output", ledger=ledger, canary=True
    )
    main_job = manifest(
        commit=commit,
        output_dir=artifact_dir / "main-output",
        ledger=ledger,
        canary=False,
        canary_receipt=args.canary_receipt,
    )
    if args.validate:
        from research.colab_cli.supervisor import validate

        if not args.canary_receipt:
            validate(canary, repo=ROOT)
        from tools.ops.dispatch import require_resume_policy

        require_resume_policy(main_job)
        if args.canary_receipt:
            validate(main_job, repo=ROOT)
    canary_path = artifact_dir / "a1-canary.json"
    main_path = artifact_dir / ("a1-main.json" if args.canary_receipt else "a1-main-template.json")
    if not canary_path.exists():
        write_once(canary_path, json.dumps(canary, indent=2, sort_keys=True) + "\n")
    write_once(main_path, json.dumps(main_job, indent=2, sort_keys=True) + "\n")
    handoff = artifact_dir / "DISPATCH.txt"
    if not handoff.exists():
        write_once(handoff, dispatch_text(canary_path, main_path, commit))
    print(
        json.dumps(
            {"artifact_dir": str(artifact_dir), "commit": commit, "prepared": True}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
