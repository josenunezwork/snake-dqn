"""Prepare immutable, resumable A1-PROF Colab packages; never allocate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_units
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
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100"
)
LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")


def _git_file(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def require_source_pin(commit: str) -> None:
    """Refuse source drift and pins which are not yet published to origin."""
    _require_commit(commit)
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


def _namespace(commit: str, *, smoke: bool, canary: bool) -> str:
    return a1_prof_units.namespace(commit, smoke=smoke, canary=canary)


def _allocation(*, smoke: bool, canary: bool) -> tuple[str, ...]:
    return a1_prof_units.allocation(smoke=smoke, canary=canary)


def empty_ledger(commit: str, *, smoke: bool = False, canary: bool = False) -> dict[str, Any]:
    """Create the first-attempt ledger without inspecting any measurement output."""
    return a1_prof_units.empty_ledger(
        _namespace(commit, smoke=smoke, canary=canary), _allocation(smoke=smoke, canary=canary)
    )


def _ledger_inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
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


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    """Bind a successful, hash-complete real-entrypoint Mac CPU preflight."""
    receipt_path = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestRefused(f"invalid A1-PROF preflight receipt: {receipt_path}") from exc
    if not isinstance(value, dict):
        raise ManifestRefused("A1-PROF preflight receipt must be an object")
    complete = Path(str(receipt_path) + ".complete")
    sidecar = Path(str(receipt_path) + ".sha256")
    if not complete.is_file() or not sidecar.is_file() or complete.read_text() != "complete\n":
        raise ManifestRefused("A1-PROF preflight receipt sidecars are missing or invalid")
    if sidecar.read_text().split()[0] != sha256_file(receipt_path):
        raise ManifestRefused("A1-PROF preflight SHA256 sidecar differs")
    elapsed = value.get("entrypoint_run_wall_seconds")
    source = value.get("source_sha256")
    if (
        value.get("status") != "PREFLIGHT"
        or value.get("commit") != commit
        or value.get("device") != "cpu"
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed <= 0
        or not isinstance(source, dict)
        or not source
        or value.get("units") != ["cell-scripted-e1", "cell-frozen-e1", "cell-mixed-e1"]
    ):
        raise ManifestRefused("preflight must be a successful current-pin CPU three-roster receipt")
    runtime_receipt = Path(str(value.get("receipt_path", "")))
    if not runtime_receipt.is_file() or value.get("receipt_sha256") != sha256_file(runtime_receipt):
        raise ManifestRefused("A1-PROF preflight runtime receipt hash differs")
    from research.gpu_sim_spike.benchmark import source_hashes

    a1_prof_units._verify_sidecars(runtime_receipt)
    runtime = json.loads(runtime_receipt.read_text())
    if (
        runtime.get("status") != "PREFLIGHT"
        or runtime.get("commit") != commit
        or runtime.get("device") != "cpu"
        or runtime.get("completed_units") != value["units"]
        or runtime.get("source_sha256") != source
        or source != source_hashes()
    ):
        raise ManifestRefused("A1-PROF preflight runtime/source identity differs")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "entrypoint_run_wall_seconds": float(elapsed),
        "receipt_path": str(runtime_receipt),
        "receipt_sha256": value["receipt_sha256"],
        "source_sha256": source,
    }


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary_ledger: Path,
    canary: bool,
    canary_receipt: Path | None = None,
    rate: float = 10.0,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one A100-only package with a matched-family canary contract."""
    _require_commit(commit)
    output_dir = output_dir.resolve()
    if not output_dir.is_absolute():
        raise ManifestRefused("output directory must be absolute")
    if ledger.name != "verified-ledger.json" or canary_ledger.name != "verified-ledger.json":
        raise ManifestRefused("portable ledgers must use verified-ledger.json basename")
    if not ledger.is_file() or not canary_ledger.is_file():
        raise ManifestRefused("A1-PROF verified ledgers are absent")
    if not 0 < rate <= 10:
        raise ManifestRefused("rate ceiling must be positive and <=10 units/hour")
    a1_prof_units.load_ledger(
        ledger,
        _namespace(commit, smoke=False, canary=False),
        _allocation(smoke=False, canary=False),
    )
    a1_prof_units.load_ledger(
        canary_ledger,
        _namespace(commit, smoke=False, canary=True),
        _allocation(smoke=False, canary=True),
    )
    timeout, estimate, seconds, unit_cap, reserve = (
        (30, 10, 900, 5.0, 1.0) if canary else (90, 30, 2400, 20.0, 4.0)
    )
    job_mode = "canary" if canary else "main"
    result: dict[str, Any] = {
        "job_id": f"a1-prof-a100-{job_mode}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "A1-PROF development-only stage timings; no labels, gate, or release evidence",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_ENTRYPOINT": "a1-prof",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_DEVICE": "cuda",
            "GPU_SIM_SPIKE_SECONDS": str(seconds),
            "GPU_SIM_SPIKE_THREADS": "1",
            "GPU_SIM_SPIKE_CPU_WORKERS": "12",
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": str(rate),
            "A1_PROF_DEVICE": "cuda",
            "A1_PROF_LEDGER_PATH": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "A1_PROF_SECONDS": str(seconds),
            "A1_PROF_CANARY": "1" if canary else "0",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir),
        # Both payload bundles are uploaded for both modes.  This makes the
        # canary a real transport proof for the exact main input inventory.
        "inputs": checkpoint_inputs()
        + _ledger_inputs(ledger, "inputs")
        + _ledger_inputs(canary_ledger, "canary-inputs"),
        "budget": {
            "unit_cap": unit_cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": ["a1-prof.json"],
        "source_paths": _source_paths(),
        "resume": {
            "unit_kind": "profile-cell",
            "ledger_path": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "a1_prof": {
            "base_envelope": {
                "estimate_minutes": 30,
                "wall_minutes": 45,
                "unit_cap": 10,
                "reserve_units": 2,
            },
            "effective_envelope": {
                "estimate_minutes": 30,
                "wall_minutes": 90,
                "unit_cap": 20,
                "reserve_units": 4,
            },
            "runner_seconds": {"base": 1200, "effective": 2400},
            "matrix": {
                "worlds": [64, 256, 1024],
                "rosters": ["frozen", "mixed", "scripted"],
                "repeats": 3,
                "ticks_per_repeat": 3,
            },
            "canary_matrix": {
                "worlds": [64, 256, 1024],
                "rosters": ["frozen", "mixed", "scripted"],
                "repeats": 1,
                "ticks_per_repeat": 1,
            },
            "outcome_free": True,
            "analysis": "Mac-side only; remote outputs are sealed measurement payloads",
        },
    }
    if preflight is not None:
        result["a1_prof"]["mac_preflight"] = preflight
    if canary:
        result["env"].update({"SNAKE_COLAB_CANARY": "1", "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE})
        result["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    elif canary_receipt is not None:
        receipt = canary_receipt.resolve(strict=True)
        result["canary"] = {"receipt": {"path": str(receipt), "sha256": sha256_file(receipt)}}
    result = convert_public_checkpoints(result, repo=ROOT)
    assert_upload_budget(result, repo=ROOT)
    return result


def dispatch_text(canary_path: Path, commit: str, suffix: str = "") -> str:
    if suffix and (not suffix.isascii() or not suffix.isdigit()):
        raise ManifestRefused("dispatch suffix must contain ASCII digits only")
    attempt = "-" + suffix if suffix else ""
    heading = (
        "# A1-PROF is prepared only; refresh account assignment/rate evidence before dispatch."
    )
    return f"""{heading}
REPO={ROOT}
ART={canary_path.parent}
PIN={commit}
PY="$REPO/venv/bin/python"
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-canary{attempt} "$ART/{canary_path.name}" \\
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-canary" --dispatch-txt "$ART/DISPATCH.txt"
# Require PASS, verified exports, and verified release; then bind that exact receipt.
"$PY" -m research.gpu_sim_spike.a1_prof_pack --commit "$PIN" --artifact-dir "$ART" \\
  --canary-receipt "$ART/canary-output/receipt.json" --validate
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-main{attempt} "$ART/a1-prof-main.json" \\
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-main" --dispatch-txt "$ART/DISPATCH.txt"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--dispatch-suffix", default="")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    commit = args.commit or git_head()
    require_source_pin(commit)
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.ledger is not None:
        raise ManifestRefused("A1-PROF first-job pack refuses a recovery --ledger input")
    preflight = preflight_binding(
        (
            args.preflight.resolve(strict=True)
            if args.preflight is not None
            else artifact_dir / "mac-preflight" / "mac-preflight.json"
        ),
        commit,
    )
    ledger = artifact_dir / "verified-ledger.json"
    if not ledger.exists():
        a1_prof_units.publish_json(ledger, empty_ledger(commit))
    elif a1_prof_units.load_ledger(
        ledger,
        _namespace(commit, smoke=False, canary=False),
        _allocation(smoke=False, canary=False),
    ):
        raise ManifestRefused("A1-PROF first-job pack refuses a nonempty recovery ledger")
    canary_ledger = artifact_dir / "canary-ledger" / "verified-ledger.json"
    if not canary_ledger.exists():
        a1_prof_units.publish_json(canary_ledger, empty_ledger(commit, canary=True))
    canary = manifest(
        commit=commit,
        output_dir=artifact_dir / "canary-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=True,
        rate=args.rate_units_per_hour,
        preflight=preflight,
    )
    main_job = manifest(
        commit=commit,
        output_dir=artifact_dir / "main-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=False,
        canary_receipt=args.canary_receipt,
        rate=args.rate_units_per_hour,
        preflight=preflight,
    )
    if args.validate:
        from research.colab_cli.supervisor import validate
        from tools.ops.dispatch import require_resume_policy

        if args.canary_receipt is None:
            validate(canary, repo=ROOT)
        require_resume_policy(main_job)
        if args.canary_receipt is not None:
            validate(main_job, repo=ROOT)
    canary_path = artifact_dir / "a1-prof-canary.json"
    main_path = artifact_dir / (
        "a1-prof-main.json" if args.canary_receipt else "a1-prof-main-template.json"
    )
    for path, value in ((canary_path, canary), (main_path, main_job)):
        content = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text() != content:
                raise ManifestRefused(f"immutable pack differs: {path}")
        else:
            write_once(path, content)
    handoff = artifact_dir / "DISPATCH.txt"
    if not handoff.exists():
        write_once(handoff, dispatch_text(canary_path, commit, args.dispatch_suffix))
    print(
        json.dumps(
            {"artifact_dir": str(artifact_dir), "commit": commit, "prepared": True}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
