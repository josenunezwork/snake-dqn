"""Prepare the sparse vector61 A100 comparison package without allocation."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, sparse_units
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
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sparse-bench-a100"
)
PRIMARY_PY = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")


def require_source_pin(commit: str) -> None:
    """Use A1-PROF's exact package closure/published-pin admission."""
    a1_prof_pack.require_source_pin(commit)


def _inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    value = json.loads(ledger.read_text())
    paths = [ledger, Path(str(ledger) + ".sha256"), Path(str(ledger) + ".complete")]
    for row in value.get("units", []):
        payload = ledger.parent / row["path"]
        paths.extend((payload, Path(str(payload) + ".sha256"), Path(str(payload) + ".complete")))
    return [
        {
            "path": str(path.resolve(strict=True)),
            "sha256": sha256_file(path),
            "dest": f"{prefix}/{path.relative_to(ledger.parent).as_posix()}",
        }
        for path in paths
    ]


def _preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text())
    elapsed = value.get("entrypoint_run_wall_seconds")
    if (
        value.get("status") != "PREFLIGHT"
        or value.get("commit") != commit
        or value.get("device") != "cpu"
        or value.get("completed_units") != list(sparse_units.allocation(smoke=True))
        or not isinstance(elapsed, (int, float))
        or elapsed <= 0
    ):
        raise ManifestRefused(
            "sparse preflight must be complete current-pin CPU all-roster evidence"
        )
    for suffix in (".sha256", ".complete"):
        if not Path(str(path) + suffix).is_file():
            raise ManifestRefused("sparse preflight sidecars are missing")
    if Path(str(path) + ".sha256").read_text().split()[0] != sha256_file(path):
        raise ManifestRefused("sparse preflight digest differs")
    runtime = Path(str(value.get("receipt_path", "")))
    if not runtime.is_file():
        raise ManifestRefused("sparse preflight runtime receipt is absent")
    from research.gpu_sim_spike.benchmark import source_hashes

    sparse_units._verify(runtime)
    body = json.loads(runtime.read_text())
    if (
        value.get("receipt_sha256") != sha256_file(runtime)
        or body.get("status") != "PREFLIGHT"
        or body.get("commit") != commit
        or body.get("device") != "cpu"
        or body.get("completed_units") != value["completed_units"]
        or body.get("source_sha256") != value.get("source_sha256")
        or body.get("source_sha256") != source_hashes()
    ):
        raise ManifestRefused("sparse preflight runtime/source identity differs")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_sha256": value["source_sha256"],
        "entrypoint_run_wall_seconds": float(elapsed),
        "receipt_path": str(runtime),
        "receipt_sha256": value["receipt_sha256"],
    }


def _a1_v4_canary_binding(path: Path, commit: str) -> dict[str, Any]:
    """Require the exact current-pin v4 profiler canary before sparse spending."""
    receipt = path.resolve(strict=True)
    value = json.loads(receipt.read_text())
    saved = receipt.parent / "manifest.json"
    if not saved.is_file() or value.get("manifest_sha256") != sha256_file(saved):
        raise ManifestRefused("A1-PROF v4 receipt lacks its bound probe manifest")
    probe = json.loads(saved.read_text())
    if (
        probe.get("commit") != commit
        or probe.get("env", {}).get("GPU_SIM_SPIKE_ENTRYPOINT") != "a1-prof"
        or probe.get("a1_prof", {}).get("version") != 4
    ):
        raise ManifestRefused("A1-PROF receipt is not a current same-pin v4 probe")
    # Re-use the supervisor's independent release/export/transport proof rather
    # than inventing a second terminal receipt schema.
    target = json.loads(saved.read_text())
    target["env"].pop("SNAKE_COLAB_CANARY", None)
    target["env"].pop("SNAKE_COLAB_CANARY_PACKAGE", None)
    target["canary"] = {"receipt": {"path": str(receipt), "sha256": sha256_file(receipt)}}
    try:
        from research.colab_cli.canary import check

        check(target, ROOT)
    except Exception as exc:
        raise ManifestRefused("A1-PROF v4 canary lacks verified release/evidence") from exc
    try:
        from research.gpu_sim_spike import a1_prof_analyze

        analysis = a1_prof_analyze.aggregate([receipt.parent / "exports"], commit, canary=True)
        if len(analysis.get("cells", [])) != 9:
            raise ValueError("A1-PROF v4 canary did not export its complete nine-cell matrix")
        for cell in analysis["cells"]:
            if cell.get("metrics", {}).get("attribution_status") != "COMPLETE":
                raise ValueError("A1-PROF v4 diagnostic reconciliation is incomplete")
            if not cell.get("native_launch_event_ticks") or any(
                not tick.get("launches") for tick in cell["native_launch_event_ticks"]
            ):
                raise ValueError("A1-PROF v4 canary lacks native launch timing")
            traces = cell.get("profiler_ticks", [])
            if not traces or any(not trace.get("kernel_inventory") for trace in traces):
                raise ValueError("A1-PROF v4 canary lacks native kernel inventory")
    except Exception as exc:
        raise ManifestRefused("A1-PROF v4 canary lacks complete native producer proof") from exc
    return {"path": str(receipt), "sha256": sha256_file(receipt), "commit": commit, "version": 4}


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary_ledger: Path,
    canary: bool,
    preflight: dict[str, Any],
    canary_receipt: Path | None = None,
    rate: float = 10.0,
) -> dict[str, Any]:
    _require_commit(commit)
    if not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be positive and <=10 units/hour")
    for candidate, mode in ((ledger, False), (canary_ledger, True)):
        sparse_units.load_ledger(
            candidate,
            sparse_units.namespace(commit, canary=mode),
            sparse_units.allocation(canary=mode),
        )
    # Q2 sub-envelope from Astra 01: base 108m/18u including 3u reserve,
    # doubled once to 216m/36u, shared by canary/main/<=2 mechanical attempts.
    timeout, estimate, seconds, cap, reserve = (
        (24, 10, 600, 4.0, 2 / 3) if canary else (192, 80, 4800, 32.0, 16 / 3)
    )
    mode = "canary" if canary else "main"
    ledger_dest = "canary-inputs/verified-ledger.json" if canary else "inputs/verified-ledger.json"
    value: dict[str, Any] = {
        "job_id": f"sparse-bench-a100-{mode}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": (
            "Development-only exact sparse vector61 comparison; "
            "no labels, gate, serving, or release evidence."
        ),
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_ENTRYPOINT": "sparse-bench",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_DEVICE": "cuda",
            "GPU_SIM_SPIKE_SECONDS": str(seconds),
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": str(rate),
            "GPU_SIM_SPIKE_THREADS": "1",
            "GPU_SIM_SPIKE_CPU_WORKERS": "12",
            "SPARSE_BENCH_DEVICE": "cuda",
            "SPARSE_BENCH_SECONDS": str(seconds),
            "SPARSE_BENCH_LEDGER_PATH": ledger_dest,
            "SPARSE_BENCH_CANARY": "1" if canary else "0",
            "GPU_SIM_VECTOR_FLOOD": "dense",
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
        "output_dir": str(output_dir.resolve()),
        "inputs": checkpoint_inputs()
        + _inputs(ledger, "inputs")
        + _inputs(canary_ledger, "canary-inputs"),
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": ["sparse-bench.json"],
        "source_paths": a1_prof_pack._source_paths(),
        "resume": {
            "unit_kind": "sparse-comparison-cell",
            "ledger_path": ledger_dest,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sparse_bench": {
            "requires_a1_prof_version": 4,
            "requires_same_pin_a1_prof_v4_canary_receipt": True,
            "base_envelope": {
                "forecast_minutes": 90,
                "wall_minutes": 108,
                "unit_cap": 18,
                "reserve_units": 3,
            },
            "effective_envelope": {
                "forecast_minutes": 90,
                "wall_minutes": 216,
                "unit_cap": 36,
                "reserve_units": 6,
            },
            "scope": ("canary, main and at most two mechanical completion attempts combined"),
            "matrix": {
                "worlds": [1024],
                "rosters": ["scripted", "frozen", "mixed"],
                "repeats": 3,
                "warmups": 3,
                "matched_backends": ["dense", "sparse"],
            },
            "kill_rule": "STOP_ROUTE on any feature mismatch or any E1024 roster speedup below 2x",
        },
        "mac_preflight": preflight,
    }
    if canary:
        value["env"].update({"SNAKE_COLAB_CANARY": "1", "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE})
        value["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    else:
        if canary_receipt is None:
            value["canary"] = {
                "required": "same-pin sparse canary receipt plus A1-PROF-v4 canary receipt"
            }
        else:
            receipt = canary_receipt.resolve(strict=True)
            value["canary"] = {"receipt": {"path": str(receipt), "sha256": sha256_file(receipt)}}
    value = convert_public_checkpoints(value, repo=ROOT)
    assert_upload_budget(value, repo=ROOT)
    return value


def _dispatch_text(artifact: Path, commit: str) -> str:
    return f"""# Prepared only; refresh assignment/rate evidence and verify the v4 canary.
REPO={ROOT}
ART={artifact}
cd "$REPO"
"$ART/dispatch-canary-after-a1-prof-v4.sh"
bash "$ART/dispatch-main-after-canary.sh"
"""


def _canary_script(artifact: Path, commit: str, preflight: Path, rate: float) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={ROOT}
ART={artifact}
PIN={commit}
PY="$REPO/venv/bin/python"
[[ -x "$PY" ]] || PY={PRIMARY_PY}
export PATH="$(dirname "$PY"):$PATH"
: "${{A1_PROF_V4_RECEIPT:?provide the original broker v4 canary receipt path}}"
RECEIPT="$A1_PROF_V4_RECEIPT"
[[ -f "$RECEIPT" ]] || {{ echo "missing current same-pin A1-PROF v4 canary receipt" >&2; exit 1; }}
cd "$REPO"
"$PY" -m research.gpu_sim_spike.sparse_pack \\
  --commit "$PIN" --artifact-dir "$ART" --a1-prof-v4-receipt "$RECEIPT" \\
  --preflight {shlex.quote(str(preflight.resolve()))} \\
  --rate-units-per-hour {rate} --validate
exec "$REPO/tools/ops/dispatch.sh" sparse-bench-a100-canary-{commit[:8]} \\
  "$ART/sparse-bench-canary.json" --broker "$REPO" --cwd "$REPO" \\
  --state-dir "$ART/state-canary" --dispatch-txt "$ART/DISPATCH.txt"
"""


def _main_script(artifact: Path, commit: str, preflight: Path, rate: float) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={ROOT}
ART={artifact}
PIN={commit}
PY="$REPO/venv/bin/python"
[[ -x "$PY" ]] || PY={PRIMARY_PY}
export PATH="$(dirname "$PY"):$PATH"
SPARSE_RECEIPT="$ART/canary-output/receipt.json"
: "${{A1_PROF_V4_RECEIPT:?provide the original broker v4 canary receipt path}}"
A1_V4_RECEIPT="$A1_PROF_V4_RECEIPT"
[[ -f "$SPARSE_RECEIPT" ]] || {{ echo "missing sparse canary receipt" >&2; exit 1; }}
[[ -f "$A1_V4_RECEIPT" ]] || {{ echo "missing same-pin A1-PROF v4 canary receipt" >&2; exit 1; }}
cd "$REPO"
"$PY" -m research.gpu_sim_spike.sparse_pack \\
  --commit "$PIN" --artifact-dir "$ART" --canary-receipt "$SPARSE_RECEIPT" \\
  --a1-prof-v4-receipt "$A1_V4_RECEIPT" \\
  --preflight {shlex.quote(str(preflight.resolve()))} \\
  --rate-units-per-hour {rate} --validate
exec "$REPO/tools/ops/dispatch.sh" sparse-bench-a100-main-{commit[:8]} \\
  "$ART/sparse-bench-main.json" --broker "$REPO" --cwd "$REPO" \\
  --state-dir "$ART/state-main" --dispatch-txt "$ART/DISPATCH.txt"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--a1-prof-v4-receipt", type=Path)
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    commit = args.commit or git_head()
    require_source_pin(commit)
    art = args.artifact_dir.resolve()
    art.mkdir(parents=True, exist_ok=True)
    preflight_path = (args.preflight or art / "mac-preflight" / "mac-preflight.json").resolve()
    preflight = _preflight_binding(preflight_path, commit)
    ledger = art / "verified-ledger.json"
    canary_ledger = art / "canary-ledger" / "verified-ledger.json"
    if not ledger.exists():
        sparse_units.publish_json(
            ledger,
            sparse_units.empty_ledger(sparse_units.namespace(commit), sparse_units.allocation()),
        )
    if not canary_ledger.exists():
        sparse_units.publish_json(
            canary_ledger,
            sparse_units.empty_ledger(
                sparse_units.namespace(commit, canary=True), sparse_units.allocation(canary=True)
            ),
        )
    a1_v4 = (
        _a1_v4_canary_binding(args.a1_prof_v4_receipt, commit) if args.a1_prof_v4_receipt else None
    )
    canary = manifest(
        commit=commit,
        output_dir=art / "canary-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
        rate=args.rate_units_per_hour,
    )
    if args.canary_receipt and a1_v4 is None:
        raise ManifestRefused("sparse main requires the original A1-PROF v4 canary receipt")
    if a1_v4 is not None:
        canary["a1_prof_v4_canary"] = a1_v4
    main_job = manifest(
        commit=commit,
        output_dir=art / "main-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=args.canary_receipt,
        rate=args.rate_units_per_hour,
    )
    if a1_v4 is not None:
        main_job["a1_prof_v4_canary"] = a1_v4
    if args.validate:
        from research.colab_cli.supervisor import validate
        from tools.ops.dispatch import require_resume_policy

        if args.canary_receipt is None:
            validate(canary, repo=ROOT)
        require_resume_policy(main_job)
        if args.canary_receipt is not None:
            validate(main_job, repo=ROOT)
    for path, data in (
        (
            art / ("sparse-bench-canary.json" if a1_v4 else "sparse-bench-canary-template.json"),
            canary,
        ),
        (
            art
            / (
                "sparse-bench-main.json"
                if args.canary_receipt
                else (
                    "sparse-bench-main-a1-template.json"
                    if a1_v4
                    else "sparse-bench-main-template.json"
                )
            ),
            main_job,
        ),
    ):
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != text:
            raise ManifestRefused(f"immutable pack differs: {path}")
        if not path.exists():
            write_once(path, text)
    handoff = art / "DISPATCH.txt"
    if not handoff.exists():
        write_once(handoff, _dispatch_text(art, commit))
    script = art / "dispatch-main-after-canary.sh"
    text = _main_script(art, commit, preflight_path, args.rate_units_per_hour)
    if not script.exists():
        write_once(script, text)
        script.chmod(0o755)
    elif script.read_text() != text:
        raise ManifestRefused(f"immutable pack differs: {script}")
    canary_script = art / "dispatch-canary-after-a1-prof-v4.sh"
    text = _canary_script(art, commit, preflight_path, args.rate_units_per_hour)
    if not canary_script.exists():
        write_once(canary_script, text)
        canary_script.chmod(0o755)
    elif canary_script.read_text() != text:
        raise ManifestRefused(f"immutable pack differs: {canary_script}")
    print(
        json.dumps({"artifact_dir": str(art), "commit": commit, "prepared": True}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
