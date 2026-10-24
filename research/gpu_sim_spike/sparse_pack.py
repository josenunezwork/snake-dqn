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
    if "exactness_fixture_sha256" in value:
        probe_binding = value.get("exactness_probe", {})
        probe_path = Path(str(probe_binding.get("path", "")))
        sparse_units._verify(probe_path)
        probe = json.loads(probe_path.read_text())
        records = probe.get("records", [])
        if (
            probe_binding.get("sha256") != sha256_file(probe_path)
            or probe.get("status") != "PASS"
            or probe.get("device") != "cpu"
            or probe.get("commit") != commit
            or probe.get("source_sha256") != source_hashes()
            or probe.get("native_cuda_executed") is not False
            or probe.get("fixture_sha256") != value["exactness_fixture_sha256"]
            or len(records) != 4
            or any(not row.get("match") for row in records)
        ):
            raise ManifestRefused("preflight exactness fixture/probe binding differs")
    return {
        **(
            {
                "exactness_fixture_sha256": value["exactness_fixture_sha256"],
                "exactness_probe": value["exactness_probe"],
            }
            if "exactness_fixture_sha256" in value
            else {}
        ),
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_sha256": value["source_sha256"],
        "entrypoint_run_wall_seconds": float(elapsed),
        "receipt_path": str(runtime),
        "receipt_sha256": value["receipt_sha256"],
    }


def _a1_v4_exports_binding(exports: Path) -> dict[str, Any]:
    """Bind recovered diagnostic units, without requiring a successful job receipt.

    These prior-pin units are provenance only, never a sparse resume ledger or
    native canary proof. Verify complete payload triples and their frozen identity;
    do not score outcomes or require large optional Chrome traces.
    """
    from research.gpu_sim_spike import a1_prof_units

    root = exports.resolve(strict=True)
    saved = root.parent / "manifest.json"
    probe = json.loads(saved.read_text())
    commit = probe.get("commit", "")
    _require_commit(commit)
    if (
        probe.get("env", {}).get("GPU_SIM_SPIKE_ENTRYPOINT") != "a1-prof"
        or probe.get("a1_prof", {}).get("version") != 4
    ):
        raise ManifestRefused("recovered exports must belong to an A1-PROF v4 manifest")
    canary = probe.get("env", {}).get("A1_PROF_CANARY") == "1"
    namespace = a1_prof_units.namespace(commit, canary=canary)
    allocation = a1_prof_units.allocation(canary=canary)
    units = []
    for path in sorted((root / a1_prof_units.UNIT_DIRECTORY).glob("*.json")):
        digest = a1_prof_units._verify_sidecars(path)
        value = json.loads(path.read_text())
        if (
            value.get("schema") != a1_prof_units.SCHEMA
            or value.get("namespace") != namespace
            or value.get("allocation") != list(allocation)
            or value.get("unit_id") != path.stem
            or path.stem not in allocation
        ):
            raise ManifestRefused("recovered A1-PROF unit has foreign identity/allocation")
        units.append({"path": str(path), "sha256": digest, "size": path.stat().st_size})
    if not units:
        raise ManifestRefused("no complete recovered A1-PROF v4 units")
    return {
        "role": "optional diagnostic provenance; not canary or sparse resume proof",
        "manifest": {"path": str(saved), "sha256": sha256_file(saved)},
        "commit": commit,
        "version": 4,
        "namespace": namespace,
        "units": units,
    }


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
    exactness_fixture: Path | None = None,
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
            "a1_prof_v4_exports_optional": True,
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
    if exactness_fixture is not None:
        fixture = exactness_fixture.resolve(strict=True)
        sparse_units._verify(fixture)
        if preflight.get("exactness_fixture_sha256") != sha256_file(fixture):
            raise ManifestRefused("exactness fixture must match the Mac preflight")
        value["inputs"].append(
            {
                "path": str(fixture),
                "sha256": sha256_file(fixture),
                "dest": "exactness/failed-a100-state.npz",
            }
        )
        value["env"]["SPARSE_EXACTNESS_FIXTURE"] = "exactness/failed-a100-state.npz"
        value["required_exports"].append("vector-exactness.json")
    value["sparse_bench"]["declared_cpu_repair"] = {
        "baseline_pin": "42c84584d5edef40009ad7278cb912b7f8bd4a9a",
        "scope": "all vector61 arithmetic, sectors, dead rows, pixel float64 trend carry",
        "dense_is_repaired_baseline": True,
        "both_backends_require_cpu_bytes": True,
    }
    if canary:
        value["env"].update({"SNAKE_COLAB_CANARY": "1", "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE})
        value["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    else:
        if canary_receipt is None:
            value["canary"] = {"required": "same-pin sparse canary receipt"}
        else:
            receipt = canary_receipt.resolve(strict=True)
            value["canary"] = {"receipt": {"path": str(receipt), "sha256": sha256_file(receipt)}}
    value = convert_public_checkpoints(value, repo=ROOT)
    assert_upload_budget(value, repo=ROOT)
    return value


def _dispatch_text(artifact: Path, commit: str) -> str:
    return f"""# Prepared only; ops refreshes assignment/rate evidence at dispatch.
REPO={ROOT}
ART={artifact}
cd "$REPO"
"$ART/dispatch-canary.sh"
bash "$ART/dispatch-main-after-canary.sh"
"""


def _canary_script(
    artifact: Path,
    commit: str,
    preflight: Path,
    rate: float,
    exports: Path | None = None,
    exactness_fixture: Path | None = None,
) -> str:
    binding = (
        "" if exports is None else " --a1-prof-v4-exports " + shlex.quote(str(exports.resolve()))
    )
    if exactness_fixture is not None:
        binding += " --exactness-fixture " + shlex.quote(str(exactness_fixture.resolve()))
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={ROOT}
ART={artifact}
PIN={commit}
PY="$REPO/venv/bin/python"
[[ -x "$PY" ]] || PY={PRIMARY_PY}
export PATH="$(dirname "$PY"):$PATH"
cd "$REPO"
"$PY" -m research.gpu_sim_spike.sparse_pack \\
  --commit "$PIN" --artifact-dir "$ART" {binding} \\
  --preflight {shlex.quote(str(preflight.resolve()))} \\
  --rate-units-per-hour {rate} --validate
exec "$REPO/tools/ops/dispatch.sh" sparse-bench-a100-canary-{commit[:8]} \\
  "$ART/sparse-bench-canary.json" --broker "$REPO" --cwd "$REPO" \\
  --state-dir "$ART/state-canary" --dispatch-txt "$ART/DISPATCH.txt"
"""


def _main_script(
    artifact: Path,
    commit: str,
    preflight: Path,
    rate: float,
    exports: Path | None = None,
    exactness_fixture: Path | None = None,
) -> str:
    binding = (
        "" if exports is None else " --a1-prof-v4-exports " + shlex.quote(str(exports.resolve()))
    )
    if exactness_fixture is not None:
        binding += " --exactness-fixture " + shlex.quote(str(exactness_fixture.resolve()))
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={ROOT}
ART={artifact}
PIN={commit}
PY="$REPO/venv/bin/python"
[[ -x "$PY" ]] || PY={PRIMARY_PY}
export PATH="$(dirname "$PY"):$PATH"
SPARSE_RECEIPT="$ART/canary-output/receipt.json"
[[ -f "$SPARSE_RECEIPT" ]] || {{ echo "missing sparse canary receipt" >&2; exit 1; }}
cd "$REPO"
"$PY" -m research.gpu_sim_spike.sparse_pack \\
  --commit "$PIN" --artifact-dir "$ART" --canary-receipt "$SPARSE_RECEIPT" \\
  {binding} \\
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
    parser.add_argument("--a1-prof-v4-exports", type=Path)
    parser.add_argument("--exactness-fixture", type=Path)
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
    a1_v4 = _a1_v4_exports_binding(args.a1_prof_v4_exports) if args.a1_prof_v4_exports else None
    canary = manifest(
        commit=commit,
        output_dir=art / "canary-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
        rate=args.rate_units_per_hour,
        exactness_fixture=args.exactness_fixture,
    )
    if a1_v4 is not None:
        canary["a1_prof_v4_recovered_exports"] = a1_v4
    main_job = manifest(
        commit=commit,
        output_dir=art / "main-output",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=args.canary_receipt,
        rate=args.rate_units_per_hour,
        exactness_fixture=args.exactness_fixture,
    )
    if a1_v4 is not None:
        main_job["a1_prof_v4_recovered_exports"] = a1_v4
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
            art / "sparse-bench-canary.json",
            canary,
        ),
        (
            art
            / (
                "sparse-bench-main.json"
                if args.canary_receipt
                else "sparse-bench-main-template.json"
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
    text = _main_script(
        art,
        commit,
        preflight_path,
        args.rate_units_per_hour,
        args.a1_prof_v4_exports,
        args.exactness_fixture,
    )
    if not script.exists():
        write_once(script, text)
        script.chmod(0o755)
    elif script.read_text() != text:
        raise ManifestRefused(f"immutable pack differs: {script}")
    canary_script = art / "dispatch-canary.sh"
    text = _canary_script(
        art,
        commit,
        preflight_path,
        args.rate_units_per_hour,
        args.a1_prof_v4_exports,
        args.exactness_fixture,
    )
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
