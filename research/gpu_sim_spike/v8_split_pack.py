"""Prepare immutable V8-SPLIT-01 packets; never allocate a Colab runtime."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, v8_split_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/v8_split_bench.sh"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
RUNTIME_BASE_PIN = "85180521e87ffc3ddc79296bfb09dcd2c89fdc62"
RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_V8_ROUTE_20261023.md"
)
RULING_SHA256 = "5358934afc504cbee94a74622dea0534c831de5e8d7d8f471d5ad4177871a108"
PARITY_RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_GPU_PARITY_STANDARD_20261023.md"
)
PARITY_RULING_SHA256 = "51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b"
PROTOCOL = ROOT / PACKAGE / "V8_SPLIT_INTENT_20261023.md"
DEFAULT_ARTIFACT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-split-20261023-01"
)
PRIMARY_PY = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"


def _verify_file(path: Path) -> str:
    return v8_split_units._verify(path)


def _ruling_inputs() -> list[dict[str, str]]:
    if not RULING.is_file() or sha256_file(RULING) != RULING_SHA256:
        raise ManifestRefused("binding V8 route ruling hash differs")
    if not PROTOCOL.is_file():
        raise ManifestRefused("V8 split protocol is missing")
    if not PARITY_RULING.is_file() or sha256_file(PARITY_RULING) != PARITY_RULING_SHA256:
        raise ManifestRefused("binding GPU parity-standard ruling hash differs")
    return [
        {
            "path": str(RULING),
            "sha256": RULING_SHA256,
            "dest": "inputs/JUDGMENT_V8_ROUTE_20261023.md",
        },
        {
            "path": str(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
            "dest": "inputs/V8_SPLIT_INTENT_20261023.md",
        },
        {
            "path": str(PARITY_RULING),
            "sha256": PARITY_RULING_SHA256,
            "dest": "inputs/JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
        },
    ]


def require_source_pin(commit: str) -> None:
    """Require a clean published frozen split runtime and its qualified transport."""
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("full lowercase commit required")
    if git_head() != commit:
        raise ManifestRefused("pack checkout must be exactly pinned")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ManifestRefused("frozen split source checkout must be clean")
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if "origin/" not in published:
        raise ManifestRefused("frozen split source pin must be published")
    allowed = {
        f"{PACKAGE}/v8_split_bench.py",
        f"{PACKAGE}/v8_split_units.py",
        f"{PACKAGE}/v8_split_analyze.py",
        f"{PACKAGE}/v8_split_pack.py",
        f"{PACKAGE}/v8_split_preflight.py",
        f"{PACKAGE}/v8_split_bench.sh",
        f"{PACKAGE}/v8_split_discrepancies.py",
        f"{PACKAGE}/v8_split_native.py",
        f"{PACKAGE}/test_v8_split_parity.py",
        f"{PACKAGE}/test_v8_split_packet.py",
        f"{PACKAGE}/test_v8_split_contract.py",
        f"{PACKAGE}/V8_SPLIT_INTENT_20261023.md",
        f"{PACKAGE}/V8_SPLIT_FROZEN_COMPONENTS.json",
        f"{PACKAGE}/fixtures/v8_split/device_sim_85180521.py",
    }
    changed = set(
        filter(
            None,
            subprocess.check_output(
                ["git", "diff", "--name-only", RUNTIME_BASE_PIN, commit], cwd=ROOT, text=True
            ).splitlines(),
        )
    )
    if not changed <= allowed:
        raise ManifestRefused(
            "frozen split runtime differs from 85180521 outside its declared new files"
        )
    from tools.ops.qualification import OPS, check

    check(ROOT, OPS / "colab-cli-qualified.json")
    check(ROOT, OPS / "colab-cli-qualified.json", ref=TRANSPORT_PIN)
    _ruling_inputs()


def _ledger_inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    body = json.loads(ledger.read_text())
    paths = [ledger, Path(str(ledger) + ".sha256"), Path(str(ledger) + ".complete")]
    for row in body["units"]:
        payload = ledger.parent / row["path"]
        paths.extend((payload, Path(str(payload) + ".sha256"), Path(str(payload) + ".complete")))
        fixture = json.loads(payload.read_text())["content"].get("recorded_fixture")
        if fixture:
            bound = ledger.parent / fixture["path"]
            paths.extend((bound, Path(str(bound) + ".sha256"), Path(str(bound) + ".complete")))
    return [
        {
            "path": str(p.resolve(strict=True)),
            "sha256": sha256_file(p),
            "dest": f"{prefix}/{p.relative_to(ledger.parent).as_posix()}",
        }
        for p in paths
    ]


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    _verify_file(path)
    value = json.loads(path.read_text())
    receipt = Path(str(value.get("receipt_path", "")))
    _verify_file(receipt)
    body = json.loads(receipt.read_text())
    register = receipt.parent / "discrepancy-register.json"
    _verify_file(register)
    register_body = json.loads(register.read_text())
    expected = list(v8_split_units.allocation(smoke=True))
    for item in (value, body):
        if (
            item.get("status") != "PREFLIGHT"
            or item.get("commit") != commit
            or item.get("device") != "cpu"
            or item.get("completed_units") != expected
            or not isinstance(item.get("source_sha256"), dict)
        ):
            raise ManifestRefused("V8 split needs exact current-pin CPU three-roster preflight")
    elapsed = value.get("entrypoint_run_wall_seconds")
    if (
        value.get("receipt_sha256") != sha256_file(receipt)
        or not isinstance(elapsed, (int, float))
        or elapsed <= 0
    ):
        raise ManifestRefused("V8 split preflight receipt/time binding differs")
    if value["source_sha256"] != body["source_sha256"]:
        raise ManifestRefused("V8 split preflight source binding differs")
    from research.gpu_sim_spike.benchmark import source_hashes

    if body["source_sha256"] != source_hashes():
        raise ManifestRefused("V8 split preflight source closure differs from current pin")
    if register_body.get("schema") != "v8-split-discrepancies/v1":
        raise ManifestRefused("V8 split preflight lacks source-specific discrepancy register")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "receipt_path": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "entrypoint_run_wall_seconds": elapsed,
        "source_sha256": value["source_sha256"],
        "discrepancy_register": {"path": str(register), "sha256": sha256_file(register)},
    }


def _preflight_inputs(preflight: dict[str, Any]) -> list[dict[str, str]]:
    register = preflight["discrepancy_register"]
    return [
        {
            "path": register["path"],
            "sha256": register["sha256"],
            "dest": "inputs/mac-discrepancy-register.json",
        }
    ]


def _canary_binding(receipt: Path, commit: str, preflight: dict[str, Any]) -> dict[str, Any]:
    if not receipt.is_file():
        raise ManifestRefused("same-pin V8 split canary receipt is absent")
    value = json.loads(receipt.read_text())
    manifest_path = receipt.parent / "manifest.json"
    if (
        value.get("status") != "PASS"
        or value.get("commit") != commit
        or value.get("stop_verified") is not True
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != value.get("manifest_sha256")
    ):
        raise ManifestRefused("same-pin native V8 split canary PASS/release receipt required")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("script") != SCRIPT
        or manifest.get("commit") != commit
        or manifest.get("mac_preflight") != preflight
    ):
        raise ManifestRefused("V8 split canary receipt is not from this exact packet")
    producer = receipt.parent / "exports" / "v8-split-bench.json"
    _verify_file(producer)
    output = json.loads(producer.read_text())
    if (
        output.get("status") != "MEASURED_PENDING_MAC_REVIEW"
        or output.get("commit") != commit
        or output.get("namespace") != v8_split_units.namespace(commit, canary=True)
        or output.get("completed_units") != list(v8_split_units.allocation(canary=True))
    ):
        raise ManifestRefused("V8 split canary producer evidence is incomplete")
    native = receipt.parent / "exports" / "native-parity.json"
    _verify_file(native)
    native_body = json.loads(native.read_text())
    if (
        native_body.get("status") != "PASS"
        or native_body.get("native_cuda_executed") is not True
        or native_body.get("commit") != commit
        or native_body.get("source_sha256") != output.get("source_sha256")
    ):
        raise ManifestRefused("V8 split canary native parity gate did not PASS")
    inventory = value.get("exports")
    if not isinstance(inventory, list):
        raise ManifestRefused("V8 split canary terminal export inventory is absent")
    listed = {row.get("path"): row for row in inventory if isinstance(row, dict)}
    if len(listed) != len(inventory):
        raise ManifestRefused("V8 split canary terminal export inventory is malformed")
    actual_payloads = {
        path.relative_to(receipt.parent / "exports").as_posix()
        for path in (receipt.parent / "exports").rglob("*")
        if path.is_file() and not path.name.endswith((".sha256", ".complete"))
    }
    if set(listed) != actual_payloads:
        raise ManifestRefused("V8 split canary terminal export inventory is incomplete")
    for name, row in listed.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ManifestRefused("V8 split canary terminal export path is unsafe")
        path = receipt.parent / "exports" / name
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("sha256") != sha256_file(path)
            or row.get("size") != path.stat().st_size
        ):
            raise ManifestRefused("V8 split canary terminal export inventory differs")
    for name in ("v8-split-bench.json", "native-parity.json", "discrepancy-register.json"):
        if not isinstance(listed.get(name), dict):
            raise ManifestRefused("V8 split canary terminal export inventory differs")
    from research.gpu_sim_spike import v8_split_analyze

    analysis = v8_split_analyze.aggregate([receipt.parent / "exports"], commit, canary=True)
    if analysis.get("verdict") != "PASS_CANARY":
        raise ManifestRefused("independent V8 split canary reducer did not PASS_CANARY")
    elapsed, benchmark = value.get("elapsed_seconds"), output.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not isinstance(benchmark, (int, float)):
        raise ManifestRefused("V8 split canary lacks operational timing for main estimate")
    # One canary E64 sample exists per roster. Main uses nine timing passes per
    # arm and frozen E64/E256/E1024 shape factors 1/4/16: 9*(1+4+16)/3 = 63.
    main_seconds = max(180.0, elapsed - benchmark + benchmark * 63)
    if main_seconds > 20 * 60:
        raise ManifestRefused(
            "canary operational timing cannot support the bounded 20-minute main estimate"
        )
    return {
        "path": str(receipt.resolve()),
        "sha256": sha256_file(receipt),
        "analysis": analysis,
        "main_estimate_minutes": max(3, math.ceil(main_seconds / 60)),
    }


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary_ledger: Path,
    canary: bool,
    preflight: dict[str, Any],
    rate: float = 10.0,
    canary_receipt: Path | None = None,
) -> dict[str, Any]:
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be positive and <=10 u/h")
    v8_split_units.load_ledger(
        ledger, v8_split_units.namespace(commit), v8_split_units.allocation()
    )
    v8_split_units.load_ledger(
        canary_ledger,
        v8_split_units.namespace(commit, canary=True),
        v8_split_units.allocation(canary=True),
    )
    binding = _canary_binding(canary_receipt, commit, preflight) if canary_receipt else None
    timeout, estimate, seconds, cap, reserve = (
        (24, 10, 600, 4.0, 2 / 3)
        if canary
        else (48, binding["main_estimate_minutes"] if binding else 20, 1200, 8.0, 4 / 3)
    )
    ledger_dest = "canary-inputs/verified-ledger.json" if canary else "inputs/verified-ledger.json"
    result: dict[str, Any] = {
        "job_id": f"gpu-v8-split-{'canary' if canary else 'main'}-{commit[:12]}-01",
        "commit": commit,
        "script": SCRIPT,
        "claim": (
            "Prospective V8 geometry attribution only; no labels, release, or "
            "CPU closed-loop claim."
        ),
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/v8_requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "V8_SPLIT_BENCH_DEVICE": "cuda",
            "V8_SPLIT_BENCH_SECONDS": str(seconds),
            "V8_SPLIT_BENCH_LEDGER_PATH": ledger_dest,
            "V8_SPLIT_BENCH_CANARY": "1" if canary else "0",
            "V8_SPLIT_EXPECTED_DISCREPANCIES": "inputs/mac-discrepancy-register.json",
            "MAX_JOBS": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
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
        + _ledger_inputs(ledger, "inputs")
        + _ledger_inputs(canary_ledger, "canary-inputs")
        + _preflight_inputs(preflight)
        + _ruling_inputs(),
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 2 * 1024**2,
        "required_exports": [
            "v8-split-bench.json",
            "discrepancy-register.json",
            "native-parity.json",
        ],
        "resume": {
            "unit_kind": "v8-split-cell",
            "ledger_path": ledger_dest,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": TRANSPORT_PIN,
        "mac_preflight": preflight,
        "ruling": {"path": "inputs/JUDGMENT_V8_ROUTE_20261023.md", "sha256": RULING_SHA256},
        "parity_ruling": {
            "path": "inputs/JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
            "sha256": PARITY_RULING_SHA256,
        },
        "protocol": {"path": "inputs/V8_SPLIT_INTENT_20261023.md", "sha256": sha256_file(PROTOCOL)},
        "v8_split": {
            "runtime_base_pin": RUNTIME_BASE_PIN,
            "effective_envelope": {"wall_minutes": 72, "units": 12},
            "canary": {"wall_minutes": 24, "units": 4, "estimate_minutes_max": 10},
            "main": {"wall_minutes": 48, "units": 8, "estimate_minutes_max": 20},
            "matrix": {
                "order": list(v8_split_units.allocation()),
                "arms": ["R", "G", "GV"],
                "repeats": 3,
                "warmups": 3,
            },
            "admission_guard": (
                "record original T0, deadline, Q2/Q4 remaining allocation, live balance and "
                "liabilities; smallest allowance wins"
            ),
        },
    }
    if canary:
        result["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
        result["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    else:
        result["canary"] = {"required": "independently accepted same-pin V8 split canary receipt"}
        if canary_receipt:
            result["canary"] = {"receipt": binding}
    result = convert_public_checkpoints(result, repo=ROOT)
    assert_upload_budget(result, repo=ROOT)
    return result


def _write(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise ManifestRefused(f"immutable packet differs: {path}")
    if not path.exists():
        write_once(path, text)


def _dispatch(art: Path, commit: str, preflight: Path, main: bool, rate: float) -> str:
    receipt = ' --canary-receipt "$ART/canary-output/receipt.json"' if main else ""
    job = "main" if main else "canary"
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={shlex.quote(str(ROOT))}
ART={shlex.quote(str(art))}
PY={shlex.quote(PRIMARY_PY)}
cd "$REPO"
"$PY" -m research.gpu_sim_spike.v8_split_pack --commit {commit} \\
  --artifact-dir "$ART" --preflight {shlex.quote(str(preflight))} \\
  --rate-units-per-hour {rate}{receipt} --validate
exec "$REPO/tools/ops/dispatch.sh" gpu-v8-split-{job}-{commit[:12]}-01 \\
  "$ART/v8-split-{job}.json" --broker "$REPO" --cwd "$REPO" \\
  --state-dir "$ART/state-{job}" --dispatch-txt "$ART/DISPATCH.txt"
"""


def prepare(
    art: Path,
    commit: str,
    preflight_path: Path,
    rate: float,
    *,
    canary_receipt: Path | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    require_source_pin(commit)
    preflight = preflight_binding(preflight_path, commit)
    art.mkdir(parents=True, exist_ok=True)
    ledgers = (
        (art / "verified-ledger.json", False),
        (art / "canary-ledger/verified-ledger.json", True),
    )
    for path, canary in ledgers:
        if not path.exists():
            v8_split_units.publish_json(
                path,
                v8_split_units.empty_ledger(
                    v8_split_units.namespace(commit, canary=canary),
                    v8_split_units.allocation(canary=canary),
                ),
            )
    common = dict(
        commit=commit,
        ledger=ledgers[0][0],
        canary_ledger=ledgers[1][0],
        preflight=preflight,
        rate=rate,
    )
    canary = manifest(**common, output_dir=art / "canary-output", canary=True)
    main = manifest(
        **common, output_dir=art / "main-output", canary=False, canary_receipt=canary_receipt
    )
    if validate:
        from research.colab_cli.supervisor import validate as supervisor_validate
        from tools.ops.dispatch import require_resume_policy

        require_resume_policy(main)
        supervisor_validate(main if canary_receipt else canary, repo=ROOT)
    for name, value in (
        ("v8-split-canary.json", canary),
        ("v8-split-main.json" if canary_receipt else "v8-split-main-template.json", main),
    ):
        _write(art / name, json.dumps(value, indent=2, sort_keys=True) + "\n")
    for name, is_main in (("dispatch-canary.sh", False), ("dispatch-main-after-canary.sh", True)):
        path = art / name
        _write(path, _dispatch(art, commit, preflight_path, is_main, rate))
        path.chmod(0o755)
    _write(
        art / "DISPATCH.txt",
        f"cd {ROOT}\nnice -n 15 bash {art}/dispatch-canary.sh\n"
        "# Only after independently accepted canary release:\n"
        f"nice -n 15 bash {art}/dispatch-main-after-canary.sh\n",
    )
    return {"artifact_dir": str(art), "commit": commit, "prepared": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.artifact_dir.resolve(),
                args.commit,
                args.preflight,
                args.rate_units_per_hour,
                canary_receipt=args.canary_receipt,
                validate=args.validate,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
