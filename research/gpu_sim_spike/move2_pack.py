"""Prepare, but never dispatch, the fresh MOVE2 A100 benchmark packet."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, move2_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/move2_bench.sh"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_GPU_PARITY_STANDARD_20261023.md"
)
RULING_SHA256 = "51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b"
DEFAULT_ARTIFACT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02"
)
LEDGER_DEST, CANARY_LEDGER_DEST = (
    "inputs/verified-ledger.json",
    "canary-inputs/verified-ledger.json",
)


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")


def require_source_pin(commit: str) -> None:
    """Bind the fresh source checkout and the already-qualified transport tree."""
    _require_commit(commit)
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if "origin/" not in published:
        raise ManifestRefused("source pin must be published before packet preparation")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ManifestRefused("frozen source checkout must be clean")
    for path in sorted((ROOT / PACKAGE).rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".cu", ".cpp", ".h", ".txt"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        pinned = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if pinned != path.read_bytes():
            raise ManifestRefused(f"reviewed source differs from pinned commit: {relative}")
    if not RULING.is_file() or sha256_file(RULING) != RULING_SHA256:
        raise ManifestRefused("binding GPU parity ruling hash differs")
    from tools.ops.qualification import OPS, check

    check(ROOT, OPS / "colab-cli-qualified.json")
    check(ROOT, OPS / "colab-cli-qualified.json", ref=TRANSPORT_PIN)


def _ledger_inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    value = json.loads(ledger.read_text())
    paths = [ledger]
    for row in value["units"] + value["fixtures"]:
        paths.append(ledger.parent / row["path"])
    return [
        {
            "path": str(candidate.resolve(strict=True)),
            "sha256": sha256_file(candidate),
            "dest": prefix + "/" + candidate.relative_to(ledger.parent).as_posix(),
        }
        for path in paths
        for candidate in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    ]


def _ruling_input() -> dict[str, str]:
    return {
        "path": str(RULING),
        "sha256": RULING_SHA256,
        "dest": "inputs/JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
    }


def _validate_preflight(path: Path, commit: str) -> dict[str, Any]:
    move2_units._verify(path)
    value = json.loads(path.read_text())
    if (
        value.get("status") != "PREFLIGHT"
        or value.get("commit") != commit
        or value.get("device") != "cpu"
    ):
        raise ManifestRefused("MOVE2 preflight must be successful on the current source pin")
    for name in (
        "receipt_path",
        "receipt_sha256",
        "cpu_discrepancies_path",
        "cpu_discrepancies_sha256",
    ):
        if not value.get(name):
            raise ManifestRefused("MOVE2 preflight lacks required diagnostic binding")
    receipt = Path(value["receipt_path"])
    diagnostic = Path(value["cpu_discrepancies_path"])
    if (
        move2_units._verify(receipt) != value["receipt_sha256"]
        or move2_units._verify(diagnostic) != value["cpu_discrepancies_sha256"]
    ):
        raise ManifestRefused("MOVE2 preflight artifact hash differs")
    receipt_value = json.loads(receipt.read_text())
    if (
        receipt_value.get("status") != "PASS"
        or receipt_value.get("commit") != commit
        or receipt_value.get("namespace") != move2_units.namespace(commit, smoke=True)
        or receipt_value.get("completed_units") != list(move2_units.allocation(smoke=True))
        or not isinstance(receipt_value.get("source_sha256"), dict)
    ):
        raise ManifestRefused("MOVE2 preflight receipt did not complete the smoke allocation")
    diagnostic_value = json.loads(diagnostic.read_text())
    if (
        diagnostic_value.get("status") != "DIAGNOSTIC"
        or diagnostic_value.get("commit") != commit
        or diagnostic_value.get("source_sha256") != receipt_value["source_sha256"]
    ):
        raise ManifestRefused("MOVE2 CPU discrepancies must remain diagnostic-only")
    register = Path(value.get("discrepancy_register_path", ""))
    if not register.is_file() or not value.get("discrepancy_register_sha256"):
        raise ManifestRefused("MOVE2 preflight lacks the immutable discrepancy register")
    if move2_units._verify(register) != value["discrepancy_register_sha256"]:
        raise ManifestRefused("MOVE2 discrepancy register hash differs")
    register_value = json.loads(register.read_text())
    if (
        register_value.get("commit") != commit
        or register_value.get("ruling_sha256") != RULING_SHA256
        or register_value.get("source_sha256") != receipt_value["source_sha256"]
    ):
        raise ManifestRefused("MOVE2 discrepancy register is not bound to the ruling/current pin")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "receipt": {"path": str(receipt.resolve()), "sha256": value["receipt_sha256"]},
        "cpu_discrepancies": {
            "path": str(diagnostic.resolve()),
            "sha256": value["cpu_discrepancies_sha256"],
        },
        "discrepancy_register": {
            "path": str(register.resolve()),
            "sha256": value["discrepancy_register_sha256"],
        },
    }


def _preflight_inputs(preflight: dict[str, Any]) -> list[dict[str, str]]:
    """Upload frozen CPU diagnostics to both modes; canary must reproduce them exactly."""
    result = []
    for key, dest in (
        ("cpu_discrepancies", "inputs/mac-cpu-discrepancies.json"),
        ("discrepancy_register", "inputs/discrepancy-register.json"),
    ):
        binding = preflight[key]
        result.append({"path": binding["path"], "sha256": binding["sha256"], "dest": dest})
    return result


def native_report_valid(body: dict[str, Any]) -> bool:
    """Accept only the event-probe report emitted after real native CUDA calls."""
    from research.gpu_sim_spike import move2_events

    extension = body.get("extension_sha256")
    if (
        body.get("status") != "PASS"
        or body.get("portability_identity") != "PASS"
        or body.get("fallback") is not False
        or not isinstance(extension, str)
        or not re.fullmatch(r"[0-9a-f]{64}", extension)
    ):
        return False
    observed = body.get("native_calls")
    if not isinstance(observed, dict) or any(
        not isinstance(observed.get(name), int) or observed[name] <= 0
        for name in ("respawn", "collisions")
    ):
        return False
    cases = body.get("cases")
    if not isinstance(cases, list):
        return False
    covered = set()
    for row in cases:
        if not isinstance(row, dict) or row.get("identity", {}).get("match") is not True:
            return False
        calls = row.get("native_calls")
        if not isinstance(calls, dict) or any(
            not isinstance(calls.get(name), int) or calls[name] < 0
            for name in ("respawn", "collisions")
        ):
            return False
        required = "respawn" if row.get("case") in move2_events.CASES[:4] else "collisions"
        if calls[required] != 1:
            return False
        covered.add(row.get("case"))
    return covered == set(move2_events.CASES) and len(cases) == len(move2_events.CASES)


def _validate_canary(receipt: Path, commit: str, preflight: dict[str, Any]) -> dict[str, str]:
    """Check the broker receipt and producer exports; revalidation later runs canary.check."""
    value = json.loads(receipt.read_text())
    if (
        value.get("status") != "PASS"
        or value.get("commit") != commit
        or value.get("stop_verified") is not True
    ):
        raise ManifestRefused("same-pin MOVE2 native canary PASS receipt required")
    saved = receipt.parent / "manifest.json"
    if not saved.is_file() or sha256_file(saved) != value.get("manifest_sha256"):
        raise ManifestRefused("MOVE2 canary broker manifest binding differs")
    probe = json.loads(saved.read_text())
    if (
        probe.get("script") != SCRIPT
        or probe.get("commit") != commit
        or probe.get("mac_preflight") != preflight
    ):
        raise ManifestRefused("MOVE2 canary receipt is not from this exact packet")
    exports = receipt.parent / "exports"
    producer = exports / "move2-bench.json"
    move2_units._verify(producer)
    producer_value = json.loads(producer.read_text())
    if (
        producer_value.get("status") != "PASS"
        or producer_value.get("commit") != commit
        or producer_value.get("namespace") != move2_units.namespace(commit, canary=True)
        or producer_value.get("completed_units") != list(move2_units.allocation(canary=True))
    ):
        raise ManifestRefused("MOVE2 canary producer receipt is incomplete")
    for name in ("native_parity", "cpu_discrepancies"):
        artifact = exports / (
            "native-parity.json" if name == "native_parity" else "cpu-discrepancies.json"
        )
        move2_units._verify(artifact)
        body = json.loads(artifact.read_text())
        expected = "PASS" if name == "native_parity" else "DIAGNOSTIC"
        if body.get("status") != expected or body.get("commit") != commit:
            raise ManifestRefused(f"MOVE2 canary {name} is not admissible")
        if name == "native_parity" and not native_report_valid(body):
            raise ManifestRefused("MOVE2 canary native event coverage is incomplete")
    return {"path": str(receipt.resolve()), "sha256": sha256_file(receipt)}


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary_ledger: Path,
    canary: bool,
    rate: float,
    preflight: dict[str, str],
    canary_receipt: Path | None = None,
) -> dict[str, Any]:
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be positive and <=10 u/h")
    move2_units.load_ledger(ledger, move2_units.namespace(commit), move2_units.allocation())
    move2_units.load_ledger(
        canary_ledger,
        move2_units.namespace(commit, canary=True),
        move2_units.allocation(canary=True),
    )
    timeout, estimate, seconds, cap, reserve = (
        (24, 10, 600, 4.0, 2 / 3) if canary else (192, 80, 4800, 32.0, 16 / 3)
    )
    env = {
        "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/move_requirements.txt",
        "PYTHONUNBUFFERED": "1",
        "GPU_SIM_SPIKE_COMMIT": commit,
        "MOVE2_BENCH_DEVICE": "cuda",
        "MOVE2_BENCH_SECONDS": str(seconds),
        "MOVE2_BENCH_LEDGER_PATH": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
        "MOVE2_BENCH_MODE": "--canary" if canary else "",
        "MOVE2_BENCH_REPEATS": "3",
        "MOVE2_BENCH_WARMUPS": "3",
        "MOVE2_EXPECTED_CPU_DISCREPANCIES_PATH": "inputs/mac-cpu-discrepancies.json",
        "MOVE2_CPU_DISCREPANCIES_SHA256": preflight["cpu_discrepancies"]["sha256"],
        "MOVE2_DISCREPANCY_REGISTER_SHA256": preflight["discrepancy_register"]["sha256"],
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    if canary:
        env.update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
    result: dict[str, Any] = {
        "job_id": f"gpu-move-a100-20261023-02-{'canary' if canary else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": (
            "MOVE2 benchmark-only frozen-device optimization identity; no labels, CPU PASS, "
            "gate, or release evidence"
        ),
        "env": env,
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
        + [_ruling_input()],
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 2 * 1024**2,
        "required_exports": [
            "move2-bench.json",
            "verified-ledger.json",
            "cpu-discrepancies.json",
            "native-parity.json",
        ],
        "resume": {
            "unit_kind": "timing-cell",
            "ledger_path": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": TRANSPORT_PIN,
        "mac_preflight": preflight,
        "ruling": {
            "path": "inputs/JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
            "sha256": RULING_SHA256,
        },
        "move_envelope": {
            "forecast_minutes": 90,
            "base": {"wall_minutes": 108, "units": 18, "reserve_units": 3},
            "effective": {"wall_minutes": 216, "units": 36, "reserve_units": 6},
            "canary": {"wall_minutes": 24, "units": 4},
            "main": {"wall_minutes": 192, "units": 32},
            "no_new_allowance": True,
        },
    }
    result["canary"] = (
        {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
        if canary
        else {"required": "same-pin native MOVE2 canary receipt"}
    )
    if not canary and canary_receipt:
        result["canary"] = {"receipt": _validate_canary(canary_receipt, commit, preflight)}
    return convert_public_checkpoints(result, repo=ROOT)


def _write(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise ManifestRefused(f"immutable packet differs: {path}")
    if not path.exists():
        write_once(path, text)


def _dispatch(art: Path, commit: str, preflight: Path, *, main: bool, rate: float) -> str:
    receipt = ' --canary-receipt "$ART/canary-output/receipt.json"' if main else ""
    job = "main" if main else "canary"
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={shlex.quote(str(ROOT))}
ART={shlex.quote(str(art))}
PY={shlex.quote(str(ROOT / 'venv/bin/python'))}
export PATH="$(dirname "$PY"):$PATH"
cd "$REPO"
"$PY" -m research.gpu_sim_spike.move2_pack --commit {commit} --artifact-dir "$ART" \\
  --preflight {shlex.quote(str(preflight))} --rate-units-per-hour {rate}{receipt} --validate
exec "$REPO/tools/ops/dispatch.sh" gpu-move2-{job}-{commit[:8]} "$ART/move2-{job}.json" \\
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-{job}" --dispatch-txt "$ART/DISPATCH.txt"
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
    if git_head() != commit:
        raise ManifestRefused("pack checkout must be exactly pinned")
    preflight = _validate_preflight(preflight_path, commit)
    art.mkdir(parents=True, exist_ok=True)
    ledgers = (
        (art / "verified-ledger.json", False),
        (art / "canary-ledger/verified-ledger.json", True),
    )
    for path, is_canary in ledgers:
        if not path.exists():
            move2_units.publish_json(
                path,
                move2_units.empty_ledger(
                    move2_units.namespace(commit, canary=is_canary),
                    move2_units.allocation(canary=is_canary),
                ),
            )
    main_ledger, canary_ledger = ledgers[0][0], ledgers[1][0]
    canary = manifest(
        commit=commit,
        output_dir=art / "canary-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=True,
        rate=rate,
        preflight=preflight,
    )
    main = manifest(
        commit=commit,
        output_dir=art / "main-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        rate=rate,
        preflight=preflight,
        canary_receipt=canary_receipt,
    )
    assert_upload_budget(canary, repo=ROOT)
    assert_upload_budget(main, repo=ROOT)
    if canary_receipt is None:
        main["scientific_admission"] = False
        main["admission_reason"] = "TEMPLATE_ONLY: current-pin native A100 PASS canary is required"
    if validate:
        from research.colab_cli.supervisor import validate as validate_manifest

        validate_manifest(main if canary_receipt else canary, repo=ROOT)
    _write(art / "move2-canary.json", json.dumps(canary, indent=2, sort_keys=True) + "\n")
    _write(
        art / ("move2-main.json" if canary_receipt else "move2-main-template.json"),
        json.dumps(main, indent=2, sort_keys=True) + "\n",
    )
    for name, is_main in (("dispatch-canary.sh", False), ("dispatch-main-after-canary.sh", True)):
        path = art / name
        _write(path, _dispatch(art, commit, preflight_path, main=is_main, rate=rate))
        path.chmod(0o755)
    _write(
        art / "DISPATCH.txt",
        f"cd {ROOT}\n"
        f"nice -n 15 bash {art}/dispatch-canary.sh\n"
        "# Only after the immutable same-pin native PASS canary receipt:\n"
        f"nice -n 15 bash {art}/dispatch-main-after-canary.sh\n",
    )
    return {
        "prepared": True,
        "artifact_dir": str(art),
        "commit": commit,
        "main_admitted": canary_receipt is not None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit")
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--rate-units-per-hour", type=float, default=10.0)
    p.add_argument("--canary-receipt", type=Path)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            prepare(
                a.artifact_dir.resolve(),
                a.commit or git_head(),
                a.preflight.resolve(),
                a.rate_units_per_hour,
                canary_receipt=a.canary_receipt,
                validate=a.validate,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
