"""Prepare immutable qualified A100 V8 packets; never allocate a runtime."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, v8_units
from research.gpu_sim_spike.benchmark import source_hashes
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/v8_bench.sh"
DEFAULT_ARTIFACT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-sprint-20261023-02"
)
PRIMARY_PY = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"


def require_source_pin(commit: str) -> None:
    a1_prof_pack.require_source_pin(commit)


def _inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    body = json.loads(ledger.read_text())
    paths = [ledger, Path(str(ledger) + ".sha256"), Path(str(ledger) + ".complete")]
    for row in body["units"]:
        payload = ledger.parent / row["path"]
        paths += [payload, Path(str(payload) + ".sha256"), Path(str(payload) + ".complete")]
        fixture = json.loads(payload.read_text())["content"].get("recorded_fixture")
        if fixture:
            target = ledger.parent / fixture["path"]
            paths += [target, Path(str(target) + ".sha256"), Path(str(target) + ".complete")]
    return [
        {
            "path": str(p.resolve(strict=True)),
            "sha256": sha256_file(p),
            "dest": f"{prefix}/{p.relative_to(ledger.parent).as_posix()}",
        }
        for p in paths
    ]


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    v8_units._verify(path)
    value = json.loads(path.read_text())
    runtime = Path(str(value.get("receipt_path", "")))
    v8_units._verify(runtime)
    body = json.loads(runtime.read_text())
    elapsed = value.get("entrypoint_run_wall_seconds")
    for receipt in (value, body):
        if (
            receipt.get("status") != "PREFLIGHT"
            or receipt.get("commit") != commit
            or receipt.get("device") != "cpu"
            or receipt.get("completed_units") != list(v8_units.allocation(smoke=True))
            or receipt.get("source_sha256") != source_hashes()
        ):
            raise ManifestRefused("V8 needs exact current-pin three-roster CPU preflight")
    if (
        value.get("receipt_sha256") != sha256_file(runtime)
        or not isinstance(elapsed, (int, float))
        or elapsed <= 0
    ):
        raise ManifestRefused("V8 preflight receipt/time binding differs")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt_path": str(runtime),
        "receipt_sha256": sha256_file(runtime),
        "entrypoint_run_wall_seconds": elapsed,
        "source_sha256": value["source_sha256"],
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
) -> dict[str, Any]:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ManifestRefused("full lowercase commit required")
    if not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be positive and <=10 u/h")
    for led, mode in ((ledger, False), (canary_ledger, True)):
        v8_units.load_ledger(
            led, v8_units.namespace(commit, canary=mode), v8_units.allocation(canary=mode)
        )
    timeout, estimate, seconds, cap, reserve = (
        (24, 10, 600, 4.0, 2 / 3) if canary else (192, 80, 4800, 32.0, 16 / 3)
    )
    destination = "canary-inputs/verified-ledger.json" if canary else "inputs/verified-ledger.json"
    value = {
        "job_id": f"gpu-v8-indexed-{'canary' if canary else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "Development exactness/timing only; no closed-loop CPU qualification or labels.",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/v8_requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "V8_BENCH_DEVICE": "cuda",
            "V8_BENCH_SECONDS": str(seconds),
            "V8_BENCH_LEDGER_PATH": destination,
            "V8_BENCH_CANARY": "1" if canary else "0",
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
        "required_exports": ["v8-bench.json"],
        "source_paths": a1_prof_pack._source_paths(),
        "resume": {
            "unit_kind": "v8-indexed-cell",
            "ledger_path": destination,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "v8_bench": {
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
            "scope": (
                "canary/main/at most two manual mechanical completions combined; "
                "Q2/Q4/global subordinate"
            ),
            "matrix": {
                "worlds": [64] if canary else [64, 256, 1024],
                "rosters": ["scripted", "frozen", "mixed"],
                "repeats": 1 if canary else 3,
                "warmups": 1 if canary else 3,
            },
            "kill_rule": "any mismatch, or either vector61_non_flood/v8_combined E1024 speedup <2x",
            "transport_reference_pin": "313758255ee5b9f1422a0202c14a80cc32554857",
        },
        "mac_preflight": preflight,
    }
    if canary:
        value["env"].update({"SNAKE_COLAB_CANARY": "1", "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE})
        value["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    elif canary_receipt:
        path = canary_receipt.resolve(strict=True)
        value["canary"] = {"receipt": {"path": str(path), "sha256": sha256_file(path)}}
    else:
        value["canary"] = {"required": "same-pin native V8 canary receipt"}
    value = convert_public_checkpoints(value, repo=ROOT)
    assert_upload_budget(value, repo=ROOT)
    return value


def _dispatch(artifact: Path, commit: str, preflight: Path, main: bool, rate: float) -> str:
    receipt = ' --canary-receipt "$ART/canary-output/receipt.json"' if main else ""
    job = "main" if main else "canary"
    return f"""#!/usr/bin/env bash
set -euo pipefail
REPO={shlex.quote(str(ROOT))}
ART={shlex.quote(str(artifact))}
PIN={commit}
PY={PRIMARY_PY}
export PATH="$(dirname "$PY"):$PATH"
cd "$REPO"
"$PY" -m research.gpu_sim_spike.v8_pack \\
  --commit "$PIN" --artifact-dir "$ART" \\
  --preflight {shlex.quote(str(preflight))} \\
  --rate-units-per-hour {rate}{receipt} --validate
exec "$REPO/tools/ops/dispatch.sh" gpu-v8-indexed-{job}-{commit[:8]} \\
  "$ART/v8-{job}.json" --broker "$REPO" --cwd "$REPO" \\
  --state-dir "$ART/state-{job}" --dispatch-txt "$ART/DISPATCH.txt"
"""


def _write(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise ManifestRefused(f"immutable pack differs: {path}")
    else:
        write_once(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    commit = args.commit or git_head()
    require_source_pin(commit)
    if git_head() != commit:
        raise ManifestRefused("pack checkout must be exactly pinned")
    art = args.artifact_dir.resolve()
    art.mkdir(parents=True, exist_ok=True)
    pf_path = (args.preflight or art / "mac-preflight/mac-preflight.json").resolve()
    pf = preflight_binding(pf_path, commit)
    for name, mode in (
        ("verified-ledger.json", False),
        ("canary-ledger/verified-ledger.json", True),
    ):
        path = art / name
        if not path.exists():
            v8_units.publish_json(
                path,
                v8_units.empty_ledger(
                    v8_units.namespace(commit, canary=mode), v8_units.allocation(canary=mode)
                ),
            )
    common = {
        "commit": commit,
        "ledger": art / "verified-ledger.json",
        "canary_ledger": art / "canary-ledger/verified-ledger.json",
        "preflight": pf,
        "rate": args.rate_units_per_hour,
    }
    canary = manifest(**common, output_dir=art / "canary-output", canary=True)
    science = manifest(
        **common, output_dir=art / "main-output", canary=False, canary_receipt=args.canary_receipt
    )
    if args.validate:
        from research.colab_cli.supervisor import validate
        from tools.ops.dispatch import require_resume_policy

        validate(canary, repo=ROOT)
        require_resume_policy(science)
        if args.canary_receipt:
            validate(science, repo=ROOT)
    for name, value in (
        ("v8-canary.json", canary),
        ("v8-main.json" if args.canary_receipt else "v8-main-template.json", science),
    ):
        _write(art / name, json.dumps(value, indent=2, sort_keys=True) + "\n")
    for name, is_main in (("dispatch-canary.sh", False), ("dispatch-main-after-canary.sh", True)):
        path = art / name
        _write(path, _dispatch(art, commit, pf_path, is_main, args.rate_units_per_hour))
        path.chmod(0o755)
    _write(
        art / "DISPATCH.txt",
        f"cd {ROOT}\nnice -n 15 bash {art}/dispatch-canary.sh\n"
        "# After verified same-pin canary/release:\n"
        f"nice -n 15 bash {art}/dispatch-main-after-canary.sh\n",
    )
    print(json.dumps({"artifact_dir": str(art), "commit": commit, "prepared": True}))


if __name__ == "__main__":
    main()
