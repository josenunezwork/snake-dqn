"""Prepare T4A-DEV-01 A100 canary and main packets; never dispatches a runtime.

T4A is a pre-registered, outcome-free batch-invariance premise test.  Its
canary performs the fixed-pin native compile proof and its main is only written
after the exact canary export inventory has been released and independently
reduced on the Mac.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import (
    a1_prof_pack,
    native_compile,
    native_compile_pack,
    r2_pack,
    t4a_units,
)
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)
from research.metadata_archive import bundle_inputs

ROOT = r2_pack.ROOT
PACKAGE = r2_pack.PACKAGE
SCRIPT = f"{PACKAGE}/t4a_bench.sh"
R2_PIN = "e66a701fd43951dc80e976485ed77113a91dddcb"
ENGINE_FILES = (
    "r2_cache.py",
    "v8_device.py",
    "masks_device.py",
    "anchors_device.py",
    "ego_device.py",
    "vector_device.py",
    "vector_geometry.py",
    "device_sim.py",
    "realistic_benchmark.py",
    "device_policy.py",
    "integration_engine.py",
    "a100_ruler.py",
    "r2_bench.py",
    "r2_identity.py",
)
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever")
BINDINGS = (
    (
        ARTIFACTS / "exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md",
        "d0041e5413a6456fd338865465e33ef7bbfc06227631f0739621ccc788e3f10c",
    ),
    (
        ARTIFACTS / "exp/JUDGMENT_GPU_LABEL_PATH_20261024.md",
        "ef488af238e1f60818dc5baea9fbc17001ea9fe09c1590867f3ce00cc08f4c00",
    ),
)
PROTOCOL = ROOT / PACKAGE / "T4A_PROTOCOL_20261024.md"
LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"
# The ruling freezes a 120 minute base series and owner 2x effective timeout:
# canary 15/30 minutes, main 105/210 minutes, inclusive of fixed-pin compile.
CANARY = {"est_minutes": 15, "timeout_minutes": 30}
MAIN = {"est_minutes": 105, "timeout_minutes": 210}
PRODUCER_SECONDS = {"canary": 12 * 60, "main": 100 * 60}
RATE_UNITS_PER_HOUR = 10.0
CLEANUP_RESERVE_UNITS = 0.01
MAX_MANUAL_ATTEMPTS = 3
MECHANICAL_FAILURE_CLASSES = {"RPC_EXEC_POLL_TIMEOUT"}
MECHANICAL_ERROR_MARKERS = ("timeout", "transport", "interrupted", "transfer")
AMENDMENT_SCHEMA = "gpu-sim-spike-t4a-operational-amendment/v1"


def _wall_compatibility(envelope: dict[str, int]) -> dict[str, Any]:
    """Required broker fields, derived only from the binding wall timeout."""
    unit_cap = envelope["timeout_minutes"] * RATE_UNITS_PER_HOUR / 60.0
    return {
        "unit_cap": unit_cap,
        "rate_units_per_hour": RATE_UNITS_PER_HOUR,
        "reserve_units": CLEANUP_RESERVE_UNITS,
        "other_assignments": {},
        "classification": "wall-derived compatibility field; not a science budget or unit ceiling",
        "derivation": "effective timeout_minutes * pre-priced rate_units_per_hour / 60",
    }


def _attempt(value: int) -> int:
    """Validate an explicit, outcome-blind original/completion attempt."""
    if type(value) is not int or not 1 <= value <= MAX_MANUAL_ATTEMPTS:
        raise ManifestRefused("T4A attempt must be an explicit integer from 1 through 3")
    return value


def require_source_pin(commit: str) -> None:
    """Require a clean published pin and an unchanged R2 I-prime engine."""
    r2_pack.require_source_pin(commit)
    for name in ENGINE_FILES:
        relative = f"{PACKAGE}/{name}"
        old = subprocess.check_output(
            ["git", "rev-parse", f"{R2_PIN}:{relative}"], cwd=ROOT, text=True
        )
        new = subprocess.check_output(
            ["git", "rev-parse", f"{commit}:{relative}"], cwd=ROOT, text=True
        )
        if old != new:
            raise ManifestRefused(f"T4A must retain the R2 I-prime engine: {name}")
    for path, digest in BINDINGS:
        if not path.is_file() or sha256_file(path) != digest:
            raise ManifestRefused(f"T4A authority input differs: {path}")
    if not PROTOCOL.is_file():
        raise ManifestRefused("T4A pre-data protocol is absent")


def _bindings() -> list[dict[str, str]]:
    rows = [
        {"path": str(path), "sha256": digest, "dest": "inputs/authority/" + path.name}
        for path, digest in BINDINGS
    ]
    rows.append(
        {
            "path": str(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
            "dest": "inputs/T4A_PROTOCOL_20261024.md",
        }
    )
    return rows


def native_compile_template(commit: str, packet: Path) -> dict[str, Any]:
    """Write an unadmitted proposed L4 compiler packet; never dispatch it."""
    value = native_compile_pack.manifest(commit, packet / "native-compile-output")
    value["job_id"] = f"t4a-native-compile-l4-{commit[:12]}-proposed"
    value["requirements"] = {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 2, "est_minutes": 5}
    value["timeout_minutes"] = 10
    value["compile_envelope"] = {
        "base": {"timeout_minutes": 5},
        "effective": {"timeout_minutes": 10},
        "estimate_basis": "proposed compiler prerequisite, not adopted or measured on this pin",
    }
    value["budget"] = {
        "unit_cap": 0.5,
        "rate_units_per_hour": 3.0,
        "reserve_units": CLEANUP_RESERVE_UNITS,
        "other_assignments": {},
        "classification": "wall-derived compatibility field; not a science budget or unit ceiling",
    }
    value["env"].update(NUMBA_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    value["resume_exemption"] = {
        "kind": "short-canary",
        "reason": "Proposed compiler prerequisite pending a Fable pre-data amendment",
        "timeout_minutes": 10,
    }
    value["scientific_admission"] = False
    value["admission_reason"] = "REQUIRES_FABLE_PRE_DATA_OPERATIONAL_AMENDMENT"
    value["t4a_operational_split"] = {
        "proposed": True,
        "compile_base_minutes": 5,
        "compile_effective_minutes": 10,
        "requires_amendment": True,
    }
    return value


def _amendment(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    body = json.loads(path.read_text())
    expected = {
        "schema": AMENDMENT_SCHEMA,
        "version": 1,
        "study": "T4A-DEV-01",
        "verdict": "APPROVE_PRE_DATA_OPERATIONAL_AMENDMENT",
        "judge": "Fable",
        "pre_data": True,
        "science_unchanged": True,
        "base_minutes": {"compile": 5, "canary": 10, "main": 105},
        "effective_minutes": {"compile": 10, "canary": 20, "main": 210},
        "authority_sha256": BINDINGS[0][1],
    }
    if body != expected:
        raise ManifestRefused("T4A operational amendment is not the exact approved Fable schema")
    return {"path": str(path), "sha256": sha256_file(path), "body": body}


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    """Bind a measured current-pin CPU real-entrypoint tiny matrix."""
    t4a_units._verify(path)
    body = json.loads(path.read_text())
    receipt = Path(str(body.get("receipt_path", "")))
    if not receipt.is_file() or t4a_units._verify(receipt) != body.get("receipt_sha256"):
        raise ManifestRefused("T4A preflight receipt binding differs")
    expected = list(t4a_units.allocation(smoke=True))
    if (
        (body.get("status"), body.get("commit"), body.get("device"), body.get("completed_units"))
        != ("PREFLIGHT", commit, "cpu", expected)
        or not isinstance(body.get("entrypoint_run_wall_seconds"), (int, float))
        or body["entrypoint_run_wall_seconds"] <= 0
    ):
        raise ManifestRefused("T4A needs a current-pin CPU real-entrypoint preflight")
    from research.gpu_sim_spike.benchmark import source_hashes

    runtime = json.loads(receipt.read_text())
    if (
        runtime.get("commit"),
        runtime.get("status"),
        runtime.get("completed_units"),
        runtime.get("source_sha256"),
    ) != (commit, "MEASURED_PENDING_MAC_REVIEW", expected, body.get("source_sha256")):
        raise ManifestRefused("T4A preflight runtime receipt is incomplete")
    if runtime.get("source_sha256") != source_hashes():
        raise ManifestRefused("T4A preflight source closure differs")
    from research.gpu_sim_spike import t4a_analyze

    reduced = t4a_analyze.aggregate(receipt.parent, commit, smoke=True)
    if reduced.get("verdict") != "PREFLIGHT":
        raise ManifestRefused("T4A CPU real-entrypoint smoke did not reduce to PREFLIGHT")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "receipt_path": str(receipt),
        "receipt_sha256": body["receipt_sha256"],
        "source_sha256": body["source_sha256"],
        "entrypoint_run_wall_seconds": body["entrypoint_run_wall_seconds"],
    }


def canary_binding(receipt: Path, commit: str, preflight: dict[str, Any]) -> dict[str, Any]:
    """Validate the released native canary and rerun the frozen Mac reducer."""
    if not receipt.is_file():
        raise ManifestRefused("same-pin T4A canary receipt is absent")
    body = json.loads(receipt.read_text())
    manifest = receipt.parent / "manifest.json"
    if (body.get("status"), body.get("commit"), body.get("stop_verified")) != (
        "PASS",
        commit,
        True,
    ):
        raise ManifestRefused("same-pin native T4A PASS/release receipt required")
    if not manifest.is_file() or sha256_file(manifest) != body.get("manifest_sha256"):
        raise ManifestRefused("T4A canary manifest binding differs")
    saved = json.loads(manifest.read_text())
    if saved.get("script") != SCRIPT or saved.get("mac_preflight") != preflight:
        raise ManifestRefused("T4A canary was not created from this packet")
    exports = receipt.parent / "exports"
    listed = body.get("exports")
    expected = {row.get("path"): row for row in listed or [] if isinstance(row, dict)}
    actual = {
        file.relative_to(exports).as_posix(): {
            "path": file.relative_to(exports).as_posix(),
            "sha256": sha256_file(file),
            "size": file.stat().st_size,
        }
        for file in exports.rglob("*")
        if file.is_file() and not file.name.endswith((".sha256", ".complete"))
    }
    if not listed or len(expected) != len(listed) or expected != actual:
        raise ManifestRefused("T4A canary terminal export inventory differs")
    bench = json.loads((exports / "t4a-bench.json").read_text())
    if bench.get("status") != "MEASURED_PENDING_MAC_REVIEW" or bench.get("completed_units") != list(
        t4a_units.allocation(canary=True)
    ):
        raise ManifestRefused("T4A canary did not complete its three roster units")
    native = json.loads((exports / "native-parity.json").read_text())
    if native.get("status") != "PASS" or native.get("native_cuda_executed") is not True:
        raise ManifestRefused("T4A fixed-pin CUDA native compile proof did not pass")
    contents = t4a_units.load_ledger(
        exports / "verified-ledger.json",
        t4a_units.namespace(commit, canary=True),
        t4a_units.allocation(),
        exports=exports,
    )
    projection = _main_projection({**bench, "units": list(contents.values())}, native)
    if projection["projected_main_seconds"] > PRODUCER_SECONDS["main"]:
        raise ManifestRefused(
            "T4A canary forecast cannot fit the unchanged H256 matrix within 100 minutes"
        )
    from research.gpu_sim_spike import t4a_analyze

    reduced = t4a_analyze.aggregate(exports, commit, canary=True)
    if reduced.get("verdict") != "PASS_CANARY":
        raise ManifestRefused("independent T4A Mac canary reduction did not pass")
    return {
        "path": str(receipt.resolve()),
        "sha256": sha256_file(receipt),
        "analysis": reduced,
        "main_projection": projection,
        "operational_amendment": saved.get("t4a_envelope", {}).get("operational_amendment"),
    }


def _main_projection(bench: dict[str, Any], native: dict[str, Any]) -> dict[str, float | int]:
    """Conservatively forecast H256 from sealed H2 per-record timings.

    Setup/build, event coverage, and native compile stay whole-run costs.  Only
    the observed tick work scales by H256/H2.  Missing/foreign timing data
    refuses main preparation; it is not repaired or estimated after the fact.
    """
    contents = bench.get("units")
    if not isinstance(contents, list) or len(contents) != len(t4a_units.allocation(canary=True)):
        raise ManifestRefused("T4A canary lacks all roster timing payloads")
    build_seconds = tick_seconds = comparison_seconds = event_seconds = 0.0
    terminal_elapsed = None
    for index, content in enumerate(contents):
        if not isinstance(content, dict) or content.get("horizon") != 2:
            raise ManifestRefused("T4A canary timing payload is malformed")
        records = content.get("records")
        if not isinstance(records, list) or len(records) != 18:
            raise ManifestRefused("T4A canary timing matrix is incomplete")
        unit_build = unit_tick = 0.0
        for record in records:
            if not isinstance(record, dict):
                raise ManifestRefused("T4A canary timing record is malformed")
            build = record.get("build_seconds")
            ticks = record.get("tick_seconds")
            if (
                isinstance(build, bool)
                or not isinstance(build, (int, float))
                or not math.isfinite(build)
                or build < 0
                or not isinstance(ticks, list)
                or len(ticks) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in ticks
                )
            ):
                raise ManifestRefused("T4A canary build/tick timing is malformed")
            unit_build += float(build)
            unit_tick += sum(float(value) for value in ticks)
            compared = record.get("comparison_seconds")
            if (
                isinstance(compared, bool)
                or not isinstance(compared, (int, float))
                or not math.isfinite(compared)
                or compared < 0
            ):
                raise ManifestRefused("T4A comparison timing absent or malformed")
            comparison_seconds += compared
        events = content.get("event_seconds")
        if (
            isinstance(events, bool)
            or not isinstance(events, (int, float))
            or not math.isfinite(events)
            or events < 0
        ):
            raise ManifestRefused("T4A event timing absent or malformed")
        event_seconds += events
        elapsed = content.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ManifestRefused("T4A canary event/setup timing is malformed")
        build_seconds += unit_build
        tick_seconds += unit_tick
        if index == len(contents) - 1:
            terminal_elapsed = float(elapsed)
    native_seconds = bench.get("native_elapsed_seconds")
    if (
        isinstance(native_seconds, bool)
        or not isinstance(native_seconds, (int, float))
        or not math.isfinite(native_seconds)
        or native_seconds < 0
    ):
        raise ManifestRefused("T4A native proof elapsed time is malformed")
    if terminal_elapsed is None:
        raise ManifestRefused("T4A canary terminal elapsed time is absent")
    # ``elapsed_seconds`` is measured from producer start, so only the final
    # roster payload covers one whole canary.  Summing cumulative unit values
    # would double-count earlier rosters and conceal the actual forecast.
    residual = (
        terminal_elapsed
        - float(native_seconds)
        - build_seconds
        - tick_seconds
        - comparison_seconds
        - event_seconds
    )
    if residual < 0:
        raise ManifestRefused("T4A canary timing components exceed terminal elapsed time")
    # Comparison and evidence formatting scale with H too.  Treat every
    # unclassified residual second as H-dependent, including prior JSON writes.
    projected = (
        float(native_seconds)
        + build_seconds
        + event_seconds
        + (tick_seconds + comparison_seconds + residual) * 128
    )
    return {
        "native_seconds": float(native_seconds),
        "build_seconds": build_seconds,
        "event_and_setup_seconds": event_seconds,
        "canary_tick_seconds": tick_seconds,
        "canary_comparison_seconds": comparison_seconds,
        "scaled_residual_seconds": residual,
        "horizon_multiplier": 128,
        "projected_main_seconds": projected,
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
    template: bool = False,
    attempt: int = 1,
) -> dict[str, Any]:
    """Make a packet only; ``template`` remains non-admitted until canary reduction."""
    attempt = _attempt(attempt)
    t4a_units.load_ledger(ledger, t4a_units.namespace(commit), t4a_units.allocation())
    t4a_units.load_ledger(
        canary_ledger, t4a_units.namespace(commit, canary=True), t4a_units.allocation(canary=True)
    )
    if template and (canary or canary_receipt):
        raise ManifestRefused("only an unadmitted main may be a template")
    if not canary and not template and canary_receipt is None:
        raise ManifestRefused("T4A main requires a released independently-reduced canary")
    lane, envelope = ("canary", CANARY) if canary else ("main", MAIN)
    binding = None if canary or template else canary_binding(canary_receipt, commit, preflight)
    result: dict[str, Any] = {
        "job_id": f"gpu-t4a-dev-01-{lane}-{commit[:12]}-{attempt:02d}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "T4A-DEV-01 excluded batch-invariance premise evidence only; "
        "no label, hero, evaluation, release or serving claim.",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/integration_requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "GPU_SIM_V8_INDEXED": "0",
            "GPU_SIM_R2_DEBUG": "0",
            "T4A_BENCH_DEVICE": "cuda",
            "T4A_BENCH_SECONDS": str(PRODUCER_SECONDS[lane]),
            "T4A_BENCH_LEDGER_PATH": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "T4A_BENCH_SMOKE": "0",
            "T4A_BENCH_CANARY": "1" if canary else "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": envelope["est_minutes"],
        },
        "timeout_minutes": envelope["timeout_minutes"],
        "output_dir": str(output_dir.resolve()),
        "inputs": checkpoint_inputs()
        + r2_pack._ledger_inputs(ledger, "inputs")
        + r2_pack._ledger_inputs(canary_ledger, "canary-inputs")
        + _bindings(),
        "budget": _wall_compatibility(envelope),
        "lease_seconds": 600,
        "max_export_bytes": 1024**3,
        "required_exports": ["t4a-bench.json", "native-parity.json", "verified-ledger.json"],
        "resume": {
            "unit_kind": "roster-replay-matrix",
            "ledger_path": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": r2_pack.TRANSPORT_PIN,
        "mac_preflight": preflight,
        "t4a_envelope": {
            "ruling": {"path": str(BINDINGS[0][0]), "sha256": BINDINGS[0][1], "section": "2.1"},
            "gpu_label_path": {"path": str(BINDINGS[1][0]), "sha256": BINDINGS[1][1]},
            "r2_pin": R2_PIN,
            "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
            "canary": CANARY,
            "main": MAIN,
            "series_base_minutes": 120,
            "series_effective_minutes": 240,
            "producer_deadline_seconds": PRODUCER_SECONDS[lane],
            "allocation": list(t4a_units.allocation(canary=canary)),
            "matrix": {
                "E": [4, 64, 1024],
                "layouts": 3,
                "replays": 2,
                "horizon": 2 if canary else 256,
                "padded_forward_rows": 1024,
            },
            "compile_proof": "fixed-pin native CUDA proof is required inside the canary",
            "attempt": attempt,
            "manual_completion_attempts": MAX_MANUAL_ATTEMPTS,
            "retry_rule": "manual outcome-blind completion after verified mechanical failure, "
            "stop and missing units; fresh manifest/output/receipt on each attempt",
            "acceptance": "T4A_PROTOCOL_20261024.md / t4a_analyze.py (frozen before data)",
        },
        "canary": (
            {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
            if canary
            else {"required": "same-pin native T4A canary receipt and Mac PASS_CANARY reduction"}
        ),
        "native_cuda_compile": {"required": "same-pin released native compile receipt"},
        "scientific_admission": not template,
    }
    if binding is not None:
        result["canary"] = {"receipt": binding}
        if binding.get("operational_amendment"):
            result["t4a_envelope"]["canary"] = {"est_minutes": 10, "timeout_minutes": 20}
            result["t4a_envelope"]["compiler"] = {"est_minutes": 5, "timeout_minutes": 10}
            result["t4a_envelope"]["operational_amendment"] = binding["operational_amendment"]
    if canary:
        result["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
    if template:
        result["admission_reason"] = (
            "TEMPLATE_ONLY: same-pin native T4A canary PASS_CANARY is required"
        )
    result = convert_public_checkpoints(result, repo=ROOT)
    result["inputs"] = bundle_inputs(
        result["inputs"], output_dir.parent / f"metadata-{lane}.tar", reuse_verified=True
    )
    result["env"]["METADATA_ARCHIVE_SHA256"] = next(
        row["sha256"] for row in result["inputs"] if row["dest"] == "metadata.tar"
    )
    assert_upload_budget(result, repo=ROOT)
    return result


def _write(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ManifestRefused(f"immutable packet differs: {path}")
    if not path.exists():
        write_once(path, text)


def bind_compile(packet: Path, receipt: Path, amendment: Path, *, attempt: int = 1) -> Path:
    """Bind a ratified operational split and released compile proof to a new canary.

    The original frozen canary draft is retained unchanged.  Absence of the
    exact amendment is a refusal, not authorization to run the compiler.
    """
    attempt = _attempt(attempt)
    source = packet / f"t4a-canary-attempt-{attempt:02d}.json"
    if not source.is_file():
        raise ManifestRefused("T4A canary template is absent")
    draft = json.loads(source.read_text())
    commit = draft.get("commit")
    if not isinstance(commit, str):
        raise ManifestRefused("T4A canary template commit is malformed")
    proof = native_compile.binding(receipt, commit, ROOT)
    approved = _amendment(amendment)
    compiled = json.loads(json.dumps(draft))
    compiled["native_cuda_compile"] = {"receipt": proof}
    compiled["requirements"]["est_minutes"] = 10
    compiled["timeout_minutes"] = 20
    compiled["budget"] = _wall_compatibility({"timeout_minutes": 20})
    compiled["t4a_envelope"]["canary"] = {"est_minutes": 10, "timeout_minutes": 20}
    compiled["t4a_envelope"]["operational_amendment"] = approved
    compiled["inputs"].append(
        {
            "path": approved["path"],
            "sha256": approved["sha256"],
            "dest": "inputs/authority/t4a-operational-amendment.json",
        }
    )
    # Keep the original bound metadata archive: it contains the verified
    # ledgers, protocol and rulings. The amendment is one extra private input.
    compiled["scientific_admission"] = True
    compiled.pop("admission_reason", None)
    output = packet / f"t4a-canary-compiled-attempt-{attempt:02d}.json"
    _write(output, compiled)
    return output


def admit_compiler(packet: Path, amendment: Path) -> Path:
    """Create the proposed compiler packet only after the exact judge amendment."""
    approved = _amendment(amendment)
    value = json.loads((packet / "native-compile-template.json").read_text())
    value["scientific_admission"] = True
    value.pop("admission_reason", None)
    value["t4a_operational_split"]["proposed"] = False
    value["t4a_operational_split"]["amendment"] = approved
    value["inputs"].append(
        {
            "path": approved["path"],
            "sha256": approved["sha256"],
            "dest": "t4a-operational-amendment.json",
        }
    )
    output = packet / "native-compile-admitted.json"
    _write(output, value)
    return output


def ledgers(packet: Path, commit: str, *, attempt: int = 1) -> tuple[Path, Path]:
    attempt = _attempt(attempt)
    attempt_root = packet / f"attempt-{attempt:02d}"
    pairs = (
        (attempt_root / "verified-ledger.json", False),
        (attempt_root / "canary-ledger/verified-ledger.json", True),
    )
    for path, canary in pairs:
        if not path.exists():
            t4a_units.publish_json(
                path,
                t4a_units.empty_ledger(
                    t4a_units.namespace(commit, canary=canary), t4a_units.allocation(canary=canary)
                ),
            )
    return pairs[0][0], pairs[1][0]


def _mechanical_failure(receipt: dict[str, Any], exports: Path) -> None:
    """Accept only a sealed outcome-blind interruption, never a behavior result."""
    failure = exports / "t4a-failure.json"
    if failure.is_file():
        t4a_units._verify(failure)
        body = json.loads(failure.read_text())
        if body.get("status") != "INCOMPLETE":
            raise ManifestRefused("T4A completion refuses behavioral/evidence-invalid prior output")
        reason = str(body.get("reason", "")).lower()
        if any(marker in reason for marker in MECHANICAL_ERROR_MARKERS):
            return
        raise ManifestRefused(
            "T4A completion failure record is not a known mechanical interruption"
        )
    failure_class = receipt.get("failure_class")
    error = str(receipt.get("error") or receipt.get("reason") or "").lower()
    if failure_class in MECHANICAL_FAILURE_CLASSES or any(
        marker in error for marker in MECHANICAL_ERROR_MARKERS
    ):
        return
    raise ManifestRefused("T4A completion requires a known mechanical supervisor failure")


def recover_completion(
    completion_from: Path,
    *,
    commit: str,
    attempt: int,
    main_ledger: Path,
    canary_ledger: Path,
) -> tuple[Path, Path]:
    """Build a fresh verified ledger from one released mechanical prior attempt."""
    attempt = _attempt(attempt)
    if attempt == 1:
        raise ManifestRefused("attempt 1 cannot have completion-from evidence")
    receipt_path = completion_from.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    previous = attempt - 1
    manifest_path = receipt_path.parent / "manifest.json"
    if (
        receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("commit") != commit
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != receipt.get("manifest_sha256")
    ):
        raise ManifestRefused("T4A completion needs a released same-pin INCOMPLETE receipt")
    manifest_body = json.loads(manifest_path.read_text())
    if (
        manifest_body.get("commit") != commit
        or manifest_body.get("script") != SCRIPT
        or manifest_body.get("t4a_envelope", {}).get("attempt") != previous
        or not str(manifest_body.get("job_id", "")).endswith(f"-{previous:02d}")
    ):
        raise ManifestRefused(
            "T4A completion prior manifest does not bind the preceding exact packet"
        )
    exports = receipt_path.parent / "exports"
    if not exports.is_dir():
        raise ManifestRefused("T4A completion prior export directory is absent")
    _mechanical_failure(receipt, exports)
    was_canary = manifest_body.get("env", {}).get("T4A_BENCH_CANARY") == "1"
    base = canary_ledger if was_canary else main_ledger
    target = base.parent / "recovered" / base.name
    t4a_units.build_ledger(exports, target, commit=commit, canary=was_canary)
    return (main_ledger, target) if was_canary else (target, canary_ledger)


def prepare_main(
    packet: Path,
    commit: str,
    preflight_path: Path,
    canary_receipt: Path,
    *,
    attempt: int = 1,
    completion_from: Path | None = None,
) -> Path:
    require_source_pin(commit)
    attempt = _attempt(attempt)
    if attempt == 1 and completion_from is not None:
        raise ManifestRefused("attempt 1 cannot have completion-from evidence")
    if attempt > 1 and completion_from is None:
        raise ManifestRefused("T4A completion attempt requires --completion-from prior receipt")
    preflight = preflight_binding(preflight_path, commit)
    ledger, canary_ledger = ledgers(packet, commit, attempt=attempt)
    if completion_from is not None:
        ledger, canary_ledger = recover_completion(
            completion_from,
            commit=commit,
            attempt=attempt,
            main_ledger=ledger,
            canary_ledger=canary_ledger,
        )
    value = manifest(
        commit=commit,
        output_dir=packet / f"main-output-attempt-{attempt:02d}",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=canary_receipt,
        attempt=attempt,
    )
    from research.colab_cli.supervisor import validate

    validate(value, repo=ROOT)
    path = packet / f"t4a-main-attempt-{attempt:02d}.json"
    _write(path, value)
    return path


def prepare(
    packet: Path,
    commit: str,
    preflight_path: Path,
    *,
    validate: bool = False,
    attempt: int = 1,
    completion_from: Path | None = None,
) -> dict[str, Any]:
    attempt = _attempt(attempt)
    if attempt == 1 and completion_from is not None:
        raise ManifestRefused("attempt 1 cannot have completion-from evidence")
    if attempt > 1 and completion_from is None:
        raise ManifestRefused("T4A completion attempt requires --completion-from prior receipt")
    require_source_pin(commit)
    if git_head() != commit:
        raise ManifestRefused("packet source is not the requested commit")
    preflight = preflight_binding(preflight_path, commit)
    packet.mkdir(parents=True, exist_ok=True)
    ledger, canary_ledger = ledgers(packet, commit, attempt=attempt)
    if completion_from is not None:
        ledger, canary_ledger = recover_completion(
            completion_from,
            commit=commit,
            attempt=attempt,
            main_ledger=ledger,
            canary_ledger=canary_ledger,
        )
    canary = manifest(
        commit=commit,
        output_dir=packet / f"canary-output-attempt-{attempt:02d}",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
        attempt=attempt,
    )
    template = manifest(
        commit=commit,
        output_dir=packet / f"main-output-attempt-{attempt:02d}",
        ledger=ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        template=True,
        attempt=attempt,
    )
    if validate:
        from research.colab_cli.supervisor import validate as validate_manifest

        validate_manifest(canary, repo=ROOT)
    _write(packet / f"t4a-canary-attempt-{attempt:02d}.json", canary)
    _write(packet / f"t4a-main-template-attempt-{attempt:02d}.json", template)
    if attempt == 1:
        _write(packet / "native-compile-template.json", native_compile_template(commit, packet))
    return {"prepared": True, "packet": str(packet), "commit": commit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--completion-from", type=Path)
    parser.add_argument("--compile-receipt", type=Path)
    parser.add_argument("--protocol-amendment", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--admit-compiler", action="store_true")
    args = parser.parse_args()
    if args.compile_receipt is not None and args.protocol_amendment is None:
        raise ManifestRefused("compile receipt and protocol amendment must be supplied together")
    if args.admit_compiler and (
        args.protocol_amendment is None or args.compile_receipt is not None
    ):
        raise ManifestRefused("compiler admission requires only the approved protocol amendment")
    if (
        args.protocol_amendment is not None
        and args.compile_receipt is None
        and not args.admit_compiler
    ):
        raise ManifestRefused("select --admit-compiler or supply the compile receipt")
    prepared = prepare(
        args.packet,
        args.commit,
        args.preflight,
        validate=args.validate,
        attempt=args.attempt,
        completion_from=args.completion_from,
    )
    if args.compile_receipt is not None:
        prepared["compiled_canary"] = str(
            bind_compile(
                args.packet, args.compile_receipt, args.protocol_amendment, attempt=args.attempt
            )
        )
    if args.admit_compiler:
        prepared["admitted_compiler"] = str(admit_compiler(args.packet, args.protocol_amendment))
    if args.canary_receipt is not None:
        prepared["main"] = str(
            prepare_main(
                args.packet,
                args.commit,
                args.preflight,
                args.canary_receipt,
                attempt=args.attempt,
                completion_from=args.completion_from,
            )
        )
    print(json.dumps(prepared))


if __name__ == "__main__":
    main()
