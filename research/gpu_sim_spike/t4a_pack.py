"""Prepare T4A-DEV-01 A100 canary and main packets; never dispatches a runtime.

T4A is a pre-registered, outcome-free batch-invariance premise test.  Its
canary loads the separately compiled fixed-pin binaries and its main is only written
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
    t4a_native,
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
COMPILE_AMENDMENT = ARTIFACTS / "exp/fable-20261024/JUDGMENT_T4A_COMPILE_AMENDMENT.md"
COMPILE_AMENDMENT_SHA256 = "1eac049440eaad214400493037c88b7098abbde530be0125ea15b30363c1a57f"
DISPATCH_RULING = ARTIFACTS / "exp/fable-20261024/JUDGMENT_T4A_DISPATCH.md"
DISPATCH_RULING_SHA256 = "d333e186c53d9be8de5f73aebeb21bb788361be9e8fd64d6f7cac7ab8a4afd34"
REVIEW02_RULING = ARTIFACTS / "exp/fable-20261024/JUDGMENT_T4A_REVIEW02.md"
REVIEW02_RULING_SHA256 = "ee638c7f2b0d155e37ea1dbf47bc8344bdcca7a0369b7d2c3c5460e79e126214"
RESUME_RULING = ARTIFACTS / "exp/fable-20261024/JUDGMENT_T4A_RESUME_NO_CLI.md"
RESUME_RULING_SHA256 = "9733d5034b61265d2eba516e9303b401492eb3092e2309d89e2cb1cb82fdbd63"
BINDINGS += (
    (COMPILE_AMENDMENT, COMPILE_AMENDMENT_SHA256),
    (DISPATCH_RULING, DISPATCH_RULING_SHA256),
    (REVIEW02_RULING, REVIEW02_RULING_SHA256),
    (RESUME_RULING, RESUME_RULING_SHA256),
)
PROTOCOL = ROOT / PACKAGE / "T4A_PROTOCOL_20261024.md"
PROTOCOL_SHA256 = "b2e55b1f640a4de8872c3feefdc51d74311d19e87cc58bdba9f8510e17db71bf"
LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"
# The ruling freezes a 120 minute base series and owner 2x effective timeout:
# compiler 5/10, canary 10/20, main 105/210, per adopted sibling amendment.
COMPILER = {"est_minutes": 5, "timeout_minutes": 10}
CANARY = {"est_minutes": 10, "timeout_minutes": 20}
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
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ManifestRefused("T4A frozen protocol bytes differ")


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


def scientific_admission() -> dict[str, Any]:
    """Cite the adopted pre-data authority, without implying review-03 ACCEPT."""
    return {
        "study": "T4A-DEV-01",
        "ruling_sha256": DISPATCH_RULING_SHA256,
        "amendment_sha256": COMPILE_AMENDMENT_SHA256,
        "review02_ruling_sha256": REVIEW02_RULING_SHA256,
        "resume_ruling_sha256": RESUME_RULING_SHA256,
        "scope": "separate same-pin A100 compile; Astra review-03 ACCEPT required before dispatch",
    }


def native_compile_template(commit: str, packet: Path) -> dict[str, Any]:
    """Prepare the adopted bounded A100 compile-only prerequisite."""
    value = native_compile_pack.manifest(commit, packet / "native-compile-output")
    value["job_id"] = f"t4a-native-compile-a100-{commit[:12]}-01"
    value["requirements"] = {
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 5,
    }
    value["timeout_minutes"] = 10
    value["compile_envelope"] = {
        "base": {"timeout_minutes": 5},
        "effective": {"timeout_minutes": 10},
        "series_base_minutes": 120,
        "series_effective_minutes": 240,
        "estimate_basis": "adopted Fable pre-data compile split; not measured on this pin",
    }
    value["budget"] = _wall_compatibility(COMPILER)
    value["env"].update(
        NUMBA_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        T4A_NATIVE_BINARIES="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    value["required_exports"] += [
        f"native-binaries/{spec['name']}.so" for spec in native_compile.inventory(ROOT)[1]
    ]
    value["max_export_bytes"] = 256 * 1024**2
    value["resume_exemption"] = {
        "kind": "short-canary",
        "reason": "Adopted Fable same-pin compile only; no kernels, matrix or outcomes",
        "timeout_minutes": 10,
    }
    value["scientific_admission"] = scientific_admission()
    value["t4a_operational_split"] = {"proposed": False, "amendment": _amendment(COMPILE_AMENDMENT)}
    value["inputs"] += _bindings()
    value["inputs"].append(
        {
            "path": str(COMPILE_AMENDMENT),
            "sha256": COMPILE_AMENDMENT_SHA256,
            "dest": "inputs/authority/t4a-compile-amendment.md",
        }
    )
    return value


def _amendment(path: Path) -> dict[str, Any]:
    """Accept the actual immutable adopted Markdown, never a synthetic approval."""
    path = path.resolve(strict=True)
    if sha256_file(path) != COMPILE_AMENDMENT_SHA256:
        raise ManifestRefused("T4A amendment bytes differ from the adopted Fable ruling")
    return {"path": str(path), "sha256": COMPILE_AMENDMENT_SHA256}


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


def canary_binding(
    receipt: Path,
    commit: str,
    preflight: dict[str, Any],
    *,
    attempt: int = 1,
    remaining_units: tuple[str, ...] | None = None,
) -> dict[str, Any]:
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
    if saved.get("t4a_envelope", {}).get("attempt") != _attempt(attempt):
        raise ManifestRefused("T4A main requires the SAME attempt's fresh canary receipt")
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
    if set(contents) != set(t4a_units.allocation(canary=True)):
        raise ManifestRefused("T4A canary verified timing allocation is incomplete")
    projection = _main_projection(
        {**bench, "units": [contents[unit] for unit in t4a_units.allocation(canary=True)]},
        native,
        remaining_units=remaining_units,
    )
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
        "native_cuda_compile": _verified_compile(saved, commit),
    }


def _main_projection(
    bench: dict[str, Any],
    native: dict[str, Any],
    *,
    remaining_units: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Conservatively forecast H256 from sealed H2 per-record timings.

    Remaining-roster setup/build and event coverage stay whole-roster costs;
    native proof and unclassified residual stay whole-run costs. Only the
    observed tick/comparison work scales by H256/H2. Missing/foreign timing data
    refuses main preparation; it is not repaired or estimated after the fact.
    """
    contents = bench.get("units")
    if not isinstance(contents, list) or len(contents) != len(t4a_units.allocation(canary=True)):
        raise ManifestRefused("T4A canary lacks all roster timing payloads")
    cells = t4a_units.allocation()
    remaining = cells if remaining_units is None else remaining_units
    if (
        not remaining
        or len(set(remaining)) != len(remaining)
        or any(u not in cells for u in remaining)
    ):
        raise ManifestRefused("T4A remaining allocation is empty or foreign")
    build_seconds = tick_seconds = comparison_seconds = event_seconds = 0.0
    remaining_build = remaining_tick = remaining_comparison = remaining_events = 0.0
    terminal_elapsed = None
    for index, content in enumerate(contents):
        if not isinstance(content, dict) or content.get("horizon") != 2:
            raise ManifestRefused("T4A canary timing payload is malformed")
        records = content.get("records")
        if not isinstance(records, list) or len(records) != 18:
            raise ManifestRefused("T4A canary timing matrix is incomplete")
        unit_build = unit_tick = unit_comparison = 0.0
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
            unit_comparison += compared
        events = content.get("event_seconds")
        if (
            isinstance(events, bool)
            or not isinstance(events, (int, float))
            or not math.isfinite(events)
            or events < 0
        ):
            raise ManifestRefused("T4A event timing absent or malformed")
        event_seconds += events
        if cells[index] in remaining:
            remaining_build += unit_build
            remaining_tick += unit_tick
            remaining_comparison += unit_comparison
            remaining_events += events
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
        + remaining_build
        + remaining_events
        + (remaining_tick + remaining_comparison + residual) * 128
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
        "remaining_units": list(remaining),
        "remaining_build_seconds": remaining_build,
        "remaining_event_seconds": remaining_events,
        "remaining_tick_seconds": remaining_tick,
        "remaining_comparison_seconds": remaining_comparison,
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
    verified = t4a_units.load_ledger(ledger, t4a_units.namespace(commit), t4a_units.allocation())
    t4a_units.load_ledger(
        canary_ledger, t4a_units.namespace(commit, canary=True), t4a_units.allocation(canary=True)
    )
    if template and (canary or canary_receipt):
        raise ManifestRefused("only an unadmitted main may be a template")
    if not canary and not template and canary_receipt is None:
        raise ManifestRefused("T4A main requires a released independently-reduced canary")
    lane, envelope = ("canary", CANARY) if canary else ("main", MAIN)
    remaining = tuple(unit for unit in t4a_units.allocation() if unit not in verified)
    binding = (
        None
        if canary or template
        else canary_binding(
            canary_receipt, commit, preflight, attempt=attempt, remaining_units=remaining
        )
    )
    result: dict[str, Any] = {
        "job_id": f"gpu-t4a-dev-01-{lane}-{commit[:12]}-{attempt:02d}",
        "commit": commit,
        "script": SCRIPT,
        "device": "cuda",
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
            "compiler": COMPILER,
            "operational_amendment": _amendment(COMPILE_AMENDMENT),
            "compile_proof": "same-pin separate A100 compile receipt and exact binaries required",
            "attempt": attempt,
            "remaining_main_units": list(remaining),
            "resume_ruling_sha256": RESUME_RULING_SHA256,
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
        "scientific_admission": False if template else scientific_admission(),
        "scientific_admission_basis": scientific_admission(),
    }
    if binding is not None:
        compile_proof = binding["native_cuda_compile"]
        result["native_cuda_compile"] = {"receipt": compile_proof}
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
        result["inputs"],
        output_dir.parent / f"metadata-{lane}-attempt-{attempt:02d}.tar",
        reuse_verified=True,
    )
    result["env"]["METADATA_ARCHIVE_SHA256"] = next(
        row["sha256"] for row in result["inputs"] if row["dest"] == "metadata.tar"
    )
    if binding is not None:
        _bind_native_inputs(result, compile_proof)
        approved = _amendment(COMPILE_AMENDMENT)
        result["inputs"].append(
            {
                "path": approved["path"],
                "sha256": approved["sha256"],
                "dest": "inputs/authority/t4a-operational-amendment.md",
            }
        )
        # Both ledgers are already in the same deterministic metadata archive.
        # A main must ship exactly the upload inventory proved by its canary.
        saved = json.loads((canary_receipt.parent / "manifest.json").read_text())

        def identity(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
            return sorted((r["dest"], r["sha256"]) for r in rows)

        if identity(result["inputs"]) != identity(saved["inputs"]):
            raise ManifestRefused("main input bytes differ from the compiled canary")
    assert_upload_budget(result, repo=ROOT)
    return result


def _write(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ManifestRefused(f"immutable packet differs: {path}")
    if not path.exists():
        write_once(path, text)


def _verified_compile(value: dict[str, Any], commit: str) -> dict[str, Any]:
    proof = value.get("native_cuda_compile", {}).get("receipt", {})
    if not isinstance(proof, dict) or not proof.get("path"):
        raise ManifestRefused("same-pin compile receipt absent from canary")
    if native_compile.binding(Path(proof["path"]), commit, ROOT) != proof:
        raise ManifestRefused("same-pin compile receipt binding differs")
    return proof


def _bind_native_inputs(value: dict[str, Any], proof: dict[str, Any]) -> None:
    rows = t4a_native.private_upload_inputs(proof, value["commit"])
    value["inputs"].extend(rows)
    report = next(row for row in rows if row["dest"].endswith("native-compile.json"))
    value["env"].update(t4a_native.runtime_environment(report["sha256"]))


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
    _bind_native_inputs(compiled, proof)
    compiled["requirements"]["est_minutes"] = 10
    compiled["timeout_minutes"] = 20
    compiled["budget"] = _wall_compatibility({"timeout_minutes": 20})
    compiled["t4a_envelope"]["canary"] = {"est_minutes": 10, "timeout_minutes": 20}
    compiled["t4a_envelope"]["operational_amendment"] = approved
    compiled["inputs"].append(
        {
            "path": approved["path"],
            "sha256": approved["sha256"],
            "dest": "inputs/authority/t4a-operational-amendment.md",
        }
    )
    # Keep the original bound metadata archive: it contains the verified
    # ledgers, protocol and rulings. The amendment is one extra private input.
    compiled["scientific_admission"] = scientific_admission()
    compiled.pop("admission_reason", None)
    output = packet / f"t4a-canary-compiled-attempt-{attempt:02d}.json"
    _write(output, compiled)
    return output


def admit_compiler(packet: Path, amendment: Path) -> Path:
    """Create the proposed compiler packet only after the exact judge amendment."""
    approved = _amendment(amendment)
    value = json.loads((packet / "native-compile-template.json").read_text())
    value["scientific_admission"] = scientific_admission()
    value.pop("admission_reason", None)
    value["t4a_operational_split"]["proposed"] = False
    value["t4a_operational_split"]["amendment"] = approved

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


def attempt_ledgers(
    packet: Path,
    commit: str,
    *,
    attempt: int,
    ledger: Path | None = None,
    completion_from: Path | None = None,
) -> tuple[Path, Path]:
    """Verify recovery bytes and give every attempt an empty, fresh H2 canary.

    ``--ledger`` is the explicit Mac-verified input. ``--completion-from`` must
    establish mechanical interruption and verified stop for completion attempts.
    Neither option dispatches or retries anything; retry remains manual.
    """
    attempt = _attempt(attempt)
    if attempt == 1 and completion_from is not None:
        raise ManifestRefused("attempt 1 cannot have completion-from evidence")
    if attempt > 1 and completion_from is None:
        raise ManifestRefused("T4A completion requires --completion-from verified stop receipt")
    main, fresh_canary = ledgers(packet, commit, attempt=attempt)
    if t4a_units.load_ledger(
        fresh_canary, t4a_units.namespace(commit, canary=True), t4a_units.allocation(canary=True)
    ):
        raise ManifestRefused("every attempt requires an empty fresh H2 canary ledger")
    recovered = None
    if completion_from is not None:
        prior = json.loads((completion_from.parent / "manifest.json").read_text())
        if prior.get("env", {}).get("T4A_BENCH_CANARY") != "0":
            raise ManifestRefused("completion must resume main; each attempt needs a fresh canary")
        recovered, _ = recover_completion(
            completion_from,
            commit=commit,
            attempt=attempt,
            main_ledger=main,
            canary_ledger=fresh_canary,
        )
    if ledger is not None:
        verified = t4a_units.load_ledger(
            ledger, t4a_units.namespace(commit), t4a_units.allocation()
        )
        # Rebuild exclusively from immutable path/size/hash-verified payloads;
        # require exact ledger membership, including refusal of unlisted units.
        selected = main.parent / "verified-input" / "verified-ledger.json"
        rebuilt = t4a_units.build_ledger(ledger.parent, selected, commit=commit)
        if {row["unit_id"] for row in rebuilt["units"]} != set(verified):
            raise ManifestRefused("verified input ledger omits immutable completed units")
        if recovered is not None and selected.read_bytes() != recovered.read_bytes():
            raise ManifestRefused("provided ledger differs from verified interruption exports")
        main = selected
    elif recovered is not None:
        main = recovered
    verified = t4a_units.load_ledger(main, t4a_units.namespace(commit), t4a_units.allocation())
    if attempt == 1 and verified:
        raise ManifestRefused("attempt 1 requires an empty main ledger")
    if attempt > 1 and (not verified or len(verified) == len(t4a_units.allocation())):
        raise ManifestRefused("completion requires verified units and missing main units")
    return main, fresh_canary


def prepare_main(
    packet: Path,
    commit: str,
    preflight_path: Path,
    canary_receipt: Path,
    *,
    attempt: int = 1,
    completion_from: Path | None = None,
    ledger: Path | None = None,
) -> Path:
    require_source_pin(commit)
    attempt = _attempt(attempt)
    preflight = preflight_binding(preflight_path, commit)
    ledger, canary_ledger = attempt_ledgers(
        packet, commit, attempt=attempt, ledger=ledger, completion_from=completion_from
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
    ledger: Path | None = None,
) -> dict[str, Any]:
    attempt = _attempt(attempt)
    require_source_pin(commit)
    if git_head() != commit:
        raise ManifestRefused("packet source is not the requested commit")
    preflight = preflight_binding(preflight_path, commit)
    packet.mkdir(parents=True, exist_ok=True)
    ledger, canary_ledger = attempt_ledgers(
        packet, commit, attempt=attempt, ledger=ledger, completion_from=completion_from
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
    parser.add_argument(
        "--ledger", type=Path, help="Mac-verified main ledger; never a canary ledger"
    )
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
        ledger=args.ledger,
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
                ledger=args.ledger,
            )
        )
    print(json.dumps(prepared))


if __name__ == "__main__":
    main()
