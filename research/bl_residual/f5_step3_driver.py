"""Sealed-step-3 runtime driver for the F5 long-body timing supplement.

This is deliberately a small operational producer.  It has no reducer and no
science inputs: its only terminal records are redacted timing dispositions.
``--dry-run`` is entirely offline; ``--qualify`` performs the new-application
CUDA/native-v6 admission; and ``--measure`` additionally requires the separate
independent acceptance input before it may call the frozen workload helper.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f5_job, f5_qualification, f5_timing_supplement

SCHEMA = "bl-f5-step3-control/v1"
LEDGER_SCHEMA = "bl-f5-step3-component-ledger/v1"
UNIT_SCHEMA = "bl-f5-step3-component-unit/v1"
COMPONENT_ID = "f5-step3-long-body-v1"
FIXTURE_IDS = tuple(range(672, 684))


class Step3Refusal(ValueError):
    """Refusal before a potentially scientific or unrecoverable action."""


def _refuse(reason: str) -> None:
    raise Step3Refusal("F5 step-3 refused: " + reason)


def _sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()


def _safe_name(name: object) -> Path:
    if not isinstance(name, str):
        _refuse("input name is malformed")
    path = Path(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        _refuse("input name escapes uploaded inputs")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _refuse("required JSON input is absent or unsafe")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Step3Refusal("required JSON input is malformed") from exc
    if not isinstance(value, dict):
        _refuse("required JSON input is not an object")
    return value


def _validate_source_hashes(control: Mapping[str, Any]) -> None:
    pins = control.get("source_hashes")
    if not isinstance(pins, dict) or not pins:
        _refuse("complete executable source inventory is absent")
    root = Path(__file__).resolve().parents[2]
    checked: dict[str, str] = {}
    for relative, expected in pins.items():
        item = _safe_name(relative)
        if item.parts[0] not in {"research", "src", "configs"} or not _sha(expected):
            _refuse("executable source inventory contains an invalid pin")
        path = root / item
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            _refuse("executable source differs from sealed inventory")
        checked[str(relative)] = str(expected)
    from research.bl_residual.f5_package import source_inventory

    if checked != source_inventory(root):
        _refuse("executable source inventory omits or adds a runtime dependency")
    if _canonical_sha(checked) != control.get("source_sha256"):
        _refuse("source inventory digest differs")


def validate_control(
    control_path: Path, fixtures: Path, inputs: Path, *, measure: bool = False
) -> dict[str, Any]:
    """Admit only sealed bytes; this function never opens CUDA or a fixture world."""
    control = _read_json(control_path)
    required = {
        "schema",
        "namespace",
        "allocation",
        "source_hashes",
        "source_sha256",
        "fixture_descriptor_sha256",
        "input_hashes",
        "environment_policy",
        "environment_reference",
        "execution_envelope",
    }
    if not required <= set(control) or control.get("schema") != SCHEMA:
        _refuse("control schema or required authority fields differ")
    if (
        not isinstance(control["namespace"], str)
        or not control["namespace"]
        or not _sha(control["allocation"])
    ):
        _refuse("component namespace or allocation is malformed")
    if not all(_sha(control[name]) for name in ("source_sha256", "fixture_descriptor_sha256")):
        _refuse("source or fixture identity is malformed")
    _validate_source_hashes(control)
    if (
        fixtures.is_symlink()
        or not fixtures.is_file()
        or digest(fixtures) != control["fixture_descriptor_sha256"]
    ):
        _refuse("fixture descriptor bytes differ")
    try:
        rows = json.loads(fixtures.read_text())
        canonical = f5_timing_supplement.frozen_fixtures(
            rows["fixtures"] if isinstance(rows, dict) else rows
        )
    except (json.JSONDecodeError, KeyError, TypeError, f5_timing_supplement.SupplementHold) as exc:
        raise Step3Refusal(
            "fixture descriptor does not contain exactly frozen timing fixtures"
        ) from exc
    if tuple(row["allocation_index"] for row in canonical) != FIXTURE_IDS:
        _refuse("fixture descriptor differs from 672 through 683")
    hashes = control["input_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        _refuse("input identity inventory is absent")
    if any(name == "verified-ledger.json" or name.startswith("units/") for name in hashes):
        _refuse("resume ledger and units must use verified component identity, not static pins")
    for name, entry in hashes.items():
        relative = _safe_name(name)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "size"}
            or not _sha(entry.get("sha256"))
            or type(entry.get("size")) is not int
            or entry["size"] <= 0
        ):
            _refuse("input identity entry is malformed")
        path = inputs / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["size"]
            or digest(path) != entry["sha256"]
        ):
            _refuse("uploaded input differs from sealed identity")
    if measure:
        qualification_input = hashes.get("qualification.json")
        if (
            not isinstance(qualification_input, Mapping)
            or not _sha(control.get("qualification_sha256"))
            or qualification_input.get("sha256") != control["qualification_sha256"]
        ):
            _refuse("sealed qualification input is absent or differs")
        qualification = _read_json(inputs / "qualification.json")
        if (
            qualification.get("status") != "PASS"
            or qualification.get("source_sha256") != control["source_sha256"]
            or qualification.get("fixture_descriptor_sha256")
            != control["fixture_descriptor_sha256"]
            or qualification.get("native_cuda_cpu_parity") != "PASS"
        ):
            _refuse("sealed qualification input does not match current driver identity")
    if (
        control["environment_policy"]
        != __import__(
            "research.bl_residual.f5_environment", fromlist=["policy_binding"]
        ).policy_binding()
    ):
        _refuse("native-v6 environment policy differs")
    if not isinstance(control["environment_reference"], Mapping) or not isinstance(
        control["execution_envelope"], Mapping
    ):
        _refuse("environment reference or execution envelope is malformed")
    envelope = control["execution_envelope"]
    if (
        envelope.get("host_workers") != 6
        or envelope.get("threads_per_worker") != 1
        or not isinstance(envelope.get("forecast_status"), str)
        or (not measure and not envelope["forecast_status"].startswith("HOLD"))
    ):
        _refuse("sealed execution envelope lacks six single-thread workers or HOLD forecast")
    if measure and (
        envelope.get("forecast_status") != "QUALIFIED_COMPLETE"
        or envelope.get("unknown_costs") != []
        or type(envelope.get("measured_total_seconds")) not in (int, float)
        or not math.isfinite(envelope["measured_total_seconds"])
        or envelope["measured_total_seconds"] <= 0
        or envelope["measured_total_seconds"] > 3600
        or envelope.get("base_timeout_minutes") != 60
        or envelope.get("effective_timeout_minutes") != 120
    ):
        _refuse("measure requires finite qualified 60/120-minute envelope")
    return control


def _binding(control: Mapping[str, Any]) -> dict[str, str]:
    return {
        "namespace": control["namespace"],
        "allocation": control["allocation"],
        "source_sha256": control["source_sha256"],
        "fixtures_sha256": control["fixture_descriptor_sha256"],
    }


def empty_ledger(control: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "component_id": COMPONENT_ID,
        "identity": _binding(control),
        "units": [],
    }


def _unit_path(output: Path, fixture_id: int) -> Path:
    return output / "units" / f"fixture-{fixture_id}.json"


def _verify_unit(
    output: Path, item: Mapping[str, Any], binding: Mapping[str, str]
) -> dict[str, Any]:
    if (
        set(item) != {"id", "path", "sha256", "size"}
        or type(item.get("id")) is not int
        or item["id"] not in FIXTURE_IDS
    ):
        _refuse("component ledger unit entry is malformed")
    relative = _safe_name(item["path"])
    path = output / relative
    if path != _unit_path(output, item["id"]):
        _refuse("component ledger unit path differs")
    if type(item["size"]) is not int or item["size"] <= 0 or not _sha(item["sha256"]):
        _refuse("component ledger unit identity is malformed")
    try:
        actual = verify_export(path)
    except ValueError as exc:
        raise Step3Refusal("completed component unit has partial or missing marker") from exc
    if actual != item["sha256"] or path.stat().st_size != item["size"]:
        _refuse("completed component unit path size or hash differs")
    raw = _read_json(path)
    if (
        set(raw)
        != {
            "schema",
            "component_id",
            "identity",
            "fixture_id",
            "record",
            "elapsed_seconds",
            "collection_elapsed_seconds",
        }
        or raw.get("schema") != UNIT_SCHEMA
        or raw.get("component_id") != COMPONENT_ID
        or raw.get("identity") != dict(binding)
        or raw.get("fixture_id") != item["id"]
    ):
        _refuse("completed component unit wrapper differs")
    record = raw.get("record")
    if not isinstance(record, dict) or any(
        not isinstance(raw.get(name), (int, float))
        or isinstance(raw.get(name), bool)
        or not math.isfinite(raw[name])
        or raw[name] < 0
        for name in ("elapsed_seconds", "collection_elapsed_seconds")
    ):
        _refuse("completed component record is malformed")
    f5_timing_supplement.validate_operational_record(record)
    if record.get("allocation_index") != item["id"]:
        _refuse("completed component unit fixture differs")
    if (
        record.get("fixture_descriptor_sha256") != binding["fixtures_sha256"]
        or record.get("source_provenance", {}).get("source_sha256") != binding["source_sha256"]
    ):
        _refuse("completed component record source or fixture identity differs")
    return raw


def load_verified_units(ledger: Path, control: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Return only atomically complete, bound units; any loss is a refusal, never recovery."""
    if ledger.is_symlink() or not ledger.is_file():
        _refuse("dedicated verified ledger is absent")
    try:
        verify_export(ledger)
    except ValueError as exc:
        raise Step3Refusal(
            "dedicated verified ledger is partial or lacks completion marker"
        ) from exc
    value = _read_json(ledger)
    if (
        value.get("schema") != LEDGER_SCHEMA
        or value.get("component_id") != COMPONENT_ID
        or value.get("identity") != _binding(control)
        or not isinstance(value.get("units"), list)
    ):
        _refuse("dedicated verified ledger is foreign")
    result: dict[int, dict[str, Any]] = {}
    for item in value["units"]:
        raw = _verify_unit(ledger.parent, item, _binding(control))
        fixture_id = int(item["id"])
        if fixture_id in result:
            _refuse("dedicated verified ledger has duplicate fixture")
        result[fixture_id] = raw
    return result


def _write_ledger(
    ledger: Path, control: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> None:
    export_json(ledger, {**empty_ledger(control), "units": list(entries)})
    verify_export(ledger)


def _qualification(
    control: Mapping[str, Any], inputs: Path, output: Path, *, fixture_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one source-native real CUDA action and CPU parity without timing fixtures."""
    actual = f5_job._environment_admission(None)
    environment = f5_job._environment_evidence(
        control, actual, refusal_output=output / "environment-refusal.json"
    )
    export_json(output / "environment.json", actual)
    checkpoint = inputs / "b8.pth"
    if not checkpoint.is_file() or digest(checkpoint) != f5_job.B8_SHA256:
        _refuse("current qualification requires pinned B8 input")
    from src.model.ego2s_network import load_ego2s_checkpoint

    model = load_ego2s_checkpoint(str(checkpoint), "cuda:0").eval()
    # The selection helpers themselves enforce real CUDA, native observation, shape and finite Qs.
    observation, mask = f5_qualification._native_observation(fixture_seed)
    cuda_q, cuda_action = f5_qualification._select_cuda(model, observation, mask)
    cpu_q, cpu_action = f5_qualification._cpu_action(model, observation, mask)
    if cuda_action != cpu_action or not np.allclose(
        cuda_q, cpu_q, rtol=f5_qualification.PARITY_RTOL, atol=f5_qualification.PARITY_ATOL
    ):
        _refuse("current B8 CUDA and CPU action parity differs")
    return (
        {
            "schema": "bl-f5-step3-qualification/v1",
            "status": "PASS",
            "fixture_descriptor_sha256": control["fixture_descriptor_sha256"],
            "source_sha256": control["source_sha256"],
            "environment_sha256": _canonical_sha(environment),
            "native_cuda_cpu_parity": "PASS",
            "physical_assignment": "NVIDIA_L4",
            "redacted": True,
        },
        actual,
    )


def _acceptance_target_sha256(control: Mapping[str, Any]) -> str:
    """Hash the independently reviewable control target without its future review input."""
    target = dict(control)
    hashes = dict(target.get("input_hashes", {}))
    hashes.pop("external-acceptance.json", None)
    target["input_hashes"] = hashes
    return _canonical_sha(target)


def _acceptance(control: Mapping[str, Any], inputs: Path, control_path: Path) -> None:
    entry = control["input_hashes"].get("external-acceptance.json")
    if entry is None:
        _refuse("measure requires separately bound independent acceptance input")
    value = _read_json(inputs / "external-acceptance.json")
    if (
        value.get("status") != "ACCEPTED"
        or value.get("source_sha256") != control["source_sha256"]
        or value.get("fixture_descriptor_sha256") != control["fixture_descriptor_sha256"]
        or value.get("acceptance_target_sha256") != _acceptance_target_sha256(control)
        or value.get("execution_envelope_sha256") != _canonical_sha(control["execution_envelope"])
        or value.get("qualification_sha256") != control["qualification_sha256"]
    ):
        _refuse("independent acceptance does not bind current package identity")


def _measure_worker(row: Mapping[str, Any], checkpoint: str) -> dict[str, Any]:
    """One single-thread host worker; it does not initialize CUDA."""
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    started = time.monotonic()
    return {
        "record": f5_timing_supplement.measure_fixture(row, Path(checkpoint)),
        "elapsed_seconds": time.monotonic() - started,
    }


def run(
    *, control_path: Path, fixtures: Path, inputs: Path, output: Path, ledger: Path, mode: str
) -> dict[str, Any]:
    if mode not in {"dry-run", "qualify", "family-canary", "measure"}:
        _refuse("unknown driver mode")
    admitted = mode in {"family-canary", "measure"} or (
        mode == "dry-run"
        and _read_json(control_path).get("execution_envelope", {}).get("forecast_status")
        == "QUALIFIED_COMPLETE"
    )
    control = validate_control(control_path, fixtures, inputs, measure=admitted)
    rows = json.loads(fixtures.read_text())
    frozen = f5_timing_supplement.frozen_fixtures(
        rows["fixtures"] if isinstance(rows, dict) else rows
    )
    if mode == "dry-run":
        load_verified_units(ledger, control)
        if admitted:
            _acceptance(control, inputs, control_path)
        return {"status": "DRY_RUN_VALID", "fixtures": list(FIXTURE_IDS), "runtime_calls": False}
    completed = load_verified_units(ledger, control)
    # The supervisor creates its output root before launching the child. Only
    # that empty directory (or an absent path) is fresh; never reuse evidence.
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        _refuse("attempt output must be fresh (absent or an empty directory)")
    if admitted:
        _acceptance(control, inputs, control_path)
    output.mkdir(parents=True, exist_ok=True)
    qualification, actual_environment = _qualification(
        control, inputs, output, fixture_seed=int(frozen[0]["seed"])
    )
    export_json(output / "qualification.json", qualification)
    if mode == "qualify":
        return {"status": "QUALIFIED", "qualification": "qualification.json"}
    if mode == "family-canary":
        # Same uploaded control/acceptance/qualification/ledger bytes as measure.
        # Exercise the real producer export seam without collecting or replaying
        # any timing unit. This excluded transport record is never a timing result.
        _summary(output, control, completed, role="EXCLUDED_FAMILY_CANARY")
        return {"status": "FAMILY_CANARY", "timing_units_executed": 0}
    if mode != "measure":
        _refuse("unknown driver mode")
    # ``ledger`` is a sealed input from an earlier attempt (empty on the first
    # attempt).  This attempt never rewrites it; each successor ledger lives
    # below the fresh output root.
    entries: list[dict[str, Any]] = []
    dispositions = {fixture_id: raw["record"]["crossing"] for fixture_id, raw in completed.items()}
    # Preserve completed units byte-for-byte in the fresh attempt root.  A
    # carried ABSENT is as final as a present crossing and is never recomputed.
    for fixture_id, raw in sorted(completed.items()):
        path = _unit_path(output, fixture_id)
        copied = export_json(path, raw)
        entries.append(
            {
                "id": fixture_id,
                "path": path.relative_to(output).as_posix(),
                "sha256": copied["sha256"],
                "size": copied["size"],
            }
        )
    checkpoint = inputs / "b8.pth"
    pending = [row for row in frozen if row["allocation_index"] not in completed]
    # Spawn rather than fork: the six host workers execute source CPU worlds
    # only, after the parent has completed the sole CUDA parity application.
    prior_collection_seconds = max(
        (float(raw["collection_elapsed_seconds"]) for raw in completed.values()), default=0.0
    )
    pool_started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for row, worker_result in zip(
            pending,
            pool.map(_measure_worker, pending, [str(checkpoint)] * len(pending)),
            strict=True,
        ):
            record = worker_result["record"]
            elapsed_seconds = worker_result["elapsed_seconds"]
            fixture_id = row["allocation_index"]
            record["fixture_descriptor_sha256"] = control["fixture_descriptor_sha256"]
            from research.bl_residual import f5_environment

            record["source_provenance"] = {
                "source_sha256": control["source_sha256"],
                "environment_sha256": f5_environment.receipt_sha256(actual_environment),
                "native_v6": True,
            }
            record["kernel_provenance"] = {
                "assignment_raw_sha256": f5_environment.receipt_sha256(actual_environment),
                "platform": actual_environment["platform"],
                "release": actual_environment["os_kernel"]["release"],
            }
            record["operational_payload_sha256"] = f5_timing_supplement.operational_payload_sha256(
                record
            )
            f5_timing_supplement.validate_operational_record(record)
            dispositions[fixture_id] = record["crossing"]
            path = _unit_path(output, fixture_id)
            exported = export_json(
                path,
                {
                    "schema": UNIT_SCHEMA,
                    "component_id": COMPONENT_ID,
                    "identity": _binding(control),
                    "fixture_id": fixture_id,
                    "record": record,
                    "elapsed_seconds": elapsed_seconds,
                    "collection_elapsed_seconds": (
                        prior_collection_seconds + time.monotonic() - pool_started
                    ),
                },
            )
            entries.append(
                {
                    "id": fixture_id,
                    "path": path.relative_to(output).as_posix(),
                    "sha256": exported["sha256"],
                    "size": exported["size"],
                }
            )
            # A later recovery gets a fresh output and never rewrites this ledger.
            _write_ledger(output / f"verified-ledger-{fixture_id}.json", control, entries)
    _summary(
        output,
        control,
        {entry["id"]: {"record": {"crossing": dispositions[entry["id"]]}} for entry in entries},
        role="TIMING_SUPPLEMENT",
    )
    _write_ledger(output / "verified-ledger-final.json", control, entries)
    return {
        "status": "MEASURED",
        "completed": len(entries),
        "ledger_successor": "verified-ledger-final.json",
    }


def _summary(
    output: Path, control: Mapping[str, Any], completed: Mapping[int, Any], *, role: str
) -> None:
    """Publish the same scanner-visible producer export in both final modes."""
    export_json(
        output / "supplement-summary.json",
        {
            "schema": "bl-f5-step3-supplement-summary/v1",
            "component_id": COMPONENT_ID,
            "identity": _binding(control),
            "role": role,
            "completed_fixture_ids": sorted(completed),
            "dispositions": {
                str(key): raw["record"]["crossing"] for key, raw in sorted(completed.items())
            },
            "fit_usable": False,
            "science_admission": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--family-canary", action="store_true")
    mode.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    selected = (
        "dry-run"
        if args.dry_run
        else "qualify" if args.qualify else "family-canary" if args.family_canary else "measure"
    )
    print(
        json.dumps(
            run(
                control_path=args.control,
                fixtures=args.fixtures,
                inputs=args.inputs,
                output=args.output,
                ledger=args.ledger,
                mode=selected,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
