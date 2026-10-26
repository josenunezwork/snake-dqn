"""Pure, sealed timing-qualification planning for selected F5 arm W.

This module deliberately has no runner, filesystem traversal, payload reader, or
dispatch integration.  It describes the pre-fit R8/R8b calibration contract and
checks a redacted operational receipt supplied by a later qualified runner.
Absent measurements are a HOLD; estimates are never manufactured here.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ALLOCATION_SHA256 = "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
SOURCE_BYTES = 45_397_643
UPLOAD_CHUNK_BYTES = 2 * 1024 * 1024
F5_TIMING_TRANSFER_SECONDS = 300
TIMING_GROUPS = ("watch6", "scripted", "frozen", "mixed")
SEMANTIC_COMMIT = "124dd22cd560a74a210a61f0ba473461328e82f8"


class TimingHold(ValueError):
    """A qualification refusal with no scientific value in its text."""


def _hold(reason: str) -> None:
    raise TimingHold("HOLD_TIMING_QUALIFICATION: " + reason)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        _hold(reason)


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _canonical_timing_parents() -> list[dict[str, Any]]:
    """The round-0 timing suffixes, without opening any allocation artifact."""
    parents: list[dict[str, Any]] = []
    allocation_index = 672
    for stream in TIMING_GROUPS:
        for group_index in range(3):
            parents.append(
                {
                    "allocation_index": allocation_index,
                    "group": f"timing/{stream}",
                    "group_index": group_index,
                    "parent_id": (
                        "bl-f5-predat-v2/20261025/r=0/" f"timing/{stream}/{group_index:04d}"
                    ),
                    "timing_only": True,
                }
            )
            allocation_index += 1
    return parents


def _timing_parents(allocation: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the only permitted timing parents from allocation metadata."""
    parents: list[dict[str, Any]] = []
    for row in allocation:
        if not isinstance(row, Mapping):
            _hold("malformed allocation row")
        if row.get("timing_only") is True:
            parents.append(dict(row))
    _require(len(parents) == 12, "requires exactly twelve timing-only parents")
    groups = Counter(row.get("group") for row in parents)
    _require(
        groups == Counter({f"timing/{group}": 3 for group in TIMING_GROUPS}),
        "requires three disjoint parents per timing stream",
    )
    ids = [row.get("parent_id") for row in parents]
    _require(all(isinstance(parent, str) and parent for parent in ids), "malformed timing parent")
    _require(len(ids) == len(set(ids)), "timing parents are not disjoint")
    canonical = _canonical_timing_parents()
    observed = {
        (
            row.get("allocation_index"),
            row.get("parent_id"),
            row.get("group"),
            row.get("group_index"),
        )
        for row in parents
    }
    expected = {
        (row["allocation_index"], row["parent_id"], row["group"], row["group_index"])
        for row in canonical
    }
    _require(observed == expected, "timing allocation identity drift")
    return sorted(parents, key=lambda row: int(row["allocation_index"]))


def build_plan(allocation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the immutable, measurement-free F5 timing qualification plan."""
    parents = _timing_parents(allocation)
    return {
        "schema": "bl-f5-sealed-timing-qualification/v1",
        "status": "PREPARED_NOT_MEASURED",
        "selection": {"arm": "W", "cov_selected": False, "allocation_sha256": ALLOCATION_SHA256},
        "source": {
            "private_input_bytes": SOURCE_BYTES,
            "semantic_commit": SEMANTIC_COMMIT,
            "forbidden_paths": ("D4096", "t34-round0-study-01"),
        },
        "timing_parents": parents,
        "calibration": {
            "sealed": True,
            "outcome_fields_permitted": False,
            "required_streams": TIMING_GROUPS,
            "minimum_per_stream": 3,
            "coverage": {
                "full_h300": True,
                "long_body": True,
                "full_h5000_natural": True,
                "watch_fixture_frames": 14_400,
                "early_only_refuses": True,
            },
            "per_worker_seconds": ("t_root", "t_branch", "t_nat"),
            "forecast_equations": {
                "archive_hours": "(128*t_nat + 14720*t_root + 88320*t_branch)/6 + setup_io_hours",
                "cov_hours": "(missing_roots*t_root + missing_actions*t_branch)/6 + setup_io_hours",
                "paired_fit_hours": (
                    "paired_fit_complete_seconds/3600 + " "setup_io_hours + imbalance_hours"
                ),
                "probe_hours": (
                    "96*max_complete_probe_seconds/2/3600 + " "setup_io_hours + imbalance_hours"
                ),
                "gameplay_hours": (
                    "1152*max_complete_gameplay_seconds/2/3600 + "
                    "setup_io_hours + imbalance_hours"
                ),
                "product_hours": (
                    "256*max_complete_product_seconds/2/3600 + " "setup_io_hours + imbalance_hours"
                ),
                "rule": "use maxima; add max-parent imbalance and measured setup/io",
            },
            "stage_workloads": {
                "paired_fit": 1,
                "probe": 96,
                "gameplay": 1152,
                "product": 256,
                "mac_workers": 2,
            },
        },
        "transport": {
            "upload_chunk_bytes": UPLOAD_CHUNK_BYTES,
            "upload_workers": 2,
            "export_readers": 1,
            "max_call_seconds": 60,
            "transfer_seconds_inclusive": 150,
            "staging_seconds_inclusive": 300,
        },
        "stage_envelopes_hours": {
            "paired_fit_including_canary": {"base": 2, "effective": 4},
            "cov_completion": {"base": 4, "effective": 8, "selected": False},
            "archive_retention": {"base": 8, "effective": 16},
            "mac_probe_reduction": {"base": 4, "effective": 8},
            "mac_gameplay": {"base": 12, "effective": 24},
            "mac_product": {"base": 12, "effective": 24},
            "canary_minutes": {"base": 20, "effective": 40},
        },
        "placement": {
            "fit": {
                "kind": "L4",
                "assignments": 1,
                "min_vcpu": 6,
                "cuda_processes": 1,
                "fp32": True,
            },
            "cpu_label_workers": 6,
            "mac_workers": 2,
            "mac_nice": 15,
            "math_threads": 1,
        },
        "admission": "HOLD until a complete operational-only proof validates",
    }


def qualification_plan(allocation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compatibility entry point for the F5 package builder."""
    return build_plan(allocation)


def _identity(proof: Mapping[str, Any], name: str, expected: Mapping[str, Any]) -> None:
    actual = proof.get(name)
    _require(isinstance(actual, Mapping), "missing " + name)
    for key, value in expected.items():
        _require(actual.get(key) == value, name + " drift")


def _saved_json(output: Path, name: str) -> Mapping[str, Any]:
    """Read one broker artifact without following a replacement symlink."""
    _require(output.is_dir() and not output.is_symlink(), "recovery output is absent or unsafe")
    path = output / name
    _require(path.is_file() and not path.is_symlink(), "saved " + name + " is absent or unsafe")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TimingHold("HOLD_TIMING_QUALIFICATION: saved " + name + " is malformed") from exc
    _require(isinstance(value, Mapping), "saved " + name + " is malformed")
    return value


def _validate_saved_recovery(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any], output: Path
) -> tuple[float, Mapping[str, float]]:
    """Bind a complete timing proof to immutable broker receipt and staging evidence."""
    from research.colab_cli.canary import CanaryError, StagingCapError, staging_measurement
    from research.colab_cli.transfer_policy import receipt_limit

    saved_receipt = _saved_json(output, "receipt.json")
    _require(saved_receipt == receipt, "saved receipt differs from supplied recovery receipt")
    terminal = receipt.get("terminal")
    _require(
        receipt.get("status") == "PASS"
        and receipt.get("stop_verified") is True
        and receipt.get("cleanup_error") is None
        and isinstance(terminal, Mapping)
        and terminal.get("returncode") == 0,
        "timing receipt is not a completed verified qualification",
    )
    try:
        transfer_bound = receipt_limit(dict(manifest), dict(receipt), output)
    except (OSError, TypeError, ValueError) as exc:
        raise TimingHold(
            "HOLD_TIMING_QUALIFICATION: timing receipt inventory authority differs"
        ) from exc
    measurement = _saved_json(output, "upload-measurement.json")
    try:
        measured = staging_measurement(dict(manifest), dict(measurement))
    except (CanaryError, StagingCapError, TypeError, ValueError) as exc:
        raise TimingHold("HOLD_TIMING_QUALIFICATION: saved staging measurement is invalid") from exc
    return transfer_bound, measured


def validate_operational_proof(
    proof: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    batch_identity: Mapping[str, Any],
    allocation: Sequence[Mapping[str, Any]] | None = None,
    recovery_manifest: Mapping[str, Any] | None = None,
    recovery_receipt: Mapping[str, Any] | None = None,
    recovery_output: Path | None = None,
) -> dict[str, Any]:
    """Accept only a complete redacted timing proof; otherwise raise ``TimingHold``.

    The proof is intentionally operational: durations, count/coverage flags and
    pinned identities.  Losses, Q values, labels, actions, episode scores and
    outcome payloads are prohibited.
    """
    _require(isinstance(proof, Mapping), "malformed proof")
    _require(proof.get("schema") == "bl-f5-timing-proof/v1", "wrong proof schema")
    _require(proof.get("status") == "COMPLETE", "measurements unavailable")
    _require(proof.get("allocation_sha256") == ALLOCATION_SHA256, "allocation hash drift")
    _identity(proof, "source_identity", source_identity)
    _identity(proof, "environment_identity", environment_identity)
    _identity(proof, "batch_identity", batch_identity)
    prohibited = {"loss", "losses", "q", "q_values", "labels", "outcomes", "actions", "scores"}
    _require(not (set(proof) & prohibited), "scientific payload in timing proof")
    transport = proof.get("transport")
    _require(isinstance(transport, Mapping), "missing transport proof")
    for key, expected in (
        ("upload_chunk_bytes", UPLOAD_CHUNK_BYTES),
        ("upload_workers", 2),
        ("export_readers", 1),
    ):
        _require(transport.get(key) == expected, "transport configuration drift")
    transfer_bound = 150
    authority = transport.get("f5_timing_transfer_exception")
    if authority is not None:
        from research.colab_cli.transfer_policy import f5_timing_authority_binding

        inventory_sha256 = (
            authority.get("inventory_sha256") if isinstance(authority, Mapping) else None
        )
        _require(isinstance(inventory_sha256, str), "timing transfer authority malformed")
        _require(
            authority == f5_timing_authority_binding(inventory_sha256, "timing"),
            "timing transfer authority drift",
        )
        _require(
            isinstance(recovery_manifest, Mapping)
            and isinstance(recovery_receipt, Mapping)
            and isinstance(recovery_output, Path),
            "timing allowance requires actual broker manifest and receipt",
        )
        _require(
            recovery_manifest.get("f5_timing_transfer_exception") == authority,
            "timing receipt authority differs",
        )
        _require(
            source_identity.get("commit") == recovery_manifest.get("commit"),
            "timing receipt source differs",
        )
        transfer_bound, measured_staging = _validate_saved_recovery(
            recovery_manifest, recovery_receipt, recovery_output
        )
        _require(
            transport.get("transfer_seconds") == measured_staging["transfer_seconds"]
            and transport.get("staging_seconds") == measured_staging["elapsed_seconds"],
            "proof transport does not match saved staging measurement",
        )
        bound_inventory = recovery_manifest["f5_timing_synthetic_inventory"]["inputs"]
        supplied_inventory = {
            row.get("name"): row
            for row in proof.get("input_inventory", [])
            if isinstance(row, Mapping)
        }
        _require(
            len(supplied_inventory) == len(bound_inventory)
            and all(
                supplied_inventory.get(row["dest"], {}).get("size") == row["size_bytes"]
                for row in bound_inventory
            ),
            "timing allowance inventory does not cover exact operation",
        )
    for key, bound in (
        ("max_call_seconds", 60),
        ("transfer_seconds", transfer_bound),
        ("staging_seconds", 300),
    ):
        value = transport.get(key)
        _require(_positive_number(value) and value <= bound, "transport timing refusal")
    private_bytes = transport.get("private_input_bytes")
    _require(
        isinstance(private_bytes, int)
        and not isinstance(private_bytes, bool)
        and 0 < private_bytes <= SOURCE_BYTES,
        "private input size exceeds corpus cap",
    )
    inventory = proof.get("input_inventory")
    _require(
        isinstance(inventory, Sequence) and not isinstance(inventory, (str, bytes)),
        "missing input inventory",
    )
    _require(len(inventory) > 0, "empty input inventory")
    private_total = 0
    names: set[str] = set()
    for item in inventory:
        _require(isinstance(item, Mapping), "malformed input inventory")
        name, size, scope = item.get("name"), item.get("size"), item.get("scope")
        _require(
            isinstance(name, str) and name and name not in names, "duplicate input inventory item"
        )
        names.add(name)
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0, "invalid input size"
        )
        if scope == "private":
            private_total += size
        elif scope == "public_b8":
            _require(item.get("release_qualified") is True, "unqualified public B8 input")
        else:
            _hold("unknown input scope")
    _require(private_total == private_bytes, "private inventory total drift")
    placement = proof.get("placement")
    _require(isinstance(placement, Mapping), "missing placement proof")
    _require(
        placement.get("l4_assignments") == 1
        and isinstance(placement.get("vcpu"), int)
        and not isinstance(placement.get("vcpu"), bool)
        and placement["vcpu"] >= 6,
        "L4 placement drift",
    )
    _require(
        placement.get("cuda_processes") == 1 and placement.get("precision") == "fp32",
        "fit placement drift",
    )
    _require(placement.get("cpu_label_workers") == 6, "CPU label placement drift")
    _require(
        placement.get("mac_workers") == 2 and placement.get("mac_nice") == 15, "Mac placement drift"
    )
    _require(placement.get("math_threads") == 1, "thread placement drift")
    samples = proof.get("samples")
    _require(
        isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)), "missing samples"
    )
    counts: Counter[str] = Counter()
    coverage = {
        stream: {
            "full_h300": False,
            "long_body": False,
            "full_h5000_natural": False,
            "watch_fixture": False,
        }
        for stream in TIMING_GROUPS
    }
    maxima = {"t_root": 0.0, "t_branch": 0.0, "t_nat": 0.0}
    stage_maxima = {"paired_fit": 0.0, "probe": 0.0, "gameplay": 0.0, "product": 0.0}
    expected_fixtures = (
        _timing_parents(allocation) if allocation is not None else _canonical_timing_parents()
    )
    expected_fixture_keys = {
        (row["allocation_index"], row["parent_id"]) for row in expected_fixtures
    }
    seen_fixtures: set[tuple[object, object]] = set()
    for sample in samples:
        _require(isinstance(sample, Mapping), "malformed timing sample")
        stream = sample.get("stream")
        _require(stream in TIMING_GROUPS, "unknown timing stream")
        fixture = (sample.get("allocation_index"), sample.get("parent_id"))
        _require(fixture in expected_fixture_keys, "science or foreign timing fixture")
        _require(fixture not in seen_fixtures, "duplicate timing fixture")
        seen_fixtures.add(fixture)
        _require(sample.get("timing_only") is True, "science fixture is forbidden")
        _require(
            sample.get("stream") == str(sample["parent_id"]).split("/")[-2], "fixture stream drift"
        )
        counts[str(stream)] += 1
        for key in maxima:
            value = sample.get(key)
            _require(_positive_number(value), "missing timing measurement")
            maxima[key] = max(maxima[key], float(value))
        for stage in stage_maxima:
            value = sample.get(stage + "_seconds")
            _require(_positive_number(value), "missing complete fixture stage measurement")
            stage_maxima[stage] = max(stage_maxima[stage], float(value))
        coverage[str(stream)]["full_h300"] |= sample.get("full_h300") is True
        coverage[str(stream)]["long_body"] |= sample.get("long_body") is True
        coverage[str(stream)]["full_h5000_natural"] |= sample.get("full_h5000_natural") is True
        coverage[str(stream)]["watch_fixture"] |= sample.get("watch_fixture_frames") == 14_400
    _require(all(counts[group] >= 3 for group in TIMING_GROUPS), "insufficient stream coverage")
    _require(seen_fixtures == expected_fixture_keys, "missing timing fixture")
    _require(
        all(
            requirement["full_h300"]
            and requirement["long_body"]
            and requirement["full_h5000_natural"]
            for requirement in coverage.values()
        )
        and coverage["watch6"]["watch_fixture"],
        "early-only or incomplete stream coverage",
    )
    forecast = proof.get("forecast")
    _require(isinstance(forecast, Mapping), "missing forecast")
    for key in ("setup_io_hours", "max_parent_imbalance_hours"):
        _require(_positive_number(forecast.get(key)), "missing conservative overhead")
    archive = (
        (128 * maxima["t_nat"] + 14_720 * maxima["t_root"] + 88_320 * maxima["t_branch"]) / 6 / 3600
    )
    archive += float(forecast["setup_io_hours"]) + float(forecast["max_parent_imbalance_hours"])
    _require(archive <= 8, "archive base envelope exceeded")
    stage_setup = forecast.get("stage_setup_io_hours")
    _require(isinstance(stage_setup, Mapping), "missing per-stage setup/io")
    stage_workloads = {"paired_fit": 1, "probe": 96, "gameplay": 1152, "product": 256}
    stage_workload_hours: dict[str, float] = {}
    stage_base_hours: dict[str, float] = {}
    for stage, ceiling in (("paired_fit", 2), ("probe", 4), ("gameplay", 12), ("product", 12)):
        overhead = stage_setup.get(stage)
        _require(_positive_number(overhead), "missing conservative stage overhead")
        divisor = 1 if stage == "paired_fit" else 2
        stage_workload_hours[stage] = stage_workloads[stage] * stage_maxima[stage] / divisor / 3600
        stage_base_hours[stage] = (
            stage_workload_hours[stage]
            + float(overhead)
            + float(forecast["max_parent_imbalance_hours"])
        )
        _require(stage_base_hours[stage] <= ceiling, "stage base envelope exceeded")
    return {
        "status": "QUALIFIED",
        "max_seconds": maxima,
        "archive_base_hours": archive,
        "stage_workload_hours": stage_workload_hours,
        "stage_base_hours": stage_base_hours,
    }
