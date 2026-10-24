"""Outcome-blind W12 operational guard and parent-throughput helpers.

This module reads only immutable attempt wrappers, T-EQ timing metadata, and
frozen allocation identities.  It deliberately never opens F0 label payloads
or rate-sealed outcome diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .exports import _verify, publish_json

W6 = 6
W12 = 12
THERMAL_CLEAN_SECONDS = 24 * 60 * 60
W12_SANITY_SECONDS = 3 * 60 * 60
W12_SANITY_MULTIPLIER = 1.15
W12_RATE_DRIFT_MULTIPLIER = 1.36


def _read_triplet(path: Path) -> dict[str, Any]:
    """Read a completion-last JSON triplet, rejecting non-object documents."""
    _verify(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid sealed JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"sealed JSON must be an object: {path}")
    return value


def _finite_positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} is not finite")
    if value <= 0:
        raise ValueError(f"{what} must be positive")
    return float(value)


def _attempt_pairs(state_dir: Path) -> list[tuple[Path, dict[str, Any], Path, dict[str, Any]]]:
    """Verify the complete immutable start/terminal sequence without outcomes."""
    starts = sorted(state_dir.glob("attempt-*.start.json"), key=lambda p: p.stat().st_mtime_ns)
    terminals = sorted(
        state_dir.glob("attempt-*.terminal.json"), key=lambda p: p.stat().st_mtime_ns
    )
    if {p.name.replace(".start.json", "") for p in starts} != {
        p.name.replace(".terminal.json", "") for p in terminals
    }:
        raise ValueError("W12 resume state has an unverified attempt hole; HOLD")
    pairs = []
    for start_path in starts:
        terminal_path = start_path.with_name(
            start_path.name.replace(".start.json", ".terminal.json")
        )
        start, terminal = _read_triplet(start_path), _read_triplet(terminal_path)
        if terminal.get("start_sha256") != _verify(start_path):
            raise ValueError("W12 terminal start binding differs; HOLD")
        if terminal.get("workers") != start.get("workers"):
            raise ValueError("W12 terminal worker width differs; HOLD")
        _finite_positive(terminal.get("elapsed_seconds"), "attempt elapsed_seconds")
        pairs.append((start_path, start, terminal_path, terminal))
    return pairs


def _fallback_reason(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("reason"), str):
        return value["reason"]
    return None


def _canonical_series_state_dirs(state_dir: Path) -> list[Path]:
    """Find sibling canonical state directories without following arbitrary paths."""
    parent = state_dir.resolve().parent
    found = {state_dir.resolve()}
    for candidate in parent.glob("*state*"):
        if candidate.is_dir():
            found.add(candidate.resolve())
    # The only production state roots authorized by the ruling share one
    # artifact parent.  This closes a cross-dose thermal bypass while keeping
    # synthetic test state directories local.
    markers = {"bl-labels-f0", "bl-labels-f0-speed"}
    parts = state_dir.resolve().parts
    marker_index = next((index for index, part in enumerate(parts) if part in markers), None)
    if marker_index is not None:
        artifact_parent = Path(*parts[:marker_index])
        artifact_root = artifact_parent / "bl-labels-f0"
        roots = (
            artifact_root / "mac-pack-02",
            artifact_parent / "bl-labels-f0-speed" / "d1024-d4096-pack-01",
        )
        for root in roots:
            if root.is_dir():
                for candidate in root.rglob("*state*"):
                    if candidate.is_dir():
                        found.add(candidate.resolve())
    return sorted(found)


def _clean_thermal_interval(value: object) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("W12 thermal clean interval is malformed; HOLD")
    start = _finite_positive(value[0], "thermal clean interval start")
    end = _finite_positive(value[1], "thermal clean interval end")
    if end < start:
        raise ValueError("W12 thermal clean interval reverses time; HOLD")
    return start, end


def resolve_resume_workers(
    requested_workers: int, state_dir: Path, *, now: float | None = None
) -> tuple[int, list[str]]:
    """Return requested width unless sealed operational evidence requires W6.

    RSS and sanity fallbacks are sticky for the series.  A thermal trip blocks
    W12 until a full 24 hours after the latest sealed trip.  Unknown receipts
    and incomplete attempt pairs fail closed rather than silently selecting W6.
    """
    if requested_workers not in (W6, W12):
        raise ValueError("W12 runtime accepts only W6 or W12")
    pairs = _attempt_pairs(state_dir)
    # The slowdown ruling is independent from the thermal statistic but uses
    # the same immutable attempt chain.  Its W6 host-check HOLD must happen
    # before this function permits any continuation at either width.
    from . import slowdown_resume

    slowdown_workers, slowdown_reasons = slowdown_resume.resume_policy(
        requested_workers, state_dir, now=now
    )
    if requested_workers == W6:
        return slowdown_workers, slowdown_reasons
    sticky, thermal_epochs, clean_intervals = [], [], []
    for _start_path, _start, _terminal_path, terminal in pairs:
        reason = _fallback_reason(terminal.get("operational_width_fallback"))
        if reason in {"rss", "sanity"}:
            sticky.append(reason)
        epochs = terminal.get("thermal_trip_epochs", [])
        if not isinstance(epochs, list):
            raise ValueError("W12 thermal trip history is malformed; HOLD")
        for epoch in epochs:
            thermal_epochs.append(_finite_positive(epoch, "thermal trip epoch"))
    # Thermal return is factory-wide: a new future-dose state directory may
    # not bypass a trip in its canonical sibling series.  RSS/sanity remains
    # sticky only for the resumed state whose operational fallback recorded it.
    slowdown_epochs = []
    for related in _canonical_series_state_dirs(state_dir):
        if related == state_dir.resolve():
            related_pairs = pairs
        else:
            related_pairs = _attempt_pairs(related)
        for _start_path, _start, _terminal_path, terminal in related_pairs:
            for epoch in terminal.get("thermal_trip_epochs", []):
                thermal_epochs.append(_finite_positive(epoch, "thermal trip epoch"))
            intervals = terminal.get("thermal_clean_intervals", [])
            if not isinstance(intervals, list):
                raise ValueError("W12 thermal clean history is malformed; HOLD")
            clean_intervals.extend(_clean_thermal_interval(interval) for interval in intervals)
        # Like thermal, a qualifying W12 slowdown pair belongs to the factory
        # series.  A later D1024/D4096 state directory cannot bypass D256's
        # sealed cooldown by changing roots.
        epoch = slowdown_resume.escalation(related)["w12_epoch"]
        if epoch is not None:
            slowdown_epochs.append(float(epoch))
    if sticky:
        return W6, [f"operational_width_fallback:{reason}" for reason in sorted(set(sticky))]
    # A qualifying pair of W12 slowdown trips has the same 24-hour clean
    # return rule as a thermal trip.  Add its epoch to the shared evidence
    # path so an isolated W12 cooldown cannot bypass the verified clean proof.
    thermal_epochs.extend(slowdown_epochs)
    if thermal_epochs:
        current = time.time() if now is None else _finite_positive(now, "current epoch")
        latest = max(thermal_epochs)
        clean = any(
            start >= latest and end <= current and end - start >= THERMAL_CLEAN_SECONDS
            for start, end in clean_intervals
        )
        if current < latest + THERMAL_CLEAN_SECONDS or not clean:
            return W6, ["thermal_clean_window_incomplete"]
    return slowdown_workers, slowdown_reasons


def _provenance(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _verify(path)}


def d256_w6_parent_rates(control: dict[str, Any], pack: Path) -> dict[str, Any]:
    """Derive W6 completed-parent wall rates from the verified D256 attempt chain."""
    # Import lazily: d256_resume is the authoritative chain verifier and may
    # itself import this module from launcher integration.
    from . import d256_resume

    attempts, _elapsed = d256_resume.attempt_chain(pack, control)
    streams = {
        item["allocation_id"]: item.get("stream")
        for item in control.get("allocation", [])
        if isinstance(item, dict) and isinstance(item.get("allocation_id"), str)
    }
    if not streams or any(not isinstance(stream, str) or not stream for stream in streams.values()):
        raise ValueError("D256 allocation stream identity is malformed; HOLD")
    counts_by_stream: dict[str, int] = {stream: 0 for stream in set(streams.values())}
    elapsed_total = 0.0
    completed_total = 0
    evidence = []
    for output in attempts:
        terminal_path = output / "runtime-result.json"
        terminal = _read_triplet(terminal_path)
        if terminal.get("workers") != W6:
            continue
        charge = _finite_positive(terminal.get("elapsed_seconds"), "D256 W6 elapsed_seconds")
        completed = terminal.get("completed_parents")
        if not isinstance(completed, list) or len(completed) != len(set(completed)):
            raise ValueError("D256 completed parent metadata is malformed; HOLD")
        for parent in completed:
            if parent not in streams:
                raise ValueError("D256 completed parent is foreign; HOLD")
            stream = streams[parent]
            counts_by_stream[stream] += 1
            completed_total += 1
        elapsed_total += charge
        evidence.append(_provenance(terminal_path))
    if completed_total <= 0 or elapsed_total <= 0:
        raise ValueError("D256 has no verified W6 completed-parent baseline; HOLD")
    return {
        "schema": "bl-labels-f0-w12-w6-parent-rate/v1",
        "source": "d256-w6-attempts",
        "rates_per_wall_hour": {"all": completed_total * 3600.0 / elapsed_total},
        "completed_parents": {"all": completed_total},
        "elapsed_seconds": {"all": elapsed_total},
        "stream_diagnostics": {"completed_parents": counts_by_stream},
        "provenance": evidence,
    }


def future_w6_parent_rates(control: dict[str, Any], teq_receipt: Path) -> dict[str, Any]:
    """Derive a future-pin W6 baseline from sealed T-EQ timing/phase metadata."""
    receipt = _read_triplet(teq_receipt)
    pin = control.get("source_commit")
    if not isinstance(pin, str) or receipt.get("new", {}).get("commit") != pin:
        raise ValueError("T-EQ receipt does not bind the future production pin; HOLD")
    table_path = teq_receipt.parent / "timing-table.json"
    table = _read_triplet(table_path)
    if receipt.get("timing_table_sha256") != _verify(table_path):
        raise ValueError("T-EQ timing table binding differs; HOLD")
    cells = table.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("T-EQ timing table is missing W6 cells; HOLD")
    bank = receipt.get("bank", {})
    bank_path = Path(bank.get("path", "")) if isinstance(bank, dict) else Path()
    if not bank_path.is_file() or _verify(bank_path) != bank.get("sha256"):
        raise ValueError("T-EQ B-EQ bank binding differs; HOLD")
    bank_value = _read_triplet(bank_path)
    bank_allocation = bank_value.get("allocation")
    if not isinstance(bank_allocation, list) or not bank_allocation:
        raise ValueError("T-EQ B-EQ allocation is malformed; HOLD")
    streams_by_index = []
    for item in bank_allocation:
        stream = item.get("stream") if isinstance(item, dict) else None
        if not isinstance(stream, str) or not stream:
            raise ValueError("T-EQ B-EQ stream identity is malformed; HOLD")
        streams_by_index.append(stream)
    counts_by_stream: dict[str, int] = {stream: 0 for stream in set(streams_by_index)}
    total_parents = 0
    total_elapsed = 0.0
    phase_provenance = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("pin") != pin or cell.get("workers") != W6:
            continue
        parents = cell.get("parents")
        seconds = cell.get("elapsed_seconds")
        if type(parents) is not int or parents <= 0 or parents > len(streams_by_index):
            raise ValueError("T-EQ parent count is malformed; HOLD")
        order = cell.get("order")
        if type(order) is not int or order < 0:
            raise ValueError("T-EQ timing cell order is malformed; HOLD")
        phase_path = teq_receipt.parent / f"timing-{order}" / "phase-ledger.json"
        phase = _read_triplet(phase_path)
        indexes = list(range(parents))
        if (
            phase.get("pin") != pin
            or phase.get("workers") != W6
            or phase.get("parents") is not None
            or phase.get("parent_indexes") != indexes
            or _finite_positive(phase.get("elapsed_seconds"), "T-EQ phase elapsed_seconds")
            != _finite_positive(seconds, "T-EQ W6 elapsed_seconds")
        ):
            raise ValueError("T-EQ W6 phase ledger differs from timing table; HOLD")
        phase_elapsed = _finite_positive(seconds, "T-EQ W6 elapsed_seconds")
        for index in indexes:
            stream = streams_by_index[index]
            counts_by_stream[stream] += 1
            total_parents += 1
        total_elapsed += phase_elapsed
        phase_provenance.append(_provenance(phase_path))
    if total_parents <= 0 or total_elapsed <= 0:
        raise ValueError("T-EQ has no matching future W6 timing baseline; HOLD")
    return {
        "schema": "bl-labels-f0-w12-w6-parent-rate/v1",
        "source": "teq-w6-timing-table",
        "rates_per_wall_hour": {"all": total_parents * 3600.0 / total_elapsed},
        "completed_parents": {"all": total_parents},
        "elapsed_seconds": {"all": total_elapsed},
        "stream_diagnostics": {"completed_parents": counts_by_stream},
        "provenance": [
            _provenance(teq_receipt),
            _provenance(table_path),
            _provenance(bank_path),
            *phase_provenance,
        ],
    }


def d4096_old_w6_parent_rates(control: dict[str, Any]) -> dict[str, Any]:
    """Load the sealed D1024 OLD W6 same-stream throughput comparator.

    D4096's three-hour W12 check is deliberately not allowed to fall back to
    the D256 rate, a Watch6 prose rate, or the Canary-B rate.  The readiness
    packet must instead bind an immutable, post-terminal D1024 metadata
    receipt with actual completed-parent and elapsed-wall exposure for every
    stream in the D4096 allocation.  This function only reads that receipt;
    it never locates or opens a live D1024 state/output directory.
    """
    gate = control.get("mac_gate")
    if not isinstance(gate, dict) or gate.get("stage") != "d4096":
        raise ValueError("D4096 OLD W6 comparator requires D4096 admission; HOLD")
    bindings = gate.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("D4096 lacks sealed collection readiness binding; HOLD")
    sealed_collection = bindings.get("sealed_collection")
    if not isinstance(sealed_collection, dict):
        raise ValueError("D4096 lacks sealed collection readiness binding; HOLD")
    readiness_binding = sealed_collection.get("readiness")
    if not isinstance(readiness_binding, dict):
        raise ValueError("D4096 sealed readiness binding is malformed; HOLD")
    readiness_path = readiness_binding.get("path")
    readiness_sha = readiness_binding.get("sha256")
    if not isinstance(readiness_path, str) or not isinstance(readiness_sha, str):
        raise ValueError("D4096 sealed readiness binding is malformed; HOLD")
    readiness = _read_triplet(Path(readiness_path))
    if _verify(Path(readiness_path)) != readiness_sha:
        raise ValueError("D4096 sealed readiness binding changed; HOLD")
    metadata_binding = readiness.get("d1024_w6_metadata")
    if not isinstance(metadata_binding, dict):
        raise ValueError("D4096 readiness lacks D1024 W6 metadata binding; HOLD")
    metadata_path = metadata_binding.get("path")
    metadata_sha = metadata_binding.get("sha256")
    if not isinstance(metadata_path, str) or not isinstance(metadata_sha, str):
        raise ValueError("D4096 D1024 W6 metadata binding is malformed; HOLD")
    metadata_file = Path(metadata_path)
    metadata = _read_triplet(metadata_file)
    if _verify(metadata_file) != metadata_sha:
        raise ValueError("D4096 D1024 W6 metadata binding changed; HOLD")
    if (
        metadata.get("schema") != "bl-labels-f0-d1024-old-w6-throughput/v1"
        or metadata.get("stage") != "d1024"
        or metadata.get("workers") != W6
        or metadata.get("source_commit") != control.get("source_commit")
    ):
        raise ValueError("D4096 D1024 W6 metadata identity differs; HOLD")
    terminal_binding = metadata.get("d1024_terminal")
    if not isinstance(terminal_binding, dict):
        raise ValueError("D4096 D1024 metadata lacks terminal binding; HOLD")
    terminal_path = terminal_binding.get("path")
    terminal_sha = terminal_binding.get("sha256")
    if not isinstance(terminal_path, str) or not isinstance(terminal_sha, str):
        raise ValueError("D4096 D1024 terminal binding is malformed; HOLD")
    # This is only a post-terminal identity proof.  Counts and elapsed exposure
    # remain the metadata receipt's explicit, stream-specific fields.
    terminal_file = Path(terminal_path)
    terminal = _read_triplet(terminal_file)
    if _verify(terminal_file) != terminal_sha:
        raise ValueError("D4096 D1024 terminal binding changed; HOLD")
    if terminal.get("status") != "COMPLETE":
        raise ValueError("D4096 D1024 terminal is not COMPLETE; HOLD")
    allocation = control.get("allocation")
    if not isinstance(allocation, list) or not allocation:
        raise ValueError("D4096 allocation is malformed; HOLD")
    if any(
        not isinstance(row, dict) or not isinstance(row.get("stream"), str) or not row["stream"]
        for row in allocation
    ):
        raise ValueError("D4096 allocation stream identity is malformed; HOLD")
    required_streams = {row["stream"] for row in allocation}
    streams = metadata.get("streams")
    if not isinstance(streams, dict) or set(streams) != required_streams:
        raise ValueError("D4096 D1024 metadata does not cover exactly its streams; HOLD")
    rates: dict[str, float] = {}
    completed: dict[str, int] = {}
    elapsed: dict[str, float] = {}
    for stream in sorted(required_streams):
        item = streams.get(stream)
        if not isinstance(item, dict):
            raise ValueError("D4096 D1024 stream metadata is malformed; HOLD")
        count = item.get("completed_parents")
        if type(count) is not int or count <= 0:
            raise ValueError("D4096 D1024 stream completion count is malformed; HOLD")
        seconds = _finite_positive(item.get("elapsed_seconds"), "D1024 stream elapsed_seconds")
        completed[stream] = count
        elapsed[stream] = seconds
        rates[stream] = count * 3600.0 / seconds
    return {
        "schema": "bl-labels-f0-w12-w6-parent-rate/v1",
        "source": "d1024-old-w6-sealed-metadata",
        "rates_per_wall_hour": rates,
        "completed_parents": completed,
        "elapsed_seconds": elapsed,
        "stream_diagnostics": {"completed_parents": completed},
        "provenance": [
            _provenance(Path(readiness_path)),
            _provenance(metadata_file),
            _provenance(terminal_file),
        ],
    }


def seal_d1024_old_w6_throughput(
    d1024_control: Path, terminals: list[Path], out: Path | None
) -> dict[str, Any]:
    """Publish the metadata-only D1024 OLD W6 comparator for D4096.

    ``terminals`` must be the ordered, complete D1024 attempt chain.  The
    builder reads only start/terminal receipts and the final operational
    slowdown snapshots: completed parent IDs, allocation stream membership,
    and active-stream wall exposure.  It intentionally never reads an output
    directory, ledger unit, life export, or outcome-bearing payload.
    """
    if out is not None and any(
        Path(str(out) + suffix).exists() for suffix in ("", ".sha256", ".complete")
    ):
        raise FileExistsError("D1024 throughput receipt is create-only")
    control = _read_triplet(d1024_control)
    gate = control.get("mac_gate")
    source_commit = control.get("source_commit")
    allocation = control.get("allocation")
    if (
        not isinstance(gate, dict)
        or gate.get("stage") != "d1024"
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or not isinstance(allocation, list)
        or not allocation
    ):
        raise ValueError("D1024 control identity is malformed; HOLD")
    parent_streams: dict[str, str] = {}
    for row in allocation:
        parent = row.get("allocation_id") if isinstance(row, dict) else None
        stream = row.get("stream") if isinstance(row, dict) else None
        if (
            not isinstance(parent, str)
            or not parent
            or not isinstance(stream, str)
            or not stream
            or parent in parent_streams
        ):
            raise ValueError("D1024 allocation stream identity is malformed; HOLD")
        parent_streams[parent] = stream
    if not terminals:
        raise ValueError("D1024 terminal chain is empty; HOLD")
    from . import frozen_factory

    seen_parents: set[str] = set()
    terminal_values: list[tuple[Path, dict[str, Any]]] = []
    names = [path.name for path in terminals]
    for index, terminal_path in enumerate(terminals):
        terminal = _read_triplet(terminal_path)
        if terminal.get("schema") != "bl-labels-f0-mac-driver/v1" or terminal.get(
            "namespace"
        ) != control.get("namespace"):
            raise ValueError("D1024 terminal namespace differs; HOLD")
        if (
            terminal.get("workers") != W6
            or terminal.get("collector_source_commit") != frozen_factory.OLD_PIN
            or terminal.get("w12_calibration_source_commit") != frozen_factory.OLD_PIN
        ):
            raise ValueError("D1024 terminal OLD W6 identity differs; HOLD")
        expected_status = "COMPLETE" if index == len(terminals) - 1 else "PREEMPTED"
        if terminal.get("status") != expected_status:
            raise ValueError("D1024 terminal sequence is not completed in order; HOLD")
        if terminal.get("attempts_ledger") != names[: index + 1]:
            raise ValueError("D1024 terminal attempt ledger differs; HOLD")
        start_name = terminal.get("start_receipt")
        if not isinstance(start_name, str) or Path(start_name).name != start_name:
            raise ValueError("D1024 terminal start receipt is malformed; HOLD")
        start_path = terminal_path.parent / start_name
        start = _read_triplet(start_path)
        if (
            terminal.get("start_sha256") != _verify(start_path)
            or start.get("namespace") != control.get("namespace")
            or start.get("source_commit") != source_commit
            or start.get("workers") != W6
        ):
            raise ValueError("D1024 terminal start binding differs; HOLD")
        completed = terminal.get("completed_parents")
        if not isinstance(completed, list) or len(completed) != len(set(completed)):
            raise ValueError("D1024 completed-parent metadata is malformed; HOLD")
        for parent in completed:
            if parent not in parent_streams or parent in seen_parents:
                raise ValueError("D1024 completed parent is foreign or duplicated; HOLD")
            seen_parents.add(parent)
        terminal_values.append((terminal_path, terminal))
    if seen_parents != set(parent_streams):
        raise ValueError("D1024 terminal chain is not complete; HOLD")
    final_path, final = terminal_values[-1]
    slowdown = final.get("slowdown_state_by_stream")
    streams = set(parent_streams.values())
    if not isinstance(slowdown, dict) or set(slowdown) != streams:
        raise ValueError("D1024 final stream exposure differs; HOLD")
    result_streams: dict[str, dict[str, float | int]] = {}
    for stream in sorted(streams):
        snapshot = slowdown.get(stream)
        raw_walls = snapshot.get("width_wall_seconds") if isinstance(snapshot, dict) else None
        if not isinstance(raw_walls, dict):
            raise ValueError("D1024 stream wall exposure is malformed; HOLD")
        elapsed = _finite_positive(raw_walls.get(str(W6)), "D1024 W6 stream elapsed_seconds")
        completed = sum(1 for value in seen_parents if parent_streams[value] == stream)
        if completed <= 0:
            raise ValueError("D1024 stream has no completed parents; HOLD")
        result_streams[stream] = {
            "completed_parents": completed,
            "elapsed_seconds": elapsed,
        }
    value = {
        "schema": "bl-labels-f0-d1024-old-w6-throughput/v1",
        "stage": "d1024",
        "workers": W6,
        "source_commit": source_commit,
        "d1024_control": _provenance(d1024_control),
        "d1024_terminal": _provenance(final_path),
        "terminal_chain": [_provenance(path) for path, _value in terminal_values],
        "streams": result_streams,
    }
    if out is not None:
        publish_json(out, value)
    return value


def verify_d1024_old_w6_throughput(path: Path) -> dict[str, Any]:
    """Recompute a receipt from its complete immutable metadata chain."""
    value = _read_triplet(path)
    control_binding = value.get("d1024_control")
    terminal_chain = value.get("terminal_chain")
    if not isinstance(control_binding, dict) or not isinstance(terminal_chain, list):
        raise ValueError("D1024 W6 throughput lacks verified source metadata")
    bindings = [control_binding, *terminal_chain]
    for binding in bindings:
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or _verify(Path(binding["path"])) != binding["sha256"]
        ):
            raise ValueError("D1024 W6 throughput source binding differs")
    expected = seal_d1024_old_w6_throughput(
        Path(control_binding["path"]), [Path(row["path"]) for row in terminal_chain], None
    )
    if value != expected:
        raise ValueError("D1024 W6 throughput differs from recomputed terminal metadata")
    return value


def resolve_w6_parent_rates(
    control: dict[str, Any], *, d256_pack: Path | None = None, teq_receipt: Path | None = None
) -> dict[str, Any]:
    """Select the binding W6 baseline source from the stage, failing closed."""
    stage = control.get("mac_gate", {}).get("stage")
    if stage == "d256":
        if d256_pack is None:
            package_root = control.get("mac_gate", {}).get("package_root")
            if isinstance(package_root, str):
                d256_pack = Path(package_root)
        if d256_pack is None:
            raise ValueError("D256 W12 needs an original W6 attempt baseline; HOLD")
        return d256_w6_parent_rates(control, d256_pack)
    if stage == "d4096":
        return d4096_old_w6_parent_rates(control)
    if stage == "d1024":
        from . import d256_resume, w12_calibration

        if control.get("source_commit") == d256_resume.OLD_PIN:
            pack = d256_pack or w12_calibration.PACK
            return d256_w6_parent_rates(d256_resume.original_control(pack), pack)
        if teq_receipt is None:
            binding = control.get("mac_gate", {}).get("bindings", {}).get("speed_equivalence", {})
            source = binding.get("path") if isinstance(binding, dict) else None
            if isinstance(source, str):
                teq_receipt = Path(source)
        if teq_receipt is None:
            raise ValueError("future W12 needs a sealed T-EQ W6 timing baseline; HOLD")
        return future_w6_parent_rates(control, teq_receipt)
    raise ValueError("W12 parent baseline stage is not authorized; HOLD")


def w12_sanity_decision(
    completed_parents: Mapping[str, int],
    elapsed_seconds: float,
    w6_rates: Mapping[str, float],
    *,
    stream_elapsed_seconds: Mapping[str, float] | None = None,
) -> tuple[bool, dict[str, float]]:
    """Evaluate the one three-hour W12 wall-throughput guard per stream.

    D1024 retains the historical all-stream rule when no exposure map is
    supplied.  D4096 may supply its sealed admission exposure map because its
    frozen allocation can spend the opening interval on a single stream.  An
    unadmitted stream is then omitted rather than treated as a zero-rate
    failure; every admitted stream has its own matched wall denominator.
    """
    elapsed = _finite_positive(elapsed_seconds, "W12 sanity elapsed_seconds")
    if elapsed < W12_SANITY_SECONDS:
        return True, {}
    exposure: dict[str, float] | None = None
    if stream_elapsed_seconds is not None:
        if not isinstance(stream_elapsed_seconds, Mapping):
            raise ValueError("W12 stream exposure is malformed; HOLD")
        if set(stream_elapsed_seconds) != set(w6_rates):
            raise ValueError("W12 stream exposure does not match baseline streams; HOLD")
        exposure = {}
        for stream, value in stream_elapsed_seconds.items():
            if not isinstance(stream, str) or not stream:
                raise ValueError("W12 stream exposure name is malformed; HOLD")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("W12 stream exposure is malformed; HOLD")
            if value < 0 or value > elapsed:
                raise ValueError("W12 stream exposure is outside the sanity window; HOLD")
            exposure[stream] = float(value)
        if not any(exposure.values()):
            raise ValueError("W12 sanity has no admitted stream exposure; HOLD")
    actual = {}
    for stream, baseline in w6_rates.items():
        if not isinstance(stream, str) or not stream:
            raise ValueError("W6 stream name is malformed; HOLD")
        base = _finite_positive(baseline, "W6 parent rate")
        count = completed_parents.get(stream)
        if type(count) is not int or count < 0:
            raise ValueError("W12 completed parent count is malformed; HOLD")
        denominator = elapsed if exposure is None else exposure[stream]
        if denominator == 0:
            if count != 0:
                raise ValueError("W12 completion has no matching stream exposure; HOLD")
            continue
        actual[stream] = count * 3600.0 / denominator
        if actual[stream] < W12_SANITY_MULTIPLIER * base:
            return False, actual
    return True, actual


def rate_drift_baseline(canary_b_rate: float) -> float:
    """Apply the ruling's width-normalized Canary-B baseline exactly once."""
    return W12_RATE_DRIFT_MULTIPLIER * _finite_positive(canary_b_rate, "Canary-B rate")


def main() -> None:
    """Create the sealed, metadata-only D1024 OLD W6 throughput receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1024-control", required=True, type=Path)
    parser.add_argument("--terminal", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    seal_d1024_old_w6_throughput(args.d1024_control, args.terminal, args.out)


if __name__ == "__main__":
    main()
