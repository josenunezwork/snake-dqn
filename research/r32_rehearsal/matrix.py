"""Portable Codex-V lifecycle matrix for R32.

The matrix exercises the isolated lifecycle adapter with temporary exports. It
does not open scientific payloads and it has no dispatch capability.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any, Callable

from research.bl_ppo.exports import digest
from research.r32_rehearsal import lifecycle


def _spec(namespace: str = "r32-synthetic-matrix") -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    source = repo / "research/r32_rehearsal/lifecycle.py"
    return {
        "synthetic": True,
        "namespace": namespace,
        "units": [{"index": index, "name": f"unit-{index:04d}.json"} for index in range(4)],
        "sources": {str(source.relative_to(repo)): digest(source)},
        "scientific_stop_required": [0, 1],
    }


def _expect_error(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except (OSError, ValueError, FileExistsError) as exc:
        return type(exc).__name__
    raise AssertionError("fault unexpectedly accepted")


def _publish_all(root: Path, spec: dict[str, Any], indexes: list[int]) -> None:
    for index in indexes:
        lifecycle.publish_unit(root, spec, index, {"index": index, "synthetic": True})


def run_matrix(output: Path) -> dict[str, Any]:
    """Run the three positive traversals and twelve independent fault cases."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    lifecycle.check_sources(spec, Path(__file__).resolve().parents[2])

    # Corrupt exports are deliberately confined to a disposable fixture tree;
    # the returned summary is the only durable runtime output.
    with tempfile.TemporaryDirectory(prefix="r32-lifecycle-") as fixture_dir:
        root = Path(fixture_dir)
        clean = root / "positive-clean"
        lifecycle.start_attempt(clean, spec)
        _publish_all(clean, spec, [0, 1, 2, 3])
        clean_ledger = lifecycle.build_ledger([clean], spec, root / "clean-ledger.json")
        clean_status = lifecycle.disposition(
            spec, clean_ledger, stop_verified=True, outcome="clean"
        )

        before = root / "positive-before-marker"
        lifecycle.start_attempt(before, spec)
        # A payload written before its sidecar/marker is not a verified unit.
        (before / "unit-0000.json").write_text('{"pending":true}\n')
        lifecycle.publish_unit(before, spec, 1, {"index": 1, "synthetic": True})
        after = root / "positive-after-marker"
        lifecycle.start_attempt(after, spec)
        lifecycle.publish_unit(after, spec, 0, {"index": 0, "synthetic": True})
        after_marker_ledger = lifecycle.build_ledger(
            [before, after], spec, root / "after-marker-ledger.json"
        )
        if sorted(
            after_marker_ledger["units"][i]["index"]
            for i in range(len(after_marker_ledger["units"]))
        ) != [0, 1]:
            raise AssertionError("post-marker interruption did not preserve the completed unit")
        _publish_all(after, spec, [2, 3])
        resumed_ledger = lifecycle.build_ledger([before, after], spec, root / "resumed-ledger.json")
        resumed_status = lifecycle.disposition(
            spec, resumed_ledger, stop_verified=True, outcome="mechanical"
        )
        if resumed_status["verified"] != clean_status["verified"]:
            raise AssertionError("mechanical resume changed the eligible completed-unit union")

        scientific = root / "positive-scientific-stop"
        lifecycle.start_attempt(scientific, spec)
        _publish_all(scientific, spec, [0, 1])
        scientific_ledger = lifecycle.build_ledger(
            [scientific], spec, root / "scientific-ledger.json"
        )
        scientific_status = lifecycle.disposition(
            spec, scientific_ledger, stop_verified=True, outcome="scientific_stop"
        )
        if scientific_status != {
            "status": "COMPLETE",
            "verified": [0, 1],
            "missing": [],
            "not_run_by_rule": [2, 3],
            "replacement_allowed": False,
            "scientific_result": False,
        }:
            raise AssertionError(
                "scientific stop did not preserve its exact downstream NOT_RUN set"
            )

        faults: dict[str, dict[str, Any]] = {}

        wrong_sidecar = root / "fault-wrong-sidecar"
        lifecycle.start_attempt(wrong_sidecar, spec)
        lifecycle.publish_unit(wrong_sidecar, spec, 0, {"fault": 1})
        (wrong_sidecar / "unit-0000.json.sha256").rename(wrong_sidecar / "wrong.sha256")
        faults["wrong_sha_sidecar_basename"] = {
            "classification": "REFUSE",
            "error": _expect_error(lambda: lifecycle.build_ledger([wrong_sidecar], spec)),
        }

        truncated = root / "fault-truncated"
        lifecycle.start_attempt(truncated, spec)
        lifecycle.publish_unit(truncated, spec, 0, {"fault": 2})
        (truncated / "unit-0000.json").write_text("{}\n")
        faults["truncated_payload"] = {
            "classification": "REFUSE",
            "error": _expect_error(lambda: lifecycle.build_ledger([truncated], spec)),
        }

        early_marker = root / "fault-marker-first"
        lifecycle.start_attempt(early_marker, spec)
        (early_marker / "unit-0000.json.complete").write_text("complete\n")
        faults["marker_before_payload"] = {
            "classification": "REFUSE",
            "error": _expect_error(lambda: lifecycle.build_ledger([early_marker], spec)),
        }

        stale = copy.deepcopy(spec)
        stale["sources"] = {next(iter(stale["sources"])): "0" * 64}
        faults["stale_source_proof"] = {
            "classification": "REFUSE",
            "error": _expect_error(
                lambda: lifecycle.check_sources(stale, Path(__file__).resolve().parents[2])
            ),
        }

        wrong_checkpoint = root / "fault-wrong-checkpoint"
        lifecycle.start_attempt(wrong_checkpoint, spec)
        lifecycle.publish_unit(wrong_checkpoint, spec, 0, {"checkpoint": "synthetic"})
        faults["wrong_checkpoint_bytes"] = {
            "classification": "REFUSE",
            "error": _expect_error(
                lambda: lifecycle.require_snapshot(wrong_checkpoint / "unit-0000.json", "0" * 64)
            ),
        }

        foreign_spec = _spec("r32-synthetic-foreign")
        foreign = root / "fault-foreign"
        lifecycle.start_attempt(foreign, foreign_spec)
        faults["foreign_namespace_allocation"] = {
            "classification": "REFUSE",
            "error": _expect_error(lambda: lifecycle.build_ledger([foreign], spec)),
        }

        duplicate_a = root / "fault-duplicate-a"
        duplicate_b = root / "fault-duplicate-b"
        lifecycle.start_attempt(duplicate_a, spec)
        lifecycle.start_attempt(duplicate_b, spec)
        lifecycle.publish_unit(duplicate_a, spec, 0, {"duplicate": "a"})
        lifecycle.publish_unit(duplicate_b, spec, 0, {"duplicate": "b"})
        faults["duplicate_immutable_unit"] = {
            "classification": "REFUSE",
            "error": _expect_error(
                lambda: lifecycle.build_ledger([duplicate_a, duplicate_b], spec)
            ),
        }

        faults["ancestor_split_leakage"] = {
            "classification": "REFUSE",
            "error": _expect_error(lambda: lifecycle.validate_ancestry(["a"], ["a"])),
        }

        faults["missing_required_snapshot"] = {
            "classification": "REFUSE",
            "error": _expect_error(
                lambda: lifecycle.require_snapshot(root / "missing-snapshot.json", "0" * 64)
            ),
        }

        unknown_ledger = lifecycle.empty_ledger(spec)
        unknown_status = lifecycle.disposition(
            spec, unknown_ledger, stop_verified=False, outcome="mechanical"
        )
        if unknown_status["status"] != "STOP_UNVERIFIED" or unknown_status["replacement_allowed"]:
            raise AssertionError("unknown runtime stop was treated as a verified replacement")
        faults["unknown_runtime_stop"] = {
            "classification": "STOP_UNVERIFIED",
            "status": unknown_status["status"],
            "replacement_allowed": unknown_status["replacement_allowed"],
            "completion_error": _expect_error(
                lambda: lifecycle.completion_envelope(
                    spec,
                    unknown_ledger,
                    stop_verified=False,
                    failure_verified=True,
                    attempts=1,
                    spent_minutes=0.0,
                    spent_units=0.0,
                )
            ),
        }

        timeout = root / "fault-timeout-after-export"
        lifecycle.start_attempt(timeout, spec)
        _publish_all(timeout, spec, [0, 1])
        timeout_ledger = lifecycle.build_ledger([timeout], spec)
        timeout_status = lifecycle.disposition(
            spec, timeout_ledger, stop_verified=False, outcome="mechanical"
        )
        if timeout_status["status"] != "STOP_UNVERIFIED" or timeout_status["verified"] != [0, 1]:
            raise AssertionError("timeout-after-export erased or completed verified units")
        faults["timeout_after_successful_export"] = {
            "classification": "STOP_UNVERIFIED",
            "status": timeout_status["status"],
            "verified": timeout_status["verified"],
            "missing": timeout_status["missing"],
        }

        changed_name = root / "fault-changed-filename"
        lifecycle.start_attempt(changed_name, spec)
        (changed_name / "unit-9999.json.complete").write_text("complete\n")
        faults["changed_output_filename_second_look"] = {
            "classification": "REFUSE",
            "error": _expect_error(lambda: lifecycle.build_ledger([changed_name], spec)),
        }

    return {
        "schema": "r32-codex-v-lifecycle-matrix/v1",
        "positive": {
            "clean": clean_status,
            "interrupted_resumed": resumed_status,
            "scientific_stop": scientific_status,
            "interruption_evidence": {
                "before_marker_ignored": True,
                "after_marker_preserved": True,
                "mid_verified": [0, 1],
            },
        },
        "faults": faults,
        "counts": {"positive": 3, "faults": 12},
    }
