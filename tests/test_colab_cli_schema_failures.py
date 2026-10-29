"""Shared Colab manifest-schema and receipt failure-class contracts."""

from __future__ import annotations

import pytest

from research.colab_cli import failures, schema


def _manifest() -> dict[str, object]:
    return {
        "budget": {"rate_units_per_hour": 1.0, "unit_cap": 1.0, "reserve_units": 0.0},
        "required_exports": ["reports/result.json", "checkpoint/state.pt"],
    }


def _entry(path: str) -> dict[str, object]:
    return {"path": path, "sha256": "a" * 64, "size": 0}


def test_shared_schema_rejects_historical_other_assignment_list_shape() -> None:
    manifest = _manifest()
    manifest["budget"]["other_assignments"] = [{"endpoint": "a", "rate_units_per_hour": 1.0}]

    with pytest.raises(schema.SchemaError, match="must map endpoint"):
        schema.validate_manifest(manifest)


@pytest.mark.parametrize("field", ["rate_units_per_hour", "unit_cap"])
def test_shared_schema_accepts_zero_warning_thresholds(field: str) -> None:
    manifest = _manifest()
    manifest["budget"][field] = 0.0

    assert schema.validate_manifest(manifest) == tuple(manifest["required_exports"])
    del manifest["budget"][field]
    schema.validate_manifest(manifest)


def test_terminal_inventory_requires_scanner_records_for_declared_exports() -> None:
    with pytest.raises(schema.SchemaError, match="checkpoint/state.pt"):
        schema.validate_export_inventory(_manifest(), [_entry("reports/result.json")])

    actual = schema.validate_export_inventory(
        _manifest(), [_entry("reports/result.json"), _entry("checkpoint/state.pt")]
    )
    assert actual == ("checkpoint/state.pt", "reports/result.json")


def test_terminal_failure_classes_never_conflate_child_and_generation() -> None:
    foreign = failures.terminal_failure(
        {"generation": "old", "returncode": 23, "status": "failed"}, "current"
    )
    assert foreign is not None
    assert foreign.failure_class is failures.FailureClass.FOREIGN_GENERATION

    child = failures.terminal_failure(
        {
            "generation": "current",
            "returncode": 23,
            "status": "failed",
            "last_traceback_line": "ValueError: actual producer fault",
        },
        "current",
    )
    assert child is not None
    assert child.failure_class is failures.FailureClass.CHILD_NONZERO_EXIT
    assert "ValueError: actual producer fault" in str(child)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("upload exceeds bounded transfer size"), failures.FailureClass.UPLOAD_CAP),
        (
            RuntimeError("deadline reached during telemetry recovery"),
            failures.FailureClass.RPC_EXEC_POLL_TIMEOUT,
        ),
    ],
)
def test_classify_known_supervisor_boundaries(
    error: BaseException, expected: failures.FailureClass
) -> None:
    assert failures.classify_exception(error) is expected


@pytest.mark.parametrize("value", [None, [], "endpoint", 1])
def test_explicit_other_assignments_must_be_map(value):
    manifest = _manifest()
    manifest["budget"]["other_assignments"] = value
    with pytest.raises(schema.SchemaError, match="must map endpoint"):
        schema.validate_manifest(manifest)


def test_generation_fence_rpc_error_is_foreign():
    assert (
        failures.classify_exception(
            RuntimeError("lease generation fence mismatch; lease was not renewed")
        )
        is failures.FailureClass.FOREIGN_GENERATION
    )


def test_budget_is_optional():
    manifest = _manifest()
    del manifest["budget"]
    schema.validate_manifest(manifest)
