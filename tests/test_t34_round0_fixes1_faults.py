"""Typed, non-scientific fault boundaries for T34 Round 0 FIX-FIRST #1."""

from __future__ import annotations

import numpy as np
import pytest

from research.t34_exit_b8 import faults as exit_faults
from research.t34_exit_b8.policy import PolicyEvaluation, action_grid
from research.t34_round0 import runner
from research.t34_round0.faults import (
    CapExpired,
    CapExpiry,
    ExportSerializationError,
    IllegalActionFault,
    NonFiniteQLossFault,
    OneActionShareFault,
    OperationalFailure,
)


def test_round0_fault_imports_reexport_canonical_exit_classes() -> None:
    assert CapExpired is exit_faults.CapExpired
    assert CapExpiry is exit_faults.CapExpiry
    assert ExportSerializationError is exit_faults.ExportSerializationError
    assert IllegalActionFault is exit_faults.IllegalActionFault
    assert NonFiniteQLossFault is exit_faults.NonFiniteQLossFault
    assert OneActionShareFault is exit_faults.OneActionShareFault
    assert OperationalFailure is exit_faults.OperationalFailure


def test_fault_markers_require_dedicated_exception_types() -> None:
    assert runner._fault_marker(NonFiniteQLossFault("any message")) == "nonfinite_q"
    assert runner._fault_marker(IllegalActionFault("any message")) == "illegal_action"
    assert runner._fault_marker(OneActionShareFault("any message")) == "action_collapse_100k"
    assert runner._fault_marker(ValueError("illegal action; non-finite Q")) is None


def test_policy_boundary_raises_the_registered_typed_faults() -> None:
    with pytest.raises(NonFiniteQLossFault):
        PolicyEvaluation(q=np.full(6, np.nan), legal=np.ones(6, dtype=bool))

    with pytest.raises(IllegalActionFault):
        action_grid(type("World", (), {"sim": object()})(), 6)


def test_export_errors_are_explicit_operational_faults(monkeypatch, tmp_path) -> None:
    from research.t34_exit_b8 import teacher_v2

    monkeypatch.setattr(
        runner, "write_world", lambda _path, _value: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(ExportSerializationError):
        runner._write_export(tmp_path / "trace.json", {})
    monkeypatch.setattr(
        runner, "_write_new", lambda _path, _value: (_ for _ in ()).throw(ValueError("fixture"))
    )
    with pytest.raises(ExportSerializationError):
        runner._write_json_export(tmp_path / "receipt.json", {"schema": "fixture"})
    monkeypatch.setattr(
        teacher_v2, "write_world", lambda _path, _value: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(ExportSerializationError):
        teacher_v2._write_export(tmp_path / "teacher.json", {})
    assert (
        runner._operational_kind(ExportSerializationError("fixture")) == "export_or_serialization"
    )


def test_cap_expiry_is_distinct_from_an_unexplained_abort() -> None:
    assert runner._operational_kind(CapExpiry("fixture")) == "cap_expired"
    assert runner._operational_kind(RuntimeError("fixture")) is None


def test_worker_deadline_raises_cap_expiry_before_incomplete_teacher_result(monkeypatch) -> None:
    monkeypatch.setattr(runner.time, "monotonic", lambda: 2.0)
    with pytest.raises(CapExpiry):
        runner._worker_deadline(2.0)
