"""CPU-only contracts for A1's immutable per-cell resume units."""

from __future__ import annotations

import pytest

from research.gpu_sim_spike import a1_prof_units

COMMIT = "a" * 40
HARDWARE = "A100-40GB"


def _contract(smoke: bool = False) -> tuple[str, tuple[str, ...]]:
    return a1_prof_units.namespace(COMMIT, smoke), a1_prof_units.allocation(smoke)


def test_interrupted_payload_without_completion_marker_is_not_resumable(tmp_path) -> None:
    namespace, allocation = _contract()
    path = a1_prof_units.payload_path(tmp_path, allocation[0])
    path.parent.mkdir()
    path.write_text("{}\n")
    value = a1_prof_units.build_ledger(
        tmp_path, tmp_path / "ledger.json", commit=COMMIT, smoke=False
    )
    assert value["units"] == []


def test_oom_after_one_cell_preserves_it_and_a_recovery_can_skip_it(tmp_path) -> None:
    namespace, allocation = _contract()
    first = allocation[0]
    a1_prof_units.publish_unit(tmp_path, namespace, allocation, first, {"measured": {"worlds": 6}})
    ledger = tmp_path / "verified-ledger.json"
    a1_prof_units.build_ledger(tmp_path, ledger, commit=COMMIT, smoke=False)
    resumed = a1_prof_units.load_ledger(ledger, namespace, allocation)
    assert set(resumed) == {first}
    assert resumed[first] == {"measured": {"worlds": 6}}
    assert [unit for unit in allocation if unit not in resumed][0] == "cell-scripted-e256"


def test_recovery_ledger_copies_only_verified_units_and_loads_portably(tmp_path) -> None:
    namespace, allocation = _contract()
    source = tmp_path / "source"
    recovery = tmp_path / "recovery"
    a1_prof_units.publish_unit(source, namespace, allocation, allocation[0], {"complete": True})
    partial = a1_prof_units.payload_path(source, allocation[1])
    partial.parent.mkdir(exist_ok=True)
    partial.write_text("partial")
    ledger = recovery / "verified-ledger.json"
    value = a1_prof_units.build_ledger(source, ledger, commit=COMMIT, smoke=False)
    assert [row["unit_id"] for row in value["units"]] == [allocation[0]]
    assert a1_prof_units.load_ledger(ledger, namespace, allocation) == {
        allocation[0]: {"complete": True}
    }


def test_duplicate_or_foreign_corruption_is_refused(tmp_path) -> None:
    namespace, allocation = _contract()
    a1_prof_units.publish_unit(tmp_path, namespace, allocation, allocation[0], {"ok": True})
    ledger = tmp_path / "verified-ledger.json"
    value = a1_prof_units.build_ledger(tmp_path, ledger, commit=COMMIT, smoke=False)
    value["units"].append(value["units"][0])
    ledger.unlink()
    ledger.with_name(ledger.name + ".sha256").unlink()
    ledger.with_name(ledger.name + ".complete").unlink()
    a1_prof_units._publish_json(ledger, value)
    with pytest.raises(ValueError, match="duplicate"):
        a1_prof_units.load_ledger(ledger, namespace, allocation)


def test_mac_ledger_builder_refuses_a_foreign_complete_unit(tmp_path) -> None:
    namespace, allocation = _contract()
    a1_prof_units.publish_unit(tmp_path, namespace, allocation, allocation[0], {"ok": True})
    foreign = tmp_path / a1_prof_units.UNIT_DIRECTORY / "cell-scripted-e999.json"
    foreign.write_text("foreign\n")
    digest = __import__("hashlib").sha256(foreign.read_bytes()).hexdigest()
    foreign.with_name(foreign.name + ".sha256").write_text(f"{digest}  {foreign.name}\n")
    foreign.with_name(foreign.name + ".complete").write_text("complete\n")
    with pytest.raises(ValueError, match="foreign"):
        a1_prof_units.build_ledger(tmp_path, tmp_path / "ledger.json", commit=COMMIT, smoke=False)


def test_builder_is_outcome_blind_and_accepts_uninspectable_content(tmp_path) -> None:
    namespace, allocation = _contract(True)
    unit = allocation[0]
    # The builder only checks filename/sidecars; malformed JSON is caught later
    # by load_ledger, after a unit was selected mechanically by the Mac.
    path = a1_prof_units.payload_path(tmp_path, unit)
    path.parent.mkdir()
    path.write_text("this is deliberately not a measurement json\n")
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    path.with_name(path.name + ".complete").write_text("complete\n")
    ledger = tmp_path / "verified-ledger.json"
    value = a1_prof_units.build_ledger(tmp_path, ledger, commit=COMMIT, smoke=True)
    assert value["units"][0]["unit_id"] == unit
    with pytest.raises(ValueError, match="not valid JSON"):
        a1_prof_units.load_ledger(ledger, namespace, allocation)


def test_load_refuses_a_complete_payload_from_a_foreign_namespace(tmp_path) -> None:
    namespace, allocation = _contract(True)
    foreign_namespace = a1_prof_units.namespace("b" * 40, True)
    a1_prof_units.publish_unit(
        tmp_path, foreign_namespace, allocation, allocation[0], {"wrong": True}
    )
    ledger = tmp_path / "verified-ledger.json"
    a1_prof_units.build_ledger(tmp_path, ledger, commit=COMMIT, smoke=True)
    with pytest.raises(ValueError, match="foreign"):
        a1_prof_units.load_ledger(ledger, namespace, allocation)


def test_full_and_canary_shapes_and_namespace_are_distinct() -> None:
    assert len(a1_prof_units.allocation()) == 9
    assert len(a1_prof_units.allocation(True)) == 3
    assert a1_prof_units.allocation(canary=True) == a1_prof_units.allocation()
    assert a1_prof_units.namespace(COMMIT) != a1_prof_units.namespace(COMMIT, canary=True)
    assert all(cell.endswith("-e1") for cell in a1_prof_units.allocation(True))
