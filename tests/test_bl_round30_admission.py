"""R30-0 metadata admission and immutable-resume contracts."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from research.bl_round30 import admission


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _inventory(*, parents: int = 400, archive: bool = True) -> dict[str, object]:
    corpus = {
        "collection_authority": "R29 unsealed development reuse authorization",
        "source_sha256": _sha("source"),
        "label_sha256": _sha("labels"),
        "checkpoint_sha256": _sha("checkpoint"),
        "explicitly_reusable_development": True,
        "already_unsealed": True,
        "f4": False,
        "protected": False,
        "strict": False,
        "retention": False,
        "natural_archive": False,
        "probe": False,
        "gameplay": False,
        "stale": False,
        "pins_current": True,
    }
    rows = [
        {
            "ultimate_parent_id": f"p-{index:04d}",
            "rescue_bearing": index % 2 == 0,
            "previously_inspected": True,
            "prior_uses": ["historical-development"],
            "protected": False,
            "foreign": False,
            "f4": False,
            "stale": False,
        }
        for index in range(parents)
    ]
    value: dict[str, object] = {
        "schema": admission.INVENTORY_SCHEMA,
        "split_salt": "r30-freeze-20261023",
        "corpus": corpus,
        "parents": rows,
    }
    if archive:
        validation = [
            row["ultimate_parent_id"]
            for row in rows
            if admission.parent_split(str(row["ultimate_parent_id"]), str(value["split_salt"]))
            == "validation"
        ]
        value["natural_ordinary_archive"] = {
            "ultimate_parent_ids": validation,
            "sampling_probabilities": {parent: 1.0 for parent in validation},
            "includes_non_rescue": True,
            "explicitly_permitted": True,
        }
    return value


def _approval(value: admission.Admission) -> dict[str, object]:
    return {
        "schema": admission.APPROVAL_SCHEMA,
        "approved": True,
        "inventory_sha256": value.inventory_sha256,
        "allocation_sha256": value.allocation_sha256,
    }


def test_freezes_parent_split_roundtrips_and_requires_codex_v_pin() -> None:
    frozen = admission.freeze_inventory(_inventory())
    assert len(frozen.validation_parents) >= 32
    assert len(frozen.validation_rescue_parents) >= 16
    assert frozen.r30_2_status == "READY_R30_2"
    assert (
        admission.validate_frozen_roundtrip(frozen.inventory).inventory_sha256
        == frozen.inventory_sha256
    )
    admission.require_codex_v_approval(frozen, _approval(frozen))
    assert admission.partition_parents(frozen, "train") == frozen.train_parents


@pytest.mark.parametrize("key", ["protected", "foreign", "f4", "stale"])
def test_refuses_any_protected_or_foreign_parent_before_a_data_opener(key: str) -> None:
    inventory = _inventory()
    assert isinstance(inventory["parents"], list)
    inventory["parents"][0][key] = True  # type: ignore[index]
    with pytest.raises(admission.AdmissionRefused, match="parent"):
        admission.freeze_inventory(inventory)


def test_duplicate_parent_and_stale_split_pin_refuse() -> None:
    duplicate = _inventory()
    assert isinstance(duplicate["parents"], list)
    duplicate["parents"].append(copy.deepcopy(duplicate["parents"][0]))  # type: ignore[index]
    with pytest.raises(admission.AdmissionRefused, match="duplicate"):
        admission.freeze_inventory(duplicate)
    stale = _inventory()
    assert isinstance(stale["parents"], list)
    stale["parents"][0]["split"] = (
        "validation"
        if admission.parent_split(  # type: ignore[index]
            stale["parents"][0]["ultimate_parent_id"], stale["split_salt"]  # type: ignore[index]
        )
        == "train"
        else "train"
    )
    with pytest.raises(admission.AdmissionRefused, match="stale split"):
        admission.freeze_inventory(stale)


def test_missing_ordinary_sampling_probabilities_holds_r30_2() -> None:
    inventory = _inventory()
    assert isinstance(inventory["natural_ordinary_archive"], dict)
    inventory["natural_ordinary_archive"].pop("sampling_probabilities")
    frozen = admission.freeze_inventory(inventory)
    assert admission.admission_status(frozen) == "HOLD_R30_2"
    assert frozen.r30_2_status == "HOLD_R30_2"


def test_immutable_units_skip_only_verified_matching_allocation(tmp_path: Path) -> None:
    namespace, allocation = "bl-r30-r30-1", _sha("allocation")
    admission.publish_unit(
        tmp_path,
        namespace=namespace,
        allocation_sha256=allocation,
        allocation_index=2,
        content={"n": 2},
    )
    ledger_path = tmp_path / "verified-ledger.json"
    admission.build_ledger(
        tmp_path,
        ledger_path,
        namespace=namespace,
        allocation_sha256=allocation,
        allocation_indices=(1, 2, 3),
    )
    assert admission.load_verified_ledger(
        ledger_path, namespace=namespace, allocation_sha256=allocation, allocation_indices=(1, 2, 3)
    ) == {2: {"n": 2}}
    (tmp_path / "units" / "000003.json").write_text("partial")
    assert admission.load_verified_ledger(
        ledger_path, namespace=namespace, allocation_sha256=allocation, allocation_indices=(1, 2, 3)
    ) == {2: {"n": 2}}


def test_ledger_refuses_missing_marker_and_foreign_namespace(tmp_path: Path) -> None:
    namespace, allocation = "bl-r30-r30-1", _sha("allocation")
    path = admission.publish_unit(
        tmp_path,
        namespace=namespace,
        allocation_sha256=allocation,
        allocation_index=0,
        content={"ok": True},
    )
    Path(str(path) + ".complete").unlink()
    ledger = tmp_path / "ledger.json"
    admission.build_ledger(
        tmp_path, ledger, namespace=namespace, allocation_sha256=allocation, allocation_indices=(0,)
    )
    assert (
        admission.load_verified_ledger(
            ledger, namespace=namespace, allocation_sha256=allocation, allocation_indices=(0,)
        )
        == {}
    )
    foreign = tmp_path / "foreign"
    admission.publish_unit(
        foreign,
        namespace="bl-r30-other",
        allocation_sha256=allocation,
        allocation_index=0,
        content={"ok": True},
    )
    foreign_ledger = foreign / "ledger.json"
    admission.build_ledger(
        foreign,
        foreign_ledger,
        namespace="bl-r30-other",
        allocation_sha256=allocation,
        allocation_indices=(0,),
    )
    with pytest.raises(admission.AdmissionRefused, match="foreign"):
        admission.load_verified_ledger(
            foreign_ledger,
            namespace=namespace,
            allocation_sha256=allocation,
            allocation_indices=(0,),
        )
