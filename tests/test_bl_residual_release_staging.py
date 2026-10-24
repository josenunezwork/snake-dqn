"""Actual release writer -> full fit staging; synthetic metadata only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_labels import d4096_final, d4096_resume, d4096_sealed, mac_gates
from research.bl_labels.exports import publish_session
from research.bl_labels.test_d4096_resume import add_attempt, pack_fixture
from research.bl_ppo.exports import digest
from research.bl_residual import collection_receipts as receipts
from research.bl_residual import job
from research.bl_residual.protocol import F4_REGISTRATION_SHA, NAMESPACE
from tests import test_bl_residual_collection_receipts as i3


def bind(root: Path, control: dict, key: str, path: Path) -> None:
    control["stage_receipts"][key] = path.name
    control["input_hashes"][path.name] = digest(path)


def l4_pair(root: Path, *, count: int = 1) -> tuple[Path, Path]:
    """Include a charged failed attempt even though the complete union is Mac."""
    root.mkdir(parents=True)
    attempts = []
    for index in range(count):
        failed = root / f"failed-{index}.json"
        failed.write_text('{"job_id":"fixture-failed-l4","status":"INCOMPLETE"}')
        attempts.append({"receipt": {"path": str(failed), "sha256": digest(failed)}})
    identity = {
        "series_id": NAMESPACE,
        "registration_sha256": F4_REGISTRATION_SHA,
        "dose": 4096,
    }
    inventory = i3.sealed(
        root / "inventory.json",
        {
            **identity,
            "schema": "bl-labels-d4096-l4-attempt-inventory/v1",
            "all_attempts_listed": True,
            "attempts": attempts,
        },
    )
    ledger = i3.sealed(
        root / "accounting.json",
        {
            **identity,
            "schema": receipts.L4_ACCOUNTING_SCHEMA,
            "attempts": [
                {"receipt_sha256": row["receipt"]["sha256"], "charged_units": 2} for row in attempts
            ],
        },
    )
    return ledger, inventory


def project(root: Path, control: dict, ledger: Path | None, name: str) -> None:
    path = root / name
    receipts.write_projection_receipt(
        *[
            root / control["stage_receipts"][key]
            for key in ("D256_COLLECTION", "D1024_COLLECTION", "D4096_MAC_PROJECTION")
        ],
        path,
        l4_ledger=ledger,
    )
    bind(root, control, "D4096_PROJECTION", path)


def actual_release_staging(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, count: int | None = 1
) -> dict:
    """Use the real union and release writers; replace only source/admission pins."""
    monkeypatch.setattr(i3, "SERIES", NAMESPACE)
    monkeypatch.setattr(d4096_sealed, "CANONICAL_PACK", root / "canonical-pack")
    monkeypatch.setattr(job, "D4096_DISCARD_SEAL", root / "canonical-pack/discard.json")
    control = i3.staging(root)
    funding = i3.bound_value(root, control, "D4096_FUNDING")
    i3.replace_bound(
        root,
        control,
        "D4096_FUNDING",
        {
            "doses": [256, 1024],
            "inputs": {
                f"d{dose}_checkpoint_sha256": funding[f"d{dose}_fit_sha256"] for dose in (256, 1024)
            },
        },
    )
    pack, source = pack_fixture(root / "writer")
    output = add_attempt(
        pack, source, 1, workers=6, terminal_change={"status": "COMPLETE", "pending_parents": []}
    )
    for item in source["allocation"]:
        publish_session(
            output,
            source["namespace"],
            source["allocation"],
            {
                "schema": "bl-labels-f0-session/v1",
                "status": "COMPLETE",
                "namespace": source["namespace"],
                "allocation_id": item["allocation_id"],
                "allocation": item,
                "counters": {"deaths": 0},
            },
        )
    funding_path = root / control["stage_receipts"]["D4096_FUNDING"]
    union = root / "writer/final/verified-ledger.json"
    mac_gates.publish_ledger(
        pack, [output], union, control=source, funding_receipt=funding_path, require_complete=True
    )
    monkeypatch.setattr(d4096_resume, "original_control", lambda _: source)
    monkeypatch.setattr(d4096_final, "require_pinned_source", lambda pin: pin)
    options = {"no_l4_attempts": True}
    ledger = None
    if count is not None:
        ledger, inventory = l4_pair(root / "original-l4", count=count)
        options = {"l4_ledger": ledger, "l4_attempt_inventory": inventory}
        # Stage byte-identical copies with different names, as a real upload does.
        for key, path in (
            ("D4096_L4_ACCOUNTING", ledger),
            ("D4096_L4_ATTEMPT_INVENTORY", inventory),
        ):
            staged = root / f"staged-{path.name}"
            staged.write_bytes(path.read_bytes())
            Path(str(staged) + ".sha256").write_text(f"{digest(path)}  {staged.name}\n")
            Path(str(staged) + ".complete").write_text("complete\n")
            bind(root, control, key, staged)
    release = root / "actual-release.json"
    d4096_sealed.seal_collection_release(
        release, union, funding_path, pack=pack, code_commit="c" * 40, **options
    )
    bind(root, control, "D4096_COLLECTION_DISPOSITION", release)
    control["factory_union_ledger_sha256"] = digest(union)
    project(root, control, ledger, "actual-projection.json")
    i3.assert_dependent_seals(root, control)
    job._validate_staging(root, control, [4096], i3.RECIPE, set())
    return control


@pytest.mark.parametrize("count", [None, 0, 1])
def test_actual_release_writer_full_staging_accepts_exact_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int | None
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch, count=count)
    # Remote validation must use only staged copies, even when Mac originals vanish.
    for path in (tmp_path / "original-l4").glob("*"):
        path.unlink()
    job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize("substitute", [False, True])
def test_actual_release_rejects_omitted_or_substituted_empty_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, substitute: bool
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    ledger = None
    if substitute:
        original, _ = l4_pair(tmp_path / "empty", count=0)
        ledger = i3.sealed(tmp_path / "empty-accounting.json", json.loads(original.read_text()))
        bind(tmp_path, control, "D4096_L4_ACCOUNTING", ledger)
    else:
        control["stage_receipts"].pop("D4096_L4_ACCOUNTING")
    project(tmp_path, control, ledger, "empty-projection.json")
    i3.assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match="release-bound L4 ledger"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize("count", [0, 1])
def test_actual_no_l4_release_rejects_contrary_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch, count=None)
    original, _ = l4_pair(tmp_path / "contrary", count=count)
    ledger = i3.sealed(tmp_path / "contrary-accounting.json", json.loads(original.read_text()))
    bind(tmp_path, control, "D4096_L4_ACCOUNTING", ledger)
    project(tmp_path, control, ledger, "contrary-projection.json")
    with pytest.raises(ValueError, match="contradicts release no-L4"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize("declaration", [None, {}, {"status": "unknown"}])
def test_actual_release_rejects_missing_or_unknown_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declaration: dict | None
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    value = i3.bound_value(tmp_path, control, "D4096_COLLECTION_DISPOSITION")
    value.pop("l4_accounting")
    if declaration is not None:
        value["l4_accounting"] = declaration
    path = i3.sealed(tmp_path / "undeclared-release.json", value)
    bind(tmp_path, control, "D4096_COLLECTION_DISPOSITION", path)
    with pytest.raises(ValueError, match="explicit release L4"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize(
    "field,value",
    [("attempt_count", 0), ("attempt_count", True), ("charged_units", 0), ("charged_units", False)],
)
def test_actual_release_rechecks_count_and_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    declaration = copy.deepcopy(
        i3.bound_value(tmp_path, control, "D4096_COLLECTION_DISPOSITION")["l4_accounting"]
    )
    declaration[field] = value
    i3.replace_bound(
        tmp_path, control, "D4096_COLLECTION_DISPOSITION", {"l4_accounting": declaration}
    )
    with pytest.raises(ValueError, match="attempt-set, count or charge"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize("substitute", [False, True])
def test_actual_release_rejects_missing_or_substituted_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, substitute: bool
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    key = "D4096_L4_ATTEMPT_INVENTORY"
    if substitute:
        _, original = l4_pair(tmp_path / "empty", count=0)
        path = i3.sealed(tmp_path / "empty-inventory.json", json.loads(original.read_text()))
        bind(tmp_path, control, key, path)
    else:
        control["stage_receipts"].pop(key)
    with pytest.raises(ValueError, match="release-bound L4 inventory"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize(
    "edits,match",
    [
        ({"schema": "foreign"}, "complete D4096 L4"),
        ({"series_id": "foreign"}, "complete D4096 L4"),
        ({"registration_sha256": "f" * 64}, "complete D4096 L4"),
        ({"dose": 1024}, "complete D4096 L4"),
        ({"all_attempts_listed": False}, "complete D4096 L4"),
        ({"attempts": None}, "complete D4096 L4"),
        ({"attempts": []}, "attempt-set, count or charge"),
        (
            {"attempts": [{"receipt": {"path": "/missing", "sha256": "f" * 64}}]},
            "attempt-set, count or charge",
        ),
        ({"attempts": [{"receipt": {"path": "relative", "sha256": "f" * 64}}]}, "malformed"),
    ],
)
def test_actual_release_rechecks_inventory_even_with_rebound_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edits: dict, match: str
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    i3.replace_bound(tmp_path, control, "D4096_L4_ATTEMPT_INVENTORY", edits)
    declaration = i3.bound_value(tmp_path, control, "D4096_COLLECTION_DISPOSITION")["l4_accounting"]
    path = tmp_path / control["stage_receipts"]["D4096_L4_ATTEMPT_INVENTORY"]
    declaration["inventory"] = {"path": str(path), "sha256": digest(path)}
    i3.replace_bound(
        tmp_path, control, "D4096_COLLECTION_DISPOSITION", {"l4_accounting": declaration}
    )
    i3.assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match=match):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


def test_actual_release_rejects_duplicate_inventory_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    inventory = i3.bound_value(tmp_path, control, "D4096_L4_ATTEMPT_INVENTORY")
    i3.replace_bound(
        tmp_path, control, "D4096_L4_ATTEMPT_INVENTORY", {"attempts": inventory["attempts"] * 2}
    )
    declaration = i3.bound_value(tmp_path, control, "D4096_COLLECTION_DISPOSITION")["l4_accounting"]
    path = tmp_path / control["stage_receipts"]["D4096_L4_ATTEMPT_INVENTORY"]
    declaration.update(inventory={"path": str(path), "sha256": digest(path)}, attempt_count=2)
    i3.replace_bound(
        tmp_path, control, "D4096_COLLECTION_DISPOSITION", {"l4_accounting": declaration}
    )
    with pytest.raises(ValueError, match="attempt-set, count or charge"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize("count", [None, 1])
@pytest.mark.parametrize("field", ["l4_tail_accounting", "l4_tail_accounting_sha256"])
def test_actual_release_rejects_contrary_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int | None, field: str
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch, count=count)
    value = (
        "f" * 64
        if field.endswith("sha256")
        else receipts.l4_accounting(None, NAMESPACE) | {"charged_units": 1}
    )
    i3.replace_bound(tmp_path, control, "D4096_PROJECTION", {field: value})
    with pytest.raises(ValueError, match="L4"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())


@pytest.mark.parametrize("key", ["D4096_L4_ACCOUNTING", "D4096_L4_ATTEMPT_INVENTORY"])
@pytest.mark.parametrize("suffix", [".sha256", ".complete"])
def test_actual_release_requires_sealed_staged_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, suffix: str
) -> None:
    control = actual_release_staging(tmp_path, monkeypatch)
    Path(str(tmp_path / control["stage_receipts"][key]) + suffix).unlink()
    with pytest.raises(ValueError, match="missing or unsafe export"):
        job._validate_staging(tmp_path, control, [4096], i3.RECIPE, set())
