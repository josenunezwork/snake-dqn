"""Synthetic-only F5 metadata boundary proofs; no real container is opened."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from research.bl_residual import f5_status_extractor as subject


def bindings() -> dict:
    return {"n/p/0": {"parent_world": "p", "allocation_order": 17, "ordinary_only": False}}


def arrays() -> dict:
    return {
        "row_id": np.array(["n/p/0/3", "n/p/0/4"]),
        "life_id": np.array(["n/p/0", "n/p/0"]),
        "parent_world": np.array(["p", "p"]),
        "ordinary": np.array([True, False]),
        "natural_stream": np.array([True, False]),
        "legal": np.array([[True, True, False, False, False, False], [True] * 6]),
        "action_mask": np.zeros((2, 6), dtype=bool),
        "status": np.array([["EXACT", "UNKNOWN", *["NOT_EVALUATED"] * 4], ["UNKNOWN"] * 6]),
    }


def npy(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def fixture(
    path: Path,
    values: dict | None = None,
    *,
    forbidden: bytes = b"\xffnever decode",
    replace: dict | None = None,
    extra: str | None = None,
    duplicate: bool = False,
) -> Path:
    values = arrays() if values is None else values
    replace = {} if replace is None else replace
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(subject.NPZ_MEMBERS):
            raw = npy(values[name[:-4]]) if name in subject.NPZ_OPEN else forbidden
            archive.writestr(name, replace.get(name, raw))
        if extra:
            archive.writestr(extra, b"unknown schema")
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("status.npy", npy(values["status"]))
    return path


def test_full_inventory_enumerates_both_strata_and_all_missing_actions(tmp_path: Path) -> None:
    result = subject.project_synthetic_npz(fixture(tmp_path / "fixture.npz"), bindings())
    assert result["strata"] == {"ordinary": 1, "doom": 1}
    assert len(result["status_inventory"]) == 12
    assert {r["status"] for r in result["status_inventory"]} == {"EXACT", "NONEXACT"}
    assert result["count_U"] == 1
    unit = result["cov_missing_ordinary_units"][0]
    assert (unit["row_id"], unit["action"], unit["allocation_order"]) == ("n/p/0/3", 1, 17)
    assert result["count_reconciliation"]["action_units"] == 12
    assert result["count_reconciliation"]["expected_action_units"] == 12


def test_forbidden_changes_and_known_nonexact_subtypes_leave_identical_output_and_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = fixture(tmp_path / "left.npz", forbidden=b"\xff undecodable object/pickle mass")
    changed = arrays()
    changed["status"][changed["status"] == "UNKNOWN"] = "NOT_EVALUATED"
    right = fixture(tmp_path / "right.npz", changed, forbidden=b"outcome" * 9000)
    opened, decoded = [], []
    original_open, original_decode = zipfile.ZipFile.open, np.frombuffer

    def track_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original_open(self, name, *args, **kwargs)

    def track_decode(raw, *args, **kwargs):
        decoded.append((str(kwargs.get("dtype")), len(raw)))
        assert b"outcome" not in raw and b"undecodable" not in raw
        return original_decode(raw, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", track_open)
    monkeypatch.setattr(np, "frombuffer", track_decode)
    a = subject.project_synthetic_npz(left, bindings())
    trace_a = (opened[:], decoded[:])
    opened.clear()
    decoded.clear()
    b = subject.project_synthetic_npz(right, bindings())
    assert a == b
    assert trace_a == (opened, decoded)
    assert opened == sorted(subject.NPZ_OPEN) * 2
    assert not set(opened) & (subject.NPZ_MEMBERS - subject.NPZ_OPEN)


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "missing"])
def test_zip_schema_rejected_before_any_member_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    path = fixture(
        tmp_path / "bad.npz",
        extra="other.npy" if fault == "unknown" else None,
        duplicate=fault == "duplicate",
    )
    if fault == "missing":
        with zipfile.ZipFile(path) as archive:
            raw = {n: archive.read(n) for n in archive.namelist() if n != "status.npy"}
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in raw.items():
                archive.writestr(name, value)

    def no_open(*args, **kwargs):
        raise AssertionError("schema refusal must precede member open")

    monkeypatch.setattr(zipfile.ZipFile, "open", no_open)
    with pytest.raises(subject.Refused):
        subject.project_synthetic_npz(path, bindings())


@pytest.mark.parametrize(
    "fault", ["huge_shape", "object", "huge_width", "bad_shape", "trailing", "fortran"]
)
def test_every_allowed_header_checked_before_any_array_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    values = arrays()
    replacement = {}
    if fault == "huge_shape":
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<U13", "fortran_order": False, "shape": (2**60, 6)}
        )
        replacement["status.npy"] = stream.getvalue()
    elif fault == "object":
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "|O", "fortran_order": False, "shape": (2, 6)}
        )
        replacement["status.npy"] = stream.getvalue()
    elif fault == "huge_width":
        values["row_id"] = values["row_id"].astype("U999")
    elif fault == "bad_shape":
        values["status"] = np.array(["EXACT", "UNKNOWN"])
    elif fault == "trailing":
        replacement["status.npy"] = npy(values["status"]) + b"extra"
    elif fault == "fortran":
        values["status"] = np.asfortranarray(values["status"])
    path = fixture(tmp_path / "bad.npz", values, replace=replacement)

    def no_decode(*args, **kwargs):
        raise AssertionError("invalid header reached array allocation")

    monkeypatch.setattr(np, "frombuffer", no_decode)
    with pytest.raises(subject.Refused):
        subject.project_synthetic_npz(path, bindings())


@pytest.mark.parametrize(
    "fault",
    [
        "row_prefix",
        "row_decision",
        "parent",
        "life",
        "duplicate",
        "bytes_exact",
        "status",
        "stratum",
        "nonboolean",
        "int_identity",
    ],
)
def test_unknown_or_contradictory_metadata_fails_closed(tmp_path: Path, fault: str) -> None:
    values = arrays()
    if fault == "row_prefix":
        values["row_id"][0] = "evil/3"
    elif fault == "row_decision":
        values["row_id"][0] = "n/p/0/x"
    elif fault == "parent":
        values["parent_world"][0] = "x"
    elif fault == "life":
        values["life_id"][0] = "n/x/0"
    elif fault == "duplicate":
        values["row_id"][1] = values["row_id"][0]
    elif fault == "bytes_exact":
        values["status"] = values["status"].astype("S13")
    elif fault == "status":
        values["status"][0, 0] = "SURVIVED"
    elif fault == "stratum":
        values["natural_stream"][0] = False
    elif fault == "nonboolean":
        values["legal"] = values["legal"].astype(np.int8)
    elif fault == "int_identity":
        values["parent_world"] = np.array([1, 1])
    with pytest.raises(subject.Refused) as error:
        subject.project_synthetic_npz(fixture(tmp_path / "bad.npz", values), bindings())
    assert "evil" not in str(error.value) and "SURVIVED" not in str(error.value)


def test_ordinary_only_sibling_and_exclusions_are_enforced(tmp_path: Path) -> None:
    path = fixture(tmp_path / "fixture.npz")
    sibling = bindings()
    sibling["n/p/0"]["ordinary_only"] = True
    with pytest.raises(subject.Refused, match="sibling"):
        subject.project_synthetic_npz(path, sibling)
    with pytest.raises(subject.Refused, match="lineage"):
        subject.project_synthetic_npz(path, bindings(), exclusions=frozenset({"p"}))


def test_original_allocation_order_preserves_source_row_identity(tmp_path: Path) -> None:
    values = arrays()
    values["life_id"] = np.array(["n/p/0", "n/z/0"])
    values["parent_world"] = np.array(["p", "z"])
    values["row_id"] = np.array(["n/p/0/3", "n/z/0/4"])
    mapping = {
        **bindings(),
        "n/z/0": {"parent_world": "z", "allocation_order": 2, "ordinary_only": False},
    }
    result = subject.project_synthetic_npz(fixture(tmp_path / "order.npz", values), mapping)
    assert result["status_inventory"][0]["allocation_order"] == 2
    assert result["status_inventory"][0]["source_row_index"] == 1
    assert result["status_inventory"][-1]["allocation_order"] == 17
    assert result["status_inventory"][-1]["source_row_index"] == 0


def test_foreign_or_missing_life_cannot_silently_shrink_inventory(tmp_path: Path) -> None:
    mapping = {
        **bindings(),
        "n/x/0": {"parent_world": "x", "allocation_order": 20, "ordinary_only": False},
    }
    with pytest.raises(subject.Refused, match="coverage"):
        subject.project_synthetic_npz(fixture(tmp_path / "missing.npz"), mapping)


@pytest.mark.parametrize("bad", ["D4096", "t34-round0-study-01"])
def test_protected_path_refused_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    def no_open(*args, **kwargs):
        raise AssertionError("protected input opened")

    monkeypatch.setattr(Path, "open", no_open)
    with pytest.raises(subject.Refused, match="protected"):
        subject.project_synthetic_npz(tmp_path / bad / "fixture.npz", bindings())


def test_parent_symlink_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    fixture(real / "fixture.npz")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.Refused, match="symlink"):
        subject.project_synthetic_npz(alias / "fixture.npz", bindings())


def test_deadline_errors_are_redacted(tmp_path: Path) -> None:
    ticks = iter([0, 121, 122])
    with pytest.raises(subject.Refused, match="wall bound"):
        subject.project_synthetic_npz(
            fixture(tmp_path / "fixture.npz"), bindings(), clock=lambda: next(ticks)
        )


def test_container_ordinary_ceiling_filters_to_full_selected_U_without_truncation(
    tmp_path: Path,
) -> None:
    mapping, lives, rows = {}, [], []
    for i in range(969):
        life = f"n/p-{i:04d}/0"
        mapping[life] = {
            "parent_world": f"p-{i:04d}",
            "allocation_order": i,
            "ordinary_only": i >= 960,
        }
        for j in range(8):
            lives.append(life)
            rows.append(f"{life}/{j}")
    lives.append("n/p-0000/0")
    rows.append("n/p-0000/0/8")
    n = len(rows)
    ordinary = np.array([True] * (n - 1) + [False])
    values = {
        "row_id": np.array(rows),
        "life_id": np.array(lives),
        "parent_world": np.array([life.split("/")[1] for life in lives]),
        "ordinary": ordinary,
        "natural_stream": ordinary.copy(),
        "legal": np.ones((n, 6), dtype=bool),
        "action_mask": np.ones((n, 6), dtype=bool),
        "status": np.full((n, 6), "UNKNOWN"),
    }
    result = subject.project_synthetic_npz(fixture(tmp_path / "ceiling.npz", values), mapping)
    assert result["count_U"] == 46080
    assert result["if_COV_selected"] == "PENDING_R4_SELECTOR"
    assert len(result["cov_missing_ordinary_units"]) == 46080
    assert result["source_rows"] == 7753
    assert result["eligible_rows"] == 7681
    assert result["count_reconciliation"]["source_ordinary_rows"] == 7752
    assert result["strata"] == {"ordinary": 7680, "doom": 1}
    assert not any(mapping[r["life_id"]]["ordinary_only"] for r in result["status_inventory"])
    assert subject.MAX_TOTAL_ROWS == 94152
    assert subject.MAX_SELECTED_ROWS == 94080


def test_missing_acceptance_or_any_changed_source_pin_is_refused() -> None:
    pins = {key: "a" * 64 for key in ("source", "tests", "contract", "inventory")}
    receipt = {
        "schema": "bl-f5-extractor-independent-acceptance/v2",
        "decision": "ACCEPT_STATUS_ONLY_EXTRACTOR",
        "independent_reviewer": "fixture judge",
        "ruling_sha256": subject.RULING_SHA,
        "reconciliation_ruling_sha256": subject.RECONCILIATION_RULING_SHA,
        "pins": pins.copy(),
    }
    subject.validate_acceptance(receipt, pins)
    for key in pins:
        changed = {**pins, key: "b" * 64}
        with pytest.raises(subject.Refused, match="acceptance"):
            subject.validate_acceptance(receipt, changed)
    with pytest.raises(subject.Refused, match="acceptance"):
        subject.validate_acceptance({}, pins)


def test_missing_acceptance_cannot_open_real_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_source(*args, **kwargs):
        raise AssertionError("real source must remain unopened")

    monkeypatch.setattr(subject, "open_source", no_source)
    with pytest.raises(subject.Refused, match="acceptance"):
        subject.extract_accepted({})


def test_zero_U_holds_COV_without_selecting_a_mechanism(tmp_path: Path) -> None:
    values = arrays()
    values["status"][:] = "EXACT"
    result = subject.project_synthetic_npz(fixture(tmp_path / "zero.npz", values), bindings())
    assert result["count_U"] == 0
    assert result["if_COV_selected"] == "HOLD_NO_TARGET"


def test_per_life_ordinary_cap_is_enforced(tmp_path: Path) -> None:
    values = arrays()
    values = {key: np.repeat(value[:1], 10, axis=0) for key, value in values.items()}
    values["row_id"] = np.array([f"n/p/0/{i}" for i in range(10)])
    values["ordinary"][-1] = values["natural_stream"][-1] = False
    with pytest.raises(subject.Refused, match="stratum bound"):
        subject.project_synthetic_npz(fixture(tmp_path / "too-many.npz", values), bindings())


def test_CLI_exposes_no_real_input_mode() -> None:
    with pytest.raises(SystemExit):
        subject.main([])
    with pytest.raises(SystemExit):
        subject.main(["--synthetic-life", "/not/an/allowed/path"])


def sibling_bindings() -> dict:
    return {
        **bindings(),
        "n/p/1": {"parent_world": "p", "allocation_order": 18, "ordinary_only": True},
    }


def selected960_fixture(present: int = 0) -> tuple[dict, dict]:
    mapping, lives, row_ids, flags = {}, [], [], []
    for i in range(969):
        life = f"n/p-{i:04d}/0"
        mapping[life] = {
            "parent_world": f"p-{i:04d}",
            "allocation_order": i,
            "ordinary_only": i >= 960,
        }
        if i < 960:
            lives.extend([life, life])
            row_ids.extend([f"{life}/3", f"{life}/4"])
            flags.extend([True, False])
        elif i < 960 + present:
            lives.append(life)
            row_ids.append(f"{life}/3")
            flags.append(True)
    n = len(lives)
    ordinary = np.array(flags)
    return mapping, {
        "life_id": np.array(lives),
        "parent_world": np.array([life.split("/")[1] for life in lives]),
        "row_id": np.array(row_ids),
        "ordinary": ordinary,
        "natural_stream": ordinary.copy(),
        "legal": np.ones((n, 6), dtype=bool),
        "action_mask": np.ones((n, 6), dtype=bool),
        "status": np.full((n, 6), "UNKNOWN"),
    }


def test_all_absent_all_nonzero_present_and_mixed_siblings_have_identical_selected_output(
    tmp_path: Path,
) -> None:
    results = []
    for present in (0, 9, 4):
        mapping, values = selected960_fixture(present)
        result = subject.project_synthetic_npz(
            fixture(tmp_path / f"presence-{present}.npz", values), mapping
        )
        results.append(result)
        report = result["id_reconciliation"]
        assert len(report["selected_expected_ids"]) == 960
        assert report["selected_observed_ids"] == report["selected_expected_ids"]
        assert len(report["source_observed_ids"]) == 960 + present
        assert len(report["excluded_present_ids"]) == present
        assert len(report["excluded_absent_ids"]) == 9 - present
        assert report["counts"] == {
            "source_observed": 960 + present,
            "selected_expected": 960,
            "selected_observed": 960,
            "excluded_present": present,
            "excluded_absent": 9 - present,
        }
        assert report["accepted"]
        assert not report["count_evidence_present"]
        assert report["count_evidence_sha256"] is None
        assert all(
            d["classification"].startswith("EXCLUDED_BY_PREDATA_AMENDMENT_")
            for d in report["differences"]
        )
        assert result["count_reconciliation"]["eligible_lives"] == 960
        assert result["count_reconciliation"]["observed_selected_lives"] == 960
        assert result["eligible_rows"] == 1920
        assert result["source_rows"] == 1920 + present
        assert result["strata"] == {"ordinary": 960, "doom": 960}
        assert result["count_U"] == 5760
        assert len(result["status_inventory"]) == 11520
        for row in result["status_inventory"]:
            assert not mapping[row["life_id"]]["ordinary_only"]
            index = row["source_row_index"]
            assert row["row_id"] == values["row_id"][index]
            assert row["life_id"] == values["life_id"][index]
            assert row["allocation_order"] == mapping[row["life_id"]]["allocation_order"]
    for key in (
        "status_inventory",
        "cov_missing_ordinary_units",
        "status_inventory_sha256",
        "U_inventory_sha256",
    ):
        assert results[0][key] == results[1][key] == results[2][key]


@pytest.mark.parametrize("fault", ["doom", "duplicate", "ancestry", "cap", "status"])
def test_present_excluded_siblings_still_obey_original_container_checks(
    tmp_path: Path, fault: str
) -> None:
    mapping, values = selected960_fixture(9)
    if fault == "doom":
        values["ordinary"][-1] = values["natural_stream"][-1] = False
    elif fault == "duplicate":
        values["row_id"][-1] = values["row_id"][-2]
    elif fault == "ancestry":
        values["parent_world"][-1] = "foreign"
    elif fault == "status":
        values["status"][-1, 0] = "UNRECOGNIZED"
    elif fault == "cap":
        values = {k: np.concatenate([v, np.repeat(v[-1:], 8, axis=0)]) for k, v in values.items()}
        values["row_id"][-8:] = [f"n/p-0968/0/{i}" for i in range(4, 12)]
    with pytest.raises(subject.Refused):
        subject.project_synthetic_npz(fixture(tmp_path / "excluded-fault.npz", values), mapping)


@pytest.mark.parametrize("fault", ["missing_selected", "foreign"])
def test_selected_coverage_and_foreign_ids_refuse_with_amendment_report(
    tmp_path: Path, fault: str
) -> None:
    mapping, values = selected960_fixture(9)
    if fault == "missing_selected":
        values = {k: v[2:] for k, v in values.items()}
    else:
        values["life_id"][-1] = "n/alien/0"
        values["row_id"][-1] = "n/alien/0/3"
    with pytest.raises(subject.Refused) as error:
        subject.project_synthetic_npz(fixture(tmp_path / "coverage.npz", values), mapping)
    report = error.value.reconciliation
    assert not report["accepted"]
    key = "missing_ids" if fault == "missing_selected" else "extra_ids"
    assert len(report[key]) == 1


def test_excluded_sibling_cannot_supply_selected_ordinary_stratum(tmp_path: Path) -> None:
    mapping, values = selected960_fixture(9)
    keep = np.arange(len(values["ordinary"])) % 2 == 1
    keep[1920:] = True
    values = {k: v[keep] for k, v in values.items()}
    with pytest.raises(subject.Refused, match="both strata"):
        subject.project_synthetic_npz(fixture(tmp_path / "stratum.npz", values), mapping)


def test_interleaved_excluded_row_preserves_selected_source_index_and_order(tmp_path: Path) -> None:
    values = arrays()
    for key, value in list(values.items()):
        values[key] = np.concatenate([value[:1], value[:1], value[1:]])
    values["life_id"][1] = "n/p/1"
    values["row_id"][1] = "n/p/1/3"
    result = subject.project_synthetic_npz(
        fixture(tmp_path / "interleaved.npz", values), sibling_bindings()
    )
    rows = result["status_inventory"]
    assert len(rows) == 12
    assert [r["source_row_index"] for r in rows] == [0] * 6 + [2] * 6
    assert [r["row_order"] for r in rows] == [0] * 6 + [1] * 6
    assert {r["life_id"] for r in rows} == {"n/p/0"}


def test_frozen_selected_excluded_and_serialized_union_hashes() -> None:
    root = Path(subject.__file__).absolute().parents[2]
    raw = (root / subject.REVIEW_FOLDER / "extractor-input-inventory.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "cc20e1e7f5c321de5a1f63b94945c92c15c852a8499e88ab56659d1d2d7c1615"
    )
    mapping, exclusions = subject.validate_inventory(raw, root)
    assert len(mapping) == 969 and sum(not r["ordinary_only"] for r in mapping.values()) == 960
    assert len(exclusions) == 167
    inventory = json.loads(raw)
    prior = json.loads((root / subject.PRIOR_INVENTORY).read_bytes())
    union = [mapping[r["life_id"]] for r in prior["original_life_sources"]]
    assert hashlib.sha256(subject.canonical(union)).hexdigest() == subject.PRIOR_RECORDS_SHA
    assert (
        hashlib.sha256(
            subject.canonical(sorted(union, key=lambda r: r["allocation_order"]))
        ).hexdigest()
        != subject.PRIOR_RECORDS_SHA
    )
    assert len({r["parent_world"] for r in inventory["original_life_sources"]}) == 372


@pytest.mark.parametrize(
    "fault",
    [
        "selected",
        "excluded",
        "selected_pin",
        "excluded_pin",
        "counts",
        "evidence",
        "prior",
        "union",
        "exclusions",
    ],
)
def test_changed_frozen_inventory_refused(fault: str) -> None:
    root = Path(subject.__file__).absolute().parents[2]
    inventory = json.loads(
        (root / subject.REVIEW_FOLDER / "extractor-input-inventory.json").read_bytes()
    )
    if fault in ("selected", "excluded"):
        key = (
            "original_life_sources"
            if fault == "selected"
            else "excluded_ordinary_only_sibling_sources"
        )
        inventory[key][0]["allocation_order"] += 1
    elif fault in ("selected_pin", "excluded_pin"):
        key = (
            "original_life_sources_sha256"
            if fault == "selected_pin"
            else "excluded_ordinary_only_sibling_sources_sha256"
        )
        inventory[key] = "0" * 64
    elif fault == "counts":
        inventory["counts"]["total"] = 969
    elif fault == "evidence":
        inventory["retained_row_count_metadata"] = {}
    elif fault == "prior":
        inventory["prior_inventory_sha256"] = "0" * 64
    elif fault == "union":
        inventory["source_container_original_life_sources_sha256"] = "0" * 64
    elif fault == "exclusions":
        inventory["training_exclusions"] = []
    with pytest.raises(subject.Refused):
        subject.validate_inventory(subject.canonical(inventory), root)


def test_prior_inventory_bytes_checked_before_decode(tmp_path: Path) -> None:
    root = Path(subject.__file__).absolute().parents[2]
    raw = (root / subject.REVIEW_FOLDER / "extractor-input-inventory.json").read_bytes()
    prior = tmp_path / subject.PRIOR_INVENTORY
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"not json and not the pinned inventory")
    with pytest.raises(subject.Refused, match="prior inventory binding"):
        subject.validate_inventory(raw, tmp_path)


def test_CLI_refuses_unbound_receipt_paths_before_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_open(*args, **kwargs):
        raise AssertionError("unbound receipt path opened")

    monkeypatch.setattr(subject, "open_source", no_open)
    with pytest.raises(subject.Refused, match="fixed receipt"):
        subject.main(
            [
                "--acceptance",
                subject.SOURCE_PATH,
                "--authorization",
                subject.SOURCE_PATH,
                "--output",
                subject.PROPOSED_OUTPUT,
            ]
        )


def synthetic_acceptance(pins: dict) -> dict:
    return {
        "schema": "bl-f5-extractor-independent-acceptance/v2",
        "decision": "ACCEPT_STATUS_ONLY_EXTRACTOR",
        "independent_reviewer": "SYNTHETIC_ONLY",
        "ruling_sha256": subject.RULING_SHA,
        "reconciliation_ruling_sha256": subject.RECONCILIATION_RULING_SHA,
        "pins": pins,
    }


def synthetic_authorization(pins: dict) -> dict:
    return {
        "schema": "bl-f5-renewed-scan-authorization/v1",
        "decision": "AUTHORIZE_ONE_METADATA_SCAN",
        "authorized_by": "SYNTHETIC_ONLY",
        "pins": pins,
        "output_path": subject.PROPOSED_OUTPUT,
        "reconciliation_ruling_sha256": subject.RECONCILIATION_RULING_SHA,
    }


@pytest.mark.parametrize("fault", ["absent", "old_ruling", "old_output", "old_pins"])
def test_renewed_authority_is_required_and_bound(fault: str) -> None:
    pins = {key: "a" * 64 for key in ("source", "tests", "contract", "inventory")}
    authorization = synthetic_authorization(pins)
    subject.validate_authorization(authorization, pins)
    if fault == "absent":
        authorization = None
    elif fault == "old_ruling":
        authorization["reconciliation_ruling_sha256"] = subject.RULING_SHA
    elif fault == "old_output":
        authorization["output_path"] = "/historical-reservation.json"
    elif fault == "old_pins":
        authorization["pins"] = {**pins, "source": "b" * 64}
    with pytest.raises(subject.Refused, match="renewed"):
        subject.validate_authorization(authorization, pins)


def test_API_requires_renewal_before_real_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    root = Path(subject.__file__).absolute().parents[2]
    folder = root / subject.REVIEW_FOLDER
    paths = {
        "source": Path(subject.__file__),
        "tests": Path(__file__),
        "contract": folder / "extractor-contract.json",
        "inventory": folder / "extractor-input-inventory.json",
    }
    pins = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}

    def no_open(*args, **kwargs):
        raise AssertionError("real source opened without renewed authorization")

    monkeypatch.setattr(subject, "open_source", no_open)
    with pytest.raises(subject.Refused, match="renewed"):
        subject.extract_accepted(synthetic_acceptance(pins))


@pytest.mark.parametrize("refusal", [False, True])
def test_operator_entrypoint_synthetic_receipts_and_create_only_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refusal: bool,
) -> None:
    import json

    monkeypatch.setattr(subject, "RECEIPT_FOLDER", str(tmp_path))
    monkeypatch.setattr(subject, "PROPOSED_OUTPUT", str(tmp_path / "fresh.json"))
    pins = {key: "a" * 64 for key in ("source", "tests", "contract", "inventory")}
    for name, data in [
        ("acceptance", synthetic_acceptance(pins)),
        ("authorization", synthetic_authorization(pins)),
    ]:
        (tmp_path / f"{name}.json").write_bytes(subject.canonical(data))
    calls = []

    def synthetic_extract(acceptance, authorization, *, reservation):
        assert reservation.mode == "xb"
        reservation.write(b"SYNTHETIC reservation")
        reservation.flush()
        calls.append(True)
        if refusal:
            subject.reconcile_ids(
                {
                    **bindings(),
                    "n/p/1": {"parent_world": "p", "allocation_order": 18, "ordinary_only": False},
                },
                ["n/p/0"],
            )
        return {"SYNTHETIC_ONLY": True}

    monkeypatch.setattr(subject, "extract_accepted", synthetic_extract)
    args = [
        "--acceptance",
        str(tmp_path / "acceptance.json"),
        "--authorization",
        str(tmp_path / "authorization.json"),
        "--output",
        subject.PROPOSED_OUTPUT,
    ]
    assert subject.main(args) == (2 if refusal else 0)
    data = json.loads(Path(subject.PROPOSED_OUTPUT).read_bytes())
    if refusal:
        assert data["status"] == "REFUSED"
        assert data["id_reconciliation"]["missing_ids"] == ["n/p/1"]
    else:
        assert data == {"SYNTHETIC_ONLY": True}
    before = Path(subject.PROPOSED_OUTPUT).read_bytes()
    with pytest.raises(FileExistsError):
        subject.main(args)
    assert calls == [True]
    assert Path(subject.PROPOSED_OUTPUT).read_bytes() == before


def test_API_consumes_reservation_once_before_any_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    monkeypatch.setattr(subject, "PROPOSED_OUTPUT", str(tmp_path / "reservation.json"))
    root = Path(subject.__file__).absolute().parents[2]
    folder = root / subject.REVIEW_FOLDER
    paths = {
        "source": Path(subject.__file__),
        "tests": Path(__file__),
        "contract": folder / "extractor-contract.json",
        "inventory": folder / "extractor-input-inventory.json",
    }
    pins = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    acceptance, authorization = synthetic_acceptance(pins), synthetic_authorization(pins)
    calls = []
    original_open = subject.open_source

    def no_source(path):
        if path == root / subject.PRIOR_INVENTORY:
            return original_open(path)
        calls.append(True)
        subject.refuse("synthetic source stop")

    monkeypatch.setattr(subject, "open_source", no_source)
    with pytest.raises(subject.Refused, match="reservation"):
        subject.extract_accepted(acceptance, authorization)
    assert calls == []
    with Path(subject.PROPOSED_OUTPUT).open("xb") as output:
        with pytest.raises(subject.Refused, match="synthetic source stop"):
            subject.extract_accepted(acceptance, authorization, reservation=output)
        with pytest.raises(subject.Refused, match="reservation"):
            subject.extract_accepted(acceptance, authorization, reservation=output)
    assert calls == [True]
    assert Path(subject.PROPOSED_OUTPUT).stat().st_size > 0


def test_output_ancestor_substitution_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, elsewhere = tmp_path / "parent", tmp_path / "elsewhere"
    parent.mkdir()
    elsewhere.mkdir()
    original = subject.safe_path

    def substitute(path):
        original(path)
        parent.rename(tmp_path / "moved")
        parent.symlink_to(elsewhere, target_is_directory=True)

    monkeypatch.setattr(subject, "safe_path", substitute)
    with pytest.raises(OSError):
        subject.open_output(parent / "fresh.json")
    assert not (elsewhere / "fresh.json").exists()
    assert not (tmp_path / "moved" / "fresh.json").exists()


@pytest.mark.parametrize("field", ["ruling_sha256", "reconciliation_ruling_sha256"])
def test_acceptance_rejects_wrong_or_stale_ruling(field: str) -> None:
    pins = {key: "a" * 64 for key in ("source", "tests", "contract", "inventory")}
    receipt = synthetic_acceptance(pins)
    receipt[field] = "a77f026a07052cf86abf4094d73703276d68d6529b19d0b116b1739b8eaf44b7"
    with pytest.raises(subject.Refused, match="acceptance"):
        subject.validate_acceptance(receipt, pins)


def test_interrupted_CLI_keeps_consumed_output_and_cannot_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "RECEIPT_FOLDER", str(tmp_path))
    monkeypatch.setattr(subject, "PROPOSED_OUTPUT", str(tmp_path / "scan.json"))
    pins = {key: "a" * 64 for key in ("source", "tests", "contract", "inventory")}
    for name, data in [
        ("acceptance", synthetic_acceptance(pins)),
        ("authorization", synthetic_authorization(pins)),
    ]:
        (tmp_path / f"{name}.json").write_bytes(subject.canonical(data))
    calls = []

    def interrupted(acceptance, authorization, *, reservation):
        reservation.write(subject.canonical({"status": "RESERVED"}))
        reservation.flush()
        calls.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(subject, "extract_accepted", interrupted)
    args = [
        "--acceptance",
        str(tmp_path / "acceptance.json"),
        "--authorization",
        str(tmp_path / "authorization.json"),
        "--output",
        subject.PROPOSED_OUTPUT,
    ]
    with pytest.raises(KeyboardInterrupt):
        subject.main(args)
    before = Path(subject.PROPOSED_OUTPUT).read_bytes()
    assert json.loads(before)["status"] == "RESERVED"
    with pytest.raises(FileExistsError):
        subject.main(args)
    assert Path(subject.PROPOSED_OUTPUT).read_bytes() == before
    assert calls == [True]
