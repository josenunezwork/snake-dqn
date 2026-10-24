"""I3 sealed metadata fixtures only; no training, labels, rates or outcomes."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.bl_labels.d4096_design import (
    COLLECTION_DESIGN,
    OWNER_DECISION_PATH,
    OWNER_DECISION_SHA256,
)
from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import collection_receipts as receipts
from research.bl_residual import job
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import (
    B8_SHA,
    F4_REGISTRATION_SHA,
    allocation_hash,
    canonical_sha,
)

RECIPE = canonical_sha(dataclasses.asdict(FitRecipe()))
RESERVE = "a" * 64
SERIES = "synthetic-i3-series"


def sealed(path: Path, value: dict) -> Path:
    export_json(path, value)
    return path


def source_fixture(root: Path, dose: int, *, seconds: tuple[float, ...] = (3600, 7200)) -> dict:
    namespace = f"synthetic-collection-{dose}"
    attempts = []
    elapsed = 0
    terminals = []
    for index, seconds_ in enumerate(seconds, 1):
        attempt = root / f"attempt-{index}"
        attempts.append(str(attempt))
        terminals.append(f"attempt-{index}.terminal.json")
        value = {
            "schema": "bl-labels-f0-mac-driver/v1",
            "workers": 6,
            "namespace": namespace,
            "output": str(attempt),
            "status": "COMPLETE" if index == len(seconds) else "PREEMPTED",
            "pending_parents": [],
            "platform_attestation": {"platform": "darwin-arm64"},
            "elapsed_seconds": seconds_,
            "prior_elapsed_seconds": elapsed,
            "cumulative_elapsed_seconds": elapsed + seconds_,
            "start_receipt": f"attempt-{index}.start.json",
            "attempts_ledger": list(terminals),
            # These nonexistent paths must never be followed.
            "rate_sealed_path": "/forbidden/rates.json",
            "state_dir": "/forbidden/state",
        }
        sealed(attempt / "runtime-result.json", value)
        elapsed += seconds_
    ledger = sealed(
        root / "verified-ledger.json",
        {
            "schema": "bl-labels-f0-verified-ledger/v1",
            "namespace": namespace,
            "allocation": [{"allocation_id": "parent-0", "allocation_index": 0}],
            "attempt_exports": attempts,
            "units": [
                {
                    "unit_id": "parent-0--life-000000",
                    "path": "lives/parent-0--life-000000.json",
                    "sha256": "f" * 64,
                    "size": 10,
                }
            ],
        },
    )
    control = {
        "schema": "bl-residual-control/v1",
        "synthetic": False,
        "namespace": SERIES,
        "series_id": SERIES,
        "reserve_allocation_sha256": RESERVE,
        "recipe_sha256": RECIPE,
        "registration_sha256": F4_REGISTRATION_SHA,
        "doses": [dose],
        "factory_union_ledger_sha256": digest(ledger),
        "input_hashes": {"b8.pth": B8_SHA},
    }
    control["allocation"] = allocation_hash(control)
    fit_control = sealed(root / "fit-control.json", control)
    fit = sealed(
        root / "stage-fit.json",
        stage_fit(dose)
        | {
            "namespace": SERIES,
            "allocation": control["allocation"],
            "source_control_sha256": digest(fit_control),
        },
    )
    return {"ledger": ledger, "stage_fit": fit, "fit_control": fit_control, "dose": dose}


def stage_fit(dose: int) -> dict:
    return {
        "schema": "bl-residual-stage-fit/v1",
        "dose": dose,
        "fit_sha256": str(dose)[0] * 64,
        "recipe_sha256": RECIPE,
        "complete": True,
        "fit_fault_free": True,
    }


def collection(root: Path, dose: int) -> Path:
    source = source_fixture(root / "source", dose)
    out = root / "collection.json"
    receipts.write_collection_receipt(
        source["ledger"],
        digest(source["ledger"]),
        source["stage_fit"],
        out,
        fit_control=source["fit_control"],
        dose=dose,
        series_id=SERIES,
        reserve_allocation_sha256=RESERVE,
    )
    return out


def b2_fixture(collections: list[dict]) -> dict:
    walls = {str(c["dose"]): c["cumulative_wall_seconds"] for c in collections}
    return {
        "schema": receipts.MAC_PROJECTION_SCHEMA,
        "judgment_sha256": receipts.START_JUDGMENT_SHA,
        "placement": "mac",
        "verdict": "PASS",
        "selected_lives": 1024,
        "increment_lives": 3072,
        "margin": 1.2,
        "wall_cap_hours": 130,
        "collection_ledger_sha256": {
            str(c["dose"]): c["collection_ledger_sha256"] for c in collections
        },
        "cumulative_wall_seconds": walls,
        "projected_mac_hours": sum(walls.values()) / 3600 / 1024 * 3072 * 1.2,
    }


def projection_fixture(root: Path) -> tuple[list[Path], Path, Path]:
    paths = [collection(root / str(d), d) for d in (256, 1024)]
    b2 = sealed(root / "b2.json", b2_fixture([json.loads(p.read_text()) for p in paths]))
    output = root / "projection.json"
    receipts.write_projection_receipt(*paths, b2, output)
    return paths, b2, output


def staging(root: Path) -> dict:
    paths, b2, projection = projection_fixture(root)
    control = {
        "namespace": SERIES,
        "series_id": SERIES,
        "reserve_allocation_sha256": RESERVE,
        "synthetic": False,
        "factory_union_ledger_sha256": "c" * 64,
        "stage_receipts": {},
        "input_hashes": {},
    }
    bindings = {
        "D256_COLLECTION": paths[0],
        "D1024_COLLECTION": paths[1],
        "D4096_PROJECTION": projection,
        "D4096_MAC_PROJECTION": b2,
    }
    for dose in (256, 1024):
        prior = json.loads((root / str(dose) / "source/fit-control.json").read_text())
        prior_path = sealed(root / f"control-{dose}.json", prior)
        fit = json.loads((root / str(dose) / "source/stage-fit.json").read_text())
        bindings[f"D{dose}"] = sealed(root / f"fit-{dose}.json", fit)
        bindings[f"D{dose}_CONTROL"] = prior_path
    funding = {
        "schema": "bl-residual-funding-look/v1",
        "verdict": "PASS",
        "series_id": SERIES,
        "registration_sha256": F4_REGISTRATION_SHA,
        "reserve_allocation_sha256": RESERVE,
        "recipe_sha256": RECIPE,
        "d256_fit_sha256": stage_fit(256)["fit_sha256"],
        "d1024_fit_sha256": stage_fit(1024)["fit_sha256"],
    }
    bindings["D4096_FUNDING"] = sealed(root / "funding.json", funding)
    release = json.loads(
        (Path(__file__).parent / "fixtures/bl_residual_i3_collection_release.json").read_text()
    )
    assert release["schema"] == job.COLLECTION_DISPOSITION_SCHEMA
    release.update(
        series_id=SERIES,
        reserve_allocation_sha256=RESERVE,
        recipe_sha256=RECIPE,
        funding_receipt_sha256=digest(bindings["D4096_FUNDING"]),
    )
    bindings["D4096_COLLECTION_DISPOSITION"] = sealed(root / "release.json", release)
    for key, path in bindings.items():
        control["stage_receipts"][key] = str(path.relative_to(root))
        # The real pack flattens staging names. Fixtures copy to fresh basenames.
        if path.parent != root:
            flat = root / f"{key}.json"
            sealed(flat, json.loads(path.read_text()))
            path = flat
            control["stage_receipts"][key] = path.name
        control["input_hashes"][path.name] = digest(path)
    return control


def _replace_one(root: Path, control: dict, key: str, edits: dict) -> None:
    """Reseal fixture mutations to a fresh path, preserving prior evidence."""
    old = root / control["stage_receipts"][key]
    value = json.loads(old.read_text()) | edits
    new = root / f"modified-{old.name}"
    sealed(new, value)
    control["stage_receipts"][key] = new.name
    control["input_hashes"][new.name] = digest(new)


def bound_value(root: Path, control: dict, key: str) -> dict:
    return json.loads((root / control["stage_receipts"][key]).read_text())


def replace_bound(root: Path, control: dict, key: str, edits: dict) -> None:
    """Propagate dependent hashes/seals, keeping only the edited semantic defect.

    Direct edits of a dependent field remain deliberate binding-defect tests.
    Digest-tampering tests bypass this helper and corrupt bytes or sidecars.
    """
    old_sha = digest(root / control["stage_receipts"][key])
    _replace_one(root, control, key, edits)
    if key in ("D256", "D1024"):
        replace_bound(
            root,
            control,
            key + "_COLLECTION",
            {"stage_fit_receipt_sha256": digest(root / control["stage_receipts"][key])},
        )
    if key in ("D256_COLLECTION", "D1024_COLLECTION"):
        collections = [bound_value(root, control, f"D{d}_COLLECTION") for d in (256, 1024)]
        if "D4096_MAC_PROJECTION" in control["stage_receipts"]:
            b2 = b2_fixture(collections)
            replace_bound(
                root,
                control,
                "D4096_MAC_PROJECTION",
                {
                    k: b2[k]
                    for k in (
                        "collection_ledger_sha256",
                        "cumulative_wall_seconds",
                        "projected_mac_hours",
                    )
                },
            )
        _replace_one(
            root,
            control,
            "D4096_PROJECTION",
            {
                "collection_receipt_sha256": [
                    digest(root / control["stage_receipts"][f"D{d}_COLLECTION"])
                    for d in (256, 1024)
                ],
                "measurements": [
                    {
                        k: c[k]
                        for k in ("dose", "fit_sha256", "charged_units", "selected_rescue_lives")
                    }
                    for c in collections
                ],
            },
        )
    if key == "D4096_MAC_PROJECTION":
        projection = bound_value(root, control, "D4096_PROJECTION")
        new_sha = digest(root / control["stage_receipts"][key])
        _replace_one(
            root,
            control,
            "D4096_PROJECTION",
            {
                "mac_projection_receipt_sha256": new_sha,
                "qualified_placement_receipts": [
                    new_sha if sha == old_sha else sha
                    for sha in projection["qualified_placement_receipts"]
                ],
            },
        )
    if key == "D4096_FUNDING":
        _replace_one(
            root,
            control,
            "D4096_COLLECTION_DISPOSITION",
            {"funding_receipt_sha256": digest(root / control["stage_receipts"][key])},
        )


def assert_dependent_seals(root: Path, control: dict, *, skip_mac_binding: bool = False) -> None:
    """Semantic tests must reach the intended check with current digest bindings."""
    for name in control["stage_receipts"].values():
        assert verify_export(root / name) == control["input_hashes"][name]
    projection = bound_value(root, control, "D4096_PROJECTION")
    assert projection["collection_receipt_sha256"] == [
        digest(root / control["stage_receipts"][f"D{dose}_COLLECTION"]) for dose in (256, 1024)
    ]
    if not skip_mac_binding and "D4096_MAC_PROJECTION" in control["stage_receipts"]:
        b2_sha = digest(root / control["stage_receipts"]["D4096_MAC_PROJECTION"])
        assert projection["mac_projection_receipt_sha256"] == b2_sha
        assert b2_sha in projection["qualified_placement_receipts"]
    assert bound_value(root, control, "D4096_COLLECTION_DISPOSITION")[
        "funding_receipt_sha256"
    ] == digest(root / control["stage_receipts"]["D4096_FUNDING"])


def test_collection_is_metadata_only_and_create_only(tmp_path: Path, monkeypatch) -> None:
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        assert "forbidden" not in str(path)
        assert path.parent.name != "lives"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    path = collection(tmp_path, 1024)
    value = json.loads(path.read_text())
    assert verify_export(path) == digest(path)
    assert value["charged_units"] == 0
    assert value["selected_rescue_lives"] == 768
    assert value["cumulative_wall_seconds"] == 10800
    assert len(value["terminal_receipts"]) == 2
    source = tmp_path / "source"
    with pytest.raises(FileExistsError):
        receipts.write_collection_receipt(
            source / "verified-ledger.json",
            digest(source / "verified-ledger.json"),
            source / "stage-fit.json",
            path,
            fit_control=source / "fit-control.json",
            dose=1024,
            series_id=SERIES,
            reserve_allocation_sha256=RESERVE,
        )


@pytest.mark.parametrize(
    "edit",
    [
        {"status": "RUNNING"},
        {"invalid_reason": "fault"},
        {"registered_stop_reason": "STOP"},
        {"namespace": "foreign"},
        {"output": "/foreign"},
        {"pending_parents": ["p"]},
        {"elapsed_seconds": True},
        {"elapsed_seconds": -1},
        {"elapsed_seconds": 0},
        {"prior_elapsed_seconds": 1},
        {"cumulative_elapsed_seconds": 1},
        {"attempts_ledger": []},
        {"platform_attestation": {"platform": "linux"}},
        {"charged_units": 1},
        {"charged_units": None},
        {"charged_units": False},
    ],
)
def test_collection_refuses_bad_terminal_metadata(tmp_path: Path, edit: dict) -> None:
    source = source_fixture(tmp_path / "source", 256, seconds=(3600,))
    old = tmp_path / "source/attempt-1/runtime-result.json"
    # Mutation deliberately represents hostile source bytes, then is resealed.
    for suffix in ("", ".sha256", ".complete"):
        Path(str(old) + suffix).unlink()
    terminal = {
        "schema": "bl-labels-f0-mac-driver/v1",
        "workers": 6,
        "namespace": "synthetic-collection-256",
        "output": str(old.parent),
        "status": "COMPLETE",
        "pending_parents": [],
        "platform_attestation": {"platform": "darwin-arm64"},
        "elapsed_seconds": 3600,
        "prior_elapsed_seconds": 0,
        "cumulative_elapsed_seconds": 3600,
        "start_receipt": "attempt-1.start.json",
        "attempts_ledger": ["attempt-1.terminal.json"],
    }
    sealed(old, terminal | edit)
    out = tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        receipts.write_collection_receipt(
            source["ledger"],
            digest(source["ledger"]),
            source["stage_fit"],
            out,
            fit_control=source["fit_control"],
            dose=256,
            series_id=SERIES,
            reserve_allocation_sha256=RESERVE,
        )
    assert not out.exists()


def test_seals_and_ledger_pin_are_required(tmp_path: Path) -> None:
    s = source_fixture(tmp_path, 256)
    with pytest.raises(ValueError, match="foreign"):
        receipts.write_collection_receipt(
            s["ledger"],
            "0" * 64,
            s["stage_fit"],
            tmp_path / "bad.json",
            fit_control=s["fit_control"],
            dose=256,
            series_id=SERIES,
            reserve_allocation_sha256=RESERVE,
        )
    Path(str(s["ledger"]) + ".complete").unlink()
    with pytest.raises(ValueError, match="missing"):
        receipts.write_collection_receipt(
            s["ledger"],
            digest(s["ledger"]),
            s["stage_fit"],
            tmp_path / "bad.json",
            fit_control=s["fit_control"],
            dose=256,
            series_id=SERIES,
            reserve_allocation_sha256=RESERVE,
        )


def test_zero_mac_projection_passes_all_original_staging_guards(tmp_path: Path) -> None:
    control = staging(tmp_path)
    job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize(
    "key,edit",
    [
        ("D4096_FUNDING", {"verdict": "STOP_FUNDING"}),
        ("D4096_FUNDING", {"series_id": "foreign"}),
        ("D256", {"fit_fault_free": False}),
        ("D1024", {"complete": False}),
        ("D256_COLLECTION", {"placement": "l4"}),
        ("D1024_COLLECTION", {"charged_units": 0.1}),
        ("D256_COLLECTION", {"charged_units": False}),
        ("D256_COLLECTION", {"fit_sha256": "f" * 64}),
        ("D4096_PROJECTION", {"projected_units": -1}),
        ("D4096_PROJECTION", {"projected_units": True}),
        ("D4096_PROJECTION", {"projected_units": 481}),
        ("D4096_PROJECTION", {"projected_units": 1}),
        ("D4096_PROJECTION", {"qualified_placement_tree": "foreign"}),
        ("D4096_PROJECTION", {"verdict": "HOLD"}),
        ("D4096_PROJECTION", {"mac_projection_receipt_sha256": "f" * 64}),
        ("D4096_MAC_PROJECTION", {"schema": "foreign"}),
        ("D4096_MAC_PROJECTION", {"judgment_sha256": "f" * 64}),
        ("D4096_MAC_PROJECTION", {"selected_lives": 1023}),
        ("D4096_MAC_PROJECTION", {"projected_mac_hours": 131}),
        ("D4096_MAC_PROJECTION", {"wall_cap_hours": 131}),
        ("D4096_MAC_PROJECTION", {"collection_ledger_sha256": {}}),
    ],
)
def test_staging_adversarial_refusals(tmp_path: Path, key: str, edit: dict) -> None:
    control = staging(tmp_path)
    replace_bound(tmp_path, control, key, edit)
    assert_dependent_seals(
        tmp_path, control, skip_mac_binding="mac_projection_receipt_sha256" in edit
    )
    with pytest.raises(ValueError):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


def test_missing_b2_refuses(tmp_path: Path) -> None:
    control = staging(tmp_path)
    missing = copy.deepcopy(control)
    missing["stage_receipts"].pop("D4096_MAC_PROJECTION")
    with pytest.raises(ValueError, match="projection"):
        job._validate_staging(tmp_path, missing, [4096], RECIPE, set())


@pytest.mark.parametrize("key", ["D4096_MAC_PROJECTION", "D256_COLLECTION", "D4096_FUNDING"])
def test_staging_digest_tamper_refuses(tmp_path: Path, key: str) -> None:
    control = staging(tmp_path)
    path = tmp_path / control["stage_receipts"][key]
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("key", ["D4096_FUNDING", "D4096_COLLECTION_DISPOSITION"])
@pytest.mark.parametrize("suffix", [".sha256", ".complete"])
def test_required_authority_receipts_must_be_sealed(tmp_path: Path, key: str, suffix: str) -> None:
    control = staging(tmp_path)
    path = tmp_path / control["stage_receipts"][key]
    Path(str(path) + suffix).unlink()
    with pytest.raises(ValueError, match="missing"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize(
    "units,projected,passes",
    [
        (10, 72, True),
        (480 / 7.2, 480, True),
        (70, 504, False),
        (1e-12, 7.2e-12, True),
        (1e-12, 0, False),
    ],
)
def test_original_positive_unit_projection_still_passes(
    tmp_path: Path, units: float, projected: float, passes: bool
) -> None:
    control = staging(tmp_path)
    # Legacy positive-unit receipts require no Mac exception or B2 binding.
    control["stage_receipts"].pop("D4096_MAC_PROJECTION")
    for key in ("D256_COLLECTION", "D1024_COLLECTION"):
        replace_bound(tmp_path, control, key, {"charged_units": units, "placement": "l4"})
    replace_bound(tmp_path, control, "D4096_PROJECTION", {"projected_units": projected})
    assert_dependent_seals(tmp_path, control)
    if passes:
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())
    else:
        with pytest.raises(ValueError, match="projection"):
            job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("verdict", ["PASS", "STOP_FUNDING", None])
@pytest.mark.parametrize("published", [True, False])
def test_direct_f2_cannot_replace_canonical_pass_funding(
    tmp_path: Path, published: bool, verdict: str | None
) -> None:
    control = staging(tmp_path)
    replace_bound(
        tmp_path,
        control,
        "D4096_FUNDING",
        {
            "schema": "bl-residual-f2-disposition/v1",
            "verdict": verdict,
            "volume_struck": False,
            "published": published,
            "published_ref": "synthetic-published-ref",
            "source_sha256": "f" * 64,
        },
    )
    assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match="STOP_FUNDING"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize(
    "placement,has_b2", [("mac", True), ("mac", False), ("l4", True), ("l4", False)]
)
@pytest.mark.parametrize("projected", [1e-10, 1e-12])
def test_zero_measured_charges_cannot_hide_behind_tiny_positive_projection(
    tmp_path: Path, placement: str, has_b2: bool, projected: float
) -> None:
    control = staging(tmp_path)
    for dose in (256, 1024):
        replace_bound(tmp_path, control, f"D{dose}_COLLECTION", {"placement": placement})
    if not has_b2:
        control["stage_receipts"].pop("D4096_MAC_PROJECTION")
    replace_bound(tmp_path, control, "D4096_PROJECTION", {"projected_units": projected})
    assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match="projection receipt did not pass"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("verdict", [None, "STOP_FUNDING", "HOLD", "INVALID"])
def test_canonical_funding_requires_exact_pass(tmp_path: Path, verdict: str | None) -> None:
    control = staging(tmp_path)
    replace_bound(tmp_path, control, "D4096_FUNDING", {"verdict": verdict})
    assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match="STOP_FUNDING"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


def test_missing_funding_refuses_otherwise_valid_staging(tmp_path: Path) -> None:
    control = staging(tmp_path)
    assert_dependent_seals(tmp_path, control)
    control["stage_receipts"].pop("D4096_FUNDING")
    with pytest.raises(ValueError, match="requires funding"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("disposition", ["PENDING", "PENDING_SEALED", "DISCARDED", "unknown", None])
def test_collection_must_be_released(tmp_path: Path, disposition: str | None) -> None:
    control = staging(tmp_path)
    replace_bound(tmp_path, control, "D4096_COLLECTION_DISPOSITION", {"disposition": disposition})
    assert_dependent_seals(tmp_path, control)
    with pytest.raises(
        ValueError, match="NOT_RUN_BY_RULE" if disposition == "DISCARDED" else "HOLD"
    ):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "foreign"),
        ("judgment_sha256", "f" * 64),
        ("series_id", "foreign"),
        ("registration_sha256", "f" * 64),
        ("reserve_allocation_sha256", "f" * 64),
        ("recipe_sha256", "f" * 64),
        ("collection_ledger_sha256", "f" * 64),
        ("funding_receipt_sha256", "f" * 64),
    ],
)
def test_foreign_collection_release_refuses(tmp_path: Path, field: str, value: str) -> None:
    control = staging(tmp_path)
    replace_bound(tmp_path, control, "D4096_COLLECTION_DISPOSITION", {field: value})
    if field != "funding_receipt_sha256":
        assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match="foreign D4096 collection release"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


def test_missing_collection_release_refuses(tmp_path: Path) -> None:
    control = staging(tmp_path)
    control["stage_receipts"].pop("D4096_COLLECTION_DISPOSITION")
    with pytest.raises(ValueError, match="released collection disposition"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


def discard_fixture(root: Path, control: dict, *, bound: bool) -> None:
    value = json.loads(
        (Path(__file__).parent / "fixtures/bl_residual_i3_terminal_discard.json").read_text()
    )
    assert value["schema"] == job.DISCARD_SCHEMA
    path = sealed(root / ("alternate-discard.json" if bound else job.DISCARD_FILENAME), value)
    if bound:
        control["stage_receipts"]["D4096_DISCARD"] = path.name
        control["input_hashes"][path.name] = digest(path)


@pytest.mark.parametrize("bound", [True, False])
def test_terminal_discard_wins_over_pass_and_missing_funding(tmp_path: Path, bound: bool) -> None:
    control = staging(tmp_path)
    assert_dependent_seals(tmp_path, control)
    discard_fixture(tmp_path, control, bound=bound)
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())
    control["stage_receipts"].pop("D4096_FUNDING")
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


def test_discarded_collection_wins_over_missing_funding(tmp_path: Path) -> None:
    control = staging(tmp_path)
    replace_bound(tmp_path, control, "D4096_COLLECTION_DISPOSITION", {"disposition": "DISCARDED"})
    assert_dependent_seals(tmp_path, control)
    control["stage_receipts"].pop("D4096_FUNDING")
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("entrypoint", ["preflight", "worker"])
@pytest.mark.parametrize("state", ["terminal", "PENDING_SEALED", "DISCARDED", "missing"])
def test_pack_and_worker_refuse_before_any_payload_hash_or_output(
    tmp_path: Path, monkeypatch, entrypoint: str, state: str
) -> None:
    control = staging(tmp_path)
    control.update(schema="bl-residual-control/v1", doses=[4096])
    if state == "terminal":
        discard_fixture(tmp_path, control, bound=False)
    elif state == "missing":
        control["stage_receipts"].pop("D4096_COLLECTION_DISPOSITION")
    else:
        replace_bound(tmp_path, control, "D4096_COLLECTION_DISPOSITION", {"disposition": state})
    (tmp_path / "control.json").write_text(json.dumps(control))
    original = job.digest

    def guarded(path: Path) -> str:
        assert path.name in control["stage_receipts"].values(), "payload opened before refusal"
        return original(path)

    monkeypatch.setattr(job, "digest", guarded)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE|HOLD"):
        if entrypoint == "preflight":
            job.preflight_main_dispatch(tmp_path)
        else:
            job.run(tmp_path, output, mode="main")
    assert not output.exists()


@pytest.mark.parametrize("component", ["payload", "sha256", "complete", "dangling-symlink"])
def test_partial_or_malformed_terminal_discard_is_fail_closed(
    tmp_path: Path, component: str
) -> None:
    control = staging(tmp_path)
    path = tmp_path / job.DISCARD_FILENAME
    if component == "dangling-symlink":
        path.symlink_to(tmp_path / "absent")
    else:
        candidate = path if component == "payload" else Path(str(path) + "." + component)
        candidate.write_text("malformed")
    with pytest.raises(ValueError, match="NOT_RUN_BY_RULE"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


def test_b2_exact_130_hour_boundary_and_hold() -> None:
    collections = [
        {"dose": d, "cumulative_wall_seconds": 65000, "collection_ledger_sha256": str(d)[0] * 64}
        for d in (256, 1024)
    ]
    b2 = b2_fixture(collections)
    assert b2["projected_mac_hours"] == 130
    receipts.validate_mac_projection(b2, collections)
    collections[1]["cumulative_wall_seconds"] += 1
    with pytest.raises(ValueError, match="HOLD"):
        receipts.validate_mac_projection(b2_fixture(collections), collections)


def test_i1_b2_schema_fixture() -> None:
    value = json.loads(
        (Path(__file__).parent / "fixtures/bl_residual_i3_b2_mac_projection.json").read_text()
    )
    collections = [
        {"dose": dose, "cumulative_wall_seconds": 10800, "collection_ledger_sha256": sha}
        for dose, sha in ((256, "2" * 64), (1024, "1" * 64))
    ]
    receipts.validate_mac_projection(value, collections)


def test_projection_discloses_mixed_platform_design_and_observed_placement(tmp_path: Path) -> None:
    _, _, output = projection_fixture(tmp_path)
    projection = json.loads(output.read_text())

    assert projection["collection_design"] == COLLECTION_DESIGN
    assert projection["owner_decision"] == {
        "path": OWNER_DECISION_PATH,
        "sha256": OWNER_DECISION_SHA256,
    }
    assert projection["actual_execution_placement"] == {
        "source_collections": [{"platform": "mac", "doses": [256, 1024]}],
        "l4_ledger_status": "ABSENT",
        "l4_accounted_attempt_count": 0,
    }


def test_l4_tail_units_are_separate_and_bound(tmp_path: Path) -> None:
    control = staging(tmp_path)
    l4 = {
        "schema": receipts.L4_ACCOUNTING_SCHEMA,
        "series_id": SERIES,
        "registration_sha256": F4_REGISTRATION_SHA,
        "dose": 4096,
        "attempts": [{"receipt_sha256": "c" * 64, "charged_units": 700}],
    }
    path = sealed(tmp_path / "l4.json", l4)
    bound_projection = tmp_path / "bound-l4-projection.json"
    receipts.write_projection_receipt(
        tmp_path / control["stage_receipts"]["D256_COLLECTION"],
        tmp_path / control["stage_receipts"]["D1024_COLLECTION"],
        tmp_path / control["stage_receipts"]["D4096_MAC_PROJECTION"],
        bound_projection,
        l4_ledger=path,
    )
    assert json.loads(bound_projection.read_text())["actual_execution_placement"] == {
        "source_collections": [{"platform": "mac", "doses": [256, 1024]}],
        "l4_ledger_status": "BOUND",
        "l4_accounted_attempt_count": 1,
    }
    control["stage_receipts"]["D4096_L4_ACCOUNTING"] = path.name
    control["input_hashes"][path.name] = digest(path)
    replace_bound(
        tmp_path,
        control,
        "D4096_PROJECTION",
        {
            "l4_tail_accounting": receipts.l4_accounting(l4, SERIES),
            "l4_tail_accounting_sha256": digest(path),
        },
    )
    job._validate_staging(tmp_path, control, [4096], RECIPE, set())
    projection = json.loads((tmp_path / control["stage_receipts"]["D4096_PROJECTION"]).read_text())
    assert projection["projected_units"] == 0
    assert projection["l4_tail_accounting"]["charged_units"] == 700
    assert projection["actual_execution_placement"] == {
        "source_collections": [{"platform": "mac", "doses": [256, 1024]}],
        "l4_ledger_status": "ABSENT",
        "l4_accounted_attempt_count": 0,
    }
    control["stage_receipts"].pop("D4096_L4_ACCOUNTING")
    with pytest.raises(ValueError, match="L4"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("units", [None, False, -1, float("inf"), float("nan"), 960.001])
def test_l4_unknown_and_excess_charge_refused(units) -> None:
    value = {
        "schema": receipts.L4_ACCOUNTING_SCHEMA,
        "series_id": SERIES,
        "registration_sha256": F4_REGISTRATION_SHA,
        "dose": 4096,
        "attempts": [{"receipt_sha256": "c" * 64, "charged_units": units}],
    }
    with pytest.raises(ValueError):
        receipts.l4_accounting(value, SERIES)


def test_l4_duplicate_charge_refused() -> None:
    row = {"receipt_sha256": "c" * 64, "charged_units": 1}
    value = {
        "schema": receipts.L4_ACCOUNTING_SCHEMA,
        "series_id": SERIES,
        "registration_sha256": F4_REGISTRATION_SHA,
        "dose": 4096,
        "attempts": [row, row],
    }
    with pytest.raises(ValueError, match="duplicate"):
        receipts.l4_accounting(value, SERIES)


def test_foreign_collection_ledger_cannot_be_attached_to_a_passed_fit(tmp_path: Path) -> None:
    a = source_fixture(tmp_path / "a", 256)
    b = source_fixture(tmp_path / "b", 1024)
    with pytest.raises(ValueError, match="foreign"):
        receipts.write_collection_receipt(
            a["ledger"],
            digest(a["ledger"]),
            b["stage_fit"],
            tmp_path / "bad.json",
            fit_control=b["fit_control"],
            dose=1024,
            series_id=SERIES,
            reserve_allocation_sha256=RESERVE,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("collection_ledger_sha256", "f" * 64),
        ("source_control_sha256", "f" * 64),
        ("series_id", "foreign"),
        ("registration_sha256", "f" * 64),
        ("reserve_allocation_sha256", "f" * 64),
        ("qualified_tree", "foreign"),
    ],
)
def test_collection_provenance_cannot_be_rebound(tmp_path: Path, field: str, value) -> None:
    control = staging(tmp_path)
    replace_bound(tmp_path, control, "D256_COLLECTION", {field: value})
    assert_dependent_seals(tmp_path, control)
    with pytest.raises(ValueError, match="projection receipt did not pass"):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize("tail", [None, {}])
def test_bound_l4_accounting_cannot_be_silently_omitted(tmp_path: Path, tail) -> None:
    control = staging(tmp_path)
    path = sealed(
        tmp_path / "l4.json",
        {
            "schema": receipts.L4_ACCOUNTING_SCHEMA,
            "series_id": SERIES,
            "registration_sha256": F4_REGISTRATION_SHA,
            "dose": 4096,
            "attempts": [{"receipt_sha256": "c" * 64, "charged_units": 1000}],
        },
    )
    control["stage_receipts"]["D4096_L4_ACCOUNTING"] = path.name
    control["input_hashes"][path.name] = digest(path)
    replace_bound(tmp_path, control, "D4096_PROJECTION", {"l4_tail_accounting": tail})
    with pytest.raises(ValueError):
        job._validate_staging(tmp_path, control, [4096], RECIPE, set())


@pytest.mark.parametrize(
    "driver,delta,allowed",
    [
        (receipts.LEGACY_D256_DRIVER_SHA, 0.0001, True),
        (receipts.LEGACY_D256_DRIVER_SHA, 0.01, False),
        ("f" * 64, 0.0001, False),
    ],
)
def test_legacy_d256_clock_compatibility_is_exactly_scoped(
    tmp_path: Path, driver: str, delta: float, allowed: bool
) -> None:
    source = source_fixture(tmp_path, 256, seconds=(3600,))
    path = tmp_path / "attempt-1/runtime-result.json"
    terminal = json.loads(path.read_text())
    terminal["cumulative_elapsed_seconds"] += delta
    terminal["platform_attestation"]["driver_sha256"] = driver
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    sealed(path, terminal)

    def produce():
        return receipts.write_collection_receipt(
            source["ledger"],
            digest(source["ledger"]),
            source["stage_fit"],
            tmp_path / "result.json",
            fit_control=source["fit_control"],
            dose=256,
            series_id=SERIES,
            reserve_allocation_sha256=RESERVE,
        )

    if allowed:
        assert produce()["cumulative_wall_seconds"] == 3600
    else:
        with pytest.raises(ValueError, match="cumulative"):
            produce()


def test_real_cli_on_synthetic_metadata(tmp_path: Path) -> None:
    paths = []
    for dose in (256, 1024):
        source = source_fixture(tmp_path / str(dose), dose)
        out = tmp_path / f"collection-{dose}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.bl_residual.collection_receipts",
                "collection",
                "--ledger",
                str(source["ledger"]),
                "--ledger-sha256",
                digest(source["ledger"]),
                "--stage-fit",
                str(source["stage_fit"]),
                "--fit-control",
                str(source["fit_control"]),
                "--dose",
                str(dose),
                "--series-id",
                SERIES,
                "--reserve-allocation-sha256",
                RESERVE,
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        paths.append(out)
    b2 = sealed(tmp_path / "b2.json", b2_fixture([json.loads(p.read_text()) for p in paths]))
    out = tmp_path / "projection.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.bl_residual.collection_receipts",
            "projection",
            "--d256",
            str(paths[0]),
            "--d1024",
            str(paths[1]),
            "--mac-projection",
            str(b2),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert verify_export(out) == digest(out)
    assert json.loads(out.read_text())["projected_units"] == 0
