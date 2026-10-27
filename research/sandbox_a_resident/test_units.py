"""Immutable completion, skip/resume identity and corruption refusal."""

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident.units import (
    allocation,
    load_ledger,
    publish,
    unit_id,
    verified_descriptor,
)


def test_immutable_publication_and_metadata_only_ledger(tmp_path: object) -> None:
    control = {"worlds": [2], "cycles": 3}
    path = tmp_path / "unit.tar"
    publish(path, lambda p: p.write_bytes(b"opaque-checkpoint-bytes"))
    descriptor = verified_descriptor(path, unit_id(2, "integrated"))
    timing_path = path.with_suffix(".json")
    publish(timing_path, lambda p: p.write_bytes(b"{}"))
    descriptor["timing"] = verified_descriptor(timing_path, unit_id(2, "integrated"))
    ledger = tmp_path / "ledger.json"
    payload = {
        "schema": "sandbox-a-ledger/v1",
        "namespace": "a100-dev/sandbox-a",
        "allocation": allocation(control),
        "units": [descriptor],
    }
    ledger.write_text(json.dumps(payload))
    assert list(load_ledger(ledger, control)) == [unit_id(2, "integrated")]
    with pytest.raises(FileExistsError):
        publish(path, lambda p: p.write_bytes(b"overwrite"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        verified_descriptor(path, unit_id(2, "integrated"))
    payload["units"].append(descriptor)
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_ledger(ledger, control)
    payload["units"] = []
    payload["namespace"] = "foreign"
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(ledger, control)


def receipt_for(control: dict, directory: Path, exports: list[dict]) -> Path:
    """Create the metadata envelope used by ledger-only corruption tests."""
    from research.sandbox_a_resident.units import digest

    manifest = directory.parents[1] / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "commit": control["source_commit"],
                "env": {"SANDBOX_A_CONTROL_JSON": json.dumps(control)},
            }
        )
    )
    receipt = manifest.with_name("receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "commit": control["source_commit"],
                "stop_verified": True,
                "manifest_sha256": digest(manifest),
                "exports": exports,
            }
        )
    )
    return receipt


def completed_cached_receipt(root: Path) -> tuple[dict, Path, Path]:
    """Create one complete cache unit for input-path guard tests."""
    control = {"worlds": [2], "source_commit": "a" * 40}
    directory = root / "exports" / "assay"
    directory.mkdir(parents=True)
    checkpoint = publish(directory / "e2-cached.tar", lambda p: p.write_bytes(b"checkpoint"))
    timing = publish(directory / "e2-cached.json", lambda p: p.write_bytes(b"{}"))
    receipt = receipt_for(
        control,
        directory,
        [
            {**checkpoint, "path": "assay/e2-cached.tar"},
            {**timing, "path": "assay/e2-cached.json"},
        ],
    )
    return control, directory, receipt


@pytest.mark.parametrize("summary_invalid", [False, True])
def test_invalid_cell_or_summary_cannot_become_qualified_throughput(
    tmp_path: Path, summary_invalid: bool
) -> None:
    from research.sandbox_a_resident.postprocess import aggregate
    from research.sandbox_a_resident.units import digest

    control, directory, receipt_path = completed_cached_receipt(tmp_path)
    summary = {
        "allocation": allocation(control),
        "source_commit": control["source_commit"],
        "parity": {"status": "PASS"},
        "status": "INVALID" if summary_invalid else "PASS",
        "results": [{"status": "PASS" if summary_invalid else "INVALID"}],
    }
    exported = [
        publish(directory / name, lambda path, value=value: path.write_text(json.dumps(value)))
        for name, value in (("summary.json", summary), ("parity.json", {"status": "PASS"}))
    ]
    receipt = json.loads(receipt_path.read_text())
    receipt.update(status="PASS", endpoint="session-1", elapsed_seconds=60)
    receipt["exports"] += [{**row, "path": "assay/" + Path(row["path"]).name} for row in exported]
    receipt["manifest_sha256"] = digest(receipt_path.with_name("manifest.json"))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="INVALID cell/sweep"):
        aggregate(control, [receipt_path])


def test_partial_export_pair_and_old_source_cannot_become_verified(tmp_path: Path) -> None:
    from research.sandbox_a_resident.postprocess import ledger

    control = {"worlds": [2], "source_commit": "a" * 40}
    directory = tmp_path / "exports" / "assay"
    directory.mkdir(parents=True)
    checkpoint = publish(directory / "e2-cached.tar", lambda p: p.write_bytes(b"checkpoint"))
    timing = publish(directory / "e2-cached.json", lambda p: p.write_bytes(b"{}"))
    interrupted = publish(directory / "e2-integrated.tar", lambda p: p.write_bytes(b"partial"))
    (directory / "e2-integrated.json.partial").write_bytes(b"interrupted timing export")
    parity = publish(directory / "parity.json", lambda p: p.write_bytes(b"{}"))
    receipt = receipt_for(
        control,
        directory,
        [
            {**checkpoint, "path": "assay/e2-cached.tar"},
            {**timing, "path": "assay/e2-cached.json"},
            {**interrupted, "path": "assay/e2-integrated.tar"},
            {**parity, "path": "assay/parity.json"},
        ],
    )
    saved = ledger(control, [receipt])["units"]
    assert [row["id"] for row in saved] == [unit_id(2, "cached")]
    with pytest.raises(ValueError, match="metadata differs"):
        ledger({**control, "source_commit": "b" * 40}, [receipt])
    with pytest.raises(ValueError, match="protected"):
        ledger(control, [tmp_path / "f5-state"])


def test_ledger_refuses_half_markers_foreign_and_duplicate_exports(tmp_path: Path) -> None:
    from research.sandbox_a_resident.postprocess import ledger

    control = {"worlds": [2], "source_commit": "a" * 40}
    directory = tmp_path / "exports" / "assay"
    directory.mkdir(parents=True)
    timing = publish(directory / "e2-cached.json", lambda p: p.write_bytes(b"{}"))
    receipt = receipt_for(control, directory, [{**timing, "path": "assay/e2-cached.json"}])
    with pytest.raises(ValueError, match="timing completion"):
        ledger(control, [receipt])

    publish(directory / "e99-cached.tar", lambda p: p.write_bytes(b"foreign"))
    receipt = receipt_for(control, directory, [{**timing, "path": "assay/e2-cached.json"}])
    with pytest.raises(ValueError, match="foreign assay"):
        ledger(control, [receipt])


def test_ledger_refuses_foreign_or_duplicate_inventory_and_linked_paths(tmp_path: Path) -> None:
    from research.sandbox_a_resident.postprocess import ledger

    control = {"worlds": [2], "source_commit": "a" * 40}
    directory = tmp_path / "exports" / "assay"
    directory.mkdir(parents=True)
    checkpoint = publish(directory / "e2-cached.tar", lambda p: p.write_bytes(b"checkpoint"))
    timing = publish(directory / "e2-cached.json", lambda p: p.write_bytes(b"{}"))
    valid = [
        {**checkpoint, "path": "assay/e2-cached.tar"},
        {**timing, "path": "assay/e2-cached.json"},
    ]
    receipt = receipt_for(
        control, directory, valid + [{**checkpoint, "path": "assay/e2-foreign.tar"}]
    )
    with pytest.raises(ValueError, match="foreign receipt"):
        ledger(control, [receipt])

    receipt = receipt_for(control, directory, valid + [valid[0]])
    with pytest.raises(ValueError, match="duplicate receipt"):
        ledger(control, [receipt])

    linked = directory / "e2-collection.tar"
    linked.symlink_to(directory / "e2-cached.tar")
    receipt = receipt_for(control, directory, valid)
    with pytest.raises(ValueError, match="symlink"):
        ledger(control, [receipt])


def test_ledger_refuses_symlink_parents_manifest_and_sidecars(tmp_path: Path) -> None:
    from research.sandbox_a_resident.postprocess import ledger

    control, _, receipt = completed_cached_receipt(tmp_path / "parent-source")
    linked_parent = tmp_path / "linked-receipt-parent"
    linked_parent.symlink_to(receipt.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ledger(control, [linked_parent / receipt.name])

    control, _, receipt = completed_cached_receipt(tmp_path / "manifest-link")
    manifest = receipt.with_name("manifest.json")
    target = receipt.with_name("manifest-target.json")
    target.write_text(manifest.read_text())
    manifest.unlink()
    manifest.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        ledger(control, [receipt])

    control, directory, receipt = completed_cached_receipt(tmp_path / "sidecar-link")
    sidecar = Path(str(directory / "e2-cached.tar") + ".sha256")
    target = directory.parent / "sidecar-target.sha256"
    target.write_text(sidecar.read_text())
    sidecar.unlink()
    sidecar.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        ledger(control, [receipt])


def test_path_guard_allows_harmless_hex_component(tmp_path: Path) -> None:
    from research.sandbox_a_resident.postprocess import allowed_path

    harmless = tmp_path / "archive-f5a9"
    harmless.mkdir()
    assert allowed_path(harmless) == harmless.resolve()


def test_source_or_recipe_drift_changes_verified_allocation() -> None:
    control = {
        "worlds": [128, 1024],
        "source_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "recipe_parameters": {"gamma": 0.99},
    }
    assert allocation(control) != allocation({**control, "source_commit": "c" * 40})
    assert allocation(control) != allocation({**control, "recipe_parameters": {"gamma": 0.9}})


def test_a2_sweep_has_distinct_resume_units_and_exports(tmp_path: Path) -> None:
    from research.sandbox_a_resident.postprocess import expected_stems, ledger
    from research.sandbox_a_resident.units import cell_specs, unit_stem

    control = {
        "worlds": [2],
        "episode_limits": [64, 512],
        "seed_offset_base": 20_000_000,
        "source_commit": "a" * 40,
    }
    cells = cell_specs(control)
    assert len(cells) == 6
    assert len({unit_id(e, p, limit) for e, p, limit, _ in cells}) == 6
    assert [row[3] for row in cells] == list(range(20_000_000, 26_000_000, 1_000_000))
    assert expected_stems(control) == {unit_stem(e, p, limit) for e, p, limit, _ in cells}
    directory = tmp_path / "exports" / "assay"
    directory.mkdir(parents=True)
    stem = unit_stem(2, "cached", 512)
    checkpoint = publish(directory / f"{stem}.tar", lambda p: p.write_bytes(b"checkpoint"))
    timing = publish(directory / f"{stem}.json", lambda p: p.write_bytes(b"{}"))
    receipt = receipt_for(
        control,
        directory,
        [
            {**checkpoint, "path": f"assay/{stem}.tar"},
            {**timing, "path": f"assay/{stem}.json"},
        ],
    )
    value = ledger(control, [receipt])
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(value))
    assert list(load_ledger(path, control)) == [unit_id(2, "cached", 512)]
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(path, {**control, "episode_limits": [64]})


def test_a3_stride_covers_every_possible_e4096_reset_root() -> None:
    from research.sandbox_a_resident.units import cell_specs

    control = {
        "worlds": [128, 1024, 2048, 4096],
        "episode_limits": [64, 512],
        "horizon": 128,
        "cycles": 3,
        "warmups": 1,
        "seed_offset_base": 120_000_000,
        "allocation_stride": 3_000_000,
    }
    cells = cell_specs(control)
    assert len(cells) == 24
    assert [row[3] for row in cells[:3]] == [120_000_000, 123_000_000, 126_000_000]
    assert min(row[3] for row in cells) > 100_000_000
    episodes = 1 + (control["cycles"] + control["warmups"]) * control["horizon"]
    assert control["allocation_stride"] >= 4096 * episodes
