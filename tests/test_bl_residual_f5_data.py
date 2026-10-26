"""Synthetic-only tests for the F5 selected-960 data boundary."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from research.bl_residual import f5_data
from research.bl_residual.f5_data import (
    metadata_cache,
    package_selected,
    package_selected_files,
    read_selected_package,
    selected_payload_table,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _fixture(*, excluded: bool = False) -> tuple[dict, dict]:
    lives = []
    rows = []
    for index in range(960):
        parent = f"d256-parent-{index:03d}"
        life = f"ns/{parent}/0"
        lives.append(
            {
                "life_id": life,
                "parent_world": parent,
                "selection_index": index,
                "allocation_order": index * 1_000_000,
                "ordinary_only": False,
                "original_fold": 1,
                "source_namespace": "ns-mac-d256",
                "source_sha256": "a" * 64,
                "source_size": 1,
            }
        )
        for action in range(6):
            rows.append(
                {
                    "action": action,
                    "allocation_order": index * 1_000_000,
                    "legal": True,
                    "life_id": life,
                    "parent_world": parent,
                    "row_id": f"{life}/1",
                    "row_order": 0,
                    "served_selectable": True,
                    "source_row_index": index,
                    "status": "EXACT",
                    "stratum": "ordinary" if index % 2 else "doom",
                }
            )
    siblings = [
        {
            "life_id": f"ns/sibling-{index}/0",
            "ordinary_only": True,
        }
        for index in range(9)
    ]
    inventory = {
        "schema": "bl-f5-status-input-inventory/v3",
        "counts": {"ordinary_only_siblings": 0, "selected": 960, "total": 960},
        "original_life_sources": lives,
        "original_life_sources_sha256": hashlib.sha256(_canonical(lives)).hexdigest(),
        "excluded_ordinary_only_sibling_sources": siblings,
        "excluded_ordinary_only_sibling_sources_sha256": hashlib.sha256(
            _canonical(siblings)
        ).hexdigest(),
        "training_exclusions": ["d256-parent-000"] if excluded else ["dev-unused"],
        "converted_container": {"sha256": "b" * 64, "size": 1},
    }
    scan = {
        "schema": "bl-f5-status-extractor/v5",
        "source_binding": {"sha256": "b" * 64, "size": 1},
        "status_inventory": rows,
        "status_inventory_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }
    return scan, inventory


def test_metadata_cache_keeps_registered_order_and_exclusions() -> None:
    scan, inventory = _fixture(excluded=True)
    cache = metadata_cache(scan, inventory)
    assert [row["source_row_index"] for row in cache.rows] == list(range(1, 960))
    assert len(cache.selected_lives) == 960
    assert all(row["parent_world"] not in cache.excluded_parents for row in cache.rows)


def test_metadata_cache_refuses_d256_prefix_or_action_identity_change() -> None:
    scan, inventory = _fixture()
    inventory["original_life_sources"][0]["source_namespace"] = "ns-mac-d1024"
    with pytest.raises(ValueError, match="canonical hash mismatch"):
        metadata_cache(scan, inventory)
    scan, inventory = _fixture()
    scan["status_inventory"][1]["row_id"] = "ns/different/0/1"
    scan["status_inventory_sha256"] = hashlib.sha256(
        _canonical(scan["status_inventory"])
    ).hexdigest()
    with pytest.raises(ValueError, match="action-unit"):
        metadata_cache(scan, inventory)


def _payload(path: Path, rows: int = 960, empty_first_mask: bool = False) -> None:
    q = np.zeros((rows, 6), dtype=np.float32)
    q[:, 0] = 1
    ordinary = np.arange(rows) % 2 == 1
    common = {
        "local": np.zeros((rows, 7, 31, 31), dtype=np.uint8),
        "global": np.zeros((rows, 4, 37, 37), dtype=np.uint8),
        "scalars": np.zeros((rows, 60), dtype=np.float32),
        "q": q,
        "legal": np.ones((rows, 6), dtype=bool),
        "action_mask": np.ones((rows, 6), dtype=bool),
        "bare": np.zeros(rows, dtype=np.int64),
        "died": np.zeros((rows, 6), dtype=bool),
        "status": np.full((rows, 6), "EXACT"),
        "mass": np.ones((rows, 6), dtype=np.float32),
        "ordinary": ordinary,
        "natural_stream": ordinary.copy(),
        "sampling_weight": np.ones(rows, dtype=np.float32),
        "sampling_prob": np.ones(rows, dtype=np.float32),
        "parent_world": np.asarray([f"d256-parent-{i:03d}" for i in range(rows)]),
        "life_id": np.asarray([f"ns/d256-parent-{i:03d}/0" for i in range(rows)]),
        "row_id": np.asarray([f"ns/d256-parent-{i:03d}/0/1" for i in range(rows)]),
    }
    if empty_first_mask:
        common["action_mask"][0] = False
    np.savez(path, **common)


def test_selected_payload_table_hashes_before_loading_and_preserves_metadata_order(
    tmp_path: Path,
) -> None:
    scan, inventory = _fixture()
    source = tmp_path / "synthetic.npz"
    _payload(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scan["source_binding"] = {"sha256": digest, "size": source.stat().st_size}
    inventory["converted_container"] = dict(scan["source_binding"])
    cache = metadata_cache(scan, inventory)
    table = selected_payload_table(source, cache)
    assert table["source_row_index"].tolist() == list(range(960))
    assert table["life_id"].tolist()[0] == "ns/d256-parent-000/0"
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity mismatch|hash mismatch"):
        selected_payload_table(source, cache)


def test_selected_payload_table_refuses_payload_ancestry_mismatch(tmp_path: Path) -> None:
    scan, inventory = _fixture()
    source = tmp_path / "synthetic.npz"
    _payload(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scan["source_binding"] = {"sha256": digest, "size": source.stat().st_size}
    inventory["converted_container"] = dict(scan["source_binding"])
    cache = metadata_cache(scan, inventory)
    with np.load(source, allow_pickle=False) as loaded:
        values = {name: loaded[name] for name in loaded.files}
    values["parent_world"] = values["parent_world"].copy()
    values["parent_world"][0] = "foreign"
    np.savez(source, **values)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scan["source_binding"] = {"sha256": digest, "size": source.stat().st_size}
    inventory["converted_container"] = dict(scan["source_binding"])
    cache = metadata_cache(scan, inventory)
    with pytest.raises(ValueError, match="differs from frozen metadata"):
        selected_payload_table(source, cache)


def test_selected_payload_table_uses_normal_fallback_without_rewriting_raw_mask(
    tmp_path: Path,
) -> None:
    scan, inventory = _fixture()
    for action, record in enumerate(scan["status_inventory"][:6]):
        record["served_selectable"] = action < 3
    scan["status_inventory_sha256"] = hashlib.sha256(
        _canonical(scan["status_inventory"])
    ).hexdigest()
    source = tmp_path / "synthetic.npz"
    _payload(source, empty_first_mask=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scan["source_binding"] = {"sha256": digest, "size": source.stat().st_size}
    inventory["converted_container"] = dict(scan["source_binding"])
    table = selected_payload_table(source, metadata_cache(scan, inventory))
    assert not table["action_mask"][0].any()


def test_compact_package_omits_scan_and_recomputes_corpus_identity(tmp_path: Path) -> None:
    scan, inventory = _fixture()
    source = tmp_path / "synthetic.npz"
    _payload(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    scan["source_binding"] = {"sha256": source_sha, "size": source.stat().st_size}
    inventory["converted_container"] = dict(scan["source_binding"])
    pins = {
        "inventory_sha256": "c" * 64,
        "scan_sha256": "d" * 64,
        "original_source_sha256": source_sha,
    }
    bundle = tmp_path / "selected.tar.xz"
    receipt = package_selected(source, scan, inventory, bundle, pins=pins, max_bytes=20_000_000)
    table, metadata, loaded = read_selected_package(bundle, pins=pins, max_bytes=20_000_000)
    assert set(table) >= {"local", "status", "died", "row_id"}
    assert loaded["corpus_sha256"] == receipt["corpus_sha256"]
    assert metadata.scan_sha256 == scan["status_inventory_sha256"]
    assert receipt["package_bytes"] < 20_000_000


def test_compact_package_cap_failure_removes_partial_output(tmp_path: Path) -> None:
    scan, inventory = _fixture()
    source = tmp_path / "synthetic.npz"
    _payload(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    scan["source_binding"] = {"sha256": source_sha, "size": source.stat().st_size}
    inventory["converted_container"] = dict(scan["source_binding"])
    pins = {
        "inventory_sha256": "c" * 64,
        "scan_sha256": "d" * 64,
        "original_source_sha256": source_sha,
    }
    bundle = tmp_path / "too-large.tar.xz"
    with pytest.raises(ValueError, match="HOLD"):
        package_selected(source, scan, inventory, bundle, pins=pins, max_bytes=1)
    assert not bundle.exists()


def test_file_wrapper_refuses_changed_scan_before_payload_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan, inventory = _fixture()
    source = tmp_path / "synthetic.npz"
    _payload(source)
    scan_path, inventory_path = tmp_path / "scan.json", tmp_path / "inventory.json"
    scan_path.write_text(json.dumps(scan))
    inventory_path.write_text(json.dumps(inventory))
    monkeypatch.setattr(
        f5_data,
        "PACKAGE_PINS",
        {
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "scan_sha256": "f" * 64,
            "original_source_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(f5_data, "selected_payload_table", lambda *_: pytest.fail("payload opened"))
    with pytest.raises(ValueError, match="scan raw hash mismatch"):
        package_selected_files(source, scan_path, inventory_path, tmp_path / "bundle.tar.xz")


def test_compact_reader_refuses_duplicate_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    duplicate = tmp_path / "duplicate.tar.xz"
    with tarfile.open(duplicate, "w:xz") as archive:
        for name in ("selected-table.npz", "selected-table.npz", "receipt.json"):
            payload = b"x"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(f5_data, "verify_export", lambda _: "ignored")
    with pytest.raises(ValueError, match="member schema mismatch"):
        read_selected_package(duplicate, pins={key: "a" * 64 for key in f5_data.PACKAGE_PINS})
