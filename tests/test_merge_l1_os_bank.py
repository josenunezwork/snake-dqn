"""Renumbered packet union, exact sidecar binding, and fail-closed merge checks."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from research.watch_replay import merge_l1_os as merger
from research.watch_replay.install import extract_controller_bank
from research.watch_replay.run import allocation, identity
from tests.test_watch_replay import recording
from tools.ops.package_spec import PackageRefused, publish_json, unit_id, verify_immutable
from web.backend.watch_replay import Recording


def packet(root: Path, number: int = 0, count: int = 1, horizon: int = 2) -> dict:
    contract = {
        **allocation(merger.L1_OS_BANK_NAMESPACES[number], count, merger.CONTROLLER, horizon),
        "commit": "a" * 40,
    }
    publish_json(root / "contract.json", contract)
    for row in contract["rows"]:
        index = row["index"]
        recording(
            root,
            f"{index:06d}",
            horizon=horizon,
            controller=merger.CONTROLLER,
            seed=row["seed"],
            index=index,
            unit_id=unit_id(contract["namespace"], "sessions", index),
            allocation=identity(contract),
            source_commit=contract["commit"],
        )
    return contract


def test_three_44_packet_union_renumbers_and_binds_sidecars(tmp_path):
    roots = [tmp_path / f"p{i}" for i in range(3)]
    for i, root in enumerate(roots):
        packet(root, i, count=44)
    bank = tmp_path / "merged"
    report = merger.merge(list(reversed(roots)), bank, fixture=True)
    assert report["copied_count"] == 132
    assert list(report["packet_counts"].values()) == [44, 44, 44]
    assert len(list(bank.glob("*.watch.gz"))) == 132
    for i, item in enumerate(report["units"]):
        target = bank / f"{i:06d}.watch.gz"
        assert verify_immutable(target)["sha256"] == item["sha256"]
        assert Path(f"{target}.sha256").read_text().split() == [item["sha256"], target.name]
        assert target.read_bytes() == Path(item["source"]).read_bytes()
        record = Recording(target, product=False)
        try:
            assert record.header["unit_id"] == item["unit_id"]
        finally:
            record.close()
    extracted = extract_controller_bank(bank, merger.CONTROLLER, tmp_path / "pure", product=False)
    assert extracted["copied_count"] == 132
    with pytest.raises(FileExistsError):
        merger.merge(roots, bank, fixture=True)


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_or_conflicting_unit_refused_before_destination(tmp_path, conflicting):
    root, other = tmp_path / "p1", tmp_path / "p1-again"
    contract = packet(root)
    shutil.copytree(root, other)
    if conflicting:
        for path in other.glob("*.watch.gz*"):
            path.unlink()
        row = contract["rows"][0]
        recording(
            other,
            "000000",
            horizon=2,
            controller=merger.CONTROLLER,
            seed=row["seed"],
            index=0,
            unit_id=unit_id(contract["namespace"], "sessions", 0),
            allocation=identity(contract),
            source_commit=contract["commit"],
            stats={"changed": 1},
        )
    bank = tmp_path / "merged"
    with pytest.raises(ValueError, match="duplicate/conflicting"):
        merger.merge([root, other], bank, fixture=True)
    assert not bank.exists()


def test_sidecar_wrong_name_refused(tmp_path):
    root = tmp_path / "p1"
    packet(root)
    sidecar = root / "000000.watch.gz.sha256"
    sidecar.write_text(sidecar.read_text().replace("000000.watch.gz", "000044.watch.gz"))
    with pytest.raises(PackageRefused, match="sidecar"):
        merger.merge([root], tmp_path / "merged", fixture=True)
    assert not (tmp_path / "merged").exists()


def test_incomplete_union_refused(tmp_path):
    root = tmp_path / "p1"
    packet(root, count=2)
    for path in root.glob("000001.watch.gz*"):
        path.unlink()
    with pytest.raises(ValueError, match="missing units"):
        merger.merge([root], tmp_path / "merged", fixture=True)
    assert not (tmp_path / "merged").exists()


def test_fixture_not_accepted_as_product(tmp_path):
    root = tmp_path / "p1"
    packet(root)
    with pytest.raises(ValueError, match="three 44-session H6000"):
        merger.merge([root], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_product_requires_all_three_packets(tmp_path):
    root = tmp_path / "p1"
    contract = {
        **allocation(merger.L1_OS_BANK_NAMESPACES[0], 44, merger.CONTROLLER),
        "commit": "a" * 40,
    }
    publish_json(root / "contract.json", contract)
    with pytest.raises(ValueError, match="all three"):
        merger.merge([root], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()
