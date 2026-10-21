"""Bank archives preserve sidecars and remain bounded and immutable."""

from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix.bundles import pack_bank, unpack_bank
from research.colab_cli.runtime import _exports


def test_bank_roundtrip_supervisor_sidecars_and_deterministic_chunks(tmp_path: Path) -> None:
    source = tmp_path / "work"
    export_json(source / "bank.json", {"roots": []})
    export_json(source / "root.npz.root.json", {"state": "fixture"})
    first, second = tmp_path / "first" / "output", tmp_path / "second" / "output"
    one = pack_bank(source, first, 20480)
    two = pack_bank(source, second, 20480)
    assert one == two
    assert _exports(first.parent, 64 * 1024**2)
    restored = unpack_bank(first / "bank-bundle.json", tmp_path / "restored")
    assert verify_export(restored) == verify_export(source / "bank.json")
    assert (restored.parent / "root.npz.root.json").read_bytes() == (
        source / "root.npz.root.json"
    ).read_bytes()
    chunk = first / one["chunks"][0]["path"]
    chunk.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch"):
        unpack_bank(first / "bank-bundle.json", tmp_path / "changed")


def test_bundle_refuses_uncompleted_bank_members(tmp_path: Path) -> None:
    source = tmp_path / "work"
    source.mkdir()
    (source / "bank.json").write_text("{}")
    with pytest.raises(ValueError, match="missing"):
        pack_bank(source, tmp_path / "out")
