"""Mac-local runtime parity contracts for the L2 amendment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_served_search import ceiling, parity_v2
from research.bl_served_search.exports import export_json


def _bank(path: Path) -> Path:
    rows = [
        {"world_id": f"fixture-{mix}", "world_seed": index + 1, "mix": mix}
        for index, mix in enumerate(("frozen", "mixed", "scripted"))
    ]
    export_json(
        path,
        {
            "schema": "bl-served-search-bank/v1",
            "rows": [],
            "excluded_preflight_rows": rows,
        },
    )
    return path


def _fixture(
    path: Path,
    bank: Path,
    source: dict[str, str],
    *,
    search_mode: str = "exact_stop",
    native_mode: str = "stock",
    workers: int = 2,
) -> None:
    rows = json.loads(bank.read_text())["excluded_preflight_rows"]
    export_json(
        path,
        {
            "schema": "bl-served-search-l2-fixture-shard/v1",
            "status": "completed",
            "clean_exit": True,
            "identity": {
                "source_manifest": source,
                "bank_sha256": parity_v2.digest(bank),
                "search_mode": search_mode,
                "native_mode": native_mode,
                "workers": workers,
                "fixture": True,
            },
            "rows": rows,
            "rows_sha256": ceiling.digest_value(rows),
        },
    )


def test_verify_accepts_mac_identity_without_colab_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _bank(tmp_path / "bank.json")
    source = {"source.py": "0" * 64}
    reference, actual, receipt = (
        tmp_path / "reference.json",
        tmp_path / "actual.json",
        tmp_path / "parity.json",
    )
    _fixture(reference, bank, source)
    _fixture(actual, bank, source)
    mac = {"node": "mac", "machine": "arm64", "runtime_kind": "mac", "system": "Darwin"}
    monkeypatch.setattr(ceiling, "source_manifest", lambda: source)
    monkeypatch.setattr(ceiling, "runtime_identity", lambda: mac)

    result = parity_v2.verify(reference, actual, bank, receipt)

    assert result["runtime_identity"] == mac
    assert json.loads(receipt.read_text())["status"] == "VERIFIED"


def test_verify_refuses_mismatched_fixture_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _bank(tmp_path / "bank.json")
    source = {"source.py": "0" * 64}
    reference, actual = tmp_path / "reference.json", tmp_path / "actual.json"
    _fixture(reference, bank, source, search_mode="all")
    _fixture(actual, bank, source)
    monkeypatch.setattr(ceiling, "source_manifest", lambda: source)

    with pytest.raises(ValueError, match="execution modes differ"):
        parity_v2.verify(reference, actual, bank, tmp_path / "parity.json")


def test_verify_allows_matching_legacy_fixture_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _bank(tmp_path / "bank.json")
    source = {"source.py": "0" * 64}
    reference, actual, receipt = (
        tmp_path / "reference.json",
        tmp_path / "actual.json",
        tmp_path / "parity.json",
    )
    _fixture(reference, bank, source, search_mode="all", native_mode="stock", workers=1)
    _fixture(actual, bank, source, search_mode="all", native_mode="stock", workers=1)
    monkeypatch.setattr(ceiling, "source_manifest", lambda: source)
    monkeypatch.setattr(
        ceiling,
        "runtime_identity",
        lambda: {"node": "mac", "machine": "arm64", "runtime_kind": "mac", "system": "Darwin"},
    )

    assert parity_v2.verify(reference, actual, bank, receipt)["status"] == "VERIFIED"


def test_prepare_mac_runs_two_real_entrypoint_fixtures_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _bank(tmp_path / "bank.json")
    seen: list[dict] = []

    def fake_run(**kwargs) -> None:
        seen.append(kwargs)

    expected = {"status": "VERIFIED", "receipt": "ok"}
    monkeypatch.delenv("COLAB_CLI_ROOT", raising=False)
    monkeypatch.setattr(parity_v2, "_run_fixture", fake_run)
    monkeypatch.setattr(parity_v2, "verify", lambda *args: expected)
    out_dir = tmp_path / "fresh"

    assert (
        parity_v2.prepare_mac(
            bank=bank,
            b8_path=tmp_path / "b8.pth",
            checkpoint_dir=tmp_path / "checkpoints",
            out_dir=out_dir,
        )
        == expected
    )
    assert [entry["out"].parent.name for entry in seen] == ["reference", "verification"]
    assert all(entry["out"].name == "shard.json" for entry in seen)
    assert all(entry["environment"].get("COLAB_CLI_ROOT") is None for entry in seen)
    with pytest.raises(FileExistsError, match="create-only"):
        parity_v2.prepare_mac(
            bank=bank,
            b8_path=tmp_path / "b8.pth",
            checkpoint_dir=tmp_path / "checkpoints",
            out_dir=out_dir,
        )
