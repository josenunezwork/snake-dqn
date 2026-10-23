"""Fast contract tests for the outcome-blind RPC qualification loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_labels.exports import publish_json
from research.rpc_qualification import loop


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    roots = [{"root_id": f"{index:064x}", "source_session": "source"} for index in range(64)]
    bank = {"roots": roots, "source_sessions": [{"name": "source"}]}
    monkeypatch.setattr(
        loop.f1_ruler,
        "bank_digest",
        lambda _: "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d",
    )
    monkeypatch.setattr(loop.f1_ruler, "load_bank", lambda _: bank)
    source = tmp_path / "frozen"
    source.mkdir()
    (source / "member.py").write_text("frozen\n")
    manifest = {
        "commit": loop.F1_SOURCE_COMMIT,
        "archive_sha256": "a" * 64,
        "files": {"member.py": hashlib.sha256((source / "member.py").read_bytes()).hexdigest()},
    }
    publish_json(source / "frozen-source.json", manifest)
    ledger = tmp_path / "ledger.json"
    publish_json(
        ledger,
        loop.empty_ledger(
            "ns",
            "alloc",
            "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d",
            loop.F1_SOURCE_COMMIT,
            "a" * 64,
        ),
    )
    return source / "frozen-source.json", ledger, bank


def test_loop_finishes_admitted_root_then_publishes_ledger_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, ledger, bank = _fixture(tmp_path, monkeypatch)
    ticks = iter([0.0, 0.0, 0.0, 3600.0, 3601.0, 3602.0, 3603.0, 3604.0])

    def prepare(value: dict, *_: object) -> dict[str, Path]:
        return {row["root_id"]: tmp_path / (row["root_id"] + ".pickle") for row in value["roots"]}

    def execute(_job: tuple[str, float]) -> dict:
        return {
            "root_id": bank["roots"][0]["root_id"],
            "digest": "d" * 64,
            "label_row_sha256": "e" * 64,
            "digest_checks": 1,
            "bare_audit_mismatches": 0,
            "elapsed_seconds": 1.0,
        }

    result = loop.run(
        bank_path=tmp_path / "bank",
        ledger_path=ledger,
        b8=tmp_path / "b8",
        frp3=tmp_path / "frp3",
        output=tmp_path / "out",
        platform="cpu",
        workers=2,
        namespace="ns",
        allocation="alloc",
        loop_seconds=3600,
        source_closure_manifest=manifest,
        source_closure_sha256="a" * 64,
        clock=lambda: next(ticks),
        prepare=prepare,
        execute=execute,
    )
    assert result["units_completed"] == 1
    assert (tmp_path / "out/verified-ledger.json.complete").is_file()
    assert (tmp_path / "out/runtime-result.json.complete").is_file()
    assert (tmp_path / "out/units/unit-00000000.json.complete").is_file()


def test_resume_skips_verified_root_and_preserves_it_in_exported_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, ledger, bank = _fixture(tmp_path, monkeypatch)
    seen = []
    now = [0.0]

    def prepare(value: dict, *_: object) -> dict[str, Path]:
        return {row["root_id"]: tmp_path / (row["root_id"] + ".pickle") for row in value["roots"]}

    def execute(job: tuple[str, float]) -> dict:
        root_id = Path(job[0]).stem
        seen.append(root_id)
        now[0] += 3601
        return {
            "root_id": root_id,
            "digest": "d" * 64,
            "label_row_sha256": "e" * 64,
            "digest_checks": 1,
            "bare_audit_mismatches": 0,
            "elapsed_seconds": 1.0,
        }

    common = dict(
        bank_path=tmp_path / "bank",
        b8=tmp_path / "b8",
        frp3=tmp_path / "frp3",
        platform="cpu",
        workers=2,
        namespace="ns",
        allocation="alloc",
        loop_seconds=3600,
        source_closure_manifest=manifest,
        source_closure_sha256="a" * 64,
        clock=lambda: now[0],
        prepare=prepare,
        execute=execute,
    )
    loop.run(ledger_path=ledger, output=tmp_path / "first", **common)
    now[0] = 0.0
    result = loop.run(
        ledger_path=tmp_path / "first/verified-ledger.json", output=tmp_path / "second", **common
    )
    assert seen == [bank["roots"][0]["root_id"], bank["roots"][1]["root_id"]]
    assert result["units_completed"] == 2
    assert (tmp_path / "first/units/unit-00000000.json").read_bytes() == (
        tmp_path / "second/units/unit-00000000.json"
    ).read_bytes()


def test_loop_rejects_foreign_ledger_and_short_normal_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, ledger, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="at least 60"):
        loop.run(
            bank_path=tmp_path / "bank",
            ledger_path=ledger,
            b8=tmp_path / "b8",
            frp3=tmp_path / "frp3",
            output=tmp_path / "out",
            platform="cpu",
            workers=2,
            namespace="ns",
            allocation="alloc",
            loop_seconds=1,
            source_closure_manifest=manifest,
            source_closure_sha256="a" * 64,
        )
    value = json.loads(ledger.read_text())
    value["namespace"] = "foreign"
    ledger.unlink()
    (Path(str(ledger) + ".sha256")).unlink()
    (Path(str(ledger) + ".complete")).unlink()
    publish_json(ledger, value)
    with pytest.raises(ValueError, match="foreign namespace"):
        loop._load_ledger(
            ledger,
            "ns",
            "alloc",
            "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d",
            "a" * 64,
        )


def test_supervisor_precreated_empty_output_is_allowed_but_reuse_is_refused(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    loop._claim_empty_output(output)
    (output / "old").write_text("not fresh\n")
    with pytest.raises(ValueError, match="reused or nonempty"):
        loop._claim_empty_output(output)
