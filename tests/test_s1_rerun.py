"""Local-only contracts for S1's explicit operational rerun preparation."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from research.s1_mature_20261018 import rerun
from research.s1_mature_20261018.ledger import Refused


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Ledger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, Path]] = []

    def reconcile_operational_failure(self, job: str, receipt: Path, review: Path) -> None:
        self.calls.append((job, receipt, review))


def _source(root: Path) -> Path:
    return _write(
        root / "source.json",
        {
            "job_id": "s1-mature-n3-7",
            "commit": "a" * 40,
            "env": {"S1_ARM": "n3", "S1_SEED": "7"},
            "output_dir": str(root / "old"),
            "inputs": [],
            "requirements": {"gpu": "L4"},
            "budget": {"unit_cap": 2},
        },
    )


def _final_packet(root: Path) -> dict:
    exports = root / "out" / "exports"
    exports.mkdir(parents=True)
    payload = root / "payload"
    payload.mkdir()
    for name in ("final.pth", "target.pth", "summary.json", "s1-training-provenance.json"):
        (payload / name).write_text(name)
    archive = exports / "packet.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in payload.iterdir():
            tar.add(item, arcname=f"run/{item.name}")
    index = {
        "schema": "s1-chunked-export/v1",
        "chunks": [{"path": archive.name, "sha256": _digest(archive)}],
        "archive_sha256": _digest(archive),
    }
    _write(exports / "s1-final.index.json", index)
    return {"output_dir": str(root / "out")}


def test_final_packet_inspector_distinguishes_complete_partial_and_corrupt(tmp_path: Path) -> None:
    source = _final_packet(tmp_path)
    assert rerun._verified_final_export(source) is True
    exports = Path(source["output_dir"]) / "exports"
    packet = exports / "packet.tar.gz"
    packet.unlink()
    assert rerun._verified_final_export(source) is False
    packet.write_bytes(b"corrupt")
    with pytest.raises(Refused, match="corrupt or ambiguous"):
        rerun._verified_final_export(source)


@pytest.mark.parametrize("path", ["../escape", "/absolute"])
def test_final_packet_inspector_refuses_malformed_chunk_paths(tmp_path: Path, path: str) -> None:
    source = _final_packet(tmp_path)
    index = Path(source["output_dir"]) / "exports" / "s1-final.index.json"
    data = json.loads(index.read_text())
    data["chunks"][0]["path"] = path
    _write(index, data)
    with pytest.raises(Refused, match="corrupt or ambiguous"):
        rerun._verified_final_export(source)


def test_final_packet_inspector_refuses_oversized_chunk_before_read(tmp_path: Path) -> None:
    source = _final_packet(tmp_path)
    exports = Path(source["output_dir"]) / "exports"
    packet = exports / "packet.tar.gz"
    with packet.open("r+b") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(Refused, match="inspection bound"):
        rerun._verified_final_export(source)


def test_prepare_writes_fresh_bound_manual_record_without_launch(tmp_path: Path) -> None:
    source = _source(tmp_path)
    receipt, review = _write(tmp_path / "receipt.json", {}), _write(tmp_path / "review.json", {})
    worker = _Ledger()
    record = rerun.prepare(
        worker_ledger=worker,
        source_manifest=source,
        receipt=receipt,
        review=review,
        manifest_out=tmp_path / "retry.json",
        record_out=tmp_path / "record.json",
        output_dir=tmp_path / "fresh-output",
        state_dir=tmp_path / "fresh-state",
        artifact_root=tmp_path,
    )
    assert worker.calls == [("s1-mature-n3-7", receipt.resolve(), review.resolve())]
    manifest, value = json.loads((tmp_path / "retry.json").read_text()), json.loads(
        record.read_text()
    )
    assert manifest["job_id"] == "retry-n3"
    assert manifest["env"] == json.loads(source.read_text())["env"]
    assert manifest["attempt"] == {"kind": "manual-operational-rerun", "of": "s1-mature-n3-7"}
    assert value["retry_manifest"]["sha256"] == _digest(tmp_path / "retry.json")
    assert value["no_resume"] is True and value["manual_operator_action_required"] is True


def test_prepare_refuses_existing_efficacy_artifact_or_reused_path(tmp_path: Path) -> None:
    source = _source(tmp_path)
    receipt, review = _write(tmp_path / "receipt.json", {}), _write(tmp_path / "review.json", {})
    (tmp_path / "analysis.json").write_text("{}")
    with pytest.raises(Refused, match="evaluation or analysis"):
        rerun.prepare(
            worker_ledger=_Ledger(),
            source_manifest=source,
            receipt=receipt,
            review=review,
            manifest_out=tmp_path / "retry.json",
            record_out=tmp_path / "record.json",
            output_dir=tmp_path / "fresh-output",
            state_dir=tmp_path / "fresh-state",
            artifact_root=tmp_path,
        )
