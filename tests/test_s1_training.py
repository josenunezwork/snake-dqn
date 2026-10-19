"""Focused contracts for the S1 mature training adapter (no Colab allocation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import manifests
from research.s1_mature_20261018.train import (
    _publish_chunked,
    actual_recipe,
    record_tripwire,
    registered_halt,
)


def _input(tmp_path: Path, name: str, dest: str) -> manifests.ExternalInput:
    path = tmp_path / name
    path.write_bytes((name + "\n").encode())
    return manifests.ExternalInput(name, path, dest)


def test_manifest_factory_makes_ten_draft_review_rows_with_asymmetric_l4_caps(
    tmp_path: Path,
) -> None:
    seeds = tmp_path / "seed-bank.json"
    seeds.write_text(json.dumps({"schema": "s1-seed-bank/v1", "seeds": [701, 703, 704, 705, 707]}))
    inputs = [
        _input(tmp_path, "seed-bank-input.json", "seed-bank.json"),
        _input(tmp_path, "B8_s3_final.pth", "B8_s3_final.pth"),
        _input(tmp_path, "demo.tar.gz", "demo.tar.gz"),
        _input(tmp_path, "league.tar.gz", "league.tar.gz"),
        _input(tmp_path, "mac-parity-reference.json", "mac-parity-reference.json"),
        _input(tmp_path, "parity-bank.json", "parity-bank.json"),
        _input(tmp_path, "q0b-l4.json", "q0b-l4.json"),
    ]
    rows = manifests.make_manifests(
        seed_bank=seeds,
        published_commit="a" * 40,
        registration_status="DRAFT_PENDING_OWNER_REVIEW",
        owner_release="DRAFT_PENDING_OWNER_REVIEW",
        output_root=tmp_path / "outputs",
        inputs=inputs,
    )
    assert len(rows) == 10
    assert {row["requirements"]["gpu"] for row in rows} == {"L4"}
    n3 = next(row for row in rows if row["env"]["S1_ARM"] == "n3")
    n1 = next(row for row in rows if row["env"]["S1_ARM"] == "n1")
    assert (n3["requirements"]["est_minutes"], n3["timeout_minutes"]) == (30, 60)
    assert (n1["requirements"]["est_minutes"], n1["timeout_minutes"]) == (68, 136)
    assert n1["env"]["S1_TRAINING_NAMESPACE"].startswith("q1-task-aligned-n1-20261017/")
    assert n3["budget"]["unit_cap"] == 2.0
    assert n1["budget"]["unit_cap"] == pytest.approx(136 * 2 / 60)


def test_manifest_factory_refuses_nonpublished_commit_and_bad_seed_bank(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"seeds": [1, 1, 2, 3, 4]}))
    with pytest.raises(manifests.ManifestRefused, match="published_commit"):
        manifests.make_manifests(
            seed_bank=seeds,
            published_commit="draft",
            registration_status="DRAFT_PENDING_OWNER_REVIEW",
            owner_release="DRAFT_PENDING_OWNER_REVIEW",
            output_root=tmp_path / "out",
            inputs=[],
        )
    with pytest.raises(manifests.ManifestRefused, match="distinct"):
        manifests.make_manifests(
            seed_bank=seeds,
            published_commit="a" * 40,
            registration_status="DRAFT_PENDING_OWNER_REVIEW",
            owner_release="DRAFT_PENDING_OWNER_REVIEW",
            output_root=tmp_path / "out",
            inputs=[],
        )


def test_commands_freeze_mature_schedule_and_qualified_collection_settings() -> None:
    n3 = manifests.training_command("n3", 701)
    n1 = manifests.training_command("n1", 701)
    for command in (n3, n1):
        assert ["--schedule-package", "mature"] == command[
            command.index("--schedule-package") : command.index("--schedule-package") + 2
        ]
        assert ["--full-state-replay", "full"] == command[
            command.index("--full-state-replay") : command.index("--full-state-replay") + 2
        ]
        assert "--compile-learner" in command
        assert command[command.index("--compile-learner") + 1] == "reduce-overhead"
        assert command[command.index("--worlds") + 1] == "128"
    assert n3[n3.index("--actors") + 1] == "6"
    assert n3[n3.index("--learner-seats") + 1] == "3"
    assert n1[n1.index("--actors") + 1] == "3"
    assert n1[n1.index("--learner-seats") + 1] == "1"


def test_provenance_uses_actual_arguments_and_requires_missing_settings() -> None:
    raw = {
        "args": {
            "worlds": "128",
            "actors": "6",
            "learner_seats": "3",
            "transitions": "8000000",
            "namespace": "actual-namespace",
            "schedule_package": "mature",
            "collection_profile": "legacy",
        }
    }
    assert actual_recipe(raw)["namespace"] == "actual-namespace"
    raw["args"]["actors"] = "3"
    assert actual_recipe(raw)["actors"] == 3
    del raw["args"]["schedule_package"]
    with pytest.raises(KeyError):
        actual_recipe(raw)


def test_scientific_tripwire_is_distinct_from_operational_failure(tmp_path: Path) -> None:
    assert registered_halt({"halt": "non-finite gradient"})
    assert not registered_halt({"halt": "worker timed out"})
    summary = {
        "actors": 6,
        "halt": "non-finite loss",
        "stopped_for_time": False,
        "actor_exitcodes": [0] * 6,
        "ring_overrun_rows": 1,
        "nstep_accounting_reconciled": True,
    }
    with pytest.raises(RuntimeError, match="incomplete"):
        record_tripwire(tmp_path, summary, "n3", 123, tmp_path / "trainer.py")


def test_chunked_export_is_reassemblable_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use tiny chunks to exercise the split/index protocol without a multi-GiB fixture.
    import research.s1_mature_20261018.train as train

    monkeypatch.setattr(train, "MAX_EXPORT_BYTES", 4096)
    source = tmp_path / "snapshot.tar.gz"
    source.write_bytes(b"abcdef" * 1000)
    exports = tmp_path / "exports"
    _publish_chunked(source, exports, "full-state-snapshot", chunk_size=2048)
    index = json.loads((exports / "full-state-snapshot.index.json").read_text())
    rebuilt = b"".join((exports / item["path"]).read_bytes() for item in index["chunks"])
    assert hashlib.sha256(rebuilt).hexdigest() == index["archive_sha256"]
    for item in index["chunks"]:
        output = exports / item["path"]
        assert output.with_name(output.name + ".sha256").is_file()
        assert output.with_name(output.name + ".complete").is_file()


def test_recovered_snapshot_is_discoverable_by_real_trainer_loader(tmp_path: Path) -> None:
    from research.redesign_m3c_20261008 import async_checkpoint
    from research.s1_mature_20261018 import train

    source = tmp_path / "full-state"
    snapshot = source / "snapshot-00000000000000012345-abcdef01"
    snapshot.mkdir(parents=True)
    state = snapshot / "state.pt"
    state.write_bytes(b"transport fixture; loader verifies identity before deserialization")
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema": async_checkpoint.SCHEMA,
                "snapshot": snapshot.name,
                "transitions": 12345,
                "created_ns": 1,
                "files": [
                    {
                        "name": "state.pt",
                        "sha256": train.sha256_file(state),
                        "size_bytes": state.stat().st_size,
                    }
                ],
                "metadata": {},
            }
        )
    )
    exports = tmp_path / "exports"
    train._snapshot_exports(source, exports, set())
    index = exports / f"full-state-{snapshot.name}.index.json"
    recovered = train.reassemble_and_extract(index, exports, tmp_path / "recovered")
    assert recovered.name == snapshot.name
    assert len(async_checkpoint.discover_snapshots(recovered.parent, verify=True)) == 1


def test_supervisor_inventory_does_not_double_the_export_directory(tmp_path: Path) -> None:
    from research.colab_cli import runtime

    output = tmp_path / "output"
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"immutable export fixture")
    _publish_chunked(archive, output, "s1-final", chunk_size=4096)
    records = runtime._exports(tmp_path, 4096)
    assert records
    assert all(Path(item["path"]).parent == Path(".") for item in records)
