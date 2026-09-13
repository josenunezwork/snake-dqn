"""Operational tests for opt-in immutable PQN checkpoint archives."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.scripts import train_pqn
from src.training.pqn_trainer import PQNTelemetry, TripwireError


def _telemetry(update: int, steps: int) -> PQNTelemetry:
    return PQNTelemetry(
        update=update,
        agent_steps=steps,
        loss=0.1,
        grad_norm=0.2,
        mean_abs_q=0.3,
        max_abs_q=0.4,
        epsilon=0.5,
        mean_reward=0.6,
        action_entropy=0.7,
        legacy_kills_per_rollout=0.0,
        boost_fraction=0.0,
        pool_size=0,
    )


class _AcceptedTrainer:
    """Small deterministic writer whose bytes expose each accepted state."""

    def __init__(self, *, start_update: int = 0, start_steps: int = 0, updates: int = 2) -> None:
        self.update_idx = start_update
        self.agent_steps = start_steps
        self._remaining = updates
        self.saves = []

    def update(self) -> PQNTelemetry:
        assert self._remaining > 0
        self._remaining -= 1
        self.update_idx += 1
        self.agent_steps += 10
        return _telemetry(self.update_idx, self.agent_steps)

    def save_checkpoint(self, path: str) -> None:
        payload = f"update={self.update_idx};steps={self.agent_steps}".encode()
        with open(path, "wb") as fh:
            fh.write(payload)
        self.saves.append((self.update_idx, self.agent_steps, path))


class _TripwireTrainer(_AcceptedTrainer):
    def update(self) -> PQNTelemetry:
        raise TripwireError("non-finite target", incident_class="nonfinite_target")

    def numeric_recovery_metadata(self):
        return {}


def _events(archive_dir):
    return [json.loads(line) for line in (archive_dir / "events.jsonl").read_text().splitlines()]


def test_default_checkpointing_creates_no_archive_directory(tmp_path):
    trainer = _AcceptedTrainer(updates=1)

    train_pqn.train_loop(trainer, 10, 10, 1, tmp_path)

    assert (tmp_path / "latest_pqn.pth").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert len(trainer.saves) == 2  # Existing cadence plus clean-final behavior.


def test_opt_in_archive_records_start_cadence_and_clean_final(tmp_path):
    trainer = _AcceptedTrainer(updates=2)

    train_pqn.train_loop(trainer, 20, 10, 1, tmp_path, archive_checkpoints=True)

    archive_dir = next((tmp_path / "checkpoints").iterdir())
    events = _events(archive_dir)
    assert [(event["label"], event["update_idx"], event["agent_steps"]) for event in events] == [
        ("start", 0, 0),
        ("cadence", 1, 10),
        ("cadence", 2, 20),
        ("final", 2, 20),
    ]
    for event in events:
        checkpoint = archive_dir / event["checkpoint_path"]
        assert checkpoint.exists()
        assert event["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert isinstance(event["created_at_unix_ns"], int)
    assert (tmp_path / "latest_pqn.pth").read_bytes() == b"update=2;steps=20"


def test_resume_invocations_receive_distinct_archives_and_start_clocks(tmp_path):
    first = _AcceptedTrainer(start_update=4, start_steps=40, updates=1)
    second = _AcceptedTrainer(start_update=5, start_steps=50, updates=1)

    train_pqn.train_loop(first, 50, 10, 0, tmp_path, archive_checkpoints=True)
    train_pqn.train_loop(second, 60, 10, 0, tmp_path, append_history=True, archive_checkpoints=True)

    directories = sorted((tmp_path / "checkpoints").iterdir())
    assert len(directories) == 2
    assert directories[0] != directories[1]
    starts = [_events(directory)[0] for directory in directories]
    assert {(event["update_idx"], event["agent_steps"]) for event in starts} == {(4, 40), (5, 50)}


def test_tripwire_archives_start_but_never_false_final(tmp_path):
    trainer = _TripwireTrainer()

    _history, tripped = train_pqn.train_loop(trainer, 1, 10, 1, tmp_path, archive_checkpoints=True)

    assert tripped is not None
    archive_dir = next((tmp_path / "checkpoints").iterdir())
    assert [event["label"] for event in _events(archive_dir)] == ["start"]
    assert not (tmp_path / "latest_pqn.pth").exists()


def test_archive_copy_failure_propagates_without_claiming_completion(tmp_path, monkeypatch):
    trainer = _AcceptedTrainer(updates=1)

    def fail_copy(*_args, **_kwargs):
        raise OSError("archive volume is full")

    monkeypatch.setattr(train_pqn.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="archive volume is full"):
        train_pqn.train_loop(trainer, 10, 10, 0, tmp_path, archive_checkpoints=True)


def test_existing_archive_destination_is_never_truncated(tmp_path):
    trainer = _AcceptedTrainer()
    archive = train_pqn._CheckpointArchive(tmp_path, tmp_path / "latest_pqn.pth")
    destination = archive.directory / "start_update_00000000_steps_000000000000.pth"
    destination.write_bytes(b"prior-evidence")

    with pytest.raises(FileExistsError):
        archive.save(trainer, "start", replace_latest=False)

    assert destination.read_bytes() == b"prior-evidence"
    assert not archive.manifest_path.exists()


def test_archive_publication_failure_leaves_no_checkpoint_or_success_event(tmp_path, monkeypatch):
    trainer = _AcceptedTrainer()
    latest = tmp_path / "latest_pqn.pth"
    latest.write_bytes(b"accepted-latest")
    archive = train_pqn._CheckpointArchive(tmp_path, latest)
    destination = archive.directory / "cadence_update_00000000_steps_000000000000.pth"

    def fail_publish(*_args, **_kwargs):
        raise OSError("archive publication failed")

    monkeypatch.setattr(archive, "_publish_temporary", fail_publish)
    with pytest.raises(OSError, match="archive publication failed"):
        archive.save(trainer, "cadence", replace_latest=True)

    assert not destination.exists()
    assert not archive.manifest_path.exists()
    assert not list(archive.directory.glob("*.tmp"))


def test_manifest_failure_rolls_back_new_archive_but_retains_latest_and_prefix(
    tmp_path, monkeypatch
):
    initial = _AcceptedTrainer()
    latest = tmp_path / "latest_pqn.pth"
    archive = train_pqn._CheckpointArchive(tmp_path, latest)
    archive.save(initial, "start", replace_latest=False)
    prior_manifest = archive.manifest_path.read_bytes()
    prior_files = sorted(path.name for path in archive.directory.glob("*.pth"))

    accepted = _AcceptedTrainer(start_update=3, start_steps=30)

    def fail_manifest(_events):
        raise OSError("manifest volume is full")

    monkeypatch.setattr(archive, "_write_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest volume is full"):
        archive.save(accepted, "cadence", replace_latest=True)

    assert archive.manifest_path.read_bytes() == prior_manifest
    assert sorted(path.name for path in archive.directory.glob("*.pth")) == prior_files
    assert not list(archive.directory.glob("*.tmp"))
    assert latest.read_bytes() == b"update=3;steps=30"
    assert len(archive._events) == 1
