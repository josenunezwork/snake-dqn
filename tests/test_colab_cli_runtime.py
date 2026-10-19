"""Unit contracts for the runtime-side Colab CLI supervisor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "research" / "colab_cli" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("colab_cli_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_read_state_returns_only_completed_hash_verified_exports(tmp_path: Path) -> None:
    root = tmp_path / "root"
    output = root / "output"
    output.mkdir(parents=True)
    data = output / "snapshot.bin"
    data.write_bytes(b"immutable")
    digest = hashlib.sha256(b"immutable").hexdigest()
    (output / "snapshot.bin.sha256").write_text(f"{digest}  snapshot.bin\n", encoding="utf-8")
    (output / "snapshot.bin.complete").write_text("ok\n", encoding="utf-8")
    (output / "bad.bin").write_bytes(b"bad")
    (output / "bad.bin.sha256").write_text("0" * 64, encoding="utf-8")
    (output / "bad.bin.complete").touch()
    _write_json(root / "lease.json", {"generation": "g1", "renewed_at": 1})
    _write_json(
        root / "state.json",
        {
            "generation": "g1",
            "max_export_bytes": 100,
            "exports": [{"path": "snapshot.bin", "sha256": digest, "size": 9}],
        },
    )

    result = runtime.read_state(root, "g1")

    assert result["exports"] == [{"path": "snapshot.bin", "sha256": digest, "size": 9}]
    assert json.loads((root / "lease.json").read_text())["renewed_at"] > 1


def test_wrong_generation_never_renews_lease(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write_json(root / "lease.json", {"generation": "right", "renewed_at": 123.0})
    _write_json(root / "state.json", {"generation": "right", "max_export_bytes": 100})

    with pytest.raises(RuntimeError, match="generation fence"):
        runtime.read_state(root, "wrong")
    assert json.loads((root / "lease.json").read_text())["renewed_at"] == 123.0


def test_validate_rejects_long_checkpoint_grace_before_touching_root() -> None:
    base = {
        "job_id": "job",
        "generation": "g",
        "root": "/content/snake-cli/unit",
        "command": ["true"],
        "env": {},
        "timeout_seconds": 60,
        "lease_seconds": 60,
        "checkpoint_grace_seconds": 31,
        "max_export_bytes": 100,
        "release_endpoint": "unitendpoint",
    }
    with pytest.raises(runtime.SpecError, match="<= 30"):
        runtime._validate_spec(base)


def test_lease_expiry_stops_owned_child_writes_receipt_and_unassigns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watchdog cannot hang when the Mac stops renewing its lease."""
    root = tmp_path / "owned"
    root.mkdir()
    spec = {
        "job_id": "job",
        "generation": "g",
        "root": str(root),
        "command": [sys.executable, "-c", "import time; time.sleep(60)"],
        "env": {},
        "timeout_seconds": 60,
        "lease_seconds": 0.2,
        "checkpoint_grace_seconds": 0.1,
        "max_export_bytes": 100,
        "release_endpoint": "unitendpoint",
    }
    calls: list[str] = []
    monkeypatch.setattr(runtime, "_validate_spec", lambda value: value)
    monkeypatch.setattr(runtime, "_require_release_safety", lambda *_: None)
    monkeypatch.setattr(runtime, "_unassign", lambda *_: calls.append("unassign"))
    monkeypatch.setattr(runtime, "_release_credential_expiring", lambda *_: False)

    assert runtime.supervise(spec) != 0
    terminal = json.loads((root / "terminal.json").read_text())
    assert terminal["reason"] == "lease_expired"
    assert terminal["status"] == "failed"
    assert calls == ["unassign"]


@pytest.mark.parametrize("fault", ["malformed_lease", "corrupt_export"])
def test_runtime_fault_stops_child_and_unassigns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    root = tmp_path / fault
    root.mkdir()
    spec = {
        "job_id": "job",
        "generation": "g",
        "root": str(root),
        "command": [sys.executable, "-c", "import time; time.sleep(60)"],
        "env": {},
        "timeout_seconds": 60,
        "lease_seconds": 10,
        "checkpoint_grace_seconds": 0.1,
        "max_export_bytes": 100,
        "release_endpoint": "unitendpoint",
    }
    calls: list[str] = []
    monkeypatch.setattr(runtime, "_validate_spec", lambda value: value)
    monkeypatch.setattr(runtime, "_require_release_safety", lambda *_: None)
    monkeypatch.setattr(runtime, "_unassign", lambda *_: calls.append("unassign"))
    monkeypatch.setattr(runtime, "_release_credential_expiring", lambda *_: False)
    if fault == "malformed_lease":
        monkeypatch.setattr(
            runtime, "_lease_expired", lambda *_: (_ for _ in ()).throw(ValueError("bad lease"))
        )
    else:
        spec["command"] = [
            sys.executable,
            "-c",
            "import os,time,pathlib; p=pathlib.Path(os.environ['COLAB_OUTPUT_DIR'])/'bad'; "
            "p.write_bytes(b'bad'); p.with_name('bad.sha256').write_text('0'*64); "
            "p.with_name('bad.complete').touch(); time.sleep(60)",
        ]
        monkeypatch.setattr(runtime, "STATE_INTERVAL_SECONDS", 0.01)

    with pytest.raises(RuntimeError, match="runtime supervisor failure"):
        runtime.supervise(spec)
    assert calls == ["unassign"]
    assert json.loads((root / "terminal.json").read_text())["status"] == "failed"


def test_state_storage_failure_stops_owned_child_and_unassigns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "disk-full"
    root.mkdir()
    spec = {
        "job_id": "job",
        "generation": "g",
        "root": str(root),
        "command": [sys.executable, "-c", "import time; time.sleep(60)"],
        "env": {},
        "timeout_seconds": 60,
        "lease_seconds": 10,
        "checkpoint_grace_seconds": 0.1,
        "max_export_bytes": 100,
        "release_endpoint": "unitendpoint",
    }
    calls: list[str] = []
    real_write = runtime._atomic_json
    writes = 0

    def disk_full(path: Path, value: dict) -> None:
        nonlocal writes
        writes += 1
        if writes >= 3:
            raise OSError("disk full")
        real_write(path, value)

    monkeypatch.setattr(runtime, "_validate_spec", lambda value: value)
    monkeypatch.setattr(runtime, "_require_release_safety", lambda *_: None)
    monkeypatch.setattr(runtime, "_unassign", lambda *_: calls.append("unassign"))
    monkeypatch.setattr(runtime, "_release_credential_expiring", lambda *_: False)
    monkeypatch.setattr(runtime, "_atomic_json", disk_full)

    with pytest.raises(RuntimeError, match="runtime supervisor failure"):
        runtime.supervise(spec)
    assert calls == ["unassign"]


def test_initial_lease_storage_failure_still_attempts_unassign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "initial-lease"
    root.mkdir()
    spec = {
        "job_id": "job",
        "generation": "g",
        "root": str(root),
        "command": ["true"],
        "env": {},
        "timeout_seconds": 1,
        "lease_seconds": 1,
        "checkpoint_grace_seconds": 0.1,
        "max_export_bytes": 1,
        "release_endpoint": "unitendpoint",
    }
    calls: list[str] = []
    monkeypatch.setattr(runtime, "_validate_spec", lambda value: value)
    monkeypatch.setattr(runtime, "_require_release_safety", lambda *_: None)
    monkeypatch.setattr(
        runtime, "_atomic_json", lambda *_: (_ for _ in ()).throw(OSError("disk full"))
    )
    monkeypatch.setattr(runtime, "_unassign", lambda *_: calls.append("unassign"))
    with pytest.raises(OSError, match="disk full"):
        runtime.supervise(spec)
    assert calls == ["unassign"]


def test_export_cache_hashes_once_and_rejects_later_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    output = root / "output"
    output.mkdir(parents=True)
    payload = output / "snap"
    payload.write_bytes(b"one")
    digest = hashlib.sha256(b"one").hexdigest()
    (output / "snap.sha256").write_text(digest, encoding="utf-8")
    (output / "snap.complete").touch()
    calls = 0
    original = runtime._sha256

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(runtime, "_sha256", counted)
    cache = {}
    assert runtime._exports(root, 100, cache)
    assert runtime._exports(root, 100, cache)
    assert calls == 1
    payload.write_bytes(b"two")
    with pytest.raises(RuntimeError, match="immutable completed export changed"):
        runtime._exports(root, 100, cache)
