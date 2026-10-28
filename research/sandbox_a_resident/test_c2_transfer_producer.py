"""Native 17 MiB download proof against the subprocess fake, without chunk mocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.colab_cli import supervisor, transfers
from research.sandbox_a_resident import training_resume as resume
from research.sandbox_a_resident.capture_supervisor_goldens import REPO, _load

SEED = b"train01-c2-native-chunks-v1"
SIZE = 17 * 1024 * 1024 + 19


def test_native_multichunk_download_and_interrupted_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers = _load(REPO / "tests/test_colab_cli_supervisor.py", "transfer_capture_helpers")
    binary = helpers._fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "config.json", "offline-transfer")
    source = tmp_path / "remote/offline-transfer/content/checkpoint.bin"
    source.parent.mkdir(parents=True)
    content = hashlib.shake_256(SEED).digest(SIZE)
    source.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    output = tmp_path / "output"
    training = output / "exports/training"
    training.mkdir(parents=True)
    canonical = training / "iteration-0016.pt.xz"
    # This is an opaque transport fixture, never a valid/resumable checkpoint.
    transfers.download_file(cli, "/content/checkpoint.bin", canonical, SIZE, expected_sha256=sha)
    assert canonical.read_bytes() == content
    Path(str(canonical) + ".sha256").write_text(sha + "\n")
    Path(str(canonical) + ".complete").write_text("complete\n")
    calls = helpers._events(tmp_path)
    downloaded = [e["argv"][-2] for e in calls if "download" in e["argv"]]
    assert [Path(path).name for path in downloaded] == [
        "00000000.part",
        "00000001.part",
        "00000002.part",
    ]
    assert all(e["argv"][:2] == ["--auth", "oauth2"] for e in calls)
    original = cli._call_once
    observed = []

    def interrupt(*args, **kwargs) -> str:
        result = original(*args, **kwargs)
        if args[0] == "download" and args[-2].endswith("/00000001.part"):
            part = Path(args[-1])
            data = part.read_bytes()
            observed.append(
                {"index": 1, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
            # Capture before the producer's finally cleans up its own scratch.
            retained = training / ".transfer-captured.1"
            retained.write_bytes(data)
            raise KeyboardInterrupt("offline host-loss boundary after native chunk download")
        return result

    monkeypatch.setattr(cli, "_call_once", interrupt)
    with pytest.raises(KeyboardInterrupt, match="offline host-loss boundary"):
        transfers.download_file(
            cli,
            "/content/checkpoint.bin",
            training / ".transfer-interrupted",
            SIZE,
            expected_sha256=sha,
        )
    assert observed == [
        {
            "index": 1,
            "size": transfers.DOWNLOAD_CHUNK_BYTES,
            "sha256": hashlib.sha256(content[8 * 1024 * 1024 : 16 * 1024 * 1024]).hexdigest(),
        }
    ]
    # Receipt-less inspection still scans retained chunks. It grants no stop or ledger.
    resume._check_output_root_fault_evidence(output)
    (training / ".transfer-captured.1").write_bytes(b"wrong offset twin")
    with pytest.raises(resume.ResumeRefused, match="differs from verified payload"):
        resume._check_output_root_fault_evidence(output)
    evidence = {
        "seed_utf8": SEED.decode(),
        "size": SIZE,
        "sha256": sha,
        "download_paths": downloaded,
        "captured_chunks": observed,
        "transfer_records": cli.transfer_records,
        "contents_transfers": cli.contents_transfers,
    }
    (tmp_path / "native-chunk-proof.json").write_text(json.dumps(evidence, indent=2) + "\n")
