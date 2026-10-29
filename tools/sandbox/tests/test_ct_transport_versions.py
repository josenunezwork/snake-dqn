"""Keep historical capture identity separate from current host qualification."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_cuda_capture
from tools.sandbox import preflight


@pytest.mark.parametrize("tree", sorted(preflight.QUALIFIED_TRANSPORT_TREES) + ["f" * 40])
def test_only_exact_qualified_host_trees(monkeypatch: pytest.MonkeyPatch, tree: str) -> None:
    monkeypatch.setattr(preflight, "_run", lambda argv: tree if "rev-parse" in argv else "")
    if tree in preflight.QUALIFIED_TRANSPORT_TREES:
        preflight.verify_transport_tree()
    else:
        with pytest.raises(preflight.PreflightError, match="exact qualification"):
            preflight.verify_transport_tree()


def test_capture_hash_uses_frozen_git_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Transport Fixture")
    git("config", "user.email", "transport@example.invalid")
    path = tmp_path / "supervisor.py"
    path.write_bytes(b"historical source")
    git("add", ".")
    git("commit", "-qm", "capture-era transport")
    pin = git("rev-parse", "HEAD")
    path.write_bytes(b"new host source")
    git("add", ".")
    git("commit", "-qm", "host successor")
    monkeypatch.setattr(ct_cuda_capture, "ROOT", tmp_path)
    ct_cuda_capture._pinned_transport_digest.cache_clear()
    try:
        assert (
            ct_cuda_capture._pinned_transport_digest(pin, "supervisor.py")
            == hashlib.sha256(b"historical source").hexdigest()
        )
        with pytest.raises(ValueError, match="unavailable"):
            ct_cuda_capture._pinned_transport_digest(pin, "missing.py")
        with pytest.raises(ValueError, match="full commit"):
            ct_cuda_capture._pinned_transport_digest("HEAD", "supervisor.py")
    finally:
        ct_cuda_capture._pinned_transport_digest.cache_clear()
