"""Verify the OLD fallback path guard without running a factory parent."""

from pathlib import Path

import pytest

from research.bl_labels import d1024_w12, frozen_factory
from research.bl_labels.exports import empty_ledger, publish_json


@pytest.mark.parametrize(
    ("head", "dirty", "detached", "accepted"),
    [
        (frozen_factory.OLD_PIN, "", True, True),
        ("a" * 40, "", True, False),
        (frozen_factory.OLD_PIN, " M factory.py", True, False),
        (frozen_factory.OLD_PIN, "", False, False),
    ],
)
def test_old_root_requires_frozen_clean_detached_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str,
    dirty: str,
    detached: bool,
    accepted: bool,
) -> None:
    """A valid OLD pin must pass; wrong pins, dirt and branches remain refused."""
    allocation = [{"allocation_id": "parent-0", "seed": 1}]
    control = {
        "namespace": "old-path-test",
        "allocation": allocation,
        "mac_gate": {"package_root": str(tmp_path), "state_dir_name": "state"},
    }
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(control["namespace"], allocation))

    def check_output(command: list[str], **_kwargs: object) -> str:
        return head if command[1] == "rev-parse" else dirty

    monkeypatch.setattr(d1024_w12.subprocess, "check_output", check_output)
    monkeypatch.setattr(
        d1024_w12.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": int(detached)})(),
    )
    args = (control, ledger, tmp_path / "output", tmp_path / "state", tmp_path / "old")
    if accepted:
        d1024_w12._check_paths(*args)
    else:
        with pytest.raises(ValueError, match="clean detached frozen 61900af2"):
            d1024_w12._check_paths(*args)
