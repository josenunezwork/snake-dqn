"""Published inventory aliases must never expand the no-peek boundary."""

from pathlib import Path

import pytest

from research.bl_ppo import bank


def _isolated_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from research.redesign_t7_phase_r_20261016 import seed_audit
    from research.t34_round0d import bank as prior

    monkeypatch.setattr(prior, "audit", lambda *args, **kwargs: {})
    monkeypatch.setattr(seed_audit, "_registered", lambda repo: set())
    directory = tmp_path / "repo" / "research" / "example"
    directory.mkdir(parents=True)
    return directory


def test_same_directory_alias_is_covered_by_regular_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = _isolated_audit(monkeypatch, tmp_path)
    (directory / "SEED_AUDIT.json").write_text("{}")
    (directory / "S1_SEED_AUDIT.json").symlink_to("SEED_AUDIT.json")
    receipt = bank.audit(bank.materialize(), repo=tmp_path / "repo", artifacts=tmp_path)
    assert [row["path"] for row in receipt["published_inventories"]] == [
        "research/example/SEED_AUDIT.json"
    ]


def test_external_inventory_link_is_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = _isolated_audit(monkeypatch, tmp_path)
    sentinel = tmp_path / "unreadable.json"
    sentinel.write_text("MUST NOT READ")
    (directory / "OTHER_SEED_AUDIT.json").symlink_to(sentinel)
    original = Path.read_text

    def guarded(path: Path, *args, **kwargs):
        if path == sentinel or path.name == "OTHER_SEED_AUDIT.json":
            raise AssertionError("opened external alias")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(ValueError, match="unsafe published"):
        bank.audit(bank.materialize(), repo=tmp_path / "repo", artifacts=tmp_path)
