"""R-A receipt hashing against copied, outcome-free D4096 production metadata."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from research.bl_labels.d4096_resume import canonical_sha as collector_sha
from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_residual import ra_guard
from research.bl_residual.protocol import canonical_sha as residual_sha

FIXTURES = Path(__file__).parent / "fixtures/d4096_start_metadata"
ALLOCATION_SHA = "74b1c5db8cd0c082311cf2c02666418847855298762174e0b9f721762cbff229"
CONTROL_SHA = "b1182f800a23dd98117133d2051be4773ac9d5d65e01f9961e3a3ad39a563584"


def _metadata() -> tuple[bytes, dict, list[tuple[str, dict]]]:
    original = gzip.decompress((FIXTURES / "d4096-control.json.gz").read_bytes())
    admitted = json.loads(gzip.decompress((FIXTURES / "d4096-admitted.json.gz").read_bytes()))
    starts = [(p.name, json.loads(p.read_text())) for p in sorted(FIXTURES.glob("*.start.json"))]
    return original, admitted, starts


def test_real_start_receipts_bind_collector_hashes() -> None:
    original, admitted, starts = _metadata()
    assert len(starts) == 7
    assert json.loads(original)["allocation"] == admitted["allocation"]
    assert collector_sha(admitted["allocation"]) == ALLOCATION_SHA
    assert collector_sha(admitted) == CONTROL_SHA
    # This was the failed R-A convention; both hashes must differ on real metadata.
    assert residual_sha(admitted["allocation"]) != ALLOCATION_SHA
    assert residual_sha(admitted) != CONTROL_SHA
    for _, start in starts:
        assert start["allocation_sha256"] == ALLOCATION_SHA
        assert start["control_sha256"] == CONTROL_SHA


def _staged_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Relocate metadata to temporary authority paths; never open live artifacts."""
    original, admitted, starts = _metadata()
    control_path = tmp_path / "d4096-control.json"
    state = tmp_path / "d4096-state-mac-01"
    export_bytes(control_path, original)
    assert digest(control_path) == admitted["mac_gate"]["original_control_sha256"]
    admitted["mac_gate"].update(
        package_root=str(tmp_path.resolve()),
        state_dir=str(state.resolve()),
        original_control_path=str(control_path.resolve()),
    )
    export_json(tmp_path / "d4096-admitted.json", admitted)
    paths = []
    for name, start in starts:
        # Only the control's authority paths changed; the real allocation hash stays fixed.
        start["control_sha256"] = collector_sha(admitted)
        start["state_dir"] = str(state.resolve())
        path = state / name
        export_json(path, start)
        paths.append(path)
    monkeypatch.setattr(ra_guard, "D4096_CONTROL", control_path)
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    return paths


def test_authoritative_starts_accepts_real_collector_metadata(tmp_path, monkeypatch) -> None:
    paths = _staged_metadata(tmp_path, monkeypatch)
    authority = ra_guard.authoritative_starts()
    assert [row["path"] for row in authority["starts"]] == [str(path) for path in paths]
    assert {row["allocation_sha256"] for row in authority["starts"]} == {ALLOCATION_SHA}
    earliest = min(authority["starts"], key=lambda row: (row["started_epoch"], row["path"]))
    timing = ra_guard.timing_identity(Path(earliest["path"]), None)
    assert timing["d4096_start"] == earliest
    assert timing["start_authority"] == authority


@pytest.mark.parametrize("field", ["allocation_sha256", "control_sha256"])
def test_real_metadata_still_refuses_foreign_hash(tmp_path, monkeypatch, field) -> None:
    paths = _staged_metadata(tmp_path, monkeypatch)
    start = json.loads(paths[0].read_text())
    start[field] = "0" * 64
    # Re-seal only the temporary copy: rejection must be semantic, not a broken sidecar.
    replacement = paths[0].with_name("attempt-foreign.start.json")
    export_json(replacement, start)
    with pytest.raises(ValueError, match="sealed D4096 attempt-start"):
        ra_guard.authoritative_starts()
