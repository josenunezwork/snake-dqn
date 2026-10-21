from pathlib import Path

import pytest

from research.bl_e16.manifests import ManifestRefused, git_head, make_manifest


def _inputs(root: Path):
    b8 = root / "b8.pth"
    b8.write_bytes(b"b8")
    roster = []
    for n in range(5):
        p = root / f"r{n}.pth"
        p.write_bytes(bytes([n]))
        roster.append(p)
    return b8, tuple(roster)


def test_blocked_a_template_has_no_fake_parent(tmp_path):
    b8, roster = _inputs(tmp_path)
    m = make_manifest(
        commit=git_head(),
        output_dir=tmp_path / "out",
        arm="A",
        b8_checkpoint=b8,
        roster_checkpoints=roster,
    )
    assert (
        m["transport"]["dispatch_ready"] is False
        and m["e16"]["input_admission"]["status"] == "NOT_DISPATCHABLE"
    )
    assert len(m["inputs"]) == 6
    assert not any(x["dest"] == "parent/F.pth" for x in m["inputs"])
    assert m["e16"]["purpose"] == "BLOCKED_TEMPLATE_SCHEMA_VALIDATION_ONLY"


def test_geometry_records_full_forecast(tmp_path):
    b8, roster = _inputs(tmp_path)
    m = make_manifest(
        commit=git_head(),
        output_dir=tmp_path / "out",
        arm="G",
        b8_checkpoint=b8,
        roster_checkpoints=roster,
    )
    assert m["e16"]["forecast"]["training_transitions"] == 12000000
    assert m["e16"]["forecast"]["evaluation_world_steps"] == 5760000
    assert m["e16"]["forecast"]["unit_cap"] == 20.0
    assert m["e16"]["arm_precondition"].endswith("E14 pass")


def test_missing_roster_is_refused(tmp_path):
    b8, roster = _inputs(tmp_path)
    with pytest.raises(ManifestRefused, match="exactly five"):
        make_manifest(
            commit="c" * 40,
            output_dir=tmp_path / "out",
            arm="A",
            b8_checkpoint=b8,
            roster_checkpoints=roster[:4],
        )
