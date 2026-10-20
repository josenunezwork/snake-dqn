"""The reused diagnostic inventory is two checkpoints on 32 original worlds."""

import pytest

from research.t34_pathb.precheck_v2 import ROOT, materialize_t3, validate_t3_row

pytestmark = pytest.mark.skipif(
    not (ROOT / "ceiling-go-for-bank.json").is_file(),
    reason="original T3 dense artifacts are not installed",
)


def test_materializer_preserves_64_original_trajectories_on_32_seeds():
    rows = materialize_t3()
    assert len(rows) == 64
    assert len({row["seed"] for row in rows}) == 32
    assert {(row["t3_source_arm"], row["t3_env"]) for row in rows} == {
        (arm, env) for arm in ("b8_s0", "b8_s3") for env in range(32)
    }
    for env in range(32):
        assert rows[env]["seed"] == rows[32 + env]["seed"]
    changed = dict(rows[0], seed=rows[0]["seed"] + 1)
    with pytest.raises(ValueError, match="differs"):
        validate_t3_row(changed)
    rows[0]["specs"][0][1] = "tampered"
    with pytest.raises(ValueError, match="differs"):
        validate_t3_row(rows[0])
    assert materialize_t3()[0]["specs"][0][1] != "tampered"


def test_original_b8_s0_precheck_does_not_change_registered_b8_s3():
    from research.t34_pathb.adapter_v2 import B8_SHA, open_precheck, open_terminal

    row = materialize_t3()[0]
    session = open_precheck(row, 24, "bare")
    try:
        assert session.identity().checkpoint == row["t3_checkpoint"]["sha256"]
        assert session.identity().checkpoint != B8_SHA
        session.step()
        assert session.frame == 1
    finally:
        session.close()
    release = open_terminal(row, 24, "bare")
    try:
        assert release.identity().checkpoint == B8_SHA
    finally:
        release.close()
