"""Fail-closed identity fences for M3-C development probe reuse."""

import hashlib
import json

import pytest

from research.redesign_m3c_20261008 import probe_watch as pw


def _checkpoint(tmp_path, name, body):
    path = tmp_path / name
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def _row(path, sha, run="A_s0", file="final.pth"):
    return {"run": run, "file": file, "path": str(path), "sha256": sha}


def _result(mix, sha, veto="none", round_index=52, **extra):
    row = {
        "mix": mix,
        "student_sha256": sha,
        "veto": veto,
        "round_index": round_index,
        "ego2s_forward": "rowwise",
        "ego2s_mask": "advisory" if veto == "none" else None,
    }
    row.update(extra)
    return row


def test_existing_output_for_another_checkpoint_fails_closed(tmp_path):
    checkpoint, sha = _checkpoint(tmp_path, "final.pth", b"selected")
    _, other_sha = _checkpoint(tmp_path, "old.pth", b"old")
    output = tmp_path / "A_s0__final__none.jsonl"
    output.write_text(json.dumps(_result("frozen", other_sha)) + "\n")

    with pytest.raises(pw.ProbeIdentityError, match="another checkpoint.*distinct --out"):
        pw.pending_jobs([_row(checkpoint, sha)], tmp_path, ["final"], ["none"], set())


def test_existing_output_mixed_checkpoint_identities_fails_closed(tmp_path):
    checkpoint, sha = _checkpoint(tmp_path, "final.pth", b"selected")
    _, other_sha = _checkpoint(tmp_path, "old.pth", b"old")
    output = tmp_path / "A_s0__final__none.jsonl"
    output.write_text(
        "\n".join((json.dumps(_result("frozen", sha)), json.dumps(_result("scripted", other_sha))))
        + "\n"
    )

    with pytest.raises(pw.ProbeIdentityError, match="another checkpoint.*distinct --out"):
        pw.pending_jobs([_row(checkpoint, sha)], tmp_path, ["final"], ["none"], set())


def test_matching_checkpoint_reuses_only_completed_mixes(tmp_path):
    checkpoint, sha = _checkpoint(tmp_path, "final.pth", b"selected")
    output = tmp_path / "A_s0__final__none.jsonl"
    output.write_text(json.dumps(_result("frozen", sha)) + "\n")

    jobs = pw.pending_jobs([_row(checkpoint, sha)], tmp_path, ["final"], ["none"], set())

    assert {job[-1] for job in jobs} == {"scripted", "mixed"}


def test_same_public_name_from_different_attempt_hashes_fails_closed(tmp_path):
    first_dir = tmp_path / "attempt00"
    second_dir = tmp_path / "attempt01"
    first_dir.mkdir()
    second_dir.mkdir()
    first, first_sha = _checkpoint(first_dir, "final.pth", b"attempt 0")
    second, second_sha = _checkpoint(second_dir, "final.pth", b"attempt 1")

    with pytest.raises(pw.ProbeIdentityError, match="different attempts.*distinct --out"):
        pw.pending_jobs(
            [_row(first, first_sha), _row(second, second_sha)],
            tmp_path,
            ["final"],
            ["none"],
            set(),
        )


def test_manifest_witness_mismatch_fails_closed(tmp_path):
    checkpoint, sha = _checkpoint(tmp_path, "final.pth", b"selected")
    bad_sha = "0" * 64
    assert sha != bad_sha

    with pytest.raises(pw.ProbeIdentityError, match="no longer matches its landed manifest"):
        pw.pending_jobs([_row(checkpoint, bad_sha)], tmp_path, ["final"], ["none"], set())


def test_invalid_existing_identity_fails_closed(tmp_path):
    checkpoint, sha = _checkpoint(tmp_path, "final.pth", b"selected")
    (tmp_path / "A_s0__final__none.jsonl").write_text(json.dumps({"mix": "frozen"}) + "\n")

    with pytest.raises(pw.ProbeIdentityError, match="another checkpoint.*distinct --out"):
        pw.pending_jobs([_row(checkpoint, sha)], tmp_path, ["final"], ["none"], set())
