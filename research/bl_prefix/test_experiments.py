from __future__ import annotations

import pytest

from research.bl_prefix.analysis import IncompleteResult, e6_summary, e7_summary


def _e6_rows() -> list[dict]:
    rows = []
    for stratum, count in (("early", 128), ("ordinary", 128)):
        for index in range(count):
            mix = ("scripted", "frozen", "mixed")[index % 3]
            for stream in (0, 1):
                for controller, score in (("B8", 10.0), ("R", 11.0)):
                    rows.append(
                        {
                            "phase": "primary",
                            "stratum": stratum,
                            "mix": mix,
                            "source_world": f"{stratum}-{index}",
                            "stream": stream,
                            "controller": controller,
                            "score": score,
                        }
                    )
    return rows


def test_e6_world_clustered_statistic_and_epd_are_order_invariant() -> None:
    rows = _e6_rows()
    result = e6_summary(rows, "R")
    reversed_result = e6_summary(list(reversed(rows)), "R")
    assert result["decision"] == "NEEDS_SCALE"
    assert result["early"]["mean"] == pytest.approx(1.0)
    assert result["epd"] == reversed_result["epd"]
    assert result["epd"]["df"] is None  # exact all-zero within-cell variance exports as JSON.


def test_e6_rejects_missing_stream_or_scientific_cardinality() -> None:
    rows = _e6_rows()[:-1]
    with pytest.raises(IncompleteResult):
        e6_summary(rows, "R")


def test_e7_requires_both_confirmation_streams_and_explicit_survival() -> None:
    rows = []
    for source in ("a", "b", "c"):
        for ancestor in (100, 150, 200):
            for stream in (0, 1):
                rows.append(
                    {
                        "source_world": source,
                        "ancestor": ancestor,
                        "stream": stream,
                        "candidate_score": 30.0,
                        "b8_score": 0.0,
                        "candidate_survival": True,
                        "b8_survival": True,
                        "b8_death_frame": None,
                        "b8_last_living_length": None,
                        "candidate_tail_mean_length": None,
                        "fallback_count": 0,
                    }
                )
    result = e7_summary(rows, ["a", "b", "c"])
    assert result["decision"] == "TRANSFER_ADMISSIBLE"
    rows[-1].pop("candidate_survival")
    with pytest.raises(IncompleteResult):
        e7_summary(rows, ["a", "b", "c"])


def test_e6_rejects_relabelled_and_repeated_actual_worlds(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from research.bl_prefix import e6

    monkeypatch.setattr(e6, "verify_export", lambda _path: None)
    monkeypatch.setattr(
        e6, "load_root", lambda _path: SimpleNamespace(seed=101, mix="scripted", frame=228)
    )
    monkeypatch.setattr(e6, "relocate_root", lambda root, _mapping: root)
    bank = {
        "preflight": True,
        "roots": [
            {
                "stratum": "early",
                "mix": "scripted",
                "source_world": str(label),
                "root": f"root-{index}.npz",
            }
            for index, label in enumerate((101, 102, 103))
        ],
    }
    with pytest.raises(IncompleteResult, match="identity"):
        e6._roots(bank, tmp_path, {}, {101, 102, 103})
    for row in bank["roots"]:
        row["source_world"] = "101"
    with pytest.raises(IncompleteResult, match="duplicated actual"):
        e6._roots(bank, tmp_path, {}, {101})


def test_critic_receipts_bind_final_endpoint_and_finite_preparation(tmp_path) -> None:
    import hashlib
    import json

    from research.bl_ppo.exports import export_bytes, export_json
    from research.bl_prefix.e6 import _critic_ok

    checkpoints, receipts = {}, {}
    records = {}
    for arm in ("R", "F"):
        checkpoint = tmp_path / f"{arm}.pth"
        checkpoint.write_bytes(arm.encode())
        checkpoints[arm] = str(checkpoint)
        record = {
            "complete": True,
            "training_complete": True,
            "checkpoint_role": "final",
            "arm": arm,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "refit_transitions": 2_000_000,
            "late_body_ev": 0.1,
            "late_body_mse": 1.0,
            "baseline_mse": 2.0,
        }
        path = tmp_path / f"{arm}.json"
        export_json(path, record)
        receipts[arm] = str(path)
        records[arm] = record
    assert _critic_ok({"preflight": False, "critic_receipts": receipts}, checkpoints)
    bad = tmp_path / "nonfinite.json"
    export_bytes(bad, json.dumps(dict(records["R"], late_body_ev=float("nan"))).encode())
    assert not _critic_ok(
        {"preflight": False, "critic_receipts": dict(receipts, R=str(bad))}, checkpoints
    )
    (tmp_path / "R.pth").write_bytes(b"changed")
    assert not _critic_ok({"preflight": False, "critic_receipts": receipts}, checkpoints)
