"""Coordinator contracts for the T4e disk-backed return collection CLI."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from research.t4e_return_20261016 import run_collection as runner


def _job(index: int, *, holdout: bool = False) -> dict:
    return {
        "plan_index": index,
        "arm": "B8_s0",
        "mix": "scripted",
        "seed": 1000 + index,
        "holdout": holdout,
    }


def _row(index: int, *, holdout: bool = False) -> dict:
    return {
        "plan_index": index,
        "source_decision": 5,
        "length": 800,
        "q": np.arange(6, dtype=np.float32),
        "state_packed": np.asarray([index], dtype=np.uint8),
        "state_scalars": np.zeros(60, dtype=np.float32),
        "fingerprint": f"f-{index}",
        "holdout": holdout,
        "legal": np.asarray([0, 1]),
        "death_distance": 20,
        "b8_action": 1,
    }


def test_parser_caps_workers_and_check_reports_declared_resources(tmp_path, monkeypatch):
    parser = runner.build_parser()
    args = parser.parse_args(["--out", str(tmp_path / "new"), "--workers", "12", "--check"])
    monkeypatch.setattr(runner, "finals", lambda _root: {f"B8_s{i}": tmp_path for i in range(5)})
    monkeypatch.setattr(runner.collection, "historical_cuts", lambda _path: (np.zeros(9), set()))
    monkeypatch.setattr(runner, "provenance", lambda *_args: {"bound": True})
    receipt = runner.run(args)
    assert receipt["status"] == "CHECK_ONLY"
    assert receipt["workers"] == 12
    assert receipt["cpu_threads_per_worker"] == 1
    assert receipt["mix_counts_per_final"] == {"scripted": 27, "mixed": 9, "frozen": 4}
    with pytest.raises(ValueError, match="1 through 12"):
        runner.run(Namespace(**{**vars(args), "workers": 13}))


def test_staged_metadata_uses_publisher_bounded_spool_api(tmp_path, monkeypatch):
    receipt = {
        "job": _job(1),
        "decisive_spool": "decisive.npz",
        "decisive_spool_sha256": "digest",
        "candidate_spool": "candidate.npz",
        "candidate_spool_sha256": "digest",
    }
    calls = []

    def load(path: Path):
        calls.append(path.name)
        return [_row(1)]

    monkeypatch.setattr(runner.publisher, "load_spool", load)
    monkeypatch.setattr(runner, "_hash", lambda _path: "digest")
    decisive, candidates = runner.staged_metadata(tmp_path, [receipt])
    assert calls == ["decisive.npz", "candidate.npz"]
    assert decisive[0]["fingerprint"] == candidates[0]["fingerprint"] == "f-1"


def test_non_matching_runtime_error_is_not_treated_as_more_collection(tmp_path, monkeypatch):
    """Only an explicit incomplete-strata receipt may authorize another fresh cycle."""
    monkeypatch.setattr(runner, "finals", lambda _root: {f"B8_s{i}": tmp_path for i in range(5)})
    monkeypatch.setattr(runner.collection, "historical_cuts", lambda _path: (np.zeros(9), set()))
    monkeypatch.setattr(runner, "provenance", lambda *_args: {"bound": True})
    monkeypatch.setattr(runner, "require_ac", lambda: None)
    monkeypatch.setattr(runner, "make_cycle_jobs", lambda *_args: ([_job(0)], {}))
    monkeypatch.setattr(runner.publisher, "run_phase_a_to_disk", lambda *_args, **_kw: [])
    monkeypatch.setattr(runner, "staged_metadata", lambda *_args: ([], []))
    monkeypatch.setattr(
        runner, "selection", lambda *_args: (_ for _ in ()).throw(RuntimeError("disk failure"))
    )
    args = Namespace(
        out=tmp_path / "out",
        workers=1,
        artifacts=tmp_path,
        prereg=tmp_path / "pre",
        reference_bank=tmp_path / "ref",
        b8_root=tmp_path,
        check=False,
    )
    with pytest.raises(RuntimeError, match="disk failure"):
        runner.run(args)


def test_matched_small_cycle_continues_until_threshold_then_finalizes(tmp_path, monkeypatch):
    """Exact matching alone never permits publication below the registered 8,000-state floor."""
    monkeypatch.setattr(runner, "finals", lambda _root: {f"B8_s{i}": tmp_path for i in range(5)})
    monkeypatch.setattr(runner.collection, "historical_cuts", lambda _path: (np.zeros(9), set()))
    source = {
        "preregistration": {"path": "pre", "sha256": "x"},
        "reference_bank": {"path": "ref", "manifest_sha256": "x"},
        "finals": {},
        "source_tree": {"git_head": "head", "sha256": {}},
    }
    monkeypatch.setattr(runner, "provenance", lambda *_args: source)
    monkeypatch.setattr(runner, "verify_bound_inputs", lambda _source: None)
    monkeypatch.setattr(runner, "require_ac", lambda: None)
    monkeypatch.setattr(runner, "_atomic_json", lambda *_args: None)
    calls = {"phase_a": 0, "selected": 0}

    def jobs(_namespace, cycle, start, _finals, _roots):
        return ([_job(start)], {"cycle": cycle})

    def phase_a(_root, jobs, *_args, **_kwargs):
        calls["phase_a"] += 1
        return [{"job": jobs[0]}]

    def select(*_args):
        count = 1 if calls["phase_a"] == 1 else runner.collection.MIN_DECISIVE
        plan_index = 0 if calls["phase_a"] == 1 else 200
        decisive = [_row(plan_index) for _ in range(count)]
        outside = [_row(plan_index) for _ in range(count)]
        return decisive, outside, {"exact": True}

    monkeypatch.setattr(runner, "make_cycle_jobs", jobs)
    monkeypatch.setattr(runner.publisher, "run_phase_a_to_disk", phase_a)
    monkeypatch.setattr(runner, "staged_metadata", lambda *_args: ([], []))
    monkeypatch.setattr(runner, "selection", select)
    monkeypatch.setattr(
        runner.publisher,
        "publish_selected_staged",
        lambda *_args, **_kwargs: calls.__setitem__("selected", calls["selected"] + 1)
        or {"train_path": "train", "holdout_path": "holdout"},
    )
    monkeypatch.setattr(
        runner.publisher, "validate_completed_banks", lambda *_args: {"status": "PASS"}
    )
    args = Namespace(
        out=tmp_path / "out",
        workers=1,
        artifacts=tmp_path,
        prereg=tmp_path / "pre",
        reference_bank=tmp_path / "ref",
        b8_root=tmp_path,
        check=False,
    )
    result = runner.run(args)
    assert calls == {"phase_a": 2, "selected": 1}
    assert result["cycles"] == 2
    assert result["training_decisive_states"] == runner.collection.MIN_DECISIVE
