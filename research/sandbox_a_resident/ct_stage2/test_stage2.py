"""Focused downstream identity, stage resume, refusal and recipe regressions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.sandbox_a_resident.ct_fit import ArmEvaluationPredictions
from research.sandbox_a_resident.ct_h5000.task import CTControl
from research.sandbox_a_resident.ct_stage2 import run as runner
from research.sandbox_a_resident.ct_stage2.bank import BANK_BYTES, BANK_SHA256
from research.sandbox_a_resident.ct_stage2.prepare import expected_science
from research.sandbox_a_resident.ct_stage2.resume import materialize_inputs
from research.sandbox_a_resident.ct_stage2.units import (
    NAMESPACE,
    STUDY,
    build_ledger,
    complete_file,
    empty_ledger,
    load_ledger,
    publish_stage,
)
from tools.sandbox.supervisor import require_source_delivery


def config() -> dict:
    return {
        "schema": "ct-stage2-control/v1",
        "study": STUDY,
        "source_commit": "a" * 40,
        "fixture": False,
        "bank": {
            "payload": {"size": BANK_BYTES, "sha256": BANK_SHA256},
            "support_report": {"status": "PASS"},
        },
        "science": expected_science("a" * 40),
    }


def test_science_matches_inherited_recipe_and_fixture_is_separate() -> None:
    original = CTControl(source_commit="a" * 40).to_dict()
    actual = expected_science("a" * 40)
    for key in original:
        if key not in {"max_stage_export_bytes", "max_total_export_bytes"}:
            assert actual[key] == original[key], key
    assert actual["branch_horizon"] == 300
    assert actual["minibatches"] == 2000
    assert actual["bootstrap_resamples"] == 10000
    assert empty_ledger(config())["namespace"] == NAMESPACE
    assert empty_ledger(config()) != empty_ledger({**config(), "fixture": True})


def test_complete_units_skip_partial_publish_and_refuse_tampering(tmp_path: Path) -> None:
    cfg = config()
    exports = tmp_path / "exports"
    publish_stage(exports, 1, lambda p: p.write_bytes(b"payload"), cfg)
    (exports / "stage-0002.pt").write_bytes(b"unfinished")
    ledger = build_ledger(cfg, [exports])
    assert [row["stage"] for row in ledger["units"]] == [1]
    inputs = tmp_path / "inputs"
    materialize_inputs(cfg, [exports], inputs)
    path = inputs / "verified-ledger.json"
    assert list(load_ledger(path, cfg, inputs)) == [1]
    (inputs / "resume/stage-0001.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_ledger(path, cfg, inputs)


@pytest.mark.parametrize("change", ["namespace", "allocation", "duplicate", "gap", "path"])
def test_ledger_refuses_foreign_or_unsafe_units(tmp_path: Path, change: str) -> None:
    cfg = config()
    exports, inputs = tmp_path / "exports", tmp_path / "inputs"
    publish_stage(exports, 1, lambda p: p.write_bytes(b"payload"), cfg)
    materialize_inputs(cfg, [exports], inputs)
    path = inputs / "verified-ledger.json"
    ledger = json.loads(path.read_text())
    if change in {"namespace", "allocation"}:
        ledger[change] = "foreign"
    elif change == "duplicate":
        ledger["units"].append(ledger["units"][0])
    elif change == "gap":
        ledger["units"][0]["stage"] = 2
    else:
        ledger["units"][0]["payload"]["path"] = "../stage-0001.pt"
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        load_ledger(path, cfg, inputs)


def test_completion_marker_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"x")
    from research.sandbox_a_resident.units import digest

    Path(str(path) + ".sha256").write_text(digest(path) + "\n")
    Path(str(path) + ".complete").write_text("unfinished\n")
    with pytest.raises(ValueError, match="completion"):
        complete_file(path)


def test_conflicting_union_refused(tmp_path: Path) -> None:
    cfg = config()
    for i in range(2):
        publish_stage(tmp_path / str(i), 1, lambda p, i=i: p.write_bytes(bytes([i])), cfg)
    with pytest.raises(ValueError, match="conflicting"):
        build_ledger(cfg, [tmp_path / "0", tmp_path / "1"])


def test_real_execute_policy_accepts_only_exact_script() -> None:
    require_source_delivery(
        {
            "sandbox_source_delivery": "background-sparse-v1",
            "script": "research/sandbox_a_resident/ct_stage2/job.sh",
        }
    )
    with pytest.raises(ValueError, match="exclusive"):
        require_source_delivery(
            {
                "sandbox_source_delivery": "background-sparse-v1",
                "script": "research/sandbox_a_resident/ct_stage2/job.sh.bad",
            }
        )


@dataclass
class Chunk:
    chunk_index: int
    root_indices: np.ndarray

    def to_state_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_state_dict(cls, value: dict) -> Chunk:
        return cls(**value)


def test_full_unit_resume_never_repeats_branches_fits_or_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config()
    fake_data = SimpleNamespace(
        store=list(range(8)),
        anchors=SimpleNamespace(num_rows=16),
        world_partitions=("train", "assessment"),
    )
    monkeypatch.setattr(runner, "load_bank", lambda *a: fake_data)
    monkeypatch.setattr(runner, "forced_state_determinism_suite", lambda d: {"status": "PASS"})
    monkeypatch.setattr(runner, "validate_geometric_fixtures", lambda: {"status": "PASS"})
    monkeypatch.setattr(runner, "load_ego2s_checkpoint", lambda *a: torch.nn.Linear(1, 1))
    monkeypatch.setattr(runner, "_load_vector61", lambda *a: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        runner,
        "digest",
        lambda p: {
            "b8": runner.B8_S3_SHA256,
            "frp3": runner.FRP3_S12_SHA256,
            "champion": runner.CHAMPION_SHA256,
        }[p.name],
    )
    monkeypatch.setattr(runner, "ScoredRootChunk", Chunk)
    calls = []

    def score(**kwargs: object) -> Chunk:
        calls.append("score")
        return Chunk(kwargs["chunk_index"], kwargs["root_indices"])

    merged = SimpleNamespace(
        total_logical_branch_steps=100,
        total_executed_sim_steps=90,
        total_scoring_wall_sec=1,
        chunk_sha256s=["b" * 64] * 8,
    )
    monkeypatch.setattr(runner, "score_root_chunk", score)
    monkeypatch.setattr(runner, "merge_scored_chunks", lambda *a, **k: merged)
    prediction = ArmEvaluationPredictions(
        "full_bv8",
        np.zeros((8, 6)),
        np.zeros(8, dtype=int),
        np.zeros(8),
        np.zeros(8),
        False,
        np.zeros(8, dtype=int),
        {},
    )

    def fit(**kwargs: object) -> tuple:
        calls.append("fit")
        return {"full_bv8": prediction}, {"mock": True}

    def evaluate(**kwargs: object) -> dict:
        calls.append("evaluate")
        return {"verdict": "MOCK_ONLY"}

    monkeypatch.setattr(runner, "fit_all_three_arms", fit)
    monkeypatch.setattr(runner, "evaluate_ct_gates", evaluate)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(empty_ledger(cfg)))
    kwargs = {
        "bank_root": tmp_path,
        "ledger": ledger_path,
        "input_root": tmp_path,
        "checkpoint": Path("b8"),
        "frp3": Path("frp3"),
        "champion": Path("champion"),
        "device": "cpu",
        "watchdog": SimpleNamespace(progress=lambda *a: None),
    }
    assert runner.run(cfg, output=tmp_path / "first", **kwargs)[0] == 0
    assert calls == ["score"] * 8 + ["fit", "evaluate"]
    inputs = tmp_path / "resume-inputs"
    materialize_inputs(cfg, [tmp_path / "first"], inputs)
    calls.clear()
    kwargs.update(ledger=inputs / "verified-ledger.json", input_root=inputs)
    code, summary = runner.run(cfg, output=tmp_path / "second", **kwargs)
    assert code == 0 and summary["resumed_stages"] == list(range(1, 11))
    assert calls == []
