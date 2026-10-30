"""Regression proof for the independent natural-H5000 support successor."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.compute.gpu_preference import prepare_preference, variants
from research.sandbox_a_resident.ct_h5000.prepare import ct_endpoint_steps
from research.sandbox_a_resident.ct_h5000.sampling import quota_sample, reservoir_admissions
from research.sandbox_a_resident.ct_h5000.task import (
    CTControl,
    _select_matched_ordinary_roots,
    canonical_low63_seed,
    ct_seed_allocations,
    world_partition_table,
)
from research.sandbox_a_resident.ct_h5000.units import (
    allocation,
    empty_ledger,
    load_ledger,
    publish_stage_snapshot,
)
from research.sandbox_a_resident.ct_task import CTControl as OriginalControl


def test_frozen_support_recipe_and_fresh_domain() -> None:
    ctrl = CTControl()
    old = OriginalControl().to_dict()
    new = ctrl.to_dict()
    changed = {key for key in old if old[key] != new[key]}
    assert changed == {"study", "collection_ticks"}
    assert ctrl.collection_ticks == 5000 and ctrl.worlds == 1024
    assert ctrl.collection_cap_seconds == 7200 == 2 * ctrl.collection_base_seconds
    assert ct_endpoint_steps(ctrl) == 5_120_000
    assert new["lookbacks"] == [15, 35, 65, 100, 140, 185, 235, 285]
    assert [
        new[k]
        for k in (
            "min_death_length",
            "min_held_out_deaths",
            "min_held_out_death_worlds",
            "min_held_out_ordinary_roots",
            "min_held_out_anchor_rows",
        )
    ] == [600, 32, 32, 256, 10000]
    labels, parts, counts = world_partition_table()
    assert len(labels) == 1024 and len(parts["train"]) == 820 and len(parts["assessment"]) == 204
    assert counts["assessment"] == {"scripted": 68, "frozen": 68, "mixed": 68}
    alloc = ct_seed_allocations(ctrl)
    assert [row["count"] for row in alloc] == [820 * 5001, 1024, 204 * 5001]
    assert canonical_low63_seed(ctrl.study, 0) != canonical_low63_seed(OriginalControl().study, 0)
    with pytest.raises(ValueError, match="study"):
        replace(ctrl, study=OriginalControl().study)
    with pytest.raises(ValueError, match="2x"):
        replace(ctrl, collection_cap_seconds=7199)


def test_batch_reservoir_is_serial_algorithm_r_and_has_no_immutable_tail() -> None:
    cell = ("train", "scripted", 0)
    cap = 17
    reservoirs: dict = {cell: []}
    seen = {cell: 0}
    rng, oracle_rng = np.random.default_rng(172), np.random.default_rng(172)
    oracle = []
    for start, count in ((0, 1024), (1024, 2000), (3024, 4000)):
        proposals = [(w, cell) for w in range(start, start + count)]
        admissions = reservoir_admissions(
            proposals, reservoirs=reservoirs, seen_counts=seen, capacities={"train": cap}, rng=rng
        )
        for world, key, slot in admissions:
            if slot is None:
                reservoirs[key].append(world)
            else:
                reservoirs[key][slot] = world
        for world in range(start, start + count):
            if len(oracle) < cap:
                oracle.append(world)
            else:
                slot = int(oracle_rng.integers(0, world + 1))
                if slot < cap:
                    oracle[slot] = world
        assert reservoirs[cell] == oracle
        assert len(reservoirs[cell]) == cap
    assert min(reservoirs[cell]) >= cap
    assert seen[cell] == 7024


def test_quota_sample_is_random_before_ordering_and_order_invariant() -> None:
    rows = [SimpleNamespace(tick=i, world_id=i % 10, life_id=0) for i in range(1000)]
    selected = quota_sample(rows, 256, key=("assessment", "scripted", 0))
    assert [r.tick for r in selected] == [
        r.tick for r in quota_sample(rows[::-1], 256, key=("assessment", "scripted", 0))
    ]
    assert max(r.tick for r in selected) > 950
    assert 440 < np.mean([r.tick for r in selected]) < 560
    assert [r.tick for r in selected] == sorted(r.tick for r in selected)


@pytest.mark.parametrize("part,quota", [("train", 1024), ("assessment", 256)])
def test_single_cell_full_quota_after_overlap_exclusion(part: str, quota: int) -> None:
    rows = [SimpleNamespace(tick=i, world_id=0, life_id=0) for i in range(quota + 64)]
    snap = SimpleNamespace(small_tensors={"length": torch.tensor([650])})
    event = SimpleNamespace(partition=part, mix="scripted", lookback_roots=[(15, 0, snap)])
    chosen, report = _select_matched_ordinary_roots(
        {(part, "scripted", 1): rows},
        [event],
        max_ordinary_roots=1280,
        forbidden_world_life_ticks={(0, 0, i) for i in range(64)},
        min_train_ordinary_roots=quota if part == "train" else 0,
        min_assess_ordinary_roots=quota if part == "assessment" else 0,
    )
    assert len(chosen) == quota
    assert all(row.tick >= 64 for row in chosen)
    assert report["partitions"][part]["shortfall_total"] == 0


def test_fresh_ledger_roundtrip_refuses_foreign_namespace_allocation_and_tamper(
    tmp_path: Path,
) -> None:
    ctrl = CTControl(source_commit="a" * 40).to_dict()
    ledger = empty_ledger(ctrl)
    unit = publish_stage_snapshot(
        tmp_path / "units",
        1,
        lambda p: torch.save({"fixed": 1}, p),
        metadata={"allocation": allocation(ctrl), "source_commit": ctrl["source_commit"]},
    )
    ledger["units"].append(unit)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    assert set(load_ledger(path, ctrl)[0]) == {1}
    for field, value in (("namespace", "a100-dev/t34-cf-1028/stages"), ("allocation", "f" * 64)):
        path.write_text(json.dumps({**ledger, field: value}))
        with pytest.raises(ValueError, match="foreign"):
            load_ledger(path, ctrl)
    path.write_text(json.dumps(ledger))
    Path(unit["exports"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        load_ledger(path, ctrl)


def test_h100_first_a100_fallback_preserves_collection_cap_and_recipe() -> None:
    original = {
        "job_id": "sandbox-ct-h5000-20261029-01",
        "output_dir": "/tmp/h5000",
        "timeout_minutes": 300,
        "requirements": {"est_minutes": 150},
        "env": {"SANDBOX_CT_CONTROL_JSON": json.dumps(CTControl().to_dict())},
        "budget": {},
        "resume": {"skip_verified": True},
    }
    prepare_preference(original)
    h100, a100 = variants(original)
    assert h100["requirements"]["gpu"] == "H100"
    assert a100["requirements"]["gpu"] == "A100"
    assert a100["env"] == h100["env"]
    assert a100["output_dir"] != h100["output_dir"]
    assert a100["job_id"] != h100["job_id"]
    assert a100["timeout_minutes"] == h100["timeout_minutes"] == 300
    assert a100["budget"]["unit_cap"] == 40 and h100["budget"]["unit_cap"] == 60


def test_real_entrypoint_cap_and_verified_stage_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.sandbox_a_resident.ct_h5000 import run
    from research.sandbox_a_resident.ct_h5000.preflight import tiny_entrypoint
    from tools.sandbox.registry import digest

    normal, capped = tmp_path / "tiny", tmp_path / "capped"
    normal_report = tiny_entrypoint(normal)
    cap_report = tiny_entrypoint(capped, capped=True)
    assert normal_report["verdict"] == "INCOMPLETE_SUPPORT"
    assert normal_report["completed_ticks"] == 24
    assert cap_report["verdict"] == "INCOMPLETE_COLLECTION"
    assert not (capped / "exports/stage-0001.pt").exists()
    cap_summary = json.loads((capped / "exports/summary.json").read_text())
    assert cap_summary["support_report"]["status"] == "NOT_EVALUATED_INCOMPLETE_COLLECTION"
    assert (capped / "exports/summary.json.complete").is_file()
    ctrl = CTControl.from_dict(json.loads((normal / "control.json").read_text()))
    ledger = empty_ledger(ctrl.to_dict())
    header = json.loads((normal / "exports/stage-0001.json").read_text())
    info = normal / "exports/stage-0001.json"
    ledger["units"] = [
        {
            "id": header["id"],
            "stage": 1,
            "exports": [
                header["snapshot"],
                {"path": str(info), "size": info.stat().st_size, "sha256": digest(info)},
            ],
        }
    ]
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(ledger))

    def refuse_collection(*args: object, **kwargs: object) -> None:
        raise AssertionError("verified complete collection was recomputed")

    monkeypatch.setattr(run, "collect_ct_dataset", refuse_collection)
    code, summary = run.run_ct_experiment(
        ctrl,
        b8_checkpoint=normal / "b8.pth",
        frp3_checkpoint=normal / "frp3.pth",
        champion_checkpoint=normal / "champion.pth",
        output=tmp_path / "resumed",
        device="cpu",
        ledger_path=path,
    )
    assert code == 0 and summary["resumed_stages"] == [1]
    assert summary["verdict"] == normal_report["verdict"]
    assert not (tmp_path / "resumed/stage-0001.pt").exists()
