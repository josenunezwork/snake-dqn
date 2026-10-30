"""Regression tests for r50 follow-on integrity and learning controls."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.prepare import digest
from research.spatial_student_pilot.resume import UnitStore, empty_ledger

from . import test_program
from .config import RoundConfig, identity
from .data import query_gate, stratified_roots
from .fit import row_loss
from .guards import require_training_scope, teacher_audit
from .prepare import seed_audit
from .provenance import audit_ancestors
from .search import Job, label_roots, rollouts
from .test_program import FixedPolicy
from .units import HardwareUnitStore

fixture = test_program.fixture


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def ref(path: Path, **kwargs: object) -> dict:
    return dict(path=str(path.resolve()), sha256=digest(path), **kwargs)


def pilot_node(root: Path) -> tuple[dict, list[int]]:
    root.mkdir()
    cfg = PilotConfig()
    config = root / "config.json"
    dump(config, cfg.to_dict())
    manifest = root / "manifest.json"
    dump(manifest, dict(inputs=[dict(dest="config.json", sha256=digest(config))]))
    keys = [
        fn(i, phase)
        for phase, n in (("collection", cfg.worlds), ("gameplay", cfg.gameplay_worlds))
        for i in range(n)
        for fn in (cfg.world_seed, cfg.opp_seed)
    ]
    return ref(manifest, kind="pilot", inputs={"config.json": ref(config)}), keys


def round_node(root: Path, parent: dict, parent_keys: list[int]) -> tuple[dict, list[int]]:
    from .prepare import seed_keys

    root.mkdir()
    cfg = replace(RoundConfig(), implementation_sha256=identity({}))
    dump(root / "config.json", cfg.to_dict())
    inventory = dict(
        scope="mac-historical-and-prior-rounds",
        excluded_seeds=parent_keys,
        prior_round_freezes=[parent],
    )
    dump(root / "seed-inventory.json", inventory)
    dump(root / "seed-audit.json", seed_audit(cfg, inventory))
    dump(root / "verified-ledger.json", empty_ledger(cfg))
    (root / "learner.pt").write_bytes(b"hash-only fixture")
    frozen = dict(
        schema="spatial-exit-freeze/v1",
        tiny=False,
        round_index=1,
        namespace=cfg.namespace,
        allocation_index=cfg.allocation_index,
        config_sha256=cfg.sha256,
        source_sha256={},
        inputs={name: digest(root / name) for name in ("seed-inventory.json", "learner.pt")},
        ledger_sha256=digest(root / "verified-ledger.json"),
        seed_audit_sha256=digest(root / "seed-audit.json"),
    )
    dump(root / "FREEZE.json", frozen)
    return ref(root / "FREEZE.json", kind="round"), seed_keys(cfg)


def test_uniform_default_and_opt_in_disagreement(fixture: tuple) -> None:
    cfg, _, _, data, _ = fixture
    default = RoundConfig()
    assert default.query_top_fraction == 0 and default.query_floor == 0.34375
    rows = [dict(data["train"][0], uid=f"r{i}", student_action=i % 6) for i in range(1000)]
    uniform = query_gate(rows, default)
    assert {r["inclusion_probability"] for r in uniform} == {0.34375}
    assert sum(r["disagreement_top_quartile"] for r in uniform) == 250
    ranked = query_gate(
        rows, replace(default, query_ranking="disagreement", query_top_fraction=0.25)
    )
    assert sum(r["inclusion_probability"] == 1 for r in ranked) == 250
    with pytest.raises(ValueError, match="uniform"):
        replace(default, query_top_fraction=0.25)


def test_stratification_protects_late_roots_and_parent_coverage() -> None:
    cfg = replace(RoundConfig.smoke(), roots_per_parent=2)
    rows = [
        dict(
            uid=f"{p}-{k}",
            parent_id=p,
            priority=priority,
            kind=k,
            start_distribution="fresh",
            hero_length=600 if k == "late" else 5,
        )
        for p in ("a", "b", "c", "d")
        for k, priority in (("late", 100), ("ordinary", 0))
    ]
    selected = stratified_roots(rows, 8, cfg, 4, 4)
    assert len([r for r in selected if r["kind"] == "late"]) == 4
    assert len({r["parent_id"] for r in selected if r["kind"] == "late"}) == 4
    assert stratified_roots(rows[::-1], 8, cfg, 4, 4) == selected
    with pytest.raises(RuntimeError, match="mature parents"):
        stratified_roots([r for r in rows if r["parent_id"] != "d"], 8, cfg, 4, 4)


def test_provenance_resolves_all_rounds_and_detects_mutation(tmp_path: Path) -> None:
    pilot, pkeys = pilot_node(tmp_path / "pilot")
    first, keys = round_node(tmp_path / "round1", pilot, pkeys)
    cfg = replace(RoundConfig(), round_index=2)
    inventory = dict(excluded_seeds=pkeys + keys, prior_round_freezes=[first, pilot])
    result = audit_ancestors(cfg, inventory)
    assert len(result["nodes"]) == 2 and result["prior_rounds"] == {1: first["sha256"]}
    (tmp_path / "round1" / "learner.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_ancestors(cfg, inventory)


def test_provenance_refuses_truthiness_missing_seeds_foreign_and_missing_rounds(
    tmp_path: Path,
) -> None:
    cfg = RoundConfig()
    with pytest.raises(ValueError, match="path/sha256"):
        audit_ancestors(cfg, dict(prior_round_freezes=["truthy"]))
    pilot, keys = pilot_node(tmp_path / "pilot")
    with pytest.raises(ValueError, match="effective seeds"):
        audit_ancestors(cfg, dict(prior_round_freezes=[pilot], excluded_seeds=[]))
    with pytest.raises(ValueError, match="every preceding round"):
        audit_ancestors(
            replace(cfg, round_index=2), dict(prior_round_freezes=[pilot], excluded_seeds=keys)
        )
    first, rkeys = round_node(tmp_path / "round1", pilot, keys)
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_ancestors(
            replace(cfg, program="foreign", round_index=2),
            dict(prior_round_freezes=[first], excluded_seeds=keys + rkeys),
        )
    changed = copy.deepcopy(pilot)
    changed["inputs"]["config.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_ancestors(cfg, dict(prior_round_freezes=[changed], excluded_seeds=keys))


def test_margin_soft_advantage_targets_and_illegal_gradients() -> None:
    cfg = RoundConfig.smoke()
    row = dict(
        mask=torch.tensor([True, True, False, False, False, False]),
        target_action=1,
        b8_action=0,
        top2_margin=0.0,
        action_advantages=np.array([0.0, 0.0, np.nan, np.nan, np.nan, np.nan]),
    )
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], requires_grad=True)
    base = row_loss(logits, [row], cfg, searched=True)
    weighted = row_loss(logits, [row], replace(cfg, margin_threshold=10), searched=True)
    assert base.item() > 0 and weighted.item() == 0
    soft = row_loss(
        logits, [row], replace(cfg, margin_threshold=10, soft_advantage_weight=1), searched=True
    )
    soft.sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad[0, 2:] == 0).all()
    assert cfg.margin_threshold == cfg.soft_advantage_weight == 0


def test_label_advantage_and_ce2_exports(fixture: tuple) -> None:
    cfg, b8, nets, data, _ = fixture
    root = dict(data["train"][0], gate_student_logits=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    labels = label_roots(
        [root],
        cfg=cfg,
        student=FixedPolicy(2),
        b8=b8,
        networks=nets,
        continuation_sha="b8",
        opponents_sha="opponents",
        device="cpu",
    )
    label = labels[0]
    mask = np.asarray(root["mask"], bool)
    expected = label["teacher_q"] - label["teacher_q"][root["b8_action"]]
    np.testing.assert_array_equal(label["action_advantages"][mask], expected[mask])
    assert label["gate_ce_squared"] == pytest.approx(label["gate_ce"] ** 2)
    assert label["quarantined"] is True and label["continuation_mode"] == "b8"


def test_b8_continuation_does_not_depend_on_student_after_first_action(fixture: tuple) -> None:
    cfg, b8, nets, data, _ = fixture
    cfg = replace(cfg, branch_horizon=8)
    root = data["train"][:1]
    jobs = [Job(0, int(root[0]["b8_action"]), 0)]
    kwargs = dict(cfg=cfg, b8=b8, networks=nets, device="cpu")
    a = rollouts(root, jobs, student=FixedPolicy(0), **kwargs)
    b = rollouts(root, jobs, student=FixedPolicy(2), **kwargs)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    with pytest.raises(ValueError, match="paired teacher audit hash"):
        replace(cfg, continuation_mode="student")


def test_teacher_audit_uses_parent_pairs_and_rejects_degradation() -> None:
    cfg = replace(
        RoundConfig.smoke(),
        continuation_mode="student",
        teacher_audit_sha256="a" * 64,
        teacher_audit_min_parents=2,
    )
    report = dict(
        schema="spatial-exit-teacher-audit/v1",
        independent_judge="independent",
        namespace=cfg.namespace,
        student_sha256="student",
        opponents_sha256="opp",
        branch_horizon=cfg.branch_horizon,
        selection_suffixes=[0, 1],
        scoring_suffixes=[4, 5],
        pairs=[
            dict(uid=f"r{i}", parent_id=f"p{i}", b8_teacher_gain=1.0, student_teacher_gain=2.0)
            for i in range(3)
        ],
    )
    assert teacher_audit(cfg, report, "student", "opp")["passed"]
    report["pairs"][0]["student_teacher_gain"] = -100
    with pytest.raises(ValueError, match="audit failed"):
        teacher_audit(cfg, report, "student", "opp")
    with pytest.raises(ValueError, match="identity mismatch"):
        teacher_audit(cfg, report, "other-student", "opp")


def test_quarantine_refuses_product_even_if_flag_is_disabled() -> None:
    cfg = RoundConfig.smoke()
    require_training_scope(cfg, [{"quarantined": True}], "research")
    with pytest.raises(ValueError, match="quarantined"):
        require_training_scope(cfg, [{"quarantined": True}], "product")
    with pytest.raises(ValueError, match="exact product replay admission"):
        require_training_scope(
            replace(cfg, quarantine_search_targets=False), [{"quarantined": False}], "product"
        )
    with pytest.raises(ValueError, match="quarantined"):
        require_training_scope(cfg, [], "product")
    with pytest.raises(ValueError, match="5-round cap"):
        replace(cfg, round_index=6)


def test_hardware_is_inside_each_hashed_unit_and_mixed_fit_resume_refused(tmp_path: Path) -> None:
    cfg = RoundConfig.smoke()
    raw = UnitStore(tmp_path / "units", cfg, empty_ledger(cfg), tmp_path)
    store = HardwareUnitStore(raw, "cpu")
    store.publish("collect-0000", [1])
    assert store.get("collect-0000") == [1]
    assert raw.get("collect-0000")["hardware"]["gpu_sku"] is None
    store.publish("fit-0000001", {"model": 1})
    store.runtime = dict(store.runtime, gpu_sku="different-GPU")
    with pytest.raises(ValueError, match="same hardware"):
        store.get("fit-0000001")


def test_student_continuation_requires_report_at_freeze(tmp_path: Path) -> None:
    from research.spatial_student_pilot.model import SpatialStudent

    from .fit import initial_state
    from .prepare import freeze
    from .run import save_tree

    cfg = replace(RoundConfig.smoke(), continuation_mode="student", teacher_audit_sha256="a" * 64)
    save_tree(tmp_path / "learner.pt", initial_state(SpatialStudent(), cfg.seed))
    save_tree(tmp_path / "reservoir.pt", dict(schema="spatial-exit-reservoir/v1", roots=[]))
    dump(tmp_path / "registration.json", dict(status="SMOKE_ONLY"))
    dump(tmp_path / "inventory.json", dict(excluded_seeds=[]))
    with pytest.raises(ValueError, match="pinned paired teacher audit"):
        freeze(
            tmp_path / "packet",
            cfg,
            inputs={n: tmp_path / n for n in ("learner.pt", "reservoir.pt")},
            registration=tmp_path / "registration.json",
            inventory=tmp_path / "inventory.json",
        )
    assert not (tmp_path / "packet").exists()


def test_ancestor_source_bytes_are_audited_against_git(tmp_path: Path) -> None:
    import subprocess

    pilot, keys = pilot_node(tmp_path / "pilot")
    first, rkeys = round_node(tmp_path / "round1", pilot, keys)
    root = tmp_path / "round1"
    source_hashes = {"research/spatial_expert_iteration/config.py": "0" * 64}
    prior = replace(RoundConfig(), implementation_sha256=identity(source_hashes))
    dump(root / "config.json", prior.to_dict())
    frozen = json.loads((root / "FREEZE.json").read_text())
    frozen.update(
        config_sha256=prior.sha256,
        source_sha256=source_hashes,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    dump(root / "FREEZE.json", frozen)
    first = ref(root / "FREEZE.json", kind="round")
    with pytest.raises(ValueError, match="source hash mismatch"):
        audit_ancestors(
            replace(RoundConfig(), round_index=2),
            dict(prior_round_freezes=[first], excluded_seeds=keys + rkeys),
        )


def test_gpu_sku_comes_from_cuda_device_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from .units import hardware

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "NVIDIA H100 test SKU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (9, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=80)
    )
    actual = hardware("cuda")
    assert actual["gpu_sku"] == "NVIDIA H100 test SKU"
    assert actual["capability"] == [9, 0]
