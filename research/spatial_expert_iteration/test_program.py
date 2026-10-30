"""Tiny scientific-integrity, unbiased-query, fan-out and learner-resume fixtures."""

from __future__ import annotations

import copy
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from research.sandbox_a_resident.ct_run import B8_S3_SHA256
from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256
from research.spatial_student_pilot.model import SpatialStudent
from research.spatial_student_pilot.resume import UnitStore as PilotUnitStore
from research.spatial_student_pilot.resume import (
    build_ledger,
    empty_ledger,
)

from .config import RoundConfig, identity
from .data import (
    check_support_possible,
    collect,
    dataset,
    query_gate,
    select_anchors,
    validate_rows,
)
from .fit import fit, initial_state, row_loss
from .prepare import freeze, seed_audit, verify_packet, write_json
from .run import save_tree
from .search import Job, label_key, label_roots, rollouts, tensor_identity
from .units import HardwareUnitStore


def UnitStore(*args: Any, **kwargs: Any) -> HardwareUnitStore:
    return HardwareUnitStore(PilotUnitStore(*args, **kwargs), "cpu")


class FixedPolicy(nn.Module):
    def __init__(self, action: int = 1) -> None:
        super().__init__()
        self.action = action

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((args[0].shape[0], 6), device=args[0].device)
        out[:, self.action] = 1
        return out


@pytest.fixture(scope="module")
def fixture() -> tuple:
    torch.set_num_threads(1)
    cfg = RoundConfig.smoke()
    b8 = FixedPolicy()
    nets = {CHAMPION_SHA256: b8, FRP3_S12_SHA256: b8}
    kwargs = dict(
        student=FixedPolicy(2), student_sha="student-a", b8=b8, networks=nets, device="cpu"
    )
    train = collect(cfg, list(range(cfg.worlds)), **kwargs)
    held = collect(cfg, list(range(cfg.assessment_worlds)), partition="assessment", **kwargs)
    data = dataset(cfg, [train, held], [])
    return cfg, b8, nets, data, [train, held]


def test_student_drives_and_natural_ancestry_stays_frozen(fixture: tuple) -> None:
    cfg, b8, nets, data, batches = fixture
    assert {r["collector_sha256"] for r in data["train"]} == {"student-a"}
    assert all(r["student_action"] == 2 for r in data["train"])
    assert all(r["b8_action"] == 1 for r in data["train"])
    assert all(r["b8_q"].shape == (6,) for r in data["demos"])
    assert not {r["parent_id"] for r in data["train"]} & {
        r["parent_id"] for r in data["assessment"]
    }
    anchor = data["reservoir"][:1]
    continued = collect(
        cfg,
        [0],
        student=FixedPolicy(1),
        student_sha="student-b",
        b8=b8,
        networks=nets,
        device="cpu",
        anchors=anchor,
    )
    assert all(r["parent_id"] == anchor[0]["parent_id"] for r in continued["roots"])
    assert all(r["ancestor_uid"] == anchor[0]["uid"] for r in continued["roots"])
    assert all(r["collector_sha256"] == "student-b" for r in continued["roots"])
    corrupt = copy.deepcopy(anchor)
    corrupt[0]["parent_partition"] = "assessment"
    with pytest.raises(ValueError, match="ancestry"):
        validate_rows(corrupt, "train")
    with pytest.raises(ValueError, match="population"):
        dataset(cfg, batches[:1], [])
    with pytest.raises(RuntimeError, match="mature"):
        dataset(replace(cfg, min_mature_parents=10), batches, [])
    with pytest.raises(ValueError, match="wholly training"):
        collect(
            cfg,
            [0],
            student=FixedPolicy(),
            student_sha="b",
            b8=b8,
            networks=nets,
            device="cpu",
            partition="assessment",
            anchors=anchor,
        )


def test_query_gate_records_probability_without_self_labels(fixture: tuple) -> None:
    cfg, _, _, data, _ = fixture
    cfg = replace(cfg, query_ranking="disagreement", query_top_fraction=0.25, query_floor=0.125)
    rows = [dict(data["train"][0], uid=f"root-{i}") for i in range(1000)]
    audit = query_gate(rows, cfg)
    assert audit == query_gate(list(reversed(rows)), cfg)[::-1]
    assert sum(r["inclusion_probability"] == 1 for r in audit) == 250
    assert 300 < sum(r["queried"] for r in audit) < 390
    assert all("target_action" not in r for r in audit)
    with pytest.raises(ValueError, match="floor"):
        replace(cfg, query_floor=0)


def test_ipw_recovers_population_gradient_exactly() -> None:
    # Exhaust all independent query outcomes for a population of three; compare
    # the actual uniform-queried minibatch expectation with full-population CE.
    probs = [1.0, 0.25, 0.5]
    logits = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]], requires_grad=True)
    losses = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]), reduction="none")
    full_grad = torch.autograd.grad(losses.mean(), logits, retain_graph=True)[0]
    expected = torch.zeros_like(logits)
    for bits in itertools.product((False, True), repeat=2):
        selected = [True, *bits]
        chance = np.prod([p if s else 1 - p for p, s in zip(probs, selected)])
        ids = [i for i, s in enumerate(selected) if s]
        # Uniform minibatch over n_query rows, weighted n_query/(N*p).
        estimate = sum(losses[i] * len(ids) / (3 * probs[i]) for i in ids) / len(ids)
        expected += chance * torch.autograd.grad(estimate, logits, retain_graph=True)[0]
    torch.testing.assert_close(expected, full_grad)


def test_restore_once_fanout_matches_reference_with_carry_and_permutations(fixture: tuple) -> None:
    cfg, b8, nets, data, batches = fixture
    roots = batches[0]["roots"]  # cover scripted, frozen and mixed/random_safe carry
    student = FixedPolicy(2)
    jobs = [
        Job(i, int(a), k)
        for i, root in enumerate(roots)
        for a in np.flatnonzero(root["mask"])
        for k in (0, 1)
    ]
    kwargs = dict(
        cfg=replace(cfg, branch_horizon=5, branch_batch_size=7),
        student=student,
        b8=b8,
        networks=nets,
        device="cpu",
    )
    fast_trace, reference_trace = [], []
    fast = rollouts(roots, jobs, trace=fast_trace, **kwargs)
    reference = rollouts(roots, jobs, fanout=False, trace=reference_trace, **kwargs)
    np.testing.assert_array_equal(fast[0], reference[0])
    np.testing.assert_array_equal(fast[1], reference[1])
    assert fast[2] == reference[2]
    assert fast_trace == reference_trace  # full simulator bytes and opponent carry/RNG
    reversed_trace = []
    permuted = rollouts(roots, jobs[::-1], trace=reversed_trace, **kwargs)
    np.testing.assert_array_equal(permuted[0][::-1], fast[0])
    assert reversed_trace[::-1] == fast_trace
    alternate = rollouts(roots, jobs, **{**kwargs, "student": FixedPolicy(0)})
    assert not np.array_equal(fast[0], alternate[0]) or fast[2] != alternate[2]
    assert label_key(roots[0], cfg, "a", "opp") != label_key(roots[0], cfg, "b", "opp")
    assert label_key(roots[0], cfg, "a", "opp") != label_key(roots[0], cfg, "a", "opp2")


def test_soft_targets_masking_and_centering(fixture: tuple) -> None:
    cfg, _, _, data, _ = fixture
    cfg = replace(cfg, soft_weight=0.3, advantage_weight=0.2)
    root = dict(
        data["train"][0],
        target_action=1,
        mask=torch.tensor([True, True, False, False, False, False]),
        b8_action=1,
        teacher_q=torch.tensor([1e3, -1e3, float("nan"), 0, 0, 0]),
    )
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], requires_grad=True)
    loss = row_loss(logits, [root], cfg, searched=True).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[0, 2:], torch.zeros(4))
    shifted = dict(root, teacher_q=root["teacher_q"] + 100)
    torch.testing.assert_close(loss, row_loss(logits, [shifted], cfg, searched=True).sum())


def test_warm_learner_resume_and_next_round_preserve_full_state(
    tmp_path: Path, fixture: tuple
) -> None:
    cfg, b8, nets, data, _ = fixture
    torch.manual_seed(cfg.seed)
    initial = initial_state(SpatialStudent(), cfg.seed)
    labels = label_roots(
        [dict(r, queried=True, inclusion_probability=1.0) for r in data["train"]],
        cfg=cfg,
        student=FixedPolicy(),
        continuation_sha="a",
        opponents_sha="o",
        b8=b8,
        networks=nets,
        device="cpu",
    )
    full = UnitStore(tmp_path / "full", cfg, empty_ledger(cfg), tmp_path)
    kwargs = dict(initial=initial, population=cfg.candidate_roots, device="cpu")
    _, final = fit(cfg, data["demos"], labels, store=full, **kwargs)
    partial = UnitStore(tmp_path / "partial", cfg, empty_ledger(cfg), tmp_path)
    counter = {"updates": 0}
    with pytest.raises(InterruptedError):
        fit(
            cfg,
            data["demos"],
            labels,
            store=partial,
            stop=lambda: counter["updates"] == 1,
            progress=lambda x: counter.update(x),
            **kwargs,
        )
    ledger = build_ledger(cfg, [partial.output])
    resumed = UnitStore(tmp_path / "resumed", cfg, ledger, partial.output)
    _, actual = fit(cfg, data["demos"], labels, store=resumed, **kwargs)
    assert tensor_identity(final) == tensor_identity(actual)
    assert final["global_update"] == 2
    cfg2 = replace(cfg, round_index=2, updates=1)
    following = UnitStore(tmp_path / "next", cfg2, empty_ledger(cfg2), tmp_path)
    _, next_final = fit(
        cfg2,
        data["demos"],
        labels,
        store=following,
        initial=final,
        population=cfg.candidate_roots,
        device="cpu",
    )
    assert next_final["global_update"] == 3
    assert all(float(s["step"]) == 3 for s in next_final["optimizer"]["state"].values())
    corrupt = [dict(labels[0], queried=False)]
    with pytest.raises(ValueError, match="unqueried"):
        fit(
            cfg2,
            data["demos"],
            corrupt,
            store=following,
            initial=final,
            population=cfg.candidate_roots,
            device="cpu",
        )
    with pytest.raises(ValueError, match="ancestry"):
        fit(cfg, data["assessment"], labels, store=full, **kwargs)


def test_packet_freeze_refuses_drift_and_seed_overlap(tmp_path: Path) -> None:
    cfg = RoundConfig.smoke()
    inputs = tmp_path / "in"
    inputs.mkdir()
    save_tree(inputs / "learner.pt", initial_state(SpatialStudent(), cfg.seed))
    save_tree(inputs / "reservoir.pt", dict(schema="spatial-exit-reservoir/v1", roots=[]))
    write_json(inputs / "registration.json", dict(status="SMOKE_ONLY"))
    write_json(inputs / "inventory.json", dict(excluded_seeds=[]))
    packet = tmp_path / "packet"
    freeze(
        packet,
        cfg,
        inputs={name: inputs / name for name in ("learner.pt", "reservoir.pt")},
        registration=inputs / "registration.json",
        inventory=inputs / "inventory.json",
    )
    assert verify_packet(packet)[0] == cfg
    (packet / "config.json").write_text(
        (packet / "config.json").read_text().replace('"updates": 2', '"updates": 3')
    )
    with pytest.raises(ValueError, match="config drift"):
        verify_packet(packet)
    with pytest.raises(ValueError, match="collision"):
        seed_audit(cfg, dict(excluded_seeds=[cfg.world_seed(0)]))
    with pytest.raises(ValueError, match="fresh absolute"):
        freeze(
            packet,
            cfg,
            inputs={},
            registration=inputs / "registration.json",
            inventory=inputs / "inventory.json",
        )
    real = RoundConfig()
    with pytest.raises(ValueError, match="ratification"):
        freeze(
            tmp_path / "real",
            real,
            inputs={},
            registration=inputs / "registration.json",
            inventory=inputs / "inventory.json",
        )


def test_round_units_reject_foreign_allocation_and_corruption(tmp_path: Path) -> None:
    cfg = RoundConfig.smoke()
    store = UnitStore(tmp_path / "a", cfg, empty_ledger(cfg), tmp_path)
    store.publish("collect-0000", {"worlds": [0, 1]})
    ledger = build_ledger(cfg, [store.output])
    with pytest.raises(ValueError, match="foreign ledger"):
        UnitStore(tmp_path / "b", replace(cfg, round_index=2), ledger, store.output)
    with pytest.raises(ValueError, match="foreign ledger"):
        UnitStore(
            tmp_path / "different-code",
            replace(cfg, implementation_sha256="a" * 64),
            ledger,
            store.output,
        )
    with pytest.raises(ValueError, match="recomputed"):
        store.publish("collect-0000", {})
    with pytest.raises(ValueError, match="ineligible"):
        store.publish("collect-0001", {})
    entry = ledger["units"][0]
    (store.output / entry["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size"):
        UnitStore(tmp_path / "c", cfg, ledger, store.output)


def test_integer_cache_byte_identity_and_gradients(fixture: tuple) -> None:
    from research.spatial_student_pilot.observation import build_observation, compact_cat

    from .cache import IntegerRasterCache

    _, _, _, data, _ = fixture
    rows = copy.deepcopy(data["demos"][:2])
    # Repeated cells, long ordered bodies, growth debt, overlap and dead slots.
    long_body = copy.deepcopy(rows[0])
    compact = long_body["compact"]
    compact["body_cells"] = torch.zeros((1, 6, 997, 2), dtype=torch.int16)
    compact["body_count"].fill_(997)
    compact["length"].fill_(1024)
    compact["alive"][0, -1] = False
    compact["heads"].zero_()
    long_body["uid"] += "/adversarial-long-body"
    rows.append(long_body)
    cache = IntegerRasterCache(rows, "cpu", 1000000)
    expected = build_observation(compact_cat([r["compact"] for r in rows]))
    actual = cache.batch(rows)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    actor = SpatialStudent()
    actor(*expected).sum().backward()
    grads = [p.grad.clone() for p in actor.parameters()]
    actor.zero_grad()
    actor(*actual).sum().backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(grads, actor.parameters()))
    with pytest.raises(ValueError, match="byte cap"):
        IntegerRasterCache(rows, "cpu", 1)
    with pytest.raises(ValueError, match="held-out"):
        IntegerRasterCache(data["assessment"], "cpu", 1000000)


def test_pilot_import_preserves_optimizer_rng_and_checks_nomination(fixture: tuple) -> None:
    from .import_pilot import convert

    cfg, _, _, _, _ = fixture
    initial = initial_state(SpatialStudent(), cfg.seed)
    pilot = dict(initial, update=8000, final=True)
    converted = convert(pilot, pilot["model"], 8000)
    assert converted["global_update"] == 8000
    for name in ("model", "optimizer", "generator", "torch_rng", "cuda_rng"):
        assert tensor_identity(converted[name]) == tensor_identity(pilot[name])
    with pytest.raises(ValueError, match="identity"):
        convert(pilot, SpatialStudent().state_dict(), 8000)
    with pytest.raises(ValueError, match="identity"):
        convert(dict(pilot, final=False), pilot["model"], 8000)


def test_complete_round_entrypoint_core_and_verified_retry(tmp_path: Path, fixture: tuple) -> None:
    from research.spatial_student_pilot.run import Control

    from .run import round_pipeline

    cfg, b8, nets, _, _ = fixture
    torch.manual_seed(cfg.seed)
    initial = initial_state(SpatialStudent(), cfg.seed)
    frozen = dict(continuation_sha256=B8_S3_SHA256, opponents_sha256="fixture")
    first = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    control = Control(first.output, 60)
    try:
        result = round_pipeline(
            cfg,
            first,
            initial=initial,
            reservoir=[],
            frozen=frozen,
            b8=b8,
            networks=nets,
            device="cpu",
            control=control,
        )
    finally:
        control.close()
    assert result["status"] == "SMOKE_COMPLETE_NO_SCIENTIFIC_RESULT"
    assert len(result["units"]) == 6  # collection, search, 2 fits, assessment, gameplay
    ledger = build_ledger(cfg, [first.output])
    retry = UnitStore(tmp_path / "retry", cfg, ledger, first.output)
    control = Control(retry.output, 60)
    try:
        resumed = round_pipeline(
            cfg,
            retry,
            initial=initial,
            reservoir=[],
            frozen=frozen,
            b8=b8,
            networks=nets,
            device="cpu",
            control=control,
        )
    finally:
        control.close()
    assert resumed["candidate_sha256"] == result["candidate_sha256"]
    assert resumed["learner_sha256"] == result["learner_sha256"]
    assert not list((retry.output / "units").glob("*.pt.gz"))


def test_next_round_anchor_program_keeps_original_reservoir_immutable(
    tmp_path: Path, fixture: tuple
) -> None:
    from research.spatial_student_pilot.run import Control

    from .run import round_pipeline

    cfg, b8, nets, data, _ = fixture
    reservoir = copy.deepcopy(data["reservoir"])
    before = tensor_identity(reservoir)
    cfg = replace(cfg, round_index=2, anchor_worlds=2, reservoir_roots=1, candidate_roots=4)
    initial = initial_state(SpatialStudent(), cfg.seed)
    initial.update(round_index=1, global_update=cfg.updates)
    store = UnitStore(tmp_path / "round2", cfg, empty_ledger(cfg), tmp_path)
    control = Control(store.output, 60)
    try:
        result = round_pipeline(
            cfg,
            store,
            initial=initial,
            reservoir=reservoir,
            frozen=dict(continuation_sha256=B8_S3_SHA256, opponents_sha256="fixture"),
            b8=b8,
            networks=nets,
            device="cpu",
            control=control,
        )
    finally:
        control.close()
    assert result["status"] == "SMOKE_COMPLETE_NO_SCIENTIFIC_RESULT"
    assert tensor_identity(reservoir) == before
    emitted = torch.load(store.output / "reservoir.pt", weights_only=False)["roots"]
    by_uid = {r["uid"]: r for r in emitted}
    assert all(tensor_identity(r) == tensor_identity(by_uid[r["uid"]]) for r in reservoir)
    held_parents = {r["parent_id"] for r in store.get("collect-0000")[1]["roots"]}
    assert not held_parents & {r["parent_id"] for r in emitted}
    final = store.get(f"fit-{cfg.updates:07d}")
    assert final["global_update"] == 2 * cfg.updates


def test_registered_freeze_go_source_and_input_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hermetic admission fixture only. No fabricated decision ever reaches the
    # simulator: exercise the registered packet path without creating science.
    from . import prepare

    cfg = replace(
        RoundConfig.smoke(), tiny=False, search_chunk=512, implementation_sha256=identity({})
    )
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    state = initial_state(SpatialStudent(), cfg.seed)
    save_tree(inputs_dir / "learner.pt", state)
    save_tree(inputs_dir / "reservoir.pt", dict(schema="spatial-exit-reservoir/v1", roots=[]))
    save_tree(inputs_dir / "initial-actor.pth", dict(model_state_dict=state["model"]))
    write_json(
        inputs_dir / "pilot-go.json",
        dict(
            schema="spatial-student-readout/v1",
            verdict="GO",
            clauses={"fixture": True},
            candidate_sha256=prepare.digest(inputs_dir / "initial-actor.pth"),
        ),
    )
    for name in ("b8-s3.pth", "champion.pth", "frp3-s12.pth"):
        (inputs_dir / name).write_bytes(b"admission-only canonical-checkpoint fixture")
    checkpoint_sha = prepare.digest(inputs_dir / "b8-s3.pth")
    monkeypatch.setattr(prepare, "B8_S3_SHA256", checkpoint_sha)
    monkeypatch.setattr(
        prepare, "OPPONENT_SHAS", {"champion.pth": checkpoint_sha, "frp3-s12.pth": checkpoint_sha}
    )
    write_json(
        inputs_dir / "inventory.json",
        dict(
            excluded_seeds=[],
            scope="mac-historical-and-prior-rounds",
            prior_round_freezes=["fixture"],
        ),
    )
    paths = {
        name: p
        for name, p in ((p.name, p) for p in inputs_dir.iterdir())
        if name != "inventory.json"
    }
    registration = tmp_path / "registration.json"
    reg = dict(
        status="RATIFIED",
        independent_judge="admission-fixture",
        config_sha256=cfg.sha256,
        decision_rule="fixture-only",
        heldout_policy="sealed",
        round_schedule=[1, 2, 3, 4, 5],
        stopping={"kind": "round-cap", "max_rounds": 5},
        release_authority="owner-only",
        input_sha256={name: prepare.digest(p) for name, p in paths.items()},
        seed_inventory_sha256=prepare.digest(inputs_dir / "inventory.json"),
    )
    write_json(registration, reg)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "sources", lambda *a, **k: {})
    monkeypatch.setattr(prepare, "audit_ancestors", lambda *a: {"nodes": []})
    monkeypatch.setattr(
        prepare.subprocess,
        "check_output",
        lambda args, **kwargs: "a" * 40 if kwargs.get("text") else registration.read_bytes(),
    )
    packet = tmp_path / "packet"
    frozen = freeze(
        packet,
        cfg,
        inputs=paths,
        registration=registration,
        inventory=inputs_dir / "inventory.json",
    )
    assert frozen["tiny"] is False
    assert frozen["dispatchable"] is False
    assert frozen["budgets"]["reservoir_max_roots"] == cfg.reservoir_max_roots
    assert verify_packet(packet)[0] == cfg
    (inputs_dir / "pilot-go.json").write_text('{"verdict":"KILL"}')
    with pytest.raises(ValueError, match="pin every round input"):
        freeze(
            tmp_path / "reject",
            cfg,
            inputs=paths,
            registration=registration,
            inventory=inputs_dir / "inventory.json",
        )
    with pytest.raises(ValueError, match="integer effective seed"):
        seed_audit(cfg, dict(excluded_seeds=[str(cfg.world_seed(0))]))


def test_registered_chunks_pool_512_roots_and_keep_assessment_together() -> None:
    cfg = RoundConfig()
    assert cfg.search_chunk == 512
    assert cfg.assessment_roots <= cfg.search_chunk
    with pytest.raises(ValueError, match="at least 512"):
        replace(cfg, search_chunk=128)
    # Four legal actions x two suffixes at a deterministic ~34% query rate.
    # Both old and new populations have the same row/action/suffix order.
    selected = [i for i in range(512) if i % 3 == 0]
    jobs = [Job(i, a, s) for i in selected for a in range(4) for s in (0, 1)]
    slabs = [
        len(jobs[i : i + cfg.branch_batch_size]) for i in range(0, len(jobs), cfg.branch_batch_size)
    ]
    assert slabs == [1024, 344]
    assert max(slabs) == cfg.branch_batch_size  # old 128-root chunks never filled one


def test_anchor_hash_sample_without_replacement_and_order_independent(fixture: tuple) -> None:
    cfg, _, _, data, _ = fixture
    roots = [dict(data["train"][0], uid=f"root-{i}") for i in range(50)]
    cfg = replace(cfg, anchor_worlds=4)
    anchors = select_anchors(cfg, roots)
    assert len({r["uid"] for r in anchors}) == 4
    assert [r["uid"] for r in anchors] == [r["uid"] for r in select_anchors(cfg, roots[::-1])]
    assert [r["uid"] for r in anchors] != [r["uid"] for r in roots[:4]]
    assert [r["uid"] for r in anchors] != [
        r["uid"] for r in select_anchors(replace(cfg, round_index=2), roots)
    ]
    with pytest.raises(ValueError, match="distinct reservoir"):
        select_anchors(cfg, roots[:2])


def test_reused_and_anchor_maturity_cannot_satisfy_fresh_support(fixture: tuple) -> None:
    cfg, _, _, data, batches = fixture
    cfg = replace(cfg, reservoir_roots=1, mature_length=10)
    old = [dict(data["train"][0], uid="old-mature", hero_length=20)]
    fresh = copy.deepcopy(batches)
    for batch in fresh:
        for root in batch["roots"]:
            root["hero_length"] = 1 if batch["partition"] == "train" else 20
    with pytest.raises(RuntimeError, match="mature"):
        dataset(cfg, fresh, old)
    for root in fresh[0]["roots"]:
        root.update(hero_length=20, start_distribution="anchor")
    with pytest.raises(RuntimeError, match="mature"):
        dataset(cfg, fresh, old)
    for root in fresh[0]["roots"]:
        root["start_distribution"] = "fresh"
    assert len(dataset(cfg, fresh, old)["train"]) == cfg.candidate_roots


def test_reservoir_forecast_and_exact_guard_precede_round_work(fixture: tuple) -> None:
    from .reservoir import capacity_forecast, reservoir_value, serialized_size

    cfg, _, _, data, _ = fixture
    roots = data["reservoir"]
    before = tensor_identity(roots)
    forecast = capacity_forecast(cfg, roots)
    assert forecast["maximum_roots"] == len(roots) + cfg.candidate_roots
    assert forecast["forecast_bytes"] > serialized_size(roots)
    with pytest.raises(ValueError, match="pre-data.*root cap"):
        capacity_forecast(replace(cfg, reservoir_max_roots=len(roots)), roots)
    with pytest.raises(ValueError, match="pre-data.*byte forecast"):
        capacity_forecast(replace(cfg, reservoir_max_bytes=1), [])
    new = [dict(roots[0], uid="new")]
    value = reservoir_value(cfg, roots, new)
    with pytest.raises(ValueError, match="byte cap.*before search"):
        reservoir_value(replace(cfg, reservoir_max_bytes=serialized_size(value) - 1), roots, new)
    assert tensor_identity(roots) == before
    assert len(value["roots"]) == len(roots) + 1
    assert tensor_identity(reservoir_value(cfg, roots, roots)["roots"]) == before


def test_freeze_refuses_capacity_before_creating_packet(tmp_path: Path) -> None:
    cfg = replace(RoundConfig.smoke(), reservoir_max_roots=1)
    save_tree(tmp_path / "learner.pt", initial_state(SpatialStudent(), cfg.seed))
    save_tree(tmp_path / "reservoir.pt", dict(schema="spatial-exit-reservoir/v1", roots=[]))
    write_json(tmp_path / "reg.json", dict(status="SMOKE_ONLY"))
    write_json(tmp_path / "inventory.json", dict(excluded_seeds=[]))
    packet = tmp_path / "packet"
    with pytest.raises(ValueError, match="pre-data.*root cap"):
        freeze(
            packet,
            cfg,
            inputs={n: tmp_path / n for n in ("learner.pt", "reservoir.pt")},
            registration=tmp_path / "reg.json",
            inventory=tmp_path / "inventory.json",
        )
    assert not packet.exists()


def test_early_support_refuses_only_impossible_remaining_counts(fixture: tuple) -> None:
    cfg, _, _, _, batches = fixture
    check_support_possible(cfg, [])  # every world is still available
    check_support_possible(cfg, batches)
    partial = copy.deepcopy(batches)
    partial[0]["world_ids"] = [0, 1]
    partial[0]["roots"] = []
    partial[0]["demos"] = []
    check_support_possible(cfg, partial)  # two remaining worlds can still supply all counts
    partial[0]["world_ids"] = list(range(cfg.worlds))
    with pytest.raises(RuntimeError, match="INCOMPLETE_SUPPORT.*after collection batch"):
        check_support_possible(cfg, partial)


def test_cache_uid_lookup_has_no_repeated_sha_and_refuses_conflicts(
    fixture: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    from . import cache as cache_module

    _, _, _, data, _ = fixture
    rows = copy.deepcopy(data["demos"][:2])
    cache = cache_module.IntegerRasterCache(rows, "cpu", 1000000)
    monkeypatch.setattr(cache_module, "tensor_identity", lambda _: pytest.fail("hot-path SHA"))
    assert cache.batch(rows)[0].shape[0] == 2
    monkeypatch.undo()
    conflicting = copy.deepcopy(rows[0])
    conflicting["compact"]["length"] += 1
    with pytest.raises(ValueError, match="conflicting compact"):
        cache_module.IntegerRasterCache([rows[0], conflicting], "cpu", 1000000)


def test_soft_temperatures_are_separate_by_teacher(fixture: tuple) -> None:
    cfg, _, _, data, _ = fixture
    cfg = replace(cfg, soft_weight=0.5)
    row = dict(
        data["train"][0], target_action=1, teacher_q=torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
    logits = torch.tensor([[0.0, 2.0, 1.0, 3.0, 4.0, 5.0]])
    searched = row_loss(logits, [row], cfg, searched=True)
    ordinary = row_loss(logits, [row], cfg, searched=False)
    changed_b8 = replace(cfg, b8_temperature=10.0)
    changed_search = replace(cfg, search_temperature=10.0)
    assert torch.equal(searched, row_loss(logits, [row], changed_b8, searched=True))
    assert torch.equal(ordinary, row_loss(logits, [row], changed_search, searched=False))
    assert not torch.equal(searched, row_loss(logits, [row], changed_search, searched=True))
    assert not torch.equal(ordinary, row_loss(logits, [row], changed_b8, searched=False))


def test_pilot_reservoir_import_uses_registered_mature_length(fixture: tuple) -> None:
    from research.spatial_student_pilot.config import PilotConfig

    from .import_pilot import training_roots

    _, _, _, data, _ = fixture
    roots = [
        dict(data["train"][0], uid=f"r-{i}", hero_length=n)
        for i, n in enumerate((599, 600, 900, 901))
    ]

    class Store:
        def list_ids(self, prefix: str) -> list[str]:
            return ["collect-0000"]

        def get(self, key: str) -> dict:
            return dict(roots=roots + [dict(roots[-1], partition="assessment")])

    imported = training_roots(Store(), PilotConfig(), 900)
    assert [r["hero_length"] for r in imported] == [900, 901]
