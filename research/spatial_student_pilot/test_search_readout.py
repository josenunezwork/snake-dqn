"""Falsification tests for selection/scoring isolation and the frozen readout."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import capture_world_snapshot, observe_hero_bv8
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.readout import clustered_lb90, evaluate_readout
from research.spatial_student_pilot.search import (
    _Job,
    _rollouts,
    assess_roots,
    label_roots,
    paired_gameplay,
    select_mass_actions,
)


class ZeroPolicy(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.zeros((args[0].shape[0], 6), device=args[0].device)


def fixture_root() -> tuple[dict, PilotConfig, nn.Module]:
    cfg = PilotConfig().development("preflight")
    b8 = ZeroPolicy()
    sim = CTDeviceSim(cfg.sim_config(1), device="cpu", train_mode=False, allow_respawn=True)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.fill_(1001)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    opponents = CTOpponentRuntime(
        sim, [training_rosters(cfg.worlds)[0]], b8_network=b8, checkpoint_networks={}
    )
    sim.prepare()
    *_, mask, _, action = observe_hero_bv8(sim, b8)
    snap = capture_world_snapshot(sim, opponents, 0, world_id=0, life_id=0, tick=0)
    return (
        dict(
            uid=cfg.namespace + "/a0/root0",
            world_id=0,
            partition="assessment",
            kind="late",
            mix="scripted",
            roster_index=0,
            snapshot=snap.to_state_dict(),
            b8_action=int(action[0]),
            mask=mask[0].cpu().tolist(),
        ),
        cfg,
        b8,
    )


def test_mass_only_selection_and_ties() -> None:
    mask = np.array([[1, 1, 1, 0, 0, 0]], bool)
    assert select_mass_actions(
        np.array([[2, 2, 1, np.nan, np.nan, np.nan]]), mask, [1]
    ).tolist() == [1]
    assert select_mass_actions(
        np.array([[2, 2, 1, np.nan, np.nan, np.nan]]), mask, [2]
    ).tolist() == [0]
    with pytest.raises(ValueError, match="nonfinite"):
        select_mass_actions(np.array([[2, np.nan, 1, 0, 0, 0]]), mask, [0])


def test_training_labels_cannot_access_scoring_suffixes() -> None:
    root, cfg, b8 = fixture_root()
    calls = []

    def rollout(roots, jobs, **kwargs):
        calls.extend(jobs)
        return np.array([float(j.action) for j in jobs]), np.ones(len(jobs)), 0, np.zeros(6, int)

    with patch("research.spatial_student_pilot.search._rollouts", side_effect=rollout):
        labels = label_roots([root], cfg=cfg, b8=b8, networks={}, device="cpu")
    assert {job.suffix for job in calls} == {0, 1}
    assert labels[0]["target_action"] == max(np.flatnonzero(root["mask"]))
    assert "scoring_mass" not in labels[0]


def test_actual_snapshot_search_assessment_and_gameplay() -> None:
    root, cfg, b8 = fixture_root()
    labels = label_roots([root], cfg=cfg, b8=b8, networks={}, device="cpu")
    assert labels[0]["target_action"] in np.flatnonzero(root["mask"])
    assessment = assess_roots([root], ZeroPolicy(), cfg=cfg, b8=b8, networks={}, device="cpu")
    assert assessment[0]["scoring_suffixes"] == [4, 5]
    assert len(assessment[0]["closed_loop_mass"]) == 2
    assert assessment[0]["student_mass"] == assessment[0]["b8_mass"]
    worlds = [dict(world_id=0, roster_index=0, mix="scripted")]
    rows = paired_gameplay(worlds, ZeroPolicy(), cfg=cfg, b8=b8, networks={}, device="cpu")
    assert rows[0]["b8_mass"] == rows[0]["student_mass"]
    assert rows[0]["b8_alive"] == rows[0]["student_alive"]
    mask = list(root["mask"])
    mask[(root["b8_action"] + 1) % 3] = not mask[(root["b8_action"] + 1) % 3]
    bad = dict(root, mask=mask)
    with pytest.raises(ValueError, match="identity"):
        label_roots([bad], cfg=cfg, b8=b8, networks={}, device="cpu")


def endpoint() -> tuple[list[dict], list[dict]]:
    mixes = ("scripted", "frozen", "mixed")
    roots = []
    for i in range(384):
        late = i < 128
        baseline_dead = late and i < 32
        student_dead = late and i < 24
        roots.append(
            dict(
                uid=f"t34-spatial-student-20261030/v1/a0/root{i}",
                partition="assessment",
                world_id=i // 2,
                kind="late" if late else "ordinary",
                mix=mixes[(i // 2) % 3],
                b8_action=0,
                teacher_action=1,
                student_action=1,
                selection_freeze_sha256="f" * 64,
                scoring_suffixes=[4, 5],
                b8_mass=[100, 100],
                teacher_mass=[120, 120],
                student_mass=[112, 112],
                closed_loop_mass=[112, 112],
                b8_survival=[0, 0] if baseline_dead else [1, 1],
                teacher_survival=[1, 1],
                student_survival=[0, 0] if student_dead else [1, 1],
                closed_loop_survival=[1, 1],
            )
        )
    gameplay = [
        dict(
            world_id=i,
            mix=mixes[i % 3],
            b8_mass=100,
            student_mass=101,
            b8_alive=0.8,
            student_alive=0.79,
            student_action_counts=[10, 10, 10, 10, 10, 10],
            b8_initial_state_sha256="a" * 64,
            student_initial_state_sha256="a" * 64,
        )
        for i in range(192)
    ]
    return roots, gameplay


def read(roots, gameplay, **kwargs):
    with patch("research.spatial_student_pilot.readout.platform.system", return_value="Darwin"):
        return evaluate_readout(
            roots,
            gameplay,
            support=dict(
                collision_audit=True,
                resume_verified=True,
                namespace="t34-spatial-student-20261030/v1",
            ),
            latency_p95_ms=3.9,
            complete=True,
            **kwargs,
        )


def test_readout_all_clauses_and_harm_falsifications() -> None:
    roots, gameplay = endpoint()
    result = read(roots, gameplay)
    assert result["verdict"] == "GO"
    assert result["metrics"]["student_late_deaths"] == 24
    harmed = deepcopy(gameplay)
    harmed[0]["student_mass"] = 0
    for row in harmed:
        row["student_mass"] = 99
    result = read(roots, harmed)
    assert result["verdict"] == "KILL"  # stricter >=B8 mass, not 90% retention
    assert not result["clauses"]["equal_mix_gameplay_mass_at_least_b8"]
    collapse = deepcopy(gameplay)
    for row in collapse:
        row["student_action_counts"] = [100, 0, 0, 0, 0, 0]
    assert read(roots, collapse)["verdict"] == "KILL"
    illegal = deepcopy(roots)
    illegal[0]["illegal_output"] = True
    assert read(illegal, gameplay)["verdict"] == "KILL"


def test_invalid_precedes_kill_and_requires_baseline_deaths() -> None:
    roots, gameplay = endpoint()
    roots[0]["student_mass"] = [np.nan, 1]
    assert read(roots, gameplay)["verdict"] == "INVALID"
    roots, gameplay = endpoint()
    for row in roots:
        row["b8_survival"] = [1, 1]
    result = read(roots, gameplay)
    assert result["verdict"] == "KILL"
    assert result["clauses"]["baseline_late_deaths_at_least16"] is False
    assert read(roots[:-1], gameplay)["verdict"] == "INVALID"
    missing_counts = deepcopy(gameplay)
    for row in missing_counts:
        row.pop("student_action_counts")
    assert read(roots, missing_counts)["verdict"] == "INVALID"


def test_clustered_bootstrap_is_order_invariant_and_world_clustered() -> None:
    rows = [
        dict(world_id=i // 2, mix=("scripted", "frozen", "mixed")[(i // 2) % 3]) for i in range(18)
    ]
    values = [float((i // 2) % 3 + 1) for i in range(18)]
    lower = clustered_lb90(rows, values)
    assert lower == clustered_lb90(list(reversed(rows)), list(reversed(values)))
    assert lower <= 2.0
    assert clustered_lb90(rows, values, stratified=True) == 2.0


def test_pairing_partition_and_missing_count_fail_closed() -> None:
    roots, gameplay = endpoint()
    bad = deepcopy(gameplay)
    bad[0]["student_initial_state_sha256"] = "b" * 64
    assert read(roots, bad)["verdict"] == "INVALID"
    bad = deepcopy(roots)
    bad[0]["partition"] = "train"
    assert read(bad, gameplay)["verdict"] == "INVALID"
    bad = deepcopy(gameplay)
    for row in bad:
        row.pop("student_action_counts")
    assert read(roots, bad)["verdict"] == "INVALID"
    bad[0]["student_action_counts"] = [float("nan")] * 6
    assert read(roots, bad)["verdict"] == "INVALID"


def test_foreign_root_prefix_and_illegal_forced_action_are_refused() -> None:
    root, cfg, b8 = fixture_root()
    with pytest.raises(ValueError, match="namespace"):
        label_roots([dict(root, uid="foreign/a0/root0")], cfg=cfg, b8=b8, networks={}, device="cpu")
    illegal = int(np.flatnonzero(~np.asarray(root["mask"], dtype=bool))[0])
    with pytest.raises(ValueError, match="illegal forced"):
        _rollouts([root], [_Job(0, illegal, 0)], cfg=cfg, b8=b8, networks={}, device="cpu")
    roots, gameplay = endpoint()
    roots[0]["uid"] = "foreign/a0/root0"
    assert read(roots, gameplay)["verdict"] == "INVALID"


def test_restore_near_ties_are_logged_but_large_or_illegal_gaps_refuse() -> None:
    from research.spatial_student_pilot.search import RESTORE_Q_EPSILON, verify_restored_action

    mask = torch.tensor([[True, True, True, False, False, False]])
    q = torch.tensor([[0.0, RESTORE_Q_EPSILON / 2, -1.0, 0, 0, 0]])
    events = verify_restored_action(mask, mask, q, torch.tensor([1]), torch.tensor([0]))
    assert len(events) == 1 and events[0]["expected"] == 0
    q[0, 1] = 2 * RESTORE_Q_EPSILON
    with pytest.raises(ValueError, match="identity"):
        verify_restored_action(mask, mask, q, torch.tensor([1]), torch.tensor([0]))
    q[0, 1] = RESTORE_Q_EPSILON / 2
    with pytest.raises(ValueError, match="identity"):
        verify_restored_action(mask, mask, q, torch.tensor([1]), torch.tensor([2]))
    with pytest.raises(ValueError, match="identity"):
        verify_restored_action(mask, ~mask, q, torch.tensor([1]), torch.tensor([0]))


def test_ruler_gameplay_has_fixed_slab_bounded_ticks_and_setup_free_time() -> None:
    from dataclasses import replace

    cfg = replace(
        PilotConfig().development("preflight"),
        tiny=False,
        mode="ruler",
        namespace=PilotConfig().development("ruler").namespace,
    )
    # Keep tiny simulator geometry for a fast test while exercising ruler semantics.
    from unittest.mock import Mock

    proxy = Mock(wraps=cfg)
    proxy.mode = "ruler"
    proxy.horizon = 50
    proxy.world_batch_size = 3
    proxy.gameplay_worlds = 3
    proxy.sim_config = PilotConfig().development("preflight").sim_config
    worlds = [dict(world_id=i, roster_index=i, mix="scripted") for i in range(3)]
    from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256

    rows = paired_gameplay(
        worlds,
        ZeroPolicy(),
        cfg=proxy,
        b8=ZeroPolicy(),
        networks={CHAMPION_SHA256: ZeroPolicy(), FRP3_S12_SHA256: ZeroPolicy()},
        device="cpu",
        ruler_ticks=4,
    )
    for arm in ("b8", "student"):
        assert rows[0][arm + "_ticks"] == 4
        assert rows[0][arm + "_executed_world_steps"] == 12
        assert rows[0][arm + "_loop_seconds"] < rows[0][arm + "_wall_seconds"]
    with pytest.raises(ValueError, match="ruler-only"):
        paired_gameplay(
            worlds,
            ZeroPolicy(),
            cfg=PilotConfig().development("preflight"),
            b8=ZeroPolicy(),
            networks={},
            device="cpu",
            ruler_ticks=4,
        )
