"""Static live restore and complete observation+actor latency-entrypoint checks."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import capture_world_snapshot
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.latency import (
    load_final_candidate,
    probe_live,
    probe_sim,
    selected_assessment_roots,
    snapshot_to_live,
)
from research.spatial_student_pilot.model import SpatialStudent, export_checkpoint
from research.spatial_student_pilot.observation import (
    build_observation,
    build_observation_cpu_reference,
    build_observation_numpy,
    compact_from_live,
    compact_from_sim,
)
from src.game.game_state import GameState
from src.game.snake import Snake


class ZeroPolicy(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.zeros((args[0].shape[0], 6), device=args[0].device)


def fixture() -> tuple[CTDeviceSim, dict, PilotConfig]:
    cfg = PilotConfig().development("preflight")
    sim = CTDeviceSim(cfg.sim_config(1), device="cpu", train_mode=False, allow_respawn=True)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.fill_(1001)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    opponents = CTOpponentRuntime(
        sim, [training_rosters(cfg.worlds)[0]], b8_network=ZeroPolicy(), checkpoint_networks={}
    )
    sim.prepare()
    # Distinct body order, overlap and growth in a small static fixture. This
    # is an adapter test, not a naturally long-body threshold qualification.
    for slot in range(6):
        count = 7 + slot
        sim.seg_count[0, slot] = count
        sim.length[0, slot] = count + 3
        sim.head_ptr[0, slot] = 2
        sim.direction[0, slot] = slot % 4
        sim.boost_frames[0, slot] = slot % 3
        sim.frames_since_food[0, slot] = slot * 19
        sim.alive[0, slot] = slot != 4
        for k in range(count):
            sim.bodies[0, slot, (2 - k) % sim.cap] = torch.tensor([k + 1, 3])
    snapshot = capture_world_snapshot(sim, opponents, 0, world_id=0, life_id=0, tick=0)
    return sim, snapshot.to_state_dict(), cfg


def test_static_real_live_object_restore_and_numpy_observation_identity() -> None:
    sim, snapshot, cfg = fixture()
    game = snapshot_to_live(snapshot, cfg)
    assert isinstance(game, GameState)
    assert all(isinstance(snake, Snake) for snake in game.snakes)
    compact = compact_from_live(game)
    for a, b in zip(build_observation_numpy(compact), build_observation(compact_from_sim(sim))):
        assert torch.equal(a, b)
    for a, b in zip(build_observation_numpy(compact), build_observation_cpu_reference(compact)):
        assert a.numpy().tobytes() == b.tobytes()
    assert game.snakes[0].length == 10
    assert game.snakes[0].segments[0] == (10, 30)


def test_real_probe_helpers_time_complete_path_without_qualifying_fixture() -> None:
    torch.set_num_threads(1)
    sim, snapshot, cfg = fixture()
    student = SpatialStudent()
    report = probe_sim([sim], student, device="cpu", warmups=1, repeats=2)
    assert report["statistics"]["samples"] == 2
    assert report["statistics"]["p95_ms"] > 0
    assert not report["threshold_qualified"]
    game = snapshot_to_live(snapshot, cfg)
    report = probe_live(
        [game],
        student,
        device="cpu",
        warmups=1,
        repeats=2,
        legal_masks=[[False, True, False, False, False, False]],
    )
    assert report["statistics"]["samples"] == 2
    assert "live extraction" in report["scope"]
    assert not report["threshold_qualified"]
    assert not report["full_tick_qualified"]
    with pytest.raises(ValueError, match="legal mask"):
        probe_live([game], student, device="cpu", legal_masks=[[False] * 6])


def test_numpy_live_builder_handles_empty_food_and_dead_fatal_head() -> None:
    sim, _, _ = fixture()
    sim.food_present.fill_(False)
    sim.alive[0, 0] = False
    sim.bodies[0, 0, 2] = torch.tensor([-1, 25])
    compact = compact_from_sim(sim)
    for a, b in zip(build_observation_numpy(compact), build_observation(compact)):
        assert torch.equal(a, b)
    compact["alive"][0, 0] = True
    with pytest.raises(ValueError, match="outside"):
        build_observation_numpy(compact)


def _selection_batches(cfg: PilotConfig) -> list[dict]:
    """Outcome-free fabricated records exercise selection only, never latency."""
    partitions = cfg.partitions()
    train = [i for i, part in enumerate(partitions) if part == "train"]
    assess = [i for i, part in enumerate(partitions) if part == "assessment"]
    roots = []
    for part, kind, count, worlds in (
        ("train", "late", cfg.train_late, train),
        ("train", "ordinary", cfg.train_ordinary, train),
        ("assessment", "late", 256, assess),
        ("assessment", "ordinary", cfg.assessment_ordinary, assess),
    ):
        for index in range(count):
            parent = worlds[index // 2] if kind == "late" else worlds[index % len(worlds)]
            roots.append(
                dict(
                    uid=f"fixture/{part}/{kind}/{index}",
                    partition=part,
                    kind=kind,
                    world_id=parent,
                    priority=index,
                    snapshot=dict(
                        world_id=parent,
                        small_tensors={
                            "alive": torch.ones(6, dtype=torch.bool),
                            "length": torch.full((6,), 601),
                        },
                    ),
                )
            )
    batches = []
    for start in range(0, cfg.worlds, cfg.world_batch_size):
        ids = list(range(start, min(start + cfg.world_batch_size, cfg.worlds)))
        batches.append(
            dict(
                schema="spatial-collection/v1",
                world_ids=ids,
                roots=[r for r in roots if r["world_id"] in ids],
                demos=[],
                decisions=cfg.min_decisions if start == 0 else 0,
            )
        )
    batches[0]["demos"] = [dict(priority=i, world_id=train[0], tick=i) for i in range(cfg.demos)]
    return batches


def test_latency_reuses_exact_frozen_selection_from_complete_collection() -> None:
    cfg = PilotConfig()
    batches = _selection_batches(cfg)
    roots = selected_assessment_roots(cfg, batches)
    assert len(roots) == 128
    assert [r["priority"] for r in roots] == list(range(128))
    assert len({r["world_id"] for r in roots}) == 64
    # Collection stores >128 late candidates; only the same frozen priority
    # subset as fitting/assessment may be timed. Batch arrival order is irrelevant.
    assert [r["uid"] for r in roots] == [
        r["uid"] for r in selected_assessment_roots(cfg, list(reversed(batches)))
    ]
    with pytest.raises(ValueError, match="incomplete/duplicate"):
        selected_assessment_roots(cfg, batches[:-1])
    with pytest.raises(ValueError, match="incomplete/duplicate"):
        selected_assessment_roots(cfg, batches + batches[:1])


def test_candidate_requires_seal_and_bound_final_metadata(tmp_path) -> None:
    cfg = PilotConfig()
    metadata = dict(config_sha256=cfg.sha256, seed=1901, updates=8000, namespace=cfg.namespace)
    for name, change in (
        ("valid", {}),
        ("seed", {"seed": 7}),
        ("update", {"updates": 7999}),
        ("config", {"config_sha256": "wrong"}),
        ("ns", {"namespace": "wrong"}),
    ):
        path = tmp_path / (name + ".pth")
        export_checkpoint(SpatialStudent(), path, metadata={**metadata, **change})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_text(digest + "  " + path.name + "\n")
        path.with_name(path.name + ".complete").write_text("complete\n")
        if change:
            with pytest.raises(ValueError, match="allocation"):
                load_final_candidate(path, cfg)
        else:
            _, actual_meta, actual_digest = load_final_candidate(path, cfg)
            assert actual_meta == metadata
            assert actual_digest == digest
            path.with_name(path.name + ".sha256").write_text("")
            with pytest.raises(ValueError, match="digest"):
                load_final_candidate(path, cfg)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="real MPS required")
def test_real_mps_live_path_and_nondivisible_board() -> None:
    torch.set_num_threads(1)
    _, snapshot, cfg = fixture()
    game = snapshot_to_live(snapshot, cfg)
    report = probe_live([game], SpatialStudent(), device="mps", warmups=1, repeats=2)
    assert report["device"] == "mps"
    assert np.isfinite(report["sample_ms"]).all()
    assert not report["threshold_qualified"]
