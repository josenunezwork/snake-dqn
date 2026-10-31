"""Review-02 regression contracts owned by the T4A event fixture layer."""

from __future__ import annotations

import hashlib

import torch

from research.gpu_sim_spike import t4a_events
from research.gpu_sim_spike.device_sim import DeviceSim
from src.simd_env.batch_sim import BatchSimConfig


def test_ring_capture_commitment_binds_typed_prefix_bytes() -> None:
    """The reducer-facing ring witness carries a typed byte commitment."""
    prefix = torch.tensor([[[5, 4], [4, 4]]], dtype=torch.int64)
    commitment = t4a_events._tensor_commitment(prefix)

    assert commitment == {
        "dtype": "torch.int64",
        "shape": [1, 2, 2],
        "sha256": hashlib.sha256(prefix.numpy().tobytes()).hexdigest(),
    }


def test_warm_output_comparator_detects_q_only_corruption() -> None:
    """S1: a warm-only Q bit flip cannot be discarded as a matching seam."""
    source = {
        "output.q.0": torch.zeros((4, 6), dtype=torch.float32),
        "output.obs.0.0": torch.ones((4, 3), dtype=torch.float32),
        "output.v8.0.mask": torch.ones((4, 6), dtype=torch.bool),
    }
    corrupted = {name: value.clone() for name, value in source.items()}
    corrupted["output.q.0"].view(torch.uint8).reshape(-1)[0] ^= 1

    assert t4a_events._warm_outputs_equal(source, source)["equal"]
    assert not t4a_events._warm_outputs_equal(source, corrupted)["equal"]


def test_mapped_event_fields_keep_only_canonical_witness_rows() -> None:
    values = {"event.ring_wrap.prefix": torch.arange(10, dtype=torch.int64).reshape(5, 2)}
    mapped = t4a_events._mapped_event_fields(
        type("Sim", (), {"device": torch.device("cpu")})(), values
    )

    assert mapped["event.ring_wrap.prefix"].tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_internal_mismatch_seals_offending_e64_row_before_later_mutation() -> None:
    left = {"sim.frame": torch.arange(64, dtype=torch.int64)}
    right = {name: value.clone() for name, value in left.items()}
    right["sim.frame"][63] ^= 1
    found = t4a_events.compare.compare_inventory(left, right)
    assert isinstance(found, t4a_events.compare.Mismatch)

    witness = t4a_events._internal_byte_witness(found, left, right, 64)
    before = witness["actual_descriptor"]["sha256"]
    left["sim.frame"].zero_()
    right["sim.frame"].zero_()

    assert witness["physical_worlds"] == 64
    assert witness["physical_rows"][0] == 63
    assert witness["actual_descriptor"]["shape"] == [4]
    assert witness["actual_descriptor"]["sha256"] == before


def test_eval_food_refills_on_next_prepare_after_consumption() -> None:
    """Eval-mode (train_mode=False) refills immediately per eating slot during step()."""
    cfg = BatchSimConfig(
        num_envs=1, num_snakes=3, game_width=100, game_height=80, initial_food=0, max_food=1
    )
    sim = DeviceSim(cfg, train_mode=False, allow_respawn=False)
    sim.head_ptr[:, 0] = 0
    sim.seg_count[:, 0] = sim.length[:, 0] = 1
    sim.alive[:, 0] = True
    sim.bodies[:, 0, 0] = torch.tensor([4, 4])
    sim.direction[:, 0] = 1
    sim.food_present.zero_()
    sim.corpse_food.zero_()
    sim.food_count.zero_()
    sim.food_cells[:, 0] = torch.tensor([5, 4])
    sim.food_present[:, 0] = True
    sim.food_count[:] = 1

    sim.step(torch.ones((1, 3), dtype=torch.long))
    assert sim._last_ambient_food_ate[0, 0].item()
    assert (sim.food_present[0] & ~sim.corpse_food[0]).sum().item() == 1
    sim.prepare()
    assert (sim.food_present[0] & ~sim.corpse_food[0]).sum().item() == 1
