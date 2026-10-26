"""Focused contracts for B-2 heads, labels, losses, and sampler."""

import torch

from research.sandbox_b_heads.core_b2 import (
    CONFIGS,
    Config,
    build_sampling_weights,
    masks,
    prepare_sampling,
    sample_indices,
    training_loss,
)
from research.sandbox_b_heads.model_b2 import CosineFastGainResidual, make_head, optimizer_groups


def test_cached_sampler_matches_shared_helper_and_handles_one_row_batches():
    table = _table()
    config = Config("normed_linear", "hinge_top", 2.0, 1500, "decisive_life_balanced")
    indices = torch.tensor([0, 1, 2])
    plan = prepare_sampling(table, config, indices)
    torch.manual_seed(5)
    cached = sample_indices(table, config, 7, plan=plan)
    torch.manual_seed(5)
    direct = sample_indices(table, config, 7, indices)
    assert torch.equal(cached, direct)
    assert masks(table)[3][cached].sum() == 3
    assert set(cached.tolist()) <= set(indices.tolist())
    single = sample_indices(table, config, 1, plan=plan)
    assert single.tolist() == [1]  # The odd remainder always goes to complement.


def _table() -> dict[str, torch.Tensor]:
    return {
        "action_mask": torch.ones(4, 6, dtype=torch.bool),
        "status": torch.tensor([[1, 1, 0, 1, 1, 1]] * 4, dtype=torch.bool),
        "legal": torch.tensor([[1, 1, 1, 0, 1, 1]] * 4, dtype=torch.bool),
        "died": torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]] * 2, dtype=torch.bool),
        "bare": torch.tensor([0, 0, 0, 0]),
        "ordinary": torch.tensor([False, True, False, True]),
        "life_id": torch.tensor([0, 0, 1, 2]),
    }


def test_fixed_48_config_grid_is_unique_and_declared():
    assert len(CONFIGS) == 48
    assert len({config.id for config in CONFIGS}) == 48
    assert {config.ratio for config in CONFIGS} == {0.5, 2.0, 8.0, 32.0}
    assert {config.updates for config in CONFIGS} == {500, 1500}


def test_masks_penalize_selectable_unknown_and_illegal_actions_as_bad():
    selectable, survivor, bad, decisive, supervised, target = masks(_table())
    assert selectable.all()
    assert survivor[0, 1]
    assert bad[0, 2] and bad[0, 3]  # unknown and illegal, respectively
    assert decisive.tolist() == [True, False, True, False]
    assert supervised.all() and target.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_losses_and_gate_bce_produce_finite_gradients_on_sparse_masks():
    table = _table()
    base = torch.zeros(4, 6)
    for loss in ("kl_pairs", "hinge_top"):
        q = torch.randn(4, 6, requires_grad=True)
        gate = torch.randn(4, requires_grad=True)
        value = training_loss(
            q, base, table, Config("normed_linear", loss, 2.0, 500, "row_uniform"), gate
        )
        assert torch.isfinite(value)
        value.backward()
        assert torch.isfinite(q.grad).all() and torch.isfinite(gate.grad).all()


def test_life_balanced_sampler_weights_each_stratum_and_life_equally():
    table = _table()
    config = Config("normed_linear", "hinge_top", 2.0, 1500, "decisive_life_balanced")
    weights = build_sampling_weights(table, config)
    decisive = masks(table)[3]
    assert weights is not None
    assert torch.isclose(weights[decisive].sum(), torch.tensor(0.5))
    assert torch.isclose(weights[~decisive].sum(), torch.tensor(0.5))
    assert torch.isclose(weights[0], weights[2])
    sampled = sample_indices(table, config, 20)
    assert masks({key: value[sampled] for key, value in table.items()})[3].sum() == 10


def test_heads_are_finite_and_supervised_gate_stays_soft_in_eval():
    features, length, margin = torch.randn(3, 256), torch.rand(3), torch.tensor([0.0, 2.0, 32.0])
    for name in ("normed_linear", "cosine_fastgain", "supervised_gated_normed"):
        model = make_head(name)
        assert model(features, length, margin).shape == (3, 6)
        assert torch.isfinite(model(features, length, margin)).all()
    gated = make_head("supervised_gated_normed")
    gated.eval()
    with torch.no_grad():
        output, body, logits = gated.forward_components(features, length, margin)
        assert torch.equal(output, body * logits.sigmoid().unsqueeze(1))


def test_cosine_gain_and_bias_receive_tenfold_optimizer_rate():
    model = CosineFastGainResidual()
    groups = optimizer_groups(model, 0.003)
    assert [group["lr"] for group in groups] == [0.003, 0.03]
