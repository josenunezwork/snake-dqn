"""Focused executable contract tests for the E3 ranking students."""

import torch

from research.bl_info_value.students import make_students, ranking_loss


def _batch(n: int = 2):
    return {
        "local": torch.randint(0, 256, (n, 7, 31, 31), dtype=torch.uint8),
        "global": torch.randint(0, 256, (n, 4, 37, 37), dtype=torch.uint8),
        "scalars": torch.randn(n, 60),
        "geometry": torch.randn(n, 7, 83, 145),
        "history": torch.randn(n, 128, 43),
        "legal": torch.tensor([[True, True, False, True, False, True]] * n),
        "returns": torch.randn(n, 6) * 100.0,
        "action": torch.zeros(n, dtype=torch.long),
        "decisive": torch.ones(n, dtype=torch.bool),
        "world": torch.zeros(n, dtype=torch.long),
        "split": torch.zeros(n, dtype=torch.long),
    }


def test_parameter_counts_are_exact_and_all_parameters_receive_gradient():
    models = make_students()
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    assert len(set(counts.values())) == 1
    batch = _batch()
    for model in models.values():
        model.zero_grad()
        ranking_loss(model(batch), batch["returns"], batch["legal"]).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_a_and_b_ignore_unavailable_modalities_while_c_uses_history():
    models = make_students()
    batch = _batch(1)
    for arm in ("A", "B"):
        model = models[arm].eval()
        with torch.no_grad():
            before = model(batch)
            changed = dict(batch)
            changed["history"] = batch["history"] + 99.0
            if arm == "A":
                changed["geometry"] = batch["geometry"] + 99.0
            assert torch.equal(before, model(changed))
    with torch.no_grad():
        before = models["C"].eval()(batch)
        changed = dict(batch)
        changed["history"] = batch["history"] + 2.0
        assert not torch.equal(before, models["C"](changed))


def test_legal_mask_makes_loss_finite_and_ignores_invalid_scores():
    scores = torch.randn(2, 6, requires_grad=True)
    returns = torch.tensor([[0.0, 100.0, 1e9, 20.0, -1e9, 10.0]] * 2)
    legal = torch.tensor([[True, True, False, True, False, True]] * 2)
    first = ranking_loss(scores, returns, legal)
    changed = scores.detach().clone()
    changed[:, 2] = float("inf")
    changed[:, 4] = float("-inf")
    second = ranking_loss(changed, returns, legal)
    assert torch.isfinite(first) and torch.isfinite(second)
    assert torch.equal(first.detach(), second.detach())


def test_geometry_encoder_preserves_fullboard_location():
    """A four-cell board shift must not vanish into global average pooling."""
    model = make_students()["B"].eval()
    batch = _batch(1)
    first = torch.zeros_like(batch["geometry"])
    second = torch.zeros_like(first)
    first[0, 0, 30, 40] = 1.0
    second[0, 0, 30, 44] = 1.0
    with torch.no_grad():
        left = model({**batch, "geometry": first})
        right = model({**batch, "geometry": second})
    assert not torch.equal(left, right)
