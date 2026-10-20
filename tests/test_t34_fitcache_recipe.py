"""Independent pre-cache recipe oracle and v2 Amendment-1 loss semantics."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from research.t34_exit_b8_fit.data import StreamingReplayLoader, to_device
from research.t34_exit_b8_fit.interface import FakeReplaySource, normalize_row, validate_row
from research.t34_exit_b8_fit.loss import exit_fit_loss
from research.t34_exit_b8_fit.train import optimizer_for


def _oracle():
    path = Path(__file__).parent / "fixtures/t34_fitcache/data_pre_cache.py"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "144098b9688fb52c1e9c57558612b467ea26d4883694a9a41b72ba9faed68596"
    )
    name = "research.t34_exit_b8_fit._pre_cache_oracle"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(60, 6)

    def forward(self, local, glob, scalars):
        return self.linear(scalars)


@pytest.mark.parametrize("bank_counts", [False, True])
def test_frozen_batches_and_adam_trajectory_identical(tmp_path, bank_counts):
    from research.t34_exit_b8_fit.cache import build_cache

    source = FakeReplaySource(worlds=2, rows_per_world=16, seed=37)
    if bank_counts:
        source.training_row_counts = lambda epoch: (16, 16)
    cached = build_cache(source, tmp_path / "cache", identity={"fixture": 37}, include_tower=True)
    reference = _oracle().StreamingReplayLoader(source, batch_size=5, seed=11)
    candidate = StreamingReplayLoader(cached, batch_size=5, seed=11)
    torch.manual_seed(21)
    first = _Tiny()
    second = copy.deepcopy(first)
    opt1, opt2 = optimizer_for(first, True), optimizer_for(second, True)
    losses = []
    for epoch in (1, 2, 3):
        ref, got = list(reference.epoch(epoch)), list(candidate.epoch(epoch))
        assert len(ref) == len(got)
        for a, b in zip(ref, got):
            assert a.keys() == b.keys()
            for key in a:
                assert a[key].dtype == b[key].dtype
                assert a[key].tobytes() == b[key].tobytes(), key
            steps = []
            for model, opt, raw in ((first, opt1, a), (second, opt2, b)):
                batch = to_device(raw, torch.device("cpu"))
                q = model(batch["local"], batch["global"], batch["scalars"])
                loss, _ = exit_fit_loss(q, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                steps.append(loss.detach().numpy().tobytes())
            assert steps[0] == steps[1]
            losses.append(steps[0])
            for key, value in first.state_dict().items():
                assert torch.equal(value, second.state_dict()[key])
    assert len(losses) == 21


def _v2(kind):
    row = dict(next(iter(FakeReplaySource(rows_per_world=4).replay_epoch(1))).rows[0])
    row.update(label_kind=kind, t_star=4, a_star=2, F=[2, 3], death_window_id=1, decision=4)
    row.pop("forbidden_mask")
    return row


def test_confirm_rows_only_kl_even_with_contradictory_old_flags():
    from research.t34_exit_b8_fit.data import _stack, sample_weights

    row = normalize_row(_v2("shadow_confirm"))
    validate_row(row)
    assert not row["searched"] and not row["decisive"] and not row["death_window"]
    assert sample_weights([row]).tolist() == [1.0]
    row = dict(row, _sample_weight=1.0)
    batch = to_device(_stack([row]), torch.device("cpu"))
    q = torch.zeros((1, 6), requires_grad=True)
    _, pieces = exit_fit_loss(q, batch)
    assert pieces["kl"] > 0
    assert pieces["hinge"] == 0
    assert pieces["advantage_mse"] == 0
    assert pieces["total"] == pieces["kl"]


@pytest.mark.parametrize("kind", ["lad", "transition"])
def test_lad_survivor_F_never_becomes_forbidden(kind):
    row = _v2(kind)
    row["per_action"] = {str(a): {"survives": a in (2, 3)} for a in range(6)}
    result = normalize_row(row)
    validate_row(result)
    assert result["action_star"] == 2
    assert result["forbidden_mask"].tolist() == [True, True, False, False, True, True]
    assert result["searched"] and result["decisive"] and result["death_window"]


def test_ambiguous_F_and_misattached_decision_are_refused():
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_row(_v2("lad"))
    with pytest.raises(ValueError, match="t_star decision"):
        normalize_row(dict(_v2("lad"), decision=3))
    missing = _v2("lad")
    del missing["decision"]
    with pytest.raises(ValueError, match="t_star decision"):
        normalize_row(missing)


def test_cpu_prefetch_preserves_dtype_bytes():
    from research.t34_exit_b8_fit.data import device_batches

    batches = list(StreamingReplayLoader(FakeReplaySource(rows_per_world=4), 3).epoch(1))
    result = list(device_batches(iter(batches), torch.device("cpu")))
    for raw, got in zip(batches, result):
        for key in raw:
            assert raw[key].tobytes() == got[key].numpy().tobytes()


def test_confirm_auxiliary_targets_never_supervise_the_head():
    from research.t34_exit_b8_fit.interface import ReplayWorld
    from research.t34_exit_b8_fit.train import fit

    class Source(FakeReplaySource):
        def replay_epoch(self, epoch):
            for world in super().replay_epoch(epoch):
                rows = []
                for index, original in enumerate(world.rows):
                    row = dict(original)
                    row["label_kind"] = "lad" if index % 2 == 0 else "shadow_confirm"
                    row.update(t_star=index, decision=index, a_star=0, F=[])
                    row["forbidden_mask"] = [False] * 6
                    row["hindsight_log_free_region"] = float("nan") if index % 2 == 0 else 1.0
                    row["hindsight_log_free_region_mask"] = index % 2 == 1
                    rows.append(row)
                yield ReplayWorld(world.world_id, rows, True)

    model = _Tiny()  # No auxiliary head: every eligible target is a KL-only row.
    result = fit(
        source=Source(rows_per_world=4),
        model=model,
        previous_model=copy.deepcopy(model),
        allow_fake=True,
        max_updates=1,
        batch_size=4,
    )
    assert result["completed"]
