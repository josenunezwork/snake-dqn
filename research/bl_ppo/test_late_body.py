"""Unit evidence for prospective P1 late-body support collectors."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.bl_ppo.collector import CollectorConfig, _long_body_start_mask
from research.bl_ppo.refit import CompleteLifeRefitBank


def _raw(worlds: int, steps: int, offset: int, *, terminal: bool) -> SimpleNamespace:
    ids = np.repeat(np.arange(worlds, dtype=np.int64) + offset, steps)
    frame = np.tile(np.arange(steps, dtype=np.int32), worlds)
    return SimpleNamespace(
        world_id=ids,
        episode=np.zeros(len(ids), dtype=np.int64),
        length=np.full(len(ids), 900, dtype=np.int32),
        frame=frame,
        reward=np.ones(len(ids), dtype=np.float32),
        done=np.array([terminal and value == steps - 1 for value in frame]),
        active_transitions=len(ids),
    )


class _FakeCollector:
    def __init__(self, config):
        self.config, self.calls, self.frame = config, [], 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def collect(self, _adapter, *, steps=None):
        assert steps is not None
        self.calls.append(steps)
        raw = _raw(self.config.worlds, steps, self.config.world_id_offset, terminal=False)
        raw.frame += self.frame
        self.frame += steps
        raw.done = raw.frame == self.config.horizon - 1
        return raw

    def peek(self):
        return {}


class _Adapter:
    def take_rollout(self, raw, _peek):
        steps = len(raw.world_id) // len(np.unique(raw.world_id))
        worlds = len(np.unique(raw.world_id))
        self.last_critic_features = torch.zeros((steps, worlds, 257))
        return object()


def test_registered_quarter_root_schedule_is_exact_and_restart_stable() -> None:
    config = CollectorConfig(worlds=8, processes=1, preflight=True, long_body_fraction=0.25)
    worlds = np.arange(8)
    rows = np.stack(
        [_long_body_start_mask(config, worlds, np.full(8, episode)) for episode in range(4)]
    )
    assert rows.sum(axis=0).tolist() == [1] * 8
    assert rows.sum() == 8
    assert np.array_equal(
        rows, np.stack([_long_body_start_mask(config, worlds, np.full(8, e)) for e in range(4)])
    )


def test_fixed_support_is_chunked_pinned_and_seed_disjoint(monkeypatch) -> None:
    from research.bl_ppo import late_body

    config = CollectorConfig(
        worlds=2,
        processes=1,
        rollout_steps=2,
        horizon=4,
        seed=777,
        long_bodies=True,
        world_id_offset=late_body.FIXED_HOLDOUT_WORLD_OFFSET,
        preflight=True,
    )
    monkeypatch.setattr(late_body, "fixed_holdout_config", lambda _excluded, horizon: config)
    made = []

    class Collector(_FakeCollector):
        def __init__(self, value):
            super().__init__(value)
            made.append(self)

    monkeypatch.setattr(late_body, "PersistentCollector", Collector)
    bank, progress = CompleteLifeRefitBank(reservoir_size=10), []
    result = late_body.collect_fixed_late_body_holdout(
        _Adapter(), bank, excluded_seeds={1, 2}, horizon=4, on_progress=progress.append
    )
    assert made[0].calls == [2, 2]
    assert progress == [4, 8]
    assert result["transitions"] == 8 and result["first_life_target_transitions"] == 8
    assert bank.late_support_by_source()["fixed_injected_long_body"] >= 2
    seeds = late_body.fixed_holdout_seeds({1, 2}, horizon=4)
    assert len(set(seeds)) == 2 and not set(seeds) & {1, 2}


def test_supplement_excludes_mod_five_holdout_rows_and_honors_stop(monkeypatch) -> None:
    from research.bl_ppo import late_body

    monkeypatch.setattr(late_body, "PersistentCollector", _FakeCollector)
    config = CollectorConfig(
        worlds=5,
        processes=1,
        rollout_steps=2,
        horizon=2,
        seed=4,
        long_body_fraction=0.25,
        preflight=True,
    )
    bank = CompleteLifeRefitBank()
    result = late_body.collect_refit_late_supplement(_Adapter(), bank, config)
    assert result["transitions"] == 10
    assert set(bank.source_counts()["holdout"]) == set()
    assert bank.source_counts()["fit"]["refit_long_body_support_supplement"] == 8
    with pytest.raises(ValueError, match="interrupted"):
        late_body.collect_refit_late_supplement(
            _Adapter(), CompleteLifeRefitBank(), config, should_stop=lambda: True
        )
