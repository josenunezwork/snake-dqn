"""CPU-only bounded fixtures for TRAIN-01 preparation."""

import copy
import subprocess
import sys
import types
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from research.sandbox_a_resident import training_prep
from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic
from research.sandbox_a_resident.training_prep import (
    PreparationConfig,
    PreparationFailure,
    PreparationState,
    _Row,
    _take_cross_life_batch,
    _world_split,
    calibrate,
    prepare,
    solve_temperature,
)
from src.model.ego2s_network import Ego2sNet


class Task:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.tick = 0
        self.config = SimpleNamespace(horizon=5000)
        self.age = torch.zeros(6, dtype=torch.long)
        self.roster_id = torch.tensor([0, 1, 2, 0, 1, 2])
        self.sim = SimpleNamespace(
            length=torch.tensor([[100], [250], [450], [100], [250], [450]], dtype=torch.long)
        )

    def observe(self):
        return (
            torch.zeros((6, 7, 31, 31), dtype=torch.uint8),
            torch.zeros((6, 4, 37, 37), dtype=torch.uint8),
            torch.zeros((6, 60)),
            torch.ones((6, 6), dtype=torch.bool),
        )

    def step(self, action):
        del action
        self.tick += 1
        self.age += 1
        values = ([650, 850, 100], [250, 450, 650], [850, 100, 250])[self.tick % 3]
        self.sim.length[:, 0] = torch.tensor(values * 2)
        reset = self.age.ge(2)
        self.age[reset] = 0
        return SimpleNamespace(
            reward=torch.ones(6), terminal=reset, cut=torch.zeros(6, dtype=torch.bool), reset=reset
        )

    def state_dict(self):
        return {"age": self.age.clone(), "length": self.sim.length.clone()}

    def load_state_dict(self, state):
        self.age.copy_(state["age"])
        self.sim.length.copy_(state["length"])


def factory(stage: str):
    return Task(stage)


def _models():
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    return B8Actor(b8_model=b8), ClockedPopArtCritic(b8)


def _assert_exact_tree(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys()
        for key in actual:
            _assert_exact_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual)) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_exact_tree(actual_item, expected_item)
    else:
        assert actual == expected


def _frozen_training_prep_module() -> types.ModuleType:
    source = subprocess.check_output(
        [
            "git",
            "show",
            "b4225e2540c6b0cdb2e00e2ce84debc8296f5deb:research/sandbox_a_resident/training_prep.py",
        ],
        text=True,
    )
    name = "_train01_training_prep_frozen_oracle"
    module = types.ModuleType(name)
    module.__file__ = "git:b4225e2540c6b0cdb2e00e2ce84debc8296f5deb:training_prep.py"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _record_fit_batches(module, monkeypatch: pytest.MonkeyPatch):
    batches = []
    original = module._fit_batch

    def record(rows, critic, optimizer, config):
        targets = [row.reward for row in rows]
        losses = []

        def capture_loss(critic_module, _inputs, prediction):
            value = torch.tensor(targets, device=prediction.device)
            loss = (
                0.5
                * (critic_module.normalise(prediction) - critic_module.normalise(value))
                .square()
                .mean()
            )
            losses.append(float(loss.detach().cpu()))

        hook = critic.register_forward_hook(capture_loss)
        try:
            original(rows, critic, optimizer, config)
        finally:
            hook.remove()
        assert len(losses) == 1
        batches.append(
            {
                "targets": targets,
                "loss": losses[0],
                "critic": copy.deepcopy(critic.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
        )

    monkeypatch.setattr(module, "_fit_batch", record)
    return batches


def test_temperature_preserves_ties_as_the_greedy_set():
    logits = torch.tensor([[2.0, 2.0, 0.0, -1.0, -1.0, -1.0]])
    legal = torch.ones_like(logits, dtype=torch.bool)
    temperature = solve_temperature(logits, legal, 0.01)
    # Departure excludes both tied maximizing actions; a tie-break implementation
    # would calibrate a radically different temperature.
    p = torch.softmax(logits / temperature, -1)
    assert torch.isclose(1 - p[:, :2].sum(), torch.tensor(0.01), atol=2e-4)


def test_temperature_canonicalizes_all_false_advisory_masks() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0, 4.0, 3.0, 2.0]])
    # Collection uses normal actions for this transient malformed advisory row.
    # Calibration must use the identical support instead of refusing it.
    temperature = solve_temperature(logits, torch.zeros_like(logits, dtype=torch.bool), 0.01)
    assert temperature > 0.0


def test_calibration_is_bounded_and_reinstalls_exact_b8_head_scale():
    actor, _ = _models()
    original = actor.reference.adv.weight.detach().clone()
    temperature, rows = calibrate(
        factory, actor, PreparationConfig.tiny(), torch.Generator().manual_seed(4)
    )
    assert rows == 10 and temperature > 0
    # prepare does installation; calibration itself never trains or changes B8.
    torch.testing.assert_close(actor.reference.adv.weight, original)


def test_calibration_emits_collection_and_solver_stage_boundaries() -> None:
    actor, _ = _models()
    events = []
    calibrate(
        factory,
        actor,
        PreparationConfig.tiny(),
        torch.Generator().manual_seed(4),
        on_progress=lambda stage, payload: events.append((stage, dict(payload))),
    )
    assert [stage for stage, _ in events if stage.endswith(("-start", "-end"))] == [
        "calibration-start",
        "calibration-end",
        "calibration-solve-start",
        "calibration-solve-end",
    ]
    assert all("elapsed_seconds" in payload for _, payload in events)


def test_time_based_progress_pulses_before_the_row_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(training_prep.time, "monotonic", lambda: now[0])
    events = []
    reporter = training_prep._ProgressReporter(
        replace(PreparationConfig.tiny(), progress_rows=1_000, progress_seconds=60.0),
        lambda stage, payload: events.append((stage, dict(payload))),
    )
    reporter.start("critic-fit", 0, 10_000, fit_rows=0)
    now[0] = 59.0
    reporter.advance("critic-fit", 1, 10_000, fit_rows=0)
    now[0] = 60.0
    reporter.advance("critic-fit", 1, 10_000, fit_rows=0)
    reporter.end("critic-fit", 1, 10_000, fit_rows=0)
    assert [stage for stage, _ in events] == ["critic-fit-start", "critic-fit", "critic-fit-end"]
    assert events[1][1] == {"rows": 1, "limit": 10_000, "elapsed_seconds": 60, "fit_rows": 0}


def test_retained_observation_owns_only_one_selected_world_and_preserves_values() -> None:
    values = (
        torch.arange(6 * 7 * 2 * 2, dtype=torch.uint8).reshape(6, 7, 2, 2),
        torch.arange(6 * 4 * 3 * 3, dtype=torch.uint8).reshape(6, 4, 3, 3),
        torch.arange(6 * 60, dtype=torch.float32).reshape(6, 60),
    )
    selected = training_prep._selected_cpu_observation(values, torch.tensor([1, 3, 5]))
    retained = training_prep._retained_observation(selected, 1)
    for source, stored in zip(values, retained):
        torch.testing.assert_close(stored, source[3:4])
        assert stored.untyped_storage().nbytes() == stored.numel() * stored.element_size()


def test_prepare_matches_frozen_pre_copying_oracle_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The storage-copy optimization cannot alter the frozen preparation path."""
    oracle = _frozen_training_prep_module()
    try:
        torch.manual_seed(173)
        actual_actor, actual_critic = _models()
        oracle_actor, oracle_critic = _models()
        oracle_actor.load_state_dict(copy.deepcopy(actual_actor.state_dict()))
        oracle_critic.load_state_dict(copy.deepcopy(actual_critic.state_dict()))
        actual_generator = torch.Generator().manual_seed(71)
        oracle_generator = torch.Generator().set_state(actual_generator.get_state())
        actual_batches = _record_fit_batches(training_prep, monkeypatch)
        oracle_batches = _record_fit_batches(oracle, monkeypatch)

        with pytest.raises(PreparationFailure) as actual_failure:
            prepare(
                factory,
                actual_actor,
                actual_critic,
                PreparationConfig.tiny(),
                actual_generator,
            )
        with pytest.raises(oracle.PreparationFailure) as oracle_failure:
            oracle.prepare(
                factory,
                oracle_actor,
                oracle_critic,
                oracle.PreparationConfig.tiny(),
                oracle_generator,
            )

        assert actual_failure.value.state is not None
        assert oracle_failure.value.state is not None
        _assert_exact_tree(
            actual_failure.value.state.state_dict(), oracle_failure.value.state.state_dict()
        )
        _assert_exact_tree(actual_actor.state_dict(), oracle_actor.state_dict())
        _assert_exact_tree(actual_critic.state_dict(), oracle_critic.state_dict())
        torch.testing.assert_close(
            actual_generator.get_state(), oracle_generator.get_state(), rtol=0, atol=0
        )
        _assert_exact_tree(actual_batches, oracle_batches)
    finally:
        sys.modules.pop(oracle.__name__, None)


def test_calibration_requires_all_five_frozen_length_bins_before_fixed_cap():
    actor, _ = _models()
    calls = []

    class ShortTask(Task):
        def step(self, action):
            result = super().step(action)
            self.sim.length.fill_(100)
            return result

    def short_factory(stage):
        calls.append(stage)
        return ShortTask(stage)

    with pytest.raises(PreparationFailure, match="five frozen length-bin quotas"):
        calibrate(
            short_factory,
            actor,
            replace(PreparationConfig.tiny(), calibration_max_rows=12),
            torch.Generator().manual_seed(3),
        )
    assert calls == ["calibration"]


def test_failed_tiny_critic_endpoint_never_returns_actor_ready():
    actor, critic = _models()
    config = PreparationConfig.tiny()
    with pytest.raises(PreparationFailure, match="critic preparation endpoint failed"):
        prepare(factory, actor, critic, config, torch.Generator().manual_seed(5))
    # The fixed tiny task has constant return, so late EV cannot meet .10;
    # preparation failure is terminal before the PPO loop receives an actor.
    assert critic.training


def test_critic_batches_wait_for_and_mix_completed_lives() -> None:
    config = replace(PreparationConfig.tiny(), critic_batch=6)

    def row(mix: int) -> _Row:
        return _Row(
            (
                torch.zeros((1, 7, 31, 31), dtype=torch.uint8),
                torch.zeros((1, 4, 37, 37), dtype=torch.uint8),
                torch.zeros((1, 60)),
            ),
            1.0,
            1.0,
            1.0,
            mix,
        )

    staged = [[row(0) for _ in range(4)]]
    assert _take_cross_life_batch(staged, config, torch.Generator().manual_seed(7)) is None
    staged.append([row(1) for _ in range(4)])
    assert _take_cross_life_batch(staged, config, torch.Generator().manual_seed(7)) is None
    staged.append([row(2) for _ in range(4)])
    batch = _take_cross_life_batch(staged, config, torch.Generator().manual_seed(7))
    assert batch is not None and len(batch) == config.critic_batch
    assert Counter(value.mix for value in batch) == {0: 2, 1: 2, 2: 2}


def _tagged_life(identity: int, count: int) -> list[_Row]:
    observation = (torch.empty(0), torch.empty(0), torch.empty(0))
    return [_Row(observation, float(identity), 1.0, float(index), 0) for index in range(count)]


def test_many_lives_are_balanced_across_every_production_sized_batch() -> None:
    config = replace(PreparationConfig(), critic_batch=256)
    staged = [_tagged_life(identity, 500) for identity in range(12)]
    generator = torch.Generator().manual_seed(37)
    seen = set()
    for _ in range(10):
        batch = _take_cross_life_batch(staged, config, generator)
        assert batch is not None and len(batch) == 256
        counts = Counter(row.length for row in batch)
        assert len(counts) == 12
        assert max(counts.values()) - min(counts.values()) <= 1
        assert max(counts.values()) <= config.critic_batch // 8
        identities = {(row.length, row.reward) for row in batch}
        assert len(identities) == 256 and not seen.intersection(identities)
        seen.update(identities)
    assert sum(map(len, staged)) == 6000 - 2560


def test_short_residual_plus_long_life_waits_for_a_multilife_pool() -> None:
    config = replace(PreparationConfig(), critic_batch=256)
    staged = [_tagged_life(0, 1), _tagged_life(1, 5000)]
    generator = torch.Generator().manual_seed(41)
    assert _take_cross_life_batch(staged, config, generator) is None
    assert list(map(len, staged)) == [1, 5000]
    staged.extend(_tagged_life(identity, 5000) for identity in range(2, 9))
    batch = _take_cross_life_batch(staged, config, generator)
    assert batch is not None and len(batch) == 256
    assert max(Counter(row.length for row in batch).values()) <= 32


def test_more_staged_lives_than_batch_rows_never_oversizes_or_negative_slices() -> None:
    config = replace(PreparationConfig.tiny(), critic_batch=4)
    staged = [_tagged_life(identity, 3) for identity in range(20)]
    batch = _take_cross_life_batch(staged, config, torch.Generator().manual_seed(43))
    assert batch is not None and len(batch) == 4
    assert len({row.length for row in batch}) == 4
    assert sum(map(len, staged)) == 56


def test_holdout_split_is_mix_balanced_for_production_and_tiny_fixture() -> None:
    for worlds, config in ((24, PreparationConfig()), (6, PreparationConfig.tiny())):
        counts = [
            sum(
                _world_split(index, config) == "holdout" and index % 3 == mix
                for index in range(worlds)
            )
            for mix in range(3)
        ]
        assert max(counts) - min(counts) <= 1 and min(counts) > 0


def test_fit_row_accounting_only_counts_full_optimizer_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _models()
    consumed: list[int] = []
    original = training_prep._fit_batch

    def record(*args: object, **kwargs: object) -> None:
        rows = args[0]
        consumed.append(len(rows))  # type: ignore[arg-type]
        for row in rows:  # type: ignore[union-attr]
            for observation in row.observation:
                assert observation.untyped_storage().nbytes() == (
                    observation.numel() * observation.element_size()
                )
        original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(training_prep, "_fit_batch", record)
    with pytest.raises(PreparationFailure) as failed:
        prepare(factory, actor, critic, PreparationConfig.tiny(), torch.Generator().manual_seed(17))
    assert failed.value.state is not None
    assert consumed and set(consumed) == {PreparationConfig.tiny().critic_batch}
    assert failed.value.state.fit_rows == sum(consumed)


def test_baseline_scalars_are_exactly_optimizer_rows_with_staged_tails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _models()
    consumed = []
    baseline_rows = []
    original_fit = training_prep._fit_batch
    original_baseline = training_prep.fit_time_length_baseline

    def record_fit(rows, *args):
        consumed.extend((row.length, row.clock, row.reward) for row in rows)
        original_fit(rows, *args)

    def record_baseline(lengths, clocks, targets):
        baseline_rows.extend(zip(lengths.tolist(), clocks.tolist(), targets.tolist()))
        return original_baseline(lengths, clocks, targets)

    monkeypatch.setattr(training_prep, "_fit_batch", record_fit)
    monkeypatch.setattr(training_prep, "fit_time_length_baseline", record_baseline)
    with pytest.raises(PreparationFailure) as failed:
        prepare(
            factory,
            actor,
            critic,
            replace(PreparationConfig.tiny(), critic_min_fit_rows=8),
            torch.Generator().manual_seed(47),
        )
    assert failed.value.state.fit_rows == len(consumed) == len(baseline_rows) == 8
    torch.testing.assert_close(torch.tensor(baseline_rows), torch.tensor(consumed))


def test_storage_cap_counts_staged_completed_lives(monkeypatch: pytest.MonkeyPatch) -> None:
    actor, critic = _models()
    # Use a deliberately small storage cap only at this synthetic task seam.
    config = SimpleNamespace(**{**vars(PreparationConfig.tiny()), "max_open_rows": 5})
    monkeypatch.setattr(training_prep, "_fit_batch", lambda *_: pytest.fail("must not fit"))
    with pytest.raises(PreparationFailure, match="open plus staged") as failed:
        prepare(factory, actor, critic, config, torch.Generator().manual_seed(53))
    assert failed.value.state.stage == "failed"
    # On tick two, world 1's two rows have moved to staging before world 5
    # pushes total storage to six. Open-only accounting would incorrectly pass.
    assert failed.value.state.critic_rows == 12


def _h5000_storage_fixture(monkeypatch: pytest.MonkeyPatch, staged_rows: int) -> list[int]:
    """Seed the real collector just before all 24 open lives reach H5000."""
    config = PreparationConfig()
    row = _tagged_life(0, 1)[0]
    life_type = training_prep._Life
    open_lives = []

    # Share a lightweight observation: only row counts matter to this guard.
    def open_life():
        life = life_type(rows=[row] * 4998)
        open_lives.append(life)
        return life

    monkeypatch.setattr(training_prep, "_Life", open_life)
    # Include all E24 slots to test the conservative envelope, even though
    # production discards whole-world holdout slots during fitting.
    monkeypatch.setattr(training_prep, "_world_split", lambda *_: "fit")
    observed = []

    class StorageTask(Task):
        def __init__(self, stage: str) -> None:
            super().__init__(stage)
            self.age = torch.full((24,), 4998, dtype=torch.long)
            self.roster_id = torch.arange(24) % 3
            self.sim.length = torch.ones((24, 1), dtype=torch.long)

        def observe(self):
            return tuple(value[:1].expand(24, *value.shape[1:]) for value in super().observe())

        def step(self, action):
            self.age += 1
            return SimpleNamespace(reward=torch.ones(24), reset=torch.zeros(24, dtype=torch.bool))

    class ReachedStorageBoundary(Exception):
        pass

    def stage_then_check(staged, *_):
        if not staged:
            # The valid case has seven H5000 lives and a short residual;
            # the overflow case deliberately injects excess staged storage.
            staged.extend([[row] * 5000 for _ in range(7)])
            staged.append([row] * (staged_rows - 7 * 5000))
            if staged_rows == 7 * 5000 + 255:
                assert _take_cross_life_batch(staged, config, torch.Generator()) is None
            return None
        assert len(open_lives) == 24 and all(len(life.rows) == 5000 for life in open_lives)
        observed.append(sum(len(life.rows) for life in open_lives) + sum(map(len, staged)))
        raise ReachedStorageBoundary

    monkeypatch.setattr(training_prep, "_take_cross_life_batch", stage_then_check)
    actor, critic = _models()
    monkeypatch.setattr(actor, "reference_logits", lambda *_: torch.zeros((24, 6)))
    try:
        prepare(
            StorageTask,
            actor,
            critic,
            config,
            torch.Generator().manual_seed(71),
            state=PreparationState(temperature=1.0, stage="calibrated"),
        )
    except ReachedStorageBoundary:
        pass
    return observed


def test_h5000_worst_case_open_plus_undrainable_staging_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _h5000_storage_fixture(monkeypatch, 7 * 5000 + 255) == [155_255]
    assert PreparationConfig().max_open_rows > 155_255


def test_h5000_storage_cap_still_refuses_genuine_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PreparationFailure, match="open plus staged") as failed:
        _h5000_storage_fixture(monkeypatch, PreparationConfig().max_open_rows - 24 * 5000 + 1)
    assert failed.value.state.stage == "failed"
    assert failed.value.state.critic_rows == 48


def test_zero_life_mix_requires_support_until_cap_and_fails_without_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _models()
    captured_counts = []
    original_quality = training_prep.critic_quality

    class MissingMixedLife(Task):
        def step(self, action):
            step = super().step(action)
            # Holdout mixed slot 2 never completes, while slots 0 and 4 do.
            self.age[2] = self.tick
            step.reset[2] = False
            return step

    def quality(*args, **kwargs):
        captured_counts.append(args[5].tolist())
        return original_quality(*args, **kwargs)

    monkeypatch.setattr(training_prep, "critic_quality", quality)
    config = replace(
        PreparationConfig.tiny(),
        critic_min_fit_rows=8,
        critic_max_rows=72,
        min_holdout_lives_per_mix=1,
    )
    # Reproduce sparse mappings from older saved preparation state too.
    state = PreparationState(temperature=1.0, stage="calibrated", holdout_lives_by_mix={})
    with pytest.raises(PreparationFailure, match="critic preparation endpoint failed") as failed:
        prepare(
            lambda stage: MissingMixedLife(stage),
            actor,
            critic,
            config,
            torch.Generator().manual_seed(59),
            state=state,
        )
    result = failed.value.state
    assert result.stage == "failed" and result.critic_rows == config.critic_max_rows
    assert result.completed_holdout_lives >= config.min_holdout_lives
    assert result.late_holdout_rows >= config.min_holdout_late_rows
    assert result.holdout_lives_by_mix == {"scripted": 4, "frozen": 4, "mixed": 0}
    assert captured_counts == [[4, 4, 0]] and critic.training


def test_holdout_quality_fallback_preserves_actual_support_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _models()

    def reject_quality(*_args, **_kwargs):
        raise ValueError("invalid quality fixture")

    monkeypatch.setattr(training_prep, "critic_quality", reject_quality)
    with pytest.raises(PreparationFailure) as failed:
        prepare(factory, actor, critic, PreparationConfig.tiny(), torch.Generator().manual_seed(61))
    result = failed.value.state
    assert result.quality["heldout_lives"] == result.completed_holdout_lives > 0
    assert result.quality["late_body_rows"] == result.late_holdout_rows > 0


def test_age_zero_holdout_lives_are_kept_at_fit_seam() -> None:
    actor, critic = _models()
    with pytest.raises(PreparationFailure) as failed:
        prepare(
            factory,
            actor,
            critic,
            replace(PreparationConfig.tiny(), critic_min_fit_rows=8, critic_max_rows=36),
            torch.Generator().manual_seed(67),
        )
    # Fit ends on tick four with all ages zero. The next life completes at
    # tick six (36 charged rows) and must be retained rather than discarded.
    result = failed.value.state
    assert result.completed_holdout_lives == 3 and result.holdout_rows == 6


def test_critic_mode_is_restored_when_heldout_forward_raises() -> None:
    actor, critic = _models()

    def fail_only_during_frozen_holdout(module: torch.nn.Module, _args: tuple[object, ...]) -> None:
        if not module.training:
            raise RuntimeError("heldout forward failed")

    hook = critic.register_forward_pre_hook(fail_only_during_frozen_holdout)
    try:
        with pytest.raises(RuntimeError, match="heldout forward failed"):
            prepare(
                factory, actor, critic, PreparationConfig.tiny(), torch.Generator().manual_seed(19)
            )
    finally:
        hook.remove()
    assert critic.training


def test_critic_mode_is_restored_when_heldout_task_step_raises() -> None:
    class FailingHoldoutTask(Task):
        def step(self, action):
            if self.tick >= 4:
                raise RuntimeError("heldout task step failed")
            return super().step(action)

    actor, critic = _models()
    with pytest.raises(RuntimeError, match="heldout task step failed"):
        prepare(
            lambda stage: FailingHoldoutTask(stage),
            actor,
            critic,
            replace(PreparationConfig.tiny(), critic_min_fit_rows=8),
            torch.Generator().manual_seed(23),
        )
    assert critic.training


def test_critic_mode_is_restored_when_heldout_observation_raises() -> None:
    actor, critic = _models()

    class FailingObservationTask(Task):
        def observe(self):
            if not critic.training:
                raise RuntimeError("heldout observation failed")
            return super().observe()

    with pytest.raises(RuntimeError, match="heldout observation failed"):
        prepare(
            lambda stage: FailingObservationTask(stage),
            actor,
            critic,
            PreparationConfig.tiny(),
            torch.Generator().manual_seed(29),
        )
    assert critic.training


def test_calibrated_callback_is_a_single_pre_critic_boundary():
    actor, critic = _models()
    seen = []

    def boundary(state, boundary_actor, boundary_critic, optimizer):
        seen.append((state.stage, state.critic_rows, boundary_actor.temperature, optimizer))
        assert boundary_critic is critic

    with pytest.raises(PreparationFailure):
        prepare(
            factory,
            actor,
            critic,
            PreparationConfig.tiny(),
            torch.Generator().manual_seed(8),
            on_calibrated=boundary,
        )
    assert len(seen) == 1
    assert seen[0][0:2] == ("calibrated", 0)
    assert seen[0][2] > 0


def test_tiny_fixture_limits_are_explicitly_not_production_values():
    tiny, production = PreparationConfig.tiny(), PreparationConfig()
    assert tiny.critic_min_fit_rows < production.critic_min_fit_rows
    assert tiny.min_holdout_lives < production.min_holdout_lives


def test_critic_cap_charges_every_simulated_world_not_only_fit_slots():
    actor, critic = _models()
    config = replace(
        PreparationConfig.tiny(),
        critic_max_rows=6,
        critic_min_fit_rows=3,
        min_holdout_lives=0,
        min_holdout_late_rows=0,
        min_holdout_lives_per_mix=0,
    )
    with pytest.raises(PreparationFailure) as failed:
        prepare(factory, actor, critic, config, torch.Generator().manual_seed(11))
    # E6 runs one whole world tick.  Only three slots are fit, but all six
    # simulator rows are charged before the cap blocks another tick.
    assert failed.value.state is not None
    assert failed.value.state.critic_rows == 6
    assert failed.value.state.fit_rows == 0
