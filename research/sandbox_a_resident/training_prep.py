"""Bounded TRAIN-01 temperature calibration and complete-life critic preparation.

This is deliberately a sandbox-local consumer of :class:`ResidentTrainingTask`.
It does not alter the R2/T4A simulator or its policies.  The collector keeps a
complete *open life* until its natural death/H5000 cut, calculates its exact
gamma-one suffix targets, and immediately trains or validates it.  Thus the
production row targets do not imply a multi-million-row GPU observation cache.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol

import torch

from research.sandbox_a_resident.training_model import (
    B8Actor,
    ClockedPopArtCritic,
    canonical_legal_mask,
    categorical,
)


class TrainingTask(Protocol):
    """The intentionally small task seam used by preparation and the runner."""

    config: Any
    age: torch.Tensor
    roster_id: torch.Tensor

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def step(self, hero_actions: torch.Tensor) -> Any: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


TaskFactory = Callable[[str], TrainingTask]


class PreparationFailure(RuntimeError):
    """A finite pre-actor preparation endpoint did not meet its fixed contract."""

    def __init__(self, message: str, state: "PreparationState | None" = None) -> None:
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class PreparationConfig:
    """Frozen TRAIN-01 preparation limits; ``tiny`` is test-only."""

    calibration_rows: int = 10_000
    calibration_max_rows: int = 5_242_880
    calibration_departure: float = 0.01
    calibration_quota_per_bin: int = 2_000
    # Preparation deliberately uses a 24-world task.  TRAIN-01 PPO remains
    # E1024; the smaller complete-life cohort keeps every H5000 trajectory in
    # bounded CPU staging instead of pretending a 1024x5000 cache fits.
    prep_worlds: int = 24
    critic_max_rows: int = 8_388_608
    critic_min_fit_rows: int = 2_000_000
    holdout_modulus: int = 5
    holdout_remainder: int = 0
    min_holdout_lives: int = 40
    min_holdout_late_rows: int = 160
    min_holdout_lives_per_mix: int = 8
    late_length: int = 800
    min_late_ev: float = 0.10
    critic_batch: int = 2048
    metric_cache_rows: int = 16_384
    # Exactly one full H5000 life for every preparation world. uint8 B8
    # observations are roughly 12 KiB/row; this is bounded CPU staging, never
    # a GPU replay cache.
    max_open_rows: int = 120_000
    progress_rows: int = 1_500_000

    def __post_init__(self) -> None:
        if not 0 < self.calibration_departure < 1:
            raise ValueError("calibration departure must be in (0, 1)")
        if not 0 < self.calibration_rows <= self.calibration_max_rows:
            raise ValueError("invalid bounded calibration row contract")
        if self.calibration_rows != self.calibration_quota_per_bin * 5:
            raise ValueError("TRAIN-01 calibration freezes five equal length-bin quotas")
        if self.prep_worlds not in (3, 24) or self.max_open_rows < self.prep_worlds * 5000:
            raise ValueError(
                "preparation needs frozen E24 (or explicit E3 test fixture) full-life staging"
            )
        if not 0 < self.critic_min_fit_rows <= self.critic_max_rows:
            raise ValueError("invalid bounded critic row contract")
        if self.holdout_modulus < 2 or not 0 <= self.holdout_remainder < self.holdout_modulus:
            raise ValueError("invalid whole-world holdout split")
        if (
            min(self.critic_batch, self.metric_cache_rows, self.max_open_rows, self.progress_rows)
            < 1
        ):
            raise ValueError("preparation capacities must be positive")

    @classmethod
    def tiny(cls) -> "PreparationConfig":
        """Explicit CPU-only fixture limits; never a production waiver."""
        return cls(
            calibration_rows=10,
            calibration_max_rows=24,
            calibration_quota_per_bin=2,
            prep_worlds=3,
            critic_max_rows=240,
            critic_min_fit_rows=12,
            min_holdout_lives=2,
            min_holdout_late_rows=2,
            min_holdout_lives_per_mix=0,
            late_length=2,
            critic_batch=8,
            metric_cache_rows=64,
            max_open_rows=15_000,
            progress_rows=10,
        )


@dataclass
class _Row:
    observation: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    length: float
    clock: float
    reward: float
    mix: int


@dataclass
class _Life:
    rows: list[_Row] = field(default_factory=list)
    total: float = 0.0


@dataclass
class _MetricRow:
    """Small CPU-only quality record; it deliberately owns no observation tensors."""

    length: float
    clock: float
    target: float
    prediction: float | None = None


@dataclass
class PreparationState:
    """Boundary-resumable torch-save state; task_state can contain tensors."""

    schema: str = "sandbox-a-train01-prep/v1"
    stage: str = "new"
    temperature: float | None = None
    calibration_rows: int = 0
    critic_rows: int = 0
    fit_rows: int = 0
    holdout_rows: int = 0
    completed_holdout_lives: int = 0
    holdout_lives_by_mix: dict[str, int] = field(
        default_factory=lambda: {"scripted": 0, "frozen": 0, "mixed": 0}
    )
    late_holdout_rows: int = 0
    quality: dict[str, float] = field(default_factory=dict)
    task_state: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "PreparationState":
        if value.get("schema") != "sandbox-a-train01-prep/v1":
            raise PreparationFailure("foreign preparation state")
        return cls(**dict(value))


@dataclass
class PreparationResult:
    state: PreparationState
    actor: B8Actor
    critic: ClockedPopArtCritic
    critic_optimizer: torch.optim.Optimizer


def _mix_name(value: int) -> str:
    return ("scripted", "frozen", "mixed")[value] if 0 <= value < 3 else "unknown"


def _lengths(task: TrainingTask, count: int, device: torch.device) -> torch.Tensor:
    sim = getattr(task, "sim", None)
    lengths = None if sim is None else getattr(sim, "length", None)
    if isinstance(lengths, torch.Tensor) and lengths.ndim == 2:
        return lengths[:, 0].to(device=device, dtype=torch.float32)
    return torch.ones(count, device=device)


def _world_split(index: int, config: PreparationConfig) -> str:
    return "holdout" if index % config.holdout_modulus == config.holdout_remainder else "fit"


def _departure_probability(logits: torch.Tensor, legal: torch.Tensor, temperature: float) -> float:
    scaled = logits.float() / temperature
    scaled = scaled.masked_fill(~legal.bool(), torch.finfo(scaled.dtype).min)
    # A tie is a greedy *set*: all maxima are retained, not arbitrarily broken.
    maximum = scaled.max(dim=-1, keepdim=True).values
    maximizing = legal.bool() & scaled.eq(maximum)
    departed = 1.0 - (torch.softmax(scaled, dim=-1) * maximizing).sum(dim=-1)
    return float(departed.mean().cpu())


@torch.no_grad()
def solve_temperature(logits: torch.Tensor, legal: torch.Tensor, target: float = 0.01) -> float:
    """Solve the frozen B8 softmax temperature for greedy-set departure mass."""
    if logits.ndim != 2 or legal.shape != logits.shape or logits.shape[-1] != 6:
        raise ValueError("calibration logits and masks must be [N, 6]")
    if not torch.isfinite(logits).all():
        raise ValueError("calibration requires finite logits")
    # Calibration samples the same advisory masks as collection.  Preserve the
    # normal-action fallback in the saved bank so a boxed-in row cannot make a
    # production preparation fail after collection would have proceeded.
    legal = canonical_legal_mask(legal)
    lo, hi = 1e-3, 1.0
    while _departure_probability(logits, legal, hi) < target and hi < 1e4:
        hi *= 2.0
    if _departure_probability(logits, legal, hi) < target:
        raise PreparationFailure("calibration target is unreachable for this B8 state bank")
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if _departure_probability(logits, legal, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _install_temperature(actor: B8Actor, temperature: float) -> None:
    """Restore B8's exact advantage head before applying the once-frozen scale."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise PreparationFailure("invalid calibrated temperature")
    with torch.no_grad():
        actor.adv.weight.copy_(actor.reference.adv.weight / temperature)
        actor.adv.bias.copy_(actor.reference.adv.bias / temperature)
        actor.temperature.fill_(float(temperature))


def _calibration_bin(length: int) -> int:
    """Frozen TRAIN-01 support bins: 0-199, ..., 600-799, and >=800."""
    return min(max(length, 0) // 200, 4)


def _progress(
    stage: str, rows: int, limit: int, callback: Callable[[str, Mapping[str, int]], None] | None
) -> None:
    if callback is not None:
        callback(stage, {"rows": rows, "limit": limit})
    else:
        print(f"TRAIN01 {stage}: {rows}/{limit} rows", flush=True)


@torch.no_grad()
def calibrate(
    task_factory: TaskFactory,
    actor: B8Actor,
    config: PreparationConfig,
    generator: torch.Generator,
    on_progress: Callable[[str, Mapping[str, int]], None] | None = None,
) -> tuple[float, int]:
    """Collect fresh length-stratified greedy B8 states then freeze temperature."""
    del generator  # Greedy collection has no stochastic policy draw.
    task = task_factory("calibration")
    if int(task.age.numel()) != config.prep_worlds:
        raise PreparationFailure(
            "calibration task world count differs from frozen preparation cohort"
        )
    bins: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {index: [] for index in range(5)}
    rows = 0
    next_progress = config.progress_rows
    while min(len(values) for values in bins.values()) < config.calibration_quota_per_bin:
        if rows >= config.calibration_max_rows:
            raise PreparationFailure(
                "calibration state cap reached before all five frozen length-bin quotas"
            )
        local, glob, scalars, legal = task.observe()
        logits = actor.reference_logits(local, glob, scalars)
        legal = canonical_legal_mask(legal)
        length = _lengths(task, logits.shape[0], logits.device).long().clamp_min(1)
        cpu_length = length.cpu()
        # Greedy action contains no artificial terminal/cut; task reset remains natural.
        actions = categorical(logits, legal).probs.argmax(-1)
        for index in range(logits.shape[0]):
            bucket = _calibration_bin(int(cpu_length[index]))
            if len(bins[bucket]) < config.calibration_quota_per_bin:
                bins[bucket].append((logits[index].detach().cpu(), legal[index].detach().cpu()))
            rows += 1
            if rows >= config.calibration_max_rows:
                break
        task.step(actions)
        if rows >= next_progress:
            _progress("calibration", rows, config.calibration_max_rows, on_progress)
            next_progress += config.progress_rows
    chosen = [row for bucket in range(5) for row in bins[bucket]]
    if len(chosen) != config.calibration_rows:
        raise PreparationFailure("calibration quota accounting differs from frozen 10k state bank")
    logits = torch.stack([row[0] for row in chosen])
    legal = torch.stack([row[1] for row in chosen])
    return solve_temperature(logits, legal, config.calibration_departure), len(chosen)


def _batch_rows(rows: list[_Row], device: torch.device) -> tuple[torch.Tensor, ...]:
    local = torch.cat([row.observation[0] for row in rows]).to(device, non_blocking=True)
    glob = torch.cat([row.observation[1] for row in rows]).to(device, non_blocking=True)
    scalars = torch.cat([row.observation[2] for row in rows]).to(device, non_blocking=True)
    clock = torch.tensor([row.clock for row in rows], device=device)
    targets = torch.tensor([row.reward for row in rows], device=device)
    return local, glob, scalars, clock, targets


def _fit_life(
    life: _Life,
    critic: ClockedPopArtCritic,
    optimizer: torch.optim.Optimizer,
    config: PreparationConfig,
) -> None:
    if not life.rows:
        return
    training = _suffix_rows(life)
    device = next(critic.parameters()).device
    for offset in range(0, len(training), config.critic_batch):
        local, glob, scalars, clock, value = _batch_rows(
            training[offset : offset + config.critic_batch], device
        )
        critic.update_targets(value, optimizer)
        prediction = critic(local, glob, scalars, clock)
        loss = (critic.normalise(prediction) - critic.normalise(value)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite critic preparation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()


def _suffix_rows(life: _Life) -> list[_Row]:
    """Copy a completed life with exact gamma-one mass-to-go targets."""
    suffix, output = 0.0, []
    for row in reversed(life.rows):
        suffix += row.reward
        target = copy.copy(row)
        target.reward = suffix
        output.append(target)
    output.reverse()
    return output


def _evaluate(rows: list[_MetricRow]) -> dict[str, float]:
    if not rows:
        return {"mse": math.inf, "baseline_mse": math.inf, "late_ev": -math.inf}
    if any(row.prediction is None for row in rows):
        raise PreparationFailure("held-out quality record is missing frozen-critic prediction")
    predicted = torch.tensor([float(row.prediction) for row in rows])
    target = torch.tensor([row.target for row in rows])
    mse = float((predicted - target).square().mean())
    late = torch.tensor([row.length >= 800 for row in rows])
    late_target, late_prediction = target[late], predicted[late]
    variance = float(late_target.var(unbiased=False)) if late_target.numel() else 0.0
    late_ev = (
        -math.inf
        if variance <= 0
        else 1.0 - float((late_target - late_prediction).square().mean()) / variance
    )
    return {"mse": mse, "late_ev": late_ev, "baseline_mse": math.inf}


def _fit_baseline(rows: list[_MetricRow]) -> tuple[torch.Tensor, float]:
    if len(rows) < 3:
        return torch.zeros(3), math.inf
    x = torch.tensor([[1.0, row.clock, row.length] for row in rows], dtype=torch.float64)
    y = torch.tensor([row.target for row in rows], dtype=torch.float64)
    coefficient = torch.linalg.lstsq(x, y).solution
    return coefficient.float(), float(
        ((x.float() @ coefficient.float() - y.float()).square().mean())
    )


def prepare(
    task_factory: TaskFactory,
    actor: B8Actor,
    critic: ClockedPopArtCritic,
    config: PreparationConfig = PreparationConfig(),
    generator: torch.Generator | None = None,
    state: PreparationState | Mapping[str, Any] | None = None,
    on_calibrated: (
        Callable[[PreparationState, B8Actor, ClockedPopArtCritic, torch.optim.Optimizer], None]
        | None
    ) = None,
    on_progress: Callable[[str, Mapping[str, int]], None] | None = None,
) -> PreparationResult:
    """Run finite calibration then world-disjoint complete-life critic preparation.

    The returned state is saved at calibrated and critic-ready boundaries by
    the durable runner.  A failed support/quality endpoint raises before any
    PPO actor update is possible.
    """
    generator = generator or torch.Generator(device=next(actor.parameters()).device).manual_seed(0)
    result = (
        PreparationState.from_state_dict(state)
        if isinstance(state, Mapping)
        else (state or PreparationState())
    )
    if result.stage == "ready":
        return PreparationResult(
            result, actor, critic, torch.optim.Adam(critic.parameters(), lr=1e-4)
        )
    newly_calibrated = result.temperature is None
    if newly_calibrated:
        temperature, rows = calibrate(task_factory, actor, config, generator, on_progress)
        _install_temperature(actor, temperature)
        result.temperature, result.calibration_rows, result.stage = temperature, rows, "calibrated"
    else:
        _install_temperature(actor, result.temperature)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    if newly_calibrated and on_calibrated is not None:
        # This is the immutable pre-actor boundary.  No critic life has been
        # sampled yet, so the runner can publish it before entering refit.
        on_calibrated(result, actor, critic, optimizer)
    # Fit and held-out collection are separate fresh task instances.  This lets
    # us discard holdout observations before the critic is frozen, then retain
    # only final frozen-critic predictions/targets for quality.  Whole world
    # slots remain disjoint across the two passes.
    task = task_factory("refit")
    worlds = int(task.age.numel())
    if worlds != config.prep_worlds:
        raise PreparationFailure(
            f"refit task E{worlds} differs from frozen preparation E{config.prep_worlds}"
        )
    lives = [_Life() for _ in range(worlds)]
    fit_metrics: list[_MetricRow] = []
    next_progress = config.progress_rows
    while result.fit_rows < config.critic_min_fit_rows:
        # The finite cap charges every simulated world, including the holdout
        # slots discarded during fit.  E24 cannot consume the final eight rows
        # of 8,388,608, so its immutable effective endpoint is 8,388,600.
        if result.critic_rows + worlds > config.critic_max_rows:
            break
        local, glob, scalars, legal = task.observe()
        with torch.no_grad():
            logits = actor.reference_logits(local, glob, scalars)
            # Categorical.sample() has no generator argument.  Explicit
            # multinomial makes this pre-actor stochastic B8 bank resumable.
            action = torch.multinomial(
                categorical(logits, legal).probs, 1, generator=generator
            ).squeeze(-1)
        length = _lengths(task, worlds, logits.device)
        clock = (task.config.horizon - task.age).clamp_min(0).float() / task.config.horizon
        step = task.step(action)
        result.critic_rows += worlds
        rewards = step.reward.detach().float()
        completed = getattr(step, "reset", None)
        if completed is None:
            completed = step.terminal | step.cut
        # One CPU transfer per tensor per tick.  Never synchronize/copy once
        # per world; completed rows are staged only until their natural life ends.
        cpu_observation = tuple(value.detach().cpu() for value in (local, glob, scalars))
        cpu_length, cpu_clock, cpu_reward = length.cpu(), clock.cpu(), rewards.cpu()
        cpu_completed, cpu_mix = completed.cpu(), task.roster_id.cpu()
        for index in range(worlds):
            if _world_split(index, config) != "fit":
                continue
            row = _Row(
                tuple(value[index : index + 1] for value in cpu_observation),
                float(cpu_length[index]),
                float(cpu_clock[index]),
                float(cpu_reward[index]),
                int(cpu_mix[index]),
            )
            lives[index].rows.append(row)
            lives[index].total += row.reward
            if sum(len(life.rows) for life in lives) > config.max_open_rows:
                result.stage, result.task_state = "failed", task.state_dict()
                raise PreparationFailure(
                    "open complete-life trajectory storage cap exceeded", result
                )
            if bool(cpu_completed[index]):
                _fit_life(lives[index], critic, optimizer, config)
                completed_rows = _suffix_rows(lives[index])
                result.fit_rows += len(completed_rows)
                fit_metrics.extend(
                    _MetricRow(row.length, row.clock, row.reward) for row in completed_rows
                )
                if len(fit_metrics) > config.metric_cache_rows:
                    del fit_metrics[: len(fit_metrics) - config.metric_cache_rows]
                lives[index] = _Life()
        if result.critic_rows >= next_progress:
            _progress("critic-fit", result.critic_rows, config.critic_max_rows, on_progress)
            next_progress += config.progress_rows

    if result.fit_rows < config.critic_min_fit_rows:
        result.stage, result.task_state = "failed", task.state_dict()
        raise PreparationFailure(
            "critic fit row cap reached before fixed 2M completed-life rows", result
        )

    baseline_coef, _ = _fit_baseline(fit_metrics)
    # Freeze the learned critic before fresh holdout worlds.  The stored
    # validation records below contain only scalar targets and predictions.
    critic.eval()
    task = task_factory("refit")
    lives = [_Life() for _ in range(worlds)]
    holdout_metrics: list[_MetricRow] = []

    def support_complete() -> bool:
        return (
            result.completed_holdout_lives >= config.min_holdout_lives
            and result.late_holdout_rows >= config.min_holdout_late_rows
            and min(result.holdout_lives_by_mix.values(), default=0)
            >= config.min_holdout_lives_per_mix
        )

    while not support_complete():
        if result.critic_rows + worlds > config.critic_max_rows:
            break
        local, glob, scalars, legal = task.observe()
        with torch.no_grad():
            logits = actor.reference_logits(local, glob, scalars)
            action = torch.multinomial(
                categorical(logits, legal).probs, 1, generator=generator
            ).squeeze(-1)
        length = _lengths(task, worlds, logits.device)
        clock = (task.config.horizon - task.age).clamp_min(0).float() / task.config.horizon
        step = task.step(action)
        result.critic_rows += worlds
        completed = getattr(step, "reset", step.terminal | step.cut)
        rewards = step.reward.detach().float()
        cpu_observation = tuple(value.detach().cpu() for value in (local, glob, scalars))
        cpu_length, cpu_clock, cpu_reward = length.cpu(), clock.cpu(), rewards.cpu()
        cpu_completed, cpu_mix = completed.cpu(), task.roster_id.cpu()
        for index in range(worlds):
            if _world_split(index, config) != "holdout":
                continue
            lives[index].rows.append(
                _Row(
                    tuple(value[index : index + 1] for value in cpu_observation),
                    float(cpu_length[index]),
                    float(cpu_clock[index]),
                    float(cpu_reward[index]),
                    int(cpu_mix[index]),
                )
            )
            if sum(len(life.rows) for life in lives) > config.max_open_rows:
                result.stage, result.task_state = "failed", task.state_dict()
                raise PreparationFailure("open held-out life storage cap exceeded", result)
            if bool(cpu_completed[index]):
                completed_rows = _suffix_rows(lives[index])
                with torch.no_grad():
                    for offset in range(0, len(completed_rows), config.critic_batch):
                        local_b, glob_b, scalars_b, clock_b, _ = _batch_rows(
                            completed_rows[offset : offset + config.critic_batch],
                            next(critic.parameters()).device,
                        )
                        prediction = critic(local_b, glob_b, scalars_b, clock_b).cpu().tolist()
                        holdout_metrics.extend(
                            _MetricRow(row.length, row.clock, row.reward, value)
                            for row, value in zip(
                                completed_rows[offset : offset + config.critic_batch], prediction
                            )
                        )
                result.holdout_rows += len(completed_rows)
                result.completed_holdout_lives += 1
                mix = _mix_name(completed_rows[0].mix)
                result.holdout_lives_by_mix[mix] = result.holdout_lives_by_mix.get(mix, 0) + 1
                result.late_holdout_rows += sum(
                    row.length >= config.late_length for row in completed_rows
                )
                lives[index] = _Life()
        if result.critic_rows >= next_progress:
            _progress("critic-holdout", result.critic_rows, config.critic_max_rows, on_progress)
            next_progress += config.progress_rows

    metrics = _evaluate(holdout_metrics)
    if holdout_metrics and torch.isfinite(baseline_coef).all():
        x = torch.tensor([[1.0, row.clock, row.length] for row in holdout_metrics])
        y = torch.tensor([row.target for row in holdout_metrics])
        metrics["baseline_mse"] = float((x @ baseline_coef - y).square().mean())
    result.quality = metrics
    support = support_complete()
    quality = metrics["late_ev"] >= config.min_late_ev and metrics["mse"] < metrics["baseline_mse"]
    if result.fit_rows < config.critic_min_fit_rows or not support or not quality:
        result.stage = "failed"
        raise PreparationFailure(
            "critic preparation endpoint failed: "
            f"fit={result.fit_rows}, holdout_lives={result.completed_holdout_lives}, "
            f"quality={metrics}",
            result,
        )
    result.stage, result.task_state = "ready", None
    return PreparationResult(result, actor, critic, optimizer)
