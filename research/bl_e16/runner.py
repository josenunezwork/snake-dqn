"""E16's finite-bank, natural-start PPO execution harness.

This module owns no admission decision and performs no Colab operation.  It is
deliberately a small synchronous driver so that every reset identity is visible
in the exported collector state.  Production runs require a bank audit and an
externally validated plan; ``smoke`` is an explicitly non-scientific exercise
of the same actor, critic, simulator, and export path.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import hashlib
import json
import random
import signal
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.distributions import Categorical

from research.bl_e16 import bank
from research.bl_e16.model import E16Policy
from research.bl_e16.spool import StreamingLife
from research.bl_e16.training import E16Batch, E16Trainer, E16TripwireError
from research.bl_ppo.exports import export_json, export_torch, verify_export

HORIZON = 5_000
CRITIC_TRANSITIONS = 2_000_000
ACTOR_TRANSITIONS = 4_000_000
SCHEMA = "bl-e16-runner/v1"


class AdmissionRequired(RuntimeError):
    """The required pre-data plan or seed audit is absent or invalid."""


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_audited_bank(path: Path, audit_path: Path) -> dict[str, Any]:
    """Load exactly the bank and audit that authorize real natural resets."""
    verify_export(path)
    verify_export(audit_path)
    value = json.loads(path.read_text())
    receipt = json.loads(audit_path.read_text())
    bank.check_audit(value, receipt)
    return value


def require_plan(path: Path | None, arm: str) -> dict[str, Any]:
    """Call the package-local admission seam once it has been committed.

    The admission module is intentionally authored by the registration owner;
    importing it here avoids a second, divergent implementation of the gate.
    """
    if path is None:
        raise AdmissionRequired("E16 train requires a validated external --plan")
    try:
        from research.bl_e16 import admission
    except ImportError as error:
        raise AdmissionRequired("E16 admission module is not available yet") from error
    validator = getattr(admission, "require_execution_plan", None)
    if not callable(validator):
        raise AdmissionRequired("E16 admission module lacks require_execution_plan(path, arm)")
    result = validator(path, arm)
    if not isinstance(result, Mapping):
        raise AdmissionRequired("E16 admission returned no validated plan")
    return dict(result)


@functools.lru_cache(maxsize=4)
def _phase_rows(phase: str, round_index: int) -> dict[str, list[dict[str, Any]]]:
    rows = bank._rows(phase, round_index)
    return {mix: [row for row in rows if row["mix"] == mix] for mix in bank.MIX_COUNTS}


def _row_for(bank_value: Mapping[str, Any], phase: str, mix: str, ordinal: int) -> dict[str, Any]:
    """Bind a finite training identity to its fixed natural opponent mix."""
    candidates = _phase_rows(phase, int(bank_value["round_index"]))[mix]
    if not candidates:
        raise AdmissionRequired(f"bank has no {mix} roster row")
    if ordinal >= len(candidates):
        raise AdmissionRequired(f"finite {phase}/{mix} bank exhausted")
    return dict(candidates[ordinal])


def _geometry(sim: Any) -> np.ndarray:
    # E16-G must use the exact E3 geometry extractor, not a nearby B8 raster.
    from research.bl_info_value.reconstruct import _geometry as reconstruct_geometry

    return reconstruct_geometry(sim)


def _history_state(sim: Any, actions: np.ndarray | None = None) -> np.ndarray:
    from research.bl_info_value.reconstruct import _history_state as reconstruct_history

    return reconstruct_history(sim, actions)


class _E16TerminalDriver:
    """Terminal natural world with E16's mechanics-v2 registration binding."""

    def __init__(self, seed: int, roster: Any, horizon: int = HORIZON) -> None:
        from research.bl_ppo.collector import _FrozenRoster, _prepare_sim, _sim_config
        from src.simd_env.grid_sim import GridBatchSim

        self.seed, self.horizon, self.frame = int(seed), int(horizon), 0
        self._prepare = _prepare_sim
        config = dataclasses.replace(_sim_config(1), mechanics_version=2)
        self.sim = GridBatchSim(config, seeds=[self.seed], train_mode=False, allow_respawn=True)
        self.frozen = _FrozenRoster(self.sim, [self.seed], [roster])
        self.last_actions: np.ndarray | None = None
        self.last_history_state: np.ndarray | None = None

    def observe_hero(self) -> dict[str, np.ndarray]:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        self._prepare(self.sim)
        rows = np.array([[0, 0]], dtype=np.int64)
        obs = build_ego_raster(self.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        mask = self.sim.get_advisory_action_mask()[0, 0].copy()
        if not mask.any():
            mask[:3] = True
        return {
            "local": obs["local"],
            "global": obs["global"],
            "scalars": obs["scalars"],
            "mask": mask[None, :],
        }

    def step(self, action: int) -> dict[str, Any]:
        if self.frame >= self.horizon or not self.sim.get_alive()[0, 0]:
            raise RuntimeError("terminal E16 driver is finished")
        actions = np.ones((1, self.sim.S), dtype=np.int64)
        actions[0, 0] = int(action)
        self.frozen.actions(self.sim, actions)
        self.last_actions = actions.copy()
        self.last_history_state = _history_state(self.sim, actions)
        self.sim.step(actions)
        self.frozen.observe_step(self.sim)
        self.frame += 1
        alive = bool(self.sim.get_alive()[0, 0])
        return {
            "done": not alive or self.frame >= self.horizon,
            "mass": float(self.sim.length[0, 0]) if alive else 0.0,
            "alive": alive,
            "frame": self.frame,
        }

    def snapshot(self) -> dict[str, Any]:
        """Serialize the live simulator and opponent carry at a rollout boundary."""
        saved = copy.copy(self.sim)
        kernel = getattr(saved, "_kernels", None)
        if kernel is not None:
            saved._kernels = None
        return {
            "seed": self.seed,
            "horizon": self.horizon,
            "frame": self.frame,
            "sim": saved,
            "kernel_module": getattr(kernel, "__name__", None),
            "frozen": self.frozen,
            "last_actions": self.last_actions,
            "last_history_state": self.last_history_state,
        }

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> "_E16TerminalDriver":
        """Restore a previously snapshotted live world without resetting its life."""
        driver = cls.__new__(cls)
        driver.seed, driver.horizon, driver.frame = (
            int(state["seed"]),
            int(state["horizon"]),
            int(state["frame"]),
        )
        from research.bl_ppo.collector import _prepare_sim

        driver._prepare, driver.sim, driver.frozen = _prepare_sim, state["sim"], state["frozen"]
        if state.get("kernel_module"):
            import importlib

            driver.sim._kernels = importlib.import_module(state["kernel_module"])
        # NumPy pickle serializes views as separate arrays. Reattach the three
        # occupancy interiors to padded storage before any compiled featurizer
        # or dynamics call; otherwise restored observations use stale padding.
        sim = driver.sim
        pad = int(sim._grid_pad)
        for name in ("owner", "slot", "food"):
            stored = getattr(sim, f"_{name}")
            view = getattr(sim, f"_{name}_pad")[:, pad : pad + sim.grid_h, pad : pad + sim.grid_w]
            if not np.array_equal(stored, view):
                raise AdmissionRequired("checkpoint grid interior/padding identity differs")
            setattr(sim, f"_{name}", view)
        driver.last_actions = state.get("last_actions")
        driver.last_history_state = state.get("last_history_state")
        return driver


class NaturalLifeCollector:
    """One persistent natural life; rollover is allowed only after terminal H5000/death.

    The caller owns one instance per phase.  Its ordinal advances only at an
    actual terminal reset, which makes A/G's frozen phase vectors pairable.
    """

    def __init__(self, bank_value: Mapping[str, Any], phase: str) -> None:
        if phase not in {"critic", "actor", "preflight"}:
            raise ValueError("invalid E16 phase")
        self.bank, self.phase, self.ordinal = dict(bank_value), phase, 0
        self.driver: _E16TerminalDriver | None = None
        self.history: deque[np.ndarray] = deque(maxlen=128)
        self.used_seeds: list[int] = []
        self.pending_spool: StreamingLife | None = None
        self.pending_spool_state: dict[str, Any] | None = None
        self.fit_offset = 0

    def _start(self) -> None:
        if self.phase == "preflight":
            rows = self.bank["excluded_preflight_rows"]
            if self.ordinal >= len(rows):
                raise AdmissionRequired("preflight bank exhausted")
            row = dict(rows[self.ordinal])
        else:
            mix = tuple(bank.MIX_COUNTS)[self.ordinal % 3]
            per_mix = self.ordinal // 3
            row = _row_for(self.bank, self.phase, mix, per_mix)
        self.driver = _E16TerminalDriver(int(row["world_seed"]), row["opponents"], HORIZON)
        self.history.clear()
        self.used_seeds.append(int(row["world_seed"]))

    def ensure(self) -> _E16TerminalDriver:
        if self.driver is None:
            self._start()
        assert self.driver is not None
        return self.driver

    def terminal_reset(self) -> None:
        if self.driver is None:
            raise RuntimeError("cannot reset an absent E16 life")
        self.driver = None
        self.ordinal += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending_spool": (
                self.pending_spool.snapshot()
                if self.pending_spool is not None
                else self.pending_spool_state
            ),
            "fit_offset": self.fit_offset,
            "phase": self.phase,
            "ordinal": self.ordinal,
            "driver": None if self.driver is None else self.driver.snapshot(),
            "history": list(self.history),
            "used_seeds": list(self.used_seeds),
        }

    @classmethod
    def restore(
        cls, bank_value: Mapping[str, Any], state: Mapping[str, Any]
    ) -> "NaturalLifeCollector":
        collector = cls(bank_value, str(state["phase"]))
        collector.ordinal = int(state["ordinal"])
        collector.pending_spool_state = state.get("pending_spool")
        collector.fit_offset = int(state.get("fit_offset", 0))
        collector.history = deque(
            (np.asarray(x, dtype=np.float32) for x in state["history"]), maxlen=128
        )
        collector.used_seeds = [int(x) for x in state["used_seeds"]]
        collector.driver = (
            None if state["driver"] is None else _E16TerminalDriver.restore(state["driver"])
        )
        return collector


def _act(
    policy: E16Policy,
    obs: Mapping[str, np.ndarray],
    sim: Any,
    history: deque[np.ndarray],
    device: torch.device,
) -> tuple[int, dict[str, torch.Tensor]]:
    local = torch.as_tensor(obs["local"], device=device)
    global_ = torch.as_tensor(obs["global"], device=device)
    scalars = torch.as_tensor(obs["scalars"], device=device)
    mask = torch.as_tensor(obs["mask"], device=device, dtype=torch.bool)
    geometry = None
    history_tensor = None
    if policy.arm == "G":
        geometry = torch.as_tensor(_geometry(sim)[None], device=device)
        if policy.use_history:
            window = np.zeros((128, 43), dtype=np.float32)
            values = list(history)[-128:]
            if values:
                window[-len(values) :] = np.stack(values)
            history_tensor = torch.as_tensor(window[None], device=device)
    with torch.no_grad():
        logits = policy.actor_logits(
            local, global_, scalars, geometry=geometry, history=history_tensor, action_mask=mask
        )
        if not torch.isfinite(logits[mask]).all():
            raise E16TripwireError("nonfinite legal collection logits")
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        logp = distribution.log_prob(action)
        value = policy.value(
            local,
            global_,
            scalars,
            torch.tensor([sim.frame[0]], device=device),
            geometry=geometry,
            history=history_tensor,
        )
    if not torch.isfinite(value).all():
        raise E16TripwireError("nonfinite collection value")
    return int(action.item()), {
        "local": local.cpu(),
        "global_": global_.cpu(),
        "scalars": scalars.cpu(),
        "mask": mask.cpu(),
        "action": action.cpu(),
        "logp": logp.cpu(),
        "value": value.cpu(),
        "geometry": None if geometry is None else geometry.cpu(),
        "history": None if history_tensor is None else history_tensor.cpu(),
    }


def collect_natural_batch(
    policy: E16Policy,
    bank_value: Mapping[str, Any],
    *,
    phase: str,
    cursor: int,
    steps: int,
    device: str = "cpu",
    collector: NaturalLifeCollector | None = None,
) -> tuple[E16Batch, int, list[int]]:
    """Collect one real natural-start terminal life slice from frozen bank identities.

    A reset is never generated: the cursor maps to a finite phase vector.  A/G
    callers pass the same phase/cursor to obtain identical trajectory roots.
    """
    if phase not in {"critic", "actor", "preflight"} or steps < 1:
        raise ValueError("invalid E16 collection phase or step count")
    if collector is not None:
        if collector.phase != phase:
            raise ValueError("persistent collector phase differs from requested phase")
        driver, history = collector.ensure(), collector.history
        seed = driver.seed
    elif phase == "preflight":
        rows = bank_value["excluded_preflight_rows"]
        if cursor >= len(rows):
            raise AdmissionRequired("preflight bank exhausted")
        row = rows[cursor]
        seed = int(row["world_seed"])
    else:
        mix = tuple(bank.MIX_COUNTS)[cursor % 3]
        ordinal = cursor // 3
        values = bank_value["phase_seeds"][phase][mix]
        if ordinal >= len(values):
            raise AdmissionRequired(f"finite {phase}/{mix} bank exhausted")
        seed = int(values[ordinal])
        row = _row_for(bank_value, phase, mix, ordinal)
    if collector is None:
        driver = _E16TerminalDriver(seed, row["opponents"], horizon=HORIZON)
        history = deque(maxlen=128)
    records: list[dict[str, torch.Tensor | None]] = []
    rewards: list[float] = []
    dones: list[bool] = []
    frames: list[int] = []
    collect_steps = HORIZON if phase == "critic" else steps
    for _ in range(collect_steps):
        obs = driver.observe_hero()
        action, record = _act(policy, obs, driver.sim, history, torch.device(device))
        result = driver.step(action)
        if driver.last_history_state is None:
            raise RuntimeError("E16 driver did not capture the pre-transition joint action history")
        before = driver.last_history_state
        history.append(before)
        records.append(record)
        rewards.append(float(result["mass"]) / 1000.0)
        dones.append(bool(result["done"]))
        frames.append(int(result["frame"]) - 1)
        if result["done"]:
            break
    if not records:
        raise RuntimeError("natural collector produced no actor step")
    count = len(records)

    def stack(name: str) -> torch.Tensor:
        return torch.stack([record[name] for record in records if record[name] is not None])

    geometry = None if records[0]["geometry"] is None else stack("geometry")
    history_t = None if records[0]["history"] is None else stack("history")
    # A rollout cut bootstraps from the actual next state.  Terminal death and
    # H5000 are the only zero-bootstrap cases under the registered objective.
    if dones[-1] or frames[-1] + 1 >= HORIZON:
        last_value = torch.zeros(1)
    else:
        next_obs = driver.observe_hero()
        next_local = torch.as_tensor(next_obs["local"], device=device)
        next_global = torch.as_tensor(next_obs["global"], device=device)
        next_scalars = torch.as_tensor(next_obs["scalars"], device=device)
        next_geometry = None
        next_history = None
        if policy.arm == "G":
            next_geometry = torch.as_tensor(_geometry(driver.sim)[None], device=device)
            if policy.use_history:
                window = np.zeros((128, 43), dtype=np.float32)
                values = list(history)[-128:]
                if values:
                    window[-len(values) :] = np.stack(values)
                next_history = torch.as_tensor(window[None], device=device)
        with torch.no_grad():
            last_value = policy.value(
                next_local,
                next_global,
                next_scalars,
                torch.tensor([driver.frame], device=device),
                geometry=next_geometry,
                history=next_history,
            ).cpu()
    critic_targets = None
    if phase == "critic":
        returns = torch.tensor(rewards, dtype=torch.float32)
        critic_targets = torch.flip(torch.cumsum(torch.flip(returns, dims=(0,)), dim=0), dims=(0,))
        critic_targets = critic_targets.view(count, 1)
    batch = E16Batch(
        local=stack("local"),
        global_=stack("global_"),
        scalars=stack("scalars"),
        frame=torch.tensor(frames).view(count, 1),
        actions=stack("action").view(count, 1),
        logp=stack("logp").view(count, 1),
        rewards=torch.tensor(rewards).view(count, 1),
        dones=torch.tensor(dones).view(count, 1),
        values=stack("value").view(count, 1),
        last_value=last_value,
        action_masks=stack("mask"),
        geometry=geometry,
        history=history_t,
        horizon_cutoff=torch.tensor([frame + 1 >= HORIZON for frame in frames]).view(count, 1),
        critic_targets=critic_targets,
    )
    terminal = bool(dones[-1])
    if collector is not None and terminal:
        collector.terminal_reset()
    return batch, (collector.ordinal if collector is not None else cursor + 1), [seed]


def _snapshot(
    policy: E16Policy,
    trainer: E16Trainer,
    counters: Mapping[str, Any],
    identity: Mapping[str, Any],
    collectors: Mapping[str, NaturalLifeCollector],
) -> dict[str, Any]:
    """Complete boundary state; a run is checkpointed only between life slices."""
    return {
        "schema": SCHEMA,
        "identity": dict(identity),
        "policy": policy.portable_state_dict(),
        "trainer": {
            "actor": trainer.actor_optimizer.state_dict(),
            "critic": trainer.critic_optimizer.state_dict(),
            "refit_transitions": trainer.refit_transitions,
            "e12_passed": trainer.e12_passed,
        },
        "collector": {name: collector.snapshot() for name, collector in collectors.items()},
        "counters": dict(counters),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def _slice_batch(batch: E16Batch, count: int) -> E16Batch:
    """Take a prefix only after a complete critic life supplied its exact targets."""
    if not 1 <= count <= batch.actions.shape[0]:
        raise ValueError("invalid complete-life batch prefix")

    def cut(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value[:count]

    return E16Batch(
        local=cut(batch.local),
        global_=cut(batch.global_),
        scalars=cut(batch.scalars),
        frame=cut(batch.frame),
        actions=cut(batch.actions),
        logp=cut(batch.logp),
        rewards=cut(batch.rewards),
        dones=cut(batch.dones),
        values=cut(batch.values),
        last_value=(batch.last_value if count == batch.actions.shape[0] else batch.values[count]),
        action_masks=cut(batch.action_masks),
        geometry=cut(batch.geometry),
        history=cut(batch.history),
        horizon_cutoff=cut(batch.horizon_cutoff),
        critic_targets=cut(batch.critic_targets),
    )


def _require_arm_critic_qualification(path: Path | None, arm: str, critic_sha: str) -> None:
    """E12 on F does not qualify the newly fitted E16 rich critic."""
    if path is None:
        raise AdmissionRequired(
            "E16 actor block requires a hash-bound per-arm critic qualification"
        )
    verify_export(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "bl-e16-critic-qualification/v1"
        or value.get("arm") != arm
        or value.get("status") != "PASS"
        or value.get("critic_snapshot_sha256") != critic_sha
    ):
        raise AdmissionRequired("per-arm critic qualification is missing, mismatched, or not PASS")


def _to_device(batch: E16Batch, device: str) -> E16Batch:
    return E16Batch(
        **{
            field.name: (
                getattr(batch, field.name).to(device)
                if isinstance(getattr(batch, field.name), torch.Tensor)
                else getattr(batch, field.name)
            )
            for field in dataclasses.fields(batch)
        }
    )


def _observe_tripwires(
    policy: E16Policy, batch: E16Batch, windows: Any, device: str
) -> list[dict[str, Any]]:
    with torch.no_grad():
        baseline = policy.actor.teacher(
            batch.local.reshape(-1, *batch.local.shape[2:]).to(device),
            batch.global_.reshape(-1, *batch.global_.shape[2:]).to(device),
            batch.scalars.reshape(-1, batch.scalars.shape[-1]).to(device),
        )
        baseline = baseline.masked_fill(
            ~batch.action_masks.reshape(-1, 6).to(device), float("-inf")
        ).argmax(dim=-1)
    return windows.observe(batch.actions.reshape(-1).cpu().numpy(), baseline.cpu().numpy())


def _collect_critic_life(
    policy: E16Policy, collector: NaturalLifeCollector, directory: Path, device: str, progress: Any
) -> StreamingLife:
    if collector.pending_spool is None:
        collector.pending_spool = (
            StreamingLife.restore(directory, collector.pending_spool_state)
            if collector.pending_spool_state is not None
            else StreamingLife(directory)
        )
        collector.pending_spool_state = None
    spool = collector.pending_spool
    if spool.closed:
        return spool
    driver = collector.ensure()
    for _ in range(HORIZON):
        obs = driver.observe_hero()
        action, record = _act(policy, obs, driver.sim, collector.history, torch.device(device))
        result = driver.step(action)
        collector.history.append(driver.last_history_state)
        spool.append(record, float(result["mass"]) / 1000.0, result["done"], result["frame"] - 1)
        if result["done"]:
            collector.terminal_reset()
            spool.finish()
            progress()
            return spool
        progress()
    raise RuntimeError("critic life failed to terminate at H5000")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run audited A/G; scientific preparation cannot bypass its frozen receipts."""
    from research.bl_e16.admission import B8_SHA256
    from research.bl_ppo.tripwires import ActionWindows

    if args.arm not in {"A", "G"}:
        raise AdmissionRequired("R/S/P remain UNRANKED implementation stubs")
    smoke = args.command == "smoke"
    plan = {"mode": "non-scientific-smoke"} if smoke else require_plan(args.plan, args.arm)
    audited = load_audited_bank(args.bank, args.seed_audit)
    if not smoke and (audited != plan["_bank"] or _sha(args.parent) != plan["parent"]["sha256"]):
        raise AdmissionRequired("runtime bank/parent differs from frozen plan")
    if _sha(args.b8_checkpoint) != B8_SHA256:
        raise AdmissionRequired("immutable B8 identity required")
    if not smoke and (
        args.transitions != ACTOR_TRANSITIONS or args.critic_transitions != CRITIC_TRANSITIONS
    ):
        raise AdmissionRequired("fixed 2M critic + 4M actor blocks cannot be resized")
    if not 2 <= args.steps <= 64 or not 1 <= args.rollout_steps <= 32:
        raise AdmissionRequired("tiny smoke needs 2..64 steps; bounded rollouts use 1..32")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    synthetic_parent = smoke and args.parent is None
    history = args.arm == "G" and (smoke or plan["use_history"])
    if synthetic_parent:
        from research.bl_ppo.model import PPOPolicy

        policy = E16Policy(
            args.arm, PPOPolicy("F", args.b8_checkpoint, temperature=1.0), use_history=history
        ).to(args.device)
        parent_sha = "synthetic-F:" + B8_SHA256
    else:
        policy = E16Policy.from_parent_checkpoint(
            args.arm, args.parent, use_history=history, device=args.device
        )
        parent_sha = _sha(args.parent)
    critic_target = 1 if smoke else CRITIC_TRANSITIONS
    actor_target = args.steps - 1 if smoke else ACTOR_TRANSITIONS
    trainer = E16Trainer(policy, critic_refit_transitions=critic_target, history_qualified=history)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema": SCHEMA,
        "arm": args.arm,
        "bank_sha256": _sha(args.bank),
        "audit_sha256": _sha(args.seed_audit),
        "parent_sha256": parent_sha,
        "synthetic_parent": synthetic_parent,
        "b8_sha256": B8_SHA256,
        "scientific": not smoke,
        "plan_sha256": None if smoke else _sha(args.plan),
    }
    counters = {
        "critic_transitions": 0,
        "actor_transitions": 0,
        "updates": 0,
        "phase_cursors": {"critic": 0, "actor": 0, "preflight": 0},
        "used_seeds": [],
        "progress_seconds": 0.0,
        "allocation_seconds": 0.0,
    }
    collectors = {
        name: NaturalLifeCollector(audited, name) for name in ("critic", "actor", "preflight")
    }
    windows, tripwires, status, critic_qualified = ActionWindows(), [], "INCOMPLETE", False
    # Model initialization consumes different RNG amounts for A/G. Reset the
    # sampling stream AFTER construction so their frozen-parent critic lives match.
    torch.manual_seed(args.seed)
    if args.resume:
        from research.bl_e16.checkpoint import restore_snapshot

        verify_export(args.resume)
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if previous["identity"] != identity:
            raise AdmissionRequired("resume identity differs from source/parent/bank/plan")
        counters = restore_snapshot(policy, trainer, collectors, previous)
        windows = ActionWindows(**previous["action_windows"])
        tripwires = previous.get("tripwires", [])
        critic_qualified = previous.get("critic_qualified", False)
    started = reported = checkpointed = time.monotonic()
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        number: signal.signal(number, request_stop) for number in (signal.SIGTERM, signal.SIGINT)
    }

    def save_boundary() -> None:
        boundary_counters = {
            **counters,
            "allocation_seconds": counters["allocation_seconds"] + time.monotonic() - started,
        }
        snapshot = _snapshot(policy, trainer, boundary_counters, identity, collectors)
        snapshot.update(
            action_windows=dataclasses.asdict(windows),
            tripwires=tripwires,
            critic_qualified=critic_qualified,
        )
        export_torch(
            args.output
            / f"boundary-{counters['critic_transitions']}-{counters['actor_transitions']}.pt",
            snapshot,
        )

    def progress() -> None:
        nonlocal reported
        now = time.monotonic()
        elapsed = counters["allocation_seconds"] + now - started
        if not smoke and elapsed >= min(
            10 * 3600, plan["_reservations"][args.arm]["timeout_minutes"] * 60
        ):
            raise AdmissionRequired("first total wall/unit cap reached")
        if now - reported >= 120:
            print(
                json.dumps(
                    {
                        "done": counters["critic_transitions"] + counters["actor_transitions"],
                        "total": critic_target + actor_target,
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
            )
            reported = now

    try:
        while (
            counters["critic_transitions"] < critic_target
            or counters["actor_transitions"] < actor_target
        ):
            progress()
            if time.monotonic() - checkpointed >= 120:
                save_boundary()
                checkpointed = time.monotonic()
            if stop_requested or (args.stop_file and args.stop_file.exists()):
                status = "INTERRUPTED"
                break
            if trainer.actor_frozen:
                if smoke:
                    batch, _, seeds = collect_natural_batch(
                        policy,
                        audited,
                        phase="preflight",
                        cursor=0,
                        steps=1,
                        device=args.device,
                        collector=collectors["preflight"],
                    )
                    # Explicitly truncated tiny qualification smoke, never scientific MC evidence.
                    batch.critic_targets = batch.rewards.clone()
                    batches = [batch]
                else:
                    collector = collectors["critic"]
                    spool = _collect_critic_life(
                        policy,
                        collector,
                        args.output / ".critic-spool" / f"life-{collector.ordinal:06d}",
                        args.device,
                        progress,
                    )
                    seeds = [collector.used_seeds[-1]]
                    batches = spool.batches()
                fitted = 0
                for batch in batches:
                    if not smoke and fitted < collector.fit_offset:
                        fitted += batch.actions.shape[0]
                        continue
                    progress()
                    remain = critic_target - counters["critic_transitions"]
                    if remain <= 0:
                        break
                    used = _slice_batch(batch, min(remain, batch.actions.shape[0]))
                    metrics = trainer.trainer_step(_to_device(used, args.device))
                    counters["critic_transitions"] += int(metrics["transitions"])
                    counters["updates"] += 1
                    if not smoke:
                        collector.fit_offset += int(metrics["transitions"])
                if not smoke:
                    import shutil

                    shutil.rmtree(spool.directory)
                    collector.pending_spool = None
                    collector.fit_offset = 0
                counters["used_seeds"].extend(seeds)
                continue
            if not trainer.e12_passed:
                ready = args.output / "critic-ready.pt"
                critic_sha = export_torch(
                    ready,
                    {
                        "critic": policy.critic.state_dict(),
                        "parent_sha256": parent_sha,
                        "arm": args.arm,
                    },
                )["sha256"]
                if not smoke:
                    _require_arm_critic_qualification(
                        args.critic_qualification, args.arm, critic_sha
                    )
                    critic_qualified = True
                trainer.mark_e12_passed()
            phase = "preflight" if smoke else "actor"
            steps = min(args.rollout_steps, actor_target - counters["actor_transitions"])
            batch, cursor, seeds = collect_natural_batch(
                policy,
                audited,
                phase=phase,
                cursor=counters["phase_cursors"][phase],
                steps=steps,
                device=args.device,
                collector=collectors[phase],
            )
            counters["phase_cursors"][phase] = cursor
            events = _observe_tripwires(policy, batch, windows, args.device)
            if events:
                tripwires.extend(events)
                status = "KILL"
                break
            metrics = trainer.trainer_step(_to_device(batch, args.device))
            counters["actor_transitions"] += int(metrics["transitions"])
            counters["updates"] += 1
            counters["used_seeds"].extend(seeds)
        else:
            status = "COMPLETE" if not smoke else "SMOKE_COMPLETE"
    except E16TripwireError as error:
        tripwires.append({"kind": "nonfinite", "reason": str(error)})
        status = "KILL"
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        counters["allocation_seconds"] += time.monotonic() - started
        payload = _snapshot(policy, trainer, counters, identity, collectors)
        payload.update(
            action_windows=dataclasses.asdict(windows),
            tripwires=tripwires,
            critic_qualified=critic_qualified,
        )
        export_torch(args.output / "full-state.pt", payload)
        endpoint = export_torch(args.output / "endpoint.pt", policy.portable_state_dict())
        value = {
            "schema": SCHEMA,
            "status": status,
            "scientific": not smoke,
            "identity": identity,
            "counters": counters,
            "critic_qualified": critic_qualified,
            "critic_transitions": counters["critic_transitions"],
            "actor_transitions": counters["actor_transitions"],
            "tripwires": tripwires,
            "endpoint_sha256": endpoint["sha256"],
            "wall_seconds": time.monotonic() - started,
        }
        export_json(args.output / "status.json", value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (sub.add_parser("smoke"), sub.add_parser("train")):
        command.add_argument("--arm", choices=("A", "G"), required=True)
        command.add_argument("--bank", type=Path, required=True)
        command.add_argument("--seed-audit", type=Path, required=True)
        command.add_argument("--b8-checkpoint", type=Path, required=True)
        command.add_argument("--parent", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=20261021)
        command.add_argument("--rollout-steps", type=int, default=32)
        command.add_argument("--stop-file", type=Path)
        command.add_argument("--plan", type=Path)
        command.add_argument("--resume", type=Path)
        command.add_argument("--critic-qualification", type=Path)
        command.add_argument("--steps", type=int, default=8)
        command.add_argument("--transitions", type=int, default=ACTOR_TRANSITIONS)
        command.add_argument("--critic-transitions", type=int, default=CRITIC_TRANSITIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    value = run(build_parser().parse_args(argv))
    print(json.dumps(value, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
