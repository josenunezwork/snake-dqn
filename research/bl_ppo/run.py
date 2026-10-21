"""E4 ruler and E2 PPO entrypoint.

This module deliberately has no dispatch code.  It is used first for the tiny
Mac operational preflight, then by the pinned L4 shell entrypoints.  All
published results are create-only exports with the supervisor sidecars.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_json, export_torch, verify_export

HORIZON = 5000
CRITIC_REFIT_TRANSITIONS = 2_000_000
SNAPSHOT_MAX_BYTES = 160 * 1024**2
E4_SCHEMA = "bl-ppo-e4-ruler/v1"
STATUS_SCHEMA = "bl-ppo-training-status/v1"


class StopRequested(RuntimeError):
    """Raised after a signal once the current complete rollout is durable."""


class PersistentPolicyAdapter:
    """GPU policy sampler that records the exact on-policy rows sent to CPU workers."""

    def __init__(self, policy: Any, device: str) -> None:
        import torch

        self.policy, self.device = policy.to(device).train(), torch.device(device)
        self.hidden = None
        self.rollout_initial_hidden = None
        self.records: list[dict[str, Any]] = []
        self.prefix: dict[str, Any] | None = None
        self.last_actions = None
        self.last_b8_actions = None
        self.last_values = None
        self.last_critic_features = None
        from research.bl_ppo.tripwires import ActionWindows

        self.action_windows = ActionWindows()

    def __call__(self, obs: dict[str, Any], reset: Any) -> Any:
        import torch

        local = torch.as_tensor(obs["local"], device=self.device)
        global_ = torch.as_tensor(obs["global"], device=self.device)
        scalars = torch.as_tensor(obs["scalars"], device=self.device)
        mask = torch.as_tensor(obs["mask"], device=self.device, dtype=torch.bool)
        reset_t = torch.as_tensor(reset, device=self.device, dtype=torch.bool)
        remaining_horizon = torch.as_tensor(
            (HORIZON - np.asarray(obs["frame"], dtype=np.float32)) / HORIZON,
            device=self.device,
        )
        if self.hidden is None and self.policy.is_recurrent:
            self.hidden = self.policy.initial_hidden(len(local), self.device)
        hidden_before = None if self.hidden is None else self.hidden.detach().clone()
        if not self.records and self.hidden is not None:
            self.rollout_initial_hidden = self.hidden.detach().clone()
        with torch.no_grad():
            actions, logp, values, self.hidden = self.policy.act(
                local,
                global_,
                scalars,
                self.hidden,
                reset_t,
                action_mask=mask,
                remaining_horizon=remaining_horizon,
            )
            teacher_logits = self.policy.teacher(local, global_, scalars).masked_fill(
                ~mask, float("-inf")
            )
            b8_actions = teacher_logits.argmax(dim=-1)
        self.records.append(
            {
                "actions": actions.cpu(),
                "logp": logp.cpu(),
                "values": values.cpu(),
                "mask": mask.cpu(),
                "reset": reset_t.cpu(),
                "hidden_before": hidden_before,
                "remaining_horizon": remaining_horizon.cpu(),
                "critic_features": self.policy.last_critic_features.cpu(),
            }
        )
        self.records[-1]["b8_actions"] = b8_actions.cpu()
        return actions.cpu().numpy()

    def take_rollout(self, raw: Any, next_obs: dict[str, Any]) -> Any:
        """Convert one ordered collector batch to the learner's time-major contract."""
        import torch

        from research.bl_ppo.learner import PPORollout

        steps, worlds = len(self.records), self.records[0]["actions"].numel()
        if raw.active_transitions != steps * worlds:
            raise ValueError("collector/PPO decision count mismatch")

        # The collector's canonical order is world-major then time; restore [T,N].
        def raw_t(name: str, dtype: Any = None):
            x = torch.as_tensor(getattr(raw, name), device=self.device, dtype=dtype)
            return x.reshape(worlds, steps, *x.shape[1:]).transpose(0, 1).contiguous()

        def saved(name: str):
            return torch.stack([row[name] for row in self.records]).to(self.device)

        local, global_, scalars = raw_t("local"), raw_t("global_"), raw_t("scalars")
        remaining_horizon = raw_t("frame", torch.float32)
        remaining_horizon = (HORIZON - remaining_horizon) / HORIZON
        with torch.no_grad():
            nl = torch.as_tensor(next_obs["local"], device=self.device)
            ng = torch.as_tensor(next_obs["global"], device=self.device)
            ns = torch.as_tensor(next_obs["scalars"], device=self.device)
            nr = torch.as_tensor(next_obs["reset"], device=self.device, dtype=torch.bool)
            last_remaining_horizon = torch.as_tensor(
                (HORIZON - np.asarray(next_obs["frame"], dtype=np.float32)) / HORIZON,
                device=self.device,
            )
            _, last_value, _ = self.policy(
                nl, ng, ns, self.hidden, nr, remaining_horizon=last_remaining_horizon
            )
        first_hidden = self.rollout_initial_hidden
        current = {
            "local": local,
            "global_": global_,
            "scalars": scalars,
            "remaining_horizon": remaining_horizon,
            "critic_fit_mask": raw_t("world_id") % 5 != 0,
            "actions": saved("actions"),
            "logp": saved("logp"),
            "rewards": raw_t("reward"),
            "dones": raw_t("done", torch.bool),
            "values": saved("values"),
            "action_masks": saved("mask"),
            "resets": saved("reset"),
        }
        if self.policy.is_recurrent and self.prefix is not None:
            first_hidden = self.prefix["initial_hidden"]
            combined = {
                key: torch.cat((self.prefix[key], value), dim=0) for key, value in current.items()
            }
            loss_mask = torch.cat(
                (
                    torch.zeros(32, worlds, dtype=torch.bool, device=self.device),
                    torch.ones(steps, worlds, dtype=torch.bool, device=self.device),
                )
            )
        else:
            combined = current
            if self.policy.is_recurrent:
                if steps < 33:
                    raise ValueError(
                        "recurrent PPO needs at least 32 burn-in steps plus one loss step"
                    )
                loss_mask = torch.ones(steps, worlds, dtype=torch.bool, device=self.device)
                loss_mask[:32] = False
            else:
                loss_mask = torch.ones(steps, worlds, dtype=torch.bool, device=self.device)
        result = PPORollout(
            local=combined["local"],
            global_=combined["global_"],
            scalars=combined["scalars"],
            actions=combined["actions"],
            logp=combined["logp"],
            rewards=combined["rewards"],
            dones=combined["dones"],
            values=combined["values"],
            last_value=last_value,
            remaining_horizon=combined["remaining_horizon"],
            last_remaining_horizon=last_remaining_horizon,
            # A collector boundary has no terminal semantics.  Only a hero
            # death or absolute H5000 endpoint zeros its bootstrap.
            bootstrap_mask=~combined["dones"],
            critic_fit_mask=combined["critic_fit_mask"],
            action_masks=combined["action_masks"],
            resets=combined["resets"],
            initial_hidden=first_hidden,
            loss_mask=loss_mask,
        )
        self.last_actions = current["actions"].detach().cpu()
        self.last_b8_actions = torch.stack([row["b8_actions"] for row in self.records])
        self.last_values = current["values"].detach().cpu()
        self.last_critic_features = torch.stack([row["critic_features"] for row in self.records])
        if self.policy.is_recurrent and steps >= 32:
            self.prefix = {key: value[-32:].detach().clone() for key, value in current.items()}
            self.prefix["initial_hidden"] = self.records[-32]["hidden_before"].detach().clone()
        self.records.clear()
        self.rollout_initial_hidden = None
        return result

    def refresh_hidden(self) -> None:
        """Rebuild the next collector state from real context after an optimizer update."""
        import torch

        if not self.policy.is_recurrent or self.prefix is None:
            return
        hidden = self.prefix["initial_hidden"]
        with torch.no_grad():
            for step in range(32):
                _, _, hidden = self.policy(
                    self.prefix["local"][step],
                    self.prefix["global_"][step],
                    self.prefix["scalars"][step],
                    hidden,
                    self.prefix["resets"][step],
                    remaining_horizon=self.prefix["remaining_horizon"][step],
                )
        self.hidden = hidden.detach()


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return a never-before-used immutable export name."""
    directory.mkdir(parents=True, exist_ok=True)
    for ordinal in range(100000):
        path = directory / f"{stem}-{ordinal:05d}{suffix}"
        if not path.exists() and not Path(str(path) + ".complete").exists():
            return path
    raise RuntimeError("exhausted unique export names")


def e4_target_transitions(
    receipt: Path, requested: int | None = None, *, allow_scale: bool = False
) -> int:
    """Bind E2 sizing to a complete E4 receipt before any collection."""
    verify_export(receipt)
    value = json.loads(receipt.read_text())
    if value.get("schema") != E4_SCHEMA or value.get("status") != "COMPLETE":
        raise ValueError("E2 requires a complete immutable E4 ruler receipt")
    configs = value.get("configs")
    selected_processes = value.get("selected_processes")
    if not isinstance(configs, list) or selected_processes not in {6, 12}:
        raise ValueError("E4 receipt lacks its fixed 6/12 selection")
    selected = next((row for row in configs if row.get("processes") == selected_processes), None)
    rate = selected.get("allocated_hero_transitions_per_second") if selected else None
    import math

    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not rate >= 0:
        raise ValueError("E4 receipt has no finite selected allocated rate")
    if (
        value.get("scientific") is not True
        or value.get("worlds") != 128
        or value.get("long_bodies") is not True
    ):
        raise ValueError("E2 requires scientific 128-world long-body E4 evidence")
    # >=4k merely permits the separate forecast-gated 20M amendment.  It does
    # not silently turn the committed default 8M pilot into a scale run.
    permitted = 4_000_000 if rate < 1000 else 8_000_000
    if requested == 20_000_000 and rate >= 4000 and allow_scale:
        if math.ceil(1.25 * requested / rate / 60 + 15) > 135:
            raise ValueError("20M forecast cannot fit the per-arm allocation cap")
        return requested
    if requested is not None and requested != permitted:
        raise ValueError(f"E4 permits exactly {permitted} transitions per arm, not {requested}")
    return permitted


def _policy_callback(policy: Any):
    def choose(observation: dict[str, Any], reset: Any) -> Any:
        if hasattr(policy, "act"):
            return policy.act(observation, reset=reset)
        return policy(
            observation["local"], observation["global"], observation["scalars"], reset=reset
        )

    return choose


def _make_collector(args: argparse.Namespace, processes: int):
    from research.bl_ppo.bank import all_seeds, materialize
    from research.bl_ppo.collector import CollectorConfig, PersistentCollector

    config = CollectorConfig(
        worlds=args.worlds,
        processes=processes,
        rollout_steps=args.rollout_steps,
        horizon=HORIZON,
        seed=args.seed,
        roster=args.roster,
        long_bodies=args.long_bodies,
        native_v8=args.native_v8,
        preflight=args.preflight,
        excluded_seeds=frozenset(all_seeds(materialize())),
    )
    return PersistentCollector(config)


def _calibrated_temperature(args: argparse.Namespace) -> float:
    """Calibrate on an independent, no-step collector namespace before training."""
    if args.temperature is not None:
        if not args.preflight:
            raise ValueError("production temperature is calibrated, never manually supplied")
        return float(args.temperature)
    import torch

    from research.bl_ppo.bank import all_seeds, materialize
    from research.bl_ppo.collector import CollectorConfig, PersistentCollector
    from research.bl_ppo.model import PPOPolicy, calibrate_temperature

    processes = args.processes if isinstance(args.processes, int) else min(args.processes)
    config = CollectorConfig(
        worlds=args.worlds,
        processes=processes,
        rollout_steps=args.rollout_steps,
        horizon=HORIZON,
        seed=args.seed + 104729,
        roster=args.roster,
        long_bodies=args.long_bodies,
        native_v8=args.native_v8,
        preflight=args.preflight,
        excluded_seeds=frozenset(all_seeds(materialize())),
    )
    collector = PersistentCollector(config)
    try:
        obs = collector.peek()
        policy = PPOPolicy("F", args.b8_checkpoint, 1.0).to(args.device).eval()
        with torch.no_grad():
            q = policy.teacher(
                torch.as_tensor(obs["local"], device=args.device),
                torch.as_tensor(obs["global"], device=args.device),
                torch.as_tensor(obs["scalars"], device=args.device),
            )
        return calibrate_temperature(q.cpu(), action_masks=torch.as_tensor(obs["mask"]))
    finally:
        collector.close()


def run_ruler(args: argparse.Namespace) -> dict[str, Any]:
    """Run E4 (or explicitly-labelled tiny preflight) for every requested process count."""
    args.processes = [
        item for value in args.processes for item in (value if isinstance(value, list) else [value])
    ]
    args.native_v8 = args.native_v8 in (True, "1", "true", "auto", "on")
    if args.preflight and (args.worlds > 2 or max(args.processes) > 2):
        raise ValueError("tiny Mac preflight is limited to two worlds and two processes")
    if not args.preflight and (args.worlds != 128 or set(args.processes) != {6, 12}):
        raise ValueError("scientific E4 is fixed at 128 worlds and 6 versus 12 processes")
    if args.preflight and not args.allow_tiny_preflight:
        raise ValueError(
            "pass --allow-tiny-preflight to label this operational timing non-scientific"
        )

    import torch

    from research.bl_ppo.model import PPOPolicy

    calibration_started = time.monotonic()
    policy = PPOPolicy(
        arm="F", b8_checkpoint=args.b8_checkpoint, temperature=_calibrated_temperature(args)
    )
    shared_seconds = time.monotonic() - calibration_started
    runtime_root = os.environ.get("COLAB_CLI_ROOT")
    if runtime_root and (Path(runtime_root) / "state.json").is_file():
        started_at = json.loads((Path(runtime_root) / "state.json").read_text()).get("started_at")
        if isinstance(started_at, (int, float)):
            shared_seconds += max(0.0, time.time() - started_at - shared_seconds)
    rows = []
    for processes in args.processes:
        torch.manual_seed(args.seed)
        adapter = PersistentPolicyAdapter(policy, args.device)
        case_started = time.monotonic()
        collector = _make_collector(args, processes)
        try:
            result = collector.ruler(
                adapter, warmup_steps=args.warmup_steps, measured_steps=args.measured_steps
            )
        finally:
            close = getattr(collector, "close", None)
            if close:
                close()
        charged_seconds = time.monotonic() - case_started + shared_seconds / len(args.processes)
        rows.append(
            {
                "processes": processes,
                **dict(result),
                "allocated_seconds": charged_seconds,
                "allocated_hero_transitions_per_second": result["active_hero_transitions"]
                / charged_seconds,
                "allocation_scope": (
                    "runtime bootstrap, calibration, worker startup, warmup, "
                    "measurement and close; shared costs split equally; "
                    "supervisor final transfer/stop excluded"
                ),
            }
        )
    ranked = sorted(
        rows, key=lambda row: (-row["allocated_hero_transitions_per_second"], row["processes"])
    )
    value = {
        "schema": E4_SCHEMA,
        "status": "COMPLETE",
        "scientific": not args.preflight,
        "operational_note": (
            None if not args.preflight else "tiny Mac preflight; no gameplay evidence"
        ),
        "worlds": args.worlds,
        "horizon": HORIZON,
        "roster": args.roster,
        "long_bodies": args.long_bodies,
        "native_v8": args.native_v8,
        "configs": rows,
        "selected_processes": int(ranked[0]["processes"]),
        "created_unix": time.time(),
    }
    export_json(args.output, value)
    return value


def _identity(args: argparse.Namespace, temperature: float) -> dict[str, Any]:
    from research.bl_ppo.bank import canonical_sha, materialize
    from research.bl_ppo.rule import RULE

    return {
        "arm": args.arm,
        "temperature": temperature,
        "horizon": HORIZON,
        "worlds": args.worlds,
        "processes": args.processes,
        "rollout_steps": args.rollout_steps,
        "b8_sha256": digest(args.b8_checkpoint),
        "intent_sha256": digest(Path(__file__).with_name("INTENT.md")),
        "e8_amendment_sha256": digest(Path(__file__).with_name("E8_AMENDMENT.md")),
        "e9_rule_sha256": canonical_sha(RULE),
        "bank_sha256": canonical_sha(materialize()),
        "source_commit": os.environ.get("BL_PPO_COMMIT", "local-unpinned"),
        "seed": args.seed,
        "target_transitions": args.transitions,
        "native_v8": bool(args.native_v8),
        "long_bodies": args.long_bodies,
        "roster": args.roster,
        "e4_receipt_sha256": None if args.preflight else verify_export(args.e4_receipt),
        "e9_receipt_sha256": None if args.preflight else verify_export(args.e9_receipt),
    }


def _record_fixed_bank_monitor(
    adapter: Any,
    policy: Any,
    args: argparse.Namespace,
    counters: dict[str, Any],
    *,
    endpoint: bool = False,
) -> None:
    """Record current-policy drift on the immutable ordinary bank."""
    from research.bl_ppo.bank import materialize
    from research.bl_ppo.collector import make_eval_driver
    from research.bl_ppo.monitor import ProbeRoot, run_fixed_bank, run_paired_mass_probe

    before = adapter.diagnostics.snapshot()["fixed_ordinary_bank_drift_by_length_decile"]
    evidence = run_fixed_bank(
        policy, adapter.ordinary_bank, adapter.diagnostics, device=args.device
    )
    after = adapter.diagnostics.snapshot()["fixed_ordinary_bank_drift_by_length_decile"]
    rows = {}
    for decile, current in after.items():
        prior, count = before[decile], current["observations"] - before[decile]["observations"]
        rows[decile] = {
            "observations": count,
            "mean_kl_b8_to_policy": (
                (current["kl_b8_to_policy_sum"] - prior["kl_b8_to_policy_sum"]) / count
                if count
                else None
            ),
            "greedy_disagreement_rate": (
                (current["greedy_disagreements"] - prior["greedy_disagreements"]) / count
                if count
                else None
            ),
        }
    roots = [
        ProbeRoot(int(row["world_seed"]), row["opponents"], row["mix"])
        for row in materialize()["excluded_parity_rows"]
    ]
    probe = run_paired_mass_probe(
        policy,
        roots,
        make_eval_driver,
        horizon=HORIZON,
        rng_seed=args.seed + adapter.next_monitor_transition,
        device=args.device,
    )
    counters["monitor_simulator_transitions"] = (
        counters.get("monitor_simulator_transitions", 0) + probe["charged_simulator_transitions"]
    )
    report = {
        "hero_transitions": counters["hero_transitions"],
        "actor_transitions": counters["actor_transitions"],
        "scheduled_transition": None if endpoint else adapter.next_monitor_transition,
        "endpoint": endpoint,
        **evidence,
        "by_length_decile": rows,
        "stochastic_vs_greedy_mass": probe,
        "fixture_role": "registered excluded parity fixtures; descriptive only",
        "diagnostics": adapter.diagnostics.snapshot(),
    }
    adapter.fixed_bank_reports.append(report)
    export_json(args.output / f"monitor-{len(adapter.fixed_bank_reports):04d}.json", report)
    if not endpoint:
        adapter.next_monitor_transition += 1_000_000


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fixed E2 arm.  Completion and tripwire status are always exported."""
    run_started = time.monotonic()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("training and explicit recovery require a fresh output directory")
    transitions = (
        args.transitions
        if args.preflight
        else e4_target_transitions(
            args.e4_receipt, args.transitions, allow_scale=args.scale_forecast_approved
        )
    )
    if args.preflight and (not isinstance(transitions, int) or transitions < 1):
        raise ValueError("tiny preflight requires an explicit positive --transitions")
    if args.preflight:
        if args.worlds > 2 or args.processes > 2:
            raise ValueError("preflight train is limited to <=2 worlds/processes")
        if not args.allow_tiny_preflight:
            raise ValueError("preflight requires --allow-tiny-preflight")
    elif args.checkpoint_every < 1800:
        raise ValueError("scientific E2 checkpoint cadence must be at least 1800 seconds")
    else:
        # This check is intentionally before calibration, model construction or
        # worker creation: a failed E9 operating-characteristics result means
        # E2 has no prospectively admissible scientific collection design.
        if args.e9_receipt is None:
            raise ValueError("scientific E2 requires an immutable E9 OC receipt")
        verify_export(args.e9_receipt)
        from research.bl_ppo.oc import validate_admission

        validate_admission(json.loads(args.e9_receipt.read_text()))
    import random

    import numpy as np
    import torch

    from research.bl_ppo.bank import all_seeds, materialize
    from research.bl_ppo.collector import CollectorConfig, PersistentCollector
    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PPOPolicy

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.resume is not None:
        from research.bl_ppo.recovery import read_snapshot

        temperature = float(read_snapshot(args.resume)["identity"]["temperature"])
    else:
        temperature = _calibrated_temperature(args)
    policy = PPOPolicy(arm=args.arm, b8_checkpoint=args.b8_checkpoint, temperature=temperature)
    adapter = PersistentPolicyAdapter(policy, args.device)
    # Tiny preflight exercises the actor path but never represents a scientific
    # critic qualification.  The production constructor has no update-count
    # bypass and remains transition- plus heldout-gate controlled.
    learner = PPOLearner(policy, critic_refit_updates=4) if args.preflight else PPOLearner(policy)
    from research.bl_ppo.diagnostics import E8Diagnostics
    from research.bl_ppo.monitor import FixedOrdinaryBank
    from research.bl_ppo.refit import CompleteLifeRefitBank

    refit_bank = CompleteLifeRefitBank()
    # recovery captures adapter state verbatim at complete-rollout boundaries;
    # retain unfinished lives and completed held-out evidence with that state.
    adapter.refit_bank = refit_bank
    adapter.diagnostics = E8Diagnostics(require_fixed_bank=not args.preflight)
    adapter.ordinary_bank = FixedOrdinaryBank(seed=args.seed)
    adapter.next_monitor_transition = 1_000_000
    adapter.fixed_bank_reports = []
    config = CollectorConfig(
        worlds=args.worlds,
        processes=args.processes,
        rollout_steps=args.rollout_steps,
        horizon=HORIZON,
        seed=args.seed,
        roster=args.roster,
        long_bodies=args.long_bodies,
        native_v8=args.native_v8,
        preflight=args.preflight,
        excluded_seeds=frozenset(all_seeds(materialize())),
    )
    collector = PersistentCollector(config)
    counters: dict[str, Any] = {
        "hero_transitions": 0,
        "critic_refit_transitions": 0,
        "critic_refit_fit_transitions": 0,
        "actor_transitions": 0,
        "updates": 0,
        "actor_enabled": False,
        "collection_seconds": 0.0,
        "learner_seconds": 0.0,
    }
    identity = _identity(args, temperature)
    if args.resume is not None:
        from research.bl_ppo.recovery import restore_full_snapshot

        counters = restore_full_snapshot(args.resume, policy, learner, collector, adapter, identity)
        refit_bank = getattr(adapter, "refit_bank", None)
        if not isinstance(refit_bank, CompleteLifeRefitBank):
            raise ValueError("E8 resume lacks complete-life critic qualification state")
    requested_stop = {"value": False}

    def mark_stop(_signum: int, _frame: Any) -> None:
        requested_stop["value"] = True

    old_term, old_usr1 = signal.signal(signal.SIGTERM, mark_stop), signal.signal(
        signal.SIGUSR1, mark_stop
    )
    last_checkpoint = time.monotonic()
    status = "RUNNING"
    tripwires: list[dict[str, Any]] = []
    try:
        # Critic-only refit must complete on at least 2M *stochastic initial
        # actor* transitions before any residual actor / encoder update.
        while (
            counters["updates"] < 4
            if args.preflight
            else counters["critic_refit_fit_transitions"] < CRITIC_REFIT_TRANSITIONS
        ) and not requested_stop["value"]:
            raw = collector.collect(adapter)
            counters["collection_seconds"] += raw.wall_seconds
            learning_started = time.monotonic()
            if not adapter.ordinary_bank.frozen:
                import torch

                hidden_before = None
                if policy.is_recurrent:
                    hidden_before = (
                        torch.stack([row["hidden_before"] for row in adapter.records])
                        .squeeze(1)
                        .transpose(0, 1)
                        .reshape(-1, 1, 256)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                adapter.ordinary_bank.offer(
                    raw,
                    hidden_before=hidden_before,
                    actor_mode="frozen_initial_stochastic",
                    actor_sha256=f"{digest(args.b8_checkpoint)}:{temperature:.17g}",
                    context_kind=(
                        "actual_collector_hidden"
                        if policy.is_recurrent
                        else "feedforward_no_hidden"
                    ),
                )
            rollout = adapter.take_rollout(raw, collector.peek())
            # Collector order is [world, step]; adapter values are [step, world].
            # The bank retains only unfinished life prefixes plus bounded complete
            # returns, so it is safe in recovery snapshots and never treats a
            # rollout boundary as terminal evidence.
            refit_bank.add_batch(
                world_ids=torch.as_tensor(raw.world_id),
                critic_features=adapter.last_critic_features.transpose(0, 1).reshape(-1, 257),
                lengths=torch.as_tensor(raw.length),
                # Only the current raw rows enter the complete-life bank.  A
                # recurrent burn-in prefix is real context for PPO but it
                # belongs to the preceding rollout and must never duplicate
                # its critic qualification evidence.
                remaining_horizon=torch.as_tensor(
                    (HORIZON - raw.frame.astype(np.float32)) / HORIZON
                ),
                rewards=torch.as_tensor(raw.reward),
                terminals=torch.as_tensor(raw.done),
            )
            counters["hero_transitions"] += int(raw.active_transitions)
            counters["critic_refit_transitions"] += int(raw.active_transitions)
            counters["critic_refit_fit_transitions"] += int((raw.world_id % 5 != 0).sum())
            events = adapter.action_windows.observe(adapter.last_actions, adapter.last_b8_actions)
            adapter.diagnostics.observe(
                raw,
                actions=adapter.last_actions.transpose(0, 1).reshape(-1),
                b8_actions=adapter.last_b8_actions.transpose(0, 1).reshape(-1),
            )
            if events:
                tripwires.extend(events)
                status = "KILL"
                break
            learner.update(rollout, actor_enabled=False)
            adapter.refresh_hidden()
            counters["learner_seconds"] += time.monotonic() - learning_started
            counters["updates"] += 1
            if (
                not args.preflight
                and counters["hero_transitions"] >= adapter.next_monitor_transition
            ):
                if not adapter.ordinary_bank.frozen:
                    adapter.ordinary_bank.freeze()
                    adapter.diagnostics.configure_length_deciles(
                        [int(row["length"]) for row in adapter.ordinary_bank.rows()]
                    )
                _record_fixed_bank_monitor(adapter, policy, args, counters)
            print(json.dumps({"progress": counters}), flush=True)
        # A resumed actor phase reuses its immutable, pre-actor qualification;
        # recalibrating after actor updates would invalidate the held-out gate.
        if not args.preflight and not adapter.ordinary_bank.frozen:
            adapter.ordinary_bank.freeze()
            adapter.diagnostics.configure_length_deciles(
                [int(row["length"]) for row in adapter.ordinary_bank.rows()]
            )
        critic_refit = getattr(adapter, "critic_refit_receipt", None)
        if args.preflight:
            critic_refit = {
                "required_transitions": CRITIC_REFIT_TRANSITIONS,
                "observed_transitions": counters["critic_refit_transitions"],
                "observed_fit_transitions": counters["critic_refit_fit_transitions"],
                "popart": True,
                "heldout_late_body": None,
                "qualified": False,
                "status": "BYPASSED_PREFLIGHT_ONLY",
            }
            adapter.critic_refit_receipt = critic_refit
        elif not (counters.get("actor_enabled") and isinstance(critic_refit, dict)):
            # `qualify_critic` owns the held-out late-body EV/baseline gate.
            try:
                refit_bank.fit_complete_targets(learner)
                critic_metrics = refit_bank.qualification(learner)
            except ValueError as exc:
                critic_metrics = {"passed": False, "error": str(exc)}
            critic_refit = {
                "required_transitions": CRITIC_REFIT_TRANSITIONS,
                "observed_transitions": counters["critic_refit_transitions"],
                "observed_fit_transitions": counters["critic_refit_fit_transitions"],
                "popart": True,
                "heldout_late_body": {
                    "count": int(critic_metrics.get("late_body_rows", 0)),
                    "explained_variance": critic_metrics.get("explained_variance"),
                    "baseline_explained_variance": None,
                    "mse": critic_metrics.get("mse"),
                    "baseline_mse": critic_metrics.get("baseline_mse"),
                    "beats_baseline": bool(critic_metrics.get("passed", False)),
                },
                "qualified": bool(critic_metrics.get("passed", False)),
                "fit_support": critic_metrics.get("fit_support"),
                "holdout_support": critic_metrics.get("holdout_support"),
                **({"error": critic_metrics["error"]} if "error" in critic_metrics else {}),
            }
            adapter.critic_refit_receipt = critic_refit
        if not args.preflight and not bool(critic_refit["qualified"]):
            status = "INCOMPLETE"
        counters["actor_enabled"] = status == "RUNNING"
        while (
            status == "RUNNING"
            and counters["actor_transitions"] < transitions
            and not requested_stop["value"]
        ):
            remaining = transitions - counters["actor_transitions"]
            if remaining % args.worlds:
                raise ValueError(
                    "fixed E2 target must divide exactly into whole-world collector steps"
                )
            raw = collector.collect(
                adapter, steps=min(args.rollout_steps, remaining // args.worlds)
            )
            counters["collection_seconds"] += raw.wall_seconds
            learning_started = time.monotonic()
            rollout = adapter.take_rollout(raw, collector.peek())
            counters["hero_transitions"] += int(raw.active_transitions)
            counters["actor_transitions"] += int(raw.active_transitions)
            events = adapter.action_windows.observe(adapter.last_actions, adapter.last_b8_actions)
            adapter.diagnostics.observe(
                raw,
                actions=adapter.last_actions.transpose(0, 1).reshape(-1),
                b8_actions=adapter.last_b8_actions.transpose(0, 1).reshape(-1),
            )
            if events:
                tripwires.extend(events)
                status = "KILL"
                break
            learner.update(rollout, actor_enabled=True)
            adapter.refresh_hidden()
            while (
                not args.preflight
                and adapter.ordinary_bank.frozen
                and counters["hero_transitions"] >= adapter.next_monitor_transition
            ):
                _record_fixed_bank_monitor(adapter, policy, args, counters)
            counters["learner_seconds"] += time.monotonic() - learning_started
            counters["updates"] += 1
            print(json.dumps({"progress": counters}), flush=True)
            if time.monotonic() - last_checkpoint >= args.checkpoint_every:
                from research.bl_ppo.recovery import capture_full_snapshot

                export_torch(
                    _unique_path(args.output, "snapshot", ".pt"),
                    capture_full_snapshot(
                        policy,
                        learner,
                        collector,
                        adapter,
                        counters,
                        identity,
                        immutable_dir=args.output,
                    ),
                    max_payload_bytes=SNAPSHOT_MAX_BYTES,
                )
                last_checkpoint = time.monotonic()
        # The endpoint is deliberately evaluator-loadable; full recovery state
        # remains in a distinct periodic snapshot.
        if status == "RUNNING":
            if not args.preflight and adapter.ordinary_bank.frozen:
                _record_fixed_bank_monitor(adapter, policy, args, counters, endpoint=True)
            endpoint = args.output / "endpoint.pt"
            export_torch(endpoint, policy.checkpoint_dict())
            from research.bl_ppo.recovery import capture_full_snapshot

            export_torch(
                _unique_path(args.output, "final-snapshot", ".pt"),
                capture_full_snapshot(
                    policy,
                    learner,
                    collector,
                    adapter,
                    counters,
                    identity,
                    immutable_dir=args.output,
                ),
                max_payload_bytes=SNAPSHOT_MAX_BYTES,
            )
            status = "COMPLETE" if counters["actor_transitions"] >= transitions else "INTERRUPTED"
    except FloatingPointError as exc:
        tripwires.append(
            {
                "kind": "nonfinite",
                "stage": "gradient" if "gradient" in str(exc) else "loss",
                "error": str(exc),
            }
        )
        from research.bl_ppo.recovery import capture_full_snapshot

        if not adapter.records:
            export_torch(
                _unique_path(args.output, "tripwire", ".pt"),
                capture_full_snapshot(
                    policy,
                    learner,
                    collector,
                    adapter,
                    counters,
                    identity,
                    immutable_dir=args.output,
                ),
                max_payload_bytes=SNAPSHOT_MAX_BYTES,
            )
        status = "KILL"
    except ValueError as exc:
        tripwires.append({"kind": "integrity", "error": str(exc)})
        status = "INCOMPLETE"
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGUSR1, old_usr1)
        close = getattr(collector, "close", None)
        if close:
            close()
    value = {
        "schema": STATUS_SCHEMA,
        "arm": args.arm,
        "complete": status == "COMPLETE",
        "status": status,
        "tripwires": tripwires,
        "boost_contract_floor": adapter.action_windows.boost_contract_floor(),
        "e8_diagnostics": adapter.diagnostics.snapshot(),
        "fixed_ordinary_bank_reports": adapter.fixed_bank_reports,
        "stochastic_vs_greedy_mass": {
            "status": "COMPLETE" if adapter.fixed_bank_reports else "PREFLIGHT_NOT_RUN",
            "reports": [r["stochastic_vs_greedy_mass"] for r in adapter.fixed_bank_reports],
            "charged_simulator_transitions": counters.get("monitor_simulator_transitions", 0),
        },
        "counters": counters,
        "critic_refit": critic_refit if "critic_refit" in locals() else None,
        "critic_qualified": (
            bool(critic_refit.get("qualified", False)) if "critic_refit" in locals() else False
        ),
        "target_transitions": transitions,
        "wall_seconds": time.monotonic() - run_started,
        "e4_receipt_sha256": None if args.preflight else verify_export(args.e4_receipt),
        "e9_rule_sha256": identity["e9_rule_sha256"],
        "e9_receipt_sha256": identity["e9_receipt_sha256"],
        "scientific": not args.preflight,
        "stop_reason": (
            "tripwire"
            if tripwires
            else (
                "endpoint"
                if status == "COMPLETE"
                else (
                    "critic_not_qualified"
                    if status == "INCOMPLETE" and "critic_refit" in locals()
                    else status.lower()
                )
            )
        ),
    }
    if status == "COMPLETE":
        value["checkpoint_sha256"] = verify_export(args.output / "endpoint.pt")
    export_json(args.output / "training-status.json", value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--b8-checkpoint", "--checkpoint", dest="b8_checkpoint", type=Path, required=True
        )
        p.add_argument("--temperature", type=float)
        p.add_argument("--device", default="cpu")
        p.add_argument("--worlds", type=int, default=128)
        p.add_argument("--seed", type=int, default=20261021)
        p.add_argument("--roster", default="frozen-equal-mixes")
        p.add_argument("--long-bodies", "--long-body", action="store_true")
        p.add_argument("--native-v8", nargs="?", const=True, default=False)
        p.add_argument("--rollout-steps", type=int, default=128)
        p.add_argument("--preflight", action="store_true")
        p.add_argument("--allow-tiny-preflight", action="store_true")

    ruler = sub.add_parser("ruler")
    common(ruler)
    ruler.add_argument(
        "--processes",
        "--process-counts",
        type=lambda x: [int(v) for v in x.split(",")],
        nargs="+",
        default=[6, 12],
    )
    ruler.add_argument("--warmup-steps", type=int, default=128)
    ruler.add_argument("--measured-steps", type=int, default=2048)
    ruler.add_argument("--max-seconds", type=int, default=1800)
    train = sub.add_parser("train")
    common(train)
    train.add_argument("--arm", choices=("R", "F"), required=True)
    train.add_argument("--processes", type=int, required=True)
    train.add_argument("--e4-receipt", type=Path)
    train.add_argument("--e9-receipt", type=Path)
    train.add_argument("--transitions", type=int)
    train.add_argument("--checkpoint-every", type=float, default=1800.0)
    train.add_argument("--horizon", type=int, default=HORIZON)
    train.add_argument("--resume", type=Path)
    train.add_argument("--scale-forecast-approved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "horizon", HORIZON) != HORIZON:
        raise ValueError("E2 horizon is fixed at 5000")
    value = run_ruler(args) if args.command == "ruler" else run_train(args)
    print(json.dumps(value, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
