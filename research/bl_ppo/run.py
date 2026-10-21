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

from research.bl_ppo.exports import digest, export_json, export_torch, verify_export

HORIZON = 5000
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
        from research.bl_ppo.tripwires import ActionWindows

        self.action_windows = ActionWindows()

    def __call__(self, obs: dict[str, Any], reset: Any) -> Any:
        import torch

        local = torch.as_tensor(obs["local"], device=self.device)
        global_ = torch.as_tensor(obs["global"], device=self.device)
        scalars = torch.as_tensor(obs["scalars"], device=self.device)
        mask = torch.as_tensor(obs["mask"], device=self.device, dtype=torch.bool)
        reset_t = torch.as_tensor(reset, device=self.device, dtype=torch.bool)
        if self.hidden is None and self.policy.is_recurrent:
            self.hidden = self.policy.initial_hidden(len(local), self.device)
        hidden_before = None if self.hidden is None else self.hidden.detach().clone()
        if not self.records and self.hidden is not None:
            self.rollout_initial_hidden = self.hidden.detach().clone()
        with torch.no_grad():
            actions, logp, values, self.hidden = self.policy.act(
                local, global_, scalars, self.hidden, reset_t, action_mask=mask
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
        with torch.no_grad():
            nl = torch.as_tensor(next_obs["local"], device=self.device)
            ng = torch.as_tensor(next_obs["global"], device=self.device)
            ns = torch.as_tensor(next_obs["scalars"], device=self.device)
            nr = torch.as_tensor(next_obs["reset"], device=self.device, dtype=torch.bool)
            _, last_value, _ = self.policy(nl, ng, ns, self.hidden, nr)
        first_hidden = self.rollout_initial_hidden
        current = {
            "local": local,
            "global_": global_,
            "scalars": scalars,
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
            action_masks=combined["action_masks"],
            resets=combined["resets"],
            initial_hidden=first_hidden,
            loss_mask=loss_mask,
        )
        self.last_actions = current["actions"].detach().cpu()
        self.last_b8_actions = torch.stack([row["b8_actions"] for row in self.records])
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

    return {
        "arm": args.arm,
        "temperature": temperature,
        "horizon": HORIZON,
        "worlds": args.worlds,
        "processes": args.processes,
        "rollout_steps": args.rollout_steps,
        "b8_sha256": digest(args.b8_checkpoint),
        "intent_sha256": digest(Path(__file__).with_name("INTENT.md")),
        "bank_sha256": canonical_sha(materialize()),
        "source_commit": os.environ.get("BL_PPO_COMMIT", "local-unpinned"),
        "seed": args.seed,
        "target_transitions": args.transitions,
        "native_v8": bool(args.native_v8),
        "long_bodies": args.long_bodies,
        "roster": args.roster,
        "e4_receipt_sha256": None if args.preflight else verify_export(args.e4_receipt),
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fixed E2 arm.  Completion and tripwire status are always exported."""
    run_started = time.monotonic()
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
    learner = PPOLearner(policy)
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
        "updates": 0,
        "actor_enabled": False,
        "collection_seconds": 0.0,
        "learner_seconds": 0.0,
    }
    identity = _identity(args, temperature)
    if args.resume is not None:
        from research.bl_ppo.recovery import restore_full_snapshot

        counters = restore_full_snapshot(args.resume, policy, learner, collector, adapter, identity)
    requested_stop = {"value": False}

    def mark_stop(_signum: int, _frame: Any) -> None:
        requested_stop["value"] = True

    old_term, old_usr1 = signal.signal(signal.SIGTERM, mark_stop), signal.signal(
        signal.SIGUSR1, mark_stop
    )
    last_checkpoint = time.monotonic()
    status = "INCOMPLETE"
    tripwires: list[dict[str, Any]] = []
    try:
        # Critic-only refit must finish before residual actor / encoder unfreeze.
        while counters["updates"] < 4 and not requested_stop["value"]:
            raw = collector.collect(adapter)
            counters["collection_seconds"] += raw.wall_seconds
            learning_started = time.monotonic()
            rollout = adapter.take_rollout(raw, collector.peek())
            counters["hero_transitions"] += int(raw.active_transitions)
            events = adapter.action_windows.observe(adapter.last_actions, adapter.last_b8_actions)
            if events:
                tripwires.extend(events)
                status = "KILL"
                break
            learner.update(rollout, actor_enabled=False)
            adapter.refresh_hidden()
            counters["learner_seconds"] += time.monotonic() - learning_started
            counters["updates"] += 1
            print(json.dumps({"progress": counters}), flush=True)
        counters["actor_enabled"] = status != "KILL"
        while (
            status != "KILL"
            and counters["hero_transitions"] < transitions
            and not requested_stop["value"]
        ):
            remaining = transitions - counters["hero_transitions"]
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
            events = adapter.action_windows.observe(adapter.last_actions, adapter.last_b8_actions)
            if events:
                tripwires.extend(events)
                status = "KILL"
                break
            learner.update(rollout, actor_enabled=True)
            adapter.refresh_hidden()
            counters["learner_seconds"] += time.monotonic() - learning_started
            counters["updates"] += 1
            print(json.dumps({"progress": counters}), flush=True)
            if time.monotonic() - last_checkpoint >= args.checkpoint_every:
                from research.bl_ppo.recovery import capture_full_snapshot

                export_torch(
                    _unique_path(args.output, "snapshot", ".pt"),
                    capture_full_snapshot(policy, learner, collector, adapter, counters, identity),
                )
                last_checkpoint = time.monotonic()
        # The endpoint is deliberately evaluator-loadable; full recovery state
        # remains in a distinct periodic snapshot.
        if status != "KILL":
            endpoint = args.output / "endpoint.pt"
            export_torch(endpoint, policy.checkpoint_dict())
            from research.bl_ppo.recovery import capture_full_snapshot

            export_torch(
                _unique_path(args.output, "final-snapshot", ".pt"),
                capture_full_snapshot(policy, learner, collector, adapter, counters, identity),
            )
            status = "COMPLETE" if counters["hero_transitions"] >= transitions else "INTERRUPTED"
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
                capture_full_snapshot(policy, learner, collector, adapter, counters, identity),
            )
        status = "KILL"
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
        "counters": counters,
        "target_transitions": transitions,
        "wall_seconds": time.monotonic() - run_started,
        "e4_receipt_sha256": None if args.preflight else verify_export(args.e4_receipt),
        "scientific": not args.preflight,
        "stop_reason": (
            "tripwire" if tripwires else "endpoint" if status == "COMPLETE" else status.lower()
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
    train.add_argument("--transitions", type=int)
    train.add_argument("--checkpoint-every", type=float, default=120.0)
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
