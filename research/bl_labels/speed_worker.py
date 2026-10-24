"""Stdlib-only entrypoint for one isolated T-EQ parent or frozen F1 bank.

Launched by speed_run in a separate interpreter rooted at the requested pin.
No candidate scientific module is imported by this bootstrap.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Any


def isolate(root: Path, pin: str) -> None:
    """Make all scientific imports resolve inside the verified worktree."""
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(F0_COMMIT=pin, F1_COMMIT=pin)
    # A worker may inherit an already lower priority from its caller.
    os.setpriority(os.PRIO_PROCESS, 0, max(10, os.getpriority(os.PRIO_PROCESS, 0)))


def assert_origins(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"research", "src"}:
            continue
        filename = getattr(module, "__file__", None)
        if filename and not Path(filename).resolve().is_relative_to(root):
            raise RuntimeError(f"foreign scientific import: {name}")
        for directory in getattr(module, "__path__", []):
            if not Path(directory).resolve().is_relative_to(root):
                raise RuntimeError(f"foreign scientific package: {name}")


def collect(args: argparse.Namespace) -> None:
    """Instrument calls and masks externally; never add keys to science exports."""

    from research.bl_labels import factory
    from research.bl_labels.exports import publish_json
    from research.bl_served_search import b2_engine
    from src.simd_env import ego2s_policy, ego_raster

    item = json.loads(args.manifest.read_text())["allocation"][args.index]
    namespace = json.loads(args.manifest.read_text())["namespace"]
    primary: list[Any] = []
    advisory: dict[str, list[bool]] = {}
    counts = {
        "living_decisions": 0,
        "natural_rasters": 0,
        "natural_forwards": 0,
        "forced_forwards": 0,
        "forced_steps": 0,
    }
    context = ["other"]
    original_make = factory.make_driver

    def make(*values: Any) -> Any:
        driver = original_make(*values)
        primary.append(driver)
        return driver

    factory.make_driver = make
    original_forward = ego2s_policy.forward_q
    original_raster = ego_raster.build_ego_raster

    def forward(*values: Any, **kwargs: Any) -> Any:
        if context[0] == "natural":
            counts["natural_forwards"] += 1
        elif context[0] == "forced":
            counts["forced_forwards"] += 1
        return original_forward(*values, **kwargs)

    def raster(source: Any, *values: Any, **kwargs: Any) -> Any:
        if primary and source is primary[0].sim:
            counts["natural_rasters"] += 1
        return original_raster(source, *values, **kwargs)

    ego2s_policy.forward_q = forward
    ego2s_policy.build_ego_raster = raster
    ego_raster.build_ego_raster = raster
    method = (
        "prepared_decision"
        if hasattr(b2_engine.WatchDriver, "prepared_decision")
        else "legal_and_bare"
    )
    original_decision = getattr(b2_engine.WatchDriver, method)

    def decision(self: Any) -> Any:
        prior = context[0]
        # Only natural collection asks for a decision; branch rollouts call step.
        # The natural driver is replaced by restore after every Watch6 death.
        primary[:] = [self]
        if primary and self is primary[0]:
            key = f"{self.life}:{self.decision}"
            mask = self.sim.get_advisory_action_mask()[0, 0].astype(bool).tolist()
            if key in advisory:
                if advisory[key] != mask:
                    raise RuntimeError("advisory changed at one natural decision")
            else:
                advisory[key] = mask
                counts["living_decisions"] += 1
            context[0] = "natural"
        try:
            return original_decision(self)
        finally:
            context[0] = prior

    setattr(b2_engine.WatchDriver, method, decision)
    original_step = b2_engine.WatchDriver.step

    def step(self: Any, *values: Any, **kwargs: Any) -> Any:
        # OLD accepts only forced; NEW also accepts a positional prepared packet
        # (GateDriver forwards both through super). Preserve the exact call shape.
        forced = values[0] if values else kwargs.get("forced")
        prior = context[0]
        if forced is not None:
            context[0] = "forced"
            counts["forced_steps"] += 1
        elif primary and self is primary[0]:
            context[0] = "natural"
        try:
            return original_step(self, *values, **kwargs)
        finally:
            context[0] = prior

    b2_engine.WatchDriver.step = step
    target = args.output / "payloads"
    target.mkdir(parents=True)

    def publish(value: dict[str, Any]) -> None:
        if value["schema"] == "bl-labels-f0-life/v1":
            suffix = f"life-{value['life']:06d}"
        elif value["schema"] == "bl-labels-f0-session/v1":
            suffix = "session"
        else:
            raise RuntimeError("foreign factory publication")
        publish_json(target / f"{item['allocation_id']}--{suffix}.json", value)

    supplied = dict(
        item, publish_life=publish, deadline_monotonic=args.deadline, progress=lambda _p: None
    )
    entered = time.monotonic()
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    unit = factory.collect_unit(supplied, args.b8, args.frp3, namespace)
    elapsed = time.monotonic() - entered
    if profiler:
        profiler.disable()
    # collect_unit's returned unit and its callback session are different
    # scientific payloads. Retain both without overwriting either.
    publish_json(target / f"{item['allocation_id']}--unit.json", unit)
    assert_origins(args.worktree)
    diagnostic: dict[str, Any] = {
        "schema": "bl-labels-f0-speed-parent-diagnostics/v1",
        "pin": args.pin,
        "allocation_id": item["allocation_id"],
        "counts": counts,
        "advisory": advisory,
        "collector_seconds": elapsed,
        "spawn_to_entry_seconds": entered - args.launched,
    }
    if profiler:
        stats = pstats.Stats(profiler)
        seconds = dict.fromkeys(
            ("forward_q", "digests", "snapshots", "raw_observation", "v8_flood"), 0.0
        )
        for (filename, _line, name), record in stats.stats.items():
            cumulative = record[3]
            if name == "forward_q":
                seconds["forward_q"] += cumulative
            if name == "full_digest" or (
                name == "env_digest" and "grid_h5000_identity" in filename
            ):
                seconds["digests"] += cumulative
            if "_pickle.dumps" in name or "_pickle.loads" in name:
                seconds["snapshots"] += cumulative
            if name == "_raw_observation":
                seconds["raw_observation"] += cumulative
            if name == "v8_decide":
                # One outer kernel entry includes its flood descendants once.
                seconds["v8_flood"] += cumulative
        diagnostic["profile_seconds"] = seconds
        diagnostic["play_seconds"] = unit["timing_seconds"]["play"]
    publish_json(args.output / "diagnostic.json", diagnostic)


def f1(args: argparse.Namespace) -> None:
    from research.bl_labels import f1_ruler, f1_units
    from research.bl_labels.exports import publish_json

    bank_hash = f1_ruler.bank_digest(args.bank)
    bank = f1_ruler.load_bank(args.bank)
    ledger = args.output / "empty-ledger.json"
    publish_json(ledger, f1_units.empty_ledger(bank_hash, "mac", source_commit=args.pin))
    result = f1_ruler.run(
        bank, bank_hash, ledger, args.b8, args.frp3, "mac", args.output / "result", args.deadline
    )
    assert_origins(args.worktree)
    publish_json(args.output / "runtime-result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--launched", type=float, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.worktree = args.worktree.resolve()
    isolate(args.worktree, args.pin)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.bank:
        f1(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
