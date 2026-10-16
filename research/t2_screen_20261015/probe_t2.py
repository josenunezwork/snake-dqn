"""H5000 dev_probe evaluation with detached terminal-hero diagnostics."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from research.redesign_m2_20261008 import dev_probe, ni_spec
from research.redesign_scope_20261007 import grid_h5000_identity as gid


class SurvivalObserver:
    """Read detached snapshots only; death frame is one-based, length is pre-action."""

    def __init__(self, seeds: list[int]) -> None:
        self.alive_frames = {seed: 0 for seed in seeds}
        self.death_frame = dict.fromkeys(seeds)
        self.length_at_death = dict.fromkeys(seeds)

    def __call__(self, snapshot: Mapping[str, Any]) -> None:
        for index, seed in enumerate(snapshot["seeds"]):
            post, pre = snapshot["post"], snapshot["pre"]
            self.alive_frames[seed] += int(post["alive"][index])
            if pre["alive"][index] and not post["alive"][index]:
                if self.death_frame[seed] is not None:
                    raise ValueError("terminal hero died twice")
                self.death_frame[seed] = int(snapshot["frame"]) + 1
                self.length_at_death[seed] = int(pre["logical_mass"][index])


def probe(
    args: argparse.Namespace,
    ctx: Mapping[str, Any] | None = None,
    *,
    observer_factory: Callable[[list[int]], SurvivalObserver] = SurvivalObserver,
) -> dict[str, Any]:
    """Reuse dev_probe's context, seeds and hero options without changing its defaults."""
    from src.simd_env import eval_engine as ee

    ctx = gid._context(1) if ctx is None else ctx
    profile = ctx["profile"]
    if args.frames:
        profile = dataclasses.replace(profile, scored_horizon=args.frames)
    seeds = dev_probe.probe_seeds(args.worlds, args.round_index)
    rows = {
        int(row["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in row["slots"]]
        for row in ctx["ds"]._design_rows(seeds)
        if row["mix"] == args.mix
    }
    observer = observer_factory(seeds)
    started = time.perf_counter()
    recs = ee.run_simd_eval(
        ctx["lookup"][ni_spec.FRP3_S12[1]],
        rows[seeds[0]],
        args.frames or ni_spec.HORIZON,
        seeds,
        profile=profile,
        opponent_specs_by_world=rows,
        mix_id=args.mix,
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        vector61_backend=dev_probe.effective_v61_backend(args.v61_backend),
        scripted_anchors=args.anchors,
        frame_observer=observer,
        **dev_probe.probe_kwargs(args),
    )
    for rec in recs:
        assert rec["denominators"]["alive_frames"] == observer.alive_frames[rec["seed"]]
        assert rec["deaths"] == int(observer.death_frame[rec["seed"]] is not None)
    checkpoint = Path(args.student) if args.student else gid.CHECKPOINT_DIR / ni_spec.FRP3_S12[0]
    return {
        "schema": "t2-screen-probe/v1",
        "authority": "development screen, not gate evidence",
        "name": args.name,
        "arm": args.arm,
        "student": args.student,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "veto": args.veto,
        "round_index": args.round_index,
        "mix": args.mix,
        "worlds": args.worlds,
        "frames": args.frames or ni_spec.HORIZON,
        "ego2s_forward": args.ego2s_forward,
        "ego2s_mask": args.ego2s_mask,
        "v61_backend": args.v61_backend,
        "anchors": args.anchors,
        "wall": time.perf_counter() - started,
        "mass": {int(r["seed"]): r["mass_integral"] for r in recs},
        "deaths": {int(r["seed"]): r["deaths"] for r in recs},
        "alive_frames": observer.alive_frames,
        "death_frame": observer.death_frame,
        "length_at_death": observer.length_at_death,
        "causes": {int(r["seed"]): r["probes"]["death_cause"] for r in recs},
        "death_frame_convention": "one-based; death frame contributes zero alive frames",
        "length_at_death_convention": "logical mass immediately before fatal action",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = dev_probe.build_parser()
    parser.description = __doc__
    parser.add_argument("--name", required=True)
    parser.set_defaults(
        round_index=60,
        frames=5000,
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
        v61_backend="numba",
        anchors="fast",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch

    torch.set_num_threads(1)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("Set OMP_NUM_THREADS=1")
    row = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Each worker owns one output; atomic replacement prevents partial rows on restart.
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"name": args.name, "mix": args.mix, "wall": row["wall"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
