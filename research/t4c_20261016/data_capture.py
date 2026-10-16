"""CPU-only B8-owned round-62 outside-counter observation collection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.t4c_20261016.data import ROOT, OutsideCollector, _atomic_json, sha256


def run(seed: int, out: Path = ROOT / "candidates") -> dict:
    """Replay B8 on the exact saved round-62 scripted roster, with a bounded observer."""
    import numpy as np
    import torch

    from research.b8_death_census_20261014.census import b8_path
    from research.t3_diagnostics_20261016 import replay as t3
    from research.t4a_posthoc_20261016.capture import _context
    from research.redesign_m2_20261008 import ni_spec
    from src.core.device_manager import DeviceManager
    from src.simd_env import ego2s_policy, eval_engine
    from src.simd_env.ego_raster import build_ego_raster

    checkpoint = b8_path(f"b8_s{seed - 10}")
    roster_path = t3.row_path(f"CTRL_s{seed}")
    inputs = {
        str(path): sha256(path)
        for path in (checkpoint, roster_path, Path(__file__), Path(__file__).with_name("data.py"))
    }
    expected = json.loads(roster_path.read_text())
    seeds, rosters, context = _context(expected)
    ctx, profile = context
    observer = t3.SeriesObserver(seeds, t3.HORIZON)
    collector = OutsideCollector(seed, "B8")
    torch.set_num_threads(1)
    prior = DeviceManager._override_device
    DeviceManager.override_device(torch.device("cpu"))
    original = ego2s_policy.Ego2sSimdPolicy

    class ObservedPolicy(original):
        def actions(self, masks, sim, slots):
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            obs = build_ego_raster(
                sim, ego2s_policy.ego_config_for(self.net), backend="numba", rows=rows
            )
            q = ego2s_policy.forward_q(self.net, obs, self.forward).numpy()
            collector.observe(sim, rows, obs, q, q)
            advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            advisory[~advisory.any(axis=1), :3] = True
            return np.where(advisory, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)

    started = time.monotonic()
    ego2s_policy.Ego2sSimdPolicy = ObservedPolicy
    try:
        records = eval_engine.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            t3.HORIZON,
            seeds,
            profile=profile,
            opponent_specs_by_world=rosters,
            mix_id="scripted",
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            vector61_backend="numba",
            scripted_anchors="fast",
            hero_ego2s=str(checkpoint),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            frame_observer=observer,
        )
    finally:
        ego2s_policy.Ego2sSimdPolicy = original
        DeviceManager.override_device(prior)
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError("bound replay inputs changed")
    by_seed = {int(record["seed"]): record for record in records}
    if set(by_seed) != set(seeds):
        raise AssertionError("replay world set mismatch")
    for env, world_seed in enumerate(seeds):
        rec = by_seed[world_seed]
        if int(rec["denominators"]["alive_frames"]) != int(observer.alive[env].sum()):
            raise AssertionError("replay alive frame accounting mismatch")
        if not np.isclose(
            rec["mass_integral"], observer.length[env].sum() / t3.HORIZON, rtol=0, atol=1e-12
        ):
            raise AssertionError("replay mass accounting mismatch")
        if np.any(~observer.alive[env, :-1] & observer.alive[env, 1:]):
            raise AssertionError("terminal hero unexpectedly respawned")
    receipt = collector.finish(out, observer.death_frame, list(seeds), inputs)
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["validation"] = (
        "internal mass/alive/world-set identity; no pre-existing B8 round62 source row"
    )
    _atomic_json((out / f"B8_s{seed}.npz").with_suffix(".receipt.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="10,11,12,13,14")
    parser.add_argument("--out", type=Path, default=ROOT / "candidates")
    args = parser.parse_args()
    for seed in [int(value) for value in args.seeds.split(",")]:
        path = args.out / f"B8_s{seed}.npz"
        if path.exists() or path.with_suffix(".receipt.json").exists():
            raise ValueError(f"refusing existing candidate output: {path}")
        print(json.dumps(run(seed, args.out), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
