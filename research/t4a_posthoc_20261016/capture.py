#!/usr/bin/env python3
"""CPU-only round-62 raw-Q capture for the declared T4a post-hoc diagnostic.

This is diagnostic collection, not a probe or gate.  It reuses the saved CTRL
round-62 scripted roster, observes all candidate students during the one CTRL
rollout, and separately records a candidate-owned series for growth.  The
candidate-owned replay deliberately has no saved source row to compare with:
its receipt labels the resulting evaluator records as internal consistency
only, never source-row parity.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.t3_diagnostics_20261016 import replay as t3_replay  # noqa: E402
from research.t4a_posthoc_20261016 import spillover  # noqa: E402


def _context(expected: Mapping[str, Any]) -> tuple[Sequence[int], Mapping[int, Any], Any]:
    """Build the exact saved scripted round-62 roster context."""
    from research.redesign_m2_20261008 import dev_probe
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    if (
        int(expected["frames"]) != t3_replay.HORIZON
        or int(expected["round_index"]) != 62
        or expected["mix"] != "scripted"
    ):
        raise ValueError("post-hoc capture only accepts a saved H5000 scripted round-62 CTRL row")
    seeds = dev_probe.probe_seeds(int(expected["worlds"]), 62)
    ctx = gid._context(1)
    profile = dataclasses.replace(ctx["profile"], scored_horizon=t3_replay.HORIZON)
    rosters = {
        int(row["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in row["slots"]]
        for row in ctx["ds"]._design_rows(seeds)
        if row["mix"] == "scripted"
    }
    if set(rosters) != set(seeds):
        raise AssertionError("saved scripted roster does not cover every capture world")
    return seeds, rosters, (ctx, profile)


class CtrlTrajectoryCapture:
    """Capture multiple candidate raw-Q values from one CTRL featurizer result."""

    def __init__(self, candidates: Mapping[str, Path]) -> None:
        from src.model.ego2s_network import load_ego2s_checkpoint

        self.networks = {
            name: load_ego2s_checkpoint(str(path), "cpu").eval()
            for name, path in sorted(candidates.items())
        }
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in self.networks}

    def observe(
        self, sim: Any, slots: np.ndarray, obs: Mapping[str, np.ndarray], ctrl_q: np.ndarray
    ) -> None:
        """Append all living pre-action rows after exactly one raster featurization."""
        from src.simd_env.ego2s_policy import forward_q

        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        active = sim.get_alive()[rows[:, 0], rows[:, 1]].astype(bool)
        if not active.any():
            return
        fingerprints = spillover.observation_fingerprints(
            obs["local"], obs["global"], obs["scalars"]
        )
        lengths = sim.get_lengths()[rows[:, 0], rows[:, 1]]
        for name, net in self.networks.items():
            candidate_q = forward_q(net, obs, "rowwise").numpy()
            for index in np.flatnonzero(active):
                self.rows[name].append(
                    {
                        "seed_slot": int(rows[index, 0]),
                        "frame": int(sim.frame[rows[index, 0]]),
                        "length": int(lengths[index]),
                        "fingerprint": str(fingerprints[index]),
                        "ctrl_q": ctrl_q[index].astype(np.float32),
                        "candidate_q": candidate_q[index].astype(np.float32),
                    }
                )

    def write(self, out: Path, seed: int) -> dict[str, Path]:
        """Persist one unpickled capture NPZ per candidate."""
        result = {}
        for name, rows in self.rows.items():
            if not rows:
                raise ValueError(f"CTRL capture has no living rows for {name}")
            path = out / "trajectory" / f"{name}_s{seed}.npz"
            spillover._atomic_npz(
                path,
                seed_slot=np.asarray([row["seed_slot"] for row in rows], dtype=np.int16),
                frame=np.asarray([row["frame"] for row in rows], dtype=np.int16),
                length=np.asarray([row["length"] for row in rows], dtype=np.int64),
                fingerprint=np.asarray([row["fingerprint"] for row in rows], dtype="U64"),
                ctrl_q=np.asarray([row["ctrl_q"] for row in rows], dtype=np.float32),
                candidate_q=np.asarray([row["candidate_q"] for row in rows], dtype=np.float32),
            )
            result[name] = path
        return result


def _run_ctrl(seed: int, candidates: Mapping[str, Path], out: Path) -> Mapping[str, Any]:
    """Replay a saved CTRL row and capture shadows without changing CTRL actions."""
    import torch

    from research.redesign_m2_20261008 import ni_spec
    from src.core.device_manager import DeviceManager
    from src.simd_env import ego2s_policy, eval_engine
    from src.simd_env.ego_raster import build_ego_raster

    name = f"CTRL_s{seed}"
    expected = json.loads(t3_replay.row_path(name).read_text())
    identities = t3_replay._input_identities(name, expected)
    identities.update({str(path): t3_replay.sha256(path) for path in candidates.values()})
    seeds, rosters, context = _context(expected)
    ctx, profile = context
    observer = t3_replay.SeriesObserver(seeds, t3_replay.HORIZON)
    capture = CtrlTrajectoryCapture(candidates)
    torch.set_num_threads(1)
    prior_override = DeviceManager._override_device
    DeviceManager.override_device(torch.device("cpu"))
    original = ego2s_policy.Ego2sSimdPolicy

    class ObservedPolicy(original):
        def actions(self, masks, sim, slots):
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            obs = build_ego_raster(
                sim, ego2s_policy.ego_config_for(self.net), backend="numba", rows=rows
            )
            q = ego2s_policy.forward_q(self.net, obs, self.forward).numpy()
            capture.observe(sim, rows, obs, q)
            advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            advisory[~advisory.any(axis=1), :3] = True
            return np.where(advisory, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)

    ego2s_policy.Ego2sSimdPolicy = ObservedPolicy
    try:
        records = eval_engine.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            t3_replay.HORIZON,
            seeds,
            profile=profile,
            opponent_specs_by_world=rosters,
            mix_id="scripted",
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            vector61_backend="numba",
            scripted_anchors="fast",
            hero_ego2s=str(t3_replay.checkpoint(name)),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            frame_observer=observer,
        )
    finally:
        ego2s_policy.Ego2sSimdPolicy = original
        DeviceManager.override_device(prior_override)
    t3_replay._assert_parity(expected, observer, records)
    if identities != {path: t3_replay.sha256(Path(path)) for path in identities}:
        raise RuntimeError("bound CTRL replay input changed during capture")
    series = out / "series" / f"{name}.npz"
    spillover._atomic_npz(
        series, seeds=np.asarray(observer.seeds), length=observer.length, alive=observer.alive
    )
    trajectories = capture.write(out, seed)
    spillover._atomic_json(
        out / "receipts" / f"CTRL_s{seed}.json",
        {
            "schema": "t4a-posthoc-ctrl-capture/v1",
            "ctrl_series": str(series),
            "ctrl_series_sha256": t3_replay.sha256(series),
            "trajectory": {
                name: {"path": str(path), "sha256": t3_replay.sha256(path)}
                for name, path in trajectories.items()
            },
            "inputs_sha256": identities,
            "parity": "exact saved CTRL mass/deaths/alive_frames/death_frame/length_at_death",
        },
    )
    return {"trajectory": trajectories, "series": series, "records": records}


def _run_candidate_owned(seed: int, checkpoint: Path, out: Path) -> Path:
    """Record one TWO_SIDED-owned round-62 series for the growth diagnostic.

    There is intentionally no source-row parity assertion: TWO_SIDED did not
    exist when the CTRL probe receipt was written.  The receipt records this
    limited consistency boundary explicitly.
    """
    import torch

    from research.redesign_m2_20261008 import ni_spec
    from src.core.device_manager import DeviceManager
    from src.simd_env import eval_engine

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing TWO_SIDED final: {checkpoint}")
    roster_row = t3_replay.row_path(f"CTRL_s{seed}")
    identities = {
        str(checkpoint): t3_replay.sha256(checkpoint),
        str(roster_row): t3_replay.sha256(roster_row),
    }
    expected = json.loads(roster_row.read_text())
    seeds, rosters, context = _context(expected)
    ctx, profile = context
    observer = t3_replay.SeriesObserver(seeds, t3_replay.HORIZON)
    torch.set_num_threads(1)
    prior_override = DeviceManager._override_device
    DeviceManager.override_device(torch.device("cpu"))
    try:
        records = eval_engine.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            t3_replay.HORIZON,
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
        DeviceManager.override_device(prior_override)
    if identities != {path: t3_replay.sha256(Path(path)) for path in identities}:
        raise RuntimeError("bound TWO_SIDED checkpoint or CTRL roster row changed during replay")
    actual = {int(record["seed"]): record for record in records}
    if set(actual) != set(seeds):
        raise AssertionError("candidate replay did not return the saved CTRL world set")
    for index, world_seed in enumerate(seeds):
        record = actual[world_seed]
        alive = observer.alive[index]
        if np.any(~alive[:-1] & alive[1:]):
            raise AssertionError(f"candidate series resurrected in world {world_seed}")
        if float(observer.length[index].sum()) / t3_replay.HORIZON != float(
            record["mass_integral"]
        ):
            raise AssertionError(f"candidate series mass disagrees with record for {world_seed}")
        recorded_alive = int(record["denominators"]["alive_frames"])
        if int(alive.sum()) != recorded_alive:
            raise AssertionError(
                f"candidate alive-frame count disagrees with record for {world_seed}"
            )
        died = bool((~alive).any())
        if int(record["deaths"]) != int(died):
            raise AssertionError(f"candidate death count disagrees with record for {world_seed}")
        if died and observer.death_frame[index] != recorded_alive + 1:
            raise AssertionError(
                f"candidate terminal death frame disagrees with alive frames for {world_seed}"
            )
    path = out / "series" / f"TWO_SIDED_s{seed}.npz"
    spillover._atomic_npz(
        path, seeds=np.asarray(observer.seeds), length=observer.length, alive=observer.alive
    )
    spillover._atomic_json(
        out / "receipts" / f"TWO_SIDED_s{seed}.json",
        {
            "schema": "t4a-posthoc-candidate-series/v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": identities[str(checkpoint)],
            "saved_roster_row": str(roster_row),
            "saved_roster_row_sha256": identities[str(roster_row)],
            "inputs_sha256": identities,
            "series": str(path),
            "series_sha256": t3_replay.sha256(path),
            "worlds": len(seeds),
            "frames": t3_replay.HORIZON,
            "records": records,
            "parity": (
                "internal exact mass/alive_frames/death_count/death_frame; length_at_death is "
                "observer-only; no historical TWO_SIDED source row exists"
            ),
        },
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(10, 15), required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument(
        "--two-sided", type=Path, help="TWO_SIDED final for its owned growth series"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    candidates = {}
    for item in args.candidate:
        if item.count("=") != 1:
            parser.error("--candidate must be NAME=CHECKPOINT")
        name, raw_path = item.split("=", 1)
        if not name or name in candidates:
            parser.error("candidate names must be unique and nonempty")
        candidates[name] = Path(raw_path)
    result = _run_ctrl(args.seed, candidates, args.out)
    owned = _run_candidate_owned(args.seed, args.two_sided, args.out) if args.two_sided else None
    print(
        json.dumps(
            {
                "seed": args.seed,
                "trajectory": {k: str(v) for k, v in result["trajectory"].items()},
                "two_sided_series": None if owned is None else str(owned),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
