#!/usr/bin/env python3
"""CPU-only T4c same-init drift diagnostic.

The diagnostic replays a CTRL final on its own historical round-62 or round-65
scripted worlds.  It observes B8 and the three completed label arms without
altering CTRL action selection.  It is development evidence only: it neither
starts a learner nor has promotion authority.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.t4a_posthoc_20261016 import spillover  # noqa: E402

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
CTRL_ROOT = ARTIFACTS / "redesign-m6-mac-20261014"
B8_ROOT = ARTIFACTS / "redesign-m4b-mac-20261013"
T3_ROOT = ARTIFACTS / "redesign-t3-mac-20261015"
T4A_ROOT = ARTIFACTS / "redesign-t4a-mac-20261016"
BANK = ARTIFACTS / "t4-prep-20261015/bank-v2"
R62_PROBES = ARTIFACTS / "redesign-t3-probes-20261015" / "rows"
R65_PROBES = ARTIFACTS / "redesign-t4a-probes-20261016" / "rows"
DEFAULT_OUT = ARTIFACTS / "t4c-drift-20261016"
POSTHOC = ARTIFACTS / "t4a-posthoc-20261016" / "trajectory"
SEEDS = tuple(range(10, 15))
ARMS = ("B8", "TWO_SIDED", "LABELS", "LABELS_KEEPV8")


def sha256(path: Path) -> str:
    """Return the stable SHA-256 of one regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write canonical, finite JSON evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write an unpickled compressed raw-Q capture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def checkpoint(arm: str, seed: int) -> Path:
    """Return the exact final checkpoint for one diagnostic arm/seed."""
    if arm == "CTRL":
        return CTRL_ROOT / f"CTRL_s{seed}" / "final.pth"
    if arm == "B8":
        return B8_ROOT / f"B8_s{seed - 10}" / "final.pth"
    if arm == "TWO_SIDED":
        return T4A_ROOT / f"TWO_SIDED_s{seed}" / "final.pth"
    if arm in ("LABELS", "LABELS_KEEPV8"):
        return T3_ROOT / f"{arm}_s{seed}" / "final.pth"
    raise ValueError(f"unknown arm: {arm}")


def expected_row(seed: int, round_index: int) -> Path:
    """Return the historical CTRL/scripted row that owns replay identity."""
    if round_index == 62:
        return R62_PROBES / f"CTRL_s{seed}__r62__scripted.json"
    if round_index == 65:
        return R65_PROBES / f"CTRL_s{seed}__r65__scripted.json"
    raise ValueError("round_index must be 62 or 65")


def raw_rates(ctrl_q: np.ndarray, candidate_q: np.ndarray) -> dict[str, int | float]:
    """Return raw and consequential first-argmax changes on a common row set."""
    ctrl, candidate = spillover._validate_q(ctrl_q, candidate_q)
    ctrl_action, candidate_action = ctrl.argmax(axis=1), candidate.argmax(axis=1)
    changed = candidate_action != ctrl_action
    # The specified gap is CTRL Q(best) - CTRL Q(other arm's selected action).
    best = ctrl[np.arange(len(ctrl)), ctrl_action]
    other = ctrl[np.arange(len(ctrl)), candidate_action]
    consequential = changed & ((best - other) > 0.3)
    return {
        "decisions": int(len(ctrl)),
        "argmax_changes": int(changed.sum()),
        "raw_change_rate": float(changed.mean()),
        "consequential_changes": int(consequential.sum()),
        "consequential_change_rate": float(consequential.mean()),
        "threshold": 0.3,
    }


class CaptureObserver:
    """Observe raw Q values on active CTRL decisions without changing play."""

    def __init__(self, seed: int, round_index: int, outside_collector: Any | None = None) -> None:
        import torch

        from src.model.ego2s_network import load_ego2s_checkpoint

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.seed, self.round_index = seed, round_index
        # The complete, hash-bound r62 post-hoc trajectory captures already
        # contain raw CTRL/TWO_SIDED/LABELS/KEEPV8 Qs.  Reusing them after exact
        # key/fingerprint/Q validation avoids redoing three rowwise forwards.
        # The r65 extension is the same-init baseline only; the three arm
        # readouts are the pre-existing r62 `(c)` comparison.
        live_arms = ("B8",)
        self.nets = {
            arm: load_ego2s_checkpoint(str(checkpoint(arm, seed)), "cpu").eval()
            for arm in live_arms
        }
        self.rows: dict[str, list[np.ndarray]] = {
            key: []
            for key in ("length", "fingerprint", "ctrl_q", *(f"{arm}_q" for arm in live_arms))
        }
        self.outside_collector = outside_collector

    def observe(self, sim: Any, slots: np.ndarray, ctrl_q: np.ndarray) -> None:
        from src.simd_env.ego2s_policy import forward_q
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        active = sim.get_alive()[rows[:, 0], rows[:, 1]].astype(bool)
        if not active.any():
            return
        obs = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        q = {"CTRL": np.asarray(ctrl_q, dtype=np.float32)}
        q.update({arm: forward_q(net, obs, "rowwise").numpy() for arm, net in self.nets.items()})
        if self.outside_collector is not None and self.round_index == 62:
            self.outside_collector.observe(sim, rows, obs, q["CTRL"], q["B8"])
        index = np.flatnonzero(active)
        self.rows["length"].append(
            sim.get_lengths()[rows[index, 0], rows[index, 1]].astype(np.int64)
        )
        self.rows["fingerprint"].append(
            spillover.observation_fingerprints(
                obs["local"][index], obs["global"][index], obs["scalars"][index]
            )
        )
        self.rows["ctrl_q"].append(q["CTRL"][index])
        for arm in self.nets:
            self.rows[f"{arm}_q"].append(q[arm][index])

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.concatenate(value) for key, value in self.rows.items()}


def add_reused_r62_q(seed: int, arrays: dict[str, np.ndarray]) -> None:
    """Join T4a's validated r62 captures after exact CTRL-state identity checks."""
    for arm in ("TWO_SIDED", "LABELS", "LABELS_KEEPV8"):
        path = POSTHOC / f"{arm}_s{seed}.npz"
        with np.load(path, allow_pickle=False) as archive:
            prior = {key: archive[key].copy() for key in archive.files}
        required = {"length", "fingerprint", "ctrl_q", "candidate_q"}
        if not required <= set(prior):
            raise ValueError(f"posthoc capture fields missing: {path}")
        # Both captures retain active CTRL rows in evaluator order.  Exact
        # fingerprint plus CTRL raw-Q identity makes a positional join auditable.
        if (
            len(prior["length"]) != len(arrays["length"])
            or not np.array_equal(prior["length"], arrays["length"])
            or not np.array_equal(
                prior["fingerprint"].astype("U64"), arrays["fingerprint"].astype("U64")
            )
            or not np.array_equal(prior["ctrl_q"], arrays["ctrl_q"])
        ):
            raise AssertionError(f"posthoc r62 state/Q identity mismatch: {arm}_s{seed}")
        arrays[f"{arm}_q"] = prior["candidate_q"]


def replay(seed: int, round_index: int, out: Path, collect_outside: bool = False) -> dict[str, Any]:
    """Run one identity-checked CTRL replay and write its raw-Q capture."""
    import torch

    from research.redesign_m2_20261008 import dev_probe, ni_spec
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_diagnostics_20261016.replay import SeriesObserver, _assert_parity
    from src.core.device_manager import DeviceManager
    from src.simd_env import ego2s_policy, eval_engine

    capture = out / "captures" / f"CTRL_s{seed}__r{round_index}__scripted.npz"
    for target in (capture, capture.with_suffix(".receipt.json")):
        if target.exists() or target.is_symlink():
            raise ValueError(f"refusing existing diagnostic capture: {target}")
    row_path = expected_row(seed, round_index)
    expected = json.loads(row_path.read_text())
    if (
        expected["mix"] != "scripted"
        or expected["frames"] != 5000
        or expected["round_index"] != round_index
    ):
        raise ValueError("only historical H5000 CTRL/scripted rows are admissible")
    inputs = {
        str(row_path): sha256(row_path),
        str(BANK / "manifest.json"): sha256(BANK / "manifest.json"),
    }
    for arm in ("CTRL", *ARMS):
        path = checkpoint(arm, seed)
        inputs[str(path)] = sha256(path)
    collector = None
    if collect_outside:
        from research.t4c_20261016.data import OutsideCollector

        collector = OutsideCollector(seed, "CTRL", BANK)
    observer = CaptureObserver(seed, round_index, collector)
    torch.set_num_threads(1)
    DeviceManager.override_device(torch.device("cpu"))
    ctx = gid._context(1)
    profile = dataclasses.replace(ctx["profile"], scored_horizon=5000)
    seeds = dev_probe.probe_seeds(96, round_index)
    series = SeriesObserver(seeds, 5000)
    rosters = {
        int(item["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in item["slots"]]
        for item in ctx["ds"]._design_rows(seeds)
        if item["mix"] == "scripted"
    }
    original = ego2s_policy.Ego2sSimdPolicy

    class ObservedPolicy(original):
        def actions(self, masks, sim, slots):
            q = ego2s_policy.student_q(self.net, sim, slots, self.forward).numpy()
            observer.observe(sim, slots, q)
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            advisory[~advisory.any(axis=1), :3] = True
            return np.where(advisory, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)

    ego2s_policy.Ego2sSimdPolicy = ObservedPolicy
    try:
        records = eval_engine.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            5000,
            seeds,
            profile=profile,
            opponent_specs_by_world=rosters,
            mix_id="scripted",
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            vector61_backend="numba",
            scripted_anchors="fast",
            hero_ego2s=str(checkpoint("CTRL", seed)),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            frame_observer=series,
        )
    finally:
        ego2s_policy.Ego2sSimdPolicy = original
        DeviceManager.reset_for_testing()
    _assert_parity(expected, series, records)
    if inputs != {path: sha256(Path(path)) for path in inputs}:
        raise RuntimeError("a bound replay input changed while the evaluator was running")
    capture = out / "captures" / f"CTRL_s{seed}__r{round_index}__scripted.npz"
    arrays = observer.arrays()
    if round_index == 62:
        add_reused_r62_q(seed, arrays)
    atomic_npz(capture, **arrays)
    receipt: dict[str, Any] = {
        "schema": "t4c-drift-capture/v1",
        "seed": seed,
        "round_index": round_index,
        "mix": "scripted",
        "rows": int(len(arrays["length"])),
        "capture": str(capture),
        "capture_sha256": sha256(capture),
        "inputs_sha256": inputs,
        "historical_row": str(row_path),
        "historical_row_sha256": inputs[str(row_path)],
    }
    if collector is not None:
        death = np.asarray(
            [
                (
                    expected["death_frame"][str(world)]
                    if expected["death_frame"][str(world)] is not None
                    else -1
                )
                for world in seeds
            ],
            dtype=np.int32,
        )
        receipt["outside_candidate"] = collector.finish(
            ARTIFACTS / "t4c-data-20261016" / "candidates", death, list(seeds), inputs
        )
    atomic_json(capture.with_suffix(".receipt.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "summarize"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--round", dest="round_index", type=int, choices=(62, 65))
    parser.add_argument("--outside-candidates", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "replay":
        if args.seed is None or args.round_index is None:
            parser.error("replay requires --seed and --round")
        print(
            json.dumps(
                replay(args.seed, args.round_index, args.out, args.outside_candidates),
                sort_keys=True,
            )
        )
    else:
        from research.t4c_20261016.drift_report import aggregate

        print(json.dumps(aggregate(args.out), sort_keys=True))


if __name__ == "__main__":
    main()
