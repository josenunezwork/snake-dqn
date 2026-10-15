"""Capture the post-ReLU fuse output on M7 worlds; development diagnostic, not gate evidence."""

from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_key] = "2"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterator  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.b8_death_census_20261014 import census  # noqa: E402
from research.m7_feature_auc_20261014 import collect as m7  # noqa: E402

ARTIFACTS = census.ARTIFACTS
ROOT = ARTIFACTS / "t1-fuse-probe-20261015"
M7_ROOT = m7.OUT_ROOT
ARMS = tuple(f"B8_s{k}" for k in range(5)) + tuple(f"CTRL_s{k}" for k in range(10, 14))


def checkpoint_for(arm: str) -> Path:
    """Return each arm's own final checkpoint."""
    if arm not in ARMS:
        raise ValueError(arm)
    study = "redesign-m4b-mac-20261013" if arm.startswith("B8_") else "redesign-m6-mac-20261014"
    return ARTIFACTS / study / arm / "final.pth"


@contextmanager
def fuse_hook(net: torch.nn.Module) -> Iterator[list[np.ndarray]]:
    """Capture precisely fuse[2], the ReLU OUTPUT, without modifying the forward result."""
    if not isinstance(net.fuse[2], torch.nn.ReLU):
        raise ValueError("expected the fuse block's third module to be ReLU")
    rows: list[np.ndarray] = []

    def record(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        a = output.detach().cpu().numpy().copy()
        if a.ndim != 2 or a.shape[1] != 256 or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("invalid post-ReLU fuse representation")
        rows.append(a.astype(np.float16))

    handle = net.fuse[2].register_forward_hook(record)
    try:
        yield rows
    finally:
        handle.remove()


def labels(
    death_frame: np.ndarray,
    cause: np.ndarray,
    frame: np.ndarray,
    horizon: int,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """M7 labels: fatal decision included; exclude competing deaths and censored survivors."""
    died = death_frame >= 0
    within = died & (death_frame - frame < horizon)
    target = (cause == 2) | (cause == 4)
    return within & target, ~(within & ~target) & ~(~died & (frame > frames - horizon))


class Collector(m7.Collector):
    """Reuse the M7 observer, recording only L>=800 stride-3 fuse rows."""

    def __init__(self, seeds: list[int], min_len: int = 800, stride: int = 3) -> None:
        super().__init__(seeds, min_len, 10)
        self.stride = stride
        self.dec = {k: [] for k in ("features", "env", "frame", "length", "q", "action")}

    def record_fuse(
        self,
        sim: Any,
        rows: np.ndarray,
        features: np.ndarray,
        q: np.ndarray,
        act: np.ndarray,
    ) -> None:
        """Align one feature vector with each hero decision, before stepping the world."""
        if np.any(rows[:, 1] != 0) or features.shape != (len(rows), 256):
            raise ValueError("hero-only feature alignment failed")
        envs = rows[:, 0]
        length = sim.length[envs, 0]
        sel = (length >= self.min_len) & (self.frame % self.stride == 0)
        values = {
            "features": features[sel],
            "env": envs[sel].astype(np.int16),
            "frame": np.full(sel.sum(), self.frame, np.int16),
            "length": length[sel].astype(np.int32),
            "q": q[sel].astype(np.float32),
            "action": act[sel].astype(np.int8),
        }
        for key, value in values.items():
            self.dec[key].append(value)
        self.my_actions[self.frame] = (envs.copy(), act.copy())
        self.n_dec += len(envs)

    def output(self, arm: str, frames: int) -> dict[str, np.ndarray]:
        """Include every horizon's labels and eligibility mask without discarding other rows."""
        out = {key: np.concatenate(value) for key, value in self.dec.items()}
        out["world"] = np.asarray(self.seeds, np.int64)[out["env"]]
        out["arm"] = np.full(len(out["frame"]), arm)
        out["seeds"] = np.asarray(self.seeds, np.int64)
        out["death_frame"] = self.death_frame.copy()
        out["death_cause"] = self.death_cause.copy()
        for h in (50, 100, 300):
            out[f"y{h}"], out[f"keep{h}"] = labels(
                self.death_frame[out["env"]],
                self.death_cause[out["env"]],
                out["frame"],
                h,
                frames,
            )
        return out


@contextmanager
def instrumented(col: Collector) -> Iterator[None]:
    """Use M7's exact observation/action path, adding a short-lived forward hook."""
    from src.simd_env import ego2s_policy as ep
    from src.simd_env.ego_raster import build_ego_raster

    original = ep.Ego2sSimdPolicy

    class ProbePolicy(original):
        def actions(self, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
            if self.mask_mode != "advisory":
                raise ValueError("advisory mode required")
            rows = np.asarray(slots, np.int64).reshape(-1, 2)
            if not len(rows):
                return np.zeros(0, np.int64)
            obs = build_ego_raster(sim, ep.ego_config_for(self.net), backend="numba", rows=rows)
            with fuse_hook(self.net) as features:
                q = ep.forward_q(self.net, obs, self.forward).numpy()
            mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            mask[~mask.any(axis=1), :3] = True
            act = np.where(mask, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)
            col.record_fuse(sim, rows, np.concatenate(features), q, act)
            return act

    ep.Ego2sSimdPolicy = ProbePolicy
    try:
        yield
    finally:
        ep.Ego2sSimdPolicy = original


def verify_m7(arm: str, arrays: dict[str, np.ndarray], sha: str) -> dict[str, Any]:
    """Fail on any trajectory, row, action or Q difference from the matching M7 capture."""
    raw = M7_ROOT / "runs" / f"{arm}.npz"
    meta_path = raw.with_suffix(".json")
    if not raw.exists() or not meta_path.exists():
        return {"status": "missing", "path": str(raw)}
    meta = json.loads(meta_path.read_text())
    if meta["checkpoint"]["sha256"] != sha:
        raise ValueError("M7 checkpoint hash mismatch")
    with np.load(raw) as old:
        sel = (old["length"] >= 800) & (old["frame"] % 3 == 0)
        for key in ("env", "frame", "length", "q", "action"):
            if not np.array_equal(arrays[key], old[key][sel]):
                raise ValueError(f"M7 {arm} {key} identity failed")
        for key in ("seeds", "death_frame", "death_cause"):
            if not np.array_equal(arrays[key], old[key]):
                raise ValueError(f"M7 {arm} {key} identity failed")
    return {
        "status": "identical",
        "path": str(raw),
        "sha256": census.sha256_file(raw),
        "metadata_sha256": census.sha256_file(meta_path),
        "missing_for_reuse": ["full local raster", "full global raster", "fuse activations"],
    }


def capture(
    arm: str,
    out: Path = ROOT,
    worlds: int = 32,
    frames: int = 5000,
    min_len: int = 800,
) -> dict[str, Any]:
    """Run the specified arm on the unmodified M7 development worlds, no training."""
    from research.redesign_m2_20261008 import ni_spec
    from src.simd_env import eval_engine as ee

    torch.set_num_threads(2)
    start = time.perf_counter()
    checkpoint = checkpoint_for(arm)
    sha = census.sha256_file(checkpoint)
    ctx, seeds, rosters = census.load_context(worlds)
    profile = dataclasses.replace(ctx["profile"], scored_horizon=frames)
    col = Collector(seeds, min_len)
    eval_start = time.perf_counter()
    with instrumented(col):
        records = ee.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            frames,
            list(seeds),
            profile=profile,
            opponent_specs_by_world=dict(rosters),
            mix_id=census.MIX,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            vector61_backend="numba",
            scripted_anchors="fast",
            frame_observer=col.observe,
            hero_ego2s=str(checkpoint),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
        )
    elapsed = time.perf_counter() - eval_start
    arrays = col.output(arm, frames)
    if any(col.checks.values()):
        raise ValueError(f"observer integrity failed: {col.checks}")
    identity = verify_m7(arm, arrays, sha) if (worlds, frames, min_len) == (32, 5000, 800) else None
    meta = {
        "authority": "development diagnostic, not gate evidence",
        "arm": arm,
        "checkpoint": {"path": str(checkpoint), "sha256": sha},
        "worlds": seeds,
        "frames": frames,
        "min_len": min_len,
        "stride": 3,
        "round_index": 52,
        "mix": "scripted",
        "engine": "grid",
        "ego2s_forward": "rowwise",
        "ego2s_mask": "advisory",
        "veto": None,
        "threads": 2,
        "hook": "fuse[2] ReLU output",
        "feature_dtype": "float16",
        "eval_seconds": elapsed,
        "total_seconds": time.perf_counter() - start,
        "decisions": col.n_dec,
        "rows": len(arrays["frame"]),
        "checks": dict(col.checks),
        "m7_identity": identity,
        "git": census._git(),
        "episodes": census.episode_rows(records, seeds),
    }
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runs / f"{arm}.npz", **arrays)
    (runs / f"{arm}.json").write_text(json.dumps(census._jsonable(meta), indent=2) + "\n")
    print(
        json.dumps({k: meta[k] for k in ("arm", "rows", "eval_seconds", "m7_identity")}), flush=True
    )
    return meta


def main() -> None:
    """Capture arms serially; omit CTRL only if B8 capture is much slower than expected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    timings = []
    for arm in args.arms:
        if arm.startswith("CTRL") and timings and np.mean(timings) > 2 * 235:
            print("B8 capture >470s/arm: CTRL omitted under spec runtime fallback", flush=True)
            break
        meta = capture(arm, args.out)
        if arm.startswith("B8"):
            timings.append(meta["eval_seconds"])


if __name__ == "__main__":
    main()
