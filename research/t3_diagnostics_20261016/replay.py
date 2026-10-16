#!/usr/bin/env python3
"""Exact round-62 scripted replays for the T3 mass and action diagnostics.

The saved probe rows remain the authority for the original measurements.  This
module runs the same evaluator again, rejects any terminal-row mismatch, and
records the post-transition hero mass series used by the mass integral.  CTRL
replays additionally observe the two requested final students on CTRL's exact
pre-action observations without changing CTRL's action selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# This is deliberately assignment, never setdefault: a caller's MPS setting is
# not admissible for a record-identity diagnostic.
os.environ["SNAKE_DQN_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np  # noqa: E402 — enforce native thread limits before import

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
PROBES = ARTIFACTS / "redesign-t3-probes-20261015"
CTRL = ARTIFACTS / "redesign-m6-mac-20261014"
TRAIN = ARTIFACTS / "redesign-t3-mac-20261015"
HORIZON = 5000
SEEDS = tuple(range(10, 15))
LABEL_ARMS = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")


def sha256(path: Path) -> str:
    """Return the SHA-256 of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(name: str) -> Path:
    """Return the registered T3 or M6 final checkpoint for ``name``."""
    root = CTRL if name.startswith("CTRL_") else TRAIN
    return root / name / "final.pth"


def row_path(name: str) -> Path:
    """Return the only probe row used by this diagnostic."""
    return PROBES / "rows" / f"{name}__r62__scripted.json"


def _terminal_t3_status() -> dict[str, int]:
    """Read the completed T3 receipt, rejecting absent or nonzero named finals."""
    path = TRAIN / "status.txt"
    rows: dict[str, int] = {}
    for line in path.read_text().splitlines():
        try:
            name, status = line.rsplit(" exit=", 1)
            rows[name] = int(status)
        except ValueError as exc:
            raise ValueError(f"malformed T3 status row: {line!r}") from exc
    expected = {f"{arm}_s{seed}" for arm in LABEL_ARMS for seed in SEEDS}
    if set(rows) != expected or any(rows[name] != 0 for name in expected):
        raise ValueError("all 15 T3 finals must be completed successfully before replay")
    return rows


def _input_identities(name: str, expected: Mapping[str, Any]) -> dict[str, str]:
    """Validate and fingerprint every checkpoint/row consumed by one replay."""
    if Path(expected["checkpoint"]) != checkpoint(name) or expected["checkpoint_sha256"] != sha256(
        checkpoint(name)
    ):
        raise ValueError("checkpoint identity does not match the saved probe row")
    identities = {
        str(row_path(name)): sha256(row_path(name)),
        str(checkpoint(name)): sha256(checkpoint(name)),
        str(TRAIN / "status.txt"): sha256(TRAIN / "status.txt"),
    }
    statuses = _terminal_t3_status()
    if name.startswith("CTRL_"):
        seed = int(name.rsplit("s", 1)[1])
        for arm in ("LABELS", "LABELS_KEEPV8"):
            shadow_name = f"{arm}_s{seed}"
            if statuses[shadow_name] != 0:
                raise ValueError(f"shadow final did not complete: {shadow_name}")
            shadow_row = json.loads(row_path(shadow_name).read_text())
            shadow_checkpoint = checkpoint(shadow_name)
            if Path(shadow_row["checkpoint"]) != shadow_checkpoint or shadow_row[
                "checkpoint_sha256"
            ] != sha256(shadow_checkpoint):
                raise ValueError(f"shadow checkpoint identity mismatch: {shadow_name}")
            identities[str(row_path(shadow_name))] = sha256(row_path(shadow_name))
            identities[str(shadow_checkpoint)] = sha256(shadow_checkpoint)
    return identities


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _number_map(value: Mapping[str, Any]) -> dict[int, Any]:
    return {int(key): item for key, item in value.items()}


class SeriesObserver:
    """Record post-transition series under the evaluator's exact mass convention."""

    def __init__(self, seeds: Sequence[int], frames: int) -> None:
        self.seeds = tuple(int(seed) for seed in seeds)
        self.length = np.zeros((len(seeds), frames), dtype=np.int32)
        self.alive = np.zeros((len(seeds), frames), dtype=bool)
        self.death_frame = np.full(len(seeds), -1, dtype=np.int32)
        self.death_length = np.full(len(seeds), -1, dtype=np.int32)

    def __call__(self, snapshot: Mapping[str, Any]) -> None:
        frame = int(snapshot["frame"])
        pre, post = snapshot["pre"], snapshot["post"]
        alive = np.asarray(post["alive"], dtype=bool)
        self.alive[:, frame] = alive
        # Dead frames contribute exactly zero to run_simd_eval's mass sum.
        self.length[:, frame] = np.where(alive, np.asarray(post["logical_mass"], dtype=np.int32), 0)
        deaths = np.asarray(pre["alive"], dtype=bool) & ~alive
        if np.any(deaths & (self.death_frame >= 0)):
            raise AssertionError("terminal hero died more than once")
        self.death_frame[deaths] = frame + 1  # saved-row convention is one based
        self.death_length[deaths] = np.asarray(pre["logical_mass"], dtype=np.int32)[deaths]


class DisagreementObserver:
    """Observe LABELS and KEEPV8 greed on CTRL observations without changing CTRL."""

    def __init__(self, seed: int, frames: int) -> None:
        from src.model.ego2s_network import load_ego2s_checkpoint

        self.seed, self.frames = int(seed), int(frames)
        self.networks = {
            arm: load_ego2s_checkpoint(str(checkpoint(f"{arm}_s{seed}")), "cpu")
            for arm in ("LABELS", "LABELS_KEEPV8")
        }
        self.rows: list[dict[str, int | float]] = []

    @staticmethod
    def _greedy(q: np.ndarray, advisory: np.ndarray) -> np.ndarray:
        mask = np.asarray(advisory, dtype=bool).copy()
        mask[~mask.any(axis=1), :3] = True  # exact advisory normal-only fallback
        return np.where(mask, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int8)

    def observe(
        self, sim: Any, slots: np.ndarray, ctrl_q: np.ndarray, ctrl_action: np.ndarray
    ) -> None:
        from src.simd_env.ego2s_policy import student_q

        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        active = sim.get_alive()[rows[:, 0], rows[:, 1]].astype(bool)
        if not active.any():
            return
        advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
        lengths = sim.get_lengths()[rows[:, 0], rows[:, 1]]
        for arm, network in self.networks.items():
            q = student_q(network, sim, rows, "rowwise").numpy()
            action = self._greedy(q, advisory)
            for index in np.flatnonzero(active):
                self.rows.append(
                    {
                        "arm": 0 if arm == "LABELS" else 1,
                        "env": int(rows[index, 0]),
                        "frame": int(sim.frame[rows[index, 0]]),
                        "length": int(lengths[index]),
                        "ctrl_action": int(ctrl_action[index]),
                        "arm_action": int(action[index]),
                        "disagree": int(action[index] != ctrl_action[index]),
                        "boost_disagree": int(
                            action[index] != ctrl_action[index]
                            and ((action[index] >= 3) != (ctrl_action[index] >= 3))
                        ),
                        # Required diagnostic score: unmasked, all-six raw outputs.
                        "ctrl_neg_max_q": float(-np.max(ctrl_q[index])),
                    }
                )

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {key: np.empty(0, dtype=dtype) for key, dtype in _DECISION_DTYPES.items()}
        return {
            key: np.asarray([row[key] for row in self.rows], dtype=dtype)
            for key, dtype in _DECISION_DTYPES.items()
        }


_DECISION_DTYPES = {
    "arm": np.int8,
    "env": np.int16,
    "frame": np.int16,
    "length": np.int32,
    "ctrl_action": np.int8,
    "arm_action": np.int8,
    "disagree": np.int8,
    "boost_disagree": np.int8,
    "ctrl_neg_max_q": np.float32,
}


def _run(
    name: str, expected: Mapping[str, Any], decisions: DisagreementObserver | None
) -> tuple[SeriesObserver, list[Mapping[str, Any]]]:
    """Run the same round-62/scripted evaluator settings that created ``expected``."""
    import dataclasses

    import torch

    from research.redesign_m2_20261008 import dev_probe, ni_spec
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env import ego2s_policy, eval_engine

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from src.core.device_manager import DeviceManager

    DeviceManager.override_device(torch.device("cpu"))
    if expected["v61_backend"] != "numba" or expected["anchors"] != "fast":
        raise ValueError("the saved T3 row must use numba and fast anchors")
    if expected["ego2s_forward"] != "rowwise" or expected["ego2s_mask"] != "advisory":
        raise ValueError("the saved T3 row must use rowwise/advisory student selection")
    frames, worlds, round_index = (
        int(expected["frames"]),
        int(expected["worlds"]),
        int(expected["round_index"]),
    )
    if frames != HORIZON or round_index != 62 or expected["mix"] != "scripted":
        raise ValueError("only round-62 H5000 scripted rows are valid inputs")
    seeds = dev_probe.probe_seeds(worlds, round_index)
    ctx = gid._context(1)
    profile = dataclasses.replace(ctx["profile"], scored_horizon=frames)
    rosters = {
        int(row["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in row["slots"]]
        for row in ctx["ds"]._design_rows(seeds)
        if row["mix"] == "scripted"
    }
    if set(rosters) != set(seeds):
        raise AssertionError("scripted roster construction did not cover every saved world")
    observer = SeriesObserver(seeds, frames)

    original = ego2s_policy.Ego2sSimdPolicy
    if decisions is not None:

        class ObservedPolicy(original):
            def actions(self, masks, sim, slots):
                q = ego2s_policy.student_q(self.net, sim, slots, self.forward).numpy()
                rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
                advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
                action = DisagreementObserver._greedy(q, advisory)
                decisions.observe(sim, rows, q, action)
                return action

        ego2s_policy.Ego2sSimdPolicy = ObservedPolicy
    try:
        records = eval_engine.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            frames,
            seeds,
            profile=profile,
            opponent_specs_by_world=rosters,
            mix_id="scripted",
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            vector61_backend="numba",
            scripted_anchors="fast",
            hero_ego2s=str(checkpoint(name)),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            frame_observer=observer,
        )
    finally:
        ego2s_policy.Ego2sSimdPolicy = original
    return observer, records


def _assert_parity(
    expected: Mapping[str, Any], observer: SeriesObserver, records: Sequence[Mapping[str, Any]]
) -> None:
    """Reject any divergence from the original probe before persisting a replay."""
    actual = {int(record["seed"]): record for record in records}
    required = set(_number_map(expected["mass"]))
    if set(actual) != required or len(observer.seeds) != len(actual):
        raise AssertionError("replay world set differs from saved probe row")
    for env, seed in enumerate(observer.seeds):
        record = actual[seed]
        if float(record["mass_integral"]) != float(_number_map(expected["mass"])[seed]):
            raise AssertionError(f"mass parity failed for world {seed}")
        comparisons = {
            "deaths": int(record["deaths"]),
            "alive_frames": int(observer.alive[env].sum()),
            "death_frame": (
                None if observer.death_frame[env] < 0 else int(observer.death_frame[env])
            ),
            "length_at_death": (
                None if observer.death_length[env] < 0 else int(observer.death_length[env])
            ),
        }
        for key, value in comparisons.items():
            if value != _number_map(expected[key])[seed]:
                raise AssertionError(f"{key} parity failed for world {seed}: {value!r}")
        integral = float(observer.length[env].sum()) / float(HORIZON)
        if integral != float(record["mass_integral"]):
            raise AssertionError(f"recorded series integral failed for world {seed}")


def replay(name: str, out: Path) -> Mapping[str, Any]:
    """Replay one arm/seed and atomically persist series plus identity receipt."""
    expected_path = row_path(name)
    expected = json.loads(expected_path.read_text())
    inputs = _input_identities(name, expected)
    decisions = (
        DisagreementObserver(int(name.rsplit("s", 1)[1]), HORIZON)
        if name.startswith("CTRL_")
        else None
    )
    observer, records = _run(name, expected, decisions)
    _assert_parity(expected, observer, records)
    if inputs != {path: sha256(Path(path)) for path in inputs}:
        raise RuntimeError("a bound replay input changed while the evaluator was running")
    npz = out / "series" / f"{name}.npz"
    arrays = {
        "seeds": np.asarray(observer.seeds, dtype=np.int64),
        "length": observer.length,
        "alive": observer.alive,
    }
    if decisions is not None:
        arrays.update({f"decision_{key}": value for key, value in decisions.arrays().items()})
    _atomic_npz(npz, **arrays)
    receipt = {
        "schema": "t3-diagnostics-replay/v1",
        "name": name,
        "checkpoint": str(checkpoint(name)),
        "checkpoint_sha256": sha256(checkpoint(name)),
        "source_row": str(expected_path),
        "source_row_sha256": sha256(expected_path),
        "series": str(npz),
        "series_sha256": sha256(npz),
        "worlds": len(observer.seeds),
        "frames": HORIZON,
        "parity": "exact mass/deaths/alive_frames/death_frame/length_at_death",
        "decision_observations": 0 if decisions is None else len(decisions.rows),
        "inputs_sha256": inputs,
    }
    _atomic_json(out / "receipts" / f"{name}.json", receipt)
    return receipt


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    """Return stable linear quartiles, with an explicit empty-event representation."""
    if not values:
        return {"q25": None, "q50": None, "q75": None}
    result = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
    return {"q25": float(result[0]), "q50": float(result[1]), "q75": float(result[2])}


def _series(out: Path, name: str) -> Mapping[str, np.ndarray]:
    path = out / "series" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _death_frames(series: Mapping[str, np.ndarray]) -> list[int]:
    """Return one-based terminal-death frames only; survivors are censored."""
    alive = np.asarray(series["alive"], dtype=bool)
    result: list[int] = []
    for row in alive:
        dead = np.flatnonzero(~row)
        if len(dead):
            result.append(int(dead[0]) + 1)
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _mass_summary(arm: str, out: Path) -> Mapping[str, Any]:
    """Exact paired mass identity and censored death/length descriptions for one arm."""
    per_seed = []
    all_arm_deaths: list[int] = []
    all_ctrl_deaths: list[int] = []
    sample_frames = (2000, 3000, 4000, 5000)
    length_pairs: dict[str, dict[str, list[float]]] = {
        str(frame): {"arm": [], "ctrl": []} for frame in sample_frames
    }
    for seed in SEEDS:
        candidate, control = _series(out, f"{arm}_s{seed}"), _series(out, f"CTRL_s{seed}")
        if not np.array_equal(candidate["seeds"], control["seeds"]):
            raise ValueError(f"world identity mismatch for {arm}_s{seed}")
        a, c = np.asarray(candidate["length"], dtype=np.int64), np.asarray(
            control["length"], dtype=np.int64
        )
        aa, ca = np.asarray(candidate["alive"], dtype=bool), np.asarray(
            control["alive"], dtype=bool
        )
        both, arm_only, ctrl_only = aa & ca, aa & ~ca, ca & ~aa
        total_num = int((a - c).sum())
        common_num = int(((a - c) * both).sum())
        arm_survival_num = int((a * arm_only).sum())
        ctrl_survival_num = int((c * ctrl_only).sum())
        if total_num != common_num + arm_survival_num - ctrl_survival_num:
            raise AssertionError(f"mass decomposition identity failed for {arm}_s{seed}")
        divisor = a.shape[0] * HORIZON
        total, common = float(total_num / divisor), float(common_num / divisor)
        arm_survival, ctrl_survival = float(arm_survival_num / divisor), float(
            ctrl_survival_num / divisor
        )
        arm_deaths, ctrl_deaths = _death_frames(candidate), _death_frames(control)
        full_survivors = aa[:, HORIZON - 1] & ca[:, HORIZON - 1]
        seed_lengths = {}
        for frame in sample_frames:
            index = frame - 1
            arm_values, ctrl_values = a[full_survivors, index], c[full_survivors, index]
            seed_lengths[str(frame)] = {
                "worlds": int(full_survivors.sum()),
                "arm_mean": _mean(arm_values.tolist()),
                "ctrl_mean": _mean(ctrl_values.tolist()),
                "arm_minus_ctrl_mean": _mean((arm_values - ctrl_values).tolist()),
            }
        per_seed.append(
            {
                "seed": seed,
                "total_delta": total,
                "common_alive_growth": common,
                "arm_only_survival": arm_survival,
                "ctrl_only_survival": ctrl_survival,
                "identity_residual_mass_frames": total_num
                - common_num
                - arm_survival_num
                + ctrl_survival_num,
                "deaths": {
                    "arm_count": len(arm_deaths),
                    "ctrl_count": len(ctrl_deaths),
                    "arm_frame_quartiles": _quantiles(arm_deaths),
                    "ctrl_frame_quartiles": _quantiles(ctrl_deaths),
                },
                "common_survivor_length_delta": seed_lengths,
            }
        )
        all_arm_deaths.extend(arm_deaths)
        all_ctrl_deaths.extend(ctrl_deaths)
        for frame in sample_frames:
            index = frame - 1  # required values are post-transition frames 2000, ...
            length_pairs[str(frame)]["arm"].extend(a[full_survivors, index].tolist())
            length_pairs[str(frame)]["ctrl"].extend(c[full_survivors, index].tolist())
    return {
        "per_seed": per_seed,
        "mean_components": {
            key: _mean([float(row[key]) for row in per_seed])
            for key in (
                "total_delta",
                "common_alive_growth",
                "arm_only_survival",
                "ctrl_only_survival",
            )
        },
        "deaths": {
            "arm_count": len(all_arm_deaths),
            "ctrl_count": len(all_ctrl_deaths),
            "arm_frame_quartiles": _quantiles(all_arm_deaths),
            "ctrl_frame_quartiles": _quantiles(all_ctrl_deaths),
            "survivors_censored": True,
        },
        "common_survivor_length_delta": {
            frame: {
                "worlds": len(values["arm"]),
                "arm_mean": _mean(values["arm"]),
                "ctrl_mean": _mean(values["ctrl"]),
                "arm_minus_ctrl_mean": _mean(
                    [left - right for left, right in zip(values["arm"], values["ctrl"])]
                ),
            }
            for frame, values in length_pairs.items()
        },
    }


def _length_bucket(length: int) -> str:
    if length < 400:
        return "lt400"
    if length < 700:
        return "400_699"
    if length < 1000:
        return "700_999"
    return "ge1000"


def _rate(rows: np.ndarray) -> Mapping[str, float | int | None]:
    count = int(len(rows))
    if not count:
        return {"decisions": 0, "disagreements": 0, "rate": None, "boost_disagreements": 0}
    return {
        "decisions": count,
        "disagreements": int(rows["disagree"].sum()),
        "rate": float(rows["disagree"].mean()),
        "boost_disagreements": int(rows["boost_disagree"].sum()),
    }


def _disagreement_summary(out: Path) -> Mapping[str, Any]:
    """Bucket CTRL trajectory disagreements using shared global raw-Q deciles."""
    rows = []
    for seed in SEEDS:
        source = _series(out, f"CTRL_s{seed}")
        for arm_id, arm in enumerate(("LABELS", "LABELS_KEEPV8")):
            selected = source["decision_arm"] == arm_id
            for index in np.flatnonzero(selected):
                rows.append(
                    tuple(source[f"decision_{key}"][index] for key in _DECISION_DTYPES)
                    + (seed, arm)
                )
    dtype = [(key, dtype) for key, dtype in _DECISION_DTYPES.items()] + [
        ("seed", np.int16),
        ("name", "U16"),
    ]
    data = np.asarray(rows, dtype=dtype)
    # Each CTRL decision appears once per shadow; use LABELS' copy only for cutpoints.
    base = data[data["name"] == "LABELS"]
    if not len(base):
        raise ValueError("no CTRL decision observations available")
    cutpoints = np.quantile(base["ctrl_neg_max_q"].astype(np.float64), np.arange(1, 10) / 10.0)
    result: dict[str, Any] = {
        "q_score": "-max of all six raw CTRL Q outputs",
        "global_decile_cutpoints": cutpoints.tolist(),
        "arms": {},
    }
    for arm in ("LABELS", "LABELS_KEEPV8"):
        selected = data[data["name"] == arm]
        by_length = {
            bucket: _rate(
                selected[np.array([_length_bucket(int(x)) == bucket for x in selected["length"]])]
            )
            for bucket in ("lt400", "400_699", "700_999", "ge1000")
        }
        decile = (
            np.searchsorted(cutpoints, selected["ctrl_neg_max_q"].astype(np.float64), side="right")
            + 1
        )
        by_decile = {str(number): _rate(selected[decile == number]) for number in range(1, 11)}
        per_seed = {}
        for seed in SEEDS:
            seed_rows = selected[selected["seed"] == seed]
            seed_decile = (
                np.searchsorted(
                    cutpoints, seed_rows["ctrl_neg_max_q"].astype(np.float64), side="right"
                )
                + 1
            )
            per_seed[str(seed)] = {
                "overall": _rate(seed_rows),
                "by_decile": {
                    str(number): _rate(seed_rows[seed_decile == number]) for number in range(1, 11)
                },
            }
        result["arms"][arm] = {
            "overall": _rate(selected),
            "by_length": by_length,
            "by_decile": by_decile,
            "per_seed": per_seed,
        }
    return result


def summarize(out: Path) -> Mapping[str, Any]:
    """Summarize all twenty hash-bound replay receipts for the report renderer."""
    names = [f"CTRL_s{seed}" for seed in SEEDS] + [
        f"{arm}_s{seed}" for arm in LABEL_ARMS for seed in SEEDS
    ]
    receipts = {}
    for name in names:
        path = out / "receipts" / f"{name}.json"
        receipt = json.loads(path.read_text())
        expected_series = (out / "series" / f"{name}.npz").resolve()
        expected_inputs = _input_identities(name, json.loads(row_path(name).read_text()))
        if (
            receipt.get("schema") != "t3-diagnostics-replay/v1"
            or receipt.get("name") != name
            or receipt.get("frames") != HORIZON
            or receipt.get("worlds") != 96
            or Path(receipt.get("series", "")).resolve() != expected_series
            or receipt.get("series_sha256") != sha256(expected_series)
            or receipt.get("inputs_sha256") != expected_inputs
        ):
            raise ValueError(f"series hash mismatch: {name}")
        receipts[name] = receipt
    result = {
        "schema": "t3-diagnostics-summary/v1",
        "authority": "development diagnostic, not a T3 decision",
        "replays": receipts,
        "mass": {arm: _mass_summary(arm, out) for arm in LABEL_ARMS},
        "disagreements": _disagreement_summary(out),
    }
    _atomic_json(out / "summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.name) == bool(args.summarize):
        raise SystemExit("provide exactly one of --name or --summarize")
    if args.summarize:
        result = summarize(args.out)
        print(
            json.dumps({"summary": str(args.out / "summary.json"), "schema": result["schema"]}),
            flush=True,
        )
        return 0
    valid = {f"CTRL_s{seed}" for seed in SEEDS}
    valid |= {f"{arm}_s{seed}" for arm in LABEL_ARMS for seed in SEEDS}
    if args.name not in valid:
        raise SystemExit(f"unknown completed T3 arm: {args.name}")
    print(json.dumps(replay(args.name, args.out), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
