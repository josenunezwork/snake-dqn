#!/usr/bin/env python3
"""Hash-bound T3 boost reconciliation on exact round-62 trajectories.

The T3 probes execute the student with the advisory mask, while training
telemetry's ``greedy_boost_rate`` used the resolved mask.  This tool records
both (and the raw six-action argmax) on each final's own trajectory, so the
two possible explanations are distinguishable without changing a trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# Set before NumPy/Torch imports.  A caller's MPS setting is never admissible
# for this identity-sensitive diagnostic.
os.environ["SNAKE_DQN_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np  # noqa: E402

from research.t3_diagnostics_20261016 import replay as t3_replay  # noqa: E402

ARMS = ("CTRL", "LABELS", "LABELS_KEEPV8")
BUCKETS = ("lt400", "400_699", "700_999", "ge1000")


def sha256(path: Path) -> str:
    """Return SHA-256 for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _greedy(q: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """First-max action, optionally under ``mask`` with T3's normal fallback."""
    if mask is None:
        return np.asarray(q, dtype=np.float64).argmax(axis=1).astype(np.int8)
    valid = np.asarray(mask, dtype=bool).copy()
    valid[~valid.any(axis=1), :3] = True
    return np.where(valid, np.asarray(q, dtype=np.float64), -np.inf).argmax(axis=1).astype(np.int8)


def _bucket(length: np.ndarray) -> np.ndarray:
    """Return the fixed four post-hoc length-bucket codes."""
    result = np.full(len(length), 3, dtype=np.int8)
    result[np.asarray(length) < 400] = 0
    result[(np.asarray(length) >= 400) & (np.asarray(length) < 700)] = 1
    result[(np.asarray(length) >= 700) & (np.asarray(length) < 1000)] = 2
    return result


class BoostObserver:
    """Capture every live hero's raw, resolved, and advisory greed pre-action."""

    def __init__(self, checkpoint_name: str) -> None:
        from src.model.ego2s_network import load_ego2s_checkpoint

        self.network = load_ego2s_checkpoint(str(t3_replay.checkpoint(checkpoint_name)), "cpu")
        self.rows: list[tuple[int, int, int, int, int, int]] = []

    def observe(self, sim: Any, slots: np.ndarray, q: np.ndarray) -> None:
        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        alive = sim.get_alive()[rows[:, 0], rows[:, 1]].astype(bool)
        if not alive.any():
            return
        lengths = sim.get_lengths()[rows[:, 0], rows[:, 1]]
        resolved = sim.get_resolved_action_mask()[rows[:, 0], rows[:, 1]]
        advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
        raw_action = _greedy(q, None)
        resolved_action = _greedy(q, resolved)
        advisory_action = _greedy(q, advisory)
        for index in np.flatnonzero(alive):
            self.rows.append(
                (
                    int(rows[index, 0]),
                    int(sim.frame[rows[index, 0]]),
                    int(lengths[index]),
                    int(raw_action[index]),
                    int(resolved_action[index]),
                    int(advisory_action[index]),
                )
            )

    def arrays(self) -> Mapping[str, np.ndarray]:
        columns = {
            "env": (0, np.int16),
            "frame": (1, np.int16),
            "length": (2, np.int32),
            "raw_action": (3, np.int8),
            "resolved_action": (4, np.int8),
            "advisory_action": (5, np.int8),
        }
        return {
            key: (
                np.asarray([row[index] for row in self.rows], dtype=dtype)
                if self.rows
                else np.empty(0, dtype=dtype)
            )
            for key, (index, dtype) in columns.items()
        }


def _run(trajectory: str, student: str, expected: Mapping[str, Any]) -> BoostObserver:
    """Replay one probe exactly, observing its executing student before actions."""
    import dataclasses

    import torch

    from research.redesign_m2_20261008 import dev_probe, ni_spec
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.core.device_manager import DeviceManager
    from src.simd_env import ego2s_policy, eval_engine

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    DeviceManager.override_device(torch.device("cpu"))
    if expected["v61_backend"] != "numba" or expected["anchors"] != "fast":
        raise ValueError("saved probe must use numba and fast anchors")
    if expected["ego2s_forward"] != "rowwise" or expected["ego2s_mask"] != "advisory":
        raise ValueError("saved probe must use rowwise/advisory student selection")
    if (int(expected["frames"]), int(expected["round_index"]), expected["mix"]) != (
        t3_replay.HORIZON,
        62,
        "scripted",
    ):
        raise ValueError("only round-62 H5000 scripted rows are valid inputs")

    seeds = dev_probe.probe_seeds(int(expected["worlds"]), 62)
    ctx = gid._context(1)
    profile = dataclasses.replace(ctx["profile"], scored_horizon=t3_replay.HORIZON)
    rosters = {
        int(row["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in row["slots"]]
        for row in ctx["ds"]._design_rows(seeds)
        if row["mix"] == "scripted"
    }
    if set(rosters) != set(seeds):
        raise AssertionError("scripted roster construction did not cover every saved world")

    observer = BoostObserver(student)
    series = t3_replay.SeriesObserver(seeds, t3_replay.HORIZON)
    original = ego2s_policy.Ego2sSimdPolicy

    class ObservedPolicy(original):
        def actions(self, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
            q = ego2s_policy.student_q(observer.network, sim, slots, self.forward).numpy()
            observer.observe(sim, slots, q)
            executing_q = (
                q
                if trajectory == student
                else ego2s_policy.student_q(self.net, sim, slots, self.forward).numpy()
            )
            if self.mask_mode == "advisory":
                rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
                mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
            else:
                mask = masks
            return _greedy(executing_q, mask).astype(np.int64)

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
            hero_ego2s=str(t3_replay.checkpoint(trajectory)),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            frame_observer=series,
        )
    finally:
        ego2s_policy.Ego2sSimdPolicy = original
        DeviceManager.reset_for_testing()
    # The trajectory is admissible only if the observer was passive.
    t3_replay._assert_parity(expected, series, records)
    return observer


def _bound_inputs(trajectory: str, student: str) -> tuple[Mapping[str, Any], Mapping[str, str]]:
    """Validate source-row/checkpoint identities and reconstruct all input hashes."""
    if not all(
        any(value.startswith(f"{arm}_s") for arm in ARMS) for value in (trajectory, student)
    ):
        raise ValueError(f"unsupported final pair: {trajectory}, {student}")
    if trajectory.rsplit("s", 1)[1] != student.rsplit("s", 1)[1]:
        raise ValueError("trajectory and student must use the same training seed")
    expected_path = t3_replay.row_path(trajectory)
    expected = json.loads(expected_path.read_text())
    student_row_path = t3_replay.row_path(student)
    student_expected = json.loads(student_row_path.read_text())
    student_checkpoint = t3_replay.checkpoint(student)
    if Path(student_expected["checkpoint"]) != student_checkpoint or student_expected[
        "checkpoint_sha256"
    ] != sha256(student_checkpoint):
        raise ValueError("student checkpoint identity does not match its saved probe row")
    inputs = dict(t3_replay._input_identities(trajectory, expected))
    inputs[str(student_checkpoint)] = sha256(student_checkpoint)
    inputs[str(student_row_path)] = sha256(student_row_path)
    return expected, inputs


def capture(trajectory: str, student: str, out: Path) -> Mapping[str, Any]:
    """Capture ``student`` greed on the exact trajectory policy's probe states."""
    expected, inputs = _bound_inputs(trajectory, student)
    expected_path = t3_replay.row_path(trajectory)
    student_row_path = t3_replay.row_path(student)
    student_checkpoint = t3_replay.checkpoint(student)
    observer = _run(trajectory, student, expected)
    if inputs != {path: sha256(Path(path)) for path in inputs}:
        raise RuntimeError("a bound input changed while boost capture was running")
    arrays = observer.arrays()
    if not len(arrays["length"]):
        raise RuntimeError("capture observed no live hero decisions")
    path = out / "boost" / f"{student}__on__{trajectory}.npz"
    _atomic_npz(path, **arrays)
    receipt = {
        "schema": "t4a-posthoc-boost/v1",
        "student": student,
        "trajectory": trajectory,
        "checkpoint": str(student_checkpoint),
        "checkpoint_sha256": sha256(student_checkpoint),
        "trajectory_source_row": str(expected_path),
        "trajectory_source_row_sha256": sha256(expected_path),
        "student_source_row": str(student_row_path),
        "student_source_row_sha256": sha256(student_row_path),
        "decisions": int(len(arrays["length"])),
        "array": str(path),
        "array_sha256": sha256(path),
        "inputs_sha256": inputs,
        "selection": "round62 scripted H5000 rowwise advisory",
        "telemetry_greedy_mask": "resolved with normal-only fallback",
    }
    _atomic_json(out / "receipts" / f"boost-{student}__on__{trajectory}.json", receipt)
    return receipt


def _rate(actions: np.ndarray, select: np.ndarray) -> Mapping[str, Any]:
    count = int(select.sum())
    boost = int((np.asarray(actions)[select] >= 3).sum())
    return {"decisions": count, "boost_argmax": boost, "rate": None if not count else boost / count}


def summarize_arrays(arrays: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    """Summarize all action definitions by the registered four length buckets."""
    length = np.asarray(arrays["length"], dtype=np.int64)
    code = _bucket(length)
    modes = ("raw_action", "resolved_action", "advisory_action")
    result: dict[str, Any] = {
        "decisions": int(len(length)),
        "overall": {
            mode.removesuffix("_action"): _rate(
                np.asarray(arrays[mode]), np.ones(len(length), bool)
            )
            for mode in modes
        },
        "by_length": {},
    }
    for index, bucket in enumerate(BUCKETS):
        selected = code == index
        result["by_length"][bucket] = {
            "own_decision_share": None if not len(length) else float(selected.mean()),
            **{
                mode.removesuffix("_action"): _rate(np.asarray(arrays[mode]), selected)
                for mode in modes
            },
        }
    return result


def _load_summary(out: Path, student: str, trajectory: str) -> Mapping[str, Any]:
    """Verify and load one persisted capture before reporting it."""
    array_path = out / "boost" / f"{student}__on__{trajectory}.npz"
    receipt_path = out / "receipts" / f"boost-{student}__on__{trajectory}.json"
    if not array_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing capture or receipt for {student} on {trajectory}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "t4a-posthoc-boost/v1":
        raise ValueError(f"unexpected receipt schema for {student} on {trajectory}")
    if receipt.get("student") != student or receipt.get("trajectory") != trajectory:
        raise ValueError(f"receipt identity mismatch for {student} on {trajectory}")
    if Path(receipt.get("array", "")) != array_path:
        raise ValueError(f"receipt array path mismatch for {student} on {trajectory}")
    if receipt.get("selection") != "round62 scripted H5000 rowwise advisory":
        raise ValueError(f"unexpected selection contract for {student} on {trajectory}")
    if receipt.get("telemetry_greedy_mask") != "resolved with normal-only fallback":
        raise ValueError(f"unexpected telemetry mask contract for {student} on {trajectory}")
    if receipt["array_sha256"] != sha256(array_path):
        raise ValueError(f"array hash mismatch for {student} on {trajectory}")
    _, expected_bindings = _bound_inputs(trajectory, student)
    if receipt.get("inputs_sha256") != expected_bindings:
        raise ValueError(f"input binding mismatch for {student} on {trajectory}")
    with np.load(array_path, allow_pickle=False) as source:
        return summarize_arrays({key: source[key].copy() for key in source.files})


def summarize(out: Path, destination: Path) -> Mapping[str, Any]:
    """Combine captures into own-trajectory and matched CTRL-trajectory tables."""
    results: dict[str, Any] = {"schema": "t4a-posthoc-boost-summary/v1", "arms": {}, "receipts": {}}
    for arm in ARMS:
        own: dict[str, Any] = {}
        for seed in t3_replay.SEEDS:
            student = f"{arm}_s{seed}"
            own[str(seed)] = _load_summary(out, student, student)
            results["receipts"][f"{student}__on__{student}"] = {
                "path": str(out / "receipts" / f"boost-{student}__on__{student}.json"),
                "sha256": sha256(out / "receipts" / f"boost-{student}__on__{student}.json"),
            }
            results["receipts"][f"{student}__on__CTRL_s{seed}"] = {
                "path": str(out / "receipts" / f"boost-{student}__on__CTRL_s{seed}.json"),
                "sha256": sha256(out / "receipts" / f"boost-{student}__on__CTRL_s{seed}.json"),
            }
        results["arms"][arm] = {
            "own_trajectory": own,
            "ctrl_trajectory": {
                str(seed): _load_summary(out, f"{arm}_s{seed}", f"CTRL_s{seed}")
                for seed in t3_replay.SEEDS
            },
        }
    _atomic_json(destination, results)
    return results


def _percent(rate: float | None) -> str:
    """Format a possibly empty rate for Markdown."""
    return "—" if rate is None else f"{100 * rate:.2f}%"


def _pooled(seed_summaries: Mapping[str, Any], bucket: str, mode: str) -> Mapping[str, Any]:
    """Pool integer counts across the five final seeds without averaging rates."""
    values = [
        (
            seed_summaries[str(seed)]["overall"]
            if bucket == "all"
            else seed_summaries[str(seed)]["by_length"][bucket]
        )
        for seed in t3_replay.SEEDS
    ]
    decisions = sum(int(value[mode]["decisions"]) for value in values)
    boost = sum(int(value[mode]["boost_argmax"]) for value in values)
    all_decisions = sum(int(seed_summaries[str(seed)]["decisions"]) for seed in t3_replay.SEEDS)
    return {
        "decisions": decisions,
        "boost_argmax": boost,
        "rate": None if not decisions else boost / decisions,
        "decision_share": None if not all_decisions else decisions / all_decisions,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    """Render the Part-1 appendix table from a hash-verified summary JSON."""
    lines = [
        "## 4. Boost reconciliation (T4a post-hoc)",
        "",
        "Training telemetry `greedy_boost_rate` counts the greedy action under the "
        "**resolved** action mask, after its normal-only fallback, before exploration or any "
        "shield/veto choice (`train_m3c.py`). The round-62 probes execute with the advisory "
        "mask. The primary rate below therefore uses resolved greed; raw and advisory rates "
        "are retained in the hash-bound JSON as definition checks.",
        "",
        "Each row pools the five matching final seeds by integer decision counts. `Own share` "
        "is the share of the named policy's own live decisions in that length bucket; CTRL "
        "trajectory rates evaluate the named final on matching-seed CTRL observations and do "
        "not alter CTRL's executed actions.",
        "",
        "| Arm | States | Length | Decisions | Own share | Resolved boost argmax | "
        "Rate | Raw rate | Advisory rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"lt400": "<400", "400_699": "400–<700", "700_999": "700–<1000", "ge1000": "≥1000"}
    for arm in ARMS:
        for state_key, state_label in (("own_trajectory", "own"), ("ctrl_trajectory", "CTRL")):
            per_seed = summary["arms"][arm][state_key]
            for bucket in BUCKETS:
                resolved = _pooled(per_seed, bucket, "resolved")
                raw = _pooled(per_seed, bucket, "raw")
                advisory = _pooled(per_seed, bucket, "advisory")
                own_share = resolved["decision_share"] if state_key == "own_trajectory" else None
                lines.append(
                    "| {arm} | {states} | {bucket} | {n} | {share} | {boost} | {rate} | {raw} | "
                    "{advisory} |".format(
                        arm=arm,
                        states=state_label,
                        bucket=labels[bucket],
                        n=resolved["decisions"],
                        share=_percent(own_share),
                        boost=resolved["boost_argmax"],
                        rate=_percent(resolved["rate"]),
                        raw=_percent(raw["rate"]),
                        advisory=_percent(advisory["rate"]),
                    )
                )
    lines += [
        "",
        "Per-seed resolved overall rates (for audit, not a new inferential analysis):",
        "",
        "| Arm | Seed | Own resolved rate | CTRL-trajectory resolved rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        own = summary["arms"][arm]["own_trajectory"]
        ctrl = summary["arms"][arm]["ctrl_trajectory"]
        for seed in t3_replay.SEEDS:
            lines.append(
                f"| {arm} | {seed} | {_percent(own[str(seed)]['overall']['resolved']['rate'])} | "
                f"{_percent(ctrl[str(seed)]['overall']['resolved']['rate'])} |"
            )
    lines += [
        "",
        "Pooled rates above are descriptive and the five seeds reuse each arm's "
        "96 scripted worlds.",
        "",
    ]
    return "\n".join(lines)


def append_report(path: Path, summary: Mapping[str, Any]) -> None:
    """Append a generated Part-1 table once, refusing an accidental duplicate section."""
    marker = "## 4. Boost reconciliation (T4a post-hoc)"
    text = path.read_text()
    if marker in text:
        raise ValueError(f"report already contains {marker!r}")
    path.write_text(text.rstrip() + "\n\n" + render_report(summary))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one capture or summarize a complete capture directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--trajectory", required=True)
    capture_parser.add_argument("--student", required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--out", type=Path, required=True)
    summary_parser.add_argument("--json", type=Path, required=True)
    summary_parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        value = capture(args.trajectory, args.student, args.out)
    else:
        value = summarize(args.out, args.json)
        if args.report:
            append_report(args.report, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
