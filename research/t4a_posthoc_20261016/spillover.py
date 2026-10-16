#!/usr/bin/env python3
"""Summarize the pre-declared T4a spillover and shared-survivor diagnostics.

This module deliberately only consumes captured raw-Q decision rows and replay
series.  It does not train, select a study arm, or confer any gate authority.
The capture producer must write all-six *raw* Q outputs, as the pre-registration
defines argmax and ``-max Q`` without the advisory action mask.

``--bank-rows`` is an NPZ with equal-length ``side`` (``decisive`` or
``counter``), ``length``, ``fingerprint``, ``ctrl_q`` and ``candidate_q`` arrays
of shape ``(n, 6)`` for each Q field.
``--trajectory-rows`` additionally requires ``length`` and ``fingerprint``.
Fingerprints are canonical exact-observation identities.  States whose
fingerprint occurs in the bank are excluded from readout (c), so this condition
cannot silently become a claim about states outside the bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
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

SCHEMA = "t4a-posthoc-spillover/v1"
LENGTH_BUCKETS = ("lt400", "400_699", "700_999", "ge1000")
GROWTH_FRAMES = (1000, 2000, 3000, 4000, 5000)


def sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def length_bucket(length: np.ndarray | int) -> np.ndarray | str:
    """Return the registered post-hoc length bucket(s)."""
    values = np.asarray(length)
    result = np.full(values.shape, "ge1000", dtype="U8")
    result[values < 1000] = "700_999"
    result[values < 700] = "400_699"
    result[values < 400] = "lt400"
    return str(result.item()) if values.ndim == 0 else result


def deciles(scores: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Assign score deciles using the established right-closed quantile convention."""
    points = np.asarray(cutpoints, dtype=np.float64)
    if points.shape != (9,) or not np.isfinite(points).all() or np.any(np.diff(points) < 0):
        raise ValueError("decile cutpoints must be nine finite nondecreasing values")
    return np.searchsorted(points, np.asarray(scores, dtype=np.float64), side="right") + 1


def _validate_q(ctrl_q: np.ndarray, candidate_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ctrl, candidate = np.asarray(ctrl_q), np.asarray(candidate_q)
    if ctrl.ndim != 2 or ctrl.shape[1] != 6 or ctrl.shape != candidate.shape:
        raise ValueError("ctrl_q and candidate_q must have matching shape (n, 6)")
    if not len(ctrl) or not np.isfinite(ctrl).all() or not np.isfinite(candidate).all():
        raise ValueError("raw Q rows must be nonempty and finite")
    return ctrl.astype(np.float64, copy=False), candidate.astype(np.float64, copy=False)


def action_rates(ctrl_q: np.ndarray, candidate_q: np.ndarray) -> dict[str, int | float]:
    """Compute raw six-action argmax agreement and the separate cross-boost readout."""
    ctrl, candidate = _validate_q(ctrl_q, candidate_q)
    ctrl_action, candidate_action = ctrl.argmax(axis=1), candidate.argmax(axis=1)
    changed = candidate_action != ctrl_action
    cross_boost = changed & ((candidate_action >= 3) != (ctrl_action >= 3))
    changes = int(changed.sum())
    return {
        "decisions": int(len(ctrl)),
        "argmax_changes": changes,
        "argmax_change_rate": float(changed.mean()),
        "argmax_agreement": float((~changed).mean()),
        "cross_boost_changes": int(cross_boost.sum()),
        "cross_boost_rate_all_decisions": float(cross_boost.mean()),
        "cross_boost_share_of_argmax_changes": (
            None if not changes else float(cross_boost.sum() / changes)
        ),
    }


def _rows_summary(
    ctrl_q: np.ndarray,
    candidate_q: np.ndarray,
    lengths: np.ndarray | None,
    cutpoints: np.ndarray,
) -> dict[str, Any]:
    """Report the declared all-row, length-by-decile, and decile readouts."""
    ctrl, candidate = _validate_q(ctrl_q, candidate_q)
    score = -ctrl.max(axis=1)
    assigned_deciles = deciles(score, cutpoints)
    by_decile = {
        str(number): (
            action_rates(ctrl[assigned_deciles == number], candidate[assigned_deciles == number])
            if np.any(assigned_deciles == number)
            else _empty_rates()
        )
        for number in range(1, 11)
    }
    result: dict[str, Any] = {
        "overall": action_rates(ctrl, candidate),
        "by_decile": by_decile,
    }
    if lengths is not None:
        length_values = np.asarray(lengths)
        if length_values.shape != (len(ctrl),) or not np.isfinite(length_values).all():
            raise ValueError("length must be a finite vector aligned with raw Q rows")
        bucket = length_bucket(length_values)
        result["by_length_and_decile"] = {
            name: {
                str(number): (
                    action_rates(
                        ctrl[(bucket == name) & (assigned_deciles == number)],
                        candidate[(bucket == name) & (assigned_deciles == number)],
                    )
                    if np.any((bucket == name) & (assigned_deciles == number))
                    else _empty_rates()
                )
                for number in range(1, 11)
            }
            for name in LENGTH_BUCKETS
        }
    return result


def _empty_rates() -> dict[str, int | float | None]:
    return {
        "decisions": 0,
        "argmax_changes": 0,
        "argmax_change_rate": None,
        "argmax_agreement": None,
        "cross_boost_changes": 0,
        "cross_boost_rate_all_decisions": None,
        "cross_boost_share_of_argmax_changes": None,
    }


def _strata(lengths: np.ndarray, ctrl_q: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Return a stable registered length-bucket × CTRL-decile stratum key."""
    return np.char.add(
        np.char.add(length_bucket(lengths).astype("U8"), ":D"),
        deciles(-np.asarray(ctrl_q).max(axis=1), cutpoints).astype("U2"),
    )


def _matched_rows_summary(
    ctrl_q: np.ndarray,
    candidate_q: np.ndarray,
    lengths: np.ndarray,
    cutpoints: np.ndarray,
    support: set[str],
) -> dict[str, Any]:
    """Restrict one readout to the declared three-way common stratum support."""
    labels = _strata(lengths, ctrl_q, cutpoints)
    keep = np.asarray([item in support for item in labels], dtype=bool)
    if not keep.any():
        raise ValueError("a/b/c have no common length-bucket and CTRL-decile support")
    return {
        "support": sorted(support),
        "retained_rows": int(keep.sum()),
        "excluded_outside_common_support": int((~keep).sum()),
        "rates": _rows_summary(ctrl_q[keep], candidate_q[keep], lengths[keep], cutpoints),
    }


def spillover_summary(
    bank: Mapping[str, np.ndarray], trajectory: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Compute declared a/b/c spillover rows with exact-bank exclusion for (c)."""
    for source, required in (
        (bank, ("side", "length", "ctrl_q", "candidate_q", "fingerprint")),
        (trajectory, ("length", "ctrl_q", "candidate_q", "fingerprint")),
    ):
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"missing required capture fields: {missing}")
    ctrl_bank, candidate_bank = _validate_q(bank["ctrl_q"], bank["candidate_q"])
    side = np.asarray(bank["side"]).astype("U16")
    bank_lengths = np.asarray(bank["length"])
    fingerprints = np.asarray(bank["fingerprint"]).astype("U128")
    if (
        side.shape != (len(ctrl_bank),)
        or fingerprints.shape != (len(ctrl_bank),)
        or bank_lengths.shape != (len(ctrl_bank),)
    ):
        raise ValueError("bank side/length/fingerprint fields must align with bank Q rows")
    if (
        set(np.unique(side)) - {"decisive", "counter"}
        or not np.any(side == "decisive")
        or not np.any(side == "counter")
    ):
        raise ValueError("bank side must contain both decisive and counter rows")
    ctrl_trajectory, candidate_trajectory = _validate_q(
        trajectory["ctrl_q"], trajectory["candidate_q"]
    )
    lengths = np.asarray(trajectory["length"])
    trajectory_fingerprints = np.asarray(trajectory["fingerprint"]).astype("U128")
    if trajectory_fingerprints.shape != (len(ctrl_trajectory),):
        raise ValueError("trajectory fingerprint must align with trajectory Q rows")
    if not len(np.unique(fingerprints)):
        raise ValueError("bank fingerprint cannot be empty")
    cutpoints = np.quantile(-ctrl_trajectory.max(axis=1), np.arange(1, 10) / 10.0)
    outside = ~np.isin(trajectory_fingerprints, np.unique(fingerprints))
    if not outside.any():
        raise ValueError("no CTRL-trajectory states remain outside the exact bank fingerprint set")
    a = side == "decisive"
    b = side == "counter"
    c = outside
    common_support = (
        set(_strata(bank_lengths[a], ctrl_bank[a], cutpoints))
        & set(_strata(bank_lengths[b], ctrl_bank[b], cutpoints))
        & set(_strata(lengths[c], ctrl_trajectory[c], cutpoints))
    )
    result = {
        "schema": SCHEMA,
        "q_convention": "raw six-action argmax; score is -max of raw CTRL Q",
        "cutpoints_source": (
            "all captured CTRL-trajectory decision rows before exact-bank exclusion"
        ),
        "global_decile_cutpoints": cutpoints.tolist(),
        "a_decisive_bank_rows": _rows_summary(
            ctrl_bank[a], candidate_bank[a], bank_lengths[a], cutpoints
        ),
        "b_counter_bank_rows": _rows_summary(
            ctrl_bank[b], candidate_bank[b], bank_lengths[b], cutpoints
        ),
        "c_ctrl_trajectory_outside_bank_matched_length_and_decile": _rows_summary(
            ctrl_trajectory[outside], candidate_trajectory[outside], lengths[outside], cutpoints
        ),
        "c_exclusion": {
            "method": "exact canonical observation fingerprint membership",
            "trajectory_rows_total": int(len(ctrl_trajectory)),
            "excluded_bank_fingerprint_rows": int((~outside).sum()),
            "outside_bank_rows": int(outside.sum()),
        },
        "matched_common_support": {
            "method": (
                "restriction to length-bucket × common CTRL -max-Q-decile support shared by a/b/c; "
                "no extrapolation or invented threshold"
            ),
            "a_decisive_bank_rows": _matched_rows_summary(
                ctrl_bank[a], candidate_bank[a], bank_lengths[a], cutpoints, common_support
            ),
            "b_counter_bank_rows": _matched_rows_summary(
                ctrl_bank[b], candidate_bank[b], bank_lengths[b], cutpoints, common_support
            ),
            "c_ctrl_trajectory_outside_bank_rows": _matched_rows_summary(
                ctrl_trajectory[c], candidate_trajectory[c], lengths[c], cutpoints, common_support
            ),
        },
    }
    return result


def growth_summary(candidate_series: Path, ctrl_series: Path) -> dict[str, Any]:
    """Compute frame-specific and fixed-H5000 shared-survivor mean-length curves."""
    with (
        np.load(candidate_series, allow_pickle=False) as candidate,
        np.load(ctrl_series, allow_pickle=False) as ctrl,
    ):
        c, k = {name: candidate[name].copy() for name in candidate.files}, {
            name: ctrl[name].copy() for name in ctrl.files
        }
    required = {"seeds", "length", "alive"}
    if not required <= c.keys() or not required <= k.keys():
        raise ValueError("series need seeds, length, and alive arrays")
    if (
        not np.array_equal(c["seeds"], k["seeds"])
        or c["length"].shape != k["length"].shape
        or c["alive"].shape != k["alive"].shape
    ):
        raise ValueError("candidate/CTRL series must have identical world identities and shapes")
    if (
        c["length"].shape != c["alive"].shape
        or c["length"].ndim != 2
        or c["length"].shape[1] < max(GROWTH_FRAMES)
    ):
        raise ValueError("series must cover all frames 1000 through 5000")
    curve, fixed_curve = {}, {}
    fixed = np.asarray(c["alive"][:, 4999], dtype=bool) & np.asarray(
        k["alive"][:, 4999], dtype=bool
    )
    for frame in GROWTH_FRAMES:
        index = frame - 1
        survivors = np.asarray(c["alive"][:, index], dtype=bool) & np.asarray(
            k["alive"][:, index], dtype=bool
        )
        a, b = c["length"][survivors, index], k["length"][survivors, index]
        curve[str(frame)] = {
            "shared_survivors": int(survivors.sum()),
            "candidate_mean_alive_length": None if not len(a) else float(np.mean(a)),
            "ctrl_mean_alive_length": None if not len(b) else float(np.mean(b)),
            "candidate_minus_ctrl_mean_alive_length": None if not len(a) else float(np.mean(a - b)),
        }
        fixed_a, fixed_b = c["length"][fixed, index], k["length"][fixed, index]
        fixed_curve[str(frame)] = {
            "shared_survivors": int(fixed.sum()),
            "candidate_mean_alive_length": None if not len(fixed_a) else float(np.mean(fixed_a)),
            "ctrl_mean_alive_length": None if not len(fixed_b) else float(np.mean(fixed_b)),
            "candidate_minus_ctrl_mean_alive_length": (
                None if not len(fixed_a) else float(np.mean(fixed_a - fixed_b))
            ),
        }
    return {
        "frames": curve,
        "cohort": "paired worlds alive under both candidate and CTRL at each named frame",
        "fixed_h5000_shared_survivor_frames": fixed_curve,
        "fixed_h5000_cohort": (
            "paired worlds alive under both candidate and CTRL at frame 5000; "
            "supplied for T3 contrast"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically persist unpickled numerical capture rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def observation_fingerprints(
    local: np.ndarray, global_: np.ndarray, scalars: np.ndarray
) -> np.ndarray:
    """Return exact, layout-tagged fingerprints shared by bank and trajectory captures."""
    if len(local) != len(global_) or len(local) != len(scalars):
        raise ValueError("observation components must have the same row count")
    result = []
    for index in range(len(local)):
        digest = hashlib.sha256()
        for name, value in (
            ("local", local[index]),
            ("global", global_[index]),
            ("scalars", scalars[index]),
        ):
            array = np.asarray(value)
            digest.update(name.encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
        result.append(digest.hexdigest())
    return np.asarray(result, dtype="U64")


def capture_bank_rows(
    bank_dir: Path,
    ctrl_checkpoint: Path,
    candidate_checkpoint: Path,
    out: Path,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Capture hash-bound raw-Q rows for both immutable T4a bank sides on CPU.

    The implementation reads the v2 bank directly so its stored ``length`` and
    ``surviving_original`` provenance remain the side authority.  It uses one
    featurized bank observation for the two network forwards; no simulation or
    learner is started.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import torch

    from research.redesign_m3c_20261008.obs_pack import train_layout
    from research.t3_rollout_labels_20261015.bank import LabelBank
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego2s_policy import forward_q

    bank_dir, ctrl_checkpoint, candidate_checkpoint = map(
        Path, (bank_dir, ctrl_checkpoint, candidate_checkpoint)
    )
    manifest = bank_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing T4a bank manifest: {manifest}")
    metadata = json.loads(manifest.read_text())
    if metadata.get("schema") != "t4-derived-label-bank/v2":
        raise ValueError("bank must be the immutable t4-derived-label-bank/v2")
    shards = sorted(bank_dir.glob("*.npz"))
    if not shards or not ctrl_checkpoint.is_file() or not candidate_checkpoint.is_file():
        raise FileNotFoundError("bank shards or final checkpoints are missing")
    declared_shards = metadata.get("shards")
    if not isinstance(declared_shards, Mapping) or set(declared_shards) != {
        path.name for path in shards
    }:
        raise ValueError("T4a v2 manifest must bind exactly the captured bank shards")
    for path in shards:
        if (
            not isinstance(declared_shards[path.name], str)
            or sha256(path) != declared_shards[path.name]
        ):
            raise ValueError(f"T4a v2 shard digest disagrees with manifest: {path.name}")
    # Bind before loading networks or doing any forward pass, then repeat this
    # identical map after capture so an input mutation cannot be concealed.
    inputs = {
        str(manifest): sha256(manifest),
        str(ctrl_checkpoint): sha256(ctrl_checkpoint),
        str(candidate_checkpoint): sha256(candidate_checkpoint),
    }
    inputs.update({str(path): sha256(path) for path in shards})
    # LabelBank enforces the stored-observation and raw-side shape contracts.
    bank = LabelBank(bank_dir, 60, pack=train_layout())
    lengths = []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as source:
            if "length" not in source.files:
                raise ValueError(f"T4a bank shard lacks stored length: {shard}")
            lengths.append(source["length"].copy())
    length_values = np.concatenate(lengths)
    if length_values.shape != (bank.size,):
        raise ValueError("T4a bank length rows do not align with observations")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    ctrl_net = load_ego2s_checkpoint(str(ctrl_checkpoint), "cpu").eval()
    candidate_net = load_ego2s_checkpoint(str(candidate_checkpoint), "cpu").eval()
    ctrl_parts, candidate_parts = [], []
    for start in range(0, bank.size, batch_size):
        stop = min(start + batch_size, bank.size)
        observations = {
            "local": bank.local[start:stop],
            "global": bank.global_[start:stop],
            "scalars": bank.scalars[start:stop],
        }
        # Mirror CTRL-trajectory capture's rowwise all-six Q convention.  The
        # batch partition only bounds memory; it does not alter a row's Q.
        ctrl_parts.append(forward_q(ctrl_net, observations, "rowwise").numpy())
        candidate_parts.append(forward_q(candidate_net, observations, "rowwise").numpy())
    if inputs != {key: sha256(Path(key)) for key in inputs}:
        raise RuntimeError("bank or checkpoint changed during raw-Q capture")
    ctrl_values = np.concatenate(ctrl_parts).astype(np.float32, copy=False)
    candidate_values = np.concatenate(candidate_parts).astype(np.float32, copy=False)
    rows = np.arange(bank.size)
    counter = bank.counter
    ctrl_actions, candidate_actions = ctrl_values.argmax(axis=1), candidate_values.argmax(axis=1)
    ctrl_target = np.where(counter, ctrl_actions == bank.action, bank.surviving[rows, ctrl_actions])
    candidate_target = np.where(
        counter, candidate_actions == bank.action, bank.surviving[rows, candidate_actions]
    )
    _atomic_npz(
        out,
        side=np.where(bank.counter, "counter", "decisive").astype("U9"),
        length=length_values.astype(np.int64, copy=False),
        fingerprint=observation_fingerprints(bank.local, bank.global_, bank.scalars),
        ctrl_q=ctrl_values,
        candidate_q=candidate_values,
        ctrl_bank_agree=ctrl_target.astype(bool),
        candidate_bank_agree=candidate_target.astype(bool),
    )
    receipt = {
        "schema": "t4a-posthoc-bank-capture/v1",
        "device": "cpu",
        "torch_threads": 1,
        "rows": int(bank.size),
        "out": str(out),
        "out_sha256": sha256(out),
        "inputs_sha256": inputs,
        "side_definition": "counter iff stored surviving_original contains stored B8 action",
        "bank_agree_definition": (
            "counter: raw argmax equals stored B8 action; decisive: raw argmax is in "
            "stored preferred set"
        ),
        "candidate_bank_agree": {
            "overall": float(candidate_target.mean()),
            "decisive": float(candidate_target[~counter].mean()),
            "counter": float(candidate_target[counter].mean()),
        },
    }
    _atomic_json(out.with_suffix(".receipt.json"), receipt)
    return receipt


def run(
    bank_rows: Path, trajectory_rows: Path, candidate_series: Path, ctrl_series: Path, out: Path
) -> dict[str, Any]:
    """Write a hash-bound, reproducible receipt for one candidate arm."""
    inputs = (Path(bank_rows), Path(trajectory_rows), Path(candidate_series), Path(ctrl_series))
    digests = {str(path): sha256(path) for path in inputs}
    with (
        np.load(bank_rows, allow_pickle=False) as bank_archive,
        np.load(trajectory_rows, allow_pickle=False) as trajectory_archive,
    ):
        summary = spillover_summary(
            {key: bank_archive[key].copy() for key in bank_archive.files},
            {key: trajectory_archive[key].copy() for key in trajectory_archive.files},
        )
    result = {
        "schema": SCHEMA,
        "authority": "declared post-hoc diagnostic only; not GO/KILL/EXTEND or gate evidence",
        "inputs_sha256": digests,
        "spillover": summary,
        "growth": growth_summary(candidate_series, ctrl_series),
    }
    if digests != {str(path): sha256(path) for path in inputs}:
        raise RuntimeError("an input artifact changed while post-hoc diagnostics were running")
    _atomic_json(out, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-rows", type=Path)
    parser.add_argument("--trajectory-rows", type=Path)
    parser.add_argument("--candidate-series", type=Path)
    parser.add_argument("--ctrl-series", type=Path)
    parser.add_argument("--capture-bank", type=Path)
    parser.add_argument("--ctrl-checkpoint", type=Path)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    capture_requested = any((args.capture_bank, args.ctrl_checkpoint, args.candidate_checkpoint))
    analysis_requested = any(
        (args.bank_rows, args.trajectory_rows, args.candidate_series, args.ctrl_series)
    )
    if capture_requested and not analysis_requested:
        if not all((args.capture_bank, args.ctrl_checkpoint, args.candidate_checkpoint)):
            parser.error(
                "bank capture requires --capture-bank, --ctrl-checkpoint and --candidate-checkpoint"
            )
        result = capture_bank_rows(
            args.capture_bank,
            args.ctrl_checkpoint,
            args.candidate_checkpoint,
            args.out,
            batch_size=args.batch_size,
        )
        print(json.dumps({"out": str(args.out), "schema": result["schema"]}, sort_keys=True))
        return 0
    if capture_requested or not all(
        (args.bank_rows, args.trajectory_rows, args.candidate_series, args.ctrl_series)
    ):
        parser.error("analysis requires all four --*-rows/--*-series arguments")
    result = run(
        args.bank_rows, args.trajectory_rows, args.candidate_series, args.ctrl_series, args.out
    )
    print(json.dumps({"out": str(args.out), "schema": result["schema"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
