#!/usr/bin/env python3
"""Independent, hash-bound audit for the T4a Part 1 boost captures.

This module does not replay the environment or load a checkpoint.  It audits the
captured NPZ rows against their source probe rows and the already-written T3
series.  In particular, a fatal pre-action row is expected, so the row count is
``sum(alive_frames) + sum(deaths)`` rather than just ``sum(alive_frames)``.

The audit is intentionally tolerant of an in-progress capture pass: missing
captures are reported in the JSON result and the CLI exits with status 2.  A
complete, hash-valid pass exits 0; an invalid existing artifact exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ["SNAKE_DQN_DEVICE"] = "cpu"
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

SEEDS = tuple(range(10, 15))
ARMS = ("CTRL", "LABELS", "LABELS_KEEPV8")
LABEL_ARMS = ("LABELS", "LABELS_KEEPV8")
BUCKETS = ("lt400", "400_699", "700_999", "ge1000")
ARRAY_FIELDS = (
    "env",
    "frame",
    "length",
    "raw_action",
    "resolved_action",
    "advisory_action",
)
ACTION_FIELDS = ("raw_action", "resolved_action", "advisory_action")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_row(probes: Path, name: str) -> Path:
    return probes / "rows" / f"{name}__r62__scripted.json"


def _t3_series(t3: Path, name: str) -> Path:
    return t3 / "series" / f"{name}.npz"


def expected_captures() -> tuple[tuple[str, str], ...]:
    """Return the registered 25 capture pairs in stable order."""
    result = []
    for seed in SEEDS:
        result.append((f"CTRL_s{seed}", f"CTRL_s{seed}"))
        for arm in LABEL_ARMS:
            result.append((f"{arm}_s{seed}", f"{arm}_s{seed}"))
            result.append((f"{arm}_s{seed}", f"CTRL_s{seed}"))
    return tuple(result)


def _capture_path(out: Path, student: str, trajectory: str) -> Path:
    return out / "boost" / f"{student}__on__{trajectory}.npz"


def _receipt_path(out: Path, student: str, trajectory: str) -> Path:
    return out / "receipts" / f"boost-{student}__on__{trajectory}.json"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_counts(row: Mapping[str, Any]) -> tuple[int, int]:
    alive = row.get("alive_frames")
    deaths = row.get("deaths")
    _require(isinstance(alive, dict) and isinstance(deaths, dict), "probe lacks alive/death maps")
    _require(set(alive) == set(deaths), "probe alive/death world keys differ")
    alive_total = sum(int(value) for value in alive.values())
    death_total = sum(int(value) for value in deaths.values())
    _require(all(int(value) >= 0 for value in alive.values()), "negative alive-frame count")
    _require(all(int(value) in (0, 1) for value in deaths.values()), "death values are not binary")
    return alive_total, death_total


def _bucket_codes(length: np.ndarray) -> np.ndarray:
    values = np.asarray(length, dtype=np.int64)
    code = np.full(len(values), 3, dtype=np.int8)
    code[values < 400] = 0
    code[(values >= 400) & (values < 700)] = 1
    code[(values >= 700) & (values < 1000)] = 2
    return code


def bucket_counts(length: np.ndarray) -> dict[str, int]:
    """Count all rows in the registered four pre-action length buckets."""
    codes = _bucket_codes(np.asarray(length))
    return {bucket: int((codes == index).sum()) for index, bucket in enumerate(BUCKETS)}


def action_counts(actions: np.ndarray) -> dict[str, Any]:
    """Return exact six-action and boost-mode counts for one action column."""
    values = np.asarray(actions)
    _require(values.ndim == 1, "action column must be one-dimensional")
    _require(np.all(np.isfinite(values)), "action column contains non-finite values")
    _require(np.all((values >= 0) & (values <= 5)), "action outside six-action range")
    counts = np.bincount(values.astype(np.int64), minlength=6)
    total = int(len(values))
    boost = int(counts[3:].sum())
    return {
        "decisions": total,
        "counts": [int(value) for value in counts],
        "boost_argmax": boost,
        "boost_rate": None if not total else boost / total,
    }


def _validate_receipt(
    out: Path,
    student: str,
    trajectory: str,
    probes: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    receipt_path = _receipt_path(out, student, trajectory)
    array_path = _capture_path(out, student, trajectory)
    _require(receipt_path.is_file(), f"missing receipt: {receipt_path}")
    _require(array_path.is_file(), f"missing capture: {array_path}")
    receipt = json.loads(receipt_path.read_text())
    _require(receipt.get("schema") == "t4a-posthoc-boost/v1", f"bad receipt schema: {receipt_path}")
    _require(receipt.get("student") == student, f"receipt student mismatch: {receipt_path}")
    _require(
        receipt.get("trajectory") == trajectory, f"receipt trajectory mismatch: {receipt_path}"
    )
    _require(
        Path(receipt.get("array", "")).resolve() == array_path.resolve(),
        "receipt array path mismatch",
    )
    _require(
        receipt.get("selection") == "round62 scripted H5000 rowwise advisory",
        "selection contract mismatch",
    )
    _require(
        receipt.get("telemetry_greedy_mask") == "resolved with normal-only fallback",
        "telemetry mask contract mismatch",
    )
    _require(
        receipt.get("array_sha256") == sha256(array_path), f"capture hash mismatch: {array_path}"
    )
    checkpoint = Path(receipt.get("checkpoint", ""))
    _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    _require(
        receipt.get("checkpoint_sha256") == sha256(checkpoint),
        f"checkpoint hash mismatch: {checkpoint}",
    )

    inputs = receipt.get("inputs_sha256")
    _require(isinstance(inputs, dict) and inputs, f"missing input bindings: {receipt_path}")
    for source, digest in inputs.items():
        path = Path(source)
        _require(path.is_file(), f"bound input missing: {path}")
        _require(sha256(path) == digest, f"bound input hash mismatch: {path}")

    trajectory_row = _source_row(probes, trajectory)
    student_row = _source_row(probes, student)
    _require(
        Path(receipt.get("trajectory_source_row", "")).resolve() == trajectory_row.resolve(),
        "trajectory row path mismatch",
    )
    _require(
        Path(receipt.get("student_source_row", "")).resolve() == student_row.resolve(),
        "student row path mismatch",
    )
    _require(
        receipt.get("trajectory_source_row_sha256") == sha256(trajectory_row),
        "trajectory row hash mismatch",
    )
    _require(
        receipt.get("student_source_row_sha256") == sha256(student_row), "student row hash mismatch"
    )
    row = json.loads(trajectory_row.read_text())
    alive_total, death_total = _source_counts(row)
    arrays = _read_npz(array_path)
    _require(set(arrays) == set(ARRAY_FIELDS), f"unexpected capture fields: {array_path}")
    lengths = {len(np.asarray(arrays[field])) for field in ARRAY_FIELDS}
    _require(len(lengths) == 1, f"capture columns have different lengths: {array_path}")
    decisions = lengths.pop()
    _require(
        decisions == alive_total + death_total, f"count mismatch for {student} on {trajectory}"
    )
    _require(
        int(receipt.get("decisions", -1)) == decisions,
        f"receipt decision count mismatch: {array_path}",
    )
    for field in ("env", "frame", "length"):
        _require(np.all(np.isfinite(arrays[field])), f"non-finite {field} values: {array_path}")
    _require(np.all(arrays["env"] >= 0), f"negative env index: {array_path}")
    _require(
        np.all(arrays["frame"] >= 1),
        f"pre-action frame must use one-based sim.frame values: {array_path}",
    )
    _require(np.all(arrays["length"] >= 0), f"negative pre-action length: {array_path}")
    bucket_codes = _bucket_codes(arrays["length"])
    by_bucket = {
        bucket: {
            field.removesuffix("_action"): action_counts(arrays[field][bucket_codes == index])
            for field in ACTION_FIELDS
        }
        for index, bucket in enumerate(BUCKETS)
    }
    details = {
        "student": student,
        "trajectory": trajectory,
        "decisions": decisions,
        "source_alive_frames": alive_total,
        "source_deaths": death_total,
        "fatal_predecision_row_included": death_total,
        "bucket_counts": bucket_counts(arrays["length"]),
        "bucket_partition": {
            "counts_sum": int(sum(bucket_counts(arrays["length"]).values())),
            "decisions": decisions,
            "complete": int(sum(bucket_counts(arrays["length"]).values())) == decisions,
        },
        "actions_by_bucket": by_bucket,
        "actions": {
            field.removesuffix("_action"): action_counts(arrays[field]) for field in ACTION_FIELDS
        },
        "receipt": str(receipt_path),
        "array": str(array_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
    }
    return receipt, arrays, details


def _load_t3_series(t3: Path, name: str) -> dict[str, np.ndarray]:
    path = _t3_series(t3, name)
    _require(path.is_file(), f"missing T3 series: {path}")
    receipt_path = t3 / "receipts" / f"{name}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        _require(receipt.get("series_sha256") == sha256(path), f"T3 series hash mismatch: {path}")
        _require(
            Path(receipt.get("series", "")).resolve() == path.resolve(),
            f"T3 series path mismatch: {path}",
        )
    return _read_npz(path)


def _key_index(env: np.ndarray, frame: np.ndarray) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for index, key in enumerate(zip(np.asarray(env).tolist(), np.asarray(frame).tolist())):
        pair = (int(key[0]), int(key[1]))
        _require(pair not in result, f"duplicate env/frame row: {pair}")
        result[pair] = index
    return result


def compare_ctrl_actions(
    arrays: Mapping[str, np.ndarray], t3_series: Mapping[str, np.ndarray], arm: str
) -> dict[str, Any]:
    """Compare a T4a advisory capture with its exact T3 CTRL decision rows."""
    required = {
        "decision_arm",
        "decision_env",
        "decision_frame",
        "decision_length",
        "decision_arm_action",
    }
    if not required <= t3_series.keys():
        return {"status": "unavailable", "reason": "T3 series has no decision arrays"}
    arm_id = 0 if arm == "LABELS" else 1
    selected = np.asarray(t3_series["decision_arm"]) == arm_id
    prior_env = np.asarray(t3_series["decision_env"])[selected]
    prior_frame = np.asarray(t3_series["decision_frame"])[selected]
    current_index = _key_index(arrays["env"], arrays["frame"])
    prior_index = _key_index(prior_env, prior_frame)
    current_keys, prior_keys = set(current_index), set(prior_index)
    _require(current_keys == prior_keys, f"T3/T4 env/frame support differs for {arm}")
    action_mismatches = 0
    length_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    for key in sorted(current_keys):
        current = current_index[key]
        prior = np.flatnonzero(selected)[prior_index[key]]
        if int(arrays["advisory_action"][current]) != int(t3_series["decision_arm_action"][prior]):
            action_mismatches += 1
            first_mismatch = first_mismatch or {
                "env": key[0],
                "frame": key[1],
                "t4_advisory_action": int(arrays["advisory_action"][current]),
                "t3_arm_action": int(t3_series["decision_arm_action"][prior]),
            }
        if int(arrays["length"][current]) != int(t3_series["decision_length"][prior]):
            length_mismatches += 1
            first_mismatch = first_mismatch or {
                "env": key[0],
                "frame": key[1],
                "t4_length": int(arrays["length"][current]),
                "t3_length": int(t3_series["decision_length"][prior]),
            }
    return {
        "status": "pass" if not action_mismatches and not length_mismatches else "fail",
        "rows": len(current_keys),
        "action_mismatches": action_mismatches,
        "length_mismatches": length_mismatches,
        "first_mismatch": first_mismatch,
    }


def compare_own_prelength(
    arrays: Mapping[str, np.ndarray], t3_series: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Compare one-based pre-action lengths with zero-based post-transition series."""
    required = {"seeds", "length", "alive"}
    if not required <= t3_series.keys():
        return {"status": "unavailable", "reason": "T3 series lacks seeds/length/alive"}
    lengths = np.asarray(t3_series["length"])
    alive = np.asarray(t3_series["alive"], dtype=bool)
    seeds = np.asarray(t3_series["seeds"])
    _require(lengths.shape == alive.shape and lengths.ndim == 2, "invalid T3 series dimensions")
    env = np.asarray(arrays["env"], dtype=np.int64)
    frame = np.asarray(arrays["frame"], dtype=np.int64)
    _require(np.all(env < len(seeds)), "capture env exceeds T3 series worlds")
    shifted = frame > 1
    checks = int(shifted.sum())
    length_mismatches = int(
        np.sum(arrays["length"][shifted] != lengths[env[shifted], frame[shifted] - 2])
    )
    prior_was_alive = alive[env[shifted], frame[shifted] - 2]
    terminal_shift_mismatches = int(np.sum(~prior_was_alive))
    initial_rows = int((frame == 1).sum())
    invalid_initial_frames = int(np.sum(frame < 1))
    frame_support_mismatches = 0
    for world in range(len(seeds)):
        prior_dead = np.flatnonzero(~alive[world])
        expected_rows = int(prior_dead[0] + 1) if len(prior_dead) else lengths.shape[1]
        observed = np.sort(frame[env == world])
        expected = np.arange(1, expected_rows + 1, dtype=np.int64)
        if not np.array_equal(observed, expected):
            frame_support_mismatches += 1
    return {
        "status": (
            "pass"
            if not length_mismatches
            and not terminal_shift_mismatches
            and not frame_support_mismatches
            and not invalid_initial_frames
            else "fail"
        ),
        "shifted_rows_checked": checks,
        "initial_prelength_rows_unchecked": initial_rows,
        "length_mismatches": length_mismatches,
        "terminal_shift_mismatches": terminal_shift_mismatches,
        "frame_support_mismatches": frame_support_mismatches,
        "invalid_initial_frames": invalid_initial_frames,
        "initial_mass_comparison": "unavailable: prior series starts post-transition",
        "terminal_contract": (
            "one-based fatal pre-action row aligned to prior zero-based post-transition frame - 2"
        ),
        "ambiguous_comparisons": [
            "frame 1 initial pre-action length has no prior post-transition series value"
        ],
    }


def audit(
    out: Path,
    probes: Path = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t3-probes-20261015"
    ),
    t3: Path = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-diagnostics-20261016"),
) -> dict[str, Any]:
    """Audit all available Part 1 captures and return a machine-readable report."""
    out, probes, t3 = Path(out), Path(probes), Path(t3)
    result: dict[str, Any] = {
        "schema": "t4a-posthoc-part1-audit/v1",
        "authority": "independent diagnostic audit, not gate evidence",
        "cpu_only": True,
        "required_captures": len(expected_captures()),
        "captures": {},
        "missing": [],
        "errors": [],
        "ctrl_action_comparisons": {},
        "own_prelength_comparisons": {},
    }
    loaded: dict[tuple[str, str], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for student, trajectory in expected_captures():
        key = f"{student}__on__{trajectory}"
        try:
            _, arrays, details = _validate_receipt(out, student, trajectory, probes)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            message = str(exc)
            if "missing capture" in message or "missing receipt" in message:
                result["missing"].append(key)
            else:
                result["errors"].append({"capture": key, "error": message})
            continue
        loaded[(student, trajectory)] = (details, arrays)
        result["captures"][key] = details

    for arm in LABEL_ARMS:
        for seed in SEEDS:
            student, trajectory = f"{arm}_s{seed}", f"CTRL_s{seed}"
            pair = loaded.get((student, trajectory))
            if pair is None:
                continue
            try:
                series = _load_t3_series(t3, trajectory)
                result["ctrl_action_comparisons"][f"{student}__on__{trajectory}"] = (
                    compare_ctrl_actions(pair[1], series, arm)
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                result["errors"].append(
                    {"comparison": f"{student}__on__{trajectory}", "error": str(exc)}
                )

    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            pair = loaded.get((name, name))
            if pair is None:
                continue
            try:
                result["own_prelength_comparisons"][name] = compare_own_prelength(
                    pair[1], _load_t3_series(t3, name)
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                result["errors"].append({"comparison": name, "error": str(exc)})

    comparisons = list(result["ctrl_action_comparisons"].values()) + list(
        result["own_prelength_comparisons"].values()
    )
    invalid_comparisons = [row for row in comparisons if row.get("status") == "fail"]
    result["status"] = (
        "invalid"
        if result["errors"] or invalid_comparisons
        else (
            "complete"
            if not result["missing"] and len(result["captures"]) == len(expected_captures())
            else "incomplete"
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--probes",
        type=Path,
        default=Path(
            "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t3-probes-20261015"
        ),
    )
    parser.add_argument(
        "--t3",
        type=Path,
        default=Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-diagnostics-20261016"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = audit(args.out, args.probes, args.t3)
    destination = args.json or args.out / "part1-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "json": str(destination)}, sort_keys=True))
    return {"complete": 0, "incomplete": 2, "invalid": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
