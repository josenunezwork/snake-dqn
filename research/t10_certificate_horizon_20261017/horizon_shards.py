"""Launch and assemble non-overlapping T10 horizon diagnostic shards.

This wrapper never changes the T10 fit implementation.  It limits one invocation
of the imported fit to a declared subset of the canonical horizons, then makes a
fresh combined receipt from a deliberately interrupted prefix and a completed
late shard.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from research.t5b_risk_and_fuse_20261016.fuse import sha256
from research.t10_certificate_horizon_20261017 import horizon

CANONICAL_HORIZONS = (30, 60, 100, 200, 300)
PREFIX_HORIZONS = (30, 60, 100)
LATE_HORIZONS = (200, 300)


def _hashes_match_receipt(report: dict[str, Any]) -> bool:
    """Check every source named by a completed receipt against its recorded hash."""
    recorded = report.get("source_sha256_after")
    if not isinstance(recorded, dict):
        return False
    return all(
        Path(path).is_file() and sha256(Path(path)) == digest for path, digest in recorded.items()
    )


def _artifact_hashes_match_receipt(directory: Path, report: dict[str, Any]) -> bool:
    """Check each saved output named by a completed fit receipt."""
    recorded = report.get("artifact_sha256")
    if not isinstance(recorded, dict):
        return False
    return all(
        (directory / name).is_file() and sha256(directory / name) == digest
        for name, digest in recorded.items()
    )


def _prediction_keys(path: Path) -> tuple[str, ...]:
    """Return ordered OOF score names while rejecting an incomplete receipt."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"y", "state", "action", "world", "fold"}
        if not required.issubset(archive.files):
            raise ValueError(f"incomplete prediction receipt: {path}")
        keys = tuple(sorted(key for key in archive.files if key.startswith("oof_")))
    if not keys:
        raise ValueError(f"prediction receipt has no OOF scores: {path}")
    return keys


def _validate_branch_identity(reference: Path, candidate: Path) -> tuple[str, ...]:
    """Require exact legal-branch identity and feature-score columns across shards."""
    keys = _prediction_keys(reference)
    if _prediction_keys(candidate) != keys:
        raise ValueError("prediction feature-set keys differ across horizon receipts")
    with (
        np.load(reference, allow_pickle=False) as left,
        np.load(candidate, allow_pickle=False) as right,
    ):
        for key in ("state", "action", "world", "fold"):
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f"branch identity differs across horizon receipts: {key}")
    return keys


def _load_report(directory: Path) -> dict[str, Any]:
    """Load a horizon fit report from a completed or deliberately stopped output."""
    path = directory / "horizon_fits.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _settings(report: dict[str, Any]) -> dict[str, Any]:
    """Select the immutable fit settings which must agree between shards."""
    return {
        key: report[key]
        for key in (
            "schema",
            "authority",
            "data_sha256",
            "fold_receipt_sha256",
            "bootstrap_world_draws",
            "bootstrap",
            "model",
            "feature_sets",
        )
    }


def run_late(output: Path, n_boot: int = 2000) -> dict[str, Any]:
    """Run only D200/D300 with unchanged fit code and durable launch provenance."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite late-shard output: {output}")
    original = horizon.HORIZONS
    if tuple(original) != CANONICAL_HORIZONS:
        raise ValueError("imported horizon module does not expose the canonical full horizon order")
    launch = {
        "schema": "t10-horizon-late-launch-v1",
        "canonical_horizons": list(CANONICAL_HORIZONS),
        "selected_horizons": list(LATE_HORIZONS),
        "implementation": str(Path(horizon.__file__).resolve()),
        "implementation_sha256": sha256(Path(horizon.__file__)),
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": sha256(Path(__file__)),
        "n_boot": n_boot,
    }
    # ``run`` owns output creation; stage provenance beside it without changing the fit code.
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.late-launch.json"
    stage.write_text(json.dumps(launch, indent=2) + "\n")
    horizon.HORIZONS = LATE_HORIZONS
    try:
        report = horizon.run(output, n_boot=n_boot)
    finally:
        horizon.HORIZONS = original
    if report.get("status") != "PASS" or tuple(report.get("horizons", ())) != LATE_HORIZONS:
        raise RuntimeError("late shard did not complete exactly D200/D300")
    if report.get("source_sha256_before") != report.get("source_sha256_after"):
        raise RuntimeError("late shard source hashes changed during the fit")
    if sha256(Path(horizon.__file__)) != launch["implementation_sha256"]:
        raise RuntimeError("horizon implementation changed during late execution")
    if sha256(Path(__file__)) != launch["driver_sha256"]:
        raise RuntimeError("late driver changed during execution")
    if not _hashes_match_receipt(report) or not _artifact_hashes_match_receipt(output, report):
        raise RuntimeError("late shard source hashes differ from files at completion")
    shutil.move(stage, output / "late_launch.json")
    return report


def mark_prefix_boundary(prefix: Path, observed_elapsed_s: float | None = None) -> dict[str, Any]:
    """Record the task-owned intentional stop after a durable D100 receipt."""
    report = _load_report(prefix)
    if tuple(report.get("results", ())) != tuple(f"D{x}" for x in PREFIX_HORIZONS):
        raise ValueError("cannot mark boundary before exactly D30/D60/D100 are saved")
    prediction = prefix / "D100.predictions.npz"
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    stat = prefix.stat()
    receipt = {
        "schema": "t10-horizon-prefix-boundary-v1",
        "status": "INTENTIONALLY_INTERRUPTED_AFTER_D100",
        "prefix_horizons": list(PREFIX_HORIZONS),
        "d100_prediction_sha256": sha256(prediction),
        "report_sha256": sha256(prefix / "horizon_fits.json"),
        "recorded_utc": datetime.now(UTC).isoformat(),
        "directory_birthtime_utc_approx": datetime.fromtimestamp(
            stat.st_birthtime, UTC
        ).isoformat(),
        "observed_elapsed_s": observed_elapsed_s,
        "elapsed_note": "observed process runtime if supplied; otherwise null, not inferred",
    }
    (prefix / "prefix_boundary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def combine(prefix: Path, late: Path, output: Path) -> dict[str, Any]:
    """Validate a D30--D100 prefix plus D200--D300 PASS shard and copy receipts."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite combined output: {output}")
    prefix_report, late_report = _load_report(prefix), _load_report(late)
    if tuple(prefix_report.get("results", ())) != tuple(f"D{x}" for x in PREFIX_HORIZONS):
        raise ValueError("prefix receipt is not exactly D30/D60/D100")
    boundary_path = prefix / "prefix_boundary.json"
    if not boundary_path.is_file():
        raise FileNotFoundError("prefix boundary receipt is required")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("status") != "INTENTIONALLY_INTERRUPTED_AFTER_D100":
        raise ValueError("prefix boundary does not record an intentional D100 interruption")
    if boundary.get("d100_prediction_sha256") != sha256(prefix / "D100.predictions.npz"):
        raise ValueError("prefix D100 prediction changed after its interruption receipt")
    if (
        tuple(late_report.get("horizons", ())) != LATE_HORIZONS
        or late_report.get("status") != "PASS"
    ):
        raise ValueError("late receipt is not a completed D200/D300 PASS shard")
    if tuple(late_report.get("results", ())) != tuple(f"D{x}" for x in LATE_HORIZONS):
        raise ValueError("late receipt does not contain exactly D200/D300")
    if _settings(prefix_report) != _settings(late_report):
        raise ValueError(
            "immutable model, fold, bootstrap, or feature settings differ across shards"
        )
    if prefix_report.get("source_sha256_before") != late_report.get("source_sha256_before"):
        raise ValueError("source hashes differ at shard launch")
    if late_report.get("source_sha256_before") != late_report.get("source_sha256_after"):
        raise ValueError("late source hashes changed during fit")
    if not _hashes_match_receipt(late_report) or not _artifact_hashes_match_receipt(
        late, late_report
    ):
        raise ValueError("late source hashes do not match after completion")
    launch_path = late / "late_launch.json"
    if not launch_path.is_file():
        raise FileNotFoundError("late launch receipt is required")
    launch = json.loads(launch_path.read_text())
    for key in ("driver", "driver_sha256", "implementation", "implementation_sha256"):
        if key not in launch:
            raise ValueError(f"late launch receipt lacks {key}")
    if sha256(Path(launch["driver"])) != launch["driver_sha256"]:
        raise ValueError("late driver changed after execution")
    if sha256(Path(launch["implementation"])) != launch["implementation_sha256"]:
        raise ValueError("late implementation changed after execution")
    prediction_paths = {h: prefix / f"D{h}.predictions.npz" for h in PREFIX_HORIZONS}
    prediction_paths.update({h: late / f"D{h}.predictions.npz" for h in LATE_HORIZONS})
    reference = prediction_paths[30]
    for path in prediction_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_branch_identity(reference, path)
    output.mkdir(parents=True)
    copied = {}
    for value, source in prediction_paths.items():
        destination = output / source.name
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"copied prediction hash mismatch: D{value}")
        copied[destination.name] = sha256(destination)
    source_after = {path: sha256(Path(path)) for path in prefix_report["source_sha256_before"]}
    if source_after != prefix_report["source_sha256_before"]:
        raise ValueError("a prefix-bound source changed before combination")
    report = {
        "schema": "t10-horizon-combined-v1",
        "status": "PASS",
        "authority": prefix_report["authority"],
        "horizons": list(CANONICAL_HORIZONS),
        "data_sha256": prefix_report["data_sha256"],
        "fold_receipt_sha256": prefix_report["fold_receipt_sha256"],
        "bootstrap_world_draws": prefix_report["bootstrap_world_draws"],
        "bootstrap": prefix_report["bootstrap"],
        "model": prefix_report["model"],
        "feature_sets": prefix_report["feature_sets"],
        "source_sha256_before": prefix_report["source_sha256_before"],
        "source_sha256_after": source_after,
        "settings": _settings(prefix_report),
        "results": {**prefix_report["results"], **late_report["results"]},
        "provenance": {
            "prefix": str(prefix.resolve()),
            "prefix_boundary": boundary,
            "prefix_report_sha256": sha256(prefix / "horizon_fits.json"),
            "late": str(late.resolve()),
            "late_status": "PASS",
            "late_report_sha256": sha256(late / "horizon_fits.json"),
            "source_sha256_before": prefix_report["source_sha256_before"],
            "late_source_sha256_after": late_report["source_sha256_after"],
            "late_launch": launch,
        },
        "prediction_sha256": copied,
    }
    report["artifact_sha256"] = copied
    report["elapsed_s"] = late_report.get("elapsed_s")
    (output / "horizon_fits.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run an explicit late shard or safely combine previously produced receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    late = command.add_parser("late")
    late.add_argument("--out", type=Path, required=True)
    late.add_argument("--boot", type=int, default=2000)
    merge = command.add_parser("combine")
    merge.add_argument("--prefix", type=Path, required=True)
    merge.add_argument("--late", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    boundary = command.add_parser("mark-prefix-boundary")
    boundary.add_argument("--prefix", type=Path, required=True)
    boundary.add_argument("--elapsed-s", type=float, default=None)
    args = parser.parse_args()
    if args.command == "late":
        result = run_late(args.out, args.boot)
    elif args.command == "combine":
        result = combine(args.prefix, args.late, args.out)
    else:
        result = mark_prefix_boundary(args.prefix, args.elapsed_s)
    print(
        json.dumps({"status": result.get("status"), "horizons": result.get("horizons")}, indent=2)
    )


if __name__ == "__main__":
    main()
