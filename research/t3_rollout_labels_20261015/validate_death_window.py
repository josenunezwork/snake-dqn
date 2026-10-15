"""Validate death-window replay rows against the preserved unfiltered bank.

The death-window collector is intentionally supplied by the collection worker.  This
module only selects preserved, completed dying worlds and compares the decisive rows
returned by that collector with the legacy world shards.  It does not mutate the bank.

The worker API is ``collect_world_death_window(job, plan, out, run_identity, ...)``;
the optional trailing arguments are accepted by name so this validator remains a
read-only drop-in while the worker's receipt fields evolve.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from research.t3_rollout_labels_20261015.collect import _key

OBS_FIELDS = ("local", "global", "scalars", "mask")
COMPARE_FIELDS = OBS_FIELDS + ("surviving", "action", "decision", "frame")
DEFAULT_WORKER = "research.t3_rollout_labels_20261015.death_window:collect_world_death_window"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert worker receipt scalars to values accepted by ``json.dumps``."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if key not in {"rows", "row_data"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _legacy_candidates(bank: Path) -> list[dict[str, Any]]:
    """Return preserved dying worlds with at least one decisive legacy row."""
    allowlist = _json(bank / "legacy-resume.json").get("entries")
    if not isinstance(allowlist, dict):
        raise ValueError("legacy-resume.json entries must be an object")
    candidates: list[dict[str, Any]] = []
    for key in sorted(allowlist):
        receipt_path = bank / "worlds" / f"{key}.json"
        rows_path = bank / "worlds" / f"{key}.npz"
        if not receipt_path.exists() or not rows_path.exists():
            continue
        receipt = _json(receipt_path)
        if receipt.get("status") != "PASS" or receipt.get("death_decision") is None:
            continue
        with np.load(rows_path, allow_pickle=False) as data:
            if "decisive" not in data.files or "decision" not in data.files:
                continue
            decisive = np.asarray(data["decisive"], dtype=bool)
            count = int(decisive.sum())
        if count:
            candidates.append(
                {
                    "key": key,
                    "mix": str(receipt.get("mix")),
                    "arm": str(receipt.get("arm")),
                    "death_decision": int(receipt["death_decision"]),
                    "decisive_rows": count,
                    "receipt_sha256": _sha256(receipt_path),
                    "rows_sha256": _sha256(rows_path),
                }
            )
    if not candidates:
        raise RuntimeError("no preserved dying legacy world has decisive rows")
    return candidates


def select_keys(bank: Path, count: int = 5) -> list[dict[str, Any]]:
    """Select one mixed world when available, then high-yield scripted worlds.

    The selection is deterministic.  It favors rows that exercise the death-window
    path and includes all available mixes before using additional scripted worlds.
    """
    if count < 1:
        raise ValueError("count must be positive")
    candidates = _legacy_candidates(bank)
    selected: list[dict[str, Any]] = []
    for mix in ("mixed", "frozen", "scripted"):
        group = sorted(
            (item for item in candidates if item["mix"] == mix),
            key=lambda item: (-item["decisive_rows"], item["arm"], item["key"]),
        )
        if group and len(selected) < count:
            selected.append(group[0])
    remaining = sorted(
        (item for item in candidates if item["key"] not in {x["key"] for x in selected}),
        key=lambda item: (-item["decisive_rows"], item["mix"], item["arm"], item["key"]),
    )
    selected.extend(remaining[: max(0, count - len(selected))])
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} suitable dying worlds are available; need {count}"
        )
    return selected


def _load_rows(path: Path, *, decisive_only: bool) -> list[dict[str, np.ndarray]]:
    """Read row arrays without converting float observations through Python scalars."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"decision", *COMPARE_FIELDS}.difference(data.files)
        if missing:
            raise ValueError(f"{path.name} lacks row fields: {sorted(missing)}")
        selection = np.ones(len(data["decision"]), dtype=bool)
        if decisive_only:
            if "decisive" not in data.files:
                raise ValueError(f"{path.name} lacks decisive flags")
            selection = np.asarray(data["decisive"], dtype=bool)
        rows = []
        for index in np.flatnonzero(selection):
            rows.append(
                {field: np.array(data[field][index], copy=True) for field in COMPARE_FIELDS}
            )
        return rows


def _row_digest(row: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for field in COMPARE_FIELDS:
        value = np.asarray(row[field])
        digest.update(field.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _index_rows(
    rows: Iterable[dict[str, np.ndarray]], label: str
) -> dict[int, dict[str, np.ndarray]]:
    indexed: dict[int, dict[str, np.ndarray]] = {}
    for row in rows:
        decision = int(np.asarray(row["decision"]).item())
        if decision in indexed:
            raise ValueError(f"{label} has duplicate decisive decision {decision}")
        indexed[decision] = row
    return indexed


def compare_rows(
    old_rows: list[dict[str, np.ndarray]], new_rows: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    """Compare decisions, surviving sets, and observation bytes exactly."""
    old = _index_rows(old_rows, "old")
    new = _index_rows(new_rows, "death-window")
    old_decisions, new_decisions = set(old), set(new)
    missing = sorted(old_decisions - new_decisions)
    extra = sorted(new_decisions - old_decisions)
    mismatches: list[dict[str, Any]] = []
    for decision in sorted(old_decisions & new_decisions):
        left, right = old[decision], new[decision]
        differing = [
            field
            for field in COMPARE_FIELDS
            if np.asarray(left[field]).dtype != np.asarray(right[field]).dtype
            or np.asarray(left[field]).shape != np.asarray(right[field]).shape
            or np.asarray(left[field]).tobytes(order="C")
            != np.asarray(right[field]).tobytes(order="C")
        ]
        if differing:
            mismatches.append(
                {
                    "decision": decision,
                    "fields": differing,
                    "old_digest": _row_digest(left),
                    "new_digest": _row_digest(right),
                }
            )
    return {
        "old_decisive_rows": len(old_rows),
        "new_decisive_rows": len(new_rows),
        "old_decisions_sha256": hashlib.sha256(
            json.dumps(sorted(old_decisions), separators=(",", ":")).encode()
        ).hexdigest(),
        "new_decisions_sha256": hashlib.sha256(
            json.dumps(sorted(new_decisions), separators=(",", ":")).encode()
        ).hexdigest(),
        "missing_decisions": missing,
        "extra_decisions": extra,
        "row_mismatches": mismatches,
        "pass": not missing and not extra and not mismatches,
    }


def _find_rows(result: Any, out: Path, key: str) -> list[dict[str, np.ndarray]]:
    """Normalize a worker result or its emitted shard into decisive rows."""
    if isinstance(result, dict):
        for name in ("row_data", "rows_data", "decisive_rows", "rows"):
            value = result.get(name)
            if isinstance(value, list):
                return [
                    {field: np.asarray(row[field]) for field in COMPARE_FIELDS} for row in value
                ]
    for candidate in (out / "worlds" / f"{key}.npz", out / f"{key}.npz"):
        if candidate.exists():
            return _load_rows(candidate, decisive_only=True)
    matches = sorted(out.rglob(f"{key}.npz"))
    if len(matches) == 1:
        return _load_rows(matches[0], decisive_only=True)
    raise RuntimeError(f"death-window worker did not publish a row shard for {key}")


def _call_worker(
    worker: Callable[..., Any],
    job: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    threshold: float,
) -> Any:
    """Call the drop-in collector while binding only parameters it declares."""
    signature = inspect.signature(worker)
    names = signature.parameters
    kwargs: dict[str, Any] = {}
    if "out" in names:
        kwargs["out"] = out
    if "run_identity" in names:
        kwargs["run_identity"] = {"validation": "t3-death-window-v1"}
    if "threshold" in names:
        kwargs["threshold"] = threshold
    if "prefilter_threshold" in names:
        kwargs["prefilter_threshold"] = threshold
    if "validate_controls" in names:
        kwargs["validate_controls"] = True
    try:
        return worker(job, plan, **kwargs)
    except TypeError as error:
        raise TypeError(
            f"death-window worker signature {signature} did not accept validator call: {error}"
        ) from error


def _worker_from_spec(spec: str) -> Callable[..., Any]:
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--worker must be MODULE:FUNCTION")
    worker = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(worker):
        raise AttributeError(f"worker {spec} is not callable")
    return worker


def _source_hashes(repo: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash every execution dependency and reject missing paths."""
    result = {}
    for relative in sorted(set(relative_paths)):
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"required death-window source is missing: {relative}")
        result[relative] = _sha256(path)
    return result


def validate(
    bank: Path,
    namespace: Path,
    out: Path,
    keys: list[str] | None = None,
    worker_spec: str = DEFAULT_WORKER,
) -> dict[str, Any]:
    """Run exact validation for five preserved dying worlds."""
    namespace_value = _json(namespace)
    jobs = namespace_value.get("jobs")
    plans = namespace_value.get("plans")
    if not isinstance(jobs, list) or not isinstance(plans, list) or len(jobs) != len(plans):
        raise ValueError("namespace receipt has no aligned jobs/plans")
    by_key = {_key(job, plan): (job, plan) for job, plan in zip(jobs, plans)}
    selected = (
        select_keys(bank, 5)
        if keys is None
        else [item for item in _legacy_candidates(bank) if item["key"] in set(keys)]
    )
    if len(selected) != 5 or len({item["key"] for item in selected}) != 5:
        raise ValueError("validation requires exactly five unique suitable dying keys")
    worker = _worker_from_spec(worker_spec)
    repo = Path(__file__).resolve().parents[2]
    prefilter_path = bank / "prefilter.json"
    prefilter = _json(prefilter_path)
    threshold = float(prefilter.get("threshold"))
    if prefilter.get("schema") != "t3-prefilter/v1" or not np.isfinite(threshold):
        raise ValueError("validation requires a finite PASS prefilter threshold")
    source_relatives = {
        "research/t3_rollout_labels_20261015/death_window.py",
        "research/t3_rollout_labels_20261015/bank_engine.py",
        "research/t3_rollout_labels_20261015/collect.py",
        "research/t3_rollout_labels_20261015/engine.py",
        "research/t3_rollout_labels_20261015/trigger.py",
        "research/t3_rollout_labels_20261015/validate_death_window.py",
        "research/t1_commitment_20261015/replay.py",
        "research/b8_death_census_20261014/census.py",
    }
    for item in selected:
        receipt = _json(bank / "worlds" / f"{item['key']}.json")
        source_relatives.update(
            str(relative) for relative in receipt.get("run_identity", {}).get("source_sha256", {})
        )
    source_before = _source_hashes(repo, source_relatives)
    namespace_sha256 = _sha256(namespace)
    prefilter_sha256 = _sha256(prefilter_path)
    from research.b8_death_census_20261014 import census

    checkpoint_sha256 = {arm: _sha256(census.b8_path(arm)) for arm in census.B8_ARMS}
    for item in selected:
        identity = _json(bank / "worlds" / f"{item['key']}.json").get("run_identity", {})
        roster = identity.get("roster_checkpoint_sha256", {})
        if not isinstance(roster, dict):
            raise ValueError(f"legacy receipt has no roster checkpoint map: {item['key']}")
        for path_text, expected in roster.items():
            path = Path(path_text)
            actual = _sha256(path)
            if actual != str(expected):
                raise RuntimeError(f"legacy roster checkpoint changed: {path_text}")
            checkpoint_sha256[str(path)] = actual
    if out.suffix.lower() == ".json":
        receipt_path = out
        work_dir = out.with_suffix(".work")
    else:
        work_dir = out
        receipt_path = out / "validation.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    if any(work_dir.iterdir()):
        raise RuntimeError(f"validation output must be empty: {work_dir}")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for selection in selected:
        key = selection["key"]
        if key not in by_key:
            raise KeyError(f"legacy key is absent from namespace receipt: {key}")
        old_path = bank / "worlds" / f"{key}.npz"
        old_rows = _load_rows(old_path, decisive_only=True)
        world_out = work_dir / key
        world_out.mkdir(parents=True)
        world_started = time.perf_counter()
        result = _call_worker(worker, *by_key[key], world_out, threshold)
        new_rows = _find_rows(result, world_out, key)
        comparison = compare_rows(old_rows, new_rows)
        results.append(
            {
                **selection,
                "elapsed_seconds": time.perf_counter() - world_started,
                "comparison": comparison,
                "worker_result": (
                    _json_safe(result)
                    if isinstance(result, dict)
                    else {"type": type(result).__name__}
                ),
            }
        )
    source_after = _source_hashes(repo, source_relatives)
    if source_before != source_after:
        raise RuntimeError("death-window source drifted during validation")
    checkpoint_after = {
        name: (_sha256(census.b8_path(name)) if name in census.B8_ARMS else _sha256(Path(name)))
        for name in checkpoint_sha256
    }
    if checkpoint_sha256 != checkpoint_after:
        raise RuntimeError("death-window checkpoint drifted during validation")
    receipt = {
        "schema": "t3-death-window-validation/v1",
        "status": "PASS" if all(item["comparison"]["pass"] for item in results) else "FAIL",
        "worker": worker_spec,
        "bank": str(bank),
        "namespace": str(namespace),
        "namespace_sha256": namespace_sha256,
        "prefilter": str(prefilter_path),
        "prefilter_sha256": prefilter_sha256,
        "threshold": threshold,
        "keys": [item["key"] for item in results],
        "source_sha256": source_after,
        "checkpoint_sha256": checkpoint_sha256,
        "legacy_bindings": {
            item["key"]: {
                "receipt_sha256": item["receipt_sha256"],
                "rows_sha256": item["rows_sha256"],
            }
            for item in results
        },
        "work_dir": str(work_dir),
        "selected": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temp = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temp, receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", nargs=5)
    parser.add_argument(
        "--worker",
        default=DEFAULT_WORKER,
    )
    args = parser.parse_args()
    receipt = validate(args.bank, args.namespace, args.out, args.keys, args.worker)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
