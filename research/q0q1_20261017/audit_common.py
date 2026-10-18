"""Shared fail-closed primitives for the Q0 CPU audit jobs.

These helpers intentionally do no training.  Q0 may read immutable receipts and
may run the registered bounded development observation packets only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

T23_SEEDS = (7101, 7102, 7103, 7104, 7105)
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
HORIZON = 5000
MIX_COUNTS = {"scripted": 16, "frozen": 8, "mixed": 8}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic receipt after making its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(payload) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def iter_json(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield readable JSON receipts, never treating malformed data as evidence."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        value = load_json(path)
        if value is not None:
            yield path, value


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return None if not rows else sum(rows) / len(rows)


def stratum(length: int) -> str:
    """The preregistered coarse length cuts, plus an explicit middle band."""
    if length < 700:
        return "short_lt700"
    if length >= 800:
        return "long_ge800"
    return "middle_700_799"


def checkpoint_evidence(path: Path, seed: int, step: int | None = None) -> dict[str, Any]:
    """Bind bytes to the trainer's embedded identity, never to a filename."""
    import torch

    digest = sha256_file(path)
    result = {"path": str(path), "sha256": digest, "valid": False}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        transitions = meta.get("transitions")
        result.update(
            seed=meta.get("seed"), init_sha256=meta.get("init_sha256"), transitions=transitions
        )
        manifest = path.parent / "manifest.jsonl"
        matching = []
        if manifest.is_file():
            for line in manifest.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    entry.get("file") == path.name
                    and entry.get("sha256") == digest
                    and entry.get("transitions") == transitions
                ):
                    matching.append(entry)
        result["manifest_matches"] = len(matching)
        result["valid"] = (
            bool(matching)
            and meta.get("m3c", {}).get("namespace") == "t23-r0-screen/v1"
            and meta.get("seed") == seed
            and meta.get("init_sha256") == B8_SHA256
            and isinstance(transitions, int)
            and not isinstance(transitions, bool)
            and (step is None or step <= transitions < step + 4096)
        )
        if step == 8_000_000:
            pins = load_json(
                Path(__file__).resolve().parents[1]
                / "t23_r0_screen_20261017/results/t23_verdict.json"
            )
            expected = pins["frozen_checkpoint_sha256s"][f"{seed}:8000000"]
            result.update(expected_sha256=expected, matches_expected_sha256=digest == expected)
            result["valid"] = result["valid"] and digest == expected
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: invalid checkpoint metadata"
    return result


def checkpoint_inventory(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Discover every checkpoint and classify only by embedded decision counter."""
    result = {}
    for seed in T23_SEEDS:
        buckets = {"final": [], "2m": [], "4m": [], "invalid": []}
        base = root / f"seed-{seed}"
        for path in sorted(base.rglob("*")) if base.exists() else []:
            if not path.is_file() or path.suffix not in (".pt", ".pth"):
                continue
            evidence = checkpoint_evidence(path, seed)
            step = evidence.get("transitions", -1)
            label = next(
                (
                    label
                    for label, target in (
                        ("2m", 2_000_000),
                        ("4m", 4_000_000),
                        ("final", 8_000_000),
                    )
                    if isinstance(step, int) and target <= step < target + 4096
                ),
                "invalid",
            )
            if label == "final":
                evidence = checkpoint_evidence(path, seed, 8_000_000)
            buckets[label].append(evidence)
        result[str(seed)] = buckets
    return result
