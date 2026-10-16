#!/usr/bin/env python3
"""Fail-closed 0.5M/1M RETURN_C early-warning checkpoint evaluator.

The evaluator deliberately consumes only immutable held-out banks and a completed
checkpoint.  It has no training side effects.  A caller must supply the initial
checkpoint as the Q-space baseline, the matching held-out outside bank, and a
telemetry row containing the registered training contrast-absolute-error values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SCHEMA = "t4f-return-early-warning/v1"
LOOKS = frozenset((500_000, 1_000_000))


@dataclass(frozen=True)
class EarlyWarningConfig:
    """Inputs for one pre-registered RETURN_C early-warning look."""

    checkpoint: Path
    initial_checkpoint: Path
    holdout_bank: Path
    outside_bank: Path
    telemetry_jsonl: Path
    transitions: int


@dataclass(frozen=True)
class OutsideObservations:
    """Immutable held-out outside states, intentionally without branch targets."""

    local: np.ndarray
    global_: np.ndarray
    scalars: np.ndarray
    b8_action: np.ndarray
    legal_mask: np.ndarray


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one result atomically so a launch lane never reads a partial verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(raw, path)
    except BaseException:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass
        raise


def _state_groups(bank) -> list[np.ndarray]:
    """Group a bank's complete action branches by their immutable source state."""
    groups: dict[tuple[bytes, bytes], list[int]] = {}
    for index, (packed, scalars) in enumerate(zip(bank.state_packed, bank.state_scalars)):
        groups.setdefault((packed.tobytes(), scalars.tobytes()), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _q_values(net, bank, indices: np.ndarray, torch) -> np.ndarray:
    """Return CPU float64 Q values for immutable bank source states."""
    with torch.no_grad():
        q = net(
            torch.from_numpy(np.ascontiguousarray(bank.local[indices])),
            torch.from_numpy(np.ascontiguousarray(bank.global_[indices])),
            torch.from_numpy(np.ascontiguousarray(bank.state_scalars[indices])),
        )
    q = q.detach().cpu().numpy()
    if q.shape != (len(indices), 6) or not np.isfinite(q).all():
        raise ValueError("checkpoint produced invalid Q values")
    return np.asarray(q, dtype=np.float64)


def decisive_gap_report(net, bank, torch) -> tuple[np.ndarray, int]:
    """Return one ``Q(fatal)-Q(best surviving)`` gap per held-out decisive state.

    A decisive state must have exactly one B8 fatal branch (or exactly one fatal
    branch for legacy fixture banks) and at least one surviving alternative.  The
    strict shape check catches a bank-family mixup instead of silently changing
    the reported denominator.
    """
    values: list[float] = []
    excluded_nondecisive = 0
    for indices in _state_groups(bank):
        if not np.all(bank.family[indices] == 0):
            raise ValueError("held-out decisive bank contains non-decisive rows")
        actions = bank.action[indices]
        died = bank.died[indices]
        if bank.b8_action is None:
            raise ValueError("held-out decisive bank needs v2 recorded B8 actions")
        b8 = int(bank.b8_action[indices[0]])
        if not np.all(bank.b8_action[indices] == b8):
            raise ValueError("inconsistent B8 action within held-out source state")
        fatal = indices[(actions == b8) & died]
        survivors = indices[~died]
        if len(fatal) != 1 or not len(survivors):
            # Macro re-rolls can make a formerly fatal B8 boost survive.  Do not
            # substitute another fatal action: retain the registered B8-fatal
            # definition and make the changed denominator visible to the planner.
            excluded_nondecisive += 1
            continue
        q = _q_values(net, bank, indices, torch)[0]
        action_to_q = {int(action): float(q[int(action)]) for action in actions}
        values.append(
            action_to_q[int(bank.action[fatal[0]])]
            - max(action_to_q[int(a)] for a in bank.action[survivors])
        )
    if not values:
        raise ValueError("held-out decisive bank has no states")
    return np.asarray(values, dtype=np.float64), excluded_nondecisive


def decisive_gaps(net, bank, torch) -> np.ndarray:
    """Compatibility helper returning only the eligible held-out decisive gaps."""
    return decisive_gap_report(net, bank, torch)[0]


def outside_gaps(net, outside: OutsideObservations, torch, *, batch_size: int = 128) -> np.ndarray:
    """Return ``Q(B8)-max(Q(other legal actions))`` for held-out outside states."""
    if batch_size < 1:
        raise ValueError("outside batch size must be positive")
    values: list[float] = []
    for begin in range(0, len(outside.b8_action), batch_size):
        end = min(begin + batch_size, len(outside.b8_action))
        with torch.no_grad():
            q = (
                net(
                    torch.from_numpy(np.ascontiguousarray(outside.local[begin:end])),
                    torch.from_numpy(np.ascontiguousarray(outside.global_[begin:end])),
                    torch.from_numpy(np.ascontiguousarray(outside.scalars[begin:end])),
                )
                .detach()
                .cpu()
                .numpy()
            )
        if q.shape != (end - begin, 6) or not np.isfinite(q).all():
            raise ValueError("checkpoint produced invalid outside Q values")
        for row, b8, legal in zip(q, outside.b8_action[begin:end], outside.legal_mask[begin:end]):
            b8 = int(b8)
            if not legal[b8] or int(legal.sum()) < 2:
                raise ValueError("outside observation has no legal B8 action or comparator")
            other = np.asarray(legal, dtype=bool).copy()
            other[b8] = False
            values.append(float(row[b8]) - float(np.max(row[other])))
    if not values:
        raise ValueError("held-out outside observations have no states")
    return np.asarray(values, dtype=np.float64)


def load_outside_observations(location: Path) -> OutsideObservations:
    """Load and validate the hash-bound T4f held-out outside observation export."""
    location = Path(location)
    root = location.parent if location.is_file() else location
    data_path = location if location.is_file() else root / "observations.npz"
    manifest_path = root / "manifest.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise ValueError("held-out outside observations or manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text())
        expected = manifest["shards"][data_path.name]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("held-out outside manifest is invalid") from exc
    actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("held-out outside observation hash mismatch")
    if manifest.get("schema") != "t4f-heldout-outside-observations/v1":
        raise ValueError("unexpected held-out outside observation schema")
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            packed = archive["state_packed"].copy()
            scalars = archive["state_scalars"].copy()
            b8_action = archive["b8_action"].copy()
            legal_mask = archive["legal_mask"].copy()
            source_fields = {
                name: archive[name].copy()
                for name in ("source_seed", "source_decision", "source_plan_index")
            }
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError("held-out outside observations are unreadable") from exc
    if (
        packed.dtype != np.uint8
        or scalars.dtype != np.float32
        or scalars.ndim != 2
        or scalars.shape[1] != 60
        or b8_action.shape != (len(scalars),)
        or legal_mask.shape != (len(scalars), 6)
        or legal_mask.dtype != np.bool_
        or not np.issubdtype(b8_action.dtype, np.integer)
        or any(
            value.shape != (len(scalars),) or not np.issubdtype(value.dtype, np.integer)
            for value in source_fields.values()
        )
    ):
        raise ValueError("held-out outside observation shapes or dtypes are invalid")
    if (
        packed.shape[0] != len(scalars)
        or not np.isfinite(scalars).all()
        or (b8_action < 0).any()
        or (b8_action > 5).any()
    ):
        raise ValueError("held-out outside observation rows are invalid")
    from research.redesign_m3c_20261008.obs_pack import PackLayout, unpack_ref

    layout = PackLayout.from_dict(manifest.get("layout"))
    if layout is None or packed.shape[1] != layout.nbytes:
        raise ValueError("held-out outside observation layout is invalid")
    local, global_ = unpack_ref(packed, layout)
    return OutsideObservations(local, global_, scalars, b8_action.astype(np.int64), legal_mask)


def _read_telemetry(path: Path, transitions: int) -> Mapping[str, Any]:
    """Read the row at a look or the first 100k row that crossed it.

    Multi-actor packets can cross a scheduled look by a small amount.  The
    trainer saves the named checkpoint after that crossing, so its matching
    telemetry is the earliest row at or above the scheduled transition.  A
    lower-only row would describe different weights and is rejected.
    """
    if not path.is_file():
        raise ValueError("telemetry JSONL is missing")
    matched: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("telemetry JSONL contains invalid JSON") from exc
        if (
            isinstance(row.get("transitions"), int)
            and transitions <= row["transitions"] < transitions + 100_000
        ):
            matched.append(row)
    if not matched:
        raise ValueError("telemetry needs a row in the early-warning 100k transition window")
    first = min(row["transitions"] for row in matched)
    closest = [row for row in matched if row["transitions"] == first]
    if len(closest) != 1:
        raise ValueError("telemetry has duplicate nearest early-warning rows")
    return closest[0]


def _finite_field(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError(f"telemetry field {name!r} is missing or non-finite")
    return float(value)


def _telemetry_metrics(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Read registered dynamic-target error, its baseline, and the L<700 boost rate."""
    current = _finite_field(row, "return_mean_abs_contrast_error")
    baseline = _finite_field(row, "return_initial_mean_abs_contrast_error")
    low_boost = _finite_field(row, "greedy_boost_rate_lt_700")
    if current < 0 or baseline < 0 or not 0 <= low_boost <= 1:
        raise ValueError("invalid early-warning telemetry range")
    return current, baseline, low_boost


def evaluate(
    config: EarlyWarningConfig,
    *,
    bank_loader: Callable[[Path, int], Any] | None = None,
    outside_loader: Callable[[Path], OutsideObservations] | None = None,
    network_loader: Callable[[Path], Any] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint, returning a machine-readable PASS/KILL verdict.

    Raises ``ValueError`` for any absent, malformed, or incompatible input.  The
    CLI converts those failures to a durable ``NOT_EVALUATED`` receipt and exits
    nonzero, so launch code cannot interpret missing guards as a PASS.
    """
    if config.transitions not in LOOKS:
        raise ValueError("early warning supports only the 500k and 1M looks")
    for path in (config.checkpoint, config.initial_checkpoint):
        if not path.is_file():
            raise ValueError(f"checkpoint is missing: {path}")
    if bank_loader is None:
        from research.t4d_return_regression_20261016.bank import ReturnBank

        bank_loader = ReturnBank
    if network_loader is None:
        from src.model.ego2s_network import load_ego2s_checkpoint

        def network_loader(path: Path):
            return load_ego2s_checkpoint(str(path), "cpu")

    if outside_loader is None:
        outside_loader = load_outside_observations
    if torch_module is None:
        import torch

        torch_module = torch
    decisive = bank_loader(config.holdout_bank, 60)
    outside = outside_loader(config.outside_bank)
    checkpoint = network_loader(config.checkpoint).eval()
    initial = network_loader(config.initial_checkpoint).eval()
    decisive_values, decisive_excluded = decisive_gap_report(checkpoint, decisive, torch_module)
    outside_values = outside_gaps(checkpoint, outside, torch_module)
    initial_outside = outside_gaps(initial, outside, torch_module)
    telemetry_row = _read_telemetry(config.telemetry_jsonl, config.transitions)
    current_error, baseline_error, low_boost = _telemetry_metrics(telemetry_row)
    decisive_median = float(np.median(decisive_values))
    outside_median, outside_initial_median = float(np.median(outside_values)), float(
        np.median(initial_outside)
    )
    outside_delta = outside_median - outside_initial_median
    reasons: list[str] = []
    if config.transitions == 1_000_000:
        if decisive_median >= 0.0 and current_error < baseline_error:
            reasons.append("heldout_decisive_gap_nonnegative_while_train_error_fell")
        if abs(outside_delta) > 0.5:
            reasons.append("heldout_outside_gap_delta_exceeds_0.5")
        if low_boost > 0.01:
            reasons.append("lt_700_greedy_boost_rate_exceeds_1pct")
    return {
        "schema": SCHEMA,
        "status": "KILL" if reasons else "PASS",
        "transitions": config.transitions,
        "checkpoint": str(config.checkpoint),
        "initial_checkpoint": str(config.initial_checkpoint),
        "decisive": {
            "median_gap": decisive_median,
            "n_states": int(len(decisive_values)),
            "excluded_nondecisive": decisive_excluded,
        },
        "outside": {
            "median_gap": outside_median,
            "initial_median_gap": outside_initial_median,
            "delta": outside_delta,
            "n_states": int(len(outside_values)),
        },
        "train": {
            "contrast_abs_error_current": current_error,
            "contrast_abs_error_baseline": baseline_error,
            "fallen": current_error < baseline_error,
        },
        "boost": {
            "lt_700_rate": low_boost,
            "source_transition": int(telemetry_row["transitions"]),
        },
        "reasons": reasons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--holdout-bank", type=Path, required=True)
    parser.add_argument("--outside-bank", type=Path, required=True)
    parser.add_argument("--telemetry-jsonl", type=Path, required=True)
    parser.add_argument("--transitions", type=int, choices=sorted(LOOKS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = EarlyWarningConfig(
        checkpoint=args.checkpoint,
        initial_checkpoint=args.initial_checkpoint,
        holdout_bank=args.holdout_bank,
        outside_bank=args.outside_bank,
        telemetry_jsonl=args.telemetry_jsonl,
        transitions=args.transitions,
    )
    try:
        result = evaluate(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {
            "schema": SCHEMA,
            "status": "NOT_EVALUATED",
            "transitions": args.transitions,
            "reasons": [str(exc)],
        }
        _atomic_json(args.out, result)
        print(json.dumps(result, sort_keys=True))
        return 2
    _atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))
    return 3 if result["status"] == "KILL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
