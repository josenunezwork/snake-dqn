"""Fixed-recipe fitting and frozen E3 information-value decision rule.

This module is deliberately self-contained at the experiment boundary: it accepts
only reconstructed, finite arrays and publishes fenced, create-only exports.
It is development evidence only and has no T34 verdict authority.
"""

from __future__ import annotations

import copy
import io
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from research.bl_info_value.io import export_bytes, write_json
from src.scripts.eval_stats import student_t_isf

SEED = 20261021
FULL_ROWS = 1_536
FULL_WORLDS = 256
SPLIT_WORLD_COUNTS = (192, 32, 32)
SPLIT_CLASS_COUNTS = (576, 96, 96)
ARM_NAMES = ("A", "B", "C")


class IncompleteInput(ValueError):
    """Input failure which must be reported as E3 INCOMPLETE."""


def _as_1d(value: np.ndarray, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows:
        raise IncompleteInput(f"{name} must be a length-{rows} vector")
    return array


def _finite(array: np.ndarray, name: str) -> None:
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise IncompleteInput(f"{name} must be a non-object numeric array")
    if not np.isfinite(array).all():
        raise IncompleteInput(f"{name} contains non-finite values")


def validate_arrays(arrays: dict[str, np.ndarray], *, smoke: bool = False) -> int:
    """Validate the fixed E3 sample construction before any fit starts."""
    required = {
        "local",
        "global",
        "scalars",
        "geometry",
        "history",
        "legal",
        "returns",
        "action",
        "decisive",
        "world",
        "split",
        "decision",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise IncompleteInput(f"missing arrays: {', '.join(missing)}")
    rows = len(np.asarray(arrays["action"]))
    if rows == 0:
        raise IncompleteInput("empty input")
    for key in required:
        value = np.asarray(arrays[key])
        if value.ndim == 0 or value.shape[0] != rows:
            raise IncompleteInput(f"{key} has inconsistent row count")
        if key not in {"world", "split", "decision", "legal", "decisive"}:
            _finite(value, key)
    shapes = {
        "local": (rows, 7, 31, 31),
        "global": (rows, 4, 37, 37),
        "scalars": (rows, 60),
        "geometry": (rows, 7, 83, 145),
        "history": (rows, 128, 43),
    }
    for key, expected in shapes.items():
        if np.asarray(arrays[key]).shape != expected:
            raise IncompleteInput(f"{key} must have shape {expected}")
    legal = np.asarray(arrays["legal"])
    returns = np.asarray(arrays["returns"])
    if legal.ndim != 2 or legal.shape != (rows, 6):
        raise IncompleteInput("legal must have shape (N, 6)")
    if returns.ndim != 2 or returns.shape != (rows, 6):
        raise IncompleteInput("returns must have shape (N, 6)")
    _finite(returns, "returns")
    if legal.dtype.hasobject:
        raise IncompleteInput("legal must not be an object array")
    if not np.all(legal.astype(bool).any(axis=1)):
        raise IncompleteInput("every row must have a resolved legal action")
    action = _as_1d(arrays["action"], "action", rows)
    if not np.issubdtype(action.dtype, np.integer) or np.any((action < 0) | (action >= 6)):
        raise IncompleteInput("action must contain action indices in [0, 5]")
    if not np.all(legal[np.arange(rows), action].astype(bool)):
        raise IncompleteInput("recorded action is illegal")
    split = _as_1d(arrays["split"], "split", rows)
    if not np.issubdtype(split.dtype, np.integer) or set(np.unique(split)) - {0, 1, 2}:
        raise IncompleteInput("split must use only 0=train, 1=validation, 2=test")
    raw_decisive = _as_1d(arrays["decisive"], "decisive", rows)
    if raw_decisive.dtype.hasobject or not np.all((raw_decisive == 0) | (raw_decisive == 1)):
        raise IncompleteInput("decisive must be boolean or binary")
    decisive = raw_decisive.astype(bool)
    world = _as_1d(arrays["world"], "world", rows)
    decision = _as_1d(arrays["decision"], "decision", rows)
    try:
        keys = [(str(w), int(d)) for w, d in zip(world, decision)]
    except (TypeError, ValueError) as error:
        raise IncompleteInput("world/decision identities are malformed") from error
    if len(set(keys)) != rows:
        raise IncompleteInput("duplicate (world, decision) root")
    if smoke:
        return rows
    if rows != FULL_ROWS:
        raise IncompleteInput(f"expected {FULL_ROWS} roots, got {rows}")
    worlds = set(str(item) for item in world)
    if len(worlds) != FULL_WORLDS:
        raise IncompleteInput(f"expected {FULL_WORLDS} worlds, got {len(worlds)}")
    for split_id, expected_worlds, expected_per_class in zip(
        range(3), SPLIT_WORLD_COUNTS, SPLIT_CLASS_COUNTS
    ):
        where = split == split_id
        actual_worlds = set(str(item) for item in world[where])
        if len(actual_worlds) != expected_worlds:
            raise IncompleteInput(
                f"split {split_id} has {len(actual_worlds)} worlds, expected {expected_worlds}"
            )
        if (
            int(np.sum(decisive[where])) != expected_per_class
            or int(np.sum(~decisive[where])) != expected_per_class
        ):
            raise IncompleteInput(
                f"split {split_id} does not have fixed decisive/counterweight quotas"
            )
        for world_id in actual_worlds:
            if np.any((world.astype(str) == world_id) & ~where):
                raise IncompleteInput(f"world leakage for {world_id}")
    return rows


def _tensor_batch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> dict[str, Tensor]:
    batch: dict[str, Tensor] = {}
    for key, value in arrays.items():
        if key in {"world", "split", "decision", "decisive"}:
            continue
        chosen = np.asarray(value)[indices]
        if key == "legal":
            batch[key] = torch.as_tensor(chosen, device=device, dtype=torch.bool)
        elif key == "action":
            batch[key] = torch.as_tensor(chosen, device=device, dtype=torch.long)
        else:
            batch[key] = torch.as_tensor(chosen, device=device, dtype=torch.float32)
    return batch


def _masked_choice(scores: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal.astype(bool), scores, -np.inf)
    # np.argmax's first-index tie behavior is the frozen rule.
    return np.argmax(masked, axis=1).astype(np.int64)


def _regrets(
    scores: np.ndarray, arrays: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    legal = np.asarray(arrays["legal"])[indices].astype(bool)
    returns = np.asarray(arrays["returns"])[indices]
    action = np.asarray(arrays["action"])[indices]
    pick = _masked_choice(scores, legal)
    best = np.where(legal, returns, -np.inf).max(axis=1)
    return best - returns[np.arange(len(indices)), pick], pick != action


def _world_means(values: np.ndarray, worlds: np.ndarray) -> dict[str, float]:
    answer: dict[str, float] = {}
    for world in sorted(set(str(item) for item in worlds)):
        answer[world] = float(np.mean(values[worlds.astype(str) == world]))
    return answer


def _lower90(deltas: list[float]) -> float:
    if len(deltas) < 2:
        raise IncompleteInput("at least two held-out worlds are required")
    average = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1))
    if std == 0.0:
        # With zero standard error the finite one-sided bound is the observed
        # mean; emitting infinity would also violate immutable JSON export.
        return average
    return average - student_t_isf(0.10, len(deltas) - 1) * std / math.sqrt(len(deltas))


def decide_outcome(arms: dict[str, dict[str, Any]], *, smoke: bool = False) -> dict[str, Any]:
    """Apply the frozen whole-world E3 decision rule to already-held-out metrics."""
    if smoke:
        return {"outcome": "SMOKE_NOT_SCIENTIFIC", "eligible_for_go_kill": False}
    a = arms["A"]
    qualifiers: list[str] = []
    comparisons: dict[str, dict[str, Any]] = {}
    for arm_name in ("B", "C"):
        candidate = arms[arm_name]
        a_world = a["decisive_world_regret"]
        c_world = candidate["decisive_world_regret"]
        if set(a_world) != set(c_world):
            raise IncompleteInput("held-out world identities differ between arms")
        deltas = [a_world[key] - c_world[key] for key in sorted(a_world)]
        reduction = (
            None
            if a["decisive_mean_regret"] <= 0
            else 1.0 - candidate["decisive_mean_regret"] / a["decisive_mean_regret"]
        )
        lb90 = _lower90(deltas)
        qualified = bool(
            a["decisive_mean_regret"] > 0
            and reduction is not None
            and candidate["decisive_mean_regret"] <= 0.80 * a["decisive_mean_regret"]
            and lb90 > 0.0
            and candidate["unnecessary_switch_rate"] <= 0.01
        )
        comparisons[arm_name] = {
            "reduction": reduction,
            "lb90_a_minus_candidate": lb90,
            "qualifies": qualified,
        }
        if qualified:
            qualifiers.append(arm_name)
    history_reduction = None
    history_retained = False
    if "C" in qualifiers and arms["B"]["decisive_mean_regret"] > 0:
        history_reduction = (
            1.0 - arms["C"]["decisive_mean_regret"] / arms["B"]["decisive_mean_regret"]
        )
        history_retained = (
            arms["C"]["decisive_mean_regret"] <= 0.90 * arms["B"]["decisive_mean_regret"]
        )
    selected = (
        "B"
        if "B" in qualifiers and ("C" not in qualifiers or not history_retained)
        else ("C" if "C" in qualifiers and history_retained else None)
    )
    return {
        "outcome": "GO" if qualifiers else "KILL",
        "eligible_for_go_kill": True,
        "qualifying_arms": qualifiers,
        "comparisons": comparisons,
        "history_reduction_b_to_c": history_reduction,
        "history_retained": history_retained,
        "selected_architecture": selected,
    }


def _scores(model: torch.nn.Module, batch: dict[str, Tensor]) -> Tensor:
    scores = model.forward(batch)
    if scores.ndim != 2 or scores.shape[1] != 6:
        raise IncompleteInput("student forward must return (N, 6) scores")
    if not torch.isfinite(scores).all():
        raise IncompleteInput("student forward produced non-finite scores")
    return scores


def _synchronize(device: torch.device) -> None:
    """Fence CUDA timing without changing CPU behavior."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _repeated(indices: np.ndarray, size: int) -> np.ndarray:
    """Return exactly ``size`` roots by cycling only the supplied training roots."""
    if len(indices) == 0:
        raise IncompleteInput("cannot benchmark an empty training split")
    return np.resize(indices, size)


def _benchmark(
    arrays: dict[str, np.ndarray],
    students: dict[str, torch.nn.Module],
    train: np.ndarray,
    device: torch.device,
    loss_fn: Any,
) -> dict[str, dict[str, float]]:
    """Run the fixed small operational probe, separate from scientific metrics."""
    batch32 = _repeated(train, 32)
    batch192 = _repeated(train, 192)
    result: dict[str, dict[str, float]] = {}
    for name in ARM_NAMES:
        # These are newly constructed throwaway students. They cannot alter a
        # selected state dict or add another fitting look.
        model = copy.deepcopy(students[name]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        updates: list[float] = []
        model.train()
        for _ in range(3):
            batch = _tensor_batch(arrays, batch32, device)
            optimizer.zero_grad(set_to_none=True)
            _synchronize(device)
            started = time.monotonic()
            loss = loss_fn(_scores(model, batch), batch["returns"], batch["legal"])
            loss.backward()
            optimizer.step()
            _synchronize(device)
            updates.append(time.monotonic() - started)
        forwards: list[float] = []
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                batch = _tensor_batch(arrays, batch192, device)
                _synchronize(device)
                started = time.monotonic()
                _scores(model, batch)
                _synchronize(device)
                forwards.append(time.monotonic() - started)
        result[name] = {
            "batch32_update_seconds": float(np.median(updates)),
            "batch192_inference_seconds": float(np.median(forwards)),
        }
    return result


def fit_students(
    arrays: dict[str, np.ndarray],
    out: Path,
    device: str = "cpu",
    epochs: int = 60,
    smoke: bool = False,
) -> dict[str, Any]:
    """Fit the three fixed E3 students and export checkpoints/report immutably."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    validate_arrays(arrays, smoke=smoke)
    from research.bl_info_value.students import make_students, ranking_loss

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    target = torch.device(device)
    students = make_students(seed=SEED)
    if set(students) != set(ARM_NAMES):
        raise IncompleteInput("make_students must return exactly A, B, C")
    counts = {
        name: sum(item.numel() for item in model.parameters() if item.requires_grad)
        for name, model in students.items()
    }
    if len(set(counts.values())) != 1:
        raise IncompleteInput(f"students are not parameter matched: {counts}")
    train = np.flatnonzero(np.asarray(arrays["split"]) == 0)
    validation = np.flatnonzero(np.asarray(arrays["split"]) == 1)
    test = np.flatnonzero(np.asarray(arrays["split"]) == 2)
    if smoke and (not len(train) or not len(validation) or not len(test)):
        raise IncompleteInput("smoke still requires nonempty train, validation, and test splits")
    started = time.monotonic()
    benchmark = _benchmark(arrays, students, train, target, ranking_loss) if smoke else {}
    trained: dict[str, tuple[torch.nn.Module, int, float]] = {}
    for name in ARM_NAMES:
        torch.manual_seed(SEED)
        model = students[name].to(target)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        best_epoch, best_value, best_state = -1, math.inf, None
        for epoch in range(epochs):
            model.train()
            generator = np.random.default_rng(SEED + epoch)
            shuffled = train[generator.permutation(len(train))]
            for begin in range(0, len(train), 32):
                index = shuffled[begin : begin + 32]
                batch = _tensor_batch(arrays, index, target)
                optimizer.zero_grad(set_to_none=True)
                loss = ranking_loss(_scores(model, batch), batch["returns"], batch["legal"])
                if not torch.isfinite(loss):
                    raise IncompleteInput(f"non-finite loss for {name}, epoch {epoch + 1}")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise IncompleteInput(f"non-finite gradient for {name}, epoch {epoch + 1}")
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise IncompleteInput(f"non-finite parameter for {name}, epoch {epoch + 1}")
            model.eval()
            with torch.no_grad():
                score = (
                    _scores(model, _tensor_batch(arrays, validation, target)).detach().cpu().numpy()
                )
            if not np.isfinite(score).all():
                raise IncompleteInput(f"non-finite validation scores for {name}, epoch {epoch + 1}")
            regret, _ = _regrets(score, arrays, validation)
            choose = np.asarray(arrays["decisive"])[validation].astype(bool)
            validation_worlds = np.asarray(arrays["world"])[validation][choose]
            value = float(np.mean(list(_world_means(regret[choose], validation_worlds).values())))
            if not math.isfinite(value):
                raise IncompleteInput("non-finite validation metric")
            if value < best_value:
                best_epoch, best_value, best_state = (
                    epoch + 1,
                    value,
                    copy.deepcopy(model.state_dict()),
                )
            print(
                f"E3 {name} epoch={epoch + 1}/{epochs} val_decisive_regret={value:.9g}", flush=True
            )
        assert best_state is not None
        model.load_state_dict(best_state)
        buffer = io.BytesIO()
        torch.save(
            {
                "arm": name,
                "epoch": best_epoch,
                "seed": SEED,
                "state_dict": best_state,
                "resume": False,
            },
            buffer,
        )
        export_bytes(out / f"student_{name}_best.pt", buffer.getvalue())
        trained[name] = (model, best_epoch, best_value)
    arm_report: dict[str, dict[str, Any]] = {}
    for name, (model, best_epoch, best_value) in trained.items():
        model.eval()
        with torch.no_grad():
            score = _scores(model, _tensor_batch(arrays, test, target)).detach().cpu().numpy()
        if not np.isfinite(score).all():
            raise IncompleteInput(f"non-finite held-out scores for {name}")
        regret, changed = _regrets(score, arrays, test)
        decisive = np.asarray(arrays["decisive"])[test].astype(bool)
        worlds = np.asarray(arrays["world"])[test]
        action = np.asarray(arrays["action"])[test]
        returns = np.asarray(arrays["returns"])[test]
        original_best = np.where(np.asarray(arrays["legal"])[test], returns, -np.inf).max(axis=1)
        original_regret = original_best - returns[np.arange(len(test)), action]
        unnecessary = changed & (regret > original_regret + 1e-9) & ~decisive
        per_world = _world_means(regret[decisive], worlds[decisive])
        arm_report[name] = {
            "best_epoch": best_epoch,
            "validation_decisive_mean_regret": best_value,
            "decisive_mean_regret": float(np.mean(list(per_world.values()))),
            "decisive_world_regret": per_world,
            "unnecessary_switch_rate": (
                float(np.mean(unnecessary[~decisive])) if np.any(~decisive) else math.nan
            ),
        }
    decision = decide_outcome(arm_report, smoke=smoke)
    report: dict[str, Any] = {
        "schema": "bl-e3-fit-v1",
        "development_only": True,
        "rows": int(len(np.asarray(arrays["action"]))),
        "parameter_counts": counts,
        "arms": arm_report,
        "decision": decision,
        "benchmark": benchmark,
        "operational_fit_seconds": time.monotonic() - started,
    }
    write_json(out / "fit_report.json", report)
    return report
