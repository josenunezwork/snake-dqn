"""Torch-free, source-derived inventory of the C/T progress emitters.

Admission reads the sealed producer sources, never imports or executes them.
Unknown emitter shapes require an explicit code change and coverage test.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
CALLBACKS = {
    "collection-progress": "ct_task.py",
    "scoring-progress": "ct_branching.py",
    "fit-progress": "ct_fit.py",
}


def _keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        raise ValueError("progress emitter requires a literal dict")
    return {key.value for key in node.keys if isinstance(key, ast.Constant)}


def _stage_pattern(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return re.escape(node.value)
    if isinstance(node, ast.JoinedStr):
        pattern = ""
        for part in node.values:
            if isinstance(part, ast.Constant):
                pattern += re.escape(part.value)
            elif (
                isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Name)
                and part.value.id in {"stage_num", "final_stage"}
                and ast.unparse(part.format_spec) == "f'04d'"
            ):
                pattern += r"\d{4}"
            else:
                raise ValueError("unqualified dynamic progress stage")
        return pattern
    raise ValueError("unqualified progress stage expression")


@functools.lru_cache(maxsize=1)
def progress_inventory() -> dict[str, frozenset[str]]:
    """Generate stage regex -> exact allowed keys from every emit call."""
    tree = ast.parse((HERE / "ct_run.py").read_text())
    base = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rec" for t in node.targets)
    )
    common = _keys(base)
    inventory: dict[str, frozenset[str]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit"
        ):
            continue
        pattern = _stage_pattern(node.args[0])
        extra: set[str] = set()
        if len(node.args) > 1:
            arg = node.args[1]
            if isinstance(arg, ast.Dict):
                extra = _keys(arg)
            elif isinstance(arg, ast.Name) and arg.id == "info":
                stage = ast.literal_eval(node.args[0])
                callback = ast.parse((HERE / CALLBACKS[stage]).read_text())
                calls = [
                    call
                    for call in ast.walk(callback)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "log_progress"
                ]
                if not calls:
                    raise ValueError("missing progress callback emitter")
                for call in calls:
                    extra.update(_keys(call.args[0]))
            else:
                raise ValueError("unqualified progress payload expression")
        inventory[pattern] = frozenset(common | extra)
    return inventory


def validate_progress(row: dict[str, Any], *, fixture_completion: bool = False) -> None:
    """Validate closed producer keys and typed values, without exposing values."""
    stage = row.get("stage")
    keys = next(
        (
            keys
            for pattern, keys in progress_inventory().items()
            if isinstance(stage, str) and re.fullmatch(pattern, stage)
        ),
        None,
    )
    if keys is None or set(row) != keys:
        raise ValueError("unqualified progress stage or keys")
    if row.get("schema") != "t34-cf-progress/v1" or row.get("study") != "t34-cf-1028":
        raise ValueError("unqualified progress schema")
    strings = {
        "schema",
        "study",
        "stage",
        "phase",
        "forced_state_device",
        "topology_fixtures",
        "support_status",
        "verdict",
    }
    for key, value in row.items():
        if key in {"schema", "study", "stage"}:
            continue
        if key == "phase":
            if not isinstance(value, str) or not re.fullmatch(
                r"collection|scoring_chunk_\d+_(?:selection|scoring)|"
                r"fit_(?:frozen_phi|full_bv8|full_bv8_plus_topology)",
                value,
            ):
                raise ValueError("unqualified progress phase")
        elif key in strings:
            choices = {
                "forced_state_device": {"cpu", "cuda", "cuda:0"},
                "topology_fixtures": {"PASS"},
                "support_status": {"PASS"},
                # Completed outcomes are refused by fault/summary precedence.
                "verdict": ({"STOP_RESCUE_TARGET_PROGRAM"} if fixture_completion else set()),
            }
            if not isinstance(value, str) or value not in choices[key]:
                raise ValueError("unqualified progress string")
        elif key == "verified_stages":
            if not isinstance(value, list) or not all(type(v) is int and v > 0 for v in value):
                raise ValueError("unqualified verified stages")
        elif key == "compact_units":
            if not isinstance(value, list) or not all(
                isinstance(v, str) and re.fullmatch(r"a100-dev/t34-cf-1028/stages/stage-\d{4}", v)
                for v in value
            ):
                raise ValueError("unqualified compact units")
        elif type(value) not in (int, float) or not -float("inf") < value < float("inf"):
            raise ValueError("unqualified progress number")
        elif key not in {"searched_kl", "forbidden_margin", "weighted_anchor_kl"} and value < 0:
            raise ValueError("negative progress counter")
