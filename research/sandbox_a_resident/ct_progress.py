"""Torch-free, source-derived inventory of the C/T progress emitters.

Admission reads the sealed producer sources, never imports or executes them.
Unknown emitter shapes require an explicit code change and coverage test.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import math
import re
from pathlib import Path
from typing import Any

from research.sandbox_a_resident import ct_json as json

HERE = Path(__file__).parent
CALLBACKS = {
    "collection-progress": "ct_task.py",
    "scoring-progress": "ct_branching.py",
    "fit-progress": "ct_fit.py",
}
HEARTBEAT_SOURCES = ("ct_liveness.py", "ct_run.py", "ct_cuda_canary.py", "ct_mechanics.py")


@functools.lru_cache(maxsize=1)
def capture_inventory() -> dict[str, frozenset[str]]:
    """Derive closed fixture report/receive-record shapes from the real producer."""
    tree = ast.parse((HERE / "ct_cuda_canary.py").read_text())
    inventory = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"report", "cadence"}:
                    inventory[target.id] = frozenset(_keys(node.value))
                elif (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "report"
                ):
                    key = ast.literal_eval(target.slice)
                    inventory["report"] = inventory["report"] | {key}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"probes", "observations"}
        ):
            keys = frozenset(_keys(node.args[0]))
            name = (
                ("signal" if "marker" in keys else "complete")
                if node.func.value.id == "probes"
                else "observation"
            )
            inventory[name] = keys
    if set(inventory) != {"report", "cadence", "complete", "signal", "observation"}:
        raise ValueError("missing fixture capture emitter AST")
    return inventory


@functools.lru_cache(maxsize=1)
def heartbeat_inventory() -> tuple[frozenset[str], frozenset[str]]:
    """Derive closed heartbeat keys and phases from the actual producer ASTs."""
    phases: set[str] = set()
    keys: set[str] = set()
    for name in HEARTBEAT_SOURCES:
        tree = ast.parse((HERE / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and name == "ct_liveness.py":
                if any(isinstance(t, ast.Name) and t.id == "row" for t in node.targets):
                    keys.update(_keys(node.value))
                if any(isinstance(t, ast.Attribute) and t.attr == "phase" for t in node.targets):
                    if isinstance(node.value, ast.Constant):
                        phases.add(ast.literal_eval(node.value))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "set_phase"
            ):
                phase = ast.literal_eval(node.args[0])
                if not isinstance(phase, str):
                    raise ValueError("heartbeat phase must be a literal string")
                phases.add(phase)
    if not phases or not keys:
        raise ValueError("missing heartbeat emitter AST")
    return frozenset(keys), frozenset(phases)


def validate_heartbeat(row: dict[str, Any]) -> None:
    """Admit only the AST-derived status form; it conveys no work completion."""
    keys, phases = heartbeat_inventory()
    if (
        set(row) != keys
        or row.get("schema") != "t34-cf-heartbeat/v1"
        or row.get("study") != "t34-cf-1028"
        or not isinstance(row.get("phase"), str)
        or row["phase"] not in phases
        or type(row.get("elapsed_sec")) not in (int, float)
        or not 0 <= row["elapsed_sec"] < float("inf")
    ):
        raise ValueError("unqualified heartbeat schema, phase, keys or elapsed time")


def verify_cadence(log: str, record: dict[str, Any]) -> None:
    """Recompute fixture gaps from hash-bound lines and parent receive times.

    Producer elapsed counters have phase-local clocks. Include launch-to-first
    and last-to-exit gaps. This proves cadence only, never CUDA or receipt identity.
    """
    lines = [
        line
        for line in log.splitlines(keepends=True)
        if line.startswith(("CT_PROGRESS ", "CT_HEARTBEAT ", "CT_SIGUSR1_CLEAN_EXIT "))
    ]
    observations = record.get("observations")
    if (
        not isinstance(observations, list)
        or len(lines) != len(observations)
        or record.get("line_count") != len(lines)
        or record.get("clock") != "parent-monotonic-receive-including-start-and-exit"
        or record.get("limit_seconds") != 60
        or record.get("heartbeat_limit_seconds") != 30
    ):
        raise ValueError("unqualified cadence observation inventory")
    points, heartbeats = [0.0], [0.0]
    for line, observation in zip(lines, observations):
        if (
            not isinstance(observation, dict)
            or set(observation) != {"received_seconds", "line_sha256"}
            or observation["line_sha256"] != hashlib.sha256(line.encode()).hexdigest()
        ):
            raise ValueError("cadence line hash mismatch")
        value = observation["received_seconds"]
        if type(value) not in (int, float) or not points[-1] <= value < float("inf"):
            raise ValueError("non-monotonic cadence observation")
        points.append(value)
        if line.startswith("CT_HEARTBEAT "):
            validate_heartbeat(json.loads(line.partition(" ")[2]))
            heartbeats.append(value)
    exited = record.get("exit_seconds")
    if (
        type(exited) not in (int, float)
        or not points[-1] <= exited < float("inf")
        or len(heartbeats) == 1
    ):
        raise ValueError("unqualified cadence exit or missing heartbeat")
    points.append(exited)
    heartbeats.append(exited)
    for sequence, key, limit in (
        (points, "max_gap_seconds", 60),
        (heartbeats, "heartbeat_max_gap_seconds", 30),
    ):
        maximum = max(b - a for a, b in zip(sequence, sequence[1:]))
        claimed = record.get(key)
        if (
            type(claimed) not in (int, float)
            or not math.isclose(claimed, maximum, rel_tol=0, abs_tol=1e-8)
            or maximum > limit
        ):
            raise ValueError("cadence exceeds frozen limit or disagrees with observations")


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
