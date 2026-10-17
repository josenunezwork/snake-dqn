#!/usr/bin/env python3
"""Fail-closed, stdlib-only audit for the T12B Check 4 rerun.

This deliberately consumes retained evidence only.  It never imports the
runner or serving implementation, so a producer cannot certify its own output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "protocol_check4.md"
PROTOCOL_SHA256 = "ce753f69c162f036a66187ac8eac384ff84e30abcb2f13dc2849e316785559c1"
CHECKPOINT_NAME = "b8-s3"
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
INTENT_SCHEMA = "t12b-check4-intent/v1"
RESULT_SCHEMA = "t12b-check4-result/v1"
AUDIT_SCHEMA = "t12b-check4-audit/v1"
CASES = {
    "Watch6": ("watch", 6),
    "Watch12": ("watch", 12),
    "Play6": ("play", 6),
    "Play12": ("play", 12),
}
FRAME_COUNT = 1500
PACED_TICKS = 120
DEFAULT_FPS = 12
P95_LIMIT_MS = 66.667
NAMESPACE = "t12b-check4-20261017/v1"


class Invalid(RuntimeError):
    """Evidence is missing, malformed, or does not bind to this rerun."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Invalid(f"unreadable JSON {path}: {exc}") from exc


def _hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _registered_seeds() -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"{NAMESPACE}|latency|{index}".encode()).digest()[:4], "big")
        for index in range(4)
    ]


def _seed_digest(seeds: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def _nonnegative_samples(value: Any, count: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != count:
        return None
    if not all(_number(item) and float(item) >= 0 for item in value):
        return None
    return [float(item) for item in value]


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lower, upper = math.floor(index), math.ceil(index)
    return (
        ordered[lower]
        if lower == upper
        else ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    )


def _source_problems(intent: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    sources = intent.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        return ["intent.sources is empty"]
    for rel, digest in sources.items():
        path = Path(str(rel))
        if path.is_absolute() or ".." in path.parts or not _hex(digest):
            problems.append(f"unsafe/unhashed source binding: {rel}")
        elif not (REPO / path).is_file() or _sha(REPO / path) != digest:
            problems.append(f"registered source changed: {rel}")
    # The rerun must bind the serving policy path, the server clamp, and config
    # bytes.  This makes a reported ceiling inseparable from its implementation.
    for required in (
        "web/backend/ego2s_policy.py",
        "web/backend/session.py",
        "web/backend/serving_speed.py",
    ):
        if required not in sources:
            problems.append(f"intent does not bind required serving source: {required}")
    if not any(str(rel).startswith("configs/") for rel in sources):
        problems.append("intent does not bind a served configuration")
    return problems


def _prior_problems(intent: Mapping[str, Any]) -> list[str]:
    prior = intent.get("prior")
    if not isinstance(prior, Mapping):
        return ["prior qualification evidence is absent"]
    out: list[str] = []
    loaded: dict[str, Mapping[str, Any]] = {}
    for key in ("audit_path", "receipt_path"):
        path_value = prior.get(key)
        digest = prior.get(key.replace("_path", "_sha256"))
        path = Path(str(path_value))
        if (
            not isinstance(path_value, str)
            or not _hex(digest)
            or not path.is_file()
            or _sha(path) != digest
        ):
            out.append(f"prior {key} is absent or changed")
        else:
            document = _json(path)
            if not isinstance(document, Mapping):
                out.append(f"prior {key} is not an object")
            else:
                loaded[key] = document
    audit = loaded.get("audit_path", {})
    criteria = audit.get("criteria") if isinstance(audit, Mapping) else None
    if not isinstance(criteria, Mapping):
        out.append("prior audit has no criteria")
    else:
        for number in ("1", "2", "3", "5", "6", "7", "8"):
            if (
                not isinstance(criteria.get(number), Mapping)
                or criteria[number].get("pass") is not True
            ):
                out.append(f"prior audit does not carry passed check {number}")
        if not isinstance(criteria.get("4"), Mapping) or criteria["4"].get("pass") is not False:
            out.append("prior audit does not retain the failed check 4")
    receipt = loaded.get("receipt_path", {})
    if not isinstance(receipt.get("files"), Mapping) or not _hex(receipt.get("intent_sha256")):
        out.append("prior receipt is not a bound file inventory")
    return out


def _intent_problems(intent: Mapping[str, Any], root: Path) -> list[str]:
    out: list[str] = []
    if intent.get("schema") != INTENT_SCHEMA:
        out.append("wrong intent schema")
    protocol_sha = intent.get("protocol_sha256")
    if protocol_sha != PROTOCOL_SHA256 or not PROTOCOL.is_file() or _sha(PROTOCOL) != protocol_sha:
        out.append("protocol amendment bytes do not match the intent pin")
    git = intent.get("git")
    if (
        not isinstance(git, Mapping)
        or not isinstance(git.get("commit"), str)
        or not git.get("commit")
    ):
        out.append("intent lacks git commit")
    elif git.get("dirty") not in (False, [], ""):
        out.append("intent was not created from a clean tracked tree")
    checkpoint = intent.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("name") != CHECKPOINT_NAME
        or checkpoint.get("sha256") != CHECKPOINT_SHA256
    ):
        out.append("intent is not pinned to b8-s3")
    elif (
        not isinstance(checkpoint.get("path"), str)
        or not Path(checkpoint["path"]).is_file()
        or _sha(Path(checkpoint["path"])) != CHECKPOINT_SHA256
    ):
        out.append("checkpoint bytes differ from b8-s3 pin")
    if intent.get("released_default") != "frp3-s12":
        out.append("qualification changed CHECKPOINT_RELEASED_DEFAULT")
    validation = root / "validation.json"
    if (
        not _hex(intent.get("validation_sha256"))
        or not validation.is_file()
        or _sha(validation) != intent.get("validation_sha256")
    ):
        out.append("intent validation binding is absent or changed")
    seed = intent.get("seed_audit")
    if (
        not isinstance(seed, Mapping)
        or not isinstance(seed.get("path"), str)
        or not _hex(seed.get("sha256"))
    ):
        out.append("seed freshness binding is absent")
    else:
        path = Path(seed["path"])
        if not path.is_file() or _sha(path) != seed["sha256"]:
            out.append("seed freshness report bytes differ from intent")
        else:
            report = _json(path)
            expected_banks = {
                "latency": {
                    "count": 4,
                    "seeds": _registered_seeds(),
                    "seed_digest": _seed_digest(_registered_seeds()),
                }
            }
            exclusions = (
                report.get("explicit_exclusions", {}) if isinstance(report, Mapping) else {}
            )
            inherited = report.get("prior_audit", {}) if isinstance(report, Mapping) else {}
            scan = (
                report.get("unexcluded_t12b_artifact_scan", {})
                if isinstance(report, Mapping)
                else {}
            )
            if (
                not isinstance(report, Mapping)
                or report.get("pass") is not True
                or report.get("schema") != "t12b-check4-seed-audit/v1"
                or report.get("namespace") != NAMESPACE
                or report.get("allocation", {}).get("banks") != expected_banks
                or intent.get("banks") != expected_banks
                or inherited.get("pass") is not True
                or exclusions.get("collision_seeds") != []
                or scan.get("hits") != []
            ):
                out.append("seed freshness report did not pass")
    banks = intent.get("banks")
    latency = banks.get("latency") if isinstance(banks, Mapping) else None
    if (
        not isinstance(latency, Mapping)
        or latency.get("count") != 4
        or not isinstance(latency.get("seeds"), list)
        or len(latency["seeds"]) != 4
        or not _hex(latency.get("seed_digest"))
    ):
        out.append("latency seed bank is not exactly four bound seeds")
    out.extend(_source_problems(intent))
    out.extend(_prior_problems(intent))
    return out


def _paced_problems(
    attempt: Any,
    expected_actual: int,
    expected_ceiling: int,
    require_request: int | None = None,
) -> list[str]:
    if not isinstance(attempt, Mapping):
        return ["paced attempt is missing"]
    requested = attempt.get("set_speed_request_fps")
    actual = attempt.get("actual_clamped_fps")
    if (
        not _number(requested)
        or float(requested) < 1
        or (require_request is not None and float(requested) != require_request)
    ):
        return ["paced request is wrong"]
    ceiling = attempt.get("server_speed_ceiling_fps")
    if not _number(actual) or float(actual) != expected_actual:
        return ["paced result does not prove the accepted server speed"]
    if not _number(ceiling) or float(ceiling) != expected_ceiling:
        return ["paced result does not prove the server-enforced ceiling"]
    raw = _nonnegative_samples(attempt.get("raw_tick_ms"), PACED_TICKS)
    if attempt.get("ticks") != PACED_TICKS or raw is None:
        return ["paced raw samples are incomplete or invalid"]
    overruns = sum(value >= 1000.0 / actual for value in raw)
    if attempt.get("overruns") != overruns:
        return ["paced overrun count is not raw-derived"]
    return [] if overruns == 0 else ["paced run has budget overruns"]


def _case_problems(row: Any, seed: Any) -> list[str]:
    if not isinstance(row, Mapping):
        raise Invalid("case is not an object")
    name = row.get("case")
    mode_snakes = CASES.get(name)
    if (
        mode_snakes is None
        or (row.get("mode"), row.get("snakes")) != mode_snakes
        or row.get("seed") != seed
    ):
        raise Invalid("case identity does not bind the registered roster/seed")
    samples = row.get("samples")
    update = _nonnegative_samples(
        samples.get("update_ms") if isinstance(samples, Mapping) else None, FRAME_COUNT
    )
    tick = _nonnegative_samples(
        samples.get("tick_ms") if isinstance(samples, Mapping) else None, FRAME_COUNT
    )
    if update is None or tick is None:
        raise Invalid(f"{name} does not retain exactly {FRAME_COUNT} finite nonnegative samples")
    out: list[str] = []
    if _p95(update) >= P95_LIMIT_MS:
        out.append(f"{name} update p95 fails {P95_LIMIT_MS}ms")
    if row.get("advancing_measured_frames") != FRAME_COUNT:
        raise Invalid(f"{name} did not advance exactly {FRAME_COUNT} measured frames")
    served = row.get("served")
    registry = served.get("registry", served) if isinstance(served, Mapping) else {}
    if (
        not isinstance(registry, Mapping)
        or registry.get("name") != CHECKPOINT_NAME
        or registry.get("sha256") != CHECKPOINT_SHA256
    ):
        raise Invalid(f"{name} does not prove b8-s3 serving identity")
    roster = row.get("roster")
    expected_total = mode_snakes[1]
    expected_ai = expected_total if mode_snakes[0] == "watch" else expected_total - 1
    expected_human = None if mode_snakes[0] == "watch" else 0
    if (
        not isinstance(roster, Mapping)
        or roster.get("all_snakes_b8") is not True
        or roster.get("all_ai_share_served_policy") is not True
        or roster.get("ai_count") != expected_ai
        or roster.get("total_snakes") != expected_total
        or roster.get("human_id") != expected_human
        or roster.get("ai_veto_wrappers") != []
    ):
        raise Invalid(f"{name} does not prove an all-b8 driven roster")
    cap = 24 if mode_snakes[1] <= 6 else 12
    default_problems = _paced_problems(row.get("default_paced"), DEFAULT_FPS, cap, DEFAULT_FPS)
    if default_problems and default_problems != ["paced run has budget overruns"]:
        raise Invalid(f"{name} default evidence is malformed: {'; '.join(default_problems)}")
    out.extend(f"{name} default: {item}" for item in default_problems)
    if row.get("clamped_ceiling_fps") != cap:
        raise Invalid(f"{name} has an unbound or wrong roster ceiling")
    attempts = row.get("ceiling_paced_attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise Invalid(f"{name} ceiling must retain exactly one validated attempt")
    else:
        first = _paced_problems(attempts[0], cap, cap, 120)
        if first:
            if first != ["paced run has budget overruns"]:
                raise Invalid(f"{name} ceiling evidence is malformed: {'; '.join(first)}")
            out.extend(f"{name} ceiling: {item}" for item in first)
    return out


def _quiet_snapshot_problems(pre: Any) -> list[str]:
    """Validate one raw, read-only quiet-Mac observation."""
    if not isinstance(pre, Mapping) or pre.get("quiet") is not True:
        return ["quiet-Mac precondition is absent"]
    power = pre.get("power")
    processes = pre.get("processes")
    if (
        not isinstance(power, Mapping)
        or power.get("ac_power") is not True
        or power.get("lid_closed") is not False
    ):
        return ["quiet-Mac power/lid evidence is absent"]
    if (
        not isinstance(processes, Mapping)
        or processes.get("returncode") != 0
        or not isinstance(processes.get("stdout"), str)
        or not any(
            len(line.strip().split(None, 2)) == 3
            and line.strip().split(None, 2)[0].isdigit()
            and line.strip().split(None, 2)[1].isdigit()
            for line in processes["stdout"].splitlines()
        )
    ):
        return ["quiet-Mac process snapshot is absent or unparseable"]
    if pre.get("suspicious_processes") != []:
        return ["quiet-Mac has suspicious processes"]
    tmux = pre.get("tmux")
    if not isinstance(tmux, Mapping):
        return ["quiet-Mac tmux evidence is absent"]
    # New harnesses retain the raw tmux command instead of a synthesized
    # present/dead summary.  Either representation is independently useful.
    if "present" not in tmux:
        if tmux.get("returncode") not in (0, 1) or not isinstance(tmux.get("stdout"), str):
            return ["quiet-Mac tmux command evidence is malformed"]
        if pre.get("active_workload_panes") != []:
            return ["quiet-Mac has active workload tmux panes"]
    else:
        present = tmux.get("present")
        complete = (
            present is True
            and tmux.get("workload_panes_exited") is True
            and isinstance(tmux.get("panes"), list)
            and bool(tmux["panes"])
        )
        if present is not False and not complete:
            return ["quiet-Mac tmux evidence does not show no running workload panes"]
    return []


def _quiet_problems(data: Mapping[str, Any], rows: Sequence[Any]) -> list[str]:
    """Require observed tmux quiescence before and after the live measurements."""
    problems = _quiet_snapshot_problems(data.get("preconditions"))
    if problems:
        return problems
    for row in rows:
        before = row.get("quiet_before") if isinstance(row, Mapping) else None
        after = row.get("quiet_after") if isinstance(row, Mapping) else None
        for label, snapshot in (("before", before), ("after", after)):
            snapshot_problems = _quiet_snapshot_problems(snapshot)
            problems.extend(f"case {label}: {problem}" for problem in snapshot_problems)
    return problems


def audit(root: str | Path) -> dict[str, Any]:
    """Return PASS, CHECK4_FAIL, or INVALID for one completed Check 4 rerun."""
    root = Path(root)
    try:
        intent_path, latency_path, receipt_path = (
            root / "intent.json",
            root / "latency.json",
            root / "receipt.json",
        )
        intent, latency, receipt = _json(intent_path), _json(latency_path), _json(receipt_path)
        if not all(isinstance(item, Mapping) for item in (intent, latency, receipt)):
            raise Invalid("intent, latency, or receipt is not an object")
        intent_sha = _sha(intent_path)
        if receipt.get("intent_sha256") != intent_sha:
            raise Invalid("receipt does not bind intent bytes")
        files = receipt.get("files")
        expected = {
            "intent.json",
            "validation.json",
            "latency.json",
            "cases/latency-000.json",
            "cases/latency-001.json",
            "cases/latency-002.json",
            "cases/latency-003.json",
        }
        if not isinstance(files, Mapping) or set(files) != expected:
            raise Invalid("receipt inventory has omissions or unexpected files")
        for rel, digest in files.items():
            path = root / rel
            if not _hex(digest) or not path.is_file() or _sha(path) != digest:
                raise Invalid(f"receipt hash mismatch: {rel}")
        intent_problems = _intent_problems(intent, root)
        if intent_problems:
            raise Invalid("; ".join(intent_problems))
        required = {
            "schema",
            "intent_sha256",
            "kind",
            "seed_or_seeds",
            "checkpoint_sha256",
            "started_utc",
            "status",
            "error",
            "wall_seconds",
            "data",
        }
        if (
            set(latency) != required
            or latency.get("schema") != RESULT_SCHEMA
            or latency.get("intent_sha256") != intent_sha
        ):
            raise Invalid("latency envelope does not bind this intent/schema")
        if (
            latency.get("kind") != "latency"
            or latency.get("checkpoint_sha256") != CHECKPOINT_SHA256
        ):
            raise Invalid("latency envelope has wrong kind/checkpoint")
        if (
            latency.get("status") != "complete"
            or latency.get("error") is not None
            or not _number(latency.get("wall_seconds"))
            or float(latency["wall_seconds"]) < 0
        ):
            raise Invalid("latency envelope is incomplete or invalid")
        data = latency.get("data")
        if not isinstance(data, Mapping) or data.get("frames_per_case") != FRAME_COUNT:
            raise Invalid("latency payload does not bind the 1500-frame protocol")
        bank = intent["banks"]["latency"]["seeds"]
        if latency.get("seed_or_seeds") != bank:
            raise Invalid("latency envelope does not bind the registered seeds")
        rows = data.get("cases")
        if (
            not isinstance(rows, list)
            or len(rows) != 4
            or {row.get("case") for row in rows if isinstance(row, Mapping)} != set(CASES)
        ):
            raise Invalid("latency cases are incomplete")
        failures: list[str] = []
        quiet_problems = _quiet_problems(data, rows)
        if quiet_problems:
            raise Invalid("; ".join(quiet_problems))
        for index, row in enumerate(rows):
            case_path = root / "cases" / f"latency-{index:03d}.json"
            case = _json(case_path)
            if (
                not isinstance(case, Mapping)
                or case.get("schema") != RESULT_SCHEMA
                or case.get("intent_sha256") != intent_sha
                or case.get("kind") != "latency_case"
                or case.get("index") != index
                or case.get("seed_or_seeds") != bank[index]
                or case.get("checkpoint_sha256") != CHECKPOINT_SHA256
                or case.get("status") != "complete"
                or case.get("data") != row
            ):
                raise Invalid(f"case evidence does not bind aggregate row {index}")
        for row, seed in zip(rows, bank):
            failures.extend(_case_problems(row, seed))
        return {
            "schema": AUDIT_SCHEMA,
            "verdict": "SERVING_PASS" if not failures else "SERVING_FAIL",
            "intent_sha256": intent_sha,
            "protocol_sha256": intent["protocol_sha256"],
            "failures": failures,
            "check4_qualified": not failures,
        }
    except (Invalid, KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        return {
            "schema": AUDIT_SCHEMA,
            "verdict": "INVALID",
            "failures": [],
            "error": str(exc),
            "check4_qualified": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)
    return {"SERVING_PASS": 0, "SERVING_FAIL": 1, "INVALID": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
