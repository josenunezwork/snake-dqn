"""Create-only Check-4 rerun harness for the repaired T12 B8 serving path."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROTOCOL_SHA256 = "ce753f69c162f036a66187ac8eac384ff84e30abcb2f13dc2849e316785559c1"
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
NAMESPACE = "t12b-check4-20261017/v1"
INTENT_SCHEMA = "t12b-check4-intent/v1"
RESULT_SCHEMA = "t12b-check4-result/v1"
RECEIPT_SCHEMA = "t12b-check4-receipt/v1"
DEFAULT_FRAMES = 1500
DEFAULT_WARMUP = 200
PACED_TICKS = 120


class Blocked(RuntimeError):
    """Raised before timing when the machine cannot be shown to be quiet."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sources() -> dict[str, str]:
    """Bind all tracked Python/config/frontend sources before real timing begins."""
    names = git("ls-files").splitlines()
    return {
        name: sha(REPO / name)
        for name in names
        if name.endswith((".py", ".json", ".yaml", ".ts", ".tsx"))
        and name.startswith(("src/", "web/", "configs/", "research/"))
        and "/__pycache__/" not in name
        and not name.startswith("research/t12_b8_serving_20261017/results/")
    }


def _prior_bindings() -> dict[str, str]:
    root = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t12-b8-serving-20261017/run-v1")
    audit = root / "audit.json"
    receipt = root / "receipt.json"
    if not audit.is_file() or not receipt.is_file():
        raise RuntimeError("prior T12 audit/receipt are unavailable")
    prior = read(audit)
    if prior.get("verdict") != "SERVING_FAIL":
        raise RuntimeError("prior T12 failed audit is not preserved")
    return {
        "audit_path": str(audit),
        "audit_sha256": sha(audit),
        "receipt_path": str(receipt),
        "receipt_sha256": sha(receipt),
    }


def _verify_intent(root: Path) -> dict[str, Any]:
    from web.backend import served_checkpoint as registry

    intent = read(root / "intent.json")
    if sha(HERE / "protocol_check4.md") != PROTOCOL_SHA256:
        raise RuntimeError("local protocol differs from producer pin")
    if intent.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("intent does not bind protocol pin")
    if intent.get("namespace") != NAMESPACE:
        raise RuntimeError("wrong seed namespace")
    for rel, digest in intent.get("sources", {}).items():
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise RuntimeError(f"unsafe source binding: {rel!r}")
        if sha(REPO / rel) != digest:
            raise RuntimeError(f"registered source changed: {rel}")
    if registry.CHECKPOINT_RELEASED_DEFAULT != registry.NAME_FRP3_S12:
        raise RuntimeError("released default changed")
    checkpoint = intent.get("checkpoint", {})
    if checkpoint.get("name") != "b8-s3" or checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("intent candidate is not pinned b8-s3")
    if sha(Path(checkpoint["path"])) != CHECKPOINT_SHA256:
        raise RuntimeError("candidate bytes changed")
    if registry.strict_pin_problems("b8-s3"):
        raise RuntimeError("B8 strict pin is no longer valid")
    return intent


def prepare(root: Path, validation: Path, seed_audit: Path) -> None:
    """Write the real rerun's immutable intent; refuses any dirty tracked tree."""
    from research.t12b_check4_20261017 import seed_audit as seeds
    from web.backend import served_checkpoint as registry

    if root.exists():
        raise RuntimeError(f"output exists; refusing overwrite: {root}")
    if sha(HERE / "protocol_check4.md") != PROTOCOL_SHA256:
        raise RuntimeError("protocol pin mismatch")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"tracked tree is dirty: {dirty}")
    audit = read(seed_audit)
    allocation = audit.get("allocation", {})
    if audit.get("schema") != seeds.SCHEMA or audit.get("pass") is not True:
        raise RuntimeError("fresh seed audit did not pass")
    if allocation != seeds.generate_banks():
        raise RuntimeError("fresh seed allocation does not match registered recipe")
    choice = registry.resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "b8-s3"})
    if choice.name != "b8-s3" or choice.sha256 != CHECKPOINT_SHA256 or choice.reason:
        raise RuntimeError(f"candidate is not selected: {choice.to_dict()}")
    if registry.strict_pin_problems("b8-s3"):
        raise RuntimeError("B8 strict pin is invalid before Check-4 preparation")
    intent = {
        "schema": INTENT_SCHEMA,
        "created_utc": utc(),
        "protocol_sha256": PROTOCOL_SHA256,
        "namespace": NAMESPACE,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": False},
        "sources": sources(),
        "producer": {"run_path": str(HERE / "run.py"), "run_sha256": sha(HERE / "run.py")},
        "checkpoint": choice.to_dict(),
        "released_default": registry.CHECKPOINT_RELEASED_DEFAULT,
        "release_env": {"SNAKE_SERVE_CHECKPOINT": "b8-s3"},
        "seed_audit": {"path": str(seed_audit.resolve()), "sha256": sha(seed_audit)},
        "banks": allocation["banks"],
        "validation_sha256": sha(validation),
        "prior": _prior_bindings(),
    }
    write_new(root / "intent.json", intent)
    with (root / "validation.json").open("xb") as stream:
        stream.write(validation.read_bytes())
    print(json.dumps({"prepared": str(root), "intent_sha256": sha(root / "intent.json")}))


def _command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def quiet_snapshot() -> dict[str, Any]:
    """Read-only quiet-Mac evidence; never attach to or mutate foreign work."""
    from research.t12_b8_serving_20261017.measurements import \
        _mac_power_and_lid

    tmux = _command(
        [
            "tmux",
            "list-panes",
            "-a",
            "-F",
            "#{session_name}\t#{window_index}.#{pane_index}\t#{pane_pid}\t"
            "#{pane_current_command}\t#{pane_dead}",
        ]
    )
    processes = _command(["ps", "-axo", "pid=,ppid=,command="])
    power = _mac_power_and_lid()
    if tmux["returncode"] not in (0, 1) or processes["returncode"] != 0:
        raise Blocked("unable to obtain complete read-only tmux/process observation")
    parsed = []
    for line in processes["stdout"].splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            parsed.append((int(parts[0]), int(parts[1]), parts[2]))
    parent_of = {pid: ppid for pid, ppid, _command_text in parsed}
    own_and_ancestors = {os.getpid()}
    current = os.getpid()
    while True:
        parent = parent_of.get(current)
        if parent is None or parent in own_and_ancestors:
            break
        own_and_ancestors.add(parent)
        current = parent
    workload_executables = {"pytest", "uvicorn", "gunicorn"}
    suspicious = []
    for pid, _ppid, command_text in parsed:
        if pid in own_and_ancestors:
            continue
        executable = Path(command_text.split()[0]).name.lower() if command_text.split() else ""
        if "python" in executable or executable in workload_executables:
            suspicious.append(f"{pid} {command_text}")
    present = tmux["returncode"] == 0
    panes = []
    for line in tmux["stdout"].splitlines():
        fields = line.split("\t")
        if len(fields) == 5:
            panes.append(
                {
                    "raw": line,
                    "session": fields[0],
                    "window_pane": fields[1],
                    "pid": int(fields[2]) if fields[2].isdigit() else None,
                    "command": fields[3],
                    "dead": fields[4],
                }
            )
    children: dict[int, set[int]] = {}
    commands = {pid: command for pid, _ppid, command in parsed}
    for pid, ppid, _command_text in parsed:
        children.setdefault(ppid, set()).add(pid)

    def is_compute(command_text: str) -> bool:
        executable = Path(command_text.split()[0]).name.lower() if command_text.split() else ""
        # Shell and tmux command lines can retain historical launch arguments
        # after their child has exited. Only an active compute executable (or a
        # Python process, which covers project runners/tests) is a blocker.
        return "python" in executable or executable in workload_executables

    def descendant_commands(pid: int) -> list[str]:
        pending, seen, found = list(children.get(pid, set())), set(), []
        while pending:
            child = pending.pop()
            if child in seen:
                continue
            seen.add(child)
            found.append(commands.get(child, ""))
            pending.extend(children.get(child, set()))
        return found

    active_workload_panes = []
    for pane in panes:
        if pane["dead"] == "1":
            continue
        descendants = descendant_commands(pane["pid"]) if pane["pid"] is not None else []
        pane["descendant_commands"] = descendants
        if is_compute(str(pane["command"])) or any(is_compute(command) for command in descendants):
            active_workload_panes.append(pane)
    tmux_summary = {
        "present": present,
        "panes": panes,
        "all_panes_dead": present and bool(panes) and all(pane["dead"] == "1" for pane in panes),
        "workload_panes_exited": not active_workload_panes,
        "absent_or_workload_panes_exited": not present or not active_workload_panes,
        "raw": tmux,
    }
    allowed = (
        bool(power.get("ac_power"))
        and power.get("lid_closed") is False
        and not suspicious
        and not active_workload_panes
        and tmux_summary["absent_or_workload_panes_exited"]
    )
    return {
        "utc": utc(),
        "power": power,
        "tmux": tmux_summary,
        "processes": processes,
        "suspicious_processes": suspicious,
        "active_workload_panes": active_workload_panes,
        "quiet": allowed,
    }


def require_quiet() -> dict[str, Any]:
    observation = quiet_snapshot()
    if not observation["quiet"]:
        raise Blocked("quiet-Mac precondition is not met; no foreign process was touched")
    return observation


@contextlib.contextmanager
def _served_b8_environment() -> Iterator[None]:
    keys = (
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["SNAKE_SERVE_CHECKPOINT"] = "b8-s3"
        for key in keys[1:]:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _paced_probe(
    fps_request: int, seed: int, mode: str, snakes: int, warmup: int
) -> dict[str, Any]:
    """Probe live Hub-order ticks and prove the actual server-side clamp."""
    from research.t12_b8_serving_20261017 import measurements as old

    old._seed_everything(seed)
    old.set_world_seed(seed)
    session = old._new_session(mode, snakes)
    old._assert_b8_session(session)
    session.set_speed(fps_request)
    actual = float(session.speed)
    ceiling = float(session.speed_ceiling)
    warm = old._advance_warmup(
        session, old.ScriptedHuman(seed) if mode == old.MODE_PLAY else None, warmup
    )
    ticks: list[float] = []
    started = time.perf_counter()
    human = old.ScriptedHuman(seed) if mode == old.MODE_PLAY else None
    for _ in range(PACED_TICKS):
        if human is not None:
            old._stand_in_human_control(session, human)
        before = int(session.game.frame)
        _, elapsed_ms, _ = old._timed_tick(session)
        old._assert_live_advance(session, before)
        ticks.append(elapsed_ms)
        if mode == old.MODE_PLAY and session.run_over:
            session.reset_game()
            extra = old._advance_warmup(session, human, warmup)
            warm["advancing_frames"] += extra["advancing_frames"]
            warm["restarts"] += extra["restarts"]
        time.sleep(1.0 / actual)
    elapsed = time.perf_counter() - started
    budget = 1000.0 / actual
    return {
        "set_speed_request_fps": fps_request,
        "actual_clamped_fps": actual,
        "server_speed_ceiling_fps": ceiling,
        "ticks": PACED_TICKS,
        "budget_ms": budget,
        "raw_tick_ms": ticks,
        "overruns": sum(tick >= budget for tick in ticks),
        "tick_ms": old._distribution(ticks),
        "actual_achieved_fps": PACED_TICKS / elapsed,
        "warmup": warm,
        "served": old._served_identity(session),
        "roster": old._roster_identity(session),
        "last_error": session.last_error,
        "forward_mode": getattr(session.policy, "forward", None),
    }


def _case(seed: int, mode: str, snakes: int) -> dict[str, Any]:
    from research.t12_b8_serving_20261017 import measurements as old

    before = require_quiet()
    measured = old._measure_latency_case(mode, snakes, seed, DEFAULT_FRAMES, DEFAULT_WARMUP)
    # These are diagnostic bounds from the unpaced sample only. They are not a
    # validated server clamp or achieved cadence, so give them names that
    # cannot be confused with the paced `set_speed(120)` evidence below.
    measured["unpaced_max_tick_bound_fps"] = measured.pop("ui_speed_bound_fps")
    measured["unpaced_estimated_achieved_fps"] = measured.pop("estimated_achieved_fps")
    default_paced = _paced_probe(12, seed, mode, snakes, DEFAULT_WARMUP)
    ceiling_attempts = [_paced_probe(120, seed, mode, snakes, DEFAULT_WARMUP)]
    # A lower retry is deliberately unavailable unless serving source declares
    # a new lower clamp and set_speed(120) proves it. A mere lower request
    # would fake qualification of the unchanged ceiling.
    after = require_quiet()
    ceiling = ceiling_attempts[0]["actual_clamped_fps"]
    roster = dict(measured.get("roster") or {})
    roster["all_snakes_b8"] = bool(roster.get("all_ai_share_served_policy"))
    return {
        **measured,
        "roster": roster,
        "quiet_before": before,
        "quiet_after": after,
        "default_paced": default_paced,
        "ceiling_paced_attempts": ceiling_attempts,
        "clamped_ceiling_fps": ceiling,
        "forward_mode": default_paced["forward_mode"],
    }


def run(root: Path) -> int:
    """Run exactly Check 4 and persist the complete result envelope once."""
    intent = _verify_intent(root)
    path = root / "latency.json"
    if path.exists():
        raise RuntimeError(f"result exists; refusing overwrite: {path}")
    seeds = intent["banks"]["latency"]["seeds"]
    cases = (("watch", 6), ("watch", 12), ("play", 6), ("play", 12))
    envelope: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "intent_sha256": sha(root / "intent.json"),
        "kind": "latency",
        "seed_or_seeds": seeds,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "started_utc": utc(),
        "status": "error",
        "error": None,
    }
    started = time.monotonic()
    try:
        with _served_b8_environment():
            import torch

            torch.set_num_threads(1)
            preconditions = require_quiet()
            completed = []
            for index, (seed, (mode, snakes)) in enumerate(zip(seeds, cases)):
                case = _case(int(seed), mode, snakes)
                completed.append(case)
                write_new(
                    root / "cases" / f"latency-{index:03d}.json",
                    {
                        "schema": RESULT_SCHEMA,
                        "intent_sha256": envelope["intent_sha256"],
                        "kind": "latency_case",
                        "index": index,
                        "seed_or_seeds": int(seed),
                        "checkpoint_sha256": CHECKPOINT_SHA256,
                        "status": "complete",
                        "data": case,
                    },
                )
            envelope["data"] = {
                "checkpoint": "b8-s3",
                "frames_per_case": DEFAULT_FRAMES,
                "warmup_per_construction_or_play_reset": DEFAULT_WARMUP,
                "paced_ticks": PACED_TICKS,
                "preconditions": preconditions,
                "threading": {"torch_num_threads": torch.get_num_threads()},
                "cases": completed,
                "definitions": {
                    "update_ms": "wall time inside GameState.update",
                    "tick_ms": (
                        "GameSession.step followed by snapshot; network/client rendering excluded"
                    ),
                    "forward_path": "existing live batched serving",
                },
            }
            envelope["status"] = "complete"
    except Exception as exc:
        envelope["error"] = f"{type(exc).__name__}: {exc}"
        envelope["traceback"] = traceback.format_exc()
    envelope["wall_seconds"] = time.monotonic() - started
    write_new(path, envelope)
    print(
        json.dumps({"result": str(path), "status": envelope["status"], "error": envelope["error"]})
    )
    return 0 if envelope["status"] == "complete" else 1


def finalize(root: Path) -> None:
    """Bind the intent, validation record, and Check-4 result without overwriting."""
    _verify_intent(root)
    required = (
        root / "intent.json",
        root / "validation.json",
        root / "latency.json",
        *(root / "cases" / f"latency-{index:03d}.json" for index in range(4)),
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("cannot finalize an incomplete Check-4 run")
    write_new(
        root / "receipt.json",
        {
            "schema": RECEIPT_SCHEMA,
            "created_utc": utc(),
            "intent_sha256": sha(root / "intent.json"),
            "files": {str(path.relative_to(root)): sha(path) for path in required},
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--seed-audit", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.validation is None or args.seed_audit is None:
            parser.error("prepare requires --validation and --seed-audit")
        prepare(args.root, args.validation, args.seed_audit)
        return 0
    if args.action == "run":
        return run(args.root)
    finalize(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
