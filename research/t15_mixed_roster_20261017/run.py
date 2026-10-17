"""Pinned, create-only coordinator and single-unit worker for T15."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
NAMESPACE = "t15-mixed-roster-20261017/v1"
SMOKE_NAMESPACE = "t15-mixed-roster-20261017/smoke-v1"
PROTOCOL_SHA256 = "0b882fc2a11e47ce67a0ec541f560d157bd32c6e62c95ce9554683633603428d"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
PARITY_PLAN = ((5000, 6), (5000, 8), (10000, 12))


class Blocked(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")


def sources() -> dict[str, str]:
    """Bind runtime code, UI source, and the exact pre-built browser bundle."""
    names = git("ls-files").splitlines()
    allowed = ("src/", "web/", "configs/", "research/")
    tracked = {
        name: sha(REPO / name)
        for name in names
        if name.startswith(allowed)
        and name.endswith((".py", ".json", ".yaml", ".ts", ".tsx", ".css", ".html", ".js"))
        and not name.startswith("research/t15_mixed_roster_20261017/results/")
    }
    dist = REPO / "web/frontend/dist"
    if not dist.is_dir():
        raise RuntimeError("built frontend dist is required before T15 preparation")
    built = {str(path.relative_to(REPO)): sha(path) for path in dist.rglob("*") if path.is_file()}
    if not built:
        raise RuntimeError("built frontend dist is empty")
    return {**tracked, **built}


def _registry_identity() -> dict[str, Any]:
    from web.backend import served_checkpoint as registry

    if registry.CHECKPOINT_RELEASED_DEFAULT != registry.NAME_B8_S3:
        raise RuntimeError("T15 requires unchanged released default b8-s3")
    if registry.WATCH_ROSTER_RELEASED_DEFAULT != "all_b8":
        raise RuntimeError("T15 requires unchanged Watch roster default all_b8")
    b8 = registry.resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "b8-s3"})
    frp = registry.resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "frp3-s12"})
    if b8.sha256 != B8_SHA256 or b8.reason or frp.sha256 != FRP3_SHA256 or frp.reason:
        raise RuntimeError(
            f"checkpoint registry pins invalid: b8={b8.to_dict()} frp3={frp.to_dict()}"
        )
    if registry.strict_pin_problems("b8-s3") or registry.strict_pin_problems("frp3-s12"):
        raise RuntimeError("strict checkpoint pin invalid")
    return {
        "released_default": registry.CHECKPOINT_RELEASED_DEFAULT,
        "b8": b8.to_dict(),
        "frp3": frp.to_dict(),
        "watch_roster_default": registry.WATCH_ROSTER_RELEASED_DEFAULT,
        "pins_sha256": sha(Path(registry.PINS_PATH)),
    }


def verify(root: Path) -> dict[str, Any]:
    intent = read(root / "intent.json")
    if intent.get("namespace") not in (NAMESPACE, SMOKE_NAMESPACE):
        raise RuntimeError("wrong namespace")
    if (
        sha(HERE / "protocol.md") != PROTOCOL_SHA256
        or intent.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise RuntimeError("protocol bytes differ from pin")
    for rel, digest in intent["sources"].items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts or sha(REPO / rel) != digest:
            raise RuntimeError(f"registered source changed: {rel}")
    actual = _registry_identity()
    if actual != intent["registry"]:
        raise RuntimeError("registry/default/pins changed")
    return intent


def prepare(root: Path, validation: Path, seed_audit: Path, smoke: bool = False) -> None:
    from research.t15_mixed_roster_20261017 import seed_audit as seeds

    if root.exists():
        raise RuntimeError(f"output exists: {root}")
    if sha(HERE / "protocol.md") != PROTOCOL_SHA256:
        raise RuntimeError("protocol pin mismatch")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not smoke:
        raise RuntimeError(f"tracked tree is dirty: {dirty}")
    audit = read(seed_audit)
    validation_record = read(validation)
    if validation_record.get("pass") is not True:
        raise RuntimeError("passing preflight validation required")
    allocation = seeds.generate_banks(SMOKE_NAMESPACE if smoke else NAMESPACE)
    if not smoke and (
        audit.get("schema") != seeds.SCHEMA
        or audit.get("pass") is not True
        or audit.get("allocation") != allocation
    ):
        raise RuntimeError("passing matching T15 seed audit required")
    intent = {
        "schema": "t15-mixed-roster-intent/v1",
        "created_utc": utc(),
        "smoke": smoke,
        "namespace": allocation["namespace"],
        "protocol_sha256": PROTOCOL_SHA256,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)},
        "sources": sources(),
        "producer": {"path": str(HERE / "run.py"), "sha256": sha(HERE / "run.py")},
        "registry": _registry_identity(),
        "release_env": {
            "SNAKE_SERVE_CHECKPOINT": None,
            "SNAKE_SERVE_WATCH_ROSTER": "mixed",
            "SNAKE_SERVE_VETO_VARIANT": None,
            "SNAKE_SERVE_VETO_WATCH_HERO": None,
            "SNAKE_SERVE_VETO_PLAY_AI": None,
        },
        "seed_audit": {"path": str(seed_audit.resolve()), "sha256": sha(seed_audit)},
        "banks": allocation["banks"],
        "validation_sha256": sha(validation),
        "validation_schema": validation_record.get("schema"),
        "counts": {"watch": 25, "parity": 3, "latency": 3, "ui": 1, "controls": 1},
    }
    write_new(root / "intent.json", intent)
    with (root / "validation.json").open("xb") as stream:
        stream.write(validation.read_bytes())
    print(json.dumps({"prepared": str(root), "intent_sha256": sha(root / "intent.json")}))


def quiet_snapshot() -> dict[str, Any]:
    """Read-only process/power check; intentionally never inspects tmux."""
    ps = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], text=True, capture_output=True, check=False
    )
    if ps.returncode:
        raise Blocked("unable to observe processes")
    entries = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            entries.append((int(parts[0]), int(parts[1]), parts[2]))
    parents = {pid: ppid for pid, ppid, _ in entries}
    own = {os.getpid()}
    current = os.getpid()
    while current in parents and parents[current] not in own:
        current = parents[current]
        own.add(current)
    rows = []
    blockers = []
    for pid, _, command in entries:
        executable = Path(command.split()[0]).name.lower() if command.split() else ""
        lower = command.lower()
        is_compute = (
            "python" in executable
            or executable in {"pytest", "uvicorn", "gunicorn"}
            or any(term in lower for term in ("tournament_eval", "apex_train", "train_pqn"))
        )
        rows.append(
            {
                "pid": pid,
                "ppid": parents[pid],
                "executable": executable,
                "is_compute": is_compute,
                "task_ancestor": pid in own,
            }
        )
        if is_compute and pid not in own:
            blockers.append({"pid": pid, "ppid": parents[pid], "executable": executable})
    power = subprocess.run(["pmset", "-g", "batt"], text=True, capture_output=True, check=False)
    lid = subprocess.run(
        ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
        text=True,
        capture_output=True,
        check=False,
    )
    lid_text = lid.stdout.lower()
    lid_closed = (
        True
        if '"appleclamshellstate" = yes' in lid_text
        else False if '"appleclamshellstate" = no' in lid_text else None
    )
    pmset_short = "\n".join(power.stdout.strip().splitlines()[:3])
    power_evidence = {
        "ac_power": "ac power" in power.stdout.lower(),
        "lid_closed": lid_closed,
        "pmset_short": pmset_short,
        "pmset_returncode": power.returncode,
        "ioreg_returncode": lid.returncode,
    }
    quiet = bool(power_evidence["ac_power"]) and lid_closed is False and not blockers
    return {
        "utc": utc(),
        "power": power_evidence,
        "processes": {
            "command": ["ps", "-axo", "pid=,ppid=,command="],
            "returncode": ps.returncode,
            "rows": rows,
            "task_ancestor_pids": sorted(own),
        },
        "foreign_active_compute": blockers,
        "quiet": quiet,
        "tmux_touched": False,
    }


def _path(root: Path, kind: str, index: int) -> Path:
    return root / "records" / f"{kind}-{index:03d}.json"


def unit(root: Path, kind: str, index: int) -> int:
    import torch

    intent = verify(root)
    path = _path(root, kind, index)
    if path.exists():
        raise RuntimeError(f"result exists: {path}")
    if kind not in ("watch", "parity", "latency"):
        raise ValueError(kind)
    bank = intent["banks"][kind]
    if not 0 <= index < bank["count"]:
        raise ValueError(f"invalid {kind} index")
    seed = int(bank["seeds"][index])
    torch.set_num_threads(1)
    result: dict[str, Any] = {
        "schema": "t15-mixed-roster-result/v1",
        "intent_sha256": sha(root / "intent.json"),
        "kind": kind,
        "index": index,
        "seed": seed,
        "status": "error",
        "error": None,
        "started_utc": utc(),
    }
    started = time.monotonic()
    try:
        if intent["smoke"] and kind == "latency":
            raise ValueError("smoke never runs full latency")
        if kind == "watch":
            from research.t15_mixed_roster_20261017.episodes import run_watch

            data = run_watch(seed, horizon=100 if intent["smoke"] else 5000, num_snakes=6)
        elif kind == "parity":
            from research.t15_mixed_roster_20261017.parity import run_identity

            horizon, snakes = PARITY_PLAN[index]
            data = run_identity(
                seed, horizon=100 if intent["smoke"] else horizon, num_snakes=snakes
            )
        else:
            from research.t15_mixed_roster_20261017.measurements import run_latency

            before = quiet_snapshot()
            if not before["quiet"]:
                raise Blocked("quiet machine precondition failed; foreign process untouched")
            data = run_latency(seed, (6, 8, 12)[index])
            after = quiet_snapshot()
            data = {
                "measurement": data,
                "quiet_before": before,
                "quiet_after": after,
                "quiet_postcondition_pass": after["quiet"],
            }
        result.update(status="complete", data=data)
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    result["wall_seconds"] = time.monotonic() - started
    write_new(path, result)
    print(json.dumps({"result": str(path), "status": result["status"], "error": result["error"]}))
    return 0 if result["status"] == "complete" else 1


def finalize(root: Path) -> None:
    intent = verify(root)
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "receipt.json"
    )
    write_new(
        root / "receipt.json",
        {
            "schema": "t15-mixed-roster-receipt/v1",
            "created_utc": utc(),
            "intent_sha256": sha(root / "intent.json"),
            "smoke": intent["smoke"],
            "files": {str(path.relative_to(root)): sha(path) for path in files},
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "unit", "finalize", "quiet"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--seed-audit", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kind", choices=("watch", "parity", "latency"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "quiet":
        print(json.dumps(quiet_snapshot(), sort_keys=True))
    elif args.action == "prepare":
        if args.root is None:
            parser.error("prepare requires --root")
        if args.validation is None or args.seed_audit is None:
            parser.error("prepare requires --validation and --seed-audit")
        prepare(args.root, args.validation, args.seed_audit, args.smoke)
    elif args.action == "unit":
        if args.root is None:
            parser.error("unit requires --root")
        return unit(args.root, args.kind, args.index)
    else:
        if args.root is None:
            parser.error("finalize requires --root")
        finalize(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
