"""Pinned, create-only coordinator/worker CLI for the T12 Mac serving qualification."""

from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import patch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROTOCOL_SHA256 = "3d8b51675a2cdecbc476af56c48cfb4fa3ac6566eac1d4faf21c1870174bd406"
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
SMOKE_NAMESPACE = "t12-b8-serving-20261017/smoke-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sources() -> dict[str, str]:
    files = git("ls-files").splitlines()
    return {
        name: sha(REPO / name)
        for name in files
        if name.endswith((".py", ".yaml", ".json", ".ts", ".tsx"))
        and name.startswith(("src/", "web/backend/", "web/frontend/src/", "configs/", "research/"))
        and not name.startswith("research/t12_b8_serving_20261017/results/")
    }


def guard() -> None:
    from research.frp3_checkpoint_serving_20261005.serving_run import make_guard

    make_guard()()


def verify(root: Path) -> dict:
    from web.backend import served_checkpoint as registry

    intent = read(root / "intent.json")
    if sha(HERE / "protocol.md") != PROTOCOL_SHA256 or intent["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("protocol differs from pinned bytes")
    for name, digest in intent["sources"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError(f"registered source changed: {name}")
    if registry.CHECKPOINT_RELEASED_DEFAULT != registry.NAME_FRP3_S12:
        raise RuntimeError("released default changed")
    if registry.strict_pin_problems("b8-s3"):
        raise RuntimeError("B8 strict pin invalid")
    if registry.B8_S3_SHA256 in registry.v8_gated_checkpoint_sha256s():
        raise RuntimeError("B8 improperly admitted to v8 allowlist")
    if sha(Path(intent["checkpoint"]["path"])) != CHECKPOINT_SHA256:
        raise RuntimeError("candidate bytes changed")
    return intent


def prepare(root: Path, validation: Path, smoke: bool = False) -> None:
    from research.t12_b8_serving_20261017.seed_audit import generate_banks
    from web.backend import served_checkpoint as registry

    if root.exists():
        raise RuntimeError("output exists; refusing overwrite")
    if sha(HERE / "protocol.md") != PROTOCOL_SHA256:
        raise RuntimeError("protocol pin mismatch")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not smoke:
        raise RuntimeError(f"tracked tree is dirty: {dirty}")
    audit_path = HERE / "SEED_AUDIT_T12_V2.json"
    audit = read(audit_path)
    if not audit["pass"]:
        raise RuntimeError("seed freshness did not pass")
    banks = generate_banks(SMOKE_NAMESPACE) if smoke else generate_banks()
    if not smoke and banks["banks"] != audit["banks"]:
        raise RuntimeError("seed allocation differs from freshness audit")
    choice = registry.resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "b8-s3"})
    if choice.name != "b8-s3" or choice.sha256 != CHECKPOINT_SHA256 or choice.reason:
        raise RuntimeError(f"candidate not selected: {choice.to_dict()}")
    pin = registry.load_pins()["checkpoints"]["b8-s3"]
    for value in pin["strict_receipt"].values():
        if isinstance(value, dict) and "path" in value:
            path = Path(pin["strict_receipt"]["root"]) / value["path"]
            if sha(path) != value["sha256"]:
                raise RuntimeError(f"strict evidence bytes differ: {path}")
    guard()
    intent = {
        "schema": "t12-serving-intent/v1",
        "created_utc": utc(),
        "smoke": smoke,
        "protocol_sha256": PROTOCOL_SHA256,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)},
        "sources": sources(),
        "checkpoint": choice.to_dict(),
        "strict_pin": pin,
        "released_default": registry.CHECKPOINT_RELEASED_DEFAULT,
        "release_env": {
            "SNAKE_SERVE_CHECKPOINT": "b8-s3",
            "SNAKE_SERVE_VETO_WATCH_HERO": None,
            "SNAKE_SERVE_VETO_PLAY_AI": None,
            "SNAKE_SERVE_VETO_VARIANT": None,
        },
        "seed_audit": {"path": str(audit_path), "sha256": sha(audit_path)},
        "banks": banks["banks"],
        "namespace": banks["namespace"],
        "validation_sha256": sha(validation),
        "max_worker_processes": 4,
    }
    write_new(root / "intent.json", intent)
    with (root / "validation.json").open("xb") as stream:
        stream.write(validation.read_bytes())
    print(
        json.dumps({"prepared": str(root), "intent_sha256": sha(root / "intent.json")}), flush=True
    )


def defaults(seeds: list[int], frames: int = 100) -> dict:
    from research.frp3_checkpoint_serving_20261005 import serving_run as old
    from web.backend import served_checkpoint as registry
    from web.backend.safety_veto_serving import logger as veto_logger
    from web.backend.session import GameSession

    cases = [
        ("empty_env", {}),
        ("champion", {"SNAKE_SERVE_CHECKPOINT": "champion"}),
        ("frp3-s12", {"SNAKE_SERVE_CHECKPOINT": "frp3-s12"}),
        ("b8-s3", {"SNAKE_SERVE_CHECKPOINT": "b8-s3"}),
        ("unknown", {"SNAKE_SERVE_CHECKPOINT": "unknown-t12"}),
        ("v7_no_name", {"SNAKE_SERVE_VETO_VARIANT": "v7"}),
        ("unfilled", {"SNAKE_SERVE_CHECKPOINT": "b8-s3"}),
        ("both_unfilled", {"SNAKE_SERVE_CHECKPOINT": "b8-s3"}),
        ("corrupt_bytes", {"SNAKE_SERVE_CHECKPOINT": "b8-s3"}),
    ]
    output = {}
    for index, (label, env) in enumerate(cases):
        pin_doc = copy.deepcopy(registry.load_pins())
        if label in ("unfilled", "both_unfilled"):
            pin_doc["checkpoints"]["b8-s3"]["strict_receipt"] = None
        if label == "both_unfilled":
            pin_doc["checkpoints"]["frp3-s12"]["strict_receipt"] = None
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        registry.logger.addHandler(handler)
        veto_logger.addHandler(handler)
        try:
            with (
                tempfile.TemporaryDirectory(prefix="t12-refusal-") as temp,
                contextlib.ExitStack() as stack,
            ):
                stack.enter_context(old.serving_env({key: env.get(key) for key in old.RELEASE_ENV}))
                stack.enter_context(old.pins_override(pin_doc))
                if label == "corrupt_bytes":
                    corrupt = Path(temp) / registry.B8_S3_FILENAME
                    corrupt.write_bytes(b"qualification deliberately corrupt temporary candidate")
                    old_entry = registry.REGISTRY["b8-s3"]
                    replacement = dataclasses.replace(
                        old_entry, filename=str(corrupt), artifact_path=None
                    )
                    stack.enter_context(patch.dict(registry.REGISTRY, {"b8-s3": replacement}))
                old.set_world_seed(seeds[index % len(seeds)])
                session = GameSession()
                watch = {
                    "choice": session.served_checkpoint,
                    "veto": session.safety_veto_state(),
                    "last_error": session.last_error,
                }
                count = frames if label in ("champion", "frp3-s12") else 0
                for _ in range(count):
                    session.step()
                    session.snapshot()
                watch["frames_stepped"] = count
                old.control(session, "set_mode", "play")
                output[label] = {
                    "watch": watch,
                    "play_veto": session.safety_veto_state(),
                    "play_last_error": session.last_error,
                }
        finally:
            registry.logger.removeHandler(handler)
            veto_logger.removeHandler(handler)
        output[label]["stderr"] = log.getvalue()
    return {
        "cases": output,
        "released_default": registry.CHECKPOINT_RELEASED_DEFAULT,
        "v8_gated_sha256s": registry.v8_gated_checkpoint_sha256s(),
        "b8_pin": registry.load_pins()["checkpoints"]["b8-s3"],
    }


def result_path(root: Path, kind: str, index: int) -> Path:
    if kind in ("watch", "play"):
        return root / "records" / f"{kind}-{index:03d}.json"
    if kind == "parity":
        return root / f"parity-{index:03d}.json"
    return root / f"{kind}.json"


def unit(root: Path, kind: str, index: int) -> int:
    import torch

    from research.t12_b8_serving_20261017 import episodes, measurements, parity

    torch.set_num_threads(1)
    intent = verify(root)
    path = result_path(root, kind, index)
    if path.exists():
        raise RuntimeError(f"result exists; never overwritten: {path}")
    purpose = "rollback" if kind == "defaults" else kind
    seeds = intent["banks"][purpose]["seeds"]
    one = seeds[index] if kind in ("watch", "play", "parity", "confidence", "controls") else seeds
    envelope = {
        "schema": "t12-serving-result/v1",
        "intent_sha256": sha(root / "intent.json"),
        "kind": kind,
        "index": index,
        "seed_or_seeds": one,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "started_utc": utc(),
        "status": "error",
        "error": None,
    }
    started = time.monotonic()
    horizon = 100 if intent["smoke"] else 5000
    try:
        guard()
        if kind == "watch":
            data = episodes.run_watch(one, horizon)
        elif kind == "play":
            data = episodes.run_play(one, horizon)
        elif kind == "parity":
            data = parity.run_session_parity(one, horizon, ("frozen", "mixed")[index])
        elif kind.startswith("p2_"):
            data = parity.run_p2(seeds, 100 if intent["smoke"] else int(kind[4:]))
        elif kind == "controls":
            data = episodes.run_controls(one)
        elif kind == "confidence":
            data = measurements.run_confidence(one, 100 if intent["smoke"] else 1500)
        elif kind == "latency":
            data = measurements.run_latency(seeds)
        elif kind == "defaults":
            data = defaults(seeds)
        else:
            raise ValueError(kind)
        envelope.update(status="complete", data=data)
    except Exception as exc:
        envelope.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    envelope["wall_seconds"] = time.monotonic() - started
    write_new(path, envelope)
    print(
        json.dumps(
            {
                "result": str(path),
                "status": envelope["status"],
                "seconds": envelope["wall_seconds"],
                "error": envelope["error"],
            }
        ),
        flush=True,
    )
    return 0 if envelope["status"] == "complete" else 1


def finalize(root: Path) -> None:
    intent = verify(root)
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.name not in ("receipt.json", "audit.json")
    )
    write_new(
        root / "receipt.json",
        {
            "schema": "t12-serving-receipt/v1",
            "created_utc": utc(),
            "intent_sha256": sha(root / "intent.json"),
            "smoke": intent["smoke"],
            "files": {str(p.relative_to(root)): sha(p) for p in paths},
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "unit", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--kind",
        choices=(
            "watch",
            "play",
            "parity",
            "p2_h5000",
            "p2_h10000",
            "controls",
            "confidence",
            "latency",
            "defaults",
        ),
    )
    parser.add_argument("--indices", default="0", help="comma-separated unit indices")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.validation, args.smoke)
    elif args.action == "finalize":
        finalize(args.root)
    else:
        for index in map(int, args.indices.split(",")):
            code = unit(args.root, args.kind, index)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
