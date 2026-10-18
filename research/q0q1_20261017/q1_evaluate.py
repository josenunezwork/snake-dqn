"""Q1 CPU H5000 evaluator: one immutable bank shared by A--D and B8.

Each shard owns one identity (``A`` through ``D`` or ``B8``).  Consequently a
completed B8 shard is reusable for every arm only when the exact bank and
identity receipt match; analysis refuses duplicate or substituted B8 rows.
"""

from __future__ import annotations
import argparse, hashlib, json, os, platform
from pathlib import Path
from typing import Any, Sequence
from research.q0q1_20261017 import q1_bank
from research.t23_r0_screen_20261017 import evaluate as t23

SCHEMA = "q1-eval-shard/v1"
ROW_SCHEMA = "q1-eval-row/v1"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
ARMS = ("A", "B", "C", "D", "B8")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _split(text: str) -> tuple[int, int]:
    a, b = text.split("/", 1)
    index, count = int(a), int(b)
    if not 0 <= index < count:
        raise ValueError("shard index is outside shard count")
    return index, count


def shard_worlds(worlds: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    return [
        x
        for i, x in enumerate(sorted(worlds, key=lambda x: (x["mix"], x["world_seed"])))
        if i % count == index
    ]


def receipt_path(root: Path, seed: int, arm: str, index: int, count: int) -> Path:
    return root / "evaluation" / f"seed-{seed}" / arm / f"shard-{index}-of-{count}.json"


def valid_completed_shard(path: Path, identity: dict[str, Any]) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    rows = value.get("rows")
    return (
        value.get("schema") == SCHEMA
        and value.get("status") == "completed"
        and value.get("clean_exit") is True
        and value.get("identity") == identity
        and isinstance(rows, list)
        and len(rows) == identity["world_count"]
        and value.get("rows_sha256") == hashlib.sha256(canonical(rows).encode()).hexdigest()
        and all(x.get("schema") == ROW_SCHEMA for x in rows)
    )


def run(
    *,
    root: Path,
    bank_path: Path,
    b8: Path,
    checkpoint: Path,
    arm: str,
    seed: int,
    shard: str,
    workers: int = 1,
) -> Path:
    if arm not in ARMS or seed not in q1_bank.TRAINING_SEEDS:
        raise ValueError("unregistered Q1 arm or seed")
    index, count = _split(shard)
    bank = q1_bank.load_world_bank(bank_path)
    worlds = shard_worlds(q1_bank.worlds_for_seed(bank, seed), index, count)
    context = t23._context(max(1, workers))
    profile = str(context["profile"].digest)
    q1_bank_digest = hashlib.sha256(canonical(bank).encode()).hexdigest()
    checkpoint_sha = sha256(checkpoint)
    if arm == "B8" and checkpoint_sha != sha256(b8):
        raise ValueError("B8 identity must use the pinned B8 checkpoint bytes")
    identity = {
        "training_seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha,
        "b8_sha256": sha256(b8),
        "endpoint": 8_000_000,
        "shard": index,
        "shards": count,
        "world_count": len(worlds),
        "world_hashes": [x["world_semantic_sha256"] for x in worlds],
        "bank_sha256": q1_bank_digest,
        "profile_digest": profile,
        "evaluator_source_sha256": sha256(Path(__file__)),
    }
    path = receipt_path(root, seed, arm, index, count)
    if valid_completed_shard(path, identity):
        return path
    if path.exists():
        raise RuntimeError("refusing to replace a Q1 shard with changed identity")
    carrier = context["lookup"][
        __import__(
            "research.redesign_scope_20261007.grid_h5000_identity", fromlist=["FRP3_S12"]
        ).FRP3_S12[1]
    ]
    rows = []
    for world in worlds:
        value = t23.evaluate_world(world, checkpoint, carrier, context)
        value.update(
            {
                "schema": ROW_SCHEMA,
                "arm": arm,
                "training_seed": seed,
                "checkpoint_step": 8_000_000,
                "checkpoint_sha256": checkpoint_sha,
                "b8_checkpoint_sha256": identity["b8_sha256"],
                "world_bank_sha256": q1_bank_digest,
                "world_profile_sha256": profile,
                "initial_world_sha256": world["initial_world_sha256"],
                "evaluator_source_sha256": identity["evaluator_source_sha256"],
                "clean_exit": True,
            }
        )
        rows.append(value)
    _write(
        path,
        {
            "schema": SCHEMA,
            "status": "completed",
            "clean_exit": True,
            "identity": identity,
            "rows": rows,
            "rows_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "engine": {"cpu_only": True, "horizon": 5000, "terminal_hero": True},
            "platform": platform.platform(),
        },
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=Path(__file__).with_name("Q1_WORLD_BANK.json"))
    p.add_argument("--b8-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    print(
        run(
            root=a.root,
            bank_path=a.bank,
            b8=a.b8_checkpoint,
            checkpoint=a.checkpoint,
            arm=a.arm,
            seed=a.seed,
            shard=a.shard,
            workers=a.workers,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
