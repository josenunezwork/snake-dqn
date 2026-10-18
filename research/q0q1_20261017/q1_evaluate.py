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
from research.q0q1_20261017 import q1_parity
from research.t23_r0_screen_20261017 import evaluate as t23

SCHEMA = "q1-eval-shard/v1"
ROW_SCHEMA = "q1-eval-row/v1"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
ARMS = ("A", "B", "C", "D", "B8")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "research/q0q1_20261017/q1_bank.py",
        "research/q0q1_20261017/q1_evaluate.py",
        "research/t23_r0_screen_20261017/evaluate.py",
        "research/apex_safety_20260926/dev_screen.py",
        "research/redesign_scope_20261007/grid_h5000_identity.py",
    )
    return {name: sha256(root / name) for name in names}


def validate_provenance(value: dict[str, Any], arm: str, seed: int, checkpoint_sha256: str) -> None:
    required = {
        "schema": "q1-training-provenance/v1",
        "arm": arm,
        "training_seed": seed,
        "endpoint": 8_000_000,
        "clean_exit": True,
        "checkpoint_sha256": checkpoint_sha256,
        "parity_pass": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("candidate provenance lacks Q1 arm/seed/8M/clean/parity binding")
    if (
        value.get("tripwire_triggered") is True
        or not isinstance(value.get("environment_sha256"), str)
        or not isinstance(value.get("runtime_parity_receipt_sha256"), str)
    ):
        raise ValueError(
            "candidate provenance has tripwire or missing environment/runtime parity binding"
        )


def validate_candidate_provenance(
    path: Path, arm: str, seed: int, checkpoint: Path
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    validate_provenance(value, arm, seed, sha256(checkpoint))
    return value


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
    provenance_path: Path | None = None,
    parity_path: Path | None = None,
) -> Path:
    if arm not in ARMS or seed not in q1_bank.TRAINING_SEEDS:
        raise ValueError("unregistered Q1 arm or seed")
    index, count = _split(shard)
    bank = q1_bank.load_world_bank(bank_path)
    worlds = shard_worlds(q1_bank.worlds_for_seed(bank, seed), index, count)
    context = t23._context(max(1, workers))
    profile = str(context["profile"].digest)
    if bank["profile"].get("base_profile_digest") != profile:
        raise ValueError("Q1 bank profile differs from active evaluation profile")
    audit_path = bank_path.with_name("Q1_SEED_AUDIT.json")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != bank["allocation_semantic_sha256"]
        or audit.get("world_bank_file", {}).get("sha256") != sha256(bank_path)
    ):
        raise ValueError("Q1 collision audit is missing or does not bind the supplied bank")
    q1_bank_digest = hashlib.sha256(canonical(bank).encode()).hexdigest()
    checkpoint_sha = sha256(checkpoint)
    b8_sha = sha256(b8)
    if b8_sha != B8_SHA256:
        raise ValueError("B8 checkpoint does not match the released B8 SHA")
    if arm == "B8" and checkpoint_sha != b8_sha:
        raise ValueError("B8 identity must use the pinned B8 checkpoint bytes")
    if parity_path is None:
        raise ValueError("every Q1 evaluation identity requires --parity-receipt")
    parity = json.loads(parity_path.read_text())
    q1_parity.validate_runtime_receipt(
        parity, bank_sha256=sha256(bank_path), b8_sha256=b8_sha, source_manifest=source_manifest()
    )
    provenance = (
        None
        if arm == "B8"
        else validate_candidate_provenance(Path(provenance_path or ""), arm, seed, checkpoint)
    )
    identity = {
        "training_seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha,
        "b8_sha256": b8_sha,
        "endpoint": 8_000_000,
        "shard": index,
        "shards": count,
        "world_count": len(worlds),
        "world_hashes": [x["world_semantic_sha256"] for x in worlds],
        "bank_sha256": q1_bank_digest,
        "profile_digest": profile,
        "evaluator_source_sha256": sha256(Path(__file__)),
        "evaluator_source_manifest": source_manifest(),
        "runtime_parity_receipt_sha256": sha256(parity_path),
        "environment_sha256": parity["environment"]["sha256"],
        "training_provenance_sha256": None if provenance is None else sha256(provenance_path),
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
            "runtime_parity": parity,
            "training_provenance": provenance,
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
    p.add_argument("--provenance", type=Path)
    p.add_argument("--parity-receipt", type=Path, required=True)
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
            provenance_path=a.provenance,
            parity_path=a.parity_receipt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
