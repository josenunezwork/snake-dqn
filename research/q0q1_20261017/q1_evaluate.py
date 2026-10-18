"""Q1 CPU H5000 evaluator: one immutable bank shared by A--D and B8.

Each shard owns one identity (``A`` through ``D`` or ``B8``).  Consequently a
completed B8 shard is reusable for every arm only when the exact bank and
identity receipt match; analysis refuses duplicate or substituted B8 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Sequence

from research.q0q1_20261017 import q1_bank, q1_parity
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
    """Pin the transitive simulator/model execution plus Q1-owned adapters."""
    root = Path(__file__).resolve().parents[2]
    result = t23.source_manifest()
    for name in ("q1_bank.py", "q1_evaluate.py", "q1_parity.py"):
        relative = "research/q0q1_20261017/" + name
        result[relative] = sha256(root / relative)
    return result


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
    hashes = ("checkpoint_sha256", "environment_sha256", "runtime_parity_receipt_sha256")
    if value.get("tripwire_triggered") is not False or any(
        not isinstance(value.get(key), str)
        or len(value[key]) != 64
        or any(c not in "0123456789abcdef" for c in value[key])
        for key in hashes
    ):
        raise ValueError(
            "candidate provenance has tripwire or missing environment/runtime parity binding"
        )
    counters = value.get("counters")
    if not isinstance(counters, dict):
        raise ValueError("candidate provenance lacks actual counter receipt")
    decisions = counters.get("actual_learner_decisions")
    overshoot = counters.get("transition_budget_overshoot")
    bound = counters.get("registered_overshoot_bound")
    if (
        type(decisions) is not int
        or type(overshoot) is not int
        or type(bound) is not int
        or not 8_000_000 <= decisions <= 8_000_000 + bound
        or overshoot != decisions - 8_000_000
        or not 0 <= bound <= 1_000_000
    ):
        raise ValueError("candidate actual learner counter or overshoot bound is invalid")
    actors, worlds, seats = (counters.get(k) for k in ("actors", "worlds", "learner_seats"))
    if (
        any(type(v) is not int or v <= 0 for v in (actors, worlds, seats))
        or worlds != 128
        or seats != (3 if arm in ("A", "B") else 1)
        or bound != actors * worlds * seats - 1
    ):
        raise ValueError("candidate overshoot bound differs from registered actor packet")
    accounting_keys = (
        "emitted_replay_rows",
        "completed_updates",
        "pending_update_debt",
        "truncation_rows_dropped",
        "pending_nstep_rows",
    )
    if any(type(counters.get(k)) is not int or counters[k] < 0 for k in accounting_keys):
        raise ValueError("candidate replay/update/tail counters are missing or invalid")
    emitted, updates, debt, dropped, pending = (counters[k] for k in accounting_keys)
    if (
        updates == 0
        or debt >= 256
        or pending > actors * worlds * seats * 5
        or emitted != 256 * updates + debt
        or decisions - emitted != dropped + pending
    ):
        raise ValueError("candidate decision/replay/update/tail accounting does not reconcile")
    for name in ("environment", "config"):
        content = value.get(name)
        if (
            not isinstance(content, dict)
            or value.get(name + "_sha256")
            != hashlib.sha256(canonical(content).encode()).hexdigest()
        ):
            raise ValueError(f"candidate {name} hash does not bind recorded content")
    root = Path(__file__).resolve().parents[2]
    if value.get("source_sha256") != sha256(root / "research/redesign_m3c_20261008/train_m3c.py"):
        raise ValueError("candidate trainer differs from registered source")
    config = value["config"]
    required_config = {
        "seed": str(seed),
        "transitions": "8000000",
        "worlds": "128",
        "learner_seats": str(seats),
        "actors": str(actors),
        "reward": "mass",
        "kill_scale": "0.0",
        "batch_scale": "2",
        "obs_version": "bv8",
        "anchor_mode": "v8margin",
        "namespace": q1_bank.training_namespace(arm),
        "schedule_package": "mature" if arm in ("B", "D") else "restart",
        "collection_profile": "legacy" if arm in ("A", "B") else "task-aligned-n1",
    }
    if any(str(config.get(k)) != expected for k, expected in required_config.items()):
        raise ValueError("candidate recipe differs from registered arm")


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
    if workers != 1:
        raise ValueError("Q1 uses one CPU thread per world; parallelize immutable shards")
    context = t23._context(1)
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
        parity, bank_sha256=q1_bank_digest, b8_sha256=b8_sha, source_manifest=source_manifest()
    )
    if parity["runtime_artifact"]["runtime"] != q1_parity.runtime_environment():
        raise ValueError("parity receipt belongs to a different evaluation runtime")
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
        "runtime_parity_receipt_sha256": hashlib.sha256(canonical(parity).encode()).hexdigest(),
        "environment_sha256": parity["environment"]["sha256"],
        "training_provenance_sha256": (
            None
            if provenance is None
            else hashlib.sha256(canonical(provenance).encode()).hexdigest()
        ),
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
