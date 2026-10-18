#!/usr/bin/env python3
"""Create and audit the immutable T23 H5000 evaluation world bank.

This module deliberately allocates worlds only.  It does not import a learner,
load a checkpoint, or play an episode.  Roster construction is delegated to
the released B8 strict-design materializer, while allocation has a wholly new
T23 namespace so it cannot be mistaken for the T7 producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research.apex_safety_20260926 import dev_screen

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
OUTPUT = Path(__file__).with_name("WORLD_BANK.json")
AUDIT_OUTPUT = Path(__file__).with_name("SEED_AUDIT.json")
NAMESPACE = "t23-r0-h5000-terminal-hero/v2"
TRAINING_SEEDS = (7101, 7102, 7103, 7104, 7105)
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
PHASES = ("initial", "extension")
PARITY_COUNT = 4
SCHEMA = "t23-r0-world-bank/v1"
PROFILE_ID = "t23-r0-terminal-h5000/v1"


def canonical_sha(value: Any) -> str:
    """Return a portable, canonical JSON SHA-256 binding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seeds(training_seed: int, phase: str, mix: str, count: int) -> list[int]:
    key = f"seed-{training_seed}/{phase}/{mix}"
    return dev_screen.screen_seeds(count, NAMESPACE, key)


def _slot(slot: dict[str, Any]) -> dict[str, Any]:
    """Convert the shared materializer row into a portable evaluator spec."""
    member = str(slot["member_sha256"])
    # Scripted identity hashes are intentionally retained as the materializer
    # emits them; evaluator resolves them via the strict design's lookup.
    return {
        "slot": int(slot["slot"]),
        "kind": "member",
        "member_sha256": member,
        "eligible_member_sha256s": [str(x) for x in slot["eligible_member_sha256s"]],
    }


def _worlds(training_seed: int, phase: str) -> list[dict[str, Any]]:
    total = sum(MIX_COUNTS.values())
    worlds: list[dict[str, Any]] = []
    for mix, count in MIX_COUNTS.items():
        # The strict materializer creates each mix for every supplied seed.
        # Allocate per-mix namespaces so a terminal episode cannot appear in
        # another mix, phase, or seed's decision sample.
        rows = [
            row
            for row in dev_screen._design_rows(_seeds(training_seed, phase, mix, count))
            if row["mix"] == mix
        ]
        if len(rows) != count:
            raise RuntimeError(f"strict materializer did not produce {count} {mix} rows")
        for row in rows:
            slots = [_slot(slot) for slot in row["slots"]]
            roster = [slot["member_sha256"] for slot in slots]
            semantic = {
                "seed_namespace": NAMESPACE,
                "training_seed": training_seed,
                "phase": phase,
                "mix": mix,
                "world_seed": int(row["world_seed"]),
                "profile_id": PROFILE_ID,
                "horizon": 5000,
                "terminal_hero_slot": 0,
                "opponent_slots": slots,
            }
            worlds.append(
                {
                    **semantic,
                    "opponents": [["checkpoint", sha] for sha in roster],
                    "roster_semantic_sha256": canonical_sha({"mix": mix, "slots": slots}),
                    "initial_world_sha256": canonical_sha(
                        {
                            "world_seed": int(row["world_seed"]),
                            "mix": mix,
                            "profile_id": PROFILE_ID,
                            "horizon": 5000,
                            "opponent_slots": slots,
                        }
                    ),
                    "world_semantic_sha256": canonical_sha(semantic),
                }
            )
    if len(worlds) != total or len({row["world_seed"] for row in worlds}) != total:
        raise RuntimeError("T23 initial/extension mix allocation reused a world seed")
    return worlds


def _parity_worlds() -> list[dict[str, Any]]:
    # Reserved separately from decision worlds: a compact profile-cross-section
    # exposes anchor and frozen-forward parity without spending selection rows.
    requested = ("scripted", "scripted", "frozen", "mixed")
    rows = []
    for index, mix in enumerate(requested):
        seed = dev_screen.screen_seeds(1, NAMESPACE, f"mac-colab-parity/{index}/{mix}")[0]
        rows.append(next(row for row in dev_screen._design_rows([seed]) if row["mix"] == mix))
    worlds = []
    for row in rows:
        slots = [_slot(slot) for slot in row["slots"]]
        semantic = {
            "seed_namespace": NAMESPACE,
            "training_seed": None,
            "phase": "parity",
            "mix": row["mix"],
            "world_seed": int(row["world_seed"]),
            "profile_id": PROFILE_ID,
            "horizon": 5000,
            "terminal_hero_slot": 0,
            "opponent_slots": slots,
        }
        worlds.append(
            {
                **semantic,
                "opponents": [["checkpoint", slot["member_sha256"]] for slot in slots],
                "roster_semantic_sha256": canonical_sha({"mix": row["mix"], "slots": slots}),
                "initial_world_sha256": canonical_sha(
                    {
                        "world_seed": int(row["world_seed"]),
                        "mix": row["mix"],
                        "profile_id": PROFILE_ID,
                        "horizon": 5000,
                        "opponent_slots": slots,
                    }
                ),
                "world_semantic_sha256": canonical_sha(semantic),
            }
        )
    return worlds


def build_world_bank() -> dict[str, Any]:
    """Return the full deterministic initial, extension, and parity allocation."""
    allocation = {
        str(seed): {phase: _worlds(seed, phase) for phase in PHASES} for seed in TRAINING_SEEDS
    }
    all_seeds = [
        w["world_seed"] for phases in allocation.values() for rows in phases.values() for w in rows
    ]
    parity = _parity_worlds()
    if len(set(all_seeds + [w["world_seed"] for w in parity])) != len(all_seeds) + len(parity):
        raise RuntimeError("T23 world namespace collision")
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "profile": {
            "profile_id": PROFILE_ID,
            "base_profile_id": dev_screen.PROFILE_NAME,
            "base_profile_digest": dev_screen.PROFILE_DIGEST,
            "horizon": 5000,
            "engine": "grid",
            "sim_engine": "grid",
            "hero_mode": "terminal",
            "hero_ego2s": True,
            "vector61_forward": "rowwise",
            "ego2s_forward": "rowwise",
            "ego2s_mask": "advisory",
            "veto": False,
            "vector61_backend": "numba",
            "scripted_anchors": "fast",
            "source": "research.apex_safety_20260926.dev_screen._design_rows",
        },
        "mix_counts": MIX_COUNTS,
        "training_seeds": list(TRAINING_SEEDS),
        "worlds": allocation,
        "parity_worlds": parity,
        "source_bindings": {
            "strict_design": {
                "path": "research/apex_safety_20260926/dev_screen.py",
                "sha256": sha256_file(ROOT / "research/apex_safety_20260926/dev_screen.py"),
            },
            "t23_bank_producer": {
                "path": "research/t23_r0_screen_20261017/bank.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "profile_semantic_sha256": canonical_sha(
            {
                "profile_id": PROFILE_ID,
                "base_profile_id": dev_screen.PROFILE_NAME,
                "base_profile_digest": dev_screen.PROFILE_DIGEST,
                "horizon": 5000,
                "sim_engine": "grid",
                "hero_ego2s": True,
                "ego2s_forward": "rowwise",
                "ego2s_mask": "advisory",
                "veto": False,
                "vector61_backend": "numba",
                "scripted_anchors": "fast",
                "hero_mode": "terminal",
            }
        ),
        "allocation_semantic_sha256": canonical_sha(
            {"worlds": allocation, "parity_worlds": parity}
        ),
    }


def worlds_for_seed(
    bank: dict[str, Any], training_seed: int, bank_phase: str = "initial"
) -> list[dict[str, Any]]:
    """Return one training seed's fixed worlds, validating the public bank shape."""
    if bank.get("schema") != SCHEMA or bank_phase not in PHASES:
        raise ValueError("invalid T23 world bank or phase")
    rows = bank["worlds"].get(str(int(training_seed)), {}).get(bank_phase)
    if not isinstance(rows, list) or len(rows) != sum(MIX_COUNTS.values()):
        raise ValueError("T23 bank has missing or malformed training-seed rows")
    return rows


def load_world_bank(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the checked-in bank manifest for evaluator consumers."""
    bank = json.loads((OUTPUT if path is None else Path(path)).read_text())
    if bank.get("schema") != SCHEMA or bank.get("allocation_semantic_sha256") != canonical_sha(
        {"worlds": bank.get("worlds"), "parity_worlds": bank.get("parity_worlds")}
    ):
        raise ValueError("T23 world bank semantic digest mismatch")
    for seed in TRAINING_SEEDS:
        worlds_for_seed(bank, seed)
    if len(bank.get("parity_worlds", [])) != PARITY_COUNT:
        raise ValueError("T23 bank lacks its four parity worlds")
    return bank


def _scan_candidate_tokens(roots: Iterable[Path], own: Path, candidate: set[int]) -> set[int]:
    """Return candidate decimal identities visible in JSON/JSONL receipts."""
    expression = r"\b(?:" + "|".join(str(value) for value in sorted(candidate)) + r")\b"
    found: set[int] = set()
    for root in roots:
        if not root.exists():
            continue
        exclude: list[str] = []
        try:
            exclude = ["--glob", "!" + own.relative_to(root).as_posix() + "/**"]
        except ValueError:
            pass
        completed = subprocess.run(
            [
                "rg",
                "--no-ignore",
                "--no-filename",
                "--no-heading",
                "--color",
                "never",
                "--glob",
                "*.json",
                "--glob",
                "*.jsonl",
                *exclude,
                "--only-matching",
                expression,
                ".",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"strict-bank collision scan failed: {completed.stderr.strip()}")
        found.update(int(value) for value in completed.stdout.splitlines() if value.isdigit())
    return found


def _prior_strict_seed_inventory(
    repo: Path, artifacts: Path, candidate: set[int]
) -> tuple[dict[str, Any], set[int]]:
    """Bind existing strict-bank receipts and extract every visible world identity."""
    mandatory = {
        "t7_seed_audit": repo / "research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json",
        "t7_plan": repo / "research/redesign_t7_phase_r_20261016/results/plan.json",
        "frp3_strict": repo / "docs/research/frp3_strict_result_2026-10-06.md",
        "b8_release": repo / "docs/research/b8_s3_release_2026-10-17.md",
        "artifact_root": artifacts,
        # The T7 audit's strict historical boundary.  Keep these explicit here
        # rather than treating a nonempty artifact root as evidence.
        "t3_namespace": artifacts / "t3-rollout-labels-20261015/bank/namespace.json",
        "t3_manifest": artifacts / "t3-rollout-labels-20261015/bank/manifest.json",
        "t4f_manifest_train": artifacts / "t4f-return-bank-v3-20261016/banks/train/manifest.json",
        "t4f_manifest_holdout": artifacts
        / "t4f-return-bank-v3-20261016/banks/holdout/manifest.json",
        "m3_phase_r": artifacts / "redesign-m3-phase-r-20261008/plan.json",
        "m4_phase_r": artifacts / "redesign-m4-phase-r-nv-20261013/plan.json",
        "t4e_train": artifacts / "t4e-return-bank-20261016/banks/train/manifest.json",
        "t4e_holdout": artifacts / "t4e-return-bank-20261016/banks/holdout/manifest.json",
        "t4_snapshot": artifacts / "t4b-snapshot-restart-20261016/full/manifest.json",
        "t4c_bank": artifacts / "t4c-data-20261016/bank-v3/manifest.json",
        "t4c_outside": artifacts / "t4c-data-20261016/outside-bank/manifest.json",
    }
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise FileNotFoundError("historical FRP strict exclusion inventories incomplete")
    mandatory.update({f"frp_exclusion_{index}": path for index, path in enumerate(exclusions)})
    missing = [name for name, path in mandatory.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"mandatory strict-bank evidence missing: {missing}")
    values: set[int] = set()
    # T7's authoritative audit exposes the generated banks directly.
    t7 = json.loads(mandatory["t7_seed_audit"].read_text())
    if not t7.get("pass"):
        raise ValueError("T7 seed audit is not a passing strict-bank receipt")
    for entry in t7.get("banks", {}).values():
        values.update(int(x) for x in entry.get("seeds", []))
    # T7's generator includes the registered historical strict namespaces,
    # manifests, and FRP exclusion inventories; reuse its fail-closed collector
    # rather than pretending the visible world_seed fields cover binary banks.
    from research.redesign_t7_phase_r_20261016 import seed_audit as t7_audit

    values.update(t7_audit._registered(repo))
    # Include visible world_seed receipts in research and artifact JSON/JSONL.
    # This is an audit boundary, not a claim about opaque binary payloads.
    own = repo / "research/t23_r0_screen_20261017"
    # Use ripgrep's fixed-word matching so this remains bounded even when the
    # artifact root contains large opaque research inventories.  Candidate
    # decimal identities are all we need to discover a collision.
    values.update(_scan_candidate_tokens((repo / "research", artifacts), own, candidate))
    binding = {
        name: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
        for name, path in mandatory.items()
    }
    return binding, values


def audit_world_bank(repo: Path = ROOT, artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    """Fail closed if T23 collides with visible strict-bank world identities."""
    bank = build_world_bank()
    candidate = {
        row["world_seed"]
        for seed in TRAINING_SEEDS
        for phase in PHASES
        for row in bank["worlds"][str(seed)][phase]
    } | {row["world_seed"] for row in bank["parity_worlds"]}
    bindings, registered = _prior_strict_seed_inventory(Path(repo), Path(artifacts), candidate)
    internal = len(candidate) == 5 * 2 * sum(MIX_COUNTS.values()) + PARITY_COUNT
    collisions = sorted(candidate & registered)
    return {
        "schema": "t23-r0-seed-audit/v1",
        "namespace": NAMESPACE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "candidate_count": len(candidate),
        "registered_strict_count": len(registered),
        "registered_collision_seeds": collisions,
        "internal_unique": internal,
        "mandatory_evidence": bindings,
        "limitations": [
            "Visible JSON strict-bank receipts plus bound T7 allocation; "
            "opaque binaries are outside this audit."
        ],
        "pass": internal and not collisions,
    }


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    bank = build_world_bank()
    audit = audit_world_bank(args.repo, args.artifacts)
    if not audit["pass"]:
        raise SystemExit(f"T23 seed audit failed: {audit['registered_collision_seeds']}")
    _write(args.out, bank)
    audit["world_bank_file"] = {"path": str(args.out), "sha256": sha256_file(args.out)}
    _write(args.audit_out, audit)
    print(
        json.dumps(
            {
                "world_bank": str(args.out),
                "audit": str(args.audit_out),
                "worlds": audit["candidate_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
