#!/usr/bin/env python3
"""Q0a B8 initializer fidelity audit; CPU-only and observation-capped.

The saved-artifact phase validates bytes and T23's embedded initializer binding.
It remains INCOMPLETE until the bounded 4,096-observation corpus is captured
and verified. Wrong wrappers, fallback masks, and identity mismatches fail closed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.q0q1_20261017.audit_common import (  # noqa: E402
    B8_SHA256,
    T23_SEEDS,
    load_json,
    sha256_file,
    write_json,
)

SCHEMA = "q0-b8-init-fidelity/v1"
OBSERVATION_CAP = 4096
REQUIRED_COVERAGE = (
    "short_lt700",
    "long_ge800",
    "crowded",
    "open",
    "exact_q_tie",
    "boost_eligible",
    "scripted",
    "frozen",
    "mixed",
)


def checkpoint_inventory(t23_root: Path) -> dict[str, Any]:
    from research.q0q1_20261017.audit_common import checkpoint_inventory as inventory

    return {
        seed: {
            "available": any(row["valid"] for row in entries["final"]),
            "finals": entries["final"],
            "invalid": entries["invalid"],
        }
        for seed, entries in inventory(t23_root).items()
    }


def initializer_binding(t23_root: Path) -> dict[str, Any]:
    """Require the actual checkpoint metadata and committed final digest."""
    inventory = checkpoint_inventory(t23_root)
    return {
        seed: {
            "matches_released_b8": bool(item["available"]),
            "source": "final checkpoint embedded meta plus committed T23 result pin",
            "evidence": item["finals"],
        }
        for seed, item in inventory.items()
    }


def build_report(b8: Path, t23_root: Path) -> dict[str, Any]:
    available = b8.is_file()
    digest = sha256_file(b8) if available else None
    return {
        "schema": SCHEMA,
        "authority": "Q0 initialization fidelity; no efficacy decision",
        "status": "INCOMPLETE",
        "b8": {
            "path": str(b8),
            "available": available,
            "sha256": digest,
            "expected_sha256": B8_SHA256,
            "sha256_matches_release": digest == B8_SHA256 if digest else False,
        },
        "t23_final_inventory": checkpoint_inventory(t23_root),
        "t23_initializer_bindings": initializer_binding(t23_root),
        "fresh_observation_packet": {
            "status": "NOT_RUN",
            "observation_cap": OBSERVATION_CAP,
            "required_coverage": list(REQUIRED_COVERAGE),
            "required_checks": [
                "zero-update action identity",
                "wrong-wrapper rejection",
                "mask disagreement rejection",
                "fallback-mask rejection",
                "frozen-seat replay contamination rejection",
                "frozen-seat gradient and optimizer-state rejection",
            ],
            "reason": "requires the collision-audited fresh Q0 corpus and pinned evaluator adapter",
        },
    }


def prerequisites_pass(report: dict[str, Any]) -> bool:
    """All released bytes and five T23 final/init bindings are mandatory."""
    if not report["b8"]["sha256_matches_release"]:
        return False
    inventory = report["t23_final_inventory"]
    bindings = report["t23_initializer_bindings"]
    return all(
        inventory[str(seed)]["available"] and bindings[str(seed)]["matches_released_b8"]
        for seed in T23_SEEDS
    )


def assess_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a bounded captured corpus; all required support is mandatory."""
    if len(rows) > OBSERVATION_CAP:
        raise ValueError("Q0 fidelity observation cap exceeded")
    coverage = {name: 0 for name in REQUIRED_COVERAGE}
    disagreements = 0
    for row in rows:
        if row.get("fixture") and row.get("fixture") != "exact-q-tie-first-legal":
            raise ValueError("unknown Q0 fidelity fixture")
        for name in (
            "short_lt700",
            "long_ge800",
            "crowded",
            "open",
            "exact_q_tie",
            "boost_eligible",
        ):
            coverage[name] += int(bool(row.get(name)))
        coverage[str(row.get("mix"))] = coverage.get(str(row.get("mix")), 0) + 1
        disagreements += int(row.get("action_a") != row.get("action_b"))
        if (
            row.get("fallback_used")
            or row.get("mask_disagreement")
            or row.get("wrapper_disagreement")
        ):
            raise ValueError("Q0 fidelity fail-closed wrapper/mask/fallback failure")
    missing = [name for name in REQUIRED_COVERAGE if not coverage.get(name)]
    return {
        "status": (
            "COMPLETE"
            if len(rows) == OBSERVATION_CAP and not missing and not disagreements
            else "INCOMPLETE"
        ),
        "observations": len(rows),
        "coverage": coverage,
        "action_disagreements": disagreements,
        "missing_coverage": missing,
    }


class CoverageReservoir:
    """Deterministic bounded reservoir with protected first witnesses per stratum.

    All candidate rows are validated; the cap applies to retained observations,
    not to stopping every world before its long-body phase can be observed.
    """

    def __init__(self, capacity: int = OBSERVATION_CAP - 1) -> None:
        import random

        self.capacity = capacity
        self.rows: list[dict[str, Any]] = []
        self.protected: set[int] = set()
        self.covered: set[str] = set()
        self.seen = 0
        self.excluded_trapped = 0
        self.rng = random.Random(20261017)

    def add(self, row: dict[str, Any]) -> None:
        if any(
            row.get(key) for key in ("fallback_used", "mask_disagreement", "wrapper_disagreement")
        ):
            raise ValueError("Q0 fidelity fail-closed candidate wrapper/mask/fallback failure")
        if row["action_a"] != row["action_b"]:
            raise ValueError("Q0 zero-update candidate action disagreement")
        self.seen += 1
        support = {key for key in REQUIRED_COVERAGE if row.get(key)} | {row["mix"]}
        novel = support - self.covered
        if len(self.rows) < self.capacity:
            index = len(self.rows)
            self.rows.append(row)
        else:
            available = [i for i in range(self.capacity) if i not in self.protected]
            if not available:
                raise ValueError("reservoir capacity insufficient for coverage witnesses")
            if novel:
                index = self.rng.choice(available)
            else:
                draw = self.rng.randrange(self.seen)
                if draw >= self.capacity or draw in self.protected:
                    return
                index = draw
            self.rows[index] = row
        if novel:
            self.protected.add(index)
            self.covered |= support


def exact_tie_fixture(*, fallback: bool = False) -> dict[str, Any]:
    """Execute evaluator selection and the independent trainer Torch expression."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np
    import torch

    from src.simd_env import ego2s_policy

    q = torch.zeros((1, 6), dtype=torch.float32)
    mask = np.array([[False, True, True, False, True, False]])
    if fallback:
        mask[:] = False
        q[:, 3:] = 10.0
    policy = object.__new__(ego2s_policy.Ego2sSimdPolicy)
    policy.net, policy.forward, policy.mask_mode = None, "rowwise", "advisory"
    sim = SimpleNamespace(get_advisory_action_mask=lambda: mask[:, None, :])
    with patch.object(ego2s_policy, "student_q", return_value=q):
        action_a = policy.actions(mask, sim, np.array([[0, 0]]))[0]
    # This is the trainer's torch.where(...,-1e9).argmax operation, independent
    # of evaluator float64/numpy selection and including a masked first action.
    trainer_mask = mask.copy()
    if not trainer_mask.any():
        trainer_mask[:, :3] = True
    action_b = torch.where(torch.from_numpy(trainer_mask), q, torch.full_like(q, -1e9)).argmax(1)[0]
    return {
        "mix": "scripted",
        "exact_q_tie": True,
        "action_a": int(action_a),
        "action_b": int(action_b),
        "fixture": "normal-only-fallback" if fallback else "exact-q-tie-first-legal",
        "fallback_used": False,
        "mask_disagreement": False,
        "wrapper_disagreement": False,
    }


def run_observations(b8: Path, observation_receipt: Path) -> dict[str, Any]:
    """Validate corpus retained during Q0 mass-audit B8 worlds; run no extra worlds."""
    if not b8.is_file() or sha256_file(b8) != B8_SHA256:
        raise ValueError("Q0 observation packet requires released B8 bytes")
    packet = load_json(observation_receipt)
    if not packet or packet.get("b8", {}).get("sha256") != B8_SHA256:
        raise ValueError("Q0 observation receipt lacks released B8 binding")
    corpus = packet.get("development_packet", {}).get("fidelity_corpus", {})
    rows = corpus.get("rows", [])
    if not rows or corpus.get("bank_audit_pass") is not True:
        raise ValueError("Q0 mass audit has no collision-audited fidelity corpus")
    outcome = assess_observations(rows)
    fallback = exact_tie_fixture(fallback=True)
    if fallback["action_a"] != fallback["action_b"] or fallback["action_a"] != 0:
        raise ValueError("legitimate normal-only fallback selector disagreement")
    outcome["normal_only_fallback_oracle"] = fallback
    outcome.update(
        source_receipt_sha256=sha256_file(observation_receipt),
        candidate_observations_inspected=corpus.get("candidate_observations_inspected"),
        additional_episodes=0,
        sampling=corpus.get("sampling"),
    )
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--t23-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-observations", action="store_true")
    parser.add_argument("--observation-receipt", type=Path)
    args = parser.parse_args(argv)
    print("Q0 B8 initializer audit: scanning released checkpoint and T23 bindings", flush=True)
    report = build_report(args.b8, args.t23_root)
    if args.run_observations:
        if args.observation_receipt is None:
            raise ValueError("--run-observations requires the Q0 mass --observation-receipt")
        print("Q0 B8 initializer audit: validating saved mass-audit fidelity corpus", flush=True)
        report["fresh_observation_packet"] = run_observations(args.b8, args.observation_receipt)
        report["status"] = (
            "COMPLETE"
            if prerequisites_pass(report)
            and report["fresh_observation_packet"]["status"] == "COMPLETE"
            else "INCOMPLETE"
        )
    write_json(args.out, report)
    print(
        f"Q0 B8 initializer audit: {report['schema']} {report['status']} -> {args.out}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
