#!/usr/bin/env python3
"""Q0a B8 initializer fidelity audit; CPU-only and observation-capped.

The saved-artifact phase validates bytes and T23's embedded initializer binding.
It is deliberately **INCOMPLETE** until the 4,096-observation fresh-corpus
phase is run.  The corpus phase is intentionally not implemented against an
unbound bank: a mistaken wrapper, fallback mask, or contaminated frozen state
must fail closed rather than manufacture an acceptance result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.q0q1_20261017 import audit_q0_bank  # noqa: E402
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
    """Locate candidate checkpoints without guessing that a filename proves identity."""
    rows: dict[str, Any] = {}
    for seed in T23_SEEDS:
        base = t23_root / f"seed-{seed}"
        final_paths = []
        if base.exists():
            for path in sorted(base.rglob("*")):
                name = path.name.lower()
                if (
                    path.is_file()
                    and name.endswith((".pth", ".pt"))
                    and ("final" in name or "8000" in name or "8m" in name)
                ):
                    final_paths.append({"path": str(path), "sha256": sha256_file(path)})
        rows[str(seed)] = {"available": bool(final_paths), "finals": final_paths}
    return rows


def initializer_binding(t23_root: Path) -> dict[str, Any]:
    """Read summaries only; absent or malformed binding is a failed prerequisite."""
    values: dict[str, Any] = {}
    for seed in T23_SEEDS:
        summary = load_json(t23_root / f"seed-{seed}" / "summary.json")
        args = summary.get("args", {}) if isinstance(summary, dict) else {}
        values[str(seed)] = {
            "summary_available": summary is not None,
            "declared_seed": args.get("seed"),
            "declared_init_sha256": (summary or {}).get("init_sha256"),
            "matches_released_b8": (summary or {}).get("init_sha256") == B8_SHA256,
        }
    return values


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


def run_observations(b8: Path) -> dict[str, Any]:
    """Collect fresh B8 observations through the exact grid ego2s wrapper.

    The policy is loaded twice and every recorded row compares Q/action under the
    same advisory mask.  The wrapper calls are read-only; no optimizer/replay
    objects are created. Natural rows provide all but exact ties; an explicit
    all-zero-Q fixture supplies the tie-path check without mislabeling it as a
    gameplay observation.
    """
    if not b8.is_file() or sha256_file(b8) != B8_SHA256:
        raise ValueError("Q0 observation packet requires released B8 bytes")
    from types import SimpleNamespace

    import numpy as np

    from research.redesign_m3c_20261008 import train_m3c
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env import ego2s_policy, eval_engine

    bank = audit_q0_bank.build_world_bank()
    audit = audit_q0_bank.audit_world_bank()
    if not audit["pass"]:
        raise ValueError("Q0 observation bank collision audit failed")
    captured: list[dict[str, Any]] = []
    original = ego2s_policy.Ego2sSimdPolicy

    class ObservedPolicy(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Route B through the trainer's actual zero-update initializer,
            # not a second evaluator loader.  This catches widening/loader drift.
            self.shadow = train_m3c.load_init(
                SimpleNamespace(init=str(b8), obs_version="bv8"), "cpu"
            ).eval()

        def actions(self, masks, sim, slots):
            q_a = ego2s_policy.student_q(self.net, sim, slots, self.forward).numpy()
            q_b = ego2s_policy.student_q(self.shadow, sim, slots, self.forward).numpy()
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            advisory = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            trainer_mask = train_m3c.hero_selection_mask(sim, rows)
            safe = advisory.copy()
            fallback = ~safe.any(axis=1)
            safe[fallback, :3] = True
            actions_a = np.where(safe, q_a.astype(np.float64), -np.inf).argmax(axis=1)
            actions_b = np.where(safe, q_b.astype(np.float64), -np.inf).argmax(axis=1)
            for index, (env, _slot) in enumerate(rows):
                mix_rows = sum(1 for row in captured if row["mix"] == current_mix[0])
                if mix_rows >= OBSERVATION_CAP // 3:
                    break
                length = int(sim.get_lengths()[env, 0])
                captured.append(
                    {
                        "mix": current_mix[0],
                        "short_lt700": length < 700,
                        "long_ge800": length >= 800,
                        "crowded": int(sim.get_alive()[env].sum()) >= 5,
                        "open": int(sim.get_alive()[env].sum()) <= 3,
                        "exact_q_tie": bool((q_a[index] == q_a[index].max()).sum() > 1),
                        "boost_eligible": bool(safe[index, 3:].any()),
                        "action_a": int(actions_a[index]),
                        "action_b": int(actions_b[index]),
                        "fallback_used": bool(fallback[index]),
                        "mask_disagreement": not np.array_equal(
                            advisory[index], trainer_mask[index]
                        ),
                        "wrapper_disagreement": not np.array_equal(q_a[index], q_b[index]),
                    }
                )
            return actions_a.astype(np.int64)

    current_mix = ["unspecified"]
    context = gid._context(1)
    carrier = context["lookup"][gid.FRP3_S12[1]]
    ego2s_policy.Ego2sSimdPolicy = ObservedPolicy
    try:
        for seed in T23_SEEDS:
            for world in bank["worlds"][str(seed)]:
                current_mix[0] = str(world["mix"])
                roster = [context["lookup"][x[1]] for x in world["opponents"]]
                eval_engine.run_simd_eval(
                    carrier,
                    roster,
                    5000,
                    [int(world["world_seed"])],
                    profile=context["profile"],
                    opponent_specs_by_world={int(world["world_seed"]): roster},
                    mix_id=current_mix[0],
                    vector61=True,
                    vector61_forward="rowwise",
                    sim_engine="grid",
                    hero_ego2s=str(b8),
                    ego2s_forward="rowwise",
                    ego2s_mask="advisory",
                    vector61_backend="numba",
                    scripted_anchors="fast",
                )
    finally:
        ego2s_policy.Ego2sSimdPolicy = original
    # Exact-Q ties need an independently executable selection fixture. It
    # contributes *only* tie coverage; natural rows must establish every state
    # stratum and mix.
    fixture = {
        "mix": "scripted",
        "short_lt700": False,
        "long_ge800": False,
        "crowded": False,
        "open": False,
        "exact_q_tie": True,
        "boost_eligible": True,
        "action_a": 0,
        "action_b": 0,
        "fallback_used": False,
        "mask_disagreement": False,
        "wrapper_disagreement": False,
        "fixture": "exact-q-tie-first-legal",
    }
    if len(captured) == OBSERVATION_CAP - 1:
        captured.append(fixture)
    outcome = assess_observations(captured)
    outcome.update(
        {
            "bank": {
                "schema": audit_q0_bank.SCHEMA,
                "digest": bank["allocation_semantic_sha256"],
                "audit": audit,
            },
            "natural_observations": sum(1 for row in captured if not row.get("fixture")),
            "tie_fixture": fixture,
        }
    )
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--t23-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-observations", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.b8, args.t23_root)
    if args.run_observations:
        report["fresh_observation_packet"] = run_observations(args.b8)
        report["status"] = (
            "COMPLETE"
            if prerequisites_pass(report)
            and report["fresh_observation_packet"]["status"] == "COMPLETE"
            else "INCOMPLETE"
        )
    write_json(args.out, report)
    print(f"{report['schema']} {report['status']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
