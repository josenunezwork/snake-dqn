#!/usr/bin/env python3
"""Q0a B8 initializer fidelity audit; CPU-only and observation-capped.

The saved-artifact phase validates bytes and T23's embedded initializer binding.
It remains INCOMPLETE until the bounded 4,096-observation corpus is captured
and verified. Wrong wrappers, fallback masks, and identity mismatches fail closed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
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
SUPPLEMENTAL_EPISODE_CAP = 12
SUPPLEMENTAL_WAVE_SIZE = 3
OPEN_DEFINITION = "ego2s-local-no-other-snake-cell/a2"
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


def _reset_stale_open(
    rows: list[dict[str, Any]], *, open_definition: object
) -> list[dict[str, Any]]:
    """Never reinterpret a pre-A2 global-population ``open`` boolean as local space."""
    return [
        (
            dict(row)
            if open_definition == OPEN_DEFINITION or row.get("open_definition") == OPEN_DEFINITION
            else dict(row, open=False)
        )
        for row in rows
    ]


def _load_resume_index(
    path: Path, expected_sha256: str, *, fill_receipt_sha256: str
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Q0 supplemental resume index SHA-256 mismatch")
    try:
        index = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Q0 supplemental resume index is invalid") from exc
    if (
        index.get("schema") != "q0-b8-init-fidelity-fill-resume/v1"
        or index.get("fill_bank_receipt_sha256") != fill_receipt_sha256
        or not isinstance(index.get("waves"), dict)
    ):
        raise ValueError("Q0 supplemental resume index binding mismatch")
    return index


def _reservoir_from_rows(rows: list[dict[str, Any]]) -> CoverageReservoir:
    """Load natural saved candidates only; the tie oracle is appended exactly once."""
    reservoir = CoverageReservoir()
    for row in rows:
        if row.get("fixture"):
            if row.get("fixture") != "exact-q-tie-first-legal":
                raise ValueError("unknown Q0 fidelity fixture in saved corpus")
            continue
        reservoir.add(row)
    return reservoir


def _supplemental_tasks(bank: dict[str, Any], b8: Path, limit: int) -> list[dict[str, Any]]:
    worlds = bank.get("worlds")
    if not isinstance(worlds, list) or limit < 1 or limit > len(worlds):
        raise ValueError("invalid bounded supplemental fill world request")
    return [
        {
            "world": world,
            "checkpoint": str(b8),
            "arm": "b8",
            "label": "b8",
            "seed": "b8-init-fill-a2",
        }
        for world in worlds[:limit]
    ]


def _run_supplemental_fill(
    b8: Path,
    reservoir: CoverageReservoir,
    *,
    fill_receipt: Path,
    fill_receipt_sha256: str,
    episodes: int,
    workers: int,
    partials_dir: Path,
    max_seconds: int,
    resume_index: Path | None = None,
    resume_index_sha256: str | None = None,
) -> dict[str, Any]:
    """Run disjoint A2 worlds in fixed three-world waves until coverage closes.

    Every completed world is durably persisted by :mod:`mass_execution`; merging
    into the shared reservoir happens only in plan order after a whole wave.
    The fixed waves make the selected prefix independent of worker completion.
    """
    if not (1 <= episodes <= SUPPLEMENTAL_EPISODE_CAP):
        raise ValueError(f"supplemental episodes must be in 1..{SUPPLEMENTAL_EPISODE_CAP}")
    if workers < 1 or not 1 <= max_seconds <= 900:
        raise ValueError("supplemental workers must be positive and max seconds must be in 1..900")
    import llvmlite
    import numba
    import numpy
    import torch

    from research.q0q1_20261017 import init_fill_bank
    from research.q0q1_20261017.audit_t23_mass import evaluate_episode, initialize_episode_worker
    from research.q0q1_20261017.mass_execution import digest, execute

    receipt_env = {
        init_fill_bank.RECEIPT_ENV: str(fill_receipt),
        init_fill_bank.RECEIPT_SHA256_ENV: fill_receipt_sha256,
    }
    fill = init_fill_bank.load_receipt(env=receipt_env)
    if (resume_index is None) != (resume_index_sha256 is None):
        raise ValueError("supplemental resume requires both index and SHA-256")
    prior = (
        _load_resume_index(
            resume_index, resume_index_sha256, fill_receipt_sha256=fill_receipt_sha256
        )
        if resume_index is not None
        else {"waves": {}}
    )
    tasks = _supplemental_tasks(fill["bank"], b8, episodes)
    deadline = time.monotonic() + max_seconds
    started = time.monotonic()
    completed = 0
    candidates_before = reservoir.seen
    trapped_before = reservoir.excluded_trapped
    waves: list[dict[str, Any]] = []
    # The old corpus uses a fixed deterministic reservoir; all supplemental
    # candidates join that same object, after each persisted wave in plan order.
    for start in range(0, len(tasks), SUPPLEMENTAL_WAVE_SIZE):
        if not assess_observations(reservoir.rows + [exact_tie_fixture()])["missing_coverage"]:
            break
        wave = tasks[start : start + SUPPLEMENTAL_WAVE_SIZE]
        wave_dir = partials_dir / f"wave-{start // SUPPLEMENTAL_WAVE_SIZE:02d}"
        binding = {
            "schema": "q0-b8-init-fidelity-fill-execution/v1",
            "fill_bank_receipt_sha256": fill_receipt_sha256,
            "fill_bank_allocation_sha256": fill["bank"]["allocation_semantic_sha256"],
            "b8_sha256": B8_SHA256,
            "open_definition": OPEN_DEFINITION,
            "wave_tasks_sha256": digest(wave),
            "runtime": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "system": platform.system(),
                "torch": str(torch.__version__),
                "numpy": numpy.__version__,
                "numba": numba.__version__,
                "llvmlite": llvmlite.__version__,
                "threads": 1,
            },
        }
        print(
            f"Q0 B8 init fill wave {start // SUPPLEMENTAL_WAVE_SIZE + 1} "
            f"starting {len(wave)} episodes (cap={episodes}, "
            f"elapsed={time.monotonic() - started:.1f}s)",
            flush=True,
        )
        prior_wave = prior["waves"].get(str(start // SUPPLEMENTAL_WAVE_SIZE))
        if prior_wave is not None and not isinstance(prior_wave, dict):
            raise ValueError("Q0 supplemental resume wave is malformed")
        try:
            results = execute(
                wave,
                worker=evaluate_episode,
                initializer=initialize_episode_worker,
                initargs=(str(b8), deadline),
                binding=binding,
                workers=min(workers, len(wave)),
                deadline=deadline,
                partials_dir=wave_dir,
                resume_partials=(Path(prior_wave["manifest"]) if prior_wave is not None else None),
                resume_sha256=(str(prior_wave["sha256"]) if prior_wave is not None else None),
            )
        except Exception:
            manifest = wave_dir / "manifest.json"
            if manifest.is_file():
                write_json(
                    partials_dir / "resume-index.json",
                    {
                        "schema": "q0-b8-init-fidelity-fill-resume/v1",
                        "fill_bank_receipt_sha256": fill_receipt_sha256,
                        "waves": {
                            **prior["waves"],
                            **{
                                str(item["ordinal"]): {
                                    "manifest": item["partial_manifest"],
                                    "sha256": sha256_file(Path(item["partial_manifest"])),
                                }
                                for item in waves
                            },
                            str(start // SUPPLEMENTAL_WAVE_SIZE): {
                                "manifest": str(manifest),
                                "sha256": sha256_file(manifest),
                            },
                        },
                    },
                )
            raise
        for result in results:
            reservoir.excluded_trapped += int(result["excluded_trapped"])
            for candidate in result["candidates"]:
                reservoir.add(candidate)
        completed += len(results)
        state = assess_observations(reservoir.rows + [exact_tie_fixture()])
        waves.append(
            {
                "ordinal": start // SUPPLEMENTAL_WAVE_SIZE,
                "episodes": len(results),
                "partial_manifest": str(wave_dir / "manifest.json"),
                "missing_coverage_after_wave": state["missing_coverage"],
            }
        )
        index = {
            "schema": "q0-b8-init-fidelity-fill-resume/v1",
            "fill_bank_receipt_sha256": fill_receipt_sha256,
            "waves": {
                str(item["ordinal"]): {
                    "manifest": item["partial_manifest"],
                    "sha256": sha256_file(Path(item["partial_manifest"])),
                }
                for item in waves
            },
        }
        write_json(partials_dir / "resume-index.json", index)
        print(
            f"Q0 B8 init fill episodes {completed}/{episodes} done "
            f"missing={state['missing_coverage']} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    return {
        "additional_episodes": completed,
        "episode_cap": episodes,
        "wave_size": SUPPLEMENTAL_WAVE_SIZE,
        "workers": workers,
        "max_seconds": max_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "fill_bank_receipt_sha256": fill_receipt_sha256,
        "fill_bank_allocation_sha256": fill["bank"]["allocation_semantic_sha256"],
        "fill_namespace": fill["bank"]["namespace"],
        "open_definition": OPEN_DEFINITION,
        "additional_candidate_observations": reservoir.seen - candidates_before,
        "additional_excluded_trapped_observations": reservoir.excluded_trapped - trapped_before,
        "coverage_world_seeds": [int(task["world"]["world_seed"]) for task in tasks[:completed]],
        "waves": waves,
    }


def run_observations(
    b8: Path,
    observation_receipt: Path,
    *,
    supplemental_episodes: int = 0,
    supplemental_workers: int = 1,
    supplemental_partials_dir: Path | None = None,
    supplemental_max_seconds: int = 900,
    fill_bank_receipt: Path | None = None,
    fill_bank_receipt_sha256: str | None = None,
    supplemental_resume_index: Path | None = None,
    supplemental_resume_index_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the saved corpus and, only if needed, append bounded A2 worlds."""
    if not b8.is_file() or sha256_file(b8) != B8_SHA256:
        raise ValueError("Q0 observation packet requires released B8 bytes")
    packet = load_json(observation_receipt)
    if not packet or packet.get("b8", {}).get("sha256") != B8_SHA256:
        raise ValueError("Q0 observation receipt lacks released B8 binding")
    corpus = packet.get("development_packet", {}).get("fidelity_corpus", {})
    rows = corpus.get("rows", [])
    if not rows or corpus.get("bank_audit_pass") is not True:
        raise ValueError("Q0 mass audit has no collision-audited fidelity corpus")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Q0 mass audit has malformed fidelity rows")
    # r5's ``open`` field was the superseded global alive-count proxy.  It
    # cannot be transformed into the A2 local-raster predicate, so preserve its
    # other observations but remove it from A2 coverage evidence.
    reset_old_open = corpus.get("open_definition") != OPEN_DEFINITION and any(
        row.get("open_definition") != OPEN_DEFINITION for row in rows
    )
    source_rows = (
        _reset_stale_open(rows, open_definition=corpus.get("open_definition"))
        if reset_old_open
        else rows
    )
    # Validate every saved candidate, including the prior tie fixture, before
    # replacing that fixture with the current executable oracle below.
    source_assessment = assess_observations(source_rows)
    if source_assessment["action_disagreements"]:
        raise ValueError("Q0 saved fidelity corpus has action disagreement")
    reservoir = _reservoir_from_rows(source_rows)
    fill = {
        "additional_episodes": 0,
        "additional_candidate_observations": 0,
        "additional_excluded_trapped_observations": 0,
        "open_definition": corpus.get("open_definition"),
    }
    if supplemental_episodes:
        if (
            fill_bank_receipt is None
            or fill_bank_receipt_sha256 is None
            or supplemental_partials_dir is None
        ):
            raise ValueError("supplemental fill requires fill-bank receipt and partials directory")
        fill = _run_supplemental_fill(
            b8,
            reservoir,
            fill_receipt=fill_bank_receipt,
            fill_receipt_sha256=fill_bank_receipt_sha256,
            episodes=supplemental_episodes,
            workers=supplemental_workers,
            partials_dir=supplemental_partials_dir,
            max_seconds=supplemental_max_seconds,
            resume_index=supplemental_resume_index,
            resume_index_sha256=supplemental_resume_index_sha256,
        )
    outcome = assess_observations(reservoir.rows + [exact_tie_fixture()])
    fallback = exact_tie_fixture(fallback=True)
    if fallback["action_a"] != fallback["action_b"] or fallback["action_a"] != 0:
        raise ValueError("legitimate normal-only fallback selector disagreement")
    outcome["normal_only_fallback_oracle"] = fallback
    outcome.update(
        source_receipt_sha256=sha256_file(observation_receipt),
        candidate_observations_inspected=int(
            corpus.get("candidate_observations_inspected", len(rows))
        )
        + fill["additional_candidate_observations"],
        excluded_trapped_observations=int(corpus.get("excluded_trapped_observations", 0))
        + fill["additional_excluded_trapped_observations"],
        additional_episodes=fill["additional_episodes"],
        supplemental_fill=fill,
        source_open_flags_reset=reset_old_open,
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
    parser.add_argument("--supplemental-episodes", type=int, default=0)
    from research.q0q1_20261017.mass_execution import default_workers

    parser.add_argument("--supplemental-workers", type=int, default=default_workers())
    parser.add_argument("--supplemental-partials-dir", type=Path)
    parser.add_argument("--supplemental-max-seconds", type=int, default=900)
    parser.add_argument("--fill-bank-receipt", type=Path)
    parser.add_argument("--fill-bank-receipt-sha256")
    parser.add_argument("--supplemental-resume-index", type=Path)
    parser.add_argument("--supplemental-resume-index-sha256")
    args = parser.parse_args(argv)
    if args.supplemental_episodes:
        receipt_path = args.fill_bank_receipt or (
            Path(os.environ["Q0Q1_INIT_FILL_BANK_RECEIPT"])
            if os.environ.get("Q0Q1_INIT_FILL_BANK_RECEIPT")
            else None
        )
        receipt_sha256 = args.fill_bank_receipt_sha256 or os.environ.get(
            "Q0Q1_INIT_FILL_BANK_RECEIPT_SHA256"
        )
        if receipt_path is None or receipt_sha256 is None:
            parser.error("supplemental fill requires a hash-bound fill-bank receipt")
    else:
        receipt_path, receipt_sha256 = None, None
    print("Q0 B8 initializer audit: scanning released checkpoint and T23 bindings", flush=True)
    report = build_report(args.b8, args.t23_root)
    if args.run_observations:
        if args.observation_receipt is None:
            raise ValueError("--run-observations requires the Q0 mass --observation-receipt")
        print("Q0 B8 initializer audit: validating saved mass-audit fidelity corpus", flush=True)
        report["fresh_observation_packet"] = run_observations(
            args.b8,
            args.observation_receipt,
            supplemental_episodes=args.supplemental_episodes,
            supplemental_workers=args.supplemental_workers,
            supplemental_partials_dir=args.supplemental_partials_dir,
            supplemental_max_seconds=args.supplemental_max_seconds,
            fill_bank_receipt=receipt_path,
            fill_bank_receipt_sha256=receipt_sha256,
            supplemental_resume_index=args.supplemental_resume_index,
            supplemental_resume_index_sha256=args.supplemental_resume_index_sha256,
        )
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
