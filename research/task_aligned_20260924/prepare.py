"""Prepare metadata, then freeze only after a source-bound independent PASS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.task_aligned_20260924 import adapter, evaluation, runner

PACKAGE = Path(__file__).resolve().parent
SUPERVISOR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/"
    "apex-scaling-h256-diagnostic/supervise.py"
)
RUNTIME_SEEDS = {str(seed): 2026092401 + i for i, seed in enumerate(adapter.PARENT_SEEDS)}


def held_seed(profile: str, index: int) -> int:
    namespace = f"task-aligned-20260924/held-v1/{profile}/{index}"
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")


def source_closure() -> dict[str, str]:
    """Only named code/config roots; never discover historical raw evidence."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "configs"], cwd=REPO, text=True
    ).splitlines()
    files = [
        REPO / name for name in tracked if Path(name).suffix in {".py", ".yaml", ".yml", ".json"}
    ]
    files += list(PACKAGE.glob("*.py"))
    files += [path for path in PACKAGE.glob("*.md") if path.name != "review-notes.md"]
    files += [REPO / "requirements.txt", REPO / "requirements-dev.txt", SUPERVISOR]
    return {str(path.resolve()): runner.digest(path) for path in sorted(set(files))}


def make_draft(evidence_root: Path) -> dict:
    """Use completed report metadata; no checkpoint access or world materialization."""
    parents = {}
    config = None
    for seed in adapter.PARENT_SEEDS:
        bound = adapter.parent_spec(seed)
        parents[str(seed)] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(bound).items()
        }
        current = json.loads(bound.report_path.read_text())["config"]
        if config is not None and adapter.evaluation_contract(
            current
        ) != adapter.evaluation_contract(config):
            raise RuntimeError("source parent native serving contracts differ")
        config = current if config is None else config
    evaluation_spec = {
        "schema_version": 1,
        "study_id": evaluation.STUDY_ID,
        "horizon": 3000,
        "lineages": list(adapter.PARENT_SEEDS),
        "contract_sha256": evaluation.canonical_hash(adapter.evaluation_contract(config)),
        "source_config": config,
        "evaluation_device": "cpu",
        "batch_worlds": 32,
        "profiles": {
            "solo": [],
            "food_pressure": ["greedy_food"] * 4 + ["random_safe"],
            "mixed": ["greedy_food"] * 2 + ["random_safe"] * 3,
        },
        "world_seeds": {
            profile: [held_seed(profile, i) for i in range(32)]
            for profile in ("solo", "food_pressure", "mixed")
        },
        "training_seed_namespaces": [f"task-aligned/v1/{seed}" for seed in RUNTIME_SEEDS.values()],
        "held_seed_namespace": "task-aligned-20260924/held-v1",
        "training_world_seeds": [],
        "novelty_scope": "declared_seed_namespaces_only",
        "decision": {
            "relative_mass_gain": 0.1,
            "absolute_mass_gain": 1.0,
            "solo_mass_food_loss": 0.05,
            "solo_survival_loss": 0.02,
            "minimum_exposed_worlds": 8,
        },
    }
    python = str(REPO / "venv/bin/python")
    common = [
        python,
        str(PACKAGE / "stage_cli.py"),
        "--intent",
        "{intent}",
        "--root",
        "{root}",
        "--out",
        "{out}",
        "--stage-id",
        "{stage_id}",
    ]

    def stage(
        ident: str, kind: str, seconds: int, counts: tuple[int, int, int, int, int], **other: object
    ) -> dict:
        return {
            "id": ident,
            "kind": kind,
            "cap_seconds": seconds,
            "argv": common,
            "counter_caps": dict(zip(runner.COUNTERS, counts)),
            "device": "mps" if kind in {"qualification", "training"} else "cpu",
            **other,
        }

    stages = [stage("qualification", "qualification", 1800, (60000, 60000, 6000, 4, 150))]
    stages += [
        stage(
            f"train_{seed}",
            "training",
            25200,
            (9_999_360, 9_999_360, 703080, 1, 39060),
            lineage=seed,
        )
        for seed in adapter.PARENT_SEEDS
    ]
    for training_stage in stages[1:]:
        training_stage["counter_caps"].update(
            model_forward_rows=20_623_680,
            opponent_transitions=23_998_464,
            sampled_transition_draws=9_999_360,
            completed_rollouts=39_060,
            preview_calls=39_060,
            episode_starts=624_960,
            checkpoint_files_hashed=1,
        )
    stages += [
        stage(
            f"eval_{role}_{seed}",
            "evaluation",
            900,
            (288000, 288000, 9000, 1, 0),
            lineage=seed,
            role=role,
        )
        for role in ("parent", "candidate")
        for seed in adapter.PARENT_SEEDS
    ]
    stages += [
        stage("anchors", "evaluation", 900, (576000, 576000, 0, 0, 0), role="anchor"),
        stage("analysis", "analysis", 600, (0, 0, 0, 0, 0)),
        stage("audit", "audit", 900, (0, 0, 0, 0, 0)),
    ]
    stages[-1]["argv"] = [
        python,
        str(PACKAGE / "audit.py"),
        "--intent",
        "{intent}",
        "--root",
        "{root}",
        "--out",
        "{out}",
        "--heartbeat",
        "{heartbeat}",
        "--stage-id",
        "{stage_id}",
    ]
    closure = source_closure()
    for bound in parents.values():
        closure[bound["report_path"]] = bound["report_sha256"]
    return {
        "schema_version": 1,
        "study_id": evaluation.STUDY_ID,
        "admitted": False,
        "output_root": str(evidence_root / "run"),
        "repo": str(REPO),
        "python": python,
        "device": "mps",
        "handoff_seconds": 120,
        "elapsed_window_seconds": 86400,
        "supervisor_source": str(SUPERVISOR),
        "source_sha256": closure,
        "evaluation_source": str(PACKAGE / "evaluation.py"),
        "evaluation": evaluation_spec,
        "parents": parents,
        "runtime_seeds": RUNTIME_SEEDS,
        "qualification_seed": 2026092400,
        "stages": stages,
        "task_contract": adapter.task_contract(),
        "training": {
            "ceiling_rollouts": 39060,
            "minimum_rollouts": 7815,
            "scheduled_hero_slots_ceiling": 9_999_360,
            "profile_rollout_cycle": ["solo", "s2", "s2", "s6", "s6"],
            "checkpoint_rollout_fractions": [0.25, 0.5, 1.0],
            "final_only_behavioral_selection": True,
            "partial_restart": False,
        },
        "qualification_limits": {
            "native_frames": 60000,
            "hero_transitions": 60000,
            "model_forwards": 6000,
            "model_forward_rows": 150000,
            "checkpoint_loads": 4,
            "optimizer_updates": 150,
            "sampled_transition_draws": 38400,
            "episode_starts": 4096,
            "preview_calls": 256,
        },
        "recovery_seconds": 0,
        "promotion_eligible": False,
        "package_claim": "Outcome-informed fixed-cohort package utility, not isolated causality",
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "psutil", "PyYAML", "matplotlib")
        },
        "python_version": sys.version,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.draft:
        root.mkdir(parents=False, exist_ok=False)
        draft = make_draft(root)
        runner.durable(root / "draft.json", draft)
        print(json.dumps({"draft": str(root / "draft.json"), "admitted": False}))
        return
    if args.review is None:
        raise ValueError("a bound independent implementation review is required")
    original = runner.read(root / "draft.json")
    fresh = make_draft(root)
    if original != fresh:
        raise RuntimeError("draft or sources changed after review; preserve and version the draft")
    review = runner.read(args.review)
    if review.get("status") != "PASS_FOR_ADMISSION" or review.get("draft_sha256") != runner.digest(
        root / "draft.json"
    ):
        raise RuntimeError("independent source-bound admission review missing")
    now = datetime.now(timezone.utc)
    intent = dict(
        fresh,
        admitted=True,
        frozen_utc=now.isoformat(),
        deadline=(now + timedelta(seconds=86400)).isoformat(),
        implementation_review={
            "path": str(args.review.resolve()),
            "sha256": runner.digest(args.review),
        },
    )
    intent["source_sha256"] = dict(intent["source_sha256"])
    intent["source_sha256"][str(args.review.resolve())] = runner.digest(args.review)
    runner.validate_intent(intent)
    runner.durable(root / "intent.json", intent)
    print(
        json.dumps(
            {
                "intent": str(root / "intent.json"),
                "sha256": runner.digest(root / "intent.json"),
                "deadline": intent["deadline"],
                "admitted": True,
            }
        )
    )


if __name__ == "__main__":
    main()
