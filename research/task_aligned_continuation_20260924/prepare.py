"""Bind completed metadata and new source; freeze only after independent review."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.task_aligned_20260924 import adapter
from research.task_aligned_continuation_20260924 import evaluation, learner, runner

PACKAGE = Path(__file__).resolve().parent
BASE = REPO.parent / "snake-dqn-artifacts/ongoing-research-20260913"
PREVIOUS = BASE / "task-aligned-cz-integration-20260924/admission-v2"
SUPERVISOR = BASE / "apex-scaling-h256-diagnostic/supervise.py"
OLD_SPEC = PREVIOUS / "run/eval_parent_2026096102/output/evaluation-spec.json"
PREVIEW = BASE / "task-aligned-first-lineage-preview-20260924/intent.json"
RUNTIME_SEEDS = {"2026096102": 2026092411}
QUALIFICATION_SEED = 2026092410


def source_closure() -> dict[str, str]:
    """Enumerate named code/config only; never traverse historical raw trees."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "configs"], cwd=REPO, text=True
    ).splitlines()
    files = [
        REPO / name for name in tracked if Path(name).suffix in {".py", ".yaml", ".yml", ".json"}
    ]
    for directory in (PACKAGE, REPO / "research/task_aligned_20260924"):
        files += list(directory.glob("*.py"))
        files += [p for p in directory.glob("*.md") if p.name != "review-notes.md"]
    files += [REPO / "requirements.txt", REPO / "requirements-dev.txt", SUPERVISOR]
    return {str(p.resolve()): runner.digest(p) for p in sorted(set(files))}


def make_draft(root: Path) -> dict:
    """Read sealed metadata; checkpoint identities remain inert before admission."""
    old = runner.read(OLD_SPEC)
    preview = runner.read(PREVIEW)
    completed = PREVIOUS / "run/train_2026096102/completed.json"
    report_path = PREVIOUS / "run/train_2026096102/output/report.json"
    receipt, report = runner.read(completed), runner.read(report_path)
    if (
        runner.digest(completed)
        != "96f28fcf57bd45a03636aac046e58269a7b26419ebf2943889fed6b502531d65"
    ):
        raise ValueError("Completed source receipt differs")
    if runner.digest(report_path) != receipt["report_sha256"]:
        raise ValueError("Completed source report differs")
    if (
        report["final_checkpoint"]["sha256"] != learner.SOURCE_SHA256
        or report["final_update"] != 35613
    ):
        raise ValueError("Source endpoint metadata differs")
    if receipt["artifacts"][str(learner.SOURCE_PATH)] != learner.SOURCE_SHA256:
        raise ValueError("Source receipt lacks the nominated checkpoint")
    excluded = set(old["training_world_seeds"])
    excluded.update(seed for values in old["world_seeds"].values() for seed in values)
    excluded.update(
        seed for profile in preview["profiles"].values() for seed in profile["world_seeds"]
    )
    previous_intent = runner.read(PREVIOUS / "intent.json")
    if (
        runner.digest(PREVIOUS / "intent.json")
        != "9ea0f3b4e7fd449bde4cd95ba91827a6b1b3897770a86ba2db58c72939955099"
    ):
        raise ValueError("Prior admitted identity differs")
    cz_binding = previous_intent["parents"]["2026096102"]
    cz_report = Path(cz_binding["report_path"])
    if runner.digest(cz_report) != cz_binding["report_sha256"]:
        raise ValueError("Exact original6102 configuration metadata differs")
    config = runner.read(cz_report)["config"]
    contract = evaluation.canonical_hash(adapter.evaluation_contract(config))
    if contract != old["contract_sha256"]:
        raise ValueError("Source serving contract differs from completed contract")
    effective_seed = learner.effective_runtime_seed(RUNTIME_SEEDS["2026096102"])
    spec = {
        "schema_version": 1,
        "study_id": evaluation.STUDY_ID,
        "lineages": [learner.SOURCE_SEED],
        "horizon": 3000,
        "source_update": 35613,
        "candidate_update": 66618,
        "continuation_rollouts": 31005,
        "evaluation_device": "cpu",
        "batch_worlds": 32,
        "source_config": config,
        "contract_sha256": contract,
        "policy_sha256": {},
        "profiles": evaluation.PROFILES,
        "decision": evaluation.DECISION,
        "world_seeds": evaluation.make_world_seeds(),
        "training_seed_namespaces": [f"task-aligned/v1/{effective_seed}"],
        "held_seed_namespace": evaluation.HELD_NAMESPACE,
        "training_world_seeds": [],
        "excluded_world_seeds": sorted(excluded),
        "novelty_scope": "declared_seed_namespaces_only",
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
        "--heartbeat",
        "{heartbeat}",
        "--stage-id",
        "{stage_id}",
    ]

    def stage(ident, kind, cap, counts, **other):
        return {
            "id": ident,
            "kind": kind,
            "cap_seconds": cap,
            "argv": list(common),
            "counter_caps": dict(zip(runner.COUNTERS, counts)),
            "device": "mps" if kind in ("qualification", "training") else "cpu",
            **other,
        }

    stages = [
        stage("qualification", "qualification", 900, (2560, 2560, 180, 2, 10)),
        stage(
            "train_2026096102",
            "training",
            25200,
            (7937280, 7937280, 558090, 1, 31005),
            lineage=2026096102,
        ),
        stage(
            "eval_parent_2026096102",
            "evaluation",
            900,
            (288000, 288000, 9000, 1, 0),
            lineage=2026096102,
            role="parent",
        ),
        stage(
            "eval_candidate_2026096102",
            "evaluation",
            900,
            (288000, 288000, 9000, 1, 0),
            lineage=2026096102,
            role="candidate",
        ),
        stage("anchors", "evaluation", 900, (576000, 576000, 0, 0, 0), role="anchor"),
        stage("analysis", "analysis", 600, (0, 0, 0, 0, 0)),
        stage("audit", "audit", 900, (0, 0, 0, 0, 0)),
    ]
    stages[0]["counter_caps"].update(
        model_forward_rows=5280,
        opponent_transitions=6144,
        sampled_transition_draws=2560,
        completed_rollouts=10,
        preview_calls=10,
        episode_starts=208,
        checkpoint_files_hashed=2,
    )
    stages[1]["counter_caps"].update(
        model_forward_rows=16370640,
        opponent_transitions=19049472,
        sampled_transition_draws=7937280,
        completed_rollouts=31005,
        preview_calls=31005,
        episode_starts=496128,
        checkpoint_files_hashed=1,
    )
    for item in stages[2:5]:
        learned = item["role"] != "anchor"
        item["counter_caps"].update(
            model_forward_rows=288000 if learned else 0,
            opponent_transitions=960000 if learned else 1920000,
            episode_starts=96 if learned else 192,
            checkpoint_files_hashed=int(learned),
            sampled_transition_draws=0,
            completed_rollouts=0,
            preview_calls=0,
        )
    stages[-1]["argv"][1] = str(PACKAGE / "audit.py")
    inputs = [
        OLD_SPEC,
        PREVIEW,
        PREVIOUS / "intent.json",
        cz_report,
        BASE / "task-aligned-continuation-20260924/admission-v1/draft.json",
        BASE / "task-aligned-continuation-20260924/admission-v1/superseded.json",
        completed,
        report_path,
        PREVIOUS / "closeout.json",
        PREVIOUS / "run/qualification/output/report.json",
        BASE / "task-aligned-continuation-direction-20260924T133044Z.json",
    ]
    return {
        "schema_version": 1,
        "study_id": evaluation.STUDY_ID,
        "admitted": False,
        "repo": str(REPO),
        "python": python,
        "device": "mps",
        "output_root": str(root / "run"),
        "handoff_seconds": 120,
        "elapsed_window_seconds": 32400,
        "recovery_seconds": 0,
        "supervisor_source": str(SUPERVISOR),
        "source_sha256": source_closure(),
        "input_sha256": {str(p): runner.digest(p) for p in inputs},
        "source_checkpoint": learner.source_binding(),
        "evaluation_source": str(PACKAGE / "evaluation.py"),
        "evaluation": spec,
        "runtime_seeds": RUNTIME_SEEDS,
        "qualification_seed": QUALIFICATION_SEED,
        "task_contract": adapter.task_contract(),
        "stages": stages,
        "training": {
            "fixed_rollouts": 31005,
            "starting_update": 35613,
            "final_update": 66618,
            "scheduled_hero_slots_ceiling": 7937280,
            "profile_rollout_cycle": list(adapter.SCHEDULE),
            "checkpoints": "final-only",
            "partial_restart": False,
            "exact_runtime_resume": False,
            "historical_rollout_replay": False,
        },
        "qualification_limits": dict(
            stages[0]["counter_caps"], schema_fixtures=4, fixture_bytes_each=1048576
        ),
        "qualification_reuse": "Completed native collector, Q(lambda), serving and lifecycle primitives; only new restoration/serialization/harness contracts are qualified.",
        "promotion_eligible": False,
        "claim": "Outcome-selected fixed6102 continuation package; one fresh paired screen, not population or promotion evidence.",
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
        runner.durable(root / "draft.json", make_draft(root))
        print({"draft": str(root / "draft.json"), "admitted": False})
        return
    if args.review is None:
        raise ValueError("Independent source-bound review required")
    original = runner.read(root / "draft.json")
    fresh = make_draft(root)
    if fresh != original:
        raise ValueError("Preserve and version changed draft; no silent refresh")
    review = runner.read(args.review)
    if review.get("status") != "PASS_FOR_ADMISSION" or review.get("draft_sha256") != runner.digest(
        root / "draft.json"
    ):
        raise ValueError("Missing independent exact-draft admission review")
    now = datetime.now(timezone.utc)
    intent = dict(
        fresh,
        admitted=True,
        frozen_utc=now.isoformat(),
        deadline=(now + timedelta(seconds=32400)).isoformat(),
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
        {
            "intent": str(root / "intent.json"),
            "sha256": runner.digest(root / "intent.json"),
            "deadline": intent["deadline"],
        }
    )


if __name__ == "__main__":
    main()
