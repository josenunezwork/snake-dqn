#!/usr/bin/env python3
"""Prepare a fresh final-only acceptance; execute exactly the reviewed saved plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from qualification import OPS, git, save, tree


def load_transport(repo: Path) -> tuple[ModuleType, ModuleType]:
    """Import the selected clean broker, not the checkout containing this wrapper."""
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("requalification requires a clean broker checkout")
    sys.path.insert(0, str(repo))
    from research.colab_cli import acceptance, admission

    return acceptance, admission


def plan(repo: Path, parent: Path, account_map: Path, colab: str) -> Path:
    """Use a unique state parent even for two plans in the same second."""
    acceptance, _ = load_transport(repo)
    parent.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="requalify-", dir=parent)).resolve()
    record = acceptance.build_plan(
        repo,
        state,
        other_assignments=acceptance.load_other_assignments(account_map),
        profile="final-only",
    )
    run = Path(record["run_dir"])
    saved = run / "ops-review.json"
    value = {
        "repo": str(repo),
        "commit": git(repo, "rev-parse", "HEAD"),
        "tree_hash": tree(repo),
        "state_dir": str(state),
        "colab": colab,
        "plan": record["plan"],
        "manifest_hashes": {
            job["manifest"]: hashlib.sha256(Path(job["manifest"]).read_bytes()).hexdigest()
            for job in record["plan"]["jobs"]
        },
    }
    # This file is create-only review material, separate from the broker's plan.
    with saved.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return saved


def execute(saved: Path, registry: Path, token: str, oauth_python: str) -> Path:
    """Never rebuild a plan on execution, retry, or replace a failed attempt."""
    raw = saved.read_bytes()
    review = json.loads(raw)
    repo = Path(review["repo"])
    acceptance, admission = load_transport(repo)
    if token != acceptance.REVIEW_TOKEN:
        raise ValueError("execution requires COLAB_CLI_ACCEPTANCE_REVIEWED")
    if git(repo, "rev-parse", "HEAD") != review["commit"] or tree(repo) != review["tree_hash"]:
        raise ValueError("broker changed after plan review")
    prepared = review["plan"]
    if prepared["profile"] != "final-only" or prepared["commit"] != review["commit"]:
        raise ValueError("only the pinned final-only plan can execute")
    acceptance.verify_plan_sources(prepared, repo)
    for job in prepared["jobs"]:
        manifest_bytes = Path(job["manifest"]).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != review["manifest_hashes"][job["manifest"]]:
            raise ValueError("acceptance manifest changed after review")
        manifest = json.loads(manifest_bytes)
        if Path(manifest["output_dir"]).exists():
            raise ValueError("acceptance output already exists; prepare a new plan")
        if manifest["commit"] != review["commit"]:
            raise ValueError("acceptance manifest commit differs from review")
    if registry.name == "l4-label-lane-qualified.json":
        # Attempt 2 of the L4 label lane: the prerequisite is a dispatch too,
        # so it needs the independent reviewer ACCEPT at this broker pin first.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from research.rpc_qualification.admission import require_review

        require_review(review["commit"])
    # An interrupted/failed execution stays consumed. Another call must plan afresh.
    with saved.with_name("ops-execute-started.json").open("x") as stream:
        json.dump({"review_sha256": hashlib.sha256(raw).hexdigest()}, stream)
    result_code = acceptance.execute(
        prepared,
        state_dir=Path(review["state_dir"]),
        colab=review["colab"],
        review_token=token,
        oauth_python=oauth_python,
    )
    result_path = saved.with_name("acceptance-result.json")
    result = json.loads(result_path.read_text())
    if result_code != 0:
        raise ValueError("acceptance failed; qualification registry unchanged")
    admission.require_final_only(result, repo)
    acceptance.verify_plan_sources(result["final_plan"], repo)
    # Recheck after execution before publishing qualification.
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("broker changed during acceptance")
    if git(repo, "rev-parse", "HEAD") != review["commit"] or tree(repo) != review["tree_hash"]:
        raise ValueError("broker commit changed during acceptance")
    save(
        registry,
        {
            "schema": "snake-dqn-colab-cli-qualified/v1",
            "qualified_commit": review["commit"],
            "tree_hash": review["tree_hash"],
            "acceptance_id": saved.parent.name,
            "acceptance_result": str(result_path),
            "acceptance_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
    )
    return result_path


def main() -> int:
    """Parse the offline plan and explicitly authorized execute interfaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("plan")
    prepare.add_argument("--broker", required=True, type=Path)
    prepare.add_argument("--other-assignments", required=True, type=Path)
    prepare.add_argument("--state-parent", type=Path)
    prepare.add_argument("--colab", default=str(Path.home() / ".local/bin/colab"))
    run = actions.add_parser("execute")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--review-token", required=True)
    run.add_argument("--oauth-python", default=str(Path.home() / ".venvs/colab-cli/bin/python"))
    run.add_argument(
        "--registry",
        type=Path,
        help="qualification record to write on PASS (default: main's colab-cli-qualified.json); "
        "the L4 label lane uses OPS_DIR/l4-label-lane-qualified.json",
    )
    args = parser.parse_args()
    ops = Path(os.environ.get("OPS_DIR", OPS))
    try:
        if args.action == "plan":
            print(
                plan(
                    args.broker.resolve(),
                    args.state_parent or ops / "requalification",
                    args.other_assignments,
                    args.colab,
                )
            )
        else:
            print(
                execute(
                    args.plan.resolve(),
                    args.registry or ops / "colab-cli-qualified.json",
                    args.review_token,
                    args.oauth_python,
                )
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"REFUSE: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
