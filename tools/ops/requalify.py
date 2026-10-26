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


def plan(
    repo: Path,
    parent: Path,
    account_map: Path,
    colab: str,
    *,
    d1024_inventory: bool = False,
    transfer_authority_sha256: str | None = None,
    f5_timing_authority: Path | None = None,
) -> Path:
    """Use a unique state parent even for two plans in the same second."""
    acceptance, _ = load_transport(repo)
    parent.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="requalify-", dir=parent)).resolve()
    record = acceptance.build_plan(
        repo,
        state,
        other_assignments=acceptance.load_other_assignments(account_map),
        profile="final-only",
        d1024_inventory=d1024_inventory,
        transfer_authority_sha256=transfer_authority_sha256,
        f5_timing_authority=f5_timing_authority,
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
        "d1024_inventory": d1024_inventory,
        "transfer_authority_sha256": transfer_authority_sha256,
        "f5_timing_authority": (
            {
                "path": str(f5_timing_authority.resolve()),
                "sha256": hashlib.sha256(f5_timing_authority.read_bytes()).hexdigest(),
            }
            if f5_timing_authority
            else None
        ),
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
    wants_d1024 = review.get("d1024_inventory", False)
    if not isinstance(wants_d1024, bool) or wants_d1024 != (
        "d1024_synthetic_inventory" in prepared
    ):
        raise ValueError("reviewed D1024 synthetic-inventory selection differs from plan")
    declared_authority = review.get("transfer_authority_sha256")
    if declared_authority is not None and not isinstance(declared_authority, str):
        raise ValueError("reviewed D1024 transfer authority is malformed")
    if bool(declared_authority) != ("transfer_exception" in prepared):
        raise ValueError("reviewed D1024 transfer authority differs from plan")
    if declared_authority is not None and (
        declared_authority != acceptance.transfer_policy.RULING_SHA256
        or prepared.get("transfer_exception")
        != acceptance.transfer_policy.authority_binding("acceptance")
    ):
        raise ValueError("reviewed D1024 transfer authority differs from ruling")
    f5_authority = review.get("f5_timing_authority")
    if f5_authority is not None:
        if (
            not isinstance(f5_authority, dict)
            or not isinstance(f5_authority.get("path"), str)
            or not isinstance(f5_authority.get("sha256"), str)
            or not Path(f5_authority["path"]).is_file()
            or hashlib.sha256(Path(f5_authority["path"]).read_bytes()).hexdigest()
            != f5_authority["sha256"]
            or prepared.get("f5_timing_transfer_exception")
            != acceptance.transfer_policy.f5_timing_authority_binding(
                prepared.get("f5_timing_synthetic_inventory", {}).get("sha256", ""), "acceptance"
            )
        ):
            raise ValueError("reviewed F5 timing transfer authority differs from ruling")
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
    prepare.add_argument(
        "--d1024-inventory",
        action="store_true",
        help="stage the prospective synthetic D1024 inventory in this acceptance plan",
    )
    prepare.add_argument(
        "--transfer-authority-sha256",
        help="F4 ruling SHA-256 for the prospective D1024 transfer exception",
    )
    prepare.add_argument(
        "--f5-timing-authority",
        type=Path,
        help="sealed F5 timing recovery authority staged synthetically in this acceptance",
    )
    run = actions.add_parser("execute")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--review-token", required=True)
    run.add_argument("--oauth-python", default=str(Path.home() / ".venvs/colab-cli/bin/python"))
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
                    d1024_inventory=args.d1024_inventory,
                    transfer_authority_sha256=args.transfer_authority_sha256,
                    f5_timing_authority=args.f5_timing_authority,
                )
            )
        else:
            print(
                execute(
                    args.plan.resolve(),
                    ops / "colab-cli-qualified.json",
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
