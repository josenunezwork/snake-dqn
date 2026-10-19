"""Prepare an offline release-delivery candidate; it cannot pass dispatcher admission.

The published source and original scientific inputs are pinned, but historical
acceptance is deliberately retained only as evidence. After final-only acceptance,
use normal admission.prepare with freshly priced transport to create the runnable plan.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.s1_mature_20261018 import admission, review_jobs, run


def prepare(args: argparse.Namespace) -> Path:
    commit = admission.published_commit(args.repo, args.published_ref)
    previous = json.loads(args.previous_plan.read_text())
    if (
        previous.get("schema") != admission.SCHEMA
        or previous.get("status") != "PREPARED_NOT_DISPATCHED"
    ):
        raise admission.AdmissionRefused("previous plan must be an immutable prepared S1 plan")
    original = run._manifest_paths(previous)
    inputs = review_jobs.verified_inputs()
    released = {item.dest: item.from_release().as_manifest() for item in inputs}
    local = {
        item.dest: {"path": str(item.path.resolve()), "sha256": item.as_manifest()["sha256"]}
        for item in inputs
    }
    rows = []
    for job, path in original:
        row = json.loads(path.read_text())
        old = {item["dest"]: item for item in row["inputs"]}
        if {
            dest: item["sha256"]
            for dest, item in old.items()
            if dest != "cli-acceptance-result.json"
        } != {dest: item["sha256"] for dest, item in released.items()}:
            raise admission.AdmissionRefused("original plan differs from qualified input bytes")
        row.update(commit=commit, output_dir=str(args.artifact_root.resolve() / "training" / job))
        if Path(row["output_dir"]).exists():
            raise admission.AdmissionRefused("candidate requires fresh job output directories")
        row["inputs"] = [released.get(item["dest"], item) for item in row["inputs"]]
        rows.append(row)
    if args.out.exists() or args.manifest_dir.exists():
        raise admission.AdmissionRefused("candidate plan and manifest directory must be fresh")
    packaging = admission.packaging_preflight(args.repo, rows)
    staging = admission.staging_preflight(rows, packaging)
    plan = copy.deepcopy(previous)
    plan.update(
        status="PENDING_FINAL_ONLY_ACCEPTANCE",
        commit=commit,
        published_ref=args.published_ref,
        source_closure=admission.source_closure(args.repo, commit),
        source_packaging=packaging,
        input_staging=staging,
        local_inputs=local,
        previous_plan={
            "path": str(args.previous_plan.resolve()),
            "sha256": admission.sha256(args.previous_plan),
        },
        admission_blockers=[
            "fresh final-only acceptance and independent review",
            "fresh transport pricing",
            "live Mac release-input preflight",
            "staging settlement and ledger acceptance requalification",
        ],
    )
    args.manifest_dir.mkdir(parents=True)
    plan["manifests"] = []
    for row in rows:
        target = args.manifest_dir / f"{row['job_id']}.json"
        admission.write_once(target, row)
        plan["manifests"].append(
            {
                "job_id": row["job_id"],
                "path": str(target.resolve()),
                "sha256": admission.sha256(target),
            }
        )
    admission.write_once(args.out, plan)
    return args.out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=admission.ROOT)
    parser.add_argument("--published-ref", default="origin/t31-colab-safety-20261018")
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
