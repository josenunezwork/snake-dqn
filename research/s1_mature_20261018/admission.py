"""Create the frozen S1 admission plan without allocating any remote resource.

``prepare`` is deliberately local and create-only.  ``run`` lives in :mod:`run`;
keeping allocation out of this module makes a reviewable plan possible before an
operator has authority to spend a single Colab unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from research.s1_mature_20261018 import bank, final_artifacts, ledger, manifests

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "s1-mature-admission-plan/v1"
APPROVAL = manifests.APPROVAL
ACCEPTANCE_UNITS = 4.0
PARALLEL_APPROVAL = "APPROVED_S1_PARALLEL_20261019"
MAX_LANES = 3


class AdmissionRefused(RuntimeError):
    """The proposed S1 run is not yet admissible."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise AdmissionRefused(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def published_commit(repo: Path, published_ref: str = "origin/main") -> str:
    """Require clean source at an explicitly named remote-tracking branch head."""
    if not published_ref.startswith("origin/") or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-"
        for char in published_ref
    ):
        raise AdmissionRefused("published reference must name an origin remote-tracking branch")
    git(repo, "show-ref", "--verify", f"refs/remotes/{published_ref}")
    head, main = git(repo, "rev-parse", "HEAD"), git(
        repo, "rev-parse", f"refs/remotes/{published_ref}"
    )
    if head != main:
        raise AdmissionRefused(
            f"prepare requires HEAD to equal the current published {published_ref}"
        )
    if git(repo, "status", "--porcelain"):
        raise AdmissionRefused("prepare requires a clean published worktree")
    return head


def source_closure(repo: Path, commit: str) -> dict[str, str]:
    """Hash every tracked source byte visible at the pinned public commit."""
    paths = git(repo, "ls-files", "-z").split("\0")
    result: dict[str, str] = {}
    for relative in paths:
        if relative:
            target = repo / relative
            if not target.is_file():
                raise AdmissionRefused(f"tracked source is absent: {relative}")
            result[relative] = sha256(target)
    if not result or git(repo, "rev-parse", "HEAD") != commit:
        raise AdmissionRefused("source closure cannot be bound to the published commit")
    return result


def packaging_preflight(repo: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build every pinned supervisor fallback locally, before reserving any job."""
    from research.colab_cli import source

    results = {}
    qualification: dict[str, Any] = {}
    for row in rows:
        job = row["job_id"]
        try:
            with tempfile.TemporaryDirectory(prefix="s1-source-preflight-") as temporary:
                plan = source.prepare(repo, row, Path(temporary))
                if row.get("script") == "research/s1_mature_20261018/train.sh":
                    from research.s1_mature_20261018.qualification_preflight import check

                    identity = canonical({"commit": row["commit"], "inputs": row["inputs"]})
                    if identity not in qualification:
                        qualification[identity] = check(repo, row, plan, Path(temporary))
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise AdmissionRefused(f"offline source packaging refused {job}: {exc}") from exc
        results[job] = {
            "commit": plan["commit"],
            "tree": plan["tree"],
            "files": len(plan["files"]),
            "inventory_sha256": hashlib.sha256(canonical(plan["files"]).encode()).hexdigest(),
            "fallback_unpacked_size": plan["fallback_unpacked_size"],
            "fallback_size": plan["fallback_size"],
        }
        if row.get("script") == "research/s1_mature_20261018/train.sh":
            results[job]["qualification"] = qualification[identity]
    return {"status": "PASS", "builder": "research.colab_cli.source.prepare", "jobs": results}


def staging_preflight(
    rows: Iterable[dict[str, Any]],
    packaging: dict[str, Any],
    *,
    release_bytes_per_second: float = 1024**2,
) -> dict[str, Any]:
    """Refuse staging forecasts beyond the transfer cap before ledger reservation.

    Contents uses half the slow dispatch-02 observation (13 KB/s). Release
    delivery reserves 1 MiB/s with a twofold margin plus five seconds per GET.
    These are admission floors, not a claim of measured VM release goodput;
    independent bounded transfers still fail closed when the network is slower.
    """
    from research.colab_cli.inputs import PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS
    from research.colab_cli.transfers import UPLOAD_TOTAL_TIMEOUT_SECONDS

    if not math.isfinite(release_bytes_per_second) or release_bytes_per_second <= 0:
        raise AdmissionRefused("release staging rate must be finite and positive")
    jobs = {}
    for row in rows:
        inputs = row.get("inputs", [])
        if not any("url" in item for item in inputs):
            continue
        uploads, fetched, fetches = 0, 0, 0
        for item in inputs:
            if "url" in item:
                size = item.get("size_bytes")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise AdmissionRefused("release inputs need a positive pinned size")
                fetched += size
                fetches += 1
            else:
                size = Path(item["path"]).stat().st_size
                if size > 100_000:
                    raise AdmissionRefused("S1 Contents evidence exceeds 100 KB")
                uploads += size
        fallback = packaging["jobs"][row["job_id"]]["fallback_size"]
        # Include the bounded source fallback plus helper/bootstrap metadata.
        uploaded = uploads + fallback + 100_000
        contents_seconds = 30 + uploaded / 6_500
        fetch_seconds = 2 * fetched / release_bytes_per_second + 5 * fetches
        if (
            contents_seconds >= UPLOAD_TOTAL_TIMEOUT_SECONDS
            or fetch_seconds >= PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS
        ):
            raise AdmissionRefused("estimated input staging exceeds its transfer cap")
        jobs[row["job_id"]] = {
            "release_bytes": fetched,
            "contents_bytes_upper_bound": uploaded,
            "contents_seconds_upper_bound": contents_seconds,
            "release_seconds_upper_bound": fetch_seconds,
            "estimated_seconds_upper_bound": contents_seconds + fetch_seconds,
            "contents_cap_seconds": UPLOAD_TOTAL_TIMEOUT_SECONDS,
            "release_cap_seconds": PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS,
            "cap_seconds": UPLOAD_TOTAL_TIMEOUT_SECONDS + PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS,
        }
    return {
        "status": "PASS",
        "basis": "2x release 1 MiB/s floor; 2x observed dispatch-02 Contents 13 KB/s; setup slack",
        "jobs": jobs,
    }


def delivery_preflight(rows: list[dict[str, Any]], packaging: dict[str, Any]) -> dict[str, Any]:
    """Hash all release bytes on the Mac before any account or ledger operation."""
    from research.colab_cli import inputs

    public = [item for row in rows for item in row.get("inputs", []) if "url" in item]
    if not public:
        return {"status": "NOT_REQUIRED"}
    staging_preflight(rows, packaging)
    try:
        verified = inputs.preflight(public)
    except (inputs.InputError, OSError, ValueError) as exc:
        raise AdmissionRefused(f"release input preflight refused: {exc}") from exc
    # Small objects measure request latency, already covered by five seconds per
    # GET in staging_preflight, rather than sustained transfer goodput.
    rate = min(
        [1024**2] + [item["bytes_per_second"] for item in verified if item["size_bytes"] >= 1024**2]
    )
    return {
        "status": "PASS",
        "verified_inputs": verified,
        "staging": staging_preflight(rows, packaging, release_bytes_per_second=rate),
    }


def require_approved_preregistration(path: Path) -> dict[str, str]:
    text = path.read_text()
    # The original DRAFT header remains part of the immutable historical bytes
    # after ratification, so only a prospective approval line can admit it.
    marker = f"**Status: {APPROVAL}.**"
    approved = any(
        line.strip() == marker or line.strip().startswith(marker + " ")
        for line in text.splitlines()[:12]
    )
    if not approved:
        raise AdmissionRefused("preregistration lacks the prospective Claude approval status line")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def require_approved_parallel_amendment(path: Path) -> dict[str, str]:
    """Bind the prospective operational amendment required for parallel lanes."""
    text = path.read_text()
    marker = f"**Status: {PARALLEL_APPROVAL}.**"
    approved = any(
        line.strip() == marker or line.strip().startswith(marker + " ")
        for line in text.splitlines()[:12]
    )
    if not approved:
        raise AdmissionRefused("parallel amendment lacks the prospective approval status line")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _lanes(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_LANES:
        raise AdmissionRefused(f"lanes must be an integer from 1 through {MAX_LANES}")
    return value


def liabilities(manifest_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Reserve the ten primary attempts only; reruns are conditional at launch."""
    rows = list(manifest_rows)
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        job, requirements = row.get("job_id"), row.get("requirements", {})
        arm = str(row.get("env", {}).get("S1_ARM", ""))
        est, timeout = requirements.get("est_minutes"), row.get("timeout_minutes")
        expected = 30 if arm == "n3" else 68 if arm == "n1" else None
        expected_timeout = 60 if arm == "n3" else 136 if arm == "n1" else None
        if not isinstance(job, str) or est != expected or timeout != expected_timeout:
            raise AdmissionRefused(f"manifest {job!r} violates registered {arm} timing")
        hours = (float(timeout) + 5.0) / 60.0
        output[job] = {
            "units": hours * 2.0,
            "gpu_allocation_hours": hours,
            "mac_worker_hours": 0.0,
            "elapsed_hours": hours,
        }
    if len(output) != 10:
        raise AdmissionRefused("S1 requires exactly ten registered training jobs")
    return output


def _input(value: str) -> manifests.ExternalInput:
    name, path, dest = value.split(":", 2)
    return manifests.ExternalInput(name, Path(path), dest)


def priced_mac_bound(path: Path, *, lanes: int = 1) -> dict[str, Any]:
    """Read the completed real-entrypoint pricing receipt, never a hand estimate."""
    lanes = _lanes(lanes)
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise AdmissionRefused("Mac preflight/evaluation pricing receipt is not PASS")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        raise AdmissionRefused("pricing receipt lacks bound training and evaluation evidence")
    training = evidence.get("training")
    evaluation = evidence.get("evaluation")
    for name, item in (("training", training), ("evaluation", evaluation)):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            raise AdmissionRefused(f"pricing {name} evidence is unbound")
        target = Path(item["path"])
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise AdmissionRefused(f"pricing {name} evidence changed or is absent")
    training_value, evaluation_value = json.loads(Path(training["path"]).read_text()), json.loads(
        Path(evaluation["path"]).read_text()
    )
    if training_value.get("status") != "PASS" or training_value.get("counter_admission") != "PASS":
        raise AdmissionRefused("real M3-C training preflight has not passed counter admission")
    expected_training_sources = (
        "research/redesign_m3c_20261008/train_m3c.py",
        "research/s1_mature_20261018/train.py",
    )
    training_sources = training_value.get("identity", {}).get("source_manifest", {})
    if any(
        training_sources.get(relative) != sha256(ROOT / relative)
        for relative in expected_training_sources
    ):
        raise AdmissionRefused(
            "counter preflight trainer/adapter source identity changed or is absent"
        )
    if (
        evaluation_value.get("schema") != "s1-eval-timing/v1"
        or evaluation_value.get("status") != "completed"
        or int(evaluation_value.get("episode_count", 0)) < 10
    ):
        raise AdmissionRefused("disjoint >=10 episode evaluation timing fixture has not passed")
    for relative, digest in evaluation_value["identity"]["source_manifest"].items():
        if sha256(ROOT / relative) != digest:
            raise AdmissionRefused("evaluation timing source changed")
    bounds = value.get("complete_upper_bound")
    if not isinstance(bounds, dict):
        raise AdmissionRefused("pricing receipt lacks a complete_upper_bound")
    try:
        mac = float(bounds["mac_worker_hours"])
        elapsed = float(value["mac_upper_bound"]["elapsed_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdmissionRefused("pricing receipt lacks finite Mac and elapsed bounds") from exc
    if not math.isfinite(mac) or not math.isfinite(elapsed) or mac < 0 or elapsed < 0:
        raise AdmissionRefused("pricing upper bounds must be nonnegative")
    priced_lanes = value.get("parallel_lanes", 1)
    if priced_lanes != lanes:
        raise AdmissionRefused("pricing receipt lane count differs from the prepared plan")
    # The final-only transport receipt binds the current source, measured final
    # bundle sizes, and a verified L4-to-Mac acceptance transfer.  It contains
    # no full trainer state and cannot be used as an exact-resume qualification.
    transport = value.get("final_artifact_transport")
    if not isinstance(transport, dict) or transport.get("status") != "PASS":
        raise AdmissionRefused("final-artifact transport is unqualified; S1 admission is refused")
    try:
        qualified = final_artifacts.transport_bound(transport)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise AdmissionRefused(str(exc)) from exc
    return {
        "cli_acceptance_receipt": qualified["cli_acceptance_receipt"],
        "mac_worker_hours": mac,
        "elapsed_hours": elapsed,
        "lanes": lanes,
        "elapsed_schedule": value.get("elapsed_schedule"),
        "per_arm_gpu_hours": {
            arm: (30 if arm == "n3" else 68) / 60 + item["transport_seconds_upper_bound"] / 3600
            for arm, item in qualified["qualified_arms"].items()
        },
    }


def prepare(args: argparse.Namespace) -> Path:
    repo, out = args.repo.resolve(), args.out.resolve()
    lanes = _lanes(getattr(args, "lanes", 1))
    published_ref = getattr(args, "published_ref", "origin/main")
    commit = published_commit(repo, published_ref)
    prereg = require_approved_preregistration(args.preregistration.resolve())
    amendment = None
    if lanes > 1:
        amendment_path = getattr(args, "parallel_amendment", None)
        if amendment_path is None:
            raise AdmissionRefused("parallel lanes require an approved hash-bound amendment")
        amendment = require_approved_parallel_amendment(Path(amendment_path).resolve())
    seed_value, audit_value = json.loads(args.seed_bank.read_text()), json.loads(
        args.seed_audit.read_text()
    )
    if seed_value.get("seeds") != list(bank.TRAINING_SEEDS):
        raise AdmissionRefused("uploaded seed bank is not the registered S1 seed block")
    world_value = json.loads(args.world_bank.read_text())
    if audit_value.get("pass") is not True or audit_value.get(
        "world_bank_semantic_sha256"
    ) != world_value.get("allocation_semantic_sha256"):
        raise AdmissionRefused("seed audit does not pass for the selected S1 world bank")
    from research.s1_mature_20261018.review_jobs import verified_inputs

    frozen_inputs = verified_inputs()
    parsed_inputs = [_input(item) for item in args.input] if args.input else list(frozen_inputs)
    expected_inputs = {item.dest: item.as_manifest()["sha256"] for item in frozen_inputs}
    if {item.dest: item.as_manifest()["sha256"] for item in parsed_inputs} != expected_inputs:
        raise AdmissionRefused("uploaded inputs differ from the frozen qualified input inventory")
    seed_upload = [item for item in parsed_inputs if item.dest == "seed-bank.json"]
    if len(seed_upload) != 1 or seed_upload[0].path.resolve() != args.seed_bank.resolve():
        raise AdmissionRefused(
            "supervisor seed-bank input must be the exact registered seed-bank bytes"
        )
    local_inputs = {
        item.dest: {"path": str(item.path.resolve()), "sha256": item.as_manifest()["sha256"]}
        for item in parsed_inputs
    }
    if getattr(args, "input_delivery", "contents") == "public-release":
        parsed_inputs = [item.from_release() for item in parsed_inputs]
    mac_bound = priced_mac_bound(args.pricing.resolve(), lanes=lanes)
    acceptance = mac_bound["cli_acceptance_receipt"]
    parsed_inputs.append(
        manifests.ExternalInput(
            "cli-acceptance-result", Path(acceptance["path"]), "cli-acceptance-result.json"
        )
    )
    rows = manifests.make_manifests(
        seed_bank=args.seed_bank.resolve(),
        published_commit=commit,
        registration_status=APPROVAL,
        owner_release=APPROVAL,
        output_root=args.artifact_root.resolve() / "training",
        inputs=parsed_inputs,
    )
    liability = liabilities(rows)
    liability["mac-final-evaluation"] = {
        "units": 0.0,
        "gpu_allocation_hours": 0.0,
        "mac_worker_hours": mac_bound["mac_worker_hours"],
        "elapsed_hours": mac_bound["elapsed_hours"],
    }
    training = {key: sum(item[key] for item in liability.values()) for key in ledger.DIMENSIONS}
    # The timeout reservation must dominate the measured final-only transport
    # bound for each run; otherwise the supposedly inclusive estimate is false.
    for row in rows:
        arm = row["env"]["S1_ARM"]
        # Cleanup is reserved separately; the actual job timeout must cover
        # training and the final decision artifact transfer.
        if float(row["timeout_minutes"]) / 60 < mac_bound["per_arm_gpu_hours"][arm]:
            raise AdmissionRefused(
                "registered job timeout does not cover training plus final-artifact transport"
            )
    # Acceptance is a conservative two GPU/allocation hours inside its four-unit
    # allowance. Final export slack is already reserved per job above.
    training_schedule = ledger.parallel_elapsed_schedule(
        {row["job_id"]: liability[row["job_id"]] for row in rows}, lanes
    )
    complete = {
        "units": training["units"] + ACCEPTANCE_UNITS,
        "gpu_allocation_hours": training["gpu_allocation_hours"] + 2.0,
        "mac_worker_hours": mac_bound["mac_worker_hours"],
        "elapsed_hours": training_schedule["makespan_hours"] + 2.0 + mac_bound["elapsed_hours"],
    }
    if complete["units"] > 48:
        raise AdmissionRefused("training plus acceptance upper bound exceeds usable units")
    if complete["gpu_allocation_hours"] > 20:
        raise AdmissionRefused("training upper bound exceeds usable GPU allocation hours")
    if complete["mac_worker_hours"] > 19.2 or complete["elapsed_hours"] > 38.4:
        raise AdmissionRefused(
            "complete preflight/training/evaluation upper bound exceeds a usable cap"
        )
    if out.exists() or args.manifest_dir.exists():
        raise AdmissionRefused("plan or manifest directory already exists")
    packaging = packaging_preflight(repo, rows)
    staging = staging_preflight(rows, packaging)
    for row in rows:
        job_staging = staging["jobs"].get(row["job_id"])
        if job_staging is not None:
            required = (
                mac_bound["per_arm_gpu_hours"][row["env"]["S1_ARM"]]
                + job_staging["estimated_seconds_upper_bound"] / 3600
            )
            if required > row["timeout_minutes"] / 60:
                raise AdmissionRefused("training, staging and final transport exceed job timeout")
    args.manifest_dir.mkdir(parents=True)
    written: list[dict[str, str]] = []
    for row in rows:
        target = args.manifest_dir / f"{row['job_id']}.json"
        write_once(target, row)
        written.append({"job_id": row["job_id"], "path": str(target), "sha256": sha256(target)})
    mac_manifest = args.manifest_dir / "mac-final-evaluation.json"
    write_once(
        mac_manifest,
        {
            "job_id": "mac-final-evaluation",
            "kind": "mac",
            "workers": 2,
            "timeout_minutes": mac_bound["elapsed_hours"] * 60,
        },
    )
    plan = {
        "schema": SCHEMA,
        "status": "PREPARED_NOT_DISPATCHED",
        "commit": commit,
        "published_ref": published_ref,
        "preregistration": prereg,
        **({"parallel_amendment": amendment} if amendment is not None else {}),
        "source_closure": source_closure(repo, commit),
        "source_packaging": packaging,
        "input_staging": staging,
        "local_inputs": local_inputs,
        "world_bank": {"path": str(args.world_bank.resolve()), "sha256": sha256(args.world_bank)},
        "seed_audit": {"path": str(args.seed_audit.resolve()), "sha256": sha256(args.seed_audit)},
        "manifests": written,
        "liabilities": liability,
        "mac_manifest": {"path": str(mac_manifest), "sha256": sha256(mac_manifest)},
        "acceptance": {
            "units_included_maximum": ACCEPTANCE_UNITS,
            "review_required_schema": ledger.REVIEW_SCHEMA,
        },
        "pricing_receipt": {"path": str(args.pricing.resolve()), "sha256": sha256(args.pricing)},
        "complete_upper_bound": complete,
        "elapsed_schedule": training_schedule,
        "run_rules": {
            "lanes": lanes,
            "serialized": lanes == 1,
            "operational_reruns_pre_funded": False,
            "operational_rerun_admission": "full-worst-case-within-all-usable-caps-at-launch",
            "operational_reruns_max": {"n3": 1, "n1": 1},
            "operational_rerun_mode": "from-scratch-identical-seed-and-recipe-before-efficacy-look",
            "no_exact_resume": True,
            "no_intermediate_checkpoint_selection": True,
            "no_outcome_analysis_before_full_bank_and_independent_audit": True,
            "mac_workers_max": 2,
            "single_thread_per_worker": True,
        },
    }
    write_once(out, plan)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="bind approved source and render create-only manifests")
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--published-ref", default="origin/main")
    p.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "research/s1_mature_20261018/PREREGISTRATION.md",
    )
    p.add_argument("--seed-bank", type=Path, required=True)
    p.add_argument("--world-bank", type=Path, required=True)
    p.add_argument("--seed-audit", type=Path, required=True)
    p.add_argument("--pricing", type=Path, required=True)
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--parallel-amendment", type=Path)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input", action="append", default=[], metavar="NAME:PATH:DEST")
    p.add_argument(
        "--input-delivery", choices=("contents", "public-release"), default="public-release"
    )
    check = sub.add_parser("preflight-inputs", help="verify release assets without account access")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--repo", type=Path, default=ROOT)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight-inputs":
        from research.s1_mature_20261018 import run

        plan = json.loads(args.plan.read_text())
        rows = [json.loads(path.read_text()) for _, path in run._manifest_paths(plan)]
        result = delivery_preflight(rows, packaging_preflight(args.repo, rows))
        write_once(args.out, result)
        print(args.out)
        return 0
    if args.command == "prepare":
        print(prepare(args))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
