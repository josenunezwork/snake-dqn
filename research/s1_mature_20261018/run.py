"""Serialized post-acceptance S1 dispatcher.

This is intentionally an operator program, never an autonomous background loop:
one invocation handles one immutable prepared plan and stops on the first failed
receipt.  It invokes only the Colab supervisor wrapper, never ``colab`` itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from research.s1_mature_20261018 import admission, audit, bank, evaluate, ledger, rerun


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _verify_plan(path: Path, repo: Path) -> dict[str, Any]:
    plan = _read(path)
    if plan.get("schema") != admission.SCHEMA or plan.get("status") != "PREPARED_NOT_DISPATCHED":
        raise admission.AdmissionRefused("not an immutable prepared S1 admission plan")
    if admission.published_commit(repo) != plan.get("commit"):
        raise admission.AdmissionRefused("published source changed since S1 plan preparation")
    if admission.source_closure(repo, plan["commit"]) != plan.get("source_closure"):
        raise admission.AdmissionRefused("complete source closure differs from the prepared plan")
    for name in ("preregistration", "pricing_receipt", "world_bank", "seed_audit"):
        item = plan.get(name)
        if not isinstance(item, dict) or admission.sha256(Path(item["path"])) != item.get("sha256"):
            raise admission.AdmissionRefused(f"prepared {name} evidence changed or is absent")
    return plan


def _history(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, raw = value.split("=", 1)
        if not name or name in result:
            raise admission.AdmissionRefused(
                "history entries must have unique NAME=PATH identities"
            )
        result[name] = Path(raw).resolve(strict=True)
    if not {"q1", "a2", "a3"} <= set(result):
        raise admission.AdmissionRefused("Q1, A2, and A3 historical evidence is required")
    return result


def _manifest_paths(plan: dict[str, Any]) -> list[tuple[str, Path]]:
    rows = plan.get("manifests")
    if not isinstance(rows, list) or len(rows) != 10:
        raise admission.AdmissionRefused("prepared plan lacks exactly ten manifests")
    answer: list[tuple[str, Path]] = []
    for row in rows:
        target = Path(row["path"])
        if admission.sha256(target) != row["sha256"]:
            raise admission.AdmissionRefused(f"manifest changed after preparation: {target}")
        answer.append((str(row["job_id"]), target))
    return answer


def _snapshot(path: Path, *, colab: Path, account_config: Path) -> Path:
    """Capture fresh account state through the supervisor's guarded CLI wrapper."""
    from research.colab_cli.supervisor import CLI

    if path.exists():
        raise admission.AdmissionRefused(f"refusing to replace account snapshot: {path}")
    cli = CLI(str(colab), account_config)
    usage, sessions = cli.usage(), cli.sessions()
    value = {
        "schema": "s1-account-snapshot/v1",
        "captured_at": time.time(),
        "balance_units": usage["balance"],
        "usage_rate_units_per_hour": usage["rate"],
        "active_assignments": int(usage["assignments"]),
        "assignments_known": True,
        "sessions": sessions,
    }
    if len(sessions) != value["active_assignments"]:
        raise admission.AdmissionRefused(
            "usage/session assignment counts disagree; account state is unknown"
        )
    admission.write_once(path, value)
    return path


def _safe_extract_final(archive: Path, destination: Path) -> tuple[Path | None, Path | None, Path]:
    """Extract one supervisor-verified immutable final archive without path escape."""
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise admission.AdmissionRefused("unsafe final archive member")
        if destination.exists():
            expected_files = {member.name for member in members if member.isfile()}
            actual_files = {
                str(path.relative_to(destination))
                for path in destination.rglob("*")
                if path.is_file()
            }
            if actual_files != expected_files:
                raise admission.AdmissionRefused("existing recovered final inventory differs")
            for member in members:
                if member.isfile():
                    existing = destination / member.name
                    source = tar.extractfile(member)
                    if (
                        existing.is_symlink()
                        or existing.stat().st_size != member.size
                        or hashlib.file_digest(source, "sha256").hexdigest()
                        != admission.sha256(existing)
                    ):
                        raise admission.AdmissionRefused("existing recovered final bytes changed")
        else:
            destination.mkdir(parents=True)
            tar.extractall(destination, filter="data")
    finals = list(destination.rglob("final.pth"))
    targets = list(destination.rglob("target.pth"))
    summaries = list(destination.rglob("summary.json"))
    outcomes = list(destination.rglob("outcome.json"))
    if len(summaries) != 1 or not (
        (len(finals) == 1 and len(targets) == 1 and not outcomes)
        or (not finals and not targets and len(outcomes) == 1)
    ):
        raise admission.AdmissionRefused(
            "export needs one summary and either final or safety outcome"
        )
    return finals[0] if finals else None, targets[0] if targets else None, summaries[0]


def _reassemble_final(index_path: Path, exports: Path, archive: Path) -> Path:
    index = _read(index_path)
    if index.get("schema") != "s1-chunked-export/v1":
        raise admission.AdmissionRefused("invalid or reused final chunk index")
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise admission.AdmissionRefused("final chunk index is empty")
    total = 0
    for item in chunks:
        name = item.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise admission.AdmissionRefused("unsafe final export chunk name")
        piece = exports / name
        if not piece.is_file() or admission.sha256(piece) != item.get("sha256"):
            raise admission.AdmissionRefused("final export chunk is missing or changed")
        total += piece.stat().st_size
    if total + index_path.stat().st_size > admission.manifests.MAX_EXPORT_BYTES:
        raise admission.AdmissionRefused("final export exceeds aggregate size cap")
    if archive.exists():
        if archive.stat().st_size != total or admission.sha256(archive) != index.get(
            "archive_sha256"
        ):
            raise admission.AdmissionRefused("existing recovered archive changed")
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as output:
        for item in chunks:
            name, digest = item.get("path"), item.get("sha256")
            piece = exports / str(name)
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not piece.is_file()
                or admission.sha256(piece) != digest
            ):
                raise admission.AdmissionRefused("final export chunk is missing or changed")
            with piece.open("rb") as source:
                import shutil

                shutil.copyfileobj(source, output)
    if admission.sha256(archive) != index.get("archive_sha256"):
        raise admission.AdmissionRefused("reassembled final archive digest differs")
    return archive


def _provenance(
    summary: Path, checkpoint: Path, target: Path, arm: str, seed: int, out: Path
) -> Path:
    """Copy only a runtime-produced, fully bound provenance record."""
    source = summary.with_name("s1-training-provenance.json")
    value = _read(source)
    required = {
        "schema": "s1-training-provenance/v1",
        "arm": arm,
        "training_seed": seed,
        "endpoint": 8_000_000,
        "clean_exit": True,
        "tripwire_triggered": False,
        "checkpoint_sha256": admission.sha256(checkpoint),
        "target_checkpoint_sha256": admission.sha256(target),
        "summary_sha256": admission.sha256(summary),
        "trainer_source_sha256": admission.sha256(
            admission.ROOT / "research/redesign_m3c_20261008/train_m3c.py"
        ),
    }
    if any(value.get(k) != v for k, v in required.items()):
        raise admission.AdmissionRefused(
            "runtime provenance does not bind the recovered final and raw summary"
        )
    if (
        not isinstance(value.get("runtime_parity_receipt_sha256"), str)
        or len(value["runtime_parity_receipt_sha256"]) != 64
    ):
        raise admission.AdmissionRefused("runtime provenance lacks parity receipt identity")
    parity = summary.with_name("runtime-parity.json")
    if (
        admission.sha256(parity) != value["runtime_parity_receipt_sha256"]
        or _read(parity).get("pass") is not True
    ):
        raise admission.AdmissionRefused("runtime parity proof is absent, changed or failed")
    if value.get("raw_summary") != _read(summary):
        raise admission.AdmissionRefused("provenance raw summary differs from saved bytes")
    evaluate._validate_provenance(value, arm, seed, checkpoint)
    if out.exists():
        if _read(out) != value:
            raise admission.AdmissionRefused("existing recovered provenance changed")
    else:
        admission.write_once(out, value)
    return out


def _recover_training(manifest_path: Path, root: Path) -> tuple[dict[str, Any], dict | None]:
    """Recover a settled job before any later allocation; classify safety outcomes."""
    manifest = _read(manifest_path)
    job, arm, seed = (
        manifest["job_id"],
        manifest["env"]["S1_ARM"].upper(),
        int(manifest["env"]["S1_SEED"]),
    )
    exports = Path(manifest["output_dir"]) / "exports"
    finals = root / "finals"
    archive = _reassemble_final(exports / "s1-final.index.json", exports, finals / f"{job}.tar.gz")
    checkpoint, target, summary = _safe_extract_final(archive, finals / job)
    record = {
        "summary": str(summary),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "target_checkpoint": str(target) if target else None,
    }
    if checkpoint is not None:
        assert target is not None
        proof = _provenance(
            summary, checkpoint, target, arm, seed, finals / f"{job}.provenance.json"
        )
        record["provenance"] = str(proof)
        return record, None
    outcome = _read(summary.with_name("outcome.json"))
    if outcome.get("arm") != arm or outcome.get("training_seed") != seed:
        raise admission.AdmissionRefused("safety outcome differs from reserved job identity")
    receipt = Path(manifest["output_dir"]) / "receipt.json"
    outcome["evidence"] = {
        name: {"path": str(path), "sha256": admission.sha256(path)}
        for name, path in (
            ("summary", summary),
            ("log", summary.with_name("log.jsonl")),
            ("supervisor_receipt", receipt),
        )
    }
    audit.validated_halts({"training_outcomes": [outcome]}, admission.ROOT)
    return record, outcome


def _evaluate_all(plan: dict[str, Any], root: Path, bank_path: Path) -> Path:
    """Only after all ten final exports are settled, write all 15 final shards."""
    inputs = _read(Path(plan["manifests"][0]["path"]))["inputs"]
    b8 = next(Path(item["path"]) for item in inputs if item["dest"] == "B8_s3_final.pth")
    shards: list[dict[str, Any]] = []
    recovered = _read(root / "recovered-training.json")
    outcomes = json.loads((root / "training-outcomes.json").read_text())
    halted = audit.validated_halts({"training_outcomes": outcomes}, admission.ROOT)
    for seed in bank.TRAINING_SEEDS:
        for arm in ("N1", "N3"):
            if arm in halted:
                continue
            job = f"s1-mature-{arm.lower()}-{seed}"
            checkpoint, provenance = Path(recovered[job]["checkpoint"]), Path(
                recovered[job]["provenance"]
            )
            shards.append(
                _read(
                    evaluate.run(
                        root=root,
                        bank_path=bank_path,
                        checkpoint=checkpoint,
                        arm=arm,
                        seed=seed,
                        workers=2,
                        provenance_path=provenance,
                    )
                )
            )
        shards.append(
            _read(
                evaluate.run(
                    root=root, bank_path=bank_path, checkpoint=b8, arm="B8", seed=seed, workers=2
                )
            )
        )
    input_path = root / "analysis-input.json"
    admission.write_once(
        input_path,
        {
            "schema": "s1-analysis-input/v1",
            "bank": _read(bank_path),
            "bank_audit": _read(bank_path.with_name("SEED_AUDIT.json")),
            "shards": shards,
            "training_outcomes": outcomes,
        },
    )
    audit_path = root / "independent-audit.json"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(admission.ROOT / "research/s1_mature_20261018/audit.py"),
            "--input",
            str(input_path),
            "--out",
            str(audit_path),
            "--repo",
            str(admission.ROOT),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.s1_mature_20261018.analyze",
            str(input_path),
            "--independent-audit",
            str(audit_path),
            "--out",
            str(root / "analysis.json"),
        ],
        check=True,
    )
    return root / "analysis.json"


def _finish_evaluation(
    args: argparse.Namespace,
    plan: dict[str, Any],
    worker_ledger: ledger.Ledger,
    recovered: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> None:
    """Run the registered final look with the same watchdog after either dispatch path."""
    admission.write_once(args.artifact_root / "recovered-training.json", recovered)
    admission.write_once(args.artifact_root / "training-outcomes.json", outcomes)
    # The preceding L4 session makes the old pre-job snapshot stale.  Capture
    # account state again before the Mac liability becomes outstanding.
    _snapshot(
        (args.snapshots / "before-mac-final-evaluation.json").resolve(),
        colab=args.colab,
        account_config=args.account_config,
    )
    worker_ledger.record_account_snapshot(
        (args.snapshots / "before-mac-final-evaluation.json").resolve()
    )
    mac = plan["mac_manifest"]
    mac_path = Path(mac["path"])
    if admission.sha256(mac_path) != mac["sha256"]:
        raise admission.AdmissionRefused("Mac evaluation manifest changed after preparation")
    worker_ledger.reserve(
        "mac-final-evaluation", plan["liabilities"]["mac-final-evaluation"], mac_path
    )
    started = time.monotonic()
    bound = plan["liabilities"]["mac-final-evaluation"]
    timeout = min(float(bound["elapsed_hours"]), float(bound["mac_worker_hours"]) / 2.0) * 3600 - 15
    code = (
        "from pathlib import Path; "
        "from research.s1_mature_20261018.run import _read,_evaluate_all; "
        f"p=_read(Path({str(args.plan)!r})); "
        f"_evaluate_all(p,Path({str(args.artifact_root)!r}),Path(p['world_bank']['path']))"
    )
    command = [sys.executable, "-c", code]
    child = subprocess.Popen(command, start_new_session=True)
    try:
        child.wait(timeout=max(1, timeout))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise admission.AdmissionRefused(
            "Mac evaluation exceeded registered deadline; reservation remains outstanding"
        )
    if child.returncode:
        raise admission.AdmissionRefused(
            "Mac evaluation/audit child failed; reservation remains outstanding"
        )
    elapsed = (time.monotonic() - started) / 3600
    mac_receipt = args.artifact_root.resolve() / "mac-final-evaluation-receipt.json"
    admission.write_once(
        mac_receipt,
        {
            "status": "PASS",
            "job_id": "mac-final-evaluation",
            "actual": {
                "units": 0.0,
                "gpu_allocation_hours": 0.0,
                "mac_worker_hours": 2 * elapsed,
                "elapsed_hours": elapsed,
            },
        },
    )
    worker_ledger.settle_mac("mac-final-evaluation", mac_receipt)


def dispatch(args: argparse.Namespace) -> None:
    plan = _verify_plan(args.plan.resolve(), args.repo.resolve())
    expected_root = Path(_read(Path(plan["manifests"][0]["path"]))["output_dir"]).parents[1]
    if args.artifact_root.resolve() != expected_root:
        raise admission.AdmissionRefused(
            "artifact root differs from the prepared manifest output root"
        )
    reviewed_supervisor = (args.repo / "research/colab_cli/launch.sh").resolve()
    if args.supervisor.resolve() != reviewed_supervisor:
        raise admission.AdmissionRefused(
            "dispatcher requires the reviewed supervisor wrapper in its pinned repo"
        )
    paths = _manifest_paths(plan)
    worker_ledger = ledger.Ledger(args.ledger.resolve())
    worker_ledger.initialize(
        acceptance_review_path=args.acceptance_review.resolve(),
        account_snapshot_path=_snapshot(
            (args.snapshots / "initial.json").resolve(),
            colab=args.colab,
            account_config=args.account_config,
        ),
        history_refs=_history(args.history),
        planned_liabilities=plan["liabilities"],
    )
    recovered, outcomes, halted = {}, [], set()
    for job_id, manifest_path in paths:
        if _read(manifest_path)["env"]["S1_ARM"].upper() in halted:
            continue  # Unspent liabilities stay reserved in the immutable original plan.
        # Each capture must be independently fresh and prove no foreign active assignment.
        snapshot = _snapshot(
            (args.snapshots / f"before-{job_id}.json").resolve(),
            colab=args.colab,
            account_config=args.account_config,
        )
        worker_ledger.record_account_snapshot(snapshot)
        worker_ledger.reserve(job_id, plan["liabilities"][job_id], manifest_path)
        state = (args.state_root / job_id).resolve()
        state.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.supervisor),
            "--state-dir",
            str(state),
            "--colab",
            str(args.colab),
            "run",
            str(manifest_path),
        ]
        # No retry: any nonzero exit leaves the ledger reservation outstanding.
        subprocess.run(command, check=True)
        output = Path(_read(manifest_path)["output_dir"])
        worker_ledger.settle(job_id, output / "receipt.json")
        record, outcome = _recover_training(manifest_path, args.artifact_root.resolve())
        recovered[job_id] = record
        if outcome is not None:
            outcomes.append(outcome)
            halted.add(outcome["arm"])
    _finish_evaluation(args, plan, worker_ledger, recovered, outcomes)


def _retry_record(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Read one create-only manual rerun authorization; it cannot authorize a new retry."""
    value = _read(path)
    if (
        value.get("schema") != "s1-manual-operational-rerun/v1"
        or value.get("status") != "PREPARED_NOT_LAUNCHED"
    ):
        raise admission.AdmissionRefused("not a prepared manual operational rerun record")
    if value.get("no_resume") is not True or value.get("no_efficacy_look") is not True:
        raise admission.AdmissionRefused(
            "manual rerun record weakens registered recovery restrictions"
        )
    for name in (
        "source_manifest",
        "retry_manifest",
        "supervisor_receipt",
        "operational_failure_review",
    ):
        item = value.get(name)
        if not isinstance(item, dict) or admission.sha256(Path(item["path"])) != item.get("sha256"):
            raise admission.AdmissionRefused(f"manual rerun {name} evidence changed or is absent")
    source = value["source_manifest"]
    if source["sha256"] not in {row["sha256"] for row in plan["manifests"]}:
        raise admission.AdmissionRefused(
            "manual rerun source is not a manifest from this immutable plan"
        )
    state = Path(value.get("state_dir", ""))
    if not state.is_absolute() or state.exists():
        raise admission.AdmissionRefused(
            "manual rerun state directory must be a fresh absolute path"
        )
    return value


def continue_dispatch(args: argparse.Namespace) -> None:
    """Explicitly continue after one prepared operational rerun; never retries automatically."""
    plan = _verify_plan(args.plan.resolve(), args.repo.resolve())
    expected_root = Path(_read(Path(plan["manifests"][0]["path"]))["output_dir"]).parents[1]
    if args.artifact_root.resolve() != expected_root:
        raise admission.AdmissionRefused(
            "artifact root differs from the prepared manifest output root"
        )
    if args.supervisor.resolve() != (args.repo / "research/colab_cli/launch.sh").resolve():
        raise admission.AdmissionRefused(
            "dispatcher requires the reviewed supervisor wrapper in its pinned repo"
        )
    rerun._no_efficacy_artifacts(args.artifact_root.resolve())
    record = _retry_record(args.retry_record.resolve(), plan)
    worker_ledger = ledger.Ledger(args.ledger.resolve())
    source_id, retry_id = record["source_job_id"], record["retry_job_id"]
    retry_manifest = Path(record["retry_manifest"]["path"])
    worker_ledger.status()  # Revalidate all settled receipts before any new allocation.
    state = json.loads(args.ledger.read_text())
    source_entry = state.get("settled", {}).get(source_id, {})
    if (
        source_entry.get("operational_failure_review", {}).get("sha256")
        != record["operational_failure_review"]["sha256"]
    ):
        raise admission.AdmissionRefused("manual rerun record is not the reconciled ledger failure")
    if retry_id not in state.get("planned", {}):
        raise admission.AdmissionRefused("manual rerun liability was already consumed")
    paths = _manifest_paths(plan)
    replacements: dict[str, Path] = {}
    for entry in state.get("settled", {}).values():
        if entry.get("retry_of"):
            evidence = entry.get("source_manifest", {})
            target = Path(evidence.get("path", ""))
            if not target.is_file() or admission.sha256(target) != evidence.get("sha256"):
                raise admission.AdmissionRefused("settled rerun manifest changed or is absent")
            replacements[entry["retry_of"]] = target
    prior: dict[str, tuple[dict[str, Any], dict | None]] = {}
    for job_id, manifest_path in paths:
        entry = state.get("settled", {}).get(job_id)
        if entry is None or job_id == source_id:
            continue
        if entry.get("operational_failure_review"):
            if job_id not in replacements:
                raise admission.AdmissionRefused("an earlier operational loss has no settled rerun")
            manifest_path = replacements[job_id]
        prior[job_id] = _recover_training(manifest_path, args.artifact_root.resolve())
    rerun._no_efficacy_artifacts(args.artifact_root.resolve())
    if rerun._verified_final_export(_read(Path(record["source_manifest"]["path"]))):
        raise admission.AdmissionRefused("verified final appeared after rerun preparation")
    # A fresh account snapshot and the ledger's atomic arm-level reservation are
    # intentionally immediately adjacent to the one operator-requested launch.
    snapshot = _snapshot(
        (args.snapshots / f"before-{retry_id}.json").resolve(),
        colab=args.colab,
        account_config=args.account_config,
    )
    worker_ledger.record_account_snapshot(snapshot)
    worker_ledger.reserve_manual_retry(source_id, retry_id, retry_manifest)
    command = [
        str(args.supervisor),
        "--state-dir",
        record["state_dir"],
        "--colab",
        str(args.colab),
        "run",
        str(retry_manifest),
    ]
    subprocess.run(command, check=True)
    worker_ledger.settle(retry_id, Path(_read(retry_manifest)["output_dir"]) / "receipt.json")
    recovered, outcomes, halted = {}, [], set()
    for job_id, manifest_path in paths:
        if job_id == source_id:
            record_item, outcome = _recover_training(retry_manifest, args.artifact_root.resolve())
        elif job_id in prior:
            record_item, outcome = prior[job_id]
        else:
            if _read(manifest_path)["env"]["S1_ARM"].upper() in halted:
                continue
            snapshot = _snapshot(
                (args.snapshots / f"before-{job_id}.json").resolve(),
                colab=args.colab,
                account_config=args.account_config,
            )
            worker_ledger.record_account_snapshot(snapshot)
            worker_ledger.reserve(job_id, plan["liabilities"][job_id], manifest_path)
            job_state = (args.state_root / job_id).resolve()
            if job_state.exists():
                raise admission.AdmissionRefused("remaining job state directory already exists")
            subprocess.run(
                [
                    str(args.supervisor),
                    "--state-dir",
                    str(job_state),
                    "--colab",
                    str(args.colab),
                    "run",
                    str(manifest_path),
                ],
                check=True,
            )
            worker_ledger.settle(job_id, Path(_read(manifest_path)["output_dir"]) / "receipt.json")
            record_item, outcome = _recover_training(manifest_path, args.artifact_root.resolve())
        recovered[job_id] = record_item
        if outcome is not None:
            outcomes.append(outcome)
            halted.add(outcome["arm"])
    _finish_evaluation(args, plan, worker_ledger, recovered, outcomes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--continue", dest="continue_dispatch", action="store_true")
    parser.add_argument("--retry-record", type=Path)
    parser.add_argument("--repo", type=Path, default=admission.ROOT)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--acceptance-review", type=Path, required=True)
    parser.add_argument(
        "--snapshots",
        type=Path,
        required=True,
        help="new create-only live account snapshot directory",
    )
    parser.add_argument("--history", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--supervisor",
        type=Path,
        required=True,
        help="research/colab_cli/launch.sh from the reviewed published checkout",
    )
    parser.add_argument("--colab", type=Path, required=True)
    parser.add_argument(
        "--account-config",
        type=Path,
        required=True,
        help="dedicated supervisor account config, never a job lane config",
    )
    args = parser.parse_args(argv)
    if args.evaluate_only:
        plan = _read(args.plan)
        _evaluate_all(plan, args.artifact_root.resolve(), Path(plan["world_bank"]["path"]))
        return 0
    if args.continue_dispatch:
        if args.retry_record is None:
            parser.error("--continue requires --retry-record")
        continue_dispatch(args)
        return 0
    dispatch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
