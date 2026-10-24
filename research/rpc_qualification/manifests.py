"""Prepare immutable attempt-2 C1 packs (v2 faults) and dispatch priced waves."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import shlex
import subprocess
import tarfile
import uuid
from pathlib import Path
from typing import Any

from research.bl_labels.exports import _verify, publish_json
from research.bl_labels.f1_manifests import source_paths, triplet
from research.bl_labels.f1_ruler import bank_digest, load_bank
from research.bl_labels.manifests import _pool_inputs
from research.bl_upload import assert_upload_budget, convert_public_checkpoints

from .admission import (
    AUTHORITY_SHA256,
    BANK_SHA256,
    FAULT_AUTHORITY_SHA256,
    FAULT_DEFINITION,
    SCHEMA,
    SCRIPT,
    TRANSPORT_PIN,
    WORKER_PIN,
    envelope,
    require_prerequisite,
    sha,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
DEFAULT_BANK = ARTIFACTS / "bl-labels-f0/f1-pack-09/f1-root-bank.json.xz"
# Attempt 2 keeps its fix-pin qualification apart from main's registry so a
# lane result never de-qualifies the transport every other lane dispatches.
LANE_REGISTRY = ARTIFACTS / "ops/l4-label-lane-qualified.json"
PUBLICATION_BRANCH = "ops-l4-faults-v2-20261024"
HOLD_RESOLUTION = ARTIFACTS / "big-lever/exp/fable-20261024/JUDGMENT_L4_V2_HOLD_RESOLUTION.md"
HOLD_RESOLUTION_SHA256 = "adddd711494b8d855d70425e028d3606531d7b8527e4138ca8419baf217299ed"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, timeout=30).strip()


def frozen_source(out: Path) -> dict[str, Any]:
    """Ship only the frozen F1 static closure, never the mutable F1 checkout."""
    tracked = git("ls-tree", "-r", "--name-only", WORKER_PIN).splitlines()
    allowed = source_paths()
    names = [
        name
        for name in tracked
        if any(name == entry or name.startswith(entry + "/") for entry in allowed)
    ]
    raw = subprocess.check_output(
        ["git", "archive", "--format=tar", WORKER_PIN, "--", *names], cwd=REPO, timeout=60
    )
    # Deterministic XZ keeps source delivery inside the qualified 150 s
    # upload forecast without changing one byte of the frozen worker closure.
    with out.open("xb") as stream:
        stream.write(lzma.compress(raw, preset=6))
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        inventory = {
            member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for member in archive.getmembers()
            if member.isfile()
        }
    result = {"commit": WORKER_PIN, "archive_sha256": sha(out), "files": inventory}
    publish_json(out.with_name("frozen-source.json"), result)
    return result


def prepare(out: Path, bank: Path = DEFAULT_BANK) -> Path:
    """No allocation, result read or pending-acceptance waiver occurs in planning."""
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("pack preparation requires clean committed source")
    if out.exists():
        raise ValueError("pack root must be fresh")
    if sha(HOLD_RESOLUTION) != HOLD_RESOLUTION_SHA256:
        raise ValueError("hold-resolution authority differs from the frozen ruling")
    if bank_digest(bank) != BANK_SHA256:
        raise ValueError("foreign frozen F1 bank")
    load_bank(bank)
    commit = git("rev-parse", "HEAD")
    transport = git("rev-parse", f"{commit}:research/colab_cli")
    if transport != git("rev-parse", f"{TRANSPORT_PIN}:research/colab_cli"):
        raise ValueError("pack transport tree differs from the v2 fix pin")
    proof = REPO / "research/rpc_qualification/FAKE_CLI_PROOF_20261024_V2_05.json"
    from .checker import validate_fake_proof

    validate_fake_proof(proof, REPO, commit)
    out.mkdir(parents=True)
    archive = out / "f1-source.tar"
    source = frozen_source(archive)
    for suffix in ("", ".sha256", ".complete"):
        with (out / (bank.name + suffix)).open("xb") as stream:
            stream.write(Path(str(bank) + suffix).read_bytes())
    checkpoints = _pool_inputs(
        REPO.parent / "snake-dqn/saved_snakes/b8_s3_t7_20261016.pth",
        REPO.parent / "snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth",
    )
    if len(checkpoints) != 6:
        raise ValueError("all six frozen checkpoint inputs are required")
    token = uuid.uuid4().hex[:12]
    jobs = []
    for platform, mode, index in [
        *[("l4", "loop", i) for i in range(1, 4)],
        *[("cpu", "loop", i) for i in range(1, 4)],
        ("cpu", "injection", 1),
    ]:
        name = f"rq-{token}-{platform}-{mode}-{index}"
        folder = out / name
        folder.mkdir()
        namespace = f"rpc-qual/{commit}/{name}"
        allocation = name
        limits = envelope(platform, mode)
        ledger = folder / "verified-ledger.json"
        from .loop import empty_ledger

        # Runtime and Mac checker use the same frozen allocation identity.
        publish_json(
            ledger,
            empty_ledger(namespace, allocation, BANK_SHA256, WORKER_PIN, source["archive_sha256"]),
        )
        inputs = triplet(out / bank.name, "inputs/" + bank.name)
        inputs += triplet(ledger, "inputs/verified-ledger.json")
        inputs += [{"path": str(archive), "sha256": sha(archive), "dest": "inputs/f1-source.tar"}]
        inputs += [
            {
                "path": str(out / "frozen-source.json"),
                "sha256": sha(out / "frozen-source.json"),
                "dest": "inputs/frozen-source.json",
            }
        ]
        inputs += checkpoints
        from research.colab_cli import injection_manifest
        from research.rpc_qualification.injection import locator_for_unit, unit_rpc_sha256

        from .loop import unit_id

        # The runtime derives each injected unit's request ID from its unit
        # identity (loop.py descriptor); namespace and allocation are fixed
        # here, so the exact IDs are known and frozen before dispatch.
        entries = (
            [
                {
                    "injection_id": f"response-loss-{k + 1}",
                    "type": "post_send_response_loss",
                    "failure_type": "post_send_response_loss",
                    "command": "exec",
                    "request_id": request_id,
                    "request_sha256": unit_rpc_sha256(
                        locator=locator_for_unit(unit),
                        namespace=namespace,
                        allocation=allocation,
                        index=k,
                        unit_id=unit,
                        request_id=request_id,
                    ),
                    "max_calls": 1,
                }
                for k in range(3)
                for unit in [unit_id(namespace, allocation, k)]
                for request_id in [
                    hashlib.sha256(("rpc-injection/" + unit).encode()).hexdigest()[:32]
                ]
            ]
            if mode == "injection"
            else []
        )
        injections = folder / "injection-manifest.json"
        injections_sha256 = injection_manifest.write(
            injections,
            name,
            entries,
            purpose=f"attempt-2 {platform} {mode} deliberate faults (none for loops)",
        )
        rq = {
            "schema": SCHEMA,
            "authority_sha256": AUTHORITY_SHA256,
            "worker_pin": WORKER_PIN,
            "bank_sha256": BANK_SHA256,
            "frozen_source_sha256": source["archive_sha256"],
            "mode": mode,
            "injection_request_sha256": {
                entry["request_id"]: entry["request_sha256"]
                for entry in entries
                if "request_sha256" in entry
            },
            "platform": platform,
            "namespace": namespace,
            "allocation": allocation,
            "workers": 12 if platform == "l4" else 2,
            "loop_seconds": 3600 if mode == "loop" else 0,
            "envelope": limits,
            "prerequisite": None,
            "fault_authority_sha256": FAULT_AUTHORITY_SHA256,
            "fault_definition": FAULT_DEFINITION,
            "injection_manifest": {
                "path": str(injections),
                "sha256": injections_sha256,
                "request_ids": [entry["request_id"] for entry in entries],
            },
            "optional": platform == "cpu" and mode == "loop",
        }
        manifest = {
            "job_id": name,
            "commit": commit,
            "script": SCRIPT,
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_labels/requirements.txt",
                "RQ_DEVICE": "cpu",
                "PYTHONUNBUFFERED": "1",
                "RQ_MODE": mode,
                "RQ_PLATFORM": platform,
                "RQ_NAMESPACE": namespace,
                "RQ_ALLOCATION": allocation,
                "RQ_WORKERS": str(rq["workers"]),
                "RQ_LOOP_SECONDS": str(rq["loop_seconds"]),
                "RQ_SOURCE_COMMIT": WORKER_PIN,
                "RQ_SOURCE_CLOSURE_SHA256": source["archive_sha256"],
                "RQ_SOURCE_MANIFEST_SHA256": sha(out / "frozen-source.json"),
                **{
                    key: "1"
                    for key in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMBA_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
            },
            "requirements": {
                "gpu": "L4" if platform == "l4" else "cpu",
                "allowed_gpus": ["L4"] if platform == "l4" else ["cpu"],
                "min_vcpu": rq["workers"],
                "est_minutes": 75 if mode == "loop" else 15,
            },
            "timeout_minutes": limits["effective"]["timeout_minutes"],
            "output_dir": str(folder / "output"),
            "inputs": inputs,
            "budget": {
                "unit_cap": limits["effective"]["unit_cap"],
                "reserve_units": limits["effective"]["reserve_units"],
                "rate_units_per_hour": 1.54 if platform == "l4" else 0.08,
                "other_assignments": {},
            },
            "lease_seconds": 180,
            "max_export_bytes": 128 * 1024**2,
            "source_paths": sorted(set(source_paths() + ["research/rpc_qualification"])),
            "required_exports": ["runtime-result.json", "verified-ledger.json"],
            "resume": {
                "unit_kind": "f1-root-repeat",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "rq": rq,
        }
        manifest = convert_public_checkpoints(manifest, repo=REPO)
        publish_json(folder / "upload-budget.json", assert_upload_budget(manifest, repo=REPO))
        template = folder / "template.json"
        publish_json(template, manifest)
        output = folder / "output"
        jobs.append(
            {
                "job_id": name,
                "platform": platform,
                "mode": mode,
                "bank_sha256": BANK_SHA256,
                "worker_pin": WORKER_PIN,
                "namespace": namespace,
                "allocation": allocation,
                "template": str(template),
                "manifest": str(folder / "manifest.json"),
                "state_dir": str(folder / "state"),
                "output_dir": str(output),
                "receipt": str(output / "receipt.json"),
                "runtime": str(output / "exports/runtime-result.json"),
                "units_dir": str(output / "exports/units"),
                "ledger": str(folder / "published-ledger.json"),
                "injection_manifest": str(injections),
                "injection_manifest_sha256": injections_sha256,
                "optional": platform == "cpu" and mode == "loop",
            }
        )
    plan = {
        "schema": SCHEMA,
        "source_commit": commit,
        "transport_pin": TRANSPORT_PIN,
        "hold_resolution": {"path": str(HOLD_RESOLUTION), "sha256": HOLD_RESOLUTION_SHA256},
        "bank_sha256": BANK_SHA256,
        "worker_pin": WORKER_PIN,
        "authority_sha256": AUTHORITY_SHA256,
        "jobs": jobs,
        "fake_proof": {"path": str(proof), "sha256": sha(proof)},
        "publication_ref": f"refs/heads/{PUBLICATION_BRANCH}",
        "fault_definition": FAULT_DEFINITION,
        "fault_authority_sha256": FAULT_AUTHORITY_SHA256,
        "transport_tree": transport,
        "lane_registry": str(LANE_REGISTRY),
        "publication_ledger": f"docs/research/results/rpc-qualification-{token}-ledger.json",
        "repo": str(REPO),
        "package_cap_units": 16.02,
    }
    publish_json(out / "plan.json", plan)
    commands = [
        "# Attempt 2 (JUDGMENT_L4_FAULT_DEFINITION_20261024 §4 steps 3-4). Order:",
        "# 1. final-only prerequisite at the fix pin (see the handoff); set ACCEPTANCE to its",
        "#    acceptance-result.json. 2. L4 wave (3 sessions concurrently). 3. injection wave.",
        "# 4. optional CPU loop wave only if account slots are free. No automatic retries.",
    ]
    for wave_name in ("l4", "injection", "cpu"):
        commands.append(
            shlex.join(
                [
                    str(REPO.parent / "snake-dqn/venv/bin/python"),
                    "-m",
                    "research.rpc_qualification.manifests",
                    "wave",
                    "--plan",
                    str(out / "plan.json"),
                    "--wave",
                    wave_name,
                    "--acceptance",
                ]
            )
            + ' "$ACCEPTANCE"'
        )
    (out / "DISPATCH.txt").write_text("\n".join(commands) + "\n")
    publication = out / "publication.json"
    post = [
        shlex.join(
            [
                str(REPO.parent / "snake-dqn/venv/bin/python"),
                "-m",
                "research.rpc_qualification.checker",
                "publish",
                "--plan",
                str(out / "plan.json"),
                "--out",
                str(publication),
            ]
        ),
        shlex.join(
            [
                "git",
                "add",
                "--",
                plan["publication_ledger"],
                plan["publication_ledger"] + ".sha256",
                plan["publication_ledger"] + ".complete",
            ]
        ),
        shlex.join(
            [
                "git",
                "commit",
                "-m",
                "Publish RPC qualification hash ledger before one look",
                "-m",
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
            ]
        ),
        f"git push origin HEAD:{PUBLICATION_BRANCH}",
        shlex.join(
            [
                str(REPO.parent / "snake-dqn/venv/bin/python"),
                "-m",
                "research.rpc_qualification.checker",
                "verdict",
                "--plan",
                str(out / "plan.json"),
                "--publication",
                str(publication),
                "--out",
                str(out / "verdict.json"),
            ]
        ),
    ]
    (out / "PUBLISH_BEFORE_READ.txt").write_text("\n".join(post) + "\n")
    return out / "plan.json"


def dry_run(plan_path: Path) -> dict[str, Any]:
    """Check only frozen local inputs, without inventory calls or dispatch writes."""
    from .admission import require_review
    from .checker import validate_fake_proof

    _verify(plan_path)
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("hold_resolution")
        != {"path": str(HOLD_RESOLUTION), "sha256": HOLD_RESOLUTION_SHA256}
        or sha(HOLD_RESOLUTION) != HOLD_RESOLUTION_SHA256
    ):
        raise ValueError("pack hold-resolution authority differs")
    if git("rev-parse", "HEAD") != plan["source_commit"]:
        raise ValueError("source changed since pack preparation")
    if git("rev-parse", "HEAD:research/colab_cli") != plan["transport_tree"]:
        raise ValueError("transport tree differs from pack")
    proof = Path(plan["fake_proof"]["path"])
    if sha(proof) != plan["fake_proof"]["sha256"]:
        raise ValueError("fake-CLI proof differs from pack")
    validate_fake_proof(proof, REPO, plan["source_commit"])
    blockers = ["final-only prerequisite has not been run for this package"]
    for job in plan["jobs"]:
        template_path = Path(job["template"])
        _verify(template_path)
        template = json.loads(template_path.read_text())
        if template["commit"] != plan["source_commit"]:
            raise ValueError("template pin differs from pack")
        for row in template["inputs"]:
            if sha(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"template input changed: {row['dest']}")
        injection = Path(job["injection_manifest"])
        _verify(injection)
        if sha(injection) != job["injection_manifest_sha256"]:
            raise ValueError("injection manifest changed")
        from research.bl_labels.l4_faults_v2 import validate_injection_manifest

        try:
            validate_injection_manifest(json.loads(injection.read_text()), job_id=job["job_id"])
        except ValueError as error:
            blockers.append(f"{job['job_id']}: {error}")
        if any(Path(job[key]).exists() for key in ("manifest", "state_dir", "output_dir")):
            raise ValueError("pack has already consumed a dispatch slot")
    try:
        require_review(repo=REPO, package_commit=plan["source_commit"])
    except (OSError, ValueError) as error:
        blockers.append(str(error))
    return {
        "status": "HOLD_OFFLINE",
        "integrity": "PASS",
        "dispatch_ready": False,
        "plan_sha256": sha(plan_path),
        "jobs": len(plan["jobs"]),
        "transport_tree": plan["transport_tree"],
        "blockers": blockers,
        "provider_calls": 0,
        "dispatches": 0,
        "self_declared_blockers": [],
    }


def wave(plan_path: Path, wave_name: str, acceptance: Path, record: Path = LANE_REGISTRY) -> None:
    """Serialize account admissions, then leave up to three runtimes concurrent."""
    plan = json.loads(plan_path.read_text())
    _verify(plan_path)
    if git("rev-parse", "HEAD") != plan["source_commit"]:
        raise ValueError("source changed since pack preparation")
    from .admission import require_review
    from .checker import validate_fake_proof

    require_review(repo=REPO, package_commit=plan["source_commit"])
    proof = Path(plan["fake_proof"]["path"])
    if sha(proof) != plan["fake_proof"]["sha256"]:
        raise ValueError("fake-CLI proof differs from pack")
    validate_fake_proof(proof, REPO, plan["source_commit"])
    require_prerequisite(acceptance, REPO, "l4" if wave_name == "l4" else "cpu")
    selected = [
        job
        for job in plan["jobs"]
        if (
            job["mode"] == "injection"
            if wave_name == "injection"
            else job["mode"] == "loop" and job["platform"] == wave_name
        )
    ]
    if wave_name in {"cpu", "injection"}:
        for job in plan["jobs"]:
            if job["platform"] == "l4" and not Path(job["manifest"]).is_file():
                raise ValueError("dispatch the required L4 wave before CPU sessions")
    # Refuse a short account before consuming any session in the fixed wave.
    # Dispatch rechecks the complete priced account map under its normal lock.
    from research.colab_cli.supervisor import CLI

    inventory_cli = CLI(str(Path.home() / ".local/bin/colab"), plan_path.parent / "inventory.json")
    if 3 - len(inventory_cli.sessions()) < len(selected):
        raise ValueError(
            "insufficient free account slots for this whole wave; no sessions consumed"
        )
    for job in selected:
        if any(Path(job[key]).exists() for key in ("manifest", "state_dir", "output_dir")):
            raise ValueError("session manifest/state/output reused")
    for job in selected:
        manifest_path = Path(job["manifest"])
        template = json.loads(Path(job["template"]).read_text())
        template["rq"]["prerequisite"] = {
            "path": str(acceptance.resolve()),
            "sha256": sha(acceptance),
        }
        from .admission import validate_manifest

        validate_manifest(template, REPO)
        # Refuse used sessions before any dispatch side effects; never regenerate.
        if (
            manifest_path.exists()
            or Path(job["state_dir"]).exists()
            or Path(job["output_dir"]).exists()
        ):
            raise ValueError("session manifest/state/output reused")
        publish_json(manifest_path, template)
        result = subprocess.run(
            [
                str(REPO / "tools/ops/dispatch.sh"),
                job["job_id"],
                str(manifest_path),
                "--broker",
                str(REPO),
                "--cwd",
                str(REPO),
                "--state-dir",
                job["state_dir"],
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "OPS_QUALIFICATION_RECORD": str(record)},
        )
        print(result.stdout, end="", flush=True)
        if "allocation=failed" in result.stdout or "allocation=timeout" in result.stdout:
            raise ValueError("wave stopped after original session allocation failure; no retries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    offline = sub.add_parser("dry-run")
    offline.add_argument("--plan", type=Path, required=True)
    dispatch = sub.add_parser("wave")
    dispatch.add_argument("--plan", type=Path, required=True)
    dispatch.add_argument("--wave", choices=["l4", "cpu", "injection"], required=True)
    dispatch.add_argument("--acceptance", type=Path, required=True)
    dispatch.add_argument("--qualification-record", type=Path, default=LANE_REGISTRY)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.out.resolve(), args.bank))
    elif args.action == "dry-run":
        print(json.dumps(dry_run(args.plan.resolve()), indent=2, sort_keys=True))
    else:
        wave(
            args.plan.resolve(),
            args.wave,
            args.acceptance.resolve(),
            args.qualification_record.resolve(),
        )


if __name__ == "__main__":
    main()
