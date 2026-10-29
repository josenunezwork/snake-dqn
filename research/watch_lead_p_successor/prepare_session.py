"""Create a deliberately undispatched timing packet; re-pin after transport lands."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from research.bl_served_search.exports import digest
from research.watch_lead_p_successor.delivery import B8_BYTES, B8_URL
from research.watch_lead_p_successor.prepare import audit_bank, write
from research.watch_lead_p_successor.units import (
    CHECKPOINTS,
    allocation,
    binding,
    completion_attempt,
    empty_ledger,
    read_ledger,
)
from tools.sandbox.registry import load_registry, world_seed

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/watch_lead_p_successor/session_job.sh"
STUDY = "p-watch-lead-successor-timing-20261029-v1"
SMOKE_STUDY = "p-watch-lead-successor-smoke-20261029-v1"
TRANSPORT_BRANCH = "astra-colab-transport-20261029"


def fresh_bank() -> dict:
    return allocation(STUDY, 2, engineering=True)


def collision_audit(bank: dict) -> dict:
    excluded = set(load_registry()["excluded_seeds"])
    for study, count in (
        ("t34-watch-lead-1028", 36),
        ("p-successor-profile-v1", 4),
        ("p-successor-profile-v2", 4),
        ("p-watch-lead-successor-v1", 32),
        (SMOKE_STUDY, 1),
    ):
        excluded.update(world_seed(study, i) for i in range(count))
    # Existing metadata-only audit also checks TRAIN01 monitor/nomination coverage.
    previous = audit_bank()
    for start, count in ((400_000_000, 288), (500_000_000, 768)):
        import hashlib

        for i in range(start, start + count):
            raw = hashlib.sha256(f"a100-dev/sandbox-a-train01/{i}".encode()).digest()[:8]
            excluded.add(int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF)
    if len(set(bank["seeds"])) != len(bank["seeds"]) or set(bank["seeds"]) & excluded:
        raise ValueError("reject entire colliding engineering allocation")
    return {
        "bank": bank,
        "collisions": [],
        "excluded_count": len(excluded),
        "registry_sha256": previous["registry_sha256"],
        "coverage": "public registry, P0-35, profile-v1/v2, successor proposed32, "
        "smoke, TRAIN01 monitor/nomination; opaque historical inventories "
        "NOT claimed exhaustive",
        "closed_payloads_read": False,
        "engineering_seeds_burned_for_science": bank["seeds"],
    }


def source_inventory(pin: str) -> dict[str, str]:
    paths = json.loads((ROOT / "research/watch_lead_p/runtime-sources.json").read_text())
    paths += [
        str(p.relative_to(ROOT))
        for p in (ROOT / "research/watch_lead_p_successor").iterdir()
        if p.suffix in (".py", ".sh")
    ]
    paths = sorted(set(paths))
    hashes = {}
    for name in paths:
        import hashlib

        data = subprocess.check_output(["git", "show", f"{pin}:{name}"], cwd=ROOT)
        checksum = hashlib.sha256(data).hexdigest()
        if digest(ROOT / name) != checksum:
            raise ValueError(f"uncommitted source drift {name}")
        hashes[name] = checksum
    return hashes


def prepare(
    packet: Path,
    pin: str,
    transport_pin: str | None = None,
    *,
    ledger: Path | None = None,
    prior: Path | None = None,
    attempt: int = 0,
) -> None:
    if not packet.is_absolute():
        raise ValueError("absolute fresh packet path required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != pin:
        raise ValueError("current committed source pin required")
    source_tree = subprocess.check_output(
        ["git", "rev-parse", f"{pin}:research/colab_cli"], cwd=ROOT, text=True
    ).strip()
    target_tree = None
    if transport_pin is not None:
        import re

        if not re.fullmatch(r"[0-9a-f]{40}", transport_pin):
            raise ValueError("full published transport commit required")
        target_tree = subprocess.check_output(
            ["git", "rev-parse", f"{transport_pin}:research/colab_cli"], cwd=ROOT, text=True
        ).strip()
        if source_tree != target_tree:
            raise ValueError("integrate the final transport before re-pinning a packet")
    sources = source_inventory(pin)
    bank = fresh_bank()
    audit = collision_audit(bank)
    bound = binding(bank, sources)
    verified_ledger = empty_ledger(bound)
    if type(attempt) is not int or attempt not in (0, 1, 2):
        raise ValueError("original plus at most two manual completions")
    if attempt:
        if ledger is None or prior is None:
            raise ValueError("completion needs Mac-verified ledger and mechanical stop record")
        verified_ledger = json.loads(ledger.read_text())
        completion_attempt(attempt, json.loads(prior.read_text()), verified_ledger, bound, [0, 1])
    elif ledger is not None or prior is not None:
        raise ValueError("original attempt must use empty first ledger")
    read_ledger(verified_ledger, bound)
    packet.mkdir(parents=True, exist_ok=False)
    if attempt:
        write(
            packet / "completion-eligibility.json",
            {
                "attempt": attempt,
                "ledger_path": str(ledger),
                "ledger_sha256": digest(ledger),
                "prior_path": str(prior),
                "prior_sha256": digest(prior),
                "fresh_native_probe_required": True,
                "automatic_retry": False,
            },
        )
    inputs = packet / "inputs"
    inputs.mkdir()
    for name, value in (
        ("allocation.json", bank),
        ("binding.json", bound),
        ("sources.json", sources),
        ("verified-ledger.json", verified_ledger),
    ):
        write(inputs / name, value)
    checks = ROOT.parent / "snake-dqn/saved_snakes"
    for source, name in (
        ("b8_s3_t7_20261016.pth", "b8-s3.pth"),
        ("frp3_m3_s12_u60000_20261005.pth", "frp3-s12.pth"),
    ):
        if digest(checks / source) != CHECKPOINTS[name]:
            raise ValueError("checkpoint staging identity differs")
        shutil.copyfile(checks / source, inputs / name)
    write(packet / "collision-audit.json", audit)
    write(
        packet / "checkpoint-inventory.json",
        [
            {
                "path": str(inputs / name),
                "sha256": checksum,
                "size_bytes": (inputs / name).stat().st_size,
            }
            for name, checksum in CHECKPOINTS.items()
        ],
    )
    local_inputs = [
        {"path": str(p), "dest": p.name, "sha256": digest(p)}
        for p in sorted(inputs.iterdir())
        if p.name != "b8-s3.pth"
    ]
    # Existing supported supervisor HTTPS route; exact complete B8 bytes, not a new service/upload.
    manifest = {
        "job_id": f"p-successor-full-timing-dev01-a{attempt}",
        "commit": pin,
        "script": SCRIPT,
        "device": "cpu",
        "source_paths": sorted(sources),
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": 360},
        "timeout_minutes": 720,
        "output_dir": str(packet / "full/output"),
        "requirements_file": "research/watch_lead_p/requirements.txt",
        "env": {
            "SNAKE_DQN_DEVICE": "cpu",
            "P_SUCCESSOR_STAGE": "full",
            "P_SUCCESSOR_WORK_SECONDS": "42540",
            "P_SUCCESSOR_WORKERS": "1",
            "P_SUCCESSOR_INDICES": "0,1",
            "COLAB_CLI_REQUIREMENTS": "research/watch_lead_p/requirements.txt",
        },
        "inputs": local_inputs
        + [
            {
                "url": B8_URL,
                "dest": "b8-s3.pth",
                "sha256": CHECKPOINTS["b8-s3.pth"],
                "size_bytes": B8_BYTES,
            }
        ],
        "max_export_bytes": 128 * 1024 * 1024,
        "lease_seconds": 180,
        "resume": {
            "unit_kind": "paired-Watch6-session",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "budget": {
            "rate_units_per_hour": 1.54,
            "unit_cap": 18.48,
            "reserve_units": 3.696,
            "other_assignments": {},
        },
        # Shared producer export proves source/checkpoint/ledger construction. Dynamic
        # session exports are also scanned; COMPLETE runtime exit is required separately.
        "required_exports": ["successor/entrypoint.json", "successor-startup.json"],
        "canary": {
            "receipt": {
                "path": str(packet / "NOT-YET-QUALIFIED-family-receipt.json"),
                "sha256": "0" * 64,
            }
        },
    }
    write(packet / "full-session-manifest.TEMPLATE.json", manifest)
    # Native proof is limited to 30min; same entrypoint and full input inventory.
    probe = json.loads(json.dumps(manifest))
    probe.update(
        job_id=f"p-successor-transport-dev01-a{attempt}",
        timeout_minutes=30,
        output_dir=str(packet / "transport/output"),
        canary={"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT},
        required_exports=[
            "successor/entrypoint.json",
            "successor/transport-smoke.json",
            "successor-startup.json",
        ],
    )
    probe["requirements"]["est_minutes"] = 15
    probe["env"].update(
        P_SUCCESSOR_STAGE="transport",
        SNAKE_COLAB_CANARY="1",
        SNAKE_COLAB_CANARY_PACKAGE="research/watch_lead_p_successor",
    )
    probe["budget"].update(unit_cap=0.78, reserve_units=0.156)
    write(packet / "transport-manifest.TEMPLATE.json", probe)
    write(
        packet / "transport-target.json",
        {
            "required_branch": TRANSPORT_BRANCH,
            "producer": "t34-sol/astra-impl-colab-transport",
            "source_transport_tree": source_tree,
            "target_transport_tree": target_tree,
            "target_transport_commit": transport_pin,
            "status": (
                "PINNED_PENDING_INDEPENDENT_TRANSPORT_APPROVAL"
                if transport_pin
                else "REPIN_REQUIRED_AFTER_TRANSPORT_LANDS"
            ),
            "required_independent_review_status": "approved by transport qualification checker",
        },
    )
    write(
        packet / "envelope.json",
        {
            "scope": "engineering runaway guards ONLY; not measured full-session "
            "completion forecast",
            "base_timeout_minutes": 360,
            "effective_timeout_minutes": 720,
            "full_work_deadline_seconds": 42540,
            "finalization_reserve_seconds": 660,
            "full_session_estimate": None,
            "rate_planning_units_per_hour": 1.54,
            "base_units_informational": 9.24,
            "effective_units_informational": 18.48,
            "base_reserve_informational": 1.848,
            "effective_reserve_informational": 3.696,
            "attempt": attempt,
            "original_plus_two_manual_completion_hours": 36,
            "native_probes_per_original_or_completion": 1,
            "native_probe_series_hours": 1.5,
            "complete_series_hours_including_fresh_probes": 37.5,
            "complete_series_units_informational": 57.75,
            "changed_ledger_requires_fresh_probe": True,
            "unit_caps_are_not_current_owner_budget_policy": True,
            "workers": 1,
            "concurrency_qualification": "serial baseline only; sustained w>1 remains needed",
            "tail_support": "freeze both seeds; record all roots, long>=1000 and late>=5000; "
            "report unsupported honestly, never replace",
            "transfer_acceptance": "<=105s transfer (>=45s margin below unchanged150s), <=300s "
            "staging; HTTPS time included in total completion pricing",
            "L": None,
            "q_residual": None,
            "full_session_c": None,
        },
    )
    (packet / "DISPATCH-NOT-RUN.txt").write_text(
        "DRY-RUN PACKET ONLY. NO DISPATCH WAS RUN OR AUTHORIZED BY THIS PACKET.\n"
        "Templates deliberately cannot dispatch: missing family receipt and final transport pin.\n"
        "1. Wait for astra-impl-colab-transport published commit/tree and independent approval.\n"
        "2. In THIS worktree integrate that commit (review conflicts); "
        "commit and push new source.\n"
        "3. Regenerate a FRESH packet using prepare_session --transport-pin FINAL_SHA; bind final "
        "runtime commit, transport tree,\n"
        "   qualified checker/record, all source hashes, checkpoint "
        "inputs, bank, ledger, manifests and seal.\n"
        "4. Argon cross-review; operator authorizes/runs native 30min transport probe first.\n"
        "   It proves short runtime/transport only. Require actual <=105s "
        "transfer and <=300s staging.\n"
        "5. After verified stop/disappearance, bind its exact family "
        "receipt to a fresh full-session packet;\n"
        "   validate via qualified supervisor and ops dry-run. No old "
        "profile adapter: it rejects long jobs.\n"
        "6. Operator separately dispatches full natural-start H6000 three-"
        "arm engineering sessions on L4 CPU.\n"
        "   12h cap is a doubled runaway guard, not a completion "
        "prediction. Publish each whole session.\n"
        "Each manual completion uses --attempt 1 or 2 --ledger VERIFIED --prior STOP_RECORD.\n"
        "Changing the ledger requires a FRESH 30min native family probe "
        "with exactly those inputs;\n"
        "never reuse the original receipt or relax the native input SHA/size checks.\n"
        "7. Price measured full-session tails and sustained concurrency "
        "before scientific registration.\n"
        "No scientific manifest, registration, closed-bank look, automatic"
        " retry, or serving authority.\n"
    )
    write(
        packet / "seal.json",
        {str(p.relative_to(packet)): digest(p) for p in sorted(packet.rglob("*")) if p.is_file()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--transport-pin", help="final published transport producer commit")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    prepare(
        args.packet,
        args.pin,
        args.transport_pin,
        ledger=args.ledger,
        prior=args.prior,
        attempt=args.attempt,
    )


if __name__ == "__main__":
    main()
