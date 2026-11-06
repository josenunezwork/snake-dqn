"""Prepare three fresh 44-session L1-OS product packets; never allocate or dispatch.

G4 is a CPU host here. Each packet requires its own same-input H=120 canary receipt.
Only explicit new-bank receipt/checkpoint paths may be supplied; this module never
scans artifacts or opens screen, confirmation, or gate outcomes. Retry evidence is
path/size/hash and stopped mechanical lineage, never replay content or performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

from research.colab_cli import canary
from research.t34_exit_b8 import public_inputs
from research.watch_replay.prepare import release_bundle
from research.watch_replay.run import ROOT, allocation, completed, identity
from research.watch_replay.run_l1_os import checkpoint_candidates
from research.watch_replay.screen_l1_os import pack_checkpoint_resume
from research.watch_replay.seed_audit import (
    L1_OS_BANK_COUNT,
    L1_OS_BANK_NAMESPACES,
    L1_OS_CANARY_NAMESPACES,
    l1_os_bank_disjointness,
)
from research.watch_replay.source import source_paths
from research.watch_replay.uploads import CHECKPOINTS, METADATA, check_uploads, pack_inputs
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs

SCRIPT = "research/watch_replay/job_l1_os.sh"
G4_RATE = 10.05
WORKERS = 44
CANARY_HORIZON = 120
SESSION_SECONDS = 3 * 3600
MAX_EXPORT_BYTES = 2 * 1024**3
MAX_SESSION_BYTES = 40 * 1024**2
MAX_CHECKPOINT_BYTES = 128 * 1024**2
MECHANICAL_ERRORS = (
    "TIMEOUT",
    "LEASE",
    "TRANSFER",
    "UPLOAD",
    "STAGING",
    "INTERRUPT",
    "RUNTIME_LOST",
)


def forecast() -> dict:
    """Projection from G4 48-worker progress timing; not terminal completion proof."""
    central = 2.29 * 60 + 25
    high = math.ceil(2.73 * 60 + 25)
    return {
        "packets": 3,
        "sessions_per_packet": L1_OS_BANK_COUNT,
        "workers": WORKERS,
        "horizon": 6000,
        "device": "cpu",
        "gpu": "G4",
        "rate_units_per_hour": G4_RATE,
        "central_minutes": central,
        "high_minutes": high,
        "effective_timeout_minutes": 2 * high,
        "central_total_units": 3 * central / 60 * G4_RATE,
        "base_unit_cap": high / 60 * G4_RATE,
        "effective_unit_cap": 2 * high / 60 * G4_RATE,
        "basis": (
            "G4 48W progress calibration: 19 clean-start traces from frame46 to >=5900; "
            "H6000 extrapolated mean2.289h/max2.726h, rounded2.29h/2.73h plus25min"
        ),
        "qualification": (
            "progress-derived forecast; incomplete receipt and unverified stop "
            "do not prove completion; "
            "new packet canaries required; no dispatch"
        ),
        "max_session_bytes": MAX_SESSION_BYTES,
        "max_checkpoint_bytes": MAX_CHECKPOINT_BYTES,
        "max_export_bytes": MAX_EXPORT_BYTES,
    }


def _check_pin(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("requires full published current HEAD commit")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
        raise ValueError("commit must be the current HEAD")
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if not any(row.strip().startswith("origin/") for row in refs.splitlines()):
        raise ValueError("requires commit published on origin")
    paths = source_paths()
    if subprocess.check_output(["git", "diff", commit, "--", *paths], cwd=ROOT):
        raise ValueError("working executable bytes differ from source pin")
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=ROOT
    ):
        raise ValueError("untracked executable source is not in commit")


def _lineage(contract: dict, ledger: Path | None, receipts: list[Path]) -> set[int]:
    """Require the exact union of completed receipt units, with no outcome selection."""
    if len(receipts) > 2 or bool(receipts) != bool(ledger):
        raise ValueError(
            "ledger requires one or two prior receipts; at most two completion retries"
        )
    if ledger is None:
        return set()
    verified = completed(ledger, contract)
    if len(verified) == contract["count"]:
        raise ValueError("completion requires missing units")
    seen: set[str] = set()
    exports: dict[int, dict] = {}
    for attempt, path in enumerate(receipts, 1):
        receipt = json.loads(path.read_text())
        manifest_path = path.parent / "manifest.json"
        former = json.loads(manifest_path.read_text())
        sha = digest(path)
        if (
            sha in seen
            or digest(manifest_path) != receipt.get("manifest_sha256")
            or former.get("commit") != contract["commit"]
            or former.get("script") != SCRIPT
            or former.get("watch_replay", {}).get("allocation") != identity(contract)
            or former.get("watch_replay", {}).get("attempt") != attempt
            or receipt.get("stop_verified") is not True
            or receipt.get("status") == "PASS"
            or not any(word in str(receipt.get("error", "")).upper() for word in MECHANICAL_ERRORS)
        ):
            raise ValueError("completion requires exact stopped mechanically failed lineage")
        seen.add(sha)
        for row in receipt.get("exports", []):
            match = re.fullmatch(r"sessions/(\d{6})\.watch\.gz", row.get("path", ""))
            if not match:
                continue
            index = int(match[1])
            if index in exports or not 0 <= index < contract["count"]:
                raise ValueError("duplicate or foreign completed receipt unit")
            exports[index] = {k: row[k] for k in ("sha256", "size")}
    if {r["index"]: {k: r[k] for k in ("sha256", "size")} for r in verified} != exports:
        raise ValueError("ledger must contain exactly all verified prior receipt units")
    return set(exports)


def _checkpoint_roots(sources: list[Path], contract: dict) -> list[Path]:
    """Accept only explicit new-bank session/checkpoint roots with an exact contract."""
    roots = []
    for source in sources:
        if source.name == "checkpoints":
            sessions = source.parent
        elif (source / "sessions").is_dir():
            sessions = source / "sessions"
        elif (source / "exports/sessions").is_dir():
            sessions = source / "exports/sessions"
        else:
            sessions = source
        binding = sessions / "contract.json"
        if json.loads(binding.read_text()) != contract:
            raise ValueError("foreign checkpoint source contract")
        from tools.ops.package_spec import verify_immutable

        verify_immutable(binding)
        root = sessions / "checkpoints"
        if not root.is_dir():
            raise ValueError("missing checkpoint source directory")
        for unit_dir in root.glob("[0-9]" * 6):
            if not unit_dir.is_dir():
                continue
            index = int(unit_dir.name)
            if not 0 <= index < contract["count"]:
                raise ValueError("foreign checkpoint index")
            checkpoint_candidates([root], index, contract, foreign="refuse")
        roots.append(root)
    return roots


def prepare(
    out: Path,
    commit: str,
    *,
    packet: int = 1,
    ledger: Path | None = None,
    previous_receipts: list[Path] | None = None,
    checkpoint_sources: list[Path] | None = None,
    canary_receipt: Path | None = None,
) -> dict:
    """Freeze a registered packet and identical canary/main input inventory."""
    if type(packet) is not int or not 1 <= packet <= len(L1_OS_BANK_NAMESPACES):
        raise ValueError("packet must be 1..3")
    _check_pin(commit)
    audit = l1_os_bank_disjointness()
    receipts = previous_receipts or []
    contracts = {
        "main": {
            **allocation(L1_OS_BANK_NAMESPACES[packet - 1], L1_OS_BANK_COUNT, "exact-stop-l1-os"),
            "commit": commit,
        },
        "canary": {
            **allocation(
                L1_OS_CANARY_NAMESPACES[packet - 1], 1, "exact-stop-l1-os", CANARY_HORIZON
            ),
            "commit": commit,
        },
    }
    done = _lineage(contracts["main"], ledger, receipts)
    if checkpoint_sources and not receipts:
        raise ValueError("checkpoint resume requires stopped prior receipt lineage")
    roots = _checkpoint_roots(checkpoint_sources or [], contracts["main"])
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    publish_json(out / "seed-audit.json", audit)
    plan = forecast()
    publish_json(out / "forecast.json", plan)
    inputs = []
    for label, contract in contracts.items():
        publish_json(out / f"{label}-contract.json", contract)
        value = (
            json.loads(ledger.read_text())
            if label == "main" and ledger
            else empty_ledger(contract["namespace"], "sessions", identity(contract))
        )
        publish_json(out / f"{label}-ledger.json", value)
        inputs += [
            Input(out / f"{label}-{kind}.json", f"{label}-{kind}.json")
            for kind in ("contract", "ledger")
        ]
    saved = Path(os.environ.get("SNAKE_CKPT_DIR", ROOT.parent / "snake-dqn/saved_snakes"))
    pins = json.loads((ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    for name, (pin, size) in CHECKPOINTS.items():
        local = saved / name
        if (
            local.stat().st_size != size
            or digest(local) != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"local checkpoint differs from released pin: {name}")
    catalog = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    public_inputs._load(catalog)
    inputs.append(Input(catalog, "public-inputs.json"))
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs.append(Input(bundle, "release-evidence.tar.gz"))
    staged = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)
    metadata_sha = next(row["sha256"] for row in staged if row["dest"] == METADATA)
    if roots:
        archive = out / "checkpoints-resume.tar.gz"
        if pack_checkpoint_resume(roots, archive, done):
            if archive.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("resume checkpoint archive exceeds export reserve")
            staged += stage_inputs(out / "staged-resume", [Input(archive, archive.name)])
    manifests = {}
    for label in ("canary", "main"):
        base = 15 if label == "canary" else plan["high_minutes"]
        unit = base / 60 * G4_RATE
        manifest = {
            "job_id": f"watch-replay-l1-os-p{packet}-{label}-{out.name}",
            "commit": commit,
            "script": SCRIPT,
            "source_paths": source_paths(),
            "device": "cpu",
            "lease_seconds": 300,
            "timeout_minutes": 2 * base,
            "output_dir": str(out / f"output-{label}"),
            "max_export_bytes": MAX_EXPORT_BYTES,
            "requirements": {
                "gpu": "G4",
                "allowed_gpus": ["G4"],
                "min_vcpu": 48,
                "est_minutes": base,
            },
            "budget": {
                "unit_cap": 2 * unit,
                "reserve_units": 2 * unit * 0.2,
                "rate_units_per_hour": G4_RATE,
                "other_assignments": {},
            },
            "watch_replay": {
                "allocation": identity(contracts[label]),
                "attempt": len(receipts) + 1,
                "packet": packet,
                "base_minutes": base,
                "base_unit_cap": unit,
                "base_reserve_units": unit * 0.2,
                "forecast": plan,
            },
            "required_exports": ["sessions/contract.json", "sessions/operational.json"],
            "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in staged],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "PYTHONUNBUFFERED": "1",
                "SNAKE_DQN_DEVICE": "cpu",
                "WATCH_REPLAY_METADATA_SHA256": metadata_sha,
                "WATCH_REPLAY_WORKERS": "1" if label == "canary" else str(WORKERS),
                "WATCH_REPLAY_CHECKPOINT_SECONDS": "45" if label == "canary" else "1200",
            },
        }
        if label == "canary":
            manifest["required_exports"].append("sessions/000000.watch.gz")
            manifest["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/watch_replay"
            )
            manifest["resume_exemption"] = {
                "kind": "short-canary",
                "reason": "excluded new-bank H=120 session with identical main inputs",
                "timeout_minutes": 30,
            }
            manifest["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(manifest),
            }
        else:
            manifest["env"]["WATCH_REPLAY_SESSION_SECONDS"] = str(SESSION_SECONDS)
            manifest["resume"] = {
                "unit_kind": "watch-session",
                "ledger_path": "inputs/main-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        publish_json(out / f"{label}-upload-preflight.json", check_uploads(manifest))
        manifests[label] = manifest
        publish_json(out / ("canary.json" if label == "canary" else "main-unsealed.json"), manifest)
    if canary_receipt is not None:
        sealed = {
            **manifests["main"],
            "canary": {
                "schema": canary.SCHEMA,
                "receipt": {
                    "path": str(canary_receipt.resolve()),
                    "sha256": digest(canary_receipt),
                },
            },
        }
        canary.check(sealed, ROOT)
        publish_json(out / "main.json", sealed)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--packet", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--previous-receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--checkpoint-sources", type=Path, nargs="*", default=[])
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--forecast-only", action="store_true")
    args = parser.parse_args()
    if args.forecast_only:
        print(json.dumps(forecast(), indent=2))
        return
    if args.out is None or args.commit is None:
        parser.error("--out and --commit are required unless --forecast-only")
    print(
        json.dumps(
            prepare(
                args.out,
                args.commit,
                packet=args.packet,
                ledger=args.ledger,
                previous_receipts=args.previous_receipts,
                checkpoint_sources=args.checkpoint_sources,
                canary_receipt=args.canary_receipt,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
