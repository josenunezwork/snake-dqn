"""Prepare (never dispatch) the Bank-v2 paired screen packets for a Colab G4 CPU host.

Writes, under --out:
  canary.json        family canary: 1 dev pair (BANK_V2_DEV_NAMESPACE world 0) at H=120
  main-unsealed.json 96 units (48 screen worlds x 2 arms), H=6000, resume block
  main.json          main bound to a PASS canary receipt (when --canary-receipt is given)
  forecast.json      unit/minute forecast from the Bank-v2 build canary + measured G4 rates

Inputs reuse research/watch_replay/uploads.py's frozen METADATA schema (main/canary
contract + ledger, release evidence, public inputs) so staging is identical to the
served replay jobs; an optional checkpoints-resume.tar.gz (<= 2 MiB) carries per-unit
checkpoints from a stopped attempt.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from research.colab_cli import canary
from research.t34_exit_b8 import public_inputs
from research.watch_replay.prepare import release_bundle
from research.watch_replay.run import ROOT
from research.watch_replay.screen_l1_os import (
    ALLOCATION,
    allocation,
    identity,
    pack_checkpoint_resume,
    read_ledger,
)
from research.watch_replay.seed_audit import BANK_V2_DEV_NAMESPACE, BANK_V2_SCREEN_NAMESPACE
from research.watch_replay.source import source_paths
from research.watch_replay.uploads import CHECKPOINTS, METADATA, check_uploads, pack_inputs
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs

SCRIPT = "research/watch_replay/screen_job.sh"
G4_RATE = 10.05  # u/h, live-observed (CAMPAIGN_LOG 2026-11-03 11:30 row)
G4_WORKERS = 48
CANARY_HORIZON = 120
# Mac 1-thread costs from the Bank-v2 build canary (t34-sol/bank-v2-build/canary.json).
MAC_FRAME_SECONDS = 0.012
MAC_H300_ROOT_SECONDS = 0.62
MAC_OPEN_STOP_ROOT_SECONDS = 0.075
MAC_SCALED_ROOT_SECONDS = {"central": 3.0, "high": 4.5}
LONG_ROOT_SHARE = {"central": 0.4, "high": 0.6}  # f = share of exact roots at L >= 800
L1_BRANCHES_PER_SESSION = 1268.5208333333333  # N=48 screen calibration (prepare.py)
# G4 per-worker slowdown vs one Mac core: measured 1W 1.38 s vs Mac 0.62 s per H300 root,
# 48W node 14.3 roots/s (argon-l1-serve-1103 prepare.py) -> 48W = 48/14.3/0.62 = 5.41x.
G4_SLOWDOWN_48W = 48 / 14.3 / MAC_H300_ROOT_SECONDS
SETUP_EXPORT_MINUTES = 30


def forecast(scenario: str = "high", workers: int = G4_WORKERS) -> dict:
    frames = 6000
    l1 = frames * MAC_FRAME_SECONDS + L1_BRANCHES_PER_SESSION * MAC_H300_ROOT_SECONDS
    f, c = LONG_ROOT_SHARE[scenario], MAC_SCALED_ROOT_SECONDS[scenario]
    l1os = (
        frames * MAC_FRAME_SECONDS
        + 1200 * ((1 - f) * MAC_H300_ROOT_SECONDS + f * c)
        + 4800 * MAC_OPEN_STOP_ROOT_SECONDS
    )
    slow = G4_SLOWDOWN_48W
    # Longest-first on `workers` >= 48: wave 1 = all 48 l1-os units, wave 2 = 48 l1 units.
    wall_h = (l1os + l1) * slow / 3600 + SETUP_EXPORT_MINUTES / 60
    minutes = math.ceil(wall_h * 60)
    return {
        "scenario": scenario,
        "mac_unit_hours": {"exact-stop-l1": l1 / 3600, "exact-stop-l1-os": l1os / 3600},
        "mac_cpu_hours_total": 48 * (l1 + l1os) / 3600,
        "g4_slowdown_per_worker_at_48w": slow,
        "g4_unit_hours": {
            "exact-stop-l1": l1 * slow / 3600,
            "exact-stop-l1-os": l1os * slow / 3600,
        },
        "workers": workers,
        "base_minutes": minutes,
        "base_units": minutes / 60 * G4_RATE,
        "rate_units_per_hour": G4_RATE,
        "assumptions": {
            "long_root_share": f,
            "mac_scaled_root_seconds": c,
            "note": "f (share of exact roots at L>=800) is unmeasured; first G4 checkpoint "
            "and the canary refine it",
        },
    }


def prepare(
    out: Path,
    commit: str,
    *,
    ledger: Path | None = None,
    checkpoint_sources: list[Path] | None = None,
    canary_receipt: Path | None = None,
    attempt: int = 1,
    workers: int = G4_WORKERS,
    checkpoint_seconds: int = 1200,
) -> dict:
    if not 1 <= workers <= 48:
        raise ValueError("G4 workers must be 1..48")
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if len(commit) != 40 or not any(r.strip().startswith("origin/") for r in refs.splitlines()):
        raise ValueError("requires a full commit published on origin")
    if subprocess.check_output(["git", "diff", commit, "--", *source_paths()], cwd=ROOT):
        raise ValueError("working executable bytes differ from source pin")
    out.mkdir(parents=True, exist_ok=False)
    contracts = {
        "main": {**allocation(BANK_V2_SCREEN_NAMESPACE, 48, 6000), "commit": commit},
        "canary": {**allocation(BANK_V2_DEV_NAMESPACE, 1, CANARY_HORIZON), "commit": commit},
    }
    done: set[int] = set()
    if ledger is not None:
        main = contracts["main"]
        done = {
            r["index"] for r in read_ledger(ledger, main["namespace"], ALLOCATION, identity(main))
        }
    inputs: list[Input] = []
    for label, contract in contracts.items():
        publish_json(out / f"{label}-contract.json", contract)
        publish_json(
            out / f"{label}-ledger.json",
            (
                json.loads(ledger.read_text())
                if label == "main" and ledger
                else empty_ledger(contract["namespace"], ALLOCATION, identity(contract))
            ),
        )
        inputs += [
            Input(out / f"{label}-contract.json", f"{label}-contract.json"),
            Input(out / f"{label}-ledger.json", f"{label}-ledger.json"),
        ]
    saved = ROOT.parent / "snake-dqn/saved_snakes"
    pins = json.loads((ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    for name, (pin, size) in CHECKPOINTS.items():
        local = saved / name
        if (
            local.stat().st_size != size
            or digest(local) != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"local checkpoint differs from released pin: {local}")
    catalog = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    public_inputs._load(catalog)
    inputs.append(Input(catalog, "public-inputs.json"))
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs.append(Input(bundle, "release-evidence.tar.gz"))
    staged = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)
    metadata_sha256 = staged[0]["sha256"]
    resume_rows: list[dict] = []
    if checkpoint_sources:
        archive = out / "checkpoints-resume.tar.gz"
        packed = pack_checkpoint_resume(checkpoint_sources, archive, done)
        if packed:
            resume_rows = stage_inputs(
                out / "staged-resume", [Input(archive, "checkpoints-resume.tar.gz")]
            )
    plan = {s: forecast(s, workers) for s in ("central", "high")}
    publish_json(out / "forecast.json", plan)
    base = plan["high"]["base_minutes"]
    manifests = {}
    for label in ("canary", "main"):
        minutes = 15 if label == "canary" else base
        unit = minutes / 60 * G4_RATE
        manifest = {
            "job_id": f"bank-v2-screen-{label}-{out.name}",
            "commit": commit,
            "script": SCRIPT,
            "source_paths": source_paths(),
            "device": "cpu",
            "lease_seconds": 300,
            "timeout_minutes": 2 * minutes,
            "output_dir": str(out / f"output-{label}"),
            "max_export_bytes": 256 * 1024**2,
            "requirements": {
                "gpu": "G4",
                "allowed_gpus": ["G4"],
                "min_vcpu": 48,
                "est_minutes": minutes,
            },
            "budget": {
                "unit_cap": 2 * unit,
                "reserve_units": 2 * unit * 0.2,
                "rate_units_per_hour": G4_RATE,
                "other_assignments": {},
            },
            "bank_v2_screen": {
                "allocation": identity(contracts[label]),
                "attempt": attempt,
                "base_minutes": minutes,
                "base_unit_cap": unit,
                "forecast": plan if label == "main" else None,
                "checkpoint_seconds": checkpoint_seconds,
            },
            "required_exports": ["screen/contract.json"],
            "inputs": [
                {k: row[k] for k in ("path", "dest", "sha256")}
                for row in [*staged, *(resume_rows if label == "main" else [])]
            ],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "PYTHONUNBUFFERED": "1",
                "SNAKE_DQN_DEVICE": "cpu",
                "WATCH_REPLAY_METADATA_SHA256": metadata_sha256,
                "SCREEN_WORKERS": "2" if label == "canary" else str(workers),
                "SCREEN_CHECKPOINT_SECONDS": "60" if label == "canary" else str(checkpoint_seconds),
            },
        }
        if label == "canary":
            manifest["resume_exemption"] = {
                "kind": "short-canary",
                "reason": "1 dev pair at H=120 (<5 min compute); family canary for the screen",
            }
            manifest["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/watch_replay"
            )
            manifest["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(manifest),
            }
            manifest["required_exports"] += [f"screen/units/{i:06d}.json" for i in (0, 1)]
        else:
            manifest["resume"] = {
                "unit_kind": "screen-arm-unit",
                "ledger_path": "inputs/main-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        publish_json(out / f"{label}-upload-preflight.json", check_uploads(manifest))
        manifests[label] = manifest
    (out / "canary.json").write_text(
        json.dumps(manifests["canary"], indent=2, sort_keys=True) + "\n"
    )
    (out / "main-unsealed.json").write_text(
        json.dumps(manifests["main"], indent=2, sort_keys=True) + "\n"
    )
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
        (out / "main.json").write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ledger", type=Path, help="Mac-verified ledger (completion attempts)")
    parser.add_argument("--checkpoint-sources", type=Path, nargs="*", default=[])
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--workers", type=int, default=G4_WORKERS)
    parser.add_argument("--forecast-only", action="store_true")
    args = parser.parse_args()
    if args.forecast_only:
        print(json.dumps({s: forecast(s, args.workers) for s in ("central", "high")}, indent=2))
        return
    plan = prepare(
        args.out.resolve(),
        args.commit,
        ledger=args.ledger,
        checkpoint_sources=args.checkpoint_sources,
        canary_receipt=args.canary_receipt,
        attempt=args.attempt,
        workers=args.workers,
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
