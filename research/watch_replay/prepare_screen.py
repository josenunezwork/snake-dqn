"""Prepare (never dispatch) Bank-v2 screen/confirmation packets for a Colab G4 CPU host.

One packet = one G4 host session.  Writes, under --out:
  canary.json        family canary on BANK_V2_DEV_NAMESPACE world 0 (both arms)
  main-unsealed.json the study allocation (optionally one shard), resume block, a
                     host-session budget (SCREEN_SESSION_SECONDS) after which running
                     units checkpoint and stop for the next attempt
  main.json          main bound to a PASS canary receipt (when --canary-receipt is given)
  forecast.json      forecast from the measured screen G4 unit times

Multi-attempt: build ``--ledger`` with ``research.watch_replay.screen_ledger ledger``
over all prior attempt dirs, pass those dirs as ``--checkpoint-sources`` and bump
``--attempt``.  Inputs reuse research/watch_replay/uploads.py's frozen METADATA schema.
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
    STUDIES,
    admitted_identities,
    allocation,
    identity,
    pack_checkpoint_resume,
    read_ledger,
)
from research.watch_replay.screen_ledger import screen_root
from research.watch_replay.seed_audit import BANK_V2_DEV_NAMESPACE
from research.watch_replay.source import source_paths
from research.watch_replay.uploads import CHECKPOINTS, METADATA, check_uploads, pack_inputs
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs

SCRIPT = "research/watch_replay/screen_job.sh"
G4_RATE = 10.05  # u/h, live-observed (CAMPAIGN_LOG 2026-11-03 11:30 row)
G4_WORKERS = 48
CANARY_HORIZON = 120
# Measured on G4 at 48 workers, Bank-v2 screen attempt 1 (packet-02, H=6000, 93 units):
# exact-stop-l1 mean 1.00 h (max 1.10), exact-stop-l1-os mean 2.27 h (max 2.72).
G4_HOURS_PER_6000 = {"exact-stop-l1": 1.00, "exact-stop-l1-os": 2.27}
G4_MAX_HOURS_PER_6000 = {"exact-stop-l1": 1.10, "exact-stop-l1-os": 2.72}
# Frames beyond 6,000 see longer heroes (more scaled-H roots): l1-os per-frame cost
# multiplier for those frames (central / high); l1 is ~flat in L.
LONG_FRAME_FACTOR = {"central": 1.15, "high": 1.35}
REPLAY_OVERHEAD = 0.08  # forced re-simulation per resume (~65 ms/frame on G4)
SESSION_SECONDS = 9000  # 2.5 h compute budget per host session
SETUP_EXPORT_MINUTES = 25


def unit_hours(arm: str, horizon: int, scenario: str, *, worst: bool = False) -> float:
    base = (G4_MAX_HOURS_PER_6000 if worst else G4_HOURS_PER_6000)[arm] / 6000
    long = LONG_FRAME_FACTOR[scenario] if arm == "exact-stop-l1-os" else 1.0
    return base * (min(horizon, 6000) + long * max(0, horizon - 6000))


def forecast(study: str, scenario: str, shards: int = 1, workers: int = G4_WORKERS) -> dict:
    spec = STUDIES[study]
    count, horizon = spec["count"], spec["horizon"]
    arms = ("exact-stop-l1", "exact-stop-l1-os")
    worker_hours = (
        (1 + REPLAY_OVERHEAD) * count * sum(unit_hours(a, horizon, scenario) for a in arms)
    )
    # Critical path: the slowest l1-os unit (mean-based centrally, the screen's slowest
    # unit scaled in the high scenario) must fit in consecutive sessions.
    longest = (1 + REPLAY_OVERHEAD) * unit_hours(
        "exact-stop-l1-os", horizon, scenario, worst=scenario == "high"
    )
    session_h = SESSION_SECONDS / 3600
    per_shard = worker_hours / shards
    sessions_per_shard = max(
        math.ceil(per_shard / (workers * session_h)), math.ceil(longest / session_h)
    )
    session_minutes = SESSION_SECONDS / 60 + SETUP_EXPORT_MINUTES
    sessions = sessions_per_shard * shards
    return {
        "study": study,
        "scenario": scenario,
        "units": 2 * count,
        "horizon": horizon,
        "g4_unit_hours": {a: unit_hours(a, horizon, scenario) for a in arms},
        "longest_unit_hours": longest,
        "worker_hours": worker_hours,
        "shards": shards,
        "sessions_per_shard": sessions_per_shard,
        "sessions": sessions,
        "session_minutes": session_minutes,
        "parallel_wall_hours": sessions_per_shard * session_minutes / 60,
        "units_total": sessions * session_minutes / 60 * G4_RATE,
        "rate_units_per_hour": G4_RATE,
        "basis": "measured G4 48W unit times, Bank-v2 screen attempt 1 (packet-02)",
    }


def _checkpoint_dirs(sources: list[Path]) -> list[Path]:
    dirs = []
    for source in sources:
        try:
            root = screen_root(source) / "checkpoints"
        except ValueError:
            root = source
        dirs.append(root if root.is_dir() else source)
    return dirs


def prepare(
    out: Path,
    commit: str,
    *,
    study: str = "screen",
    shard: tuple[int, int] | None = None,
    ledger: Path | None = None,
    checkpoint_sources: list[Path] | None = None,
    canary_receipt: Path | None = None,
    attempt: int = 1,
    workers: int = G4_WORKERS,
    checkpoint_seconds: int = 1200,
    session_seconds: int | None = SESSION_SECONDS,
    canary_horizon: int = CANARY_HORIZON,
    canary_session_seconds: int | None = None,
    canary_checkpoint_sources: list[Path] | None = None,
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
        "main": {
            **allocation(
                STUDIES[study]["namespace"], STUDIES[study]["count"], STUDIES[study]["horizon"]
            ),
            "commit": commit,
        },
        "canary": {**allocation(BANK_V2_DEV_NAMESPACE, 1, canary_horizon), "commit": commit},
    }
    done: set[int] = set()
    main_ledger = None
    if ledger is not None:
        main = contracts["main"]
        value = json.loads(ledger.read_text())
        if value.get("identity") != identity(main) and value.get("identity") in (
            admitted_identities(main)
        ):
            # Ledger from an allowlisted predecessor commit: completed units carry that
            # commit's identity and cannot be mixed in without an analysis amendment, so
            # only an EMPTY predecessor ledger is carried over (re-bound to this contract).
            read_ledger(ledger, main["namespace"], ALLOCATION, value["identity"])
            if value["units"]:
                raise ValueError("predecessor-commit ledger with completed units: amend first")
            main_ledger = empty_ledger(main["namespace"], ALLOCATION, identity(main))
        else:
            done = {
                r["index"]
                for r in read_ledger(ledger, main["namespace"], ALLOCATION, identity(main))
            }
            main_ledger = value
    inputs: list[Input] = []
    for label, contract in contracts.items():
        publish_json(out / f"{label}-contract.json", contract)
        publish_json(
            out / f"{label}-ledger.json",
            (
                main_ledger
                if label == "main" and main_ledger is not None
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
    resume_rows: dict[str, list[dict]] = {"main": [], "canary": []}
    for label, sources in (("main", checkpoint_sources), ("canary", canary_checkpoint_sources)):
        if sources:
            archive = out / f"{label}-checkpoints-resume.tar.gz"
            if pack_checkpoint_resume(
                _checkpoint_dirs(sources), archive, done if label == "main" else set()
            ):
                resume_rows[label] = stage_inputs(
                    out / f"staged-resume-{label}", [Input(archive, "checkpoints-resume.tar.gz")]
                )
    if attempt >= 2 and resume_rows["main"] and not canary_checkpoint_sources:
        # canary.py's coverage check needs the canary to upload byte-identical inputs
        # (dest + sha256) to the main: reuse the main's archive (the canary runs with
        # --foreign-checkpoints skip, so these confirm-contract checkpoints are ignored).
        resume_rows["canary"] = resume_rows["main"]
    shards = shard[1] if shard else 1
    plan = {sc: forecast(study, sc, shards, workers) for sc in ("central", "high")}
    publish_json(out / "forecast.json", plan)
    budget = {"main": session_seconds, "canary": canary_session_seconds}
    manifests = {}
    for label in ("canary", "main"):
        if label == "canary":
            minutes = 15
        elif session_seconds is not None:
            minutes = math.ceil(session_seconds / 60 + SETUP_EXPORT_MINUTES)
        else:
            minutes = math.ceil(plan["high"]["longest_unit_hours"] * 60 + SETUP_EXPORT_MINUTES)
        unit = minutes / 60 * G4_RATE
        manifest = {
            "job_id": f"bank-v2-{study}-{label}-{out.name}",
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
                "study": study if label == "main" else "dev-canary",
                "shard": list(shard) if shard and label == "main" else None,
                "session_seconds": budget[label],
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
                for row in [*staged, *resume_rows[label]]
            ],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "PYTHONUNBUFFERED": "1",
                "SNAKE_DQN_DEVICE": "cpu",
                "WATCH_REPLAY_METADATA_SHA256": metadata_sha256,
                "SCREEN_WORKERS": "2" if label == "canary" else str(workers),
                "SCREEN_CHECKPOINT_SECONDS": "45" if label == "canary" else str(checkpoint_seconds),
            },
        }
        if budget[label] is not None:
            manifest["env"]["SCREEN_SESSION_SECONDS"] = str(budget[label])
        if label == "main" and shard is not None:
            manifest["env"]["SCREEN_SHARD"] = f"{shard[0]}/{shard[1]}"
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
            if canary_session_seconds is None:
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


def _shard(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    k, n = (int(x) for x in value.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError("shard must be k/n with 0 <= k < n")
    return k, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--study", choices=sorted(STUDIES), default="screen")
    parser.add_argument("--shard", type=_shard, help="k/n: worlds with world %% n == k")
    parser.add_argument("--ledger", type=Path, help="screen_ledger-built verified ledger")
    parser.add_argument("--checkpoint-sources", type=Path, nargs="*", default=[])
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--workers", type=int, default=G4_WORKERS)
    parser.add_argument("--session-seconds", type=int, default=SESSION_SECONDS)
    parser.add_argument("--canary-horizon", type=int, default=CANARY_HORIZON)
    parser.add_argument("--canary-session-seconds", type=int)
    parser.add_argument("--canary-checkpoint-sources", type=Path, nargs="*", default=[])
    parser.add_argument("--forecast-only", action="store_true")
    args = parser.parse_args()
    if args.forecast_only:
        shards = args.shard[1] if args.shard else 1
        print(
            json.dumps(
                {sc: forecast(args.study, sc, shards, args.workers) for sc in ("central", "high")},
                indent=2,
            )
        )
        return
    plan = prepare(
        args.out.resolve(),
        args.commit,
        study=args.study,
        shard=args.shard,
        ledger=args.ledger,
        checkpoint_sources=args.checkpoint_sources,
        canary_receipt=args.canary_receipt,
        attempt=args.attempt,
        workers=args.workers,
        session_seconds=args.session_seconds,
        canary_horizon=args.canary_horizon,
        canary_session_seconds=args.canary_session_seconds,
        canary_checkpoint_sources=args.canary_checkpoint_sources,
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
