"""Prepare branch-pinned Colab packets; never allocate, dispatch, or alter serving."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path

from research.colab_cli import canary
from research.t34_exit_b8 import public_inputs
from research.watch_replay.preflight import CANARY_HORIZON_BY_CONTROLLER
from research.watch_replay.resume import stage_session_checkpoints
from research.watch_replay.run import (
    ROOT,
    SESSION_CHECKPOINT_SCHEMA,
    allocation,
    completed,
    identity,
)
from research.watch_replay.source import source_paths
from research.watch_replay.uploads import (
    CHECKPOINTS,
    METADATA,
    check_uploads,
    pack_checkpoint_resume,
    pack_inputs,
)
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs
from web.backend.watch_replay import CONTROLLERS

GPU_RATES: dict[str, float] = {"L4": 1.54, "A100": 8.0, "G4": 10.05}
DEFAULT_WORKERS_BY_GPU: dict[str, int] = {"L4": 4, "A100": 6, "G4": 12}
MAX_WORKERS_BY_GPU: dict[str, int] = {"L4": 12, "A100": 12, "G4": 48}
MAX_COUNT_BY_GPU: dict[str, int] = {"L4": 12, "A100": 12, "G4": 48}
MIN_VCPU_BY_GPU: dict[str, int] = {"L4": 12, "A100": 12, "G4": 48}
DEFAULT_CANARY_NAMESPACE = "t34-watch-replay-full-size-fixture-v1"
G4_CANARY_NAMESPACE = "t34-watch-replay-20261103-g4-canary-l1-v1"
CANARY_NAMESPACE_BY_GPU: dict[str, str] = {
    "L4": DEFAULT_CANARY_NAMESPACE,
    "A100": DEFAULT_CANARY_NAMESPACE,
    "G4": G4_CANARY_NAMESPACE,
}
DEFAULT_CHECKPOINT_INTERVAL_BY_GPU: dict[str, int] = {"L4": 0, "A100": 0, "G4": 1500}

# Measured Watch6 decision speed on Colab G4 (48 vCPU AMD EPYC 9B45 Zen 5) vs L4/A100 (12 vCPU):
#   L4/A100: 1 worker = 4.05 s/decision, 12 vCPU = 1.69 decisions/s
#   G4:      1 worker = 1.38 s/decision (~2.935x/core), 48 workers = 14.3 decisions/s (~8.46x node)
#   A100 12-session exact-stop-l1 banks (14l1-16l1, W=6): ~3.05 h (~1.525 h/session) & ~20.5 u
L4_SINGLETON_DECISION_SECONDS = 4.05
L4_NODE_DECISIONS_PER_SECOND = 1.69
L4_12W_CONTENTION_FACTOR = (12.0 / L4_NODE_DECISIONS_PER_SECOND) / L4_SINGLETON_DECISION_SECONDS
A100_W6_CONTENTION_FACTOR = 1.0 + (5.0 / 11.0) * (L4_12W_CONTENTION_FACTOR - 1.0)
G4_SINGLETON_DECISION_SECONDS = 1.38
G4_48W_DECISIONS_PER_SECOND = 14.3
G4_PER_CORE_SPEEDUP_VS_L4 = L4_SINGLETON_DECISION_SECONDS / G4_SINGLETON_DECISION_SECONDS
G4_NODE_SPEEDUP_VS_L4 = G4_48W_DECISIONS_PER_SECOND / L4_NODE_DECISIONS_PER_SECOND
A100_L1_BANK12_W6_HOURS = 3.05
A100_L1_BANK12_W6_UNITS = 20.5
A100_L1_SESSION_SECONDS_AT_W6 = (A100_L1_BANK12_W6_HOURS / 2.0) * 3600.0
A100_L1_SESSION_SECONDS_1W = A100_L1_SESSION_SECONDS_AT_W6 / A100_W6_CONTENTION_FACTOR


def g4_contention_factor(workers: int) -> float:
    """Interpolate G4 per-worker contention from measured 1W (1.38s/dec) to 48W (14.3 dec/s)."""
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS_BY_GPU["G4"]:
        raise ValueError(f"G4 workers must be an integer in 1..{MAX_WORKERS_BY_GPU['G4']}")
    max_contention = (48.0 / G4_48W_DECISIONS_PER_SECOND) / G4_SINGLETON_DECISION_SECONDS
    return 1.0 + ((workers - 1) / 47.0) * (max_contention - 1.0)


def g4_decisions_per_second(workers: int) -> float:
    """Expected aggregate Watch6 decisions/s on G4 for `workers` concurrent workers."""
    per_worker_seconds = G4_SINGLETON_DECISION_SECONDS * g4_contention_factor(workers)
    return workers / per_worker_seconds


SCREEN_MEAN_ROOTS_BY_CONTROLLER: dict[str, float] = {
    "bare-b8": 0.0,
    "exact-stop-l2-prime": 403.7291666666667,
    "exact-stop-l1": 1200.125,
}
SCREEN_MEAN_BRANCHES_BY_CONTROLLER: dict[str, float] = {
    "bare-b8": 0.0,
    "exact-stop-l2-prime": 449.6041666666667,
    "exact-stop-l1": 1268.5208333333333,
}
BRANCH_TAIL_MARGIN = 1.35
CLOUD_SLOWDOWN_PROJECTION = 3.5
CONTENTION_PROJECTION = 1.5
MIRROR_BASE_COLAB_SECONDS = 600.0
SETUP_EXPORT_MINUTES = 30


def release_bundle(bundle: Path, pins: dict) -> None:
    """Keep upload bytes identical when binding a canary in a fresh main packet."""
    with (
        bundle.open("xb") as stream,
        gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, pin in sorted(pins["checkpoints"].items()):
                for ref in pin["strict_receipt"].values():
                    if isinstance(ref, dict):
                        path = Path(pin["strict_receipt"]["root"]) / ref["path"]
                        if digest(path) != ref["sha256"]:
                            raise ValueError("release anchor hash mismatch")
                        member = tarfile.TarInfo(name + "/" + ref["path"])
                        member.size = path.stat().st_size
                        archive.addfile(member, io.BytesIO(path.read_bytes()))


def _validate_checkpoint_resume_dir(
    checkpoint_dir: Path,
    contract: dict,
    completed_indices: set[int],
) -> None:
    from research.world_checkpoint.checkpoint import find_latest_checkpoint, load_checkpoint

    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint_resume directory does not exist: {checkpoint_dir}")
    found = 0
    for sub in sorted(checkpoint_dir.iterdir()):
        if not sub.is_dir() or len(sub.name) != 6 or not sub.name.isdigit():
            continue
        index = int(sub.name)
        if not 0 <= index < contract["count"] or index in completed_indices:
            raise ValueError(
                f"checkpoint_resume contains completed or out-of-range session index: {index}"
            )
        latest = find_latest_checkpoint(sub, verify_identity=False)
        if latest is None:
            continue
        found += 1
        ckpt = load_checkpoint(latest, verify_identity=False)
        row = contract["rows"][index]
        if (
            ckpt.get("session_checkpoint_schema") != SESSION_CHECKPOINT_SCHEMA
            or ckpt.get("allocation") != identity(contract)
            or ckpt.get("namespace") != contract["namespace"]
            or int(ckpt.get("index", -1)) != index
            or int(ckpt.get("seed", -1)) != int(row["seed"])
            or ckpt.get("controller") != contract["controller"]
            or int(ckpt.get("horizon", -1)) != int(contract["horizon"])
        ):
            raise ValueError(f"checkpoint_resume mismatch for session {index}: {latest}")
    if found == 0:
        raise ValueError(
            f"checkpoint_resume directory contains zero verified checkpoints: {checkpoint_dir}"
        )


def prepare(
    out: Path,
    commit: str,
    benchmark: Path,
    namespace: str,
    count: int,
    controller: str,
    canary_receipt: Path | None = None,
    ledger: Path | None = None,
    previous_receipts: list[Path] | None = None,
    *,
    gpu: str = "L4",
    workers: int | None = None,
    checkpoint_resume: Path | None = None,
    checkpoint_interval: int | None = None,
) -> dict:
    """Ship identical inputs for an excluded long-body canary and N sessions."""
    if gpu not in GPU_RATES:
        raise ValueError(f"unsupported watch_replay gpu {gpu!r}; expected one of {list(GPU_RATES)}")
    max_count = MAX_COUNT_BY_GPU[gpu]
    if type(count) is not int or not 1 <= count <= max_count:
        raise ValueError(f"one packet contains 1..{max_count} sessions within bounded exports")
    resolved_workers = DEFAULT_WORKERS_BY_GPU[gpu] if workers is None else workers
    max_workers = MAX_WORKERS_BY_GPU[gpu]
    if type(resolved_workers) is not int or not 1 <= resolved_workers <= max_workers:
        raise ValueError(f"workers must be an integer in 1..{max_workers}")
    resolved_ckpt_interval = (
        DEFAULT_CHECKPOINT_INTERVAL_BY_GPU[gpu]
        if checkpoint_interval is None
        else checkpoint_interval
    )
    if type(resolved_ckpt_interval) is not int or resolved_ckpt_interval < 0:
        raise ValueError("checkpoint_interval must be a non-negative integer")
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if len(commit) != 40 or not any(
        line.strip().startswith("origin/") for line in refs.splitlines()
    ):
        raise ValueError("requires full published branch pin")
    if subprocess.check_output(["git", "diff", commit, "--", *source_paths()], cwd=ROOT):
        raise ValueError("working executable bytes differ from source pin")
    measured = json.loads(benchmark.read_text())
    if measured["commit"] != commit:
        raise ValueError("Mac preflight must match current commit")
    out.mkdir(parents=True, exist_ok=False)
    canary_horizon = CANARY_HORIZON_BY_CONTROLLER[controller]
    canary_namespace = CANARY_NAMESPACE_BY_GPU[gpu]
    contracts = {
        "main": {**allocation(namespace, count, controller), "commit": commit},
        # Frozen excluded prefix from the measured full-session fixture (or G4 canary namespace).
        # L2' reaches L>=600/search within 3,000 frames; L>=1 exercises 60 roots within 300 frames
        # while fitting the fixed 15/30-minute canary guard.
        "canary": {
            **allocation(canary_namespace, 1, controller, canary_horizon),
            "commit": commit,
        },
    }
    previous_receipts = previous_receipts or []
    if len(previous_receipts) > 2 or bool(previous_receipts) != bool(ledger):
        raise ValueError(
            "one original plus at most two completion attempts; ledger requires receipts"
        )
    verified = []
    if ledger:
        verified = completed(ledger, contracts["main"])
        if len(verified) == count:
            raise ValueError("completion requires missing units")
        hashes = set()
        exports = {}
        for attempt, receipt_path in enumerate(previous_receipts, 1):
            receipt = json.loads(receipt_path.read_text())
            former_path = receipt_path.parent / "manifest.json"
            former = json.loads(former_path.read_text())
            error = str(receipt.get("error", "")).upper()
            if (
                digest(receipt_path) in hashes
                or digest(former_path) != receipt.get("manifest_sha256")
                or former.get("commit") != commit
                or former.get("watch_replay", {}).get("allocation") != identity(contracts["main"])
                or former.get("watch_replay", {}).get("attempt") != attempt
                or receipt.get("stop_verified") is not True
                or receipt.get("status") == "PASS"
                or not any(
                    word in error
                    for word in (
                        "TIMEOUT",
                        "LEASE",
                        "TRANSFER",
                        "UPLOAD",
                        "STAGING",
                        "INTERRUPT",
                        "RUNTIME_LOST",
                    )
                )
            ):
                raise ValueError("completion requires exact stopped mechanically failed lineage")
            hashes.add(digest(receipt_path))
            for row in receipt.get("exports", []):
                name = row.get("path", "")
                if name.startswith("sessions/") and name.endswith(".watch.gz"):
                    index = int(Path(name).name.split(".")[0])
                    if index in exports:
                        raise ValueError("duplicate completed units across attempts")
                    exports[index] = {"sha256": row["sha256"], "size": row["size"]}
        if {r["index"]: {k: r[k] for k in ("sha256", "size")} for r in verified} != exports:
            raise ValueError("ledger must contain exactly all verified prior receipt units")
    inputs = []
    for label, contract in contracts.items():
        publish_json(out / f"{label}-contract.json", contract)
        publish_json(
            out / f"{label}-ledger.json",
            (
                json.loads(ledger.read_text())
                if label == "main" and ledger
                else empty_ledger(contract["namespace"], "sessions", identity(contract))
            ),
        )
        inputs.extend(
            [
                Input(out / f"{label}-contract.json", f"{label}-contract.json"),
                Input(out / f"{label}-ledger.json", f"{label}-ledger.json"),
            ]
        )
    saved = ROOT.parent / "snake-dqn/saved_snakes"
    if not saved.is_dir():
        saved = ROOT / "saved_snakes"
    pins = json.loads((ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    for name, (pin, size) in CHECKPOINTS.items():
        local_ckpt = saved / name
        if (
            not local_ckpt.is_file()
            or local_ckpt.stat().st_size != size
            or digest(local_ckpt) != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"local checkpoint differs from released pin: {local_ckpt}")
    public_catalog = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    public_inputs._load(public_catalog)
    inputs.append(Input(public_catalog, "public-inputs.json"))
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs.append(Input(bundle, "release-evidence.tar.gz"))
    raw_staged = stage_inputs(out / "staged", inputs)
    staged = pack_inputs(raw_staged, out / METADATA)
    metadata_sha256 = next(row["sha256"] for row in staged if row["dest"] == METADATA)
    completed_indices = {int(r["index"]) for r in verified}
    if checkpoint_resume is None and previous_receipts:
        auto_ckpt_out = out / "staged-checkpoints-resume"
        try:
            auto_staged = stage_session_checkpoints(
                contracts["main"],
                [r.parent for r in previous_receipts],
                auto_ckpt_out,
            )
        except (FileNotFoundError, ValueError):
            auto_staged = []
        if auto_staged:
            checkpoint_resume = auto_ckpt_out
    ckpt_resume_rows: list[dict] = []
    if checkpoint_resume is not None:
        _validate_checkpoint_resume_dir(checkpoint_resume, contracts["main"], completed_indices)
        ckpt_resume_rows = pack_checkpoint_resume(checkpoint_resume, out / "staged")
        staged.extend(ckpt_resume_rows)
    singleton = max(t["seconds"] / t["branches"] for t in measured["branch_timings"])
    screen_roots = SCREEN_MEAN_ROOTS_BY_CONTROLLER[controller]
    screen_branches = SCREEN_MEAN_BRANCHES_BY_CONTROLLER[controller]
    l2_prime_branches = SCREEN_MEAN_BRANCHES_BY_CONTROLLER["exact-stop-l2-prime"]
    search_multiplier = screen_branches / l2_prime_branches if l2_prime_branches else 0.0
    branches_forecast = screen_branches * BRANCH_TAIL_MARGIN
    if gpu == "G4":
        cloud_slowdown = CLOUD_SLOWDOWN_PROJECTION / G4_PER_CORE_SPEEDUP_VS_L4
        contention = g4_contention_factor(resolved_workers)
        mirror_base_seconds = (MIRROR_BASE_COLAB_SECONDS / G4_PER_CORE_SPEEDUP_VS_L4) * contention
    else:
        cloud_slowdown = CLOUD_SLOWDOWN_PROJECTION
        contention = CONTENTION_PROJECTION
        mirror_base_seconds = MIRROR_BASE_COLAB_SECONDS
    # Planning assumptions calibrated to the N=48 screen (~2.82x search cost for exact-stop-l1).
    search_seconds = branches_forecast * singleton * cloud_slowdown * contention
    session_seconds = mirror_base_seconds + search_seconds
    minutes = math.ceil(
        session_seconds * math.ceil(count / resolved_workers) / 60 + SETUP_EXPORT_MINUTES
    )
    if minutes > 600:
        raise ValueError("forecast exceeds bounded runtime; use separate frozen packets")
    forecast: dict = {
        "controller": controller,
        "canary_horizon": canary_horizon,
        "mac_h300_singleton_seconds": singleton,
        "screen_mean_roots_per_session": screen_roots,
        "screen_mean_branches_per_session": screen_branches,
        "search_cost_multiplier_vs_l2_prime": round(search_multiplier, 6),
        "branch_tail_margin": BRANCH_TAIL_MARGIN,
        "branches_per_session_forecast": branches_forecast,
        "roots_per_session_max": 1200 if controller != "bare-b8" else 0,
        "branches_per_root_max": 6,
        "cloud_slowdown_projection": cloud_slowdown,
        "contention_projection": contention,
        "mirror_base_colab_seconds_per_session": mirror_base_seconds,
        "setup_export_minutes": SETUP_EXPORT_MINUTES,
        "search_colab_seconds_per_session": search_seconds,
        "session_colab_seconds": session_seconds,
        "gpu": gpu,
        "min_vcpu": MIN_VCPU_BY_GPU[gpu],
        "rate_units_per_hour": GPU_RATES[gpu],
        "workers": resolved_workers,
        "base_minutes": minutes,
        "effective_minutes": 2 * minutes,
        "expected_sessions_per_colab_hour": resolved_workers * 3600 / session_seconds,
        "scope": (
            "N=48 screen branch-calibrated projection (~2.82x exact-stop-l1 vs l2-prime search "
            "cost); includes mirror + setup/export"
        ),
    }
    if gpu == "G4":
        # Empirical G4 session time derived from A100 14l1-16l1 receipts (91.5 min/session at W=6),
        # de-biased for A100 W=6 contention (1.3424x -> 68.16 min at 1W) before scaling by G4
        # per-core speedup (4.05 / 1.38 = 2.9348x) and G4 worker contention. Also report the
        # conservative unadjusted bound (treating A100 W=6 as 1.0x contention).
        g4_1w_session_seconds = A100_L1_SESSION_SECONDS_1W / G4_PER_CORE_SPEEDUP_VS_L4
        g4_worker_session_seconds = g4_1w_session_seconds * contention
        g4_cons_1w_seconds = A100_L1_SESSION_SECONDS_AT_W6 / G4_PER_CORE_SPEEDUP_VS_L4
        g4_cons_worker_seconds = g4_cons_1w_seconds * contention
        forecast.update(
            {
                "canary_namespace": canary_namespace,
                "checkpoint_interval": resolved_ckpt_interval,
                "a100_w6_contention_factor": round(A100_W6_CONTENTION_FACTOR, 6),
                "g4_measured_1w_seconds_per_decision": G4_SINGLETON_DECISION_SECONDS,
                "g4_measured_48w_decisions_per_second": G4_48W_DECISIONS_PER_SECOND,
                "g4_per_core_speedup_vs_l4": round(G4_PER_CORE_SPEEDUP_VS_L4, 6),
                "g4_node_speedup_vs_l4": round(G4_NODE_SPEEDUP_VS_L4, 6),
                "g4_expected_decisions_per_second_at_workers": round(
                    g4_decisions_per_second(resolved_workers), 4
                ),
                "g4_empirical_l1_session_minutes_1w": round(g4_1w_session_seconds / 60.0, 2),
                "g4_empirical_l1_session_minutes_at_workers": round(
                    g4_worker_session_seconds / 60.0, 2
                ),
                "g4_conservative_l1_session_minutes_1w": round(g4_cons_1w_seconds / 60.0, 2),
                "g4_conservative_l1_session_minutes_at_workers": round(
                    g4_cons_worker_seconds / 60.0, 2
                ),
            }
        )
    elif resolved_ckpt_interval > 0:
        forecast["checkpoint_interval"] = resolved_ckpt_interval
    publish_json(out / "forecast.json", forecast)
    rate = GPU_RATES[gpu]
    min_vcpu = MIN_VCPU_BY_GPU[gpu]
    for label in ("canary", "main"):
        base = 15 if label == "canary" else minutes
        unit = base / 60 * rate
        manifest = {
            "job_id": f"watch-replay-{label}-{out.name}",
            "commit": commit,
            "script": "research/watch_replay/job.sh",
            "source_paths": source_paths(),
            "device": "cpu",
            "lease_seconds": 300,
            "timeout_minutes": 2 * base,
            "output_dir": str(out / f"output-{label}"),
            "max_export_bytes": min(
                2 * 1024**3, max(4 * 1024**2, contracts[label]["count"] * 128 * 1024**2 + 1024**2)
            ),
            "requirements": {
                "gpu": gpu,
                "allowed_gpus": [gpu],
                "min_vcpu": min_vcpu,
                "est_minutes": base,
            },
            "budget": {
                "unit_cap": 2 * unit,
                "reserve_units": 2 * unit * 0.2,
                "rate_units_per_hour": rate,
                "other_assignments": {},
            },
            "watch_replay": {
                "allocation": identity(contracts[label]),
                "attempt": len(previous_receipts) + 1,
                "base_minutes": base,
                "base_unit_cap": unit,
                "base_reserve_units": unit * 0.2,
                "forecast": forecast,
            },
            "resume": {
                "unit_kind": "watch-session",
                "ledger_path": f"inputs/{label}-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "required_exports": ["sessions/operational.json", "sessions/contract.json"],
            "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in staged],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "PYTHONUNBUFFERED": "1",
                "SNAKE_DQN_DEVICE": "cpu",
                "WATCH_REPLAY_CONTRACT": f"{label}-contract.json",
                "WATCH_REPLAY_LEDGER": f"{label}-ledger.json",
                "WATCH_REPLAY_METADATA_SHA256": metadata_sha256,
                "WATCH_REPLAY_WORKERS": "1" if label == "canary" else str(resolved_workers),
            },
        }
        if resolved_ckpt_interval > 0:
            manifest["env"]["WATCH_REPLAY_CHECKPOINT_INTERVAL"] = (
                "0" if label == "canary" else str(resolved_ckpt_interval)
            )
        if ckpt_resume_rows:
            manifest["env"]["WATCH_REPLAY_CHECKPOINT_RESUME_SHAS"] = json.dumps(
                {row["dest"]: row["sha256"] for row in ckpt_resume_rows},
                sort_keys=True,
                separators=(",", ":"),
            )
        skipped = {row["index"] for row in verified} if label == "main" else set()
        if label == "canary":
            manifest["required_exports"].extend(
                f"sessions/{row['index']:06d}.watch.gz"
                for row in contracts[label]["rows"]
                if row["index"] not in skipped
            )
            manifest["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/watch_replay"
            )
            manifest["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(manifest),
            }
            manifest["required_exports"].append("branches.json")
            upload_report = check_uploads(manifest)
            publish_json(out / f"{label}-upload-preflight.json", upload_report)
            (out / "canary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            upload_report = check_uploads(manifest)
            publish_json(out / f"{label}-upload-preflight.json", upload_report)
            (out / "main-unsealed.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            if canary_receipt is not None:
                sealed = {
                    **manifest,
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
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--namespace", default="t34-watch-replay-fix-20261030-bank01")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), default="exact-stop-l2-prime")
    parser.add_argument("--gpu", choices=list(GPU_RATES), default="L4")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--checkpoint-resume", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--previous-receipts", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.out.resolve(),
                args.commit,
                args.benchmark,
                args.namespace,
                args.count,
                args.controller,
                args.canary_receipt,
                args.ledger,
                args.previous_receipts,
                gpu=args.gpu,
                workers=args.workers,
                checkpoint_resume=args.checkpoint_resume,
                checkpoint_interval=args.checkpoint_interval,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
