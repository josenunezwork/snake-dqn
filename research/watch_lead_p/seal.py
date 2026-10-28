"""Create a fresh, source-bound P packet before outcomes; never dispatch it."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from research.bl_served_search.exports import digest
from research.watch_lead_p.admission import exclusion_audit
from research.watch_lead_p.protocol import NAMESPACE, SEEDS, STUDY
from research.watch_lead_p.resume import empty_ledger, read_ledger
from research.watch_lead_p.run import WORKERS

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts/t34-watch-lead-1028"
CHECKPOINTS = ROOT.parent / "snake-dqn/saved_snakes"


def write(path: Path, data: object) -> None:
    with path.open("x") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
        handle.write("\n")


def seal(
    pin: str,
    broker: Path,
    runtime: Path,
    benchmark: Path,
    attempt: int = 1,
    ledger: Path | None = None,
    units_root: Path | None = None,
    revision: int = 0,
) -> None:
    broker = broker.expanduser().resolve()
    runtime = runtime.expanduser().resolve()
    benchmark = benchmark.expanduser().resolve()
    ledger = ledger.expanduser().resolve() if ledger is not None else None
    units_root = units_root.expanduser().resolve() if units_root is not None else None
    if (
        not re.fullmatch(r"[0-9a-f]{40}", pin)
        or subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{pin}^{{commit}}"], cwd=ROOT, text=True
        ).strip()
        != pin
    ):
        raise ValueError("P runtime pin must be an existing full commit SHA")
    # Read both dependency lists at the pin and prove the complete shipped
    # runtime agrees locally before deriving a ledger or writing packet files.
    inventory = "research/watch_lead_p/science-sources.json"
    science = json.loads(
        subprocess.check_output(["git", "show", f"{pin}:{inventory}"], cwd=ROOT, text=True)
    )
    runtime_inventory = "research/watch_lead_p/runtime-sources.json"
    runtime_sources = json.loads(
        subprocess.check_output(["git", "show", f"{pin}:{runtime_inventory}"], cwd=ROOT, text=True)
    )
    paths = list(dict.fromkeys([inventory, *science, runtime_inventory, *runtime_sources]))
    changed = subprocess.run(["git", "diff", "--quiet", pin, "--", *paths], cwd=ROOT).returncode
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=ROOT,
        text=True,
    ).strip()
    if changed or dirty:
        raise ValueError("P science/runtime working tree differs from runtime pin")
    if type(attempt) is not int or not 1 <= attempt <= 99:
        raise ValueError("fresh packet attempt must be 1..99")
    if type(revision) is not int or not 0 <= revision <= 99:
        raise ValueError("packet revision must be 0..99")
    if (ledger is None) != (units_root is None):
        raise ValueError("completion requires both ledger and units root")
    verified = read_ledger(ledger, units_root, recipe_root=ROOT) if ledger is not None else {}
    if ledger is not None and not 0 < len(verified) < 32:
        raise ValueError("completion requires 1..31 verified units")
    measured = json.loads(benchmark.read_text())
    if measured["fixture_index"] != 32 or not measured["excluded_from_all_32_sessions"]:
        raise ValueError("forecast requires the excluded natural fixture")
    if measured["workers"] != WORKERS or math.ceil(measured["projected_seconds"] / 60) > 240:
        raise ValueError("measured forecast does not fit the ruled base envelope")
    packet_name = f"packet-{attempt:02d}"
    attempt_name = f"attempt-{attempt:02d}"
    relative = Path("research/watch_lead_p") / packet_name
    if revision:
        relative /= f"reseal-{revision:02d}"
    packet = ROOT / relative
    packet.mkdir(parents=True, exist_ok=False)
    target = broker / relative
    write(
        packet / "verified-ledger.json",
        json.loads(ledger.read_text()) if ledger else empty_ledger(ROOT),
    )
    write(packet / "exclusion-audit.json", exclusion_audit())
    measured["receipt_path"] = str(benchmark)
    measured["receipt_sha256"] = digest(benchmark)
    write(packet / "forecast.json", measured)
    kinds = [
        ("frozen-checkpoint", CHECKPOINTS / "b8_s3_t7_20261016.pth", "b8-s3.pth"),
        ("frozen-opponent", CHECKPOINTS / "frp3_m3_s12_u60000_20261005.pth", "frp3-s12.pth"),
        ("registry", broker / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json", "registry.json"),
        ("verified-ledger", target / "verified-ledger.json", "verified-ledger.json"),
    ]
    inputs = []
    for kind, path, dest in kinds:
        local = ROOT / path.relative_to(broker) if path.is_relative_to(broker) else path
        inputs.append({"kind": kind, "path": str(path), "sha256": digest(local), "dest": dest})
    for record in verified.values():
        for suffix in ("", ".sha256", ".complete"):
            path = (units_root / (record["path"] + suffix)).resolve()
            inputs.append(
                {
                    "kind": "verified-unit",
                    "study": STUDY,
                    "path": str(path),
                    "sha256": digest(path),
                    "dest": record["path"] + suffix,
                }
            )
    identity = f"sandbox-p-watch-lead-20261028-{attempt:02d}"
    manifest = {
        "job_id": identity,
        "commit": pin,
        "script": "research/watch_lead_p/run_job.sh",
        "device": "cpu",
        "source_paths": runtime_sources,
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": math.ceil(measured["projected_seconds"] / 60),
        },
        "timeout_minutes": 480,
        "output_dir": str(ARTIFACTS / attempt_name / "output"),
        "max_export_bytes": 64 * 1024 * 1024,
        "lease_seconds": 180,
        "inputs": [{k: r[k] for k in ("path", "sha256", "dest")} for r in inputs],
        "env": {
            "SNAKE_DQN_DEVICE": "cpu",
            "COLAB_CLI_REQUIREMENTS": "research/watch_lead_p/requirements.txt",
        },
        "requirements_file": "research/watch_lead_p/requirements.txt",
        "required_exports": ["watch-lead/inventory.json"]
        + [f"watch-lead/units/session-{i:02d}.json" for i in range(32) if i not in verified],
        "budget": {
            "unit_cap": 12.32,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "paired-Watch6-session",
            "ledger_path": "verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sandbox_admission": {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str(target / "sandbox.json"),
        },
        "sandbox_source_delivery": "background-sparse-v1",
        "technical_envelope": {
            "base_seconds": 14400,
            "effective_seconds": 28800,
            "base_unit_cap": 6.16,
            "effective_unit_cap": 12.32,
            "base_reserve_units": 0,
            "effective_reserve_units": 0,
            "workers": WORKERS,
            "forecast_seconds": measured["projected_seconds"],
        },
    }
    if manifest["requirements"]["est_minutes"] > 240:
        raise ValueError("measured forecast does not fit the ruled base envelope")
    write(packet / "dispatch-manifest.json", manifest)
    sandbox = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": [{k: v for k, v in r.items() if k != "dest"} for r in inputs]
        + [{"kind": "worlds", "study": STUDY, "indices": list(range(32)), "seeds": list(SEEDS)}],
        "salt": NAMESPACE,
        "endpoint": {"kind": "steps", "value": 32 * 3 * 6000},
        "timeout": {"base_seconds": 14400, "seconds": 28800, "purpose": "runaway-guard"},
        "excluded_studies": [
            "F5",
            "F4-fold-0",
            "R30",
            "strict",
            "probe",
            "serving",
            "D4096",
            "TRAIN-01-monitor",
            "TRAIN-01-nomination",
            "t34-cf-1028",
            "engineering-indices-32-35",
        ],
        "owner_rule": (
            "docs/research/governance_a100_sandbox_20261026/ADOPTED.md; "
            "T34 RULING 2026-10-28 section 4 P"
        ),
        "outputs": {
            "label": "simulator-D",
            "paths": [manifest["output_dir"]],
            "broker_manifest": {
                "path": str(target / "dispatch-manifest.json"),
                "sha256": digest(packet / "dispatch-manifest.json"),
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    # One top-level field per physical line; braces share first and last lines.
    lines = [json.dumps(k) + ": " + json.dumps(v, sort_keys=True) for k, v in sandbox.items()]
    (packet / "sandbox.json").write_text("{" + ",\n".join(lines) + "}\n")
    command = (
        f"{broker}/tools/ops/dispatch.sh {identity} {target}/dispatch-manifest.json "
        f"--broker {broker} --cwd {runtime} --state-dir {ARTIFACTS}/{attempt_name}/state "
        f"--dispatch-txt {target}/DISPATCH.txt"
    )
    (packet / "DISPATCH.txt").write_text(
        f"P prepared for operator dispatch; no allocation performed.\nRuntime pin: {pin}\n"
        f"Identity: {identity}\n"
        "Science: 32 complete paired 6000-frame Watch6 sessions; P48 sole primary.\n"
        f"Projected wall seconds: {measured['projected_seconds']:.3f}; "
        "base/effective 14400/28800.\n"
        "Twelve arm workers, session cohorts of 12 with P0/P48/bare; "
        "each session W1/K1; L4 >=12 vCPUs.\n"
        "Only complete three-arm paired sessions are exported or resumed. CPU science only.\n"
        "Inputs remain sealed; remote exports are diagnostics, never a terminal decision.\n"
        "One original + at most two manual completion attempts after verified mechanical failure,\n"
        "stop_verified and missing-unit accounting, fresh packet/output/receipt, "
        "same allocation and N.\n"
        "Never rerun verified units or infer timing/service qualification. "
        "No automatic expansion or O1.\n"
        "All commits local by owner instruction; source is delivered by sparse fallback.\n\n"
        f"Offline validation:\n{command} --dry-run\n\nExact operator dispatch command:\n{command}\n"
    )
    write(
        packet / "seal.json", {p.name: digest(p) for p in sorted(packet.iterdir()) if p.is_file()}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--revision", type=int, default=0)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--units-root", type=Path)
    args = parser.parse_args()
    seal(
        args.pin,
        args.broker,
        args.runtime,
        args.benchmark,
        args.attempt,
        args.ledger,
        args.units_root,
        args.revision,
    )
