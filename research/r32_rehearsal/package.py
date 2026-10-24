"""Prepare published-pin CPU packets; no allocator, dispatch or automatic retries."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.r32_rehearsal.lifecycle import allocation, empty_ledger

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "src",
    "research/r32_rehearsal",
    "research/bl_residual",
    "research/bl_ppo/__init__.py",
    "research/bl_ppo/exports.py",
    "research/a100_a2/__init__.py",
    "research/a100_a2/model.py",
    "research/colab_cli",
    "tests/test_bl_residual_f4_analyzer.py",
]
REQUIREMENTS = "research/r32_rehearsal/requirements.txt"
REGISTRY = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json")


def source_inventory() -> dict[str, str]:
    result = {}
    for name in SOURCES:
        path = ROOT / name
        paths = path.rglob("*") if path.is_dir() else [path]
        for item in sorted(paths):
            if item.is_file() and item.suffix in (".py", ".sh", ".json", ".txt"):
                result[str(item.relative_to(ROOT))] = digest(item)
    return result


def prepare_inputs(output: Path, *, tiny: bool = False) -> dict:
    """Freeze all scenarios and the independent oracle before computing outputs."""
    from research.r32_rehearsal import decisions, statistics

    units = [{"kind": "decision", "case": name} for name in decisions.case_names()]
    if tiny:
        units = units[:3]
    else:
        units += [{"kind": "lifecycle"}]
        units += [{"kind": "statistics", **unit} for unit in statistics.frozen_units()]
    output.mkdir(parents=True, exist_ok=False)
    specs = {}
    for canary in (False, True):
        selected = units[:3] if canary else units
        spec = {
            "schema": "r32-allocation/v1",
            "synthetic": True,
            "namespace": "r32-synthetic-20261023-"
            + ("canary" if canary else "tiny" if tiny else "cpu"),
            "sources": source_inventory(),
            "scientific_stop_required": [0],
            "units": [
                {"index": i, "name": f"unit-{i:04d}.json", **unit}
                for i, unit in enumerate(selected)
            ],
        }
        ledger_name = "canary-ledger.json" if canary else "verified-ledger.json"
        export_json(output / ledger_name, empty_ledger(spec))
        spec["ledger_sha256"] = digest(output / ledger_name)
        export_json(output / ("canary-spec.json" if canary else "spec.json"), spec)
        specs["canary" if canary else "main"] = {
            "allocation": allocation(spec),
            "units": len(selected),
        }
    return specs


def make_pack(output: Path, commit: str, *, preflight: Path, consumer_preflight: Path) -> dict:
    """Require qualified published source and a successful tiny actual entrypoint."""
    from tools.ops.import_preflight import check as imports
    from tools.ops.qualification import check as qualified

    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (
        commit != head
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--", *SOURCES], cwd=ROOT, text=True
        ).strip()
    ):
        raise ValueError("clean published current source required")
    tree = qualified(ROOT, REGISTRY, commit)
    from research.bl_ppo.exports import verify_export

    verify_export(preflight)
    proof = json.loads(preflight.read_text())
    if proof.get("done") != proof.get("expected") or proof.get("done", 0) < 3:
        raise ValueError("three actual tiny units required")
    preflight_inputs = preflight.parent.parent / "inputs" / "spec.json"
    if (
        not preflight_inputs.exists()
        or json.loads(preflight_inputs.read_text())["sources"] != source_inventory()
    ):
        raise ValueError("tiny preflight source identity is stale")
    verify_export(consumer_preflight)
    consumer = json.loads(consumer_preflight.read_text())
    required_consumer_sources = {
        "research/r32_rehearsal/consumer.py",
        "research/bl_residual/data.py",
        "research/bl_residual/fit.py",
        "research/bl_residual/model.py",
        "src/model/ego2s_network.py",
        "web/backend/ego2s_policy.py",
    }
    if (
        consumer.get("schema") != "t34-r32-consumer-rehearsal/v1"
        or consumer.get("synthetic") is not True
        or consumer.get("production_admission_refusal", {}).get("function")
        != "research.bl_residual.heldout.validate_candidate"
        or not consumer["production_admission_refusal"].get("error")
        or set(consumer.get("source_sha256", {})) != required_consumer_sources
    ):
        raise ValueError("actual synthetic consumer/refusal trace required")
    if any(source_inventory().get(name) != sha for name, sha in consumer["source_sha256"].items()):
        raise ValueError("consumer preflight source is stale")
    for key, filename in (
        ("zero", "zero-residual-fused.pth"),
        ("deterministic_nonzero", "nonzero-residual-fused.pth"),
        ("isolated_q_tie", "tie-head-fused.pth"),
    ):
        if (
            verify_export(consumer_preflight.parent / filename)
            != consumer["checkpoints"][key]["sha256"]
        ):
            raise ValueError("consumer exported checkpoint identity differs")
    clean = imports(
        ROOT, commit, SOURCES, REQUIREMENTS, ["research.r32_rehearsal.job:run_portable"]
    )
    output.mkdir(parents=True)
    specs = prepare_inputs(output / "inputs")
    export_json(output / "clean-import-preflight.json", clean)
    export_json(
        output / "mac-preflight.json",
        {"path": str(preflight.resolve()), "sha256": digest(preflight), **proof},
    )
    export_json(
        output / "mac-consumer-plan.json",
        {
            "schema": "r32-mac-consumer-plan/v1",
            "commit": commit,
            "source_repo": str(ROOT),
            "cwd_contract": "ops-managed detached worktree at this commit; never ambient cwd",
            "synthetic": True,
            "scientific_result": False,
            "consumer_preflight_path": str(consumer_preflight.resolve()),
            "consumer_preflight_sha256": digest(consumer_preflight),
            "output_dir": str((output / "mac-consumer-output").resolve()),
            "command": [
                "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python",
                "-m",
                "research.r32_rehearsal.consumer",
                "--b8",
                "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth",
                "--output",
                str((output / "mac-consumer-output").resolve()),
                "--mac-smoke",
            ],
            "wall_ceiling_seconds": 2700,
            "units": 0,
            "threads": 1,
            "nice_floor": 15,
            "resource_admission": (
                "shared research.mac_lease; AC/open-lid ops wrapper; admitted gap"
            ),
            "consumer_readiness": "BLOCKED_STRICT_T34_ADAPTER_AND_REGISTRY_INTERFACES",
            "dispatch_prefix": [
                str(ROOT / "tools/ops/dispatch.sh"),
                "--mac",
                "--repo",
                str(ROOT),
                f"r32-mac-{commit[:8]}",
                commit,
                "--",
            ],
        },
    )
    inventory = [
        {
            "path": str((output / "inputs" / name).resolve()),
            "dest": name,
            "sha256": digest(output / "inputs" / name),
        }
        for name in ("spec.json", "verified-ledger.json", "canary-spec.json", "canary-ledger.json")
    ]
    for mode, wall, estimate in (("canary", 20, 5), ("main", 268, 115)):
        manifest = {
            "job_id": f"r32-cpu-{mode}-{commit[:8]}",
            "commit": commit,
            "script": "research/r32_rehearsal/colab.sh",
            "source_paths": SOURCES,
            "device": "cpu",
            "requirements": {
                "gpu": "cpu",
                "allowed_gpus": ["cpu"],
                "min_vcpu": 1,
                "est_minutes": estimate,
            },
            "timeout_minutes": wall,
            "output_dir": str((output / f"{mode}-output").resolve()),
            "inputs": inventory,
            "required_exports": ["receipt.json"],
            "lease_seconds": 180,
            "env": {"R32_MODE": mode, "R32_DEVICE": "cpu", "COLAB_CLI_REQUIREMENTS": REQUIREMENTS},
            "budget": {
                "unit_cap": wall * 0.08 / 60,
                "rate_units_per_hour": 0.08,
                "reserve_units": wall * 0.08 / 360,
                "other_assignments": {},
            },
            "resume": {
                "unit_kind": "excluded-fixture-scenario",
                "ledger_path": "inputs/"
                + ("canary-ledger.json" if mode == "canary" else "verified-ledger.json"),
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "max_export_bytes": 64 * 1024**2,
            "r32": {
                "synthetic": True,
                "scientific_result": False,
                "qualified_tree": tree,
                "base_timeout_minutes": wall / 2,
                "effective_timeout_minutes": wall,
                "base_unit_cap": wall * 0.08 / 120,
                "effective_unit_cap": wall * 0.08 / 60,
                "base_reserve_units": wall * 0.08 / 720,
                "effective_reserve_units": wall * 0.08 / 360,
                "shared_series_effective_minutes": 288,
                "shared_series_effective_units": 0.384,
                "forecast_note": (
                    "Ruling allocation forecast including setup/export; tiny timings in "
                    "mac-preflight.json; rates are planning, operator verifies debit/capacity."
                ),
            },
        }
        if mode == "canary":
            manifest["canary"] = {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": manifest["script"],
            }
            manifest["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/r32_rehearsal"
            )
        else:
            manifest["canary"] = {
                "receipt": {
                    "path": str((output / "canary-output/receipt.json").resolve()),
                    "sha256": "0" * 64,
                }
            }
        export_json(output / f"{mode}-template.json", manifest)
    (output / "DISPATCH_CANARY.txt").write_text(
        "R32 CPU excluded engineering canary only. Claude verifies account debit, free "
        "assignment, capacity and source proof before dispatch. No release authority.\n"
    )
    (output / "DISPATCH_MAIN.txt").write_text(
        "HOLD pending current-pin CPU canary receipt and actual stop verification; "
        "use package admit before dispatch.\n"
    )
    value = {
        "schema": "r32-pack/v1",
        "commit": commit,
        "qualified_tree": tree,
        "specs": specs,
        "scientific_result": False,
        "status": "PREPARED_NOT_DISPATCHED",
        "canary_sha256": digest(output / "canary-template.json"),
        "main_sha256": digest(output / "main-template.json"),
        "preflight_sha256": digest(preflight),
        "clean_import_sha256": digest(output / "clean-import-preflight.json"),
    }
    export_json(output / "pack.json", value)
    return value


def admit(pack: Path, receipt: Path, output: Path) -> dict:
    """Bind verified same-pin proof in a fresh manifest, without dispatching."""
    from research.colab_cli.canary import check

    manifest = copy.deepcopy(json.loads((pack / "main-template.json").read_text()))
    manifest["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": digest(receipt)}}
    check(manifest, ROOT)
    from tools.ops.spend import read_spend

    spent = read_spend(receipt)
    if (
        spent.own_rate_units_per_hour <= 0
        or spent.elapsed_seconds > 1200
        or spent.units > 20 * 0.08 / 60
    ):
        raise ValueError("canary exceeded frozen series slice; no main admission")
    export_json(output, manifest)
    instruction = output.with_suffix(".DISPATCH.txt")
    with instruction.open("x") as handle:
        handle.write(
            "R32 CPU excluded engineering main only; same-pin canary verified. Claude verifies "
            "free funded assignment and current debit/capacity; no automatic retries or "
            "release authority.\n"
        )
    return {
        "manifest": str(output),
        "sha256": digest(output),
        "dispatch_txt": str(instruction),
        "dispatched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tiny = sub.add_parser("tiny-inputs")
    tiny.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--commit", default="HEAD")
    build.add_argument("--preflight", type=Path, required=True)
    build.add_argument("--consumer-preflight", type=Path, required=True)
    bind = sub.add_parser("admit")
    bind.add_argument("--pack", type=Path, required=True)
    bind.add_argument("--receipt", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    ledger = sub.add_parser("ledger")
    ledger.add_argument("--inputs", type=Path, required=True)
    ledger.add_argument("--attempt", type=Path, action="append", required=True)
    ledger.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "tiny-inputs":
        result = prepare_inputs(args.out, tiny=True)
    elif args.command == "build":
        result = make_pack(
            args.out,
            args.commit,
            preflight=args.preflight,
            consumer_preflight=args.consumer_preflight,
        )
    elif args.command == "admit":
        result = admit(args.pack, args.receipt, args.out)
    else:
        from research.r32_rehearsal.lifecycle import build_ledger

        result = build_ledger(
            args.attempt, json.loads((args.inputs / "spec.json").read_text()), args.out
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
