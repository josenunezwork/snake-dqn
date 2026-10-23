"""Run independent A3 assay units; retain verified units across mechanical attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export

UNITS = [
    (mode, variant)
    for variant in ("baseline", "optimized")
    for mode in ("cached", "collection", "cycle")
] + [("comparison", "baseline")]
SCHEMA = "a3-p1-ceiling/v1"


def allocation(control: dict[str, Any]) -> str:
    """Bind namespace, frozen unit order and source without inspecting outcomes."""
    return hashlib.sha256(json.dumps(control, sort_keys=True).encode()).hexdigest()


def filename(control: dict[str, Any], index: int) -> str:
    return f"{control['namespace']}--{index:03d}.json"


def read_ledger(path: Path, control: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Refuse foreign allocations and duplicate, malformed or invented unit IDs."""
    value = json.loads(path.read_text())
    if (
        value.get("schema") != SCHEMA
        or value.get("namespace") != control["namespace"]
        or value.get("allocation") != allocation(control)
    ):
        raise ValueError("foreign A3 ledger namespace/allocation")
    found = {}
    for row in value["units"]:
        index = row.get("index")
        if (
            type(index) is not int
            or not 0 <= index < len(UNITS)
            or index in found
            or row.get("name") != filename(control, index)
            or type(row.get("size")) is not int
            or row["size"] < 1
            or len(row.get("sha256", "")) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("invalid/duplicate A3 ledger unit")
        found[index] = row
    return found


def make_ledger(control: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
    """Build from names, sizes and verified hashes only; never open result payloads."""
    rows = []
    seen = set()
    names = {filename(control, i): i for i in range(len(UNITS))}
    for path in paths:
        if path.name not in names or names[path.name] in seen:
            raise ValueError("foreign or duplicate A3 unit filename")
        verify_export(path)
        index = names[path.name]
        seen.add(index)
        rows.append(
            {
                "index": index,
                "name": path.name,
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    return {
        "schema": SCHEMA,
        "namespace": control["namespace"],
        "allocation": allocation(control),
        "units": sorted(rows, key=lambda r: r["index"]),
    }


def run(
    control: dict[str, Any],
    ledger: Path,
    checkpoint: Path,
    output: Path,
    *,
    tiny: bool,
    device: str,
) -> None:
    """Run only missing independent seeded units; no trajectory is resumed."""
    verified = read_ledger(ledger, control)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "receipt.json").exists() or (output / "units").exists():
        raise ValueError("A3 output must be fresh")
    temperature = None
    if not tiny and len(verified) < len(UNITS):
        from types import SimpleNamespace

        from research.bl_ppo.run import _calibrated_temperature

        temperature = _calibrated_temperature(
            SimpleNamespace(
                temperature=None,
                preflight=False,
                processes=12,
                worlds=128,
                rollout_steps=128,
                seed=control["assay_seed"],
                roster="frozen-equal-mixes",
                long_bodies=False,
                native_v8=True,
                b8_checkpoint=checkpoint,
                device=device,
                p1_runtime=True,
            )
        )
        export_json(
            output / "temperature.json",
            {
                "rule": control["temperature_rule"],
                "temperature": temperature,
                "namespace": control["namespace"],
                "allocation": allocation(control),
            },
        )
    for index, (mode, variant) in enumerate(UNITS):
        if index in verified:
            continue
        work = output / "diagnostics" / str(index)
        command = [
            sys.executable,
            "-m",
            "research.a3_p1_ceiling.benchmark",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(work),
            "--device",
            device,
            "--unit-id",
            str(index),
            "--mode",
            mode,
            "--variant",
            variant,
            "--repeats",
            "3",
            "--seed",
            str(control["assay_seed"]),
        ]
        if temperature is not None:
            command += ["--temperature", str(temperature)]
        if tiny:
            command.append("--tiny")
        print(
            json.dumps({"progress": index, "total": len(UNITS), "mode": mode, "variant": variant}),
            flush=True,
        )
        subprocess.run(command, check=True)
        payload = work / f"unit-{index}.json"
        verify_export(payload)
        export_timing = work / "export-timing.json"
        verify_export(export_timing)
        export_json(
            output / "units" / filename(control, index),
            {
                "schema": SCHEMA,
                "namespace": control["namespace"],
                "allocation": allocation(control),
                "index": index,
                "mode": mode,
                "variant": variant,
                "tiny": tiny,
                "result": json.loads(payload.read_text()),
                "temperature": temperature,
                "export_timing": json.loads(export_timing.read_text()),
            },
        )
    export_json(
        output / "receipt.json",
        {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "namespace": control["namespace"],
            "allocation": allocation(control),
            "verified_skipped": sorted(verified),
            "completed_indices": list(range(len(UNITS))),
            "tiny": tiny,
            "scoring": "Mac aggregator only; remote diagnostics sealed",
        },
    )


def aggregate(control: dict[str, Any], ledger_path: Path) -> dict[str, Any]:
    """The sole Mac-side look, over exactly the frozen union of verified units."""
    ledger = read_ledger(ledger_path, control)
    if set(ledger) != set(range(len(UNITS))):
        raise ValueError("INCOMPLETE A3 allocation")
    results = []
    for index, row in sorted(ledger.items()):
        path = Path(row["path"])
        verify_export(path)
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError("A3 verified ledger payload changed")
        value = json.loads(path.read_text())
        if (
            value.get("schema") != SCHEMA
            or value.get("namespace") != control["namespace"]
            or value.get("allocation") != allocation(control)
            or value.get("index") != index
            or (value.get("mode"), value.get("variant")) != UNITS[index]
            or value.get("tiny") is not False
        ):
            raise ValueError("foreign/tiny A3 result")
        result = value.get("result", {})
        if (
            result.get("schema") != "a3-p1-learner-ceiling/v1"
            or result.get("shape") != {"worlds": 128, "rollout_steps": 128, "transitions": 16384}
            or result.get("mode") != UNITS[index][0]
            or result.get("variant") != UNITS[index][1]
            or result.get("actor_enabled") is not True
            or result.get("arm") != "F"
            or result.get("device") != "cuda"
            or len(result.get("phase_timers_seconds", [])) != 3
        ):
            raise ValueError("INVALID A3 nested identity/shape/sample count")
        if result.get("status") != "COMPLETE":
            raise ValueError("INCOMPLETE A3 actor update (including KL skip)")
        if UNITS[index][0] != "collection" and any(
            row.get("adam_steps") != 2 for row in result["phase_timers_seconds"]
        ):
            raise ValueError("INCOMPLETE A3 actor Adam-step count")
        results.append(value)
    baseline = results[0]["result"]["cached_learning"]["median_seconds"]
    optimized = results[3]["result"]["cached_learning"]["median_seconds"]
    comparison = results[6]["result"].get("comparison") or {}
    return {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "namespace": control["namespace"],
        "allocation": allocation(control),
        "units": results,
        "kill_seconds_per_rollout": 1.6384,
        "baseline_cached_seconds": baseline,
        "optimized_cached_seconds": optimized,
        "wp5": "PAUSE_LEARNER_FIX" if baseline > 1.6384 else "CEILING_NOT_FALSIFIED",
        "optimization": (
            "NUMERICAL_COMPARISON_PASS"
            if comparison.get("passed") is True
            else "REJECT_UNEXPLAINED_GRADIENT_CHANGE"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "ledger", "aggregate"])
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_intermixed_args()
    control = json.loads(args.control.read_text())
    if args.action == "run":
        run(control, args.ledger, args.checkpoint, args.output, tiny=args.tiny, device=args.device)
    elif args.action == "ledger":
        export_json(args.output, make_ledger(control, args.paths))
    else:
        export_json(args.output, aggregate(control, args.ledger))


if __name__ == "__main__":
    main()
