"""Q1 parity receipt comparison, retaining T23's exact fail-closed comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017 import q1_bank
from research.t23_r0_screen_20261017 import parity as t23

SCHEMA = "q1-parity/v1"
HORIZON = 5000
MIX_COUNTS = {"scripted": 2, "frozen": 1, "mixed": 1}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def runtime_environment() -> dict[str, Any]:
    """Identify the CPU runtime used for both the probe and scored shards."""
    import platform
    import sys

    import llvmlite
    import numba
    import numpy
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "cpu_threads": 1,
        "gpu_used": False,
    }


def validate_runtime_receipt(
    value: Mapping[str, Any],
    *,
    bank_sha256: str,
    b8_sha256: str,
    source_manifest: Mapping[str, str],
) -> None:
    """Validate a completed Mac/CPU parity comparison before Q1 scoring."""
    required = {
        "schema": "q1-runtime-parity-receipt/v1",
        "pass": True,
        "bank_sha256": bank_sha256,
        "b8_sha256": b8_sha256,
        "source_manifest": dict(source_manifest),
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("runtime parity receipt has wrong identity or failed comparison")
    environment = value.get("environment")
    if not isinstance(environment, Mapping) or not isinstance(environment.get("sha256"), str):
        raise ValueError("runtime parity receipt lacks evaluation-runtime environment identity")
    reference, runtime = value.get("reference"), value.get("runtime_artifact")
    if not isinstance(reference, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("runtime parity receipt lacks embedded artifacts")
    compared = compare_artifacts(reference, runtime)
    if not compared.get("pass") or value.get("comparison") != compared:
        raise ValueError("runtime parity receipt comparison is forged or stale")
    environment_hash = hashlib.sha256(canonical(runtime.get("runtime")).encode()).hexdigest()
    if environment.get("sha256") != environment_hash:
        raise ValueError("runtime parity environment hash does not bind runtime artifact")
    bank = q1_bank.load_world_bank()
    expected = {str(w["world_seed"]): w for w in bank["parity_worlds"]}
    source_hash = hashlib.sha256(canonical(dict(source_manifest)).encode()).hexdigest()
    for artifact in (reference, runtime):
        if (
            artifact.get("bank_sha256") != bank_sha256
            or artifact.get("source_sha256") != source_hash
            or artifact.get("profile_digest") != bank["profile_semantic_sha256"]
        ):
            raise ValueError("parity trace source/bank/profile is not the registered Q1 identity")
        if (
            artifact.get("runtime", {}).get("cpu_threads") != 1
            or artifact.get("runtime", {}).get("gpu_used") is not False
        ):
            raise ValueError("parity trace must use one CPU thread")
        if {row["world_id"] for row in artifact["worlds"]} != set(expected):
            raise ValueError("parity trace does not use the reserved Q1 worlds")
        for row in artifact["worlds"]:
            world = expected[row["world_id"]]
            if (
                row["world_seed"] != world["world_seed"]
                or row["mix"] != world["mix"]
                or row["roster_digest"] != world["roster_semantic_sha256"]
            ):
                raise ValueError("parity trace roster/world mismatch")
            if (
                row["world_digest"]
                != hashlib.sha256(canonical(row["frame_digests"]).encode()).hexdigest()
            ):
                raise ValueError("parity frame trace hash mismatch")
    if reference["runtime"].get("system") != "Darwin":
        raise ValueError("Q1 parity reference must originate on the Mac")
    for left, right in zip(reference["worlds"], runtime["worlds"]):
        # The historical comparator diagnoses Q ties, while masks are exact contracts.
        if [x.get("mask") for x in left["actions"]] != [x.get("mask") for x in right["actions"]]:
            raise ValueError("parity action masks differ")


def compare_artifacts(mac: Mapping[str, Any], colab: Mapping[str, Any]) -> dict[str, Any]:
    """Require Q1 bindings, then apply the historical action/frame comparator."""
    errors = []
    for name, value in (("mac", mac), ("colab", colab)):
        if value.get("schema") != SCHEMA:
            errors.append(f"{name}: wrong Q1 parity schema")
        if value.get("bank_namespace") != q1_bank.NAMESPACE:
            errors.append(f"{name}: wrong Q1 bank namespace")
    # Translate only the envelope; T23's comparator is the fixed low-level oracle.
    left = {**mac, "schema": t23.SCHEMA}
    right = {**colab, "schema": t23.SCHEMA}
    result = t23.compare_artifacts(left, right)
    result["schema"] = SCHEMA
    result["pass"] = not errors and bool(result.get("pass"))
    result["errors"] = errors + list(result.get("errors", []))
    return result


def run_sample(bank_path: Path, b8_path: Path, out: Path) -> dict[str, Any]:
    """Materialize Q1's four reserved worlds with T23's frame/action oracle."""
    if t23.sha256_file(b8_path) != t23.B8_SHA256:
        raise ValueError("released B8 SHA mismatch")
    bank = q1_bank.load_world_bank(bank_path)
    worlds = list(bank["parity_worlds"])
    if {m: sum(w["mix"] == m for w in worlds) for m in MIX_COUNTS} != MIX_COUNTS:
        raise ValueError("Q1 parity allocation is not 2/1/1")
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t23_r0_screen_20261017 import evaluate

    context = evaluate._context(1)
    actions_worlds = []
    with t23.single_cpu_threads():
        for world in worlds:
            actions, frames = [], []
            with t23._record_decisions(actions, frames):
                result = evaluate.evaluate_world(
                    world, b8_path, context["lookup"][gid.FRP3_S12[1]], context
                )
            actions_worlds.append(
                {
                    "world_id": str(world.get("world_id", world["world_seed"])),
                    "world_seed": world["world_seed"],
                    "mix": world["mix"],
                    "roster_digest": world["roster_semantic_sha256"],
                    "world_digest": t23._digest(frames),
                    "frame_digests": frames,
                    "actions": actions,
                    "record": t23._comparable_record(result),
                }
            )
    artifact = {
        "schema": SCHEMA,
        "bank_namespace": q1_bank.NAMESPACE,
        "horizon": HORIZON,
        "bank_sha256": hashlib.sha256(canonical(bank).encode()).hexdigest(),
        "b8_sha256": t23.B8_SHA256,
        "profile_digest": bank["profile_semantic_sha256"],
        "source_sha256": hashlib.sha256(
            canonical(
                __import__(
                    "research.q0q1_20261017.q1_evaluate", fromlist=["source_manifest"]
                ).source_manifest()
            ).encode()
        ).hexdigest(),
        "worlds": actions_worlds,
        "runtime": runtime_environment(),
    }
    out.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", action="store_true", help="create the Mac reference artifact"
    )
    parser.add_argument("--verify-reference", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reference and args.verify_reference is not None:
        parser.error("choose reference creation or runtime verification")
    if args.reference and __import__("platform").system() != "Darwin":
        parser.error("Mac reference creation requires Darwin")
    if not args.reference and args.verify_reference is None:
        raise SystemExit("Q1 parity reference creation requires explicit --reference")
    # The historical runner supplies the action/frame oracle; Q1 wraps the
    # resulting artifact with its own namespace so it cannot be confused with
    # a T23 reference.
    runtime = run_sample(args.bank, args.b8, args.out)
    if args.verify_reference is None:
        return 0
    reference = json.loads(args.verify_reference.read_text())
    compared = compare_artifacts(reference, runtime)
    from research.q0q1_20261017.q1_evaluate import source_manifest

    receipt = {
        "schema": "q1-runtime-parity-receipt/v1",
        "pass": compared["pass"],
        "bank_sha256": hashlib.sha256(
            canonical(q1_bank.load_world_bank(args.bank)).encode()
        ).hexdigest(),
        "b8_sha256": t23.B8_SHA256,
        "source_manifest": source_manifest(),
        "environment": {
            "sha256": hashlib.sha256(canonical(runtime["runtime"]).encode()).hexdigest(),
            "runtime": runtime["runtime"],
        },
        "reference": reference,
        "runtime_artifact": runtime,
        "comparison": compared,
    }
    args.out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
