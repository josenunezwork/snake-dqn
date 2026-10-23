"""A2 real job entrypoint: immutable completed fits; no remote scoring look."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.a100_a2.protocol import (
    B8_SHA,
    NAMESPACE,
    allocation,
    allocation_hash,
    publish_fit,
    read_ledger,
)
from research.bl_ppo.exports import digest, export_json


def synthetic_table() -> dict[str, np.ndarray]:
    """Excluded tiny fixture: five worlds, genuine model forward/backward plumbing."""
    rng = np.random.default_rng(22)
    n = 10
    alive = rng.integers(0, 301, (n, 6))
    died = alive < 180
    alive[~died] = 300
    return {
        "local": rng.integers(0, 256, (n, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 256, (n, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(n, 60)).astype(np.float32),
        "world": np.asarray([f"synthetic-{i // 2}" for i in range(n)]),
        "fold": np.arange(n) // 2,
        "legal": np.ones((n, 6), bool),
        "observed": np.ones((n, 6), bool),
        "died": died,
        "alive_steps": alive,
        "decisive": np.ones(n, bool),
        "a_star": np.zeros(n, np.int64),
        "q": np.zeros((n, 6), np.float32),
    }


def hardware(device: str, smoke: bool) -> dict[str, Any]:
    """A100 family admission is narrowed to registered 40GB hardware before labels."""
    if smoke and device == "cpu":
        return {"kind": "cpu", "synthetic": True}
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("registered A2 requires CUDA A100-40")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name or not 39_000_000_000 <= props.total_memory <= 45_000_000_000:
        raise ValueError(f"A2 refuses non-A100-40: {props.name} {props.total_memory}")
    return {
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
    }


def _run(input_dir: Path, output: Path, device: str, mode: str) -> dict[str, Any]:
    """Perform final fits or the excluded same-loop canary without aggregating outcomes."""
    from research.a100_a2.run import fit_unit

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    hw = hardware(device, smoke=mode == "smoke")
    control = json.loads((input_dir / "control.json").read_text())
    if control.get("schema") != "a2-control/v1":
        raise ValueError("unrecognized control")
    for name, expected in control["input_hashes"].items():
        if Path(name).name != name or digest(input_dir / name) != expected:
            raise ValueError("input digest mismatch")
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("B8 checkpoint identity differs")
    if (
        digest(Path(__file__).with_name("PREREGISTRATION.md"))
        != control["bindings"]["preregistration"]
    ):
        raise ValueError("registration changed")
    if (
        digest(Path(__file__).with_name("SEMANTICS_ERRATUM.md"))
        != control["bindings"]["semantics_erratum"]
    ):
        raise ValueError("source semantics clarification changed")
    if allocation_hash(control["bindings"]) != control["allocation"]:
        raise ValueError("frozen allocation changed")
    tiny = mode != "main"
    namespace = NAMESPACE + ("-canary" if tiny else "")
    alloc = (
        hashlib.sha256((control["allocation"] + "-canary").encode()).hexdigest()
        if tiny
        else control["allocation"]
    )
    ledger = read_ledger(
        input_dir / ("canary-ledger.json" if tiny else "verified-ledger.json"), namespace, alloc
    )
    units = allocation()
    if tiny:
        units = [
            u
            for u in units
            if u["table"] == "t5" and u["dose"] == 100 and u["seed"] == 20261022 and u["fold"] == 0
        ]
    if not set(ledger) <= {u["index"] for u in units}:
        raise ValueError("ledger outside frozen workload")
    output.mkdir(parents=True, exist_ok=True)
    export_json(output / "allocation.json", {"namespace": namespace, "allocation": alloc})
    export_json(output / "hardware.json", hw)
    if tiny:
        if control.get("bundle_name") == "labels.tar.xz":
            from research.a100_a2.transfer import verify_compact_archive

            verify_compact_archive(input_dir / "labels.tar.xz", control["bindings"]["labels"])
        tables = {"t5": synthetic_table()}
        metadata = {"synthetic": True, "anchors": 10}
    else:
        from research.a100_a2.data import materialize_bundle

        bundle_name = control.get("bundle_name", "labels.tar.gz")
        if bundle_name == "labels.tar.xz":
            paths = materialize_bundle(
                input_dir / bundle_name,
                output.parent / "tables",
                input_dir / "b8.pth",
                source_archive_sha256=control["bindings"]["labels"],
            )
        elif bundle_name == "labels.tar.gz":
            paths = materialize_bundle(
                input_dir / bundle_name, output.parent / "tables", input_dir / "b8.pth"
            )
        else:
            raise ValueError("unsupported A2 bundle transport")
        metadata = json.loads((output.parent / "tables/materialization.json").read_text())
        tables = {}
        for name in ("t5", "a5"):
            with np.load(paths[name], allow_pickle=False) as archive:
                tables[name] = {key: archive[key] for key in archive.files}
    lineage = export_json(output / "materialization.json", metadata)
    completed = []
    for unit in units:
        index = unit["index"]
        if index in ledger:
            continue
        spec = {k: unit[k] for k in ("encoder", "target", "dose", "seed", "fold")}
        values = fit_unit(
            tables[unit["table"]],
            str(input_dir / "b8.pth"),
            device=device,
            epochs=1 if tiny else 20,
            batch_size=2 if tiny else 256,
            **spec,
        )
        values.update(
            {k: np.asarray(unit[k]) for k in ("table", "encoder", "target", "dose", "seed")}
        )
        completed.append(publish_fit(output, index, values, namespace, alloc))
        # Operational counts only, never loss, predictions or metrics.
        progress = {
            "done": len(completed) + len(ledger),
            "total": len(units),
            "elapsed_seconds": time.monotonic() - started,
            "index": index,
        }
        (output / "progress.json").write_text(json.dumps(progress) + "\n")
        print(json.dumps(progress), flush=True)
    receipt = {
        "schema": "a2-attempt/v1",
        "namespace": namespace,
        "allocation": alloc,
        "synthetic": tiny,
        "mode": mode,
        "completed": completed,
        "skipped": sorted(ledger),
        "expected_units": len(units),
        "materialization_sha256": lineage["sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "outcomes": "sealed; only registered Mac aggregator may score",
    }
    export_json(output / "receipt.json", receipt)
    return receipt


def run(input_dir: Path, output: Path, device: str, mode: str) -> dict[str, Any]:
    """Publish an outcome-blind heartbeat even while replaying or fitting a large unit."""
    output.mkdir(parents=True, exist_ok=True)
    done = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not done.wait(60):
            print(
                json.dumps({"heartbeat": True, "elapsed_seconds": time.monotonic() - started}),
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, name="a2-operational-heartbeat", daemon=True)
    thread.start()
    try:
        return _run(input_dir, output, device, mode)
    finally:
        done.set()
        thread.join(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("smoke", "canary", "main"), required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output, args.device, args.mode)


if __name__ == "__main__":
    main()
