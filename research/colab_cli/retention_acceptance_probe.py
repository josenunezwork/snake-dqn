"""Fixed native O1 workload; emits observations, never a qualification verdict."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def publish_payload(path: Path, payload: bytes) -> dict[str, Any]:
    """Atomically publish immutable bytes, SHA sidecar, then completion marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".o1-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    with Path(str(path) + ".sha256").open("x") as stream:
        stream.write(digest + "\n")
    with Path(str(path) + ".complete").open("x") as stream:
        stream.write("complete\n")
    return {"path": str(path), "size": len(payload), "sha256": digest}


def run(output: Path, *, cpu_smoke: bool = False) -> int:
    """Compile/load one exact source key, restore fixed input, run three units."""
    root = Path(os.environ["COLAB_CLI_ROOT"])
    inputs = Path(os.environ["COLAB_INPUT_DIR"])
    frozen = json.loads((inputs / "probe-input.json").read_text())
    ledger = json.loads((inputs / "verified-ledger.json").read_text())
    if (ledger.get("namespace"), ledger.get("allocation_index")) != (
        frozen["namespace"],
        frozen["allocation_index"],
    ) or ledger.get("verified_units") != []:
        raise ValueError("O1 first native comparison requires the frozen empty verified ledger")
    fault = os.environ.get("COLAB_RETENTION_FAULT", "none")
    if fault == "child_failure":
        return 23
    import torch

    began = time.monotonic()
    source = Path(__file__).with_name("retention_probe.cu")
    if hashlib.sha256(source.read_bytes()).hexdigest() != frozen["extension_source_sha256"]:
        raise ValueError("native extension differs from frozen input")
    if cpu_smoke:
        device = "cpu"
        extension = None
    else:
        from torch.utils.cpp_extension import load

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
            raise ValueError("O1 native acceptance requires actual A100 sm_80")
        extension = load(
            name="o1_retention_probe",
            sources=[str(source)],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
        device = "cuda"
    compile_seconds = time.monotonic() - began
    durations = []
    for index in range(3):
        start = time.monotonic()
        # Every unit restores fixed scientific state, independent of prior jobs.
        torch.manual_seed(frozen["seed"])
        values = torch.arange(6, dtype=torch.float32, device=device)
        if extension is not None:
            values = extension.identity(values)
        action = int(values.argmax().item())
        if device == "cuda":
            torch.cuda.synchronize()
        if action != 5:
            raise ValueError("native CUDA observation differs from exact CPU expression")
        durations.append(time.monotonic() - start)
        if index == 0:
            publish_payload(
                output / "units/unit-000.json",
                json.dumps(
                    {
                        "namespace": frozen["namespace"],
                        "allocation_index": frozen["allocation_index"],
                        "unit_id": 0,
                        "action": action,
                    },
                    sort_keys=True,
                ).encode(),
            )
    cache_path = root / "cache-evidence.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    result = {
        "schema": "o1-native-probe/v1",
        "device": device,
        "root": str(root),
        "compile_load_seconds": compile_seconds,
        "unit_seconds": durations,
        "bootstrap": cache,
        "input_sha256": hashlib.sha256((inputs / "probe-input.json").read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256((inputs / "verified-ledger.json").read_bytes()).hexdigest(),
        "extension_cache": (
            [
                json.loads(line)
                for line in (root / "extension-cache-events.jsonl").read_text().splitlines()
            ]
            if (root / "extension-cache-events.jsonl").is_file()
            else []
        ),
    }
    publish_payload(output / "probe.json", json.dumps(result, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(Path(os.environ["COLAB_OUTPUT_DIR"]), cpu_smoke=args.cpu_smoke))
