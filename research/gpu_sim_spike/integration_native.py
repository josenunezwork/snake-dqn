"""I0 native compile/parity admission, deliberately separate from timing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import patch

from research.gpu_sim_spike import integration_units as units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes


class Reports:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call" or report.failed or report.skipped:
            self.rows.append(
                {
                    "nodeid": report.nodeid,
                    "phase": report.when,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr) if report.failed else None,
                }
            )


def run(output: Path, *, device: str = "cuda") -> dict[str, Any]:
    """Run independent frozen MOVE and CPU geometry/carry checks before I0 timing."""
    import pytest
    import torch

    reports = Reports()
    native_calls = {"indexed_body_view": 0, "indexed_features58": 0}
    binaries = {}
    geometry_code = pytest.main(
        [
            "-q",
            "research/gpu_sim_spike/test_integration_native.py",
            "-k",
            "not cuda",
        ],
        plugins=[reports],
    )
    if device == "cpu":
        status = "PREFLIGHT" if geometry_code == 0 else "INVALID"
        reason, code, move = (
            "CPU geometry/carry smoke only",
            int(geometry_code),
            {"status": "NOT_RUN"},
        )
    elif not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        status, reason, code, move = (
            "INCOMPLETE",
            "A100 native parity required",
            2,
            {"status": "NOT_RUN"},
        )
    else:
        from research.gpu_sim_spike import move2_events, vector_geometry

        move = move2_events.run("cuda")
        extension = vector_geometry._load_exact_extension()

        class CountedExtension:
            def __getattr__(self, name: str) -> Any:
                target = getattr(extension, name)
                if name not in native_calls:
                    return target

                def called(*args: Any, **kwargs: Any) -> Any:
                    native_calls[name] += 1
                    return target(*args, **kwargs)

                return called

        with patch.object(vector_geometry, "_EXTENSION", CountedExtension()):
            geometry_cuda = pytest.main(
                [
                    "-q",
                    (
                        "research/gpu_sim_spike/test_vector_geometry.py::"
                        "test_cuda_indexed_geometry_bytes_match_canonical_cpu"
                    ),
                    "research/gpu_sim_spike/test_integration_native.py",
                    "-k",
                    "cuda",
                ],
                plugins=[reports],
            )
        import hashlib

        from research.gpu_sim_spike import event_cuda, flood_cuda, move_cuda

        for name, binary in (
            ("geometry", extension),
            ("move", move_cuda._extension()),
            ("events", event_cuda._EXTENSION),
            ("flood", flood_cuda._load()),
        ):
            path = Path(binary.__file__)
            binaries[name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        status = (
            "PASS"
            if geometry_code == 0
            and geometry_cuda == 0
            and move.get("status") == "PASS"
            and len(reports.rows) == 14
            and all(row["outcome"] == "passed" for row in reports.rows)
            and all(count > 0 for count in native_calls.values())
            else "INVALID"
        )
        reason, code = (
            (None, 0) if status == "PASS" else ("MOVE2 or geometry native proof failed", 1)
        )
    value = {
        "schema": "gpu-sim-spike-i0-native-parity/v1",
        "status": status,
        "reason": reason,
        "commit": source_commit(),
        "source_sha256": source_hashes(),
        "pytest_exit": int(code),
        "rows": reports.rows,
        "native_cuda_executed": status == "PASS",
        "move2_events": move,
        "geometry_native_calls": native_calls,
        "compiled_binaries": binaries,
        "fallback": False if status == "PASS" else None,
        "proof_boundary": {
            "move": "frozen-device independent identity",
            "geometry": "CPU affected geometry/carry component",
            "cpu_closed_loop": "UNPROVED",
        },
    }
    units.publish_json(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.output)
    if value["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
