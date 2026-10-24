"""Run current-pin native gates and export their explicit admission result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import v8_split_units as units
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


def main() -> None:
    import pytest
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("native gate requires A100")
    reports = Reports()
    code = pytest.main(
        [
            "-q",
            "research/gpu_sim_spike/test_vector_geometry.py",
            "research/gpu_sim_spike/test_v8_indexed.py",
            "research/gpu_sim_spike/test_v8_split_parity.py",
            "-k",
            "cuda",
        ],
        plugins=[reports],
    )
    passed = [r for r in reports.rows if r["phase"] == "call" and r["outcome"] == "passed"]
    failures = [r for r in reports.rows if r["outcome"] == "failed"]
    status = "PASS" if code == 0 and len(passed) == 9 and len(reports.rows) == 9 else "INCOMPLETE"
    if any(
        "AssertionError" in (r["detail"] or "") or "ParityMismatch" in (r["detail"] or "")
        for r in failures
    ):
        status = "INVALID"
    units.publish_json(
        args.output,
        {
            "schema": "v8-split-native-parity/v1",
            "status": status,
            "commit": source_commit(),
            "source_sha256": source_hashes(),
            "pytest_exit": int(code),
            "rows": reports.rows,
            "native_cuda_executed": status == "PASS",
        },
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
