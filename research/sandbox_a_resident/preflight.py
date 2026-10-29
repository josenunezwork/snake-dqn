"""Automated Job A admission only: shared schema, all reset seeds, focused tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.sandbox_a_resident.prepare import registry_check

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    packet = args.packet.resolve()
    from tools.sandbox.preflight import preflight

    # The dispatch adapter runs the shared sandbox suite before this entrypoint.
    # Repeat every static binding/schema/registry check here, then run Job A's
    # complete suite below. Re-running the unrelated Job B/shared test suite here
    # doubled admission work and exceeded dispatch's 60-second validation bound.
    payload = preflight(packet / "sandbox.json", run_tests=False)
    control = json.loads((packet / "control.json").read_text())
    broker = json.loads((packet / "dispatch-manifest.json").read_text())
    if json.loads(broker["env"]["SANDBOX_A_CONTROL_JSON"]) != control:
        raise ValueError("runtime controls differ from committed packet")
    if control["source_commit"] != broker["commit"]:
        raise ValueError("runtime pin differs from frozen allocation")
    checkpoint = next(
        Path(r["path"]) for r in payload["inputs"] if r["kind"] == "frozen-checkpoint"
    )
    registry = next(Path(r["path"]) for r in payload["inputs"] if r["kind"] == "registry")
    checked = registry_check(control, registry, checkpoint)
    if args.checks_only:
        print("PASS_SANDBOX_A_STRUCTURAL_CHECKS tests=REUSED_BY_HOST allocations=0")
        return 0
    environment = dict(os.environ)
    environment.pop("SNAKE_SANDBOX_TEST_RECEIPT", None)
    environment.pop("SNAKE_SANDBOX_ALLOCATION_LEASE", None)
    environment.pop("SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT", None)
    environment.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMBA_NUM_THREADS",
                "OMP_THREAD_LIMIT",
            )
        }
    )
    environment["SNAKE_DQN_DEVICE"] = "cpu"
    from tools.ops.processes import run
    from tools.sandbox.admission import TEST_SECONDS

    run(
        [
            "nice",
            "-n",
            "15",
            str(ROOT / "venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "research/sandbox_a_resident",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=TEST_SECONDS,
    )
    print(
        f"PASS_SANDBOX_A_PREFLIGHT seeds={checked['allocated_uint64_seeds']} "
        "tests=PASS schema=PASS registry=PASS allocations=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
