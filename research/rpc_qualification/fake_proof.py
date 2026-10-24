"""Regenerate create-only fake-CLI evidence from the current source, offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from research.bl_labels.exports import publish_json
from research.rpc_qualification.checker import FAKE_PROOF_TESTS, fake_proof_source_inventory

REPO = Path(__file__).resolve().parents[2]
TESTS = FAKE_PROOF_TESTS
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def generate(out: Path) -> None:
    """Run real fake-CLI tests; bind evidence to source before and after the run."""
    if out.exists() or any(Path(str(out) + s).exists() for s in (".sha256", ".complete")):
        raise ValueError("fake proof is create-only")

    def inventory() -> dict[str, str]:
        return fake_proof_source_inventory(REPO)

    before = inventory()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        timeout=30,
    ).strip()
    command = [
        "nice",
        "-n",
        "15",
        str(REPO / "venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:xdist",
        "-p",
        "no:cacheprovider",
        *TESTS,
    ]
    with tempfile.TemporaryDirectory(prefix="l4-fake-proof-") as temporary:
        report = Path(temporary) / "junit.xml"
        # In-process pytest leaves exactly one slot for differential workers.
        # Set the macOS guard before collection to avoid a third interpreter.
        for key in THREADS:
            os.environ[key] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.origin:
            os.environ["DYLD_LIBRARY_PATH"] = str(Path(spec.origin).parent / "lib")
        trace = Path(temporary) / "differential-trace.json"
        os.environ["SNAKE_COLAB_DIFFERENTIAL_TRACE"] = str(trace)
        cpu_trace = Path(temporary) / "cpu-acceptance-trace.json"
        os.environ["SNAKE_CPU_ACCEPTANCE_TRACE"] = str(cpu_trace)
        import pytest

        result = pytest.main([*command[6:], f"--junitxml={report}"])
        if result:
            raise ValueError(f"fake-CLI tests failed: {result}")
        suites = ET.parse(report).getroot().findall("testsuite")
        counts = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        trace_bytes = trace.read_bytes()
        cpu_trace_bytes = cpu_trace.read_bytes()
    if before != inventory():
        raise ValueError("source changed while fake-CLI tests ran")
    if not counts["tests"] or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise ValueError("fake-CLI proof requires every test to pass")
    publish_json(
        out,
        {
            "schema": "rpc-qualification-fake-cli-proof/v1",
            "status": "PASS",
            "live": False,
            "source_commit": source_commit,
            "command_argv": command,
            "passed": counts["tests"],
            "tests": list(TESTS),
            "source_sha256": before,
            "transport_tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD:research/colab_cli"],
                cwd=REPO,
                text=True,
                timeout=30,
            ).strip(),
            "execution": "pytest.main in-process; one differential child at a time",
            "differential_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "differential_trace": json.loads(trace_bytes),
            "cpu_acceptance_trace_sha256": hashlib.sha256(cpu_trace_bytes).hexdigest(),
            "cpu_acceptance_trace": json.loads(cpu_trace_bytes),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    generate(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
