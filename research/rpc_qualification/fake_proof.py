"""Regenerate create-only fake-CLI evidence from the current source, offline."""

from __future__ import annotations

import argparse
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
        subprocess.run(
            [*command, f"--junitxml={report}"],
            cwd=REPO,
            check=True,
            env={**os.environ, **{key: "1" for key in THREADS}, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        suites = ET.parse(report).getroot().findall("testsuite")
        counts = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
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
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    generate(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
