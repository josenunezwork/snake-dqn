"""Run the frozen Linux contracts; absent, skipped, or xfailed work is incomplete."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_name("contract_inventory.json")


def packet() -> dict[str, Any]:
    value = json.loads(INVENTORY.read_text())
    shards = value["shards"]
    flattened = [node for shard in shards for node in shard["nodes"]]
    if (
        not flattened
        or len(set(flattened)) != len(flattened)
        or flattened != value["linux_required"]
        or [s["index"] for s in shards] != list(range(len(shards)))
    ):
        raise ValueError("INCOMPLETE: malformed frozen contract inventory")
    return value


def nodes() -> list[str]:
    return packet()["linux_required"]


def validate_collection(expected: list[str], actual: list[str], returncode: int) -> None:
    if (
        returncode
        or len(actual) != len(expected)
        or set(actual) != set(expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError(f"INCOMPLETE: required collection differs: {actual!r}")


def validate_reports(report: Path, expected_count: int, returncode: int) -> None:
    tree = ET.parse(report)
    cases = tree.findall(".//testcase")
    if (
        returncode
        or len(cases) != expected_count
        or any(
            case.find(kind) is not None
            for case in cases
            for kind in ("skipped", "failure", "error")
        )
    ):
        raise ValueError("INCOMPLETE: required check failed, skipped, xfailed, or disappeared")


def run_checks(expected: list[str]) -> dict[str, Any]:
    """Collect and run one finite shard, one test process at a time."""
    environment = os.environ.copy()
    environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="")
    prefix = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "--noconftest",
        "-p",
        "no:xdist",
    ]
    collected = subprocess.run(
        [*prefix, "--collect-only", *expected],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    actual = [
        line.strip()
        for line in collected.stdout.splitlines()
        if line.strip().startswith(("tests/", "research/")) and "::" in line
    ]
    validate_collection(expected, actual, collected.returncode)
    with tempfile.TemporaryDirectory(prefix="r33-report-") as directory:
        report = Path(directory) / "report.xml"
        result = subprocess.run(
            [*prefix, "-o", "xfail_strict=true", f"--junitxml={report}", *expected],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=480,
        )
        if not report.is_file():
            raise ValueError("INCOMPLETE: missing test report\n" + result.stdout + result.stderr)
        try:
            validate_reports(report, len(expected), result.returncode)
        except ValueError as exc:
            raise ValueError(str(exc) + "\n" + result.stdout + result.stderr) from exc
    return {"node_ids": expected, "checks_passed": len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    selected = packet()["shards"]
    if args.shard is not None:
        selected = [row for row in selected if row["index"] == args.shard]
        if not selected:
            raise SystemExit("INCOMPLETE: foreign shard")
    for shard in selected:
        print(json.dumps(run_checks(shard["nodes"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
