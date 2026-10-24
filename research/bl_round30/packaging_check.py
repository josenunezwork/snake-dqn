"""Bounded stdlib proof for the synthetic R30 upload repair, never a main gate.

The prior numerical preflight remains immutable and explicitly inherited. This
check permits only enumerated packaging changes over that exact qualified pin;
it neither imports numerical libraries nor enters the factory's execution lane.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_round30.local_check import source_identity
from research.metadata_archive import unpack

ROOT = Path(__file__).resolve().parents[2]
BASE_PIN = "c1bf44c89e116bc29f27293895bbb9da326c00ff"
PARENT_SHA = "1eeaf536bb67176201bcef70cb1316516a18784a05a9cbb643391220ed59b193"
PARENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/"
    "sol-verification/lifecycle-c1bf44c8-20261024-01/preflight.json"
)
MECHANICAL_PATHS = {
    "research/bl_round30/pack.py",
    "research/bl_round30/colab.sh",
    "research/bl_round30/packaging_check.py",
    "tests/test_bl_round30_job.py",
}
EXTRA_PATHS = ("research/metadata_archive.py", "research/t34_exit_b8/PUBLIC_INPUTS.json")


def identity() -> dict[str, str]:
    """Bind the complete original closure and the archive/catalog dependencies."""
    return {**source_identity(), **{name: digest(ROOT / name) for name in EXTRA_PATHS}}


def inherited_evidence(commit: str) -> dict:
    """Refuse any scientific, transport, test-body or nonenumerated source drift."""
    import hashlib

    if verify_export(PARENT) != PARENT_SHA:
        raise ValueError("inherited preflight digest differs")
    parent = json.loads(PARENT.read_text())
    if (
        parent.get("schema") != "r30-mac-preflight/v1"
        or parent.get("status") != "PASS"
        or parent.get("returncode") != 0
        or parent.get("source_commit") != BASE_PIN
    ):
        raise ValueError("inherited numerical preflight is not the fixed PASS")
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", BASE_PIN, commit], cwd=ROOT, text=True, timeout=10
        ).splitlines()
    )
    if not changed <= MECHANICAL_PATHS:
        raise ValueError("nonmechanical change cannot inherit numerical preflight")
    current = identity()
    for name, checksum in parent["source_identity"].items():
        frozen = subprocess.check_output(
            ["git", "show", f"{BASE_PIN}:{name}"], cwd=ROOT, timeout=10
        )
        if hashlib.sha256(frozen).hexdigest() != checksum:
            raise ValueError("inherited preflight source binding differs from base")
        if name not in MECHANICAL_PATHS and current.get(name) != checksum:
            raise ValueError(f"inherited numerical source changed: {name}")
    for name in EXTRA_PATHS:
        frozen = subprocess.check_output(
            ["git", "show", f"{BASE_PIN}:{name}"], cwd=ROOT, timeout=10
        )
        if hashlib.sha256(frozen).hexdigest() != current[name]:
            raise ValueError("shared archive/catalog differs from qualified base")
    return {
        "path": str(PARENT),
        "sha256": PARENT_SHA,
        "source_commit": BASE_PIN,
        "numerical_tests_rerun": False,
        "changed_paths": sorted(changed),
    }


def check_receipt(value: dict, commit: str) -> None:
    """Accept this distinct proof only for a synthetic rehearsal at exact bytes."""
    if (
        value.get("schema") != "r30-packaging-preflight/v1"
        or value.get("status") != "PASS"
        or value.get("synthetic_only") is not True
        or value.get("source_commit") != commit
        or value.get("source_identity") != identity()
        or value.get("inherited_numerical_evidence") != inherited_evidence(commit)
    ):
        raise ValueError("synthetic packaging preflight does not bind current source")


def run(output: Path, b8: Path) -> dict:
    """Replay real archive CLI and verify exact URL/metadata bytes without models."""
    from research.bl_round30.pack import make_pack, transport_inputs

    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        raise ValueError("packaging check requires nice >=15")
    started = time.monotonic()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, timeout=10):
        raise ValueError("packaging proof requires a clean frozen checkout")
    inherited = inherited_evidence(commit)
    initial = identity()
    checks = []
    with tempfile.TemporaryDirectory(prefix="r30-packaging-") as temporary:
        root = Path(temporary)
        package = root / "package"
        manifest = make_pack(package, commit, b8)
        rows = manifest["inputs"]
        public = next(row for row in rows if row["dest"] == "b8.pth")
        asset = json.loads((ROOT / EXTRA_PATHS[1]).read_text())["assets"][0]
        assert public == {
            "dest": "b8.pth",
            "url": asset["url"],
            "size_bytes": b8.stat().st_size,
            "sha256": digest(b8),
        }
        assert len(rows) == 2 and "path" not in public
        checks.append("registered B8 URL, exact size/SHA, no local checkpoint upload")
        metadata = next(row for row in rows if row["dest"] == "metadata.tar")
        runtime = root / "runtime-inputs"
        runtime.mkdir()
        shutil.copyfile(b8, runtime / "b8.pth")
        archive = runtime / "metadata.tar"
        shutil.copyfile(metadata["path"], archive)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "research.metadata_archive",
                "unpack",
                "--archive",
                str(archive),
                "--root",
                str(runtime),
                "--sha256",
                metadata["sha256"],
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert json.loads(result.stdout)["files"] == 6
        for original in (package / "inputs").iterdir():
            assert digest(runtime / original.name) == digest(original)
        verify_export(runtime / "control.json")
        verify_export(runtime / "verified-ledger.json")
        checks.append("real archive CLI: six metadata files/sidecars exactly preserved")
        try:
            unpack(archive, runtime, metadata["sha256"])
        except FileExistsError:
            pass
        else:
            raise AssertionError("repeated unpack did not refuse")
        bad_root = root / "bad-digest"
        bad_root.mkdir()
        try:
            unpack(archive, bad_root, "0" * 64)
        except ValueError:
            assert not list(bad_root.iterdir())
        else:
            raise AssertionError("wrong outer digest did not refuse")
        unknown = root / "unknown"
        unknown.mkdir()
        (unknown / "private.npz").write_bytes(b"x" * 1_000_001)
        try:
            transport_inputs(unknown, root / "unknown.tar")
        except ValueError as exc:
            assert "registered public URL" in str(exc)
            assert not (root / "unknown.tar").exists()
        else:
            raise AssertionError("unknown large private input did not refuse")
        checks.append("repeat extraction, corrupt archive SHA and private >1 MB refuse")
        assert manifest["scientific_admission"] is False
        template = json.loads((package / "main-template.json").read_text())
        assert template["scientific_admission"] is False and template["resume"]["skip_verified"]
        checks.append("synthetic-only exemption; scientific template HOLD/resume preserved")
    if initial != identity() or {"torch", "numpy", "numba"} & sys.modules.keys():
        raise ValueError("source drift or numerical import during packaging proof")
    elapsed = time.monotonic() - started
    if elapsed > 120:
        raise ValueError("packaging proof exceeded 120-second bound")
    value = {
        "schema": "r30-packaging-preflight/v1",
        "status": "PASS",
        "synthetic_only": True,
        "source_commit": commit,
        "source_identity": initial,
        "inherited_numerical_evidence": inherited,
        "checks": checks,
        "elapsed_seconds": elapsed,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "numerical_tests_rerun": False,
        "numerical_imports": [],
        "allocations": 0,
    }
    check_receipt(value, commit)
    for invalid in (
        {**value, "synthetic_only": False},
        {**value, "source_commit": "0" * 40},
        {**value, "source_identity": {}},
    ):
        try:
            check_receipt(invalid, commit)
        except ValueError:
            pass
        else:
            raise AssertionError("foreign mechanical preflight did not refuse")
    with tempfile.TemporaryDirectory(prefix="r30-main-refusal-") as temporary:
        root = Path(temporary)
        proof = root / "preflight.json"
        export_json(proof, value)
        try:
            make_pack(root / "main", commit, b8, inputs=root, preflight_receipt=proof)
        except ValueError as exc:
            assert "current-source Mac preflight" in str(exc)
        else:
            raise AssertionError("mechanical proof admitted scientific main")
    checks.append("foreign proof/current-source drift refuse; proof cannot admit main")
    value["elapsed_seconds"] = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "preflight.json", value)
    return value


def main() -> None:
    """Produce a separate immutable mechanical proof, never a numerical PASS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    args = parser.parse_args()

    def expired(_signal: int, _frame: object) -> None:
        raise TimeoutError("packaging proof exceeded 120-second complete deadline")

    prior = signal.signal(signal.SIGALRM, expired)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        signal.signal(signal.SIGALRM, prior)
        raise ValueError("packaging proof cannot replace an active interval timer")
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        print(json.dumps(run(args.output.resolve(), args.b8.resolve()), sort_keys=True))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    main()
