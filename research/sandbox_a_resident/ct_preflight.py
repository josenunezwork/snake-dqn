"""Offline C/T (t34-cf-1028) schema, source, seed-audit, topology, and executable-test admission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from research.sandbox_a_resident.ct_prepare import STUDY, ct_endpoint_steps
from research.sandbox_a_resident.ct_task import (
    CTControl,
    ct_seed_allocations,
    train01_protected_eval_seeds,
)
from research.sandbox_a_resident.ct_topology import TOPOLOGY_SPEC
from research.sandbox_a_resident.ct_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_FIELD,
    load_ledger,
    validate_compact_resume,
)
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

ROOT = Path(__file__).resolve().parents[2]
CT_CUDA_SEMANTICS_TEST = "research/sandbox_a_resident/test_ct_cuda_semantics.py"
CT_FOCUSED_TESTS: tuple[str, ...] = (
    "research/sandbox_a_resident/test_ct_mechanics_and_snapshots.py",
    "research/sandbox_a_resident/test_ct_branching_fit_and_stats.py",
    "research/sandbox_a_resident/test_ct_runner_and_resume.py",
    CT_CUDA_SEMANTICS_TEST,
    "tools/sandbox/tests/test_source_delivery.py",
)


def validate_uploaded_ledger(packet: Path, broker: dict[str, Any], control: dict[str, Any]) -> None:
    """Verify relative runtime ledger paths (raw .pt or compact .pt.xz) against Mac uploads."""
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    compact_by_unit = validate_compact_resume(ledger)
    uploads = {row["dest"]: row for row in broker["inputs"]}
    if len(uploads) != len(broker["inputs"]):
        raise ValueError("duplicate runtime input destination")
    referenced: set[str] = set()
    for unit in ledger.get("units", []):
        stage = unit.get("stage")
        if type(stage) is not int or stage < 1:
            raise ValueError("stage unit requires positive integer stage")
        expected = [f"resume/stage-{stage:04d}{suffix}" for suffix in (".pt", ".json")]
        if [row["path"] for row in unit["exports"]] != expected:
            raise ValueError("complete canonical stage snapshot pair (.pt + .json) required")
        u_id = unit.get("id")
        compact_entry = compact_by_unit.get(u_id)
        for row in unit["exports"]:
            destination = row["path"]
            if compact_entry is not None and destination == compact_entry["original"]["path"]:
                if destination in uploads:
                    raise ValueError("compact resume must not also upload the raw stage .pt")
                sources: list[Path] = []
                for role, suffix in (
                    ("derivative", COMPACT_DERIVATIVE_SUFFIX),
                    ("manifest", COMPACT_MANIFEST_SUFFIX),
                ):
                    item = uploads.get(destination + suffix)
                    bound = compact_entry[role]
                    if (
                        item is None
                        or bound["path"] != destination + suffix
                        or item["sha256"] != bound["sha256"]
                        or not item["path"].endswith(suffix)
                        or digest(Path(item["path"])) != bound["sha256"]
                        or Path(item["path"]).stat().st_size != bound["size"]
                    ):
                        raise ValueError(f"compact resume {role} is not a hash-bound upload")
                    sources.append(Path(item["path"][: -len(suffix)]))
                    referenced.add(destination + suffix)
                if sources[0] != sources[1]:
                    raise ValueError("compact derivative and manifest must be original siblings")
                source = sources[0]
                for suffix in (".sha256", ".complete"):
                    local_sidecar = Path(str(source) + suffix)
                    if local_sidecar.is_symlink() or not local_sidecar.is_file():
                        raise ValueError("compact resume original lacks Mac completion sidecars")
                    if (
                        suffix == ".sha256"
                        and local_sidecar.read_text().strip().split()[0] != row["sha256"]
                    ):
                        raise ValueError("compact resume original .sha256 sidecar mismatch")
                    item = uploads.get(destination + suffix)
                    if item is not None:
                        if Path(item["path"]) != local_sidecar or digest(local_sidecar) != item.get(
                            "sha256"
                        ):
                            raise ValueError("resume sidecars must be exact uploaded siblings")
                        referenced.add(destination + suffix)
                row["path"] = str(source)
                continue

            upload = uploads.get(destination)
            if upload is None or row["sha256"] != upload["sha256"]:
                raise ValueError("resume payload is not a hash-bound uploaded input")
            source = Path(upload["path"])
            if (
                source.is_symlink()
                or not source.is_file()
                or digest(source) != row["sha256"]
                or source.stat().st_size != row["size"]
            ):
                raise ValueError("resume payload source file missing or hash/size mismatch")
            referenced.add(destination)
            for suffix in (".sha256", ".complete"):
                local_sidecar = Path(str(source) + suffix)
                if local_sidecar.is_symlink() or not local_sidecar.is_file():
                    raise ValueError("resume payload lacks Mac completion sidecars")
                if (
                    suffix == ".sha256"
                    and local_sidecar.read_text().strip().split()[0] != row["sha256"]
                ):
                    raise ValueError("resume payload .sha256 sidecar mismatch")
                name = destination + suffix
                item = uploads.get(name)
                if item is not None:
                    if Path(item["path"]) != local_sidecar or digest(local_sidecar) != item.get(
                        "sha256"
                    ):
                        raise ValueError("resume sidecars must be exact uploaded siblings")
                    referenced.add(name)
            row["path"] = str(source)

    if referenced != {name for name in uploads if name.startswith("resume/")}:
        raise ValueError("unreferenced resume upload")

    ledger.pop(COMPACT_RESUME_FIELD, None)
    with tempfile.TemporaryDirectory(prefix="ct1028-verify-ledger-") as directory:
        path = Path(directory) / "ledger.json"
        path.write_text(json.dumps(ledger))
        load_ledger(path, control)


def validate_packet(
    packet: Path, *, require_committed: bool = True, allow_tiny: bool = False
) -> dict[str, Any]:
    """Bind C/T runtime controls to committed seed audit and frozen topology contract."""
    payload = preflight(
        packet / "sandbox.json", require_committed=require_committed, run_tests=False
    )
    broker = json.loads((packet / "dispatch-manifest.json").read_text())
    control = json.loads((packet / "control.json").read_text())
    allocation_index = control.get("allocation_index", 0)
    seed_offset = control.get("seed_offset", 0)
    if (
        type(allocation_index) is not int
        or allocation_index < 0
        or type(seed_offset) is not int
        or seed_offset < 0
    ):
        raise ValueError("C/T seed domain requires nonnegative integer allocation/offset")

    if allow_tiny and bool(control.get("tiny", False)):
        default_ctrl = CTControl.from_dict(control)
        if default_ctrl.study != STUDY or default_ctrl.source_commit != broker["commit"]:
            raise ValueError("tiny C/T control study or source_commit differs")
    else:
        default_ctrl = CTControl(
            study=STUDY,
            source_commit=broker["commit"],
            allocation_index=allocation_index,
            seed_offset=seed_offset,
            tiny=False,
        )
    default = default_ctrl.to_dict()
    if control != default or control != json.loads(broker["env"]["SANDBOX_CT_CONTROL_JSON"]):
        raise ValueError("C/T control differs from frozen production recipe")
    if payload["study"] != STUDY or broker["script"] != "research/sandbox_a_resident/ct_job.sh":
        raise ValueError("wrong C/T study or entrypoint")
    if broker["env"]["SANDBOX_A_SOURCE_COMMIT"] != broker["commit"]:
        raise ValueError("runtime source pin differs")
    if payload["endpoint"] != {"kind": "steps", "value": ct_endpoint_steps(default_ctrl)}:
        raise ValueError("wrong finite C/T simulator endpoint")

    for name in (
        "control.json",
        "registry-check.json",
        "topology-manifest.json",
        "verified-ledger.json",
    ):
        verify_committed_manifest(packet / name, require_committed=require_committed)

    topology_manifest = json.loads((packet / "topology-manifest.json").read_text())
    if topology_manifest != dict(TOPOLOGY_SPEC):
        raise ValueError("topology-manifest.json differs from frozen TOPOLOGY_SPEC")

    check = json.loads((packet / "registry-check.json").read_text())
    expected = ct_seed_allocations(default_ctrl)
    registry = next(r for r in payload["inputs"] if r["kind"] == "registry")
    if (
        check.get("schema") != "t34-cf-seed-audit/v1"
        or check.get("status") != "PASS"
        or check.get("study") != STUDY
        or check.get("allocations") != expected
        or check.get("count") != sum(row["count"] for row in expected)
        or check.get("registry_sha256") != registry["sha256"]
        or check.get("effective_seed_bits") != 63
        or check.get("aliases") != 0
        or check.get("excluded_intersection") != 0
        or check.get("train01_protected_count") != len(train01_protected_eval_seeds())
        or check.get("train01_excluded_intersection") != 0
        or broker["sandbox_a"].get("seed_allocations") != expected
        or broker["sandbox_a"].get("seed_audit_sha256") != digest(packet / "registry-check.json")
        or broker["sandbox_a"].get("topology_manifest_sha256")
        != digest(packet / "topology-manifest.json")
    ):
        raise ValueError("exhaustive C/T seed audit does not bind this allocation/registry")
    validate_uploaded_ledger(packet, broker, control)
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, nargs="?")
    parser.add_argument(
        "--cuda-semantics",
        action="store_true",
        help="run the logical-CUDA / meta-device semantics suite (also included in full preflight)",
    )
    parser.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="engineering test helper only; real dispatch never passes this",
    )
    args = parser.parse_args()
    if not args.cuda_semantics and args.packet is None:
        parser.error("packet directory is required unless --cuda-semantics is specified")
    check: dict[str, Any] | None = None
    if args.packet is not None:
        check = validate_packet(args.packet.resolve(), require_committed=not args.allow_uncommitted)
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        env[key] = "1"
    env["SNAKE_DQN_DEVICE"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CT_PREFLIGHT_SUBPROCESS"] = "1"
    python_bin = (
        ROOT / "venv/bin/python" if (ROOT / "venv/bin/python").is_file() else Path(sys.executable)
    )
    targets = (
        (CT_CUDA_SEMANTICS_TEST,)
        if (args.cuda_semantics and args.packet is None)
        else CT_FOCUSED_TESTS
    )
    subprocess.run(
        [
            "nice",
            "-n",
            "15",
            str(python_bin),
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            *targets,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    if check is None:
        print("PASS_CT1028_CUDA_SEMANTICS tests=PASS allocations=0")
    else:
        print(
            f"PASS_CT1028_PREFLIGHT seeds={check['count']} "
            "schema=PASS registry=PASS tests=PASS allocations=0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
