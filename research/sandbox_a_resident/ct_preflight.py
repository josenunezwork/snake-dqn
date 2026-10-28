"""Offline C/T (t34-cf-1028) schema, source, seed-audit, topology, and executable-test admission."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from research.sandbox_a_resident.ct_prepare import STUDY, ct_endpoint_steps
from research.sandbox_a_resident.ct_resume import (
    _EVIDENCE_CACHE,
    _check_attempt_base_fault_evidence,
    _check_attempt_output_dir_fault_evidence,
    _check_single_root_fault_evidence,
    _evidence_session,
    _extract_prior_admitted_coverage,
    _is_hex,
    _load_json,
    _no_symlink_path,
    _scan_retained_text,
    retained_inventory,
    verify_mechanical_stop,
)
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
from tools.ops.packet_supersession import require_current_packet
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

ROOT = Path(__file__).resolve().parents[2]
CT_CUDA_SEMANTICS_TEST = "research/sandbox_a_resident/test_ct_cuda_semantics.py"
CT_FOCUSED_TESTS: tuple[str, ...] = (
    "research/sandbox_a_resident/test_ct_mechanics_and_snapshots.py",
    "research/sandbox_a_resident/test_ct_branching_fit_and_stats.py",
    "research/sandbox_a_resident/test_ct_runner_and_resume.py",
    CT_CUDA_SEMANTICS_TEST,
    "research/sandbox_a_resident/test_ct_hardening.py",
    "research/sandbox_a_resident/test_ct_cuda_capture.py",
    "research/sandbox_a_resident/test_ct_packet08.py",
    "tools/sandbox/tests/test_source_delivery.py",
)


def _verify_packet_components(
    packet: Path, broker: dict[str, Any], *, require_committed: bool
) -> None:
    """Every packet, including the root, has a canonical manifest and mandatory seal."""
    _no_symlink_path(packet)
    names = [
        "dispatch-manifest.json",
        "sandbox.json",
        "control.json",
        "registry-check.json",
        "topology-manifest.json",
        "verified-ledger.json",
        "SEAL.json",
    ]
    if broker.get("completion_attempt") is not None:
        names.append("provenance.json")
    for name in names:
        path = packet / name
        _no_symlink_path(path)
        if not path.is_file():
            raise ValueError(f"packet lacks mandatory component: {path}")
        if require_committed:
            verify_committed_manifest(path)
    if _load_json((packet / "dispatch-manifest.json").read_text()) != broker:
        raise ValueError("loaded prior manifest differs from canonical dispatch-manifest.json")
    seal = _load_json((packet / "SEAL.json").read_text())
    if not isinstance(seal, dict):
        raise ValueError("malformed mandatory SEAL.json")
    if seal.get("manifest_sha256") != digest(packet / "dispatch-manifest.json"):
        raise ValueError("SEAL.json does not bind canonical dispatch-manifest.json")
    if broker.get("completion_attempt") is None:
        bindings = {
            "control_sha256": "control.json",
            "sandbox_sha256": "sandbox.json",
            "ledger_sha256": "verified-ledger.json",
            "registry_audit_sha256": "registry-check.json",
            "topology_manifest_sha256": "topology-manifest.json",
        }
        if seal.get("schema") != "t34-cf-1028-prep-seal/v1" or any(
            seal.get(field) != digest(packet / name) for field, name in bindings.items()
        ):
            raise ValueError("initial SEAL.json lacks packet component bindings")


def _bounded_manifest_bytes(path: Path) -> bytes:
    """Read at most one MiB and normalize unavailable lineage evidence to a refusal."""
    _no_symlink_path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("oversized lineage manifest")
        return raw
    except OSError as error:
        raise ValueError(f"unreadable lineage manifest: {path}") from error


def _check_lineage_graph(
    packet: Path,
    broker: dict[str, Any],
    control: dict[str, Any],
    *,
    original_bytes: bytes | None = None,
) -> None:
    """Bound traversal before recursive admission, including malformed cyclic graphs."""
    visited: set[str] = set()
    indices: list[int] = []
    outputs: set[str] = set()
    current, manifest = packet, broker
    raw_manifest = (
        original_bytes
        if original_bytes is not None
        else _bounded_manifest_bytes(packet / "dispatch-manifest.json")
    )
    if _load_json(raw_manifest) != broker:
        raise ValueError("lineage manifest differs from verified bytes")
    output_jobs: dict[str, str] = {}
    while True:
        _no_symlink_path(current)
        canonical = current / "dispatch-manifest.json"
        identity = os.path.normpath(str(canonical)).casefold()
        if identity in visited:
            raise ValueError("continuation lineage cycle detected")
        visited.add(identity)
        require_current_packet(manifest, canonical, root=ROOT, original_bytes=raw_manifest)
        comp = manifest.get("completion_attempt")
        index = 0 if comp is None else comp.get("index") if isinstance(comp, dict) else None
        if type(index) is not int or index < 0:
            raise ValueError("malformed completion_attempt metadata")
        if index > 2 or len(indices) > 2:
            raise ValueError("continuation lineage exceeds two completion attempts")
        indices.append(index)
        if isinstance(manifest.get("output_dir"), str):
            outputs.add(manifest["output_dir"])
            if not isinstance(manifest.get("job_id"), str) or not manifest["job_id"]:
                raise ValueError("lineage lacks job identity")
            output_jobs[manifest["output_dir"]] = manifest["job_id"]
        if comp is None:
            break
        raw = comp.get("prior_manifest_path")
        if not isinstance(raw, str):
            raise ValueError("continuation requires absolute prior_manifest_path")
        path = Path(raw)
        _no_symlink_path(path)
        if path.name != "dispatch-manifest.json":
            raise ValueError("prior_manifest_path must name canonical dispatch-manifest.json")
        if os.path.normpath(str(path)).casefold() in visited:
            raise ValueError("continuation lineage cycle detected")
        raw_manifest = _bounded_manifest_bytes(path)
        if hashlib.sha256(raw_manifest).hexdigest() != comp.get("prior_manifest_sha256"):
            raise ValueError("prior manifest SHA-256 mismatch in lineage")
        manifest = _load_json(raw_manifest.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("malformed prior manifest")
        current = path.parent
    if any(index != prior + 1 for index, prior in zip(indices, indices[1:])):
        raise ValueError("continuation index must equal prior index + 1")
    cache = _EVIDENCE_CACHE.get()
    if cache is not None:
        if not (current / "SEAL.json").is_file():
            raise ValueError(f"packet lacks mandatory component: {current / 'SEAL.json'}")
        seal = _load_json((current / "SEAL.json").read_text())
        cache["source_archive_binding"] = {
            "source_commit": control["source_commit"],
            "source_tree": seal.get("source_tree")
            or (
                subprocess.check_output(
                    ["git", "rev-parse", f"{control['source_commit']}^{{tree}}"],
                    cwd=ROOT,
                    text=True,
                    timeout=10,
                ).strip()
                if seal.get("source_sha256")
                else None
            ),
            "source_archive": seal.get("source_archive", {}),
            "source_files": seal.get("source_sha256", {}),
        }
        cache.setdefault("bound_output_dirs", set()).update(outputs)
        cache.setdefault("bound_output_jobs", {}).update(output_jobs)


@_evidence_session
def validate_uploaded_ledger(
    packet: Path,
    broker: dict[str, Any],
    control: dict[str, Any],
    *,
    require_committed: bool = True,
    manifest_bytes: bytes | None = None,
) -> None:
    """Verify relative runtime ledger paths (raw .pt or compact .pt.xz) against Mac uploads."""
    _check_lineage_graph(packet, broker, control, original_bytes=manifest_bytes)
    _verify_packet_components(packet, broker, require_committed=require_committed)
    if _load_json((packet / "control.json").read_text()) != control:
        raise ValueError("lineage control differs from frozen packet control")
    ledger_file = packet / "verified-ledger.json"
    actual_ledger_sha = digest(ledger_file)
    ledger = _load_json(ledger_file.read_text())
    compact_by_unit = validate_compact_resume(ledger)
    uploads = {row["dest"]: row for row in broker["inputs"]}
    if len(uploads) != len(broker["inputs"]):
        raise ValueError("duplicate runtime input destination")
    ledger_upload = uploads.get("verified-ledger.json")
    if (
        ledger_upload is None
        or ledger_upload.get("sha256") != actual_ledger_sha
        or Path(str(ledger_upload.get("path", ""))).resolve() != ledger_file.resolve()
    ):
        raise ValueError("verified-ledger.json is not hash-bound to packet ledger upload")

    canonical_units = copy.deepcopy(ledger.get("units", []))
    if not isinstance(canonical_units, list):
        raise ValueError("verified-ledger.json units must be a list")

    comp = broker.get("completion_attempt")
    if comp is None:
        if canonical_units:
            raise ValueError("non-empty verified-ledger.json requires completion_attempt metadata")
    else:
        if not isinstance(comp, dict) or comp.get("index") not in (1, 2):
            raise ValueError("malformed completion_attempt metadata")
        if (
            comp.get("manual") is not True
            or not isinstance(comp.get("prior_job_id"), str)
            or not comp["prior_job_id"]
            or not _is_hex(comp.get("prior_manifest_sha256"), 64)
            or comp.get("ledger_sha256") != actual_ledger_sha
        ):
            raise ValueError("completion_attempt lacks valid prior_manifest/ledger SHA-256 binding")
        if not canonical_units:
            raise ValueError("continuation packet requires at least one verified stage unit")
        actual_stages = [u.get("stage") for u in canonical_units if isinstance(u, dict)]
        if comp.get("verified_stages") != actual_stages:
            raise ValueError("completion_attempt.verified_stages differs from verified-ledger.json")
        canonical_by_stage = {
            u["stage"]: u
            for u in canonical_units
            if isinstance(u, dict) and type(u.get("stage")) is int
        }
        if comp["index"] == 1:
            if comp.get("prior_verified_stages") != [] or comp.get("prior_verified_units") != []:
                raise ValueError(
                    "completion_attempt=1 must declare empty prior_verified_stages/units"
                )
        else:
            prior_stages = comp.get("prior_verified_stages")
            prior_units = comp.get("prior_verified_units")
            if (
                not isinstance(prior_stages, list)
                or not prior_stages
                or not isinstance(prior_units, list)
                or not prior_units
            ):
                raise ValueError(
                    "completion_attempt=2 must bind non-empty prior_verified_stages and units"
                )
            if [u.get("stage") for u in prior_units if isinstance(u, dict)] != prior_stages:
                raise ValueError("prior_verified_stages disagrees with prior_verified_units")
            for p_unit in prior_units:
                if not isinstance(p_unit, dict):
                    raise ValueError("malformed prior_verified_units entry")
                p_stage = p_unit.get("stage")
                if type(p_stage) is not int or p_stage not in canonical_by_stage:
                    raise ValueError(
                        f"continuation ledger omits previously verified stage {p_stage!r}"
                    )
                if canonical_by_stage[p_stage] != p_unit:
                    raise ValueError(
                        "continuation ledger conflicts with previously verified "
                        f"stage-{p_stage:04d}"
                    )

        raw_out_dirs = comp.get("prior_output_dirs")
        raw_exp_dirs = comp.get("prior_export_roots")
        if (
            not isinstance(raw_out_dirs, list)
            or any(not isinstance(d, str) or not d for d in raw_out_dirs)
            or not isinstance(raw_exp_dirs, list)
            or not raw_exp_dirs
            or any(not isinstance(d, str) or not d for d in raw_exp_dirs)
        ):
            raise ValueError(
                "completion_attempt must bind prior_output_dirs and prior_export_roots"
            )
        for out_str in raw_out_dirs:
            _check_attempt_output_dir_fault_evidence(Path(out_str))
        for exp_str in raw_exp_dirs:
            _check_single_root_fault_evidence(Path(exp_str))

        raw_pm = comp.get("prior_manifest_path")
        if not isinstance(raw_pm, str) or not Path(raw_pm).is_absolute():
            raise ValueError("continuation requires absolute prior_manifest_path")
        pm_path = Path(raw_pm)
        _no_symlink_path(pm_path)
        if pm_path.name != "dispatch-manifest.json":
            raise ValueError("prior_manifest_path must name canonical dispatch-manifest.json")
        if (
            pm_path.is_symlink()
            or not pm_path.is_file()
            or digest(pm_path) != comp["prior_manifest_sha256"]
        ):
            raise ValueError("prior_manifest_path missing, symlinked, or SHA-256 mismatched")
        prior_broker = _load_json(pm_path.read_text())
        if (
            not isinstance(prior_broker, dict)
            or prior_broker.get("job_id") != comp["prior_job_id"]
            or prior_broker.get("commit") != broker.get("commit")
        ):
            raise ValueError("prior_manifest_path job_id or commit binding mismatch")
        # Recurse through the bound manifests, never through successor self-declarations.
        # _extract_prior_admitted_coverage enforces strictly decreasing indices (2 -> 1 -> initial).
        disk_units, disk_lsha, disk_outs, disk_exps = _extract_prior_admitted_coverage(
            prior_broker,
            control,
            completion_index=int(comp["index"]),
            prior_packet_dir=pm_path.parent,
        )
        validate_uploaded_ledger(
            pm_path.parent, prior_broker, control, require_committed=require_committed
        )
        if comp.get("prior_ledger_sha256") != disk_lsha:
            raise ValueError("prior_ledger_sha256 differs from bound prior verified-ledger.json")
        if comp.get("prior_verified_units") != [disk_units[k] for k in sorted(disk_units)]:
            raise ValueError("prior_verified_units differs from bound prior verified-ledger.json")
        if raw_out_dirs != disk_outs:
            raise ValueError("continuation omits or changes lineage-derived prior_output_dirs")
        required_exps = set(disk_exps)
        for dest, upload in uploads.items():
            if dest.startswith("resume/"):
                required_exps.add(str(Path(upload["path"]).resolve().parent))
        if not required_exps.issubset(set(raw_exp_dirs)):
            raise ValueError("continuation omits lineage-derived prior_export_roots")
        proof_file, seal_file = packet / "provenance.json", packet / "SEAL.json"
        for path in (proof_file, seal_file):
            if path.is_symlink() or not path.is_file():
                raise ValueError("continuation requires provenance.json and SEAL.json")
        proof, seal = _load_json(proof_file.read_text()), _load_json(seal_file.read_text())
        proof_sha = digest(proof_file)
        if (
            proof.get("schema") != "t34-cf-continuation-provenance/v1"
            or comp.get("provenance_sha256") != proof_sha
            or seal.get("schema") != "t34-cf-continuation-seal/v1"
            or seal.get("provenance_sha256") != proof_sha
            or seal.get("manifest_sha256") != digest(packet / "dispatch-manifest.json")
            or seal.get("ledger_sha256") != actual_ledger_sha
            or seal.get("prior_manifest_sha256") != comp["prior_manifest_sha256"]
            or proof.get("prior_manifest_path") != raw_pm
            or proof.get("prior_manifest_sha256") != comp["prior_manifest_sha256"]
        ):
            raise ValueError("continuation provenance lacks mandatory seal/manifest binding")
        roots = sorted(set(raw_out_dirs + raw_exp_dirs))
        if proof.get("retained") != {root: retained_inventory(Path(root)) for root in roots}:
            raise ValueError("retained evidence inventory changed or provenance omitted")
        receipt = proof.get("receipt")
        if not isinstance(receipt, dict):
            raise ValueError("provenance lacks mechanical-stop receipt")
        log = proof.get("supervisor_log")
        if (
            not isinstance(log, dict)
            or not isinstance(log.get("path"), str)
            or type(log.get("size")) is not int
            or log["size"] < 0
            or not _is_hex(log.get("sha256"), 64)
        ):
            raise ValueError("provenance requires bound supervisor_log path/size/SHA-256")
        log_path = Path(log["path"])
        _no_symlink_path(log_path)
        if (
            not log_path.is_file()
            or log_path.stat().st_size != log["size"]
            or digest(log_path) != log["sha256"]
        ):
            raise ValueError("bound supervisor log changed")
        _scan_retained_text(
            log_path,
            output_root=log_path.parent,
            expected_job_id=receipt.get("job_id"),
            expected_generation=receipt.get("generation"),
            supervisor_receipt=receipt,
            supervisor_receipt_path=Path(prior_broker["output_dir"]) / "receipt.json",
        )
        verify_mechanical_stop(
            receipt,
            prior_broker,
            comp["prior_manifest_sha256"],
            export_root=[Path(r) for r in raw_exp_dirs],
            supervisor_log_path=log_path,
        )
        inherited_roots: set[str] = set()
        if isinstance(prior_broker.get("completion_attempt"), dict):
            inherited_roots.update(prior_broker["completion_attempt"]["prior_output_dirs"])
            inherited_roots.update(prior_broker["completion_attempt"]["prior_export_roots"])
        for root in roots:
            if root not in inherited_roots:
                _check_attempt_base_fault_evidence(
                    Path(root),
                    expected_job_id=receipt.get("job_id"),
                    expected_generation=receipt.get("generation"),
                )

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


@_evidence_session
def validate_packet(
    packet: Path, *, require_committed: bool = True, allow_tiny: bool = False
) -> dict[str, Any]:
    """Bind C/T runtime controls to committed seed audit and frozen topology contract."""
    _no_symlink_path(packet)
    original_bytes = _bounded_manifest_bytes(packet / "dispatch-manifest.json")
    broker = _load_json(original_bytes.decode("utf-8"))
    require_current_packet(broker, packet, root=ROOT, original_bytes=original_bytes)
    payload = preflight(
        packet / "sandbox.json", require_committed=require_committed, run_tests=False
    )
    control = _load_json((packet / "control.json").read_text())
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
    if control != default or control != _load_json(broker["env"]["SANDBOX_CT_CONTROL_JSON"]):
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

    topology_manifest = _load_json((packet / "topology-manifest.json").read_text())
    if topology_manifest != dict(TOPOLOGY_SPEC):
        raise ValueError("topology-manifest.json differs from frozen TOPOLOGY_SPEC")

    check = _load_json((packet / "registry-check.json").read_text())
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
    validate_uploaded_ledger(
        packet, broker, control, require_committed=require_committed, manifest_bytes=original_bytes
    )
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, nargs="?")
    parser.add_argument(
        "--cuda-semantics",
        action="store_true",
        help="run the logical-CUDA / meta-device semantics suite (also included in full preflight)",
    )
    args = parser.parse_args()
    if not args.cuda_semantics and args.packet is None:
        parser.error("packet directory is required unless --cuda-semantics is specified")
    check: dict[str, Any] | None = None
    if args.packet is not None:
        check = validate_packet(args.packet)
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
