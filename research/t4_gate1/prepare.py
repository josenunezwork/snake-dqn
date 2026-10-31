"""Offline Mac preparation, real-entrypoint local preflight, and ready-queue sealer for Gate 1.

Subcommands (never dispatches or touches Colab):
  - ``inputs``: runs outcome-blind ``seed_audit.audit()``, stages frozen configs + ledgers +
    ``public-inputs.json``, packs them into one deterministic ``input-metadata.tar.gz`` (< 25 KB),
    and verifies ``stage_metadata`` round-trip.
  - ``preflight``: executes the REAL ``research/t4_gate1/job.sh`` on Mac CPU against tiny
    Watch6-v2 ``DeviceSim(exact_mechanics=True, hero_respawn=True)`` worlds with real pinned
    ``b8_s3`` + ``frp3_s12`` checkpoints, verifies resume-skip self-check, and runs ``readout.py``.
  - ``packet``: verifies the 40-char published ``origin/*`` commit pin, runs clean-venv
    ``import_preflight.check``, enforces ``<= 8`` transfers and ``<= 60.0 s`` upload forecast on
    both canary and main, writes validated ``canary-01/manifest.json`` and
    ``main-01/manifest.template.json`` (leaving ``main-01/manifest.json`` uncreated until canary
    passes).
  - ``seal-main``: resolves the final passing canary ``receipt.json`` (following ``ready_queue``
    retry chains), runs ``canary.check`` + ``supervisor.validate``, and writes
    ``main-01/manifest.json`` create-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.colab_cli import canary
from research.colab_cli import supervisor as colab_supervisor
from research.t4_gate1 import (
    ALLOCATION_INDEX,
    ALLOCATION_NAME,
    B8_S3_SHA256,
    CANARY_NAMESPACE,
    FRP3_S12_SHA256,
    MAIN_NAMESPACE,
    PREFLIGHT_NAMESPACE,
    STUDY,
    T4_FULL_GO_PREREQUISITES,
    T4_GATE1_THRESHOLDS,
    UNIT_KIND,
    seed_audit,
)
from research.t4_gate1.pipeline import load_and_validate_ledger, publish_verified_ledger
from research.t4_gate1.readout import aggregate_gate1_readout
from research.t4_gate1.uploads import (
    MAX_FILE_BYTES,
    METADATA,
    check_uploads,
    pack_inputs,
    stage_metadata,
)
from tools.ops import auto_retry, import_preflight, ready_queue
from tools.ops.dispatch import require_resume_policy
from tools.ops.package_spec import Input, digest, publish_json, stage_inputs

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t4_gate1"
SCRIPT = "research/t4_gate1/job.sh"
REQUIREMENTS = "research/t4_gate1/requirements.txt"
PUBLIC_INPUTS_CATALOG = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
DEFAULT_LOCAL_CKPTS = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")

QUALIFIED_COLAB_CLI_PREFIX = "36285bc6"

RUNTIME_IMPORT_ROOTS: tuple[str, ...] = (
    "research.t4_gate1.pipeline",
    "research.t4_gate1.uploads",
)

GPU_SIM_NATIVE_EXTENSIONS: tuple[str, ...] = (
    "research/gpu_sim_spike/event_cuda.cpp",
    "research/gpu_sim_spike/event_cuda.cu",
    "research/gpu_sim_spike/flood_cuda.cpp",
    "research/gpu_sim_spike/flood_cuda.cu",
    "research/gpu_sim_spike/move_cuda.cpp",
    "research/gpu_sim_spike/move_cuda.cu",
    "research/gpu_sim_spike/v8_indexed.cpp",
    "research/gpu_sim_spike/v8_indexed.cu",
    "research/gpu_sim_spike/vector_geometry.cpp",
    "research/gpu_sim_spike/vector_geometry.cu",
)

# Priced A100 rate in Colab CLI accounting (8.0 u/h conservative accounting envelope;
# observed A100 burn ~6.73 u/h -> 90m base forecast ~10.1 observed units, 12.0 priced units).
A100_PRICED_RATE_UNITS_PER_HOUR = 8.0
A100_OBSERVED_RATE_UNITS_PER_HOUR = 6.73
CANARY_BASE_EST_MINUTES = 12
MAIN_BASE_EST_MINUTES = 90


def t4_gate1_source_paths(repo: Path = ROOT) -> list[str]:
    """Compute minimal deterministic sparse source closure for Gate 1."""
    closure_modules, _ = import_preflight.import_closure(repo, list(RUNTIME_IMPORT_ROOTS))
    paths: set[str] = {
        "configs",
        "pyproject.toml",
        "requirements.txt",
        "research/apex_safety_20260926/deployment.yaml",
        "research/colab_cli",
        "research/t4_gate1",
        *GPU_SIM_NATIVE_EXTENSIONS,
    }
    for mod in closure_modules:
        mod_name = mod.partition(":")[0]
        resolved = import_preflight.module_path(repo, mod_name)
        if resolved is not None:
            paths.add(str(resolved.relative_to(repo)))
    return sorted(paths)


def build_frozen_configs() -> dict[str, dict[str, Any]]:
    """Return frozen deterministic runtime configs for main, canary, and local Mac preflight."""
    common = {
        "schema": "t34-t4-gate1-config/v1",
        "study": STUDY,
        "b8_checkpoint": "b8-s3.pth",
        "b8_sha256": B8_S3_SHA256,
        "frp3_checkpoint": "frp3-s12.pth",
        "frp3_sha256": FRP3_S12_SHA256,
        "verify_checkpoint_hashes": True,
        "seed_manifest": "seed-manifest.json",
        "compaction_interval": 10,
        "critic_base_seed": 17001,
        "critic_lr": 6e-4,
        "critic_weight_decay": 1e-3,
    }
    main_cfg = {
        **common,
        "namespace": MAIN_NAMESPACE,
        "allocation_key": "main",
        "canary_only": False,
        "run_skip_selfcheck": False,
        "session_ticks": 6000,
        "max_slab_envs": 64,
        "branch_max_slab_envs": 128,
        "max_capacity": 3200,
        "stride_mature": 4,
        "stride_early": 16,
        "min_mature_length": 600,
        "branch_horizon": 300,
        "fit_max_roots": 160,
        "test_max_roots": 128,
        "death_lookbacks": [16, 48, 112, 200],
        "ordinary_root_stride": 96,
        "train_branch_suffixes": [0, 1],
        "test_selection_suffixes": [0, 1],
        "test_scoring_suffixes": [4, 5],
        "critic_epochs": 16,
        "critic_batch_size": 256,
    }
    canary_cfg = {
        **common,
        "namespace": CANARY_NAMESPACE,
        "allocation_key": "canary",
        "canary_only": True,
        "run_skip_selfcheck": True,
        "session_ticks": 180,
        "max_slab_envs": 8,
        "branch_max_slab_envs": 32,
        "max_capacity": 1600,
        "stride_mature": 8,
        "stride_early": 16,
        "min_mature_length": 4,
        "branch_horizon": 24,
        "fit_max_roots": 4,
        "test_max_roots": 4,
        "death_lookbacks": [4, 12],
        "ordinary_root_stride": 24,
        "train_branch_suffixes": [0],
        "test_selection_suffixes": [0],
        "test_scoring_suffixes": [4],
        "critic_epochs": 2,
        "critic_batch_size": 32,
    }
    preflight_cfg = {
        **common,
        "namespace": PREFLIGHT_NAMESPACE,
        "allocation_key": "preflight",
        "canary_only": True,
        "run_skip_selfcheck": True,
        "session_ticks": 90,
        "max_slab_envs": 5,
        "branch_max_slab_envs": 16,
        "max_capacity": 800,
        "stride_mature": 6,
        "stride_early": 12,
        "min_mature_length": 4,
        "branch_horizon": 16,
        "fit_max_roots": 3,
        "test_max_roots": 3,
        "death_lookbacks": [4, 10],
        "ordinary_root_stride": 18,
        "train_branch_suffixes": [0],
        "test_selection_suffixes": [0],
        "test_scoring_suffixes": [4],
        "critic_epochs": 2,
        "critic_batch_size": 16,
    }
    return {
        "main": main_cfg,
        "canary": canary_cfg,
        "preflight": preflight_cfg,
    }


def prepare_inputs(
    out_root: Path,
    *,
    queue_subdir: str = "queue-01",
    ledger_override: Path | None = None,
    scan_filesystem: bool = True,
) -> dict[str, Any]:
    """Run outcome-blind seed audit and pack all Gate-1 small files into input-metadata.tar.gz."""
    out_root = out_root.resolve()
    queue_dir = out_root / queue_subdir
    if (queue_dir / METADATA).exists():
        raise FileExistsError(f"Refusing to overwrite packed metadata in {queue_dir}")

    audit_report = seed_audit.audit(repo=ROOT, scan_filesystem=scan_filesystem)
    if audit_report.get("status") != "CLEAR":
        raise ValueError(f"Seed disjointness audit did not return CLEAR: {audit_report}")

    out_root.mkdir(parents=True, exist_ok=True)
    seed_audit_path = out_root / "SEED_AUDIT.json"
    if not seed_audit_path.exists():
        publish_json(seed_audit_path, audit_report)

    raw_dir = queue_dir / "raw_inputs"
    raw_dir.mkdir(parents=True, exist_ok=False)

    configs = build_frozen_configs()
    publish_json(raw_dir / "main-config.json", configs["main"])
    publish_json(raw_dir / "canary-config.json", configs["canary"])
    publish_json(raw_dir / "preflight-config.json", configs["preflight"])

    seed_manifest = {
        "schema": "t34-t4-gate1-seed-manifest/v1",
        "study": STUDY,
        "allocations": audit_report["allocations"],
    }
    publish_json(raw_dir / "seed-manifest.json", seed_manifest)

    if ledger_override is not None:
        skipped = load_and_validate_ledger(
            ledger_override.resolve(),
            expected_namespace=MAIN_NAMESPACE,
            valid_replicates={0, 1},
        )
        if len(skipped) >= 2:
            raise ValueError("All 2 Gate-1 replicates already verified; never recompute")
        ledger_doc = json.loads(ledger_override.resolve().read_text(encoding="utf-8"))
        publish_json(raw_dir / "verified-ledger.json", ledger_doc)
    else:
        publish_verified_ledger(raw_dir / "verified-ledger.json", {}, namespace=MAIN_NAMESPACE)

    publish_verified_ledger(raw_dir / "canary-ledger.json", {}, namespace=CANARY_NAMESPACE)
    publish_verified_ledger(raw_dir / "preflight-ledger.json", {}, namespace=PREFLIGHT_NAMESPACE)

    pub_inputs_doc = json.loads(PUBLIC_INPUTS_CATALOG.read_text(encoding="utf-8"))
    publish_json(raw_dir / "public-inputs.json", pub_inputs_doc)

    input_specs = [
        Input(raw_dir / "canary-config.json", "canary-config.json"),
        Input(raw_dir / "canary-ledger.json", "canary-ledger.json"),
        Input(raw_dir / "main-config.json", "main-config.json"),
        Input(raw_dir / "preflight-config.json", "preflight-config.json"),
        Input(raw_dir / "preflight-ledger.json", "preflight-ledger.json"),
        Input(raw_dir / "seed-manifest.json", "seed-manifest.json"),
        Input(raw_dir / "verified-ledger.json", "verified-ledger.json"),
        Input(raw_dir / "public-inputs.json", "public-inputs.json"),
    ]
    staged = stage_inputs(queue_dir / "staged_inputs", input_specs)
    packed = pack_inputs(staged, queue_dir / METADATA)
    meta_row = next(row for row in packed if row["dest"] == METADATA)

    # Verify offline stage_metadata round-trip in an isolated temp directory.
    with tempfile.TemporaryDirectory(prefix="t4-gate1-pack-verify-") as tmp_s:
        tmp_dir = Path(tmp_s)
        shutil.copyfile(queue_dir / METADATA, tmp_dir / METADATA)
        verify_stage = stage_metadata(tmp_dir, str(meta_row["sha256"]))

    report = {
        "status": "PASS",
        "seed_audit_status": audit_report["status"],
        "total_seeds_audited": audit_report["total_seeds_audited"],
        "queue_dir": str(queue_dir),
        "metadata_archive": str(queue_dir / METADATA),
        "metadata_sha256": meta_row["sha256"],
        "metadata_size_bytes": meta_row["size_bytes"],
        "packed_rows": packed,
        "verified_stage_payloads": verify_stage["staged_payloads"],
    }
    publish_json(queue_dir / "PACKED_INPUTS_REPORT.json", report)
    return report


def run_local_preflight(
    out_root: Path,
    *,
    queue_subdir: str = "queue-01",
    local_checkpoint_dir: Path = DEFAULT_LOCAL_CKPTS,
) -> dict[str, Any]:
    """Execute the REAL research/t4_gate1/job.sh on tiny Watch6-v2 worlds and verify readout."""
    out_root = out_root.resolve()
    queue_dir = out_root / queue_subdir
    archive_path = queue_dir / METADATA
    if not archive_path.is_file():
        raise FileNotFoundError(f"Packed metadata archive missing at {archive_path}; run inputs")
    meta_sha = digest(archive_path)

    preflight_dir = out_root / "preflight"
    if preflight_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight dir: {preflight_dir}")
    runtime_root = preflight_dir / "runtime"
    runtime_inputs = runtime_root / "inputs"
    runtime_inputs.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(archive_path, runtime_inputs / METADATA)

    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()

    py_bin = str(Path(sys.executable).resolve().parent)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{py_bin}:{env.get('PATH', '')}",
            "PYTHONPATH": str(ROOT),
            "COLAB_CLI_ROOT": str(runtime_root),
            "COLAB_INPUT_DIR": str(runtime_inputs),
            "T4_GATE1_SOURCE_COMMIT": head_sha,
            "T4_GATE1_CONFIG_NAME": "preflight-config.json",
            "T4_GATE1_LEDGER_NAME": "preflight-ledger.json",
            "T4_GATE1_METADATA_SHA256": meta_sha,
            "T4_GATE1_DEVICE": "cpu",
            "T4_LOCAL_CHECKPOINT_DIR": str(local_checkpoint_dir.resolve(strict=True)),
        }
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        ["bash", str(ROOT / SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    wall_sec = round(time.perf_counter() - t0, 3)
    (preflight_dir / "job_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (preflight_dir / "job_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Local preflight job.sh failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    if "RuntimeWarning" in proc.stderr or "Mean of empty slice" in proc.stderr:
        raise RuntimeError(
            f"Local preflight emitted unexpected RuntimeWarning in stderr:\n{proc.stderr[-2000:]}"
        )

    out_gate_dir = runtime_root / "output" / "t4_gate1"
    run_summary = json.loads((out_gate_dir / "run-summary.json").read_text(encoding="utf-8"))
    skip_proof = json.loads((out_gate_dir / "skip-proof.json").read_text(encoding="utf-8"))
    if skip_proof.get("synthetic_full_ledger_selfcheck", {}).get("status") != "PASS":
        raise RuntimeError(f"Preflight skip-proof self-check did not PASS: {skip_proof}")

    readout = aggregate_gate1_readout(
        [runtime_root],
        expected_namespace=PREFLIGHT_NAMESPACE,
        expected_replicates=1,
        local_preflight=True,
        allow_canary_units=True,
    )
    publish_json(preflight_dir / "PREFLIGHT_READOUT.json", readout)

    report = {
        "schema": "t34-t4-gate1-local-preflight/v1",
        "status": "PASS",
        "entrypoint": SCRIPT,
        "head_commit_at_preflight": head_sha,
        "metadata_sha256": meta_sha,
        "wall_seconds": wall_sec,
        "run_summary": run_summary,
        "skip_proof": skip_proof,
        "world_disjointness_proof": readout["world_disjointness_proof"],
        "verified_units": readout["verified_units"],
        "preflight_readout_path": str(preflight_dir / "PREFLIGHT_READOUT.json"),
    }
    publish_json(preflight_dir / "LOCAL_PREFLIGHT_REPORT.json", report)
    return report


def _verify_published_commit_and_transport(commit: str, repo: Path = ROOT) -> dict[str, str]:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"commit must be a 40-char lowercase hex SHA, got {commit!r}")
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, check=True)
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=repo, text=True, timeout=15
    )
    if not any(line.strip().startswith("origin/") for line in refs.splitlines()):
        raise ValueError(f"commit {commit} is not pushed to any origin/* remote branch")

    src_paths = t4_gate1_source_paths(repo)
    diff = subprocess.check_output(["git", "diff", commit, "--", *src_paths], cwd=repo, timeout=15)
    if diff:
        raise ValueError("Working tree Gate-1 source closure differs from pinned commit")

    colab_cli_tree = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=repo, text=True, timeout=10
    ).strip()
    if not colab_cli_tree.startswith(QUALIFIED_COLAB_CLI_PREFIX):
        raise ValueError(
            f"Unexpected research/colab_cli tree {colab_cli_tree} (expected prefix "
            f"{QUALIFIED_COLAB_CLI_PREFIX})"
        )
    return {"commit": commit, "colab_cli_tree": colab_cli_tree}


def build_packet_manifests(
    out_root: Path,
    *,
    commit: str,
    queue_subdir: str = "queue-01",
    packet_tag: str = "01",
    attempt: int = 1,
    run_clean_import_preflight: bool = True,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Build and validate commit-pinned canary manifest + unsealed main template."""
    out_root = out_root.resolve()
    queue_dir = out_root / queue_subdir
    packed_report_path = queue_dir / "PACKED_INPUTS_REPORT.json"
    if not packed_report_path.is_file():
        raise FileNotFoundError(f"Missing {packed_report_path}; run `prepare.py inputs` first")
    preflight_report_path = out_root / "preflight" / "LOCAL_PREFLIGHT_REPORT.json"
    if not preflight_report_path.is_file():
        raise FileNotFoundError(
            f"Missing measured local preflight {preflight_report_path}; "
            "run `prepare.py preflight` first"
        )
    preflight_report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    if preflight_report.get("status") != "PASS":
        raise ValueError("Local Mac preflight report is not PASS")

    pin_info = _verify_published_commit_and_transport(commit, repo=repo)
    source_paths = t4_gate1_source_paths(repo)

    import_report_path = queue_dir / "IMPORT_PREFLIGHT.json"
    if run_clean_import_preflight:
        if import_report_path.exists():
            raise FileExistsError(f"Refusing to overwrite {import_report_path}")
        imp_res = import_preflight.check(
            repo=repo,
            commit=commit,
            source_paths=source_paths,
            requirements=REQUIREMENTS,
            modules=list(RUNTIME_IMPORT_ROOTS),
        )
        publish_json(import_report_path, imp_res)

    packed_report = json.loads(packed_report_path.read_text(encoding="utf-8"))
    packed_rows = packed_report["packed_rows"]
    metadata_sha256 = str(packed_report["metadata_sha256"])
    manifest_inputs = [{k: row[k] for k in ("path", "dest", "sha256")} for row in packed_rows]

    common_env: dict[str, str] = {
        "COLAB_CLI_REQUIREMENTS": REQUIREMENTS,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "GPU_SIM_V8_INDEXED": "0",
        "GPU_SIM_VECTOR_FLOOD": "static",
        "GPU_SIM_VECTOR_GEOMETRY": "native",
        "MAX_JOBS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OMP_THREAD_LIMIT": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "T4_GATE1_DEVICE": "cuda",
        "T4_GATE1_METADATA_SHA256": metadata_sha256,
        "T4_GATE1_SOURCE_COMMIT": commit,
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
        "VECLIB_MAXIMUM_THREADS": "1",
    }

    # --- 1. Short A100 family canary ---
    canary_dir = queue_dir / f"canary-{packet_tag}"
    canary_dir.mkdir(parents=True, exist_ok=False)
    canary_base_units = round(CANARY_BASE_EST_MINUTES / 60.0 * A100_PRICED_RATE_UNITS_PER_HOUR, 4)
    canary_base_reserve = 0.4
    canary_manifest: dict[str, Any] = {
        "job_id": f"t4-gate1-critic-canary-{packet_tag}",
        "commit": commit,
        "script": SCRIPT,
        "source_paths": source_paths,
        "device": "cuda",
        "lease_seconds": 300,
        "timeout_minutes": 2 * CANARY_BASE_EST_MINUTES,
        "output_dir": str(canary_dir / "output"),
        "max_export_bytes": MAX_FILE_BYTES,
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": CANARY_BASE_EST_MINUTES,
        },
        "budget": {
            "rate_units_per_hour": A100_PRICED_RATE_UNITS_PER_HOUR,
            "unit_cap": round(2.0 * canary_base_units, 4),
            "reserve_units": round(2.0 * canary_base_reserve, 4),
            "other_assignments": {},
        },
        "resume_exemption": {
            "kind": "short-canary",
            "reason": (
                "Short A100 family transport, DeviceSim v2 harvest, and CUDA 6-action "
                "PrivilegedFactoredCritic probe (24m timeout <= 30m cap)."
            ),
        },
        "required_exports": [
            "t4_gate1/run-summary.json",
            "t4_gate1/skip-proof.json",
            "t4_gate1/units/replicate-0000.json",
        ],
        "inputs": manifest_inputs,
        "env": {
            **common_env,
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE,
            "T4_GATE1_CONFIG_NAME": "canary-config.json",
            "T4_GATE1_LEDGER_NAME": "canary-ledger.json",
        },
        "t4_gate1": {
            "study": STUDY,
            "lane": "exploration",
            "phase": "canary",
            "namespace": CANARY_NAMESPACE,
            "base_est_minutes": CANARY_BASE_EST_MINUTES,
            "effective_timeout_minutes": 2 * CANARY_BASE_EST_MINUTES,
            "base_priced_units": canary_base_units,
            "effective_unit_cap": round(2.0 * canary_base_units, 4),
            "base_reserve_units": canary_base_reserve,
            "effective_reserve_units": round(2.0 * canary_base_reserve, 4),
        },
    }
    canary_manifest["canary"] = {
        "schema": canary.SCHEMA,
        "mode": "probe",
        "family": canary.family(canary_manifest),
    }
    require_resume_policy(canary_manifest)
    canary.check(canary_manifest, repo)
    validated_canary = colab_supervisor.validate(canary_manifest, repo=repo)
    canary_upload_budget = check_uploads(validated_canary, repo=repo)
    publish_json(queue_dir / "UPLOAD_BUDGET_CANARY.json", canary_upload_budget)
    canary_manifest_path = canary_dir / "manifest.json"
    with canary_manifest_path.open("xt", encoding="utf-8") as stream:
        stream.write(json.dumps(validated_canary, indent=2, sort_keys=True) + "\n")

    # --- 2. Main A100 Gate-1 run (unsealed template; sealed once canary passes) ---
    main_dir = queue_dir / f"main-{packet_tag}"
    main_dir.mkdir(parents=True, exist_ok=False)
    main_base_priced_units = round(
        MAIN_BASE_EST_MINUTES / 60.0 * A100_PRICED_RATE_UNITS_PER_HOUR, 4
    )
    main_base_observed_units = round(
        MAIN_BASE_EST_MINUTES / 60.0 * A100_OBSERVED_RATE_UNITS_PER_HOUR, 2
    )
    main_base_reserve = 1.2
    main_template: dict[str, Any] = {
        "job_id": f"t4-gate1-critic-main-{packet_tag}",
        "commit": commit,
        "script": SCRIPT,
        "source_paths": source_paths,
        "device": "cuda",
        "lease_seconds": 300,
        "timeout_minutes": 2 * MAIN_BASE_EST_MINUTES,
        "output_dir": str(main_dir / "output"),
        "max_export_bytes": MAX_FILE_BYTES,
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": MAIN_BASE_EST_MINUTES,
        },
        "budget": {
            "rate_units_per_hour": A100_PRICED_RATE_UNITS_PER_HOUR,
            "unit_cap": round(2.0 * main_base_priced_units, 4),
            "reserve_units": round(2.0 * main_base_reserve, 4),
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": UNIT_KIND,
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "required_exports": [
            "t4_gate1/run-summary.json",
            "t4_gate1/skip-proof.json",
        ],
        "inputs": manifest_inputs,
        "env": {
            **common_env,
            "T4_GATE1_CONFIG_NAME": "main-config.json",
            "T4_GATE1_LEDGER_NAME": "verified-ledger.json",
        },
        "t4_gate1": {
            "study": STUDY,
            "lane": "exploration",
            "phase": "main",
            "attempt": int(attempt),
            "namespace": MAIN_NAMESPACE,
            "allocation": ALLOCATION_NAME,
            "allocation_index": ALLOCATION_INDEX,
            "replicates": 2,
            "worlds_total": 384,
            "held_out_test_worlds_total": 128,
            "held_out_mature_roots_target": 256,
            "base_est_minutes": MAIN_BASE_EST_MINUTES,
            "effective_timeout_minutes": 2 * MAIN_BASE_EST_MINUTES,
            "base_observed_units_forecast": main_base_observed_units,
            "base_priced_units_forecast": main_base_priced_units,
            "effective_unit_cap_2x": round(2.0 * main_base_priced_units, 4),
            "base_reserve_units": main_base_reserve,
            "effective_reserve_units_2x": round(2.0 * main_base_reserve, 4),
            "pre_stated_thresholds": dict(T4_GATE1_THRESHOLDS),
            "full_track4_go_prerequisites": list(T4_FULL_GO_PREREQUISITES),
        },
    }
    require_resume_policy(main_template)
    main_upload_budget = check_uploads(main_template, repo=repo)
    publish_json(queue_dir / "UPLOAD_BUDGET_MAIN.json", main_upload_budget)
    main_template_path = main_dir / "manifest.template.json"
    with main_template_path.open("xt", encoding="utf-8") as stream:
        stream.write(json.dumps(main_template, indent=2, sort_keys=True) + "\n")

    packet_summary = {
        "schema": "t34-t4-gate1-packet/v1",
        "status": "READY_NOT_DISPATCHED",
        "commit": commit,
        "colab_cli_tree": pin_info["colab_cli_tree"],
        "canary_manifest": str(canary_manifest_path),
        "canary_manifest_sha256": digest(canary_manifest_path),
        "main_template": str(main_template_path),
        "main_template_sha256": digest(main_template_path),
        "main_sealed_manifest_target": str(main_dir / "manifest.json"),
        "canary_upload_budget": canary_upload_budget,
        "main_upload_budget": main_upload_budget,
        "local_preflight_wall_seconds": preflight_report["wall_seconds"],
    }
    publish_json(queue_dir / "PACKET_SUMMARY.json", packet_summary)
    return packet_summary


def resolve_canary_receipt(ops: Path, canary_id: str) -> Path:
    """Follow ready_queue auto-retry and internal -retry-1 chains to the final PASS receipt."""
    queue = auto_retry.read(ops / "ready-queue.json")
    row = ready_queue.latest(queue, canary_id)
    if row.get("status") != "PASS":
        raise ValueError(
            f"Canary {canary_id!r} has not completed with status PASS (got {row.get('status')!r})"
        )
    name = row["argv"][1]
    internal = ops / "manifests" / f"{name}-retry-1.json"
    manifest_path = internal if internal.exists() else Path(row["manifest"])
    manifest = auto_retry.read(manifest_path)
    receipt_path = Path(manifest["output_dir"]) / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Canary receipt missing at {receipt_path}")
    return receipt_path


def seal_main_manifest(
    *,
    template_path: Path,
    out_manifest_path: Path,
    broker: Path = ROOT,
    ops: Path | None = None,
    canary_queue_id: str | None = None,
    canary_receipt: Path | None = None,
    idempotent: bool = False,
) -> Path:
    """Bind passing family canary receipt into main manifest and validate with supervisor."""
    template_path = template_path.resolve(strict=True)
    out_manifest_path = out_manifest_path.resolve()
    broker = broker.resolve(strict=True)

    if (canary_receipt is None) == (ops is None or canary_queue_id is None):
        raise ValueError("Supply either --canary-receipt OR both --ops and --canary-queue-id")

    receipt_path = (
        canary_receipt.resolve(strict=True)
        if canary_receipt is not None
        else resolve_canary_receipt(ops.resolve(strict=True), str(canary_queue_id))
    )

    unsealed: Mapping[str, Any] = json.loads(template_path.read_text(encoding="utf-8"))
    sealed = json.loads(json.dumps(unsealed))
    sealed["canary"] = {
        "schema": canary.SCHEMA,
        "receipt": {
            "path": str(receipt_path),
            "sha256": digest(receipt_path),
        },
    }
    require_resume_policy(sealed)
    canary.check(sealed, broker)
    validated = colab_supervisor.validate(sealed, repo=broker)
    encoded = json.dumps(validated, indent=2, sort_keys=True) + "\n"

    if out_manifest_path.exists():
        if idempotent and out_manifest_path.read_text(encoding="utf-8") == encoded:
            return out_manifest_path
        raise FileExistsError(f"Refusing to overwrite existing sealed manifest {out_manifest_path}")

    with out_manifest_path.open("xt", encoding="utf-8") as stream:
        stream.write(encoded)
    return out_manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("inputs")
    p_in.add_argument("--out-root", type=Path, required=True)
    p_in.add_argument("--queue-subdir", type=str, default="queue-01")
    p_in.add_argument("--ledger", type=Path, default=None)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--out-root", type=Path, required=True)
    p_pre.add_argument("--queue-subdir", type=str, default="queue-01")
    p_pre.add_argument("--local-checkpoint-dir", type=Path, default=DEFAULT_LOCAL_CKPTS)

    p_pkt = sub.add_parser("packet")
    p_pkt.add_argument("--out-root", type=Path, required=True)
    p_pkt.add_argument("--commit", type=str, required=True)
    p_pkt.add_argument("--queue-subdir", type=str, default="queue-01")
    p_pkt.add_argument("--packet-tag", type=str, default="01")
    p_pkt.add_argument("--attempt", type=int, default=1)
    p_pkt.add_argument("--skip-import-preflight", action="store_true")

    p_seal = sub.add_parser("seal-main")
    p_seal.add_argument("--template", type=Path, required=True)
    p_seal.add_argument("--out", type=Path, required=True)
    p_seal.add_argument("--broker", type=Path, default=ROOT)
    p_seal.add_argument("--ops", type=Path, default=None)
    p_seal.add_argument("--canary-queue-id", type=str, default=None)
    p_seal.add_argument("--canary-receipt", type=Path, default=None)
    p_seal.add_argument("--idempotent", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "inputs":
        rep = prepare_inputs(
            args.out_root,
            queue_subdir=args.queue_subdir,
            ledger_override=args.ledger,
        )
        print(f"T4_GATE1_INPUTS {json.dumps(rep, sort_keys=True)}", flush=True)
        return 0

    if args.cmd == "preflight":
        rep = run_local_preflight(
            args.out_root,
            queue_subdir=args.queue_subdir,
            local_checkpoint_dir=args.local_checkpoint_dir,
        )
        print(
            f"T4_GATE1_PREFLIGHT status={rep['status']} wall_s={rep['wall_seconds']}",
            flush=True,
        )
        return 0

    if args.cmd == "packet":
        rep = build_packet_manifests(
            args.out_root,
            commit=args.commit,
            queue_subdir=args.queue_subdir,
            packet_tag=args.packet_tag,
            attempt=args.attempt,
            run_clean_import_preflight=not args.skip_import_preflight,
        )
        print(f"T4_GATE1_PACKET {json.dumps(rep, sort_keys=True)}", flush=True)
        return 0

    if args.cmd == "seal-main":
        out = seal_main_manifest(
            template_path=args.template,
            out_manifest_path=args.out,
            broker=args.broker,
            ops=args.ops,
            canary_queue_id=args.canary_queue_id,
            canary_receipt=args.canary_receipt,
            idempotent=args.idempotent,
        )
        print(f"T4_GATE1_SEALED {out}", flush=True)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
