"""Offline Mac preparation, timing probe, packet sealing, and post-canary main sealing for R7.

Never allocates a Colab runtime, invokes ``colab`` CLI commands, or touches held-out
Watch6 V2 exports or C/T ``assessment`` roots (``assessment_roots_used == 0``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.bl_upload import convert_public_checkpoints
from research.bl_upload import qualify as qualify_upload_budget
from research.colab_cli.supervisor import validate as validate_colab_manifest
from research.gpu_sim_spike.make_manifests import checkpoint_inputs
from research.r7_macro_fallback import CT_BANK_SHA256, CT_LABEL_SHA256S
from research.r7_macro_fallback.options import (
    run_mac_timing_probe,
    state_from_compact_snapshot,
)
from research.r7_macro_fallback.search import (
    B8_S3_SHA256,
    UNIT_KIND,
    publish_verified_ledger,
)
from research.sandbox_a_resident.ct_branching import (
    ScoredRootChunk,
    compute_q_from_survival_and_mass,
)
from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore
from research.sandbox_a_resident.ct_task import (
    ORDINARY_LENGTH_BIN_EDGES,
    ORDINARY_LENGTH_BIN_LABELS,
)
from research.sandbox_a_resident.prepare import runtime_source_paths
from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256
from tools.ops import import_preflight
from tools.ops.package_spec import Input, stage_inputs

ROOT = Path(__file__).resolve().parents[2]
QUALIFIED_TRANSPORT_TREE_PREFIX = "36285bc6"
DEFAULT_CT_BANK_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
    "t34-ct-h5000-1029/revision-02/attempt-02/output-a100-fallback/exports/ct/stage-0001.pt"
)
DEFAULT_CT_STAGE2_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
    "t34-ct-h5000-stage2-1030/attempt-01/output-a100-fallback/exports/ct"
)
DEFAULT_CKPT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
DEFAULT_OUT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-mac-checks")


def _sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _verify_sidecar_file(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    complete = Path(f"{path}.complete")
    if not (path.is_file() and sidecar.is_file() and complete.is_file()):
        raise ValueError(f"Missing immutable sidecars for {path}")
    if any(p.is_symlink() for p in (path, sidecar, complete)):
        raise ValueError(f"Refusing symlinked stage artifact {path}")
    expected_sha = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch on {path}: expected {expected_sha}, got {actual_sha}")
    return actual_sha


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _length_bin_label(length: int) -> str:
    idx = 0
    for edge in ORDINARY_LENGTH_BIN_EDGES[1:]:
        if length >= edge:
            idx += 1
        else:
            break
    return ORDINARY_LENGTH_BIN_LABELS[idx]


def r7_source_paths(commit: str, repo: Path = ROOT) -> list[str]:
    """Return sparse runtime source closure for ``research.r7_macro_fallback.search``."""
    tracked = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "src",
            "tools",
            "research/sandbox_a_resident",
            "research/r7_macro_fallback",
        ],
        cwd=repo,
        text=True,
    ).splitlines()
    r7_runtime_names = {
        "__init__.py",
        "options.py",
        "search.py",
        "job.sh",
        "requirements.txt",
    }
    ct_runtime_names = {
        "__init__.py",
        "ct_branching.py",
        "ct_json.py",
        "ct_liveness.py",
        "ct_mechanics.py",
        "ct_snapshots.py",
        "ct_task.py",
        "ct_topology.py",
        "ct_units.py",
        "device_sim.py",
        "training_rosters.py",
        "training_sim.py",
    }
    additions = [
        name
        for name in tracked
        if name in {"tools/__init__.py", "tools/ops/__init__.py", "tools/ops/package_spec.py"}
        or (
            Path(name).parent.as_posix() == "research/r7_macro_fallback"
            and Path(name).name in r7_runtime_names
        )
        or (
            Path(name).parent.as_posix() == "research/sandbox_a_resident"
            and Path(name).name in ct_runtime_names
        )
    ]
    return sorted(set(runtime_source_paths(commit)) | set(additions))


def prepare_inputs(
    *,
    ct_bank_path: Path,
    ct_stage2_dir: Path,
    checkpoint_dir: Path,
    out_root: Path,
    num_batches: int = 3,
) -> dict[str, Any]:
    """Extract C/T train-only 1-step-all-die roots & baseline without opening assessment."""
    if not ct_bank_path.is_file():
        raise FileNotFoundError(f"Missing C/T Stage-1 bank at {ct_bank_path}")
    stage1_sha = _verify_sidecar_file(ct_bank_path)
    if stage1_sha != CT_BANK_SHA256:
        raise ValueError(
            f"Unexpected C/T Stage-1 bank SHA: expected {CT_BANK_SHA256}, got {stage1_sha}"
        )
    if ct_bank_path.suffix == ".xz":
        with lzma.open(ct_bank_path, "rb") as stream:
            raw_bytes = stream.read()
        raw_stage1 = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
    else:
        raw_stage1 = torch.load(ct_bank_path, map_location="cpu", weights_only=False)
    stage1_source = str(ct_bank_path)

    if raw_stage1.get("schema") == "ct-stage-0001/v1":
        dataset_dict = raw_stage1["dataset"]
    else:
        dataset_dict = raw_stage1
    if dataset_dict.get("schema") != "ct-collected-dataset/v1":
        raise ValueError(f"Foreign Stage-1 dataset schema: {dataset_dict.get('schema')!r}")

    # Reconstruct ONLY the root snapshot store (skipping the 50k-row anchor archive).
    store = LosslessBoundedSnapshotStore.from_state_dict(dataset_dict["store"])
    total_worlds = int(dataset_dict["control"]["worlds"])
    max_body_capacity = int(dataset_dict["control"]["max_body_capacity"])

    chunk_shas: dict[str, str] = {}
    chunks_by_root: dict[int, tuple[ScoredRootChunk, int]] = {}
    for chunk_idx, expected_label_sha in enumerate(CT_LABEL_SHA256S, start=1):
        c_path = ct_stage2_dir / f"stage-{chunk_idx:04d}.pt"
        actual_c_sha = _verify_sidecar_file(c_path)
        if actual_c_sha != expected_label_sha:
            raise ValueError(
                f"Unexpected Stage-2 chunk SHA on {c_path.name}: "
                f"expected {expected_label_sha}, got {actual_c_sha}"
            )
        chunk_shas[c_path.name] = actual_c_sha
        chunk = ScoredRootChunk.from_state_dict(
            torch.load(c_path, map_location="cpu", weights_only=False)
        )
        for local_pos, r_idx in enumerate(chunk.root_indices.tolist()):
            r_int = int(r_idx)
            if r_int in chunks_by_root:
                raise ValueError(f"Duplicate root_index {r_int} across C/T Stage-2 chunks")
            chunks_by_root[r_int] = (chunk, local_pos)

    # Filter strictly to partition == "train" BEFORE touching snapshots or outcomes.
    train_indices = [idx for idx, meta in enumerate(store.metadata) if meta.partition == "train"]
    assessment_ignored = sum(1 for meta in store.metadata if meta.partition != "train")

    train_baseline_rows: list[dict[str, Any]] = []
    all_die_entries: list[dict[str, Any]] = []
    l800_timing_snapshots: list[dict[str, Any]] = []

    for r_idx in train_indices:
        meta = store.metadata[r_idx]
        if meta.partition != "train":
            raise AssertionError("Non-train root leaked past partition filter")
        chunk, pos = chunks_by_root[r_idx]
        leg = chunk.legal_mask[pos].astype(bool)
        legal_idxs = np.flatnonzero(leg)
        if legal_idxs.size == 0:
            raise ValueError(f"Empty legal_mask on train root {r_idx}")

        b8_a = int(chunk.b8_action[pos])
        s0_vec = np.nan_to_num(chunk.survive_sel_raw[pos, :, 0].astype(np.float64), nan=0.0)
        m0_vec = np.nan_to_num(chunk.mass_sel_raw[pos, :, 0].astype(np.float64), nan=0.0)
        q0_vec = compute_q_from_survival_and_mass(s0_vec[None, :], m0_vec[None, :], leg[None, :])[0]
        q0_max = float(np.max(q0_vec[legal_idxs]))
        tied0 = legal_idxs[q0_vec[legal_idxs] == q0_max]
        a_sel0 = b8_a if b8_a in tied0 else int(tied0[0])
        best_s_sel0 = float(s0_vec[a_sel0])
        all_die_sel0 = bool(np.max(s0_vec[legal_idxs]) < 0.5)

        es_s_score = float(chunk.s_score[pos, a_sel0])
        es_m_score = float(chunk.m_score[pos, a_sel0])
        ct4_a = int(chunk.a_star[pos])
        ct4_s_score = float(chunk.s_score[pos, ct4_a])
        ct4_m_score = float(chunk.m_score[pos, ct4_a])
        lbin = _length_bin_label(int(meta.hero_length))

        row_meta = {
            "root_index": int(r_idx),
            "root_uid": str(meta.root_uid),
            "partition": "train",
            "is_assessment": False,
            "kind": str(meta.kind),
            "world_id": int(meta.world_id),
            "mix": str(meta.mix),
            "life_id": int(meta.life_id),
            "tick": int(meta.tick),
            "lookback": None if meta.lookback is None else int(meta.lookback),
            "hero_length": int(meta.hero_length),
            "length_bin": lbin,
            "b8_action": b8_a,
            "exact_stop_action_sel0": int(a_sel0),
            "exact_stop_s_sel0": round(best_s_sel0, 6),
            "all_die_sel0": all_die_sel0,
            "exact_stop_s_score": round(es_s_score, 6),
            "exact_stop_m_score": round(es_m_score, 4),
            "ct4_a_star": ct4_a,
            "ct4_a_star_s_score": round(ct4_s_score, 6),
            "ct4_a_star_m_score": round(ct4_m_score, 4),
        }
        train_baseline_rows.append(row_meta)

        if all_die_sel0:
            snap = store.snapshots[r_idx]
            snap_dict = snap.to_state_dict()
            all_die_entries.append({**row_meta, "snapshot": snap_dict})
            if int(meta.hero_length) >= 800:
                l800_timing_snapshots.append(snap_dict)

    # Assign deterministic balanced batch indices 0 .. num_batches-1.
    for pos, entry in enumerate(all_die_entries):
        entry["batch_index"] = int(pos % num_batches)

    preflight_dir = out_root / "r7-preflight"
    queue_dir = out_root / "r7-queue"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = {
        "schema": "t34-r7-train-exact-stop-baseline/v1",
        "assessment_roots_used": 0,
        "assessment_roots_ignored_unopened": int(assessment_ignored),
        "stage1_source": stage1_source,
        "stage1_sha256": stage1_sha,
        "stage2_chunk_sha256s": chunk_shas,
        "train_roots_total": len(train_baseline_rows),
        "train_1step_surviving_sel0": sum(1 for r in train_baseline_rows if not r["all_die_sel0"]),
        "train_1step_all_die_sel0": len(all_die_entries),
        "train_l800_all_die_sel0": sum(1 for r in all_die_entries if r["hero_length"] >= 800),
        "train_l800_death_all_die_sel0": sum(
            1 for r in all_die_entries if r["hero_length"] >= 800 and r["kind"] == "death_lookback"
        ),
        "train_roots": train_baseline_rows,
    }
    baseline_path = preflight_dir / "train-exact-stop-baseline.json"
    baseline_path.write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    timing_states_path = preflight_dir / "mac-timing-l800-snapshots.pt"
    torch.save(
        {
            "schema": "t34-r7-mac-timing-snapshots/v1",
            "assessment_roots_used": 0,
            "snapshots": l800_timing_snapshots,
        },
        timing_states_path,
    )

    staging_tmp = preflight_dir / "_raw_inputs"
    staging_tmp.mkdir(parents=True, exist_ok=True)
    roots_buf = io.BytesIO()
    torch.save(
        {
            "schema": "t34-r7-train-all-die-roots/v1",
            "assessment_roots_used": 0,
            "total_worlds": total_worlds,
            "max_body_capacity": max_body_capacity,
            "num_batches": num_batches,
            "num_roots": len(all_die_entries),
            "stage1_sha256": stage1_sha,
            "stage2_chunk_sha256s": chunk_shas,
            "roots": all_die_entries,
        },
        roots_buf,
    )
    roots_xz_path = staging_tmp / "train-all-die-roots.pt.xz"
    roots_xz_path.write_bytes(lzma.compress(roots_buf.getvalue(), preset=6))

    canary_cfg_raw = staging_tmp / "canary-config.json"
    canary_cfg_raw.write_text(
        json.dumps(
            {
                "schema": "t34-r7-search-config/v1",
                "canary_only": True,
                "verify_checkpoint_hashes": True,
                "b8_checkpoint": "b8-s3.pth",
                "frp3_checkpoint": "frp3-s12.pth",
                "champion_checkpoint": "champion.pth",
                "roots_file": "train-all-die-roots.pt.xz",
                "batch_indices": list(range(num_batches)),
                "max_roots_per_batch": 1,
                "horizon": 12,
                "macro_ticks": 3,
                "top_m_ply1": 3,
                "selection_suffix": 0,
                "scoring_suffixes": [4, 5, 6, 7],
                "compaction_interval": 5,
                "max_slab_envs": 64,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    main_cfg_raw = staging_tmp / "main-config.json"
    main_cfg_raw.write_text(
        json.dumps(
            {
                "schema": "t34-r7-search-config/v1",
                "canary_only": False,
                "verify_checkpoint_hashes": True,
                "b8_checkpoint": "b8-s3.pth",
                "frp3_checkpoint": "frp3-s12.pth",
                "champion_checkpoint": "champion.pth",
                "roots_file": "train-all-die-roots.pt.xz",
                "batch_indices": list(range(num_batches)),
                "max_roots_per_batch": None,
                "horizon": 300,
                "macro_ticks": 15,
                "top_m_ply1": 3,
                "selection_suffix": 0,
                "scoring_suffixes": [4, 5, 6, 7],
                "compaction_interval": 10,
                "max_slab_envs": 512,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    b8_src = checkpoint_dir / "b8_s3_t7_20261016.pth"
    frp3_src = checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth"
    champ_src = checkpoint_dir / "champion_a5_freespace_20260621.pth"
    if _sha256_file(b8_src) != B8_S3_SHA256:
        raise ValueError(f"Unexpected B8 SHA at {b8_src}")
    if _sha256_file(frp3_src) != FRP3_S12_SHA256:
        raise ValueError(f"Unexpected FRP3 SHA at {frp3_src}")
    if _sha256_file(champ_src) != CHAMPION_SHA256:
        raise ValueError(f"Unexpected champion SHA at {champ_src}")
    checkpoint_inputs(checkpoint_dir)

    raw_ledger_path = staging_tmp / "verified-ledger.json"
    for suffix in ("", ".sha256", ".complete"):
        Path(f"{raw_ledger_path}{suffix}").unlink(missing_ok=True)
    publish_verified_ledger(raw_ledger_path, {})

    stage_seq = 1
    while (queue_dir / f"inputs-{stage_seq:02d}").exists():
        stage_seq += 1
    inputs_dir = queue_dir / f"inputs-{stage_seq:02d}"
    staged_rows = stage_inputs(
        inputs_dir,
        [
            Input(source=roots_xz_path, destination="train-all-die-roots.pt.xz"),
            Input(source=canary_cfg_raw, destination="canary-config.json"),
            Input(source=main_cfg_raw, destination="main-config.json"),
            Input(source=raw_ledger_path, destination="verified-ledger.json"),
        ],
    )
    staged = [
        {"dest": row["dest"], "path": row["path"], "sha256": row["sha256"]}
        for row in sorted(staged_rows, key=lambda item: item["dest"])
    ]
    (queue_dir / "STAGED_INPUTS.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "staged_inputs_dir": str(inputs_dir.resolve()),
                "inputs": staged,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    total_upload_bytes = sum(Path(item["path"]).stat().st_size for item in staged)
    summary = {
        "assessment_roots_used": 0,
        "train_roots_total": len(train_baseline_rows),
        "train_1step_all_die_sel0": len(all_die_entries),
        "train_l800_all_die_sel0": baseline_payload["train_l800_all_die_sel0"],
        "train_l800_death_all_die_sel0": baseline_payload["train_l800_death_all_die_sel0"],
        "num_batches": num_batches,
        "staged_inputs_dir": str(inputs_dir),
        "total_upload_bytes": int(total_upload_bytes),
    }
    print(f"PREPARED_R7_INPUTS {json.dumps(summary, sort_keys=True)}")
    return summary


def run_mac_timing(
    *,
    out_root: Path,
    ticks: int = 50_000,
    warmup_ticks: int = 200,
) -> dict[str, Any]:
    """Run 1-thread Mac option-tracker latency probe on C/T train L>=800 all-die snapshots."""
    preflight_dir = out_root / "r7-preflight"
    snaps_payload = torch.load(
        preflight_dir / "mac-timing-l800-snapshots.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(snaps_payload.get("assessment_roots_used", -1)) != 0:
        raise ValueError("Refusing timing snapshots with nonzero assessment_roots_used")
    from research.sandbox_a_resident.ct_snapshots import CompactWorldSnapshot

    states = [
        state_from_compact_snapshot(CompactWorldSnapshot.from_state_dict(sd))
        for sd in snaps_payload["snapshots"]
    ]
    report = run_mac_timing_probe(states, ticks=ticks, warmup_ticks=warmup_ticks)
    report["assessment_roots_used"] = 0
    out_path = preflight_dir / "MAC_TRACKER_TIMING.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"R7_MAC_TIMING {json.dumps(report, sort_keys=True)}")
    return report


def prepare_packet(
    *,
    commit: str,
    out_root: Path,
    repo: Path = ROOT,
    canary_job_id: str = "r7-macro-fallback-canary-01",
    main_job_id: str = "r7-macro-fallback-main-01",
) -> dict[str, Any]:
    """Verify published commit + sparse import closure, and write canary + main template."""
    full_commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=repo, text=True
    ).strip()
    is_published = subprocess.run(
        ["git", "merge-base", "--is-ancestor", full_commit, "origin/main"],
        cwd=repo,
    )
    if is_published.returncode != 0:
        raise ValueError(
            f"Commit {full_commit} is not yet published on origin/main; merge & push first"
        )
    transport_tree = subprocess.check_output(
        ["git", "rev-parse", f"{full_commit}:research/colab_cli"], cwd=repo, text=True
    ).strip()
    if not transport_tree.startswith(QUALIFIED_TRANSPORT_TREE_PREFIX):
        raise ValueError(
            f"Unexpected research/colab_cli tree {transport_tree} "
            f"(expected {QUALIFIED_TRANSPORT_TREE_PREFIX}*)"
        )

    paths = r7_source_paths(full_commit, repo=repo)
    queue_dir = out_root / "r7-queue"
    staged_doc = json.loads((queue_dir / "STAGED_INPUTS.json").read_text(encoding="utf-8"))
    staged = staged_doc["inputs"]
    ckpt_dir = Path(staged_doc.get("checkpoint_dir", str(DEFAULT_CKPT_DIR)))
    combined_inputs = [*checkpoint_inputs(ckpt_dir), *staged]
    required_exports = ["r7/run-summary.json", "r7/skip-proof.json"]

    canary_dir = queue_dir / "canary-01"
    canary_output_dir = (canary_dir / "output").resolve()
    canary_raw = convert_public_checkpoints(
        {
            "job_id": canary_job_id,
            "commit": full_commit,
            "script": "research/r7_macro_fallback/job.sh",
            "device": "cuda",
            "source_paths": paths,
            "requirements_file": "research/r7_macro_fallback/requirements.txt",
            "requirements": {
                "gpu": "A100",
                "allowed_gpus": ["A100"],
                "min_vcpu": 2,
                "est_minutes": 15,
            },
            "timeout_minutes": 30,
            "output_dir": str(canary_output_dir),
            "max_export_bytes": 256 * 1024 * 1024,
            "inputs": combined_inputs,
            "required_exports": required_exports,
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/r7_macro_fallback/requirements.txt",
                "R7_SOURCE_COMMIT": full_commit,
                "R7_CONFIG_NAME": "canary-config.json",
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": "research/r7_macro_fallback",
            },
            "budget": {
                "rate_units_per_hour": 8.0,
                "unit_cap": 4.0,
                "reserve_units": 0.8,
                "other_assignments": {},
            },
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    "target-platform A100 transport, sidecar export, and CUDA B8/CTDeviceSim "
                    "execution proof on 3 tiny C/T train roots before main dispatch"
                ),
            },
            "canary": {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": "research/r7_macro_fallback/job.sh",
            },
        },
        repo=repo,
    )
    if "public-inputs.json" not in {row["dest"] for row in canary_raw["inputs"]}:
        raise ValueError("convert_public_checkpoints did not bind public-inputs.json")

    preflight_report = import_preflight.check(
        repo,
        full_commit,
        canary_raw["source_paths"],
        ["research.r7_macro_fallback.search", "research.bl_upload.inputs"],
    )
    (queue_dir / "IMPORT_PREFLIGHT.json").write_text(
        json.dumps(preflight_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validated_canary = validate_colab_manifest(canary_raw, repo=repo)
    upload_qual = qualify_upload_budget(
        validated_canary,
        repo=repo,
        receipt_path=queue_dir / "UPLOAD_QUALIFICATION_CANARY.json",
    )
    _write_once_json(canary_dir / "manifest.json", validated_canary)

    main_dir = queue_dir / "main-01"
    main_output_dir = (main_dir / "output").resolve()
    # Owner's 2x rule on 0.4 A100-h (24 min base forecast -> 25 min est, 50 min effective timeout,
    # 3.33 u base -> 8.0 u effective unit_cap).
    main_template = convert_public_checkpoints(
        {
            "job_id": main_job_id,
            "commit": full_commit,
            "script": "research/r7_macro_fallback/job.sh",
            "device": "cuda",
            "source_paths": paths,
            "requirements_file": "research/r7_macro_fallback/requirements.txt",
            "requirements": {
                "gpu": "A100",
                "allowed_gpus": ["A100"],
                "min_vcpu": 2,
                "est_minutes": 25,
            },
            "timeout_minutes": 50,
            "output_dir": str(main_output_dir),
            "max_export_bytes": 256 * 1024 * 1024,
            "inputs": combined_inputs,
            "required_exports": required_exports,
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/r7_macro_fallback/requirements.txt",
                "R7_SOURCE_COMMIT": full_commit,
                "R7_CONFIG_NAME": "main-config.json",
            },
            "budget": {
                "rate_units_per_hour": 8.0,
                "unit_cap": 8.0,
                "reserve_units": 1.6,
                "other_assignments": {},
            },
            "resume": {
                "unit_kind": UNIT_KIND,
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "technical_envelope": {
                "base_forecast_a100_hours": 0.4,
                "base_est_minutes": 25,
                "effective_timeout_minutes_2x": 50,
                "base_unit_cap": 4.0,
                "effective_unit_cap_2x": 8.0,
                "assessment_roots_used": 0,
            },
        },
        repo=repo,
    )
    if "public-inputs.json" not in {row["dest"] for row in main_template["inputs"]}:
        raise ValueError("convert_public_checkpoints did not bind public-inputs.json in main")
    _write_once_json(main_dir / "manifest.template.json", main_template)
    summary = {
        "commit": full_commit,
        "transport_tree": transport_tree,
        "source_paths_count": len(canary_raw["source_paths"]),
        "normal_upload_bytes": upload_qual["normal_upload_bytes"],
        "normal_forecast_seconds": round(float(upload_qual["normal_forecast_seconds"]), 3),
        "fallback_upload_forecast_seconds": round(
            float(upload_qual["fallback_upload_forecast_seconds"]), 3
        ),
        "canary_manifest": str(canary_dir / "manifest.json"),
        "main_template": str(main_dir / "manifest.template.json"),
    }
    print(f"PREPARED_R7_PACKET {json.dumps(summary, sort_keys=True)}")
    return summary


def _locate_passing_canary_receipt(
    *,
    ops_dir: Path,
    canary_queue_id: str,
) -> Path:
    """Follow ready_queue retry chains & dispatch accounting retries to find canary receipt.json."""
    from tools.ops import auto_retry, ready_queue

    queue_file = ops_dir / "ready-queue.json"
    if not queue_file.is_file():
        raise FileNotFoundError(f"Missing ready-queue.json in {ops_dir}")
    queue = auto_retry.read(queue_file)
    final_row = ready_queue.latest(queue, canary_queue_id)
    if final_row.get("status") != "PASS":
        raise ValueError(
            f"Canary queue row {final_row.get('id')!r} has "
            f"status={final_row.get('status')!r}, not PASS"
        )
    dispatch_name = final_row["argv"][1]
    internal = ops_dir / "manifests" / f"{dispatch_name}-retry-1.json"
    live_manifest_path = internal if internal.is_file() else Path(final_row["manifest"])
    out_dir = Path(auto_retry.read(live_manifest_path)["output_dir"])
    receipt_path = (out_dir / "receipt.json").resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing canary receipt.json at {receipt_path}")
    return receipt_path


def seal_main_manifest(
    *,
    template_path: Path,
    manifest_out_path: Path,
    canary_receipt: Path | None = None,
    ops_dir: Path | None = None,
    canary_queue_id: str | None = None,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Bind the passing family canary receipt into ``main-01/manifest.json`` create-only."""
    if canary_receipt is not None:
        receipt_path = canary_receipt.expanduser().resolve(strict=True)
    elif ops_dir is not None and canary_queue_id:
        receipt_path = _locate_passing_canary_receipt(
            ops_dir=ops_dir.expanduser().resolve(),
            canary_queue_id=canary_queue_id,
        )
    else:
        raise ValueError("Provide either --canary-receipt or (--ops and --canary-queue-id)")

    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["canary"] = {
        "schema": "colab-family-canary/v1",
        "receipt": {
            "path": str(receipt_path),
            "sha256": _sha256_file(receipt_path),
        },
    }
    validated = validate_colab_manifest(template, repo=repo)
    _write_once_json(manifest_out_path, validated)
    summary = {
        "main_manifest": str(manifest_out_path),
        "main_manifest_sha256": _sha256_file(manifest_out_path),
        "bound_canary_receipt": str(receipt_path),
        "bound_canary_receipt_sha256": template["canary"]["receipt"]["sha256"],
    }
    print(f"SEALED_R7_MAIN {json.dumps(summary, sort_keys=True)}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("inputs")
    p_in.add_argument("--ct-bank-path", type=Path, default=DEFAULT_CT_BANK_PATH)
    p_in.add_argument("--ct-stage2-dir", type=Path, default=DEFAULT_CT_STAGE2_DIR)
    p_in.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p_in.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_in.add_argument("--num-batches", type=int, default=3)

    p_tm = sub.add_parser("mac-timing")
    p_tm.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_tm.add_argument("--ticks", type=int, default=50_000)
    p_tm.add_argument("--warmup-ticks", type=int, default=200)

    p_pk = sub.add_parser("packet")
    p_pk.add_argument("--commit", required=True)
    p_pk.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)

    p_sm = sub.add_parser("seal-main")
    p_sm.add_argument("--template", type=Path, required=True)
    p_sm.add_argument("--manifest", type=Path, required=True)
    p_sm.add_argument("--canary-receipt", type=Path, default=None)
    p_sm.add_argument("--ops", type=Path, default=None)
    p_sm.add_argument("--canary-queue-id", type=str, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "inputs":
        prepare_inputs(
            ct_bank_path=args.ct_bank_path,
            ct_stage2_dir=args.ct_stage2_dir,
            checkpoint_dir=args.checkpoint_dir,
            out_root=args.out_root,
            num_batches=args.num_batches,
        )
    elif args.cmd == "mac-timing":
        run_mac_timing(
            out_root=args.out_root,
            ticks=args.ticks,
            warmup_ticks=args.warmup_ticks,
        )
    elif args.cmd == "packet":
        prepare_packet(commit=args.commit, out_root=args.out_root)
    elif args.cmd == "seal-main":
        seal_main_manifest(
            template_path=args.template,
            manifest_out_path=args.manifest,
            canary_receipt=args.canary_receipt,
            ops_dir=args.ops,
            canary_queue_id=args.canary_queue_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
