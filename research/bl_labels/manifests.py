"""Prepare, but never dispatch, the F0 CPU-runtime canary pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .exports import empty_ledger, publish_json

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/bl_labels"
SCRIPT = f"{PACKAGE}/colab.sh"
QUALIFIED_RECORD = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha256() -> str:
    """Digest the F0 runtime closure as an additional manifest-visible pin."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / PACKAGE).glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def canary_allocation(seed_namespace: str = "f0-canary-v1") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream, count, horizon in (
        ("gate-frozen", 12, 5000),
        ("gate-mixed", 12, 5000),
        ("gate-scripted", 8, 5000),
        ("watch6", 4, 14400),
    ):
        for index in range(count):
            token = f"{seed_namespace}/{stream}/{index}".encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
            rows.append(
                {
                    "allocation_id": f"{stream}-{index:03d}",
                    "seed": seed,
                    "stream": stream,
                    "horizon": horizon,
                }
            )
    return rows


def materialize_gate_plans(allocation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze the existing H5000 roster plans before collection, never at runtime."""
    gate = [row for row in allocation if row["stream"].startswith("gate-")]
    if not gate:
        return allocation
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    plans = planned_rosters(
        gid._context(1),
        [int(row["seed"]) for row in gate],
        [row["stream"].removeprefix("gate-") for row in gate],
    )
    for row, plan in zip(gate, plans, strict=True):
        row["plan"] = _portable_plan(plan)
    return allocation


def _portable_plan(value: Any) -> Any:
    """Replace only checkpoint paths with input-relative runtime references."""
    if isinstance(value, dict):
        return {key: _portable_plan(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_portable_plan(child) for child in value]
    if isinstance(value, str) and value.endswith(".pth"):
        return f"input:checkpoints/{Path(value).name}"
    return value


def _pool_inputs(b8: Path, frp3: Path) -> list[dict[str, str]]:
    names = (
        "champion_a5_freespace_20260621.pth",
        "best_apex_fs.pth",
        "best_apex_stage1_fs.pth",
        "best_apex_pre_fs.pth",
    )
    pool = ROOT.parent / "snake-dqn" / "saved_snakes"
    if not all((pool / name).is_file() for name in names):
        # Structural unit tests do not carry the release catalog. Preparation
        # below refuses this partial form before it can become a pack.
        return [
            input_row(b8, "B8_s3_final.pth"),
            input_row(frp3, "checkpoints/frp3_m3_s12_u60000_20261005.pth"),
        ]
    return [
        input_row(b8, "B8_s3_final.pth"),
        *[input_row(pool / name, f"checkpoints/{name}") for name in names],
        input_row(frp3, "checkpoints/frp3_m3_s12_u60000_20261005.pth"),
    ]


def namespace(commit: str) -> str:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("F0 requires a full lowercase commit SHA")
    return f"f0-hindsight-labels-{commit}-cpu-canary"


def input_row(path: Path, dest: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"required F0 input absent: {path}")
    return {"path": str(path.resolve()), "sha256": _sha(path), "dest": dest}


def ledger_inputs(ledger: Path) -> list[dict[str, str]]:
    """Upload the ledger and every verified life it names for resume verification."""
    value = json.loads(ledger.read_text())
    paths = [ledger, Path(str(ledger) + ".sha256"), Path(str(ledger) + ".complete")]
    for row in value.get("units", []):
        payload = ledger.parent / row["path"]
        paths.extend((payload, Path(str(payload) + ".sha256"), Path(str(payload) + ".complete")))
    return [
        input_row(path, "inputs/" + path.relative_to(ledger.parent).as_posix()) for path in paths
    ]


def source_paths() -> list[str]:
    """Return the exact static import closure plus declared dynamic seams."""
    from tools.ops.import_preflight import import_closure, module_path

    roots = [
        "research.bl_labels.runtime",
        "research.bl_labels.factory",
        "research.bl_labels.replay_audit",
    ]
    modules, _statements = import_closure(ROOT, roots)
    paths = {
        module_path(ROOT, module).relative_to(ROOT).as_posix()
        for module in modules
        if module_path(ROOT, module) is not None
    }
    # These modules are runtime-selected or data-loaded, so AST imports alone
    # cannot discover them. Include their packages/configuration explicitly.
    paths.update(
        {
            PACKAGE,
            "src/simd_env/grid_sim_nb.py",
            "src/simd_env/vector61_nb.py",
            "src/simd_env/v8_nb.py",
            "configs",
            "research/colab_cli",
            "research/t34_exit_b8/PUBLIC_INPUTS.json",
            "research/apex_safety_20260926",
            "requirements.txt",
            "pyproject.toml",
        }
    )
    return sorted(paths)


def require_source_pin(commit: str, *, repo: Path = ROOT) -> None:
    """Refuse an unpublished/dirty source or a drifted qualified transport tree."""
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("F0 requires a full lowercase commit SHA")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if dirty.strip():
        raise ValueError("F0 pack preparation requires a clean pinned source tree")
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=repo, text=True
    )
    if "origin/" not in published:
        raise ValueError("F0 source pin must be published on origin")
    from tools.ops.qualification import check

    check(repo, QUALIFIED_RECORD)
    check(repo, QUALIFIED_RECORD, ref=commit)


def clean_import_preflight(commit: str, out: Path, *, repo: Path = ROOT) -> dict[str, Any]:
    """Run the declared-dependency-only sparse import gate before publishing a pack."""
    from tools.ops.import_preflight import check

    result = check(
        repo,
        commit,
        source_paths(),
        f"{PACKAGE}/requirements.txt",
        [
            "research.bl_labels.runtime",
            "research.bl_labels.factory",
            "research.bl_labels.replay_audit",
            "research.bl_upload.inputs",
        ],
    )
    publish_json(out, result)
    return result


def make_manifest(
    commit: str, output_dir: Path, control: Path, ledger: Path, b8: Path, frp3: Path
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if not output_dir.is_absolute():
        raise ValueError("F0 output directory must be absolute")
    if not QUALIFIED_RECORD.is_file():
        raise ValueError(f"qualified transport record absent: {QUALIFIED_RECORD}")
    qualified = json.loads(QUALIFIED_RECORD.read_text())
    if qualified.get("schema") != "snake-dqn-colab-cli-qualified/v1":
        raise ValueError("unrecognized qualified Colab CLI record")
    seed_audit = control.with_name("SEED_AUDIT.json")
    if not seed_audit.is_file():
        raise ValueError("F0 manifest requires the frozen seed audit")
    result = {
        "job_id": f"bl-f0-cpu-canary-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "F0 development canary only; INVALID on digest/bare-audit mismatch; no verdict",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "F0_NAMESPACE": namespace(commit),
            "F0_COMMIT": commit,
            "F0_SOURCE_SHA256": source_sha256(),
            "F0_WORKERS": "nproc",
            "F0_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "requirements": {"gpu": "cpu", "allowed_gpus": ["cpu"], "min_vcpu": 1, "est_minutes": 30},
        "timeout_minutes": 60,
        "output_dir": str(output_dir),
        "inputs": [
            input_row(control, "inputs/control.json"),
            *ledger_inputs(ledger),
            input_row(seed_audit, "inputs/SEED_AUDIT.json"),
            *_pool_inputs(b8, frp3),
        ],
        "budget": {
            "unit_cap": 0.2,
            "rate_units_per_hour": 0.08,
            "reserve_units": 0.04,
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 256 * 1024**2,
        "source_paths": source_paths(),
        "required_exports": ["runtime-result.json"],
        "qualified_transport": {
            "record": str(QUALIFIED_RECORD),
            "sha256": _sha(QUALIFIED_RECORD),
            "commit": qualified.get("qualified_commit"),
            "tree_hash": qualified.get("tree_hash"),
        },
        "resume": {
            "unit_kind": "life",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "f0": {
            "base_envelope": {"hours": 0.5, "units": 0.1},
            "effective_envelope": {"hours": 1.0, "units": 0.2},
            "gpu": 0,
            "allocation": {
                "gate": {"frozen": 12, "mixed": 12, "scripted": 8},
                "watch6": 4,
                "watch6_decisions": 14400,
                "gate_horizon": 5000,
            },
            "kill": {
                "digest_or_bare_audit_mismatch": "INVALID",
                "rescue_bearing_lives_per_mac_worker_hour_lt": 8,
            },
            "execution": "W=nproc deterministic allocation order; one thread per worker",
            "attempt_policy": (
                "One original plus at most two manually dispatched mechanical completion attempts; "
                "every attempt has a fresh output directory. The full series shares the single "
                "60-minute/0.2-unit effective envelope; no automatic retries, cap reset, or "
                "outcome-triggered rerun. Verified lives are skipped and never recomputed or "
                "relabelled."
            ),
        },
    }
    return result


def prepare(
    commit: str, artifact_dir: Path, b8: Path, frp3: Path, output_dir: Path
) -> tuple[Path, Path]:
    require_source_pin(commit)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    allocation = materialize_gate_plans(canary_allocation(namespace(commit)))
    seeds = [int(row["seed"]) for row in allocation]
    if len(seeds) != len(set(seeds)):
        raise ValueError("F0 frozen allocation has an internal seed collision")
    seed_audit = artifact_dir / "SEED_AUDIT.json"
    if not seed_audit.exists():
        from research.t3_rollout_labels_20261015.bank_engine import concrete_seed_hits

        artifacts = ROOT.parent / "snake-dqn-artifacts"
        hits = {
            str(ROOT): sorted(concrete_seed_hits(ROOT, seeds)),
            str(artifacts): sorted(concrete_seed_hits(artifacts, seeds)),
        }
        if any(hits.values()):
            raise ValueError(f"F0 fresh seed collision: {hits}")
        publish_json(
            seed_audit,
            {
                "schema": "bl-labels-f0-seed-audit/v1",
                "namespace": namespace(commit),
                "seeds": seeds,
                "hits_by_root": hits,
                "outcome_blind": True,
            },
        )
    else:
        saved_audit = json.loads(seed_audit.read_text())
        if (
            saved_audit.get("namespace") != namespace(commit)
            or saved_audit.get("seeds") != seeds
            or any(saved_audit.get("hits_by_root", {}).values())
        ):
            raise ValueError("existing F0 seed audit is foreign or records a collision")
    control = artifact_dir / "control.json"
    ledger = artifact_dir / "verified-ledger.json"
    if not control.exists():
        publish_json(
            control,
            {
                "schema": "bl-labels-f0-control/v1",
                "namespace": namespace(commit),
                "allocation": allocation,
                "workers": "nproc",
            },
        )
    else:
        saved = json.loads(control.read_text())
        if saved.get("namespace") != namespace(commit) or saved.get("allocation") != allocation:
            raise ValueError("existing F0 control differs from this frozen namespace/allocation")
    if not ledger.exists():
        publish_json(ledger, empty_ledger(namespace(commit), allocation))
    else:
        from .exports import load_ledger

        load_ledger(ledger, namespace(commit), allocation)
    preflight = artifact_dir / "clean-import-preflight.json"
    if not preflight.exists():
        clean_import_preflight(commit, preflight)
    elif json.loads(preflight.read_text()).get("commit") != commit:
        raise ValueError("existing F0 import preflight has a foreign source commit")
    manifest = make_manifest(commit, output_dir, control, ledger, b8, frp3)
    if len(manifest["inputs"]) < 10:
        raise ValueError("F0 preparation requires all six public checkpoint inputs")
    from research.bl_upload import assert_upload_budget, convert_public_checkpoints

    manifest = convert_public_checkpoints(manifest, repo=ROOT)
    assert_upload_budget(manifest, repo=ROOT)
    path = artifact_dir / "f0-cpu-canary.json"
    publish_json(path, manifest)
    return path, control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    args = parser.parse_args()
    manifest, _ = prepare(args.commit, args.artifact_dir, args.b8, args.frp3, args.output_dir)
    print(manifest)


if __name__ == "__main__":
    main()
