"""Prepare, but never dispatch, the Step -1 world-step ruler pack (CPU runtime).

Subcommands:
  prepare        create-only control, empty ledger, seed audit, metadata tar,
                 excluded family-probe (canary) manifest and main template
  bind           bind a verified family-probe receipt into the main manifest
  probe-control  (runtime) derive the excluded probe control from the upload
  ledger         Mac-side verified-unit ledger (path/size/hash only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_labels.exports import publish_json
from research.bl_labels.manifests import (
    QUALIFIED_RECORD,
    input_row,
    materialize_gate_plans,
    require_source_pin,
)

from . import job

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/bl_step_ruler"
SCRIPT = f"{PACKAGE}/colab.sh"
PROTOCOL = f"{PACKAGE}/PROTOCOL.md"
JUDGMENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_GPU_LABEL_PATH_20261024.md"
)
JUDGMENT_SHA256 = "ef488af238e1f60818dc5baea9fbc17001ea9fe09c1590867f3ce00cc08f4c00"
ROUND0_EXCLUDED = "t34-round0-study-01"
SAVED_SNAKES = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
POOL = (
    "champion_a5_freespace_20260621.pth",
    "best_apex_fs.pth",
    "best_apex_stage1_fs.pth",
    "best_apex_pre_fs.pth",
)
# Watch6 first so the long parents start before the short gate parents.
ALLOCATION = (
    ("watch6", 4, 14400),
    ("gate-frozen", 3, 5000),
    ("gate-mixed", 3, 5000),
    ("gate-scripted", 2, 5000),
)
SELF_CHECK_INDEX = 1_000_000  # dev seed index outside the allocation range
RUNTIME_ROOTS = (
    "research.bl_step_ruler.job",
    "research.bl_step_ruler.pack",
    "research.bl_step_ruler.ticks",
    "research.bl_labels.factory",
    "research.bl_labels.runtime",
    "research.bl_upload.inputs",
    "research.metadata_archive",
    "src.simd_env.grid_sim_nb",
    "src.simd_env.vector61_nb",
    "src.simd_env.v8_nb",
)
REQUIREMENTS = "research/bl_labels/requirements.txt"
CPU_RATE = 0.08
# Whole-session base forecasts; the owner's 2x applies once to caps/timeouts.
STAGES = {
    "transport_probe": {
        "setup_seconds": 300,
        "work_seconds": 180,
        "export_seconds": 60,
        "margin_seconds": 60,
        "base_units": 0.02,
        "base_reserve_units": 0.004,
    },
    "main": {
        "setup_seconds": 300,
        "work_seconds": 7200,
        "export_seconds": 120,
        "margin_seconds": 180,
        "base_units": 0.18,
        "base_reserve_units": 0.036,
    },
}
SERIES_GUARD_UNITS = 1.0
RESUME = {
    "unit_kind": "parent",
    "ledger_path": "inputs/verified-ledger.json",
    "skip_verified": True,
    "max_outcome_blind_retries": 2,
}


def namespace(commit: str) -> str:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("step ruler requires a full lowercase commit SHA")
    return f"bl-step-ruler-{commit}-dev"


def _seed(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")


def allocation(ns: str) -> list[dict[str, Any]]:
    rows = []
    for stream, count, horizon in ALLOCATION:
        for index in range(count):
            rows.append(
                {
                    "allocation_id": f"{stream}-{index:03d}",
                    "seed": _seed(f"{ns}/{stream}/{index}"),
                    "stream": stream,
                    "horizon": horizon,
                }
            )
    return rows


def self_check_seed(ns: str) -> int:
    return _seed(f"{ns}/self-check/{SELF_CHECK_INDEX}")


def stage_envelope(stage: str) -> dict[str, Any]:
    base = STAGES[stage]
    seconds = sum(
        base[key] for key in ("setup_seconds", "work_seconds", "export_seconds", "margin_seconds")
    )
    return {
        **base,
        "base_minutes": seconds / 60,
        "effective_minutes": 2 * seconds / 60,
        "effective_units": 2 * base["base_units"],
        "effective_reserve_units": 2 * base["base_reserve_units"],
        "effective_finalize_seconds": 2 * (base["export_seconds"] + base["margin_seconds"]),
        "owner_multiplier": 2,
    }


def series_effective_units() -> float:
    return sum(stage_envelope(stage)["effective_units"] for stage in STAGES)


def apply_stage(manifest: dict[str, Any], stage: str) -> None:
    envelope = stage_envelope(stage)
    manifest["timeout_minutes"] = envelope["effective_minutes"]
    manifest["requirements"]["est_minutes"] = envelope["base_minutes"]
    manifest["budget"].update(
        unit_cap=envelope["effective_units"], reserve_units=envelope["effective_reserve_units"]
    )
    manifest["env"].update(
        F0_SESSION_SECONDS=str(int(envelope["effective_minutes"] * 60)),
        F0_FINALIZE_SECONDS=str(envelope["effective_finalize_seconds"]),
    )


def probe_control(control: Path, ledger: Path, out_control: Path, out_ledger: Path) -> None:
    """Runtime: excluded probe namespace, first parent per stream at horizon 3."""
    value = json.loads(control.read_text())
    if job.load_ledger(ledger, value["namespace"], value["allocation"]):
        raise ValueError("step-ruler probe requires the original uploaded ledger to be empty")
    seen: set[str] = set()
    rows = []
    for item in value["allocation"]:
        if item["stream"] in seen:
            continue
        seen.add(item["stream"])
        rows.append({**item, "horizon": 3})
    probe = {**value, "namespace": value["namespace"] + "-transport-probe", "allocation": rows}
    publish_json(out_control, probe)
    publish_json(out_ledger, job.empty_ledger(probe["namespace"], rows))


def pool_inputs(b8: Path, frp3: Path) -> list[dict[str, str]]:
    """The six registered public checkpoint identities, from the local catalogue."""
    return [
        input_row(b8, "B8_s3_final.pth"),
        *[input_row(SAVED_SNAKES / name, f"checkpoints/{name}") for name in POOL],
        input_row(frp3, "checkpoints/frp3_m3_s12_u60000_20261005.pth"),
    ]


def seed_audit(ns: str, seeds: list[int]) -> dict[str, Any]:
    """Literal whole-word scan for every seed; Round 0 study is excluded unopened."""
    roots = (ROOT, ARTIFACTS)
    hits: dict[str, list[str]] = {}
    argv: list[list[str]] = []
    for root in roots:
        command = ["rg", "-I", "-l", "-F", "-w", "--no-config", "--glob", f"!{ROUND0_EXCLUDED}/**"]
        command.extend(item for seed in seeds for item in ("-e", str(seed)))
        command.extend(("--", str(root)))
        result = subprocess.run(command, text=True, capture_output=True, timeout=900, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"seed scan failed under {root}: {result.stderr.strip()}")
        hits[str(root)] = sorted(line for line in result.stdout.splitlines() if line)
        argv.append(command)
    if any(hits.values()):
        raise ValueError(f"step-ruler dev seed collision: {hits}")
    return {
        "schema": "bl-step-ruler-seed-audit/v1",
        "namespace": ns,
        "seeds": seeds,
        "hits_by_root": hits,
        "excluded_unopened": [ROUND0_EXCLUDED],
        "argv_by_root": argv,
        "outcome_blind": True,
    }


def source_paths() -> list[str]:
    from tools.ops.import_preflight import import_closure, module_path

    modules, _statements = import_closure(ROOT, list(RUNTIME_ROOTS))
    paths = {
        module_path(ROOT, module).relative_to(ROOT).as_posix()
        for module in modules
        if module_path(ROOT, module) is not None
    }
    paths.update(
        {
            PACKAGE,
            "research/bl_labels",
            "src/simd_env/grid_sim_nb.py",
            "src/simd_env/vector61_nb.py",
            "src/simd_env/v8_nb.py",
            "configs",
            "research/colab_cli",
            "research/bl_upload",
            "research/metadata_archive.py",
            "research/t34_exit_b8/PUBLIC_INPUTS.json",
            "research/apex_safety_20260926",
            "requirements.txt",
            "pyproject.toml",
        }
    )
    return sorted(paths)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_main(
    commit: str, output_dir: Path, inputs: list[dict[str, Any]], archive_sha: str, ns: str
) -> dict[str, Any]:
    qualified = json.loads(QUALIFIED_RECORD.read_text())
    manifest: dict[str, Any] = {
        "job_id": f"bl-step-ruler-main-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": (
            "Step -1 world-step ruler: dev-namespace F0 work/timing counters only; "
            "no labels, outcomes, gate, fit or serving claim"
        ),
        "env": {
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS,
            "PYTHONUNBUFFERED": "1",
            "F0_COMMIT": commit,
            "STEP_RULER_NAMESPACE": ns,
            "STEP_RULER_SOURCE_SHA256": job.source_sha256(ROOT),
            "METADATA_ARCHIVE_SHA256": archive_sha,
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "requirements": {"gpu": "cpu", "allowed_gpus": ["cpu"], "min_vcpu": 1, "est_minutes": 0},
        "timeout_minutes": 0,
        "output_dir": str(output_dir),
        "inputs": inputs,
        "budget": {
            "unit_cap": 0,
            "rate_units_per_hour": CPU_RATE,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 64 * 1024**2,
        "source_paths": source_paths(),
        "required_exports": ["runtime-result.json"],
        "qualified_transport": {
            "record": str(QUALIFIED_RECORD),
            "sha256": _sha(QUALIFIED_RECORD),
            "commit": qualified.get("qualified_commit"),
            "tree_hash": qualified.get("tree_hash"),
        },
        "resume": dict(RESUME),
        "canary": {"template": True, "family": SCRIPT},
        "step_ruler": {
            "judgment": str(JUDGMENT),
            "judgment_sha256": JUDGMENT_SHA256,
            "protocol": PROTOCOL,
            "protocol_sha256": _sha(ROOT / PROTOCOL),
            "stages": {stage: stage_envelope(stage) for stage in STAGES},
            "series_effective_units": series_effective_units(),
            "series_guard_units": SERIES_GUARD_UNITS,
            "gpu": 0,
            "allocation": {stream: count for stream, count, _h in ALLOCATION},
            "attempt_policy": (
                "One original plus at most two manually dispatched, outcome-blind completion "
                "attempts after a verified stop and missing units; each attempt uses a fresh "
                "manifest/output and the Mac-verified ledger; all attempts share the main "
                "effective envelope. No automatic retry or cap reset."
            ),
            "never": [
                "leased Mac execution",
                "exp/mac-lease.json*",
                "t34-round0-study-01",
                "label rows/outcomes/rescue counts in exports",
            ],
        },
    }
    apply_stage(manifest, "main")
    return manifest


def make_probe(main: dict[str, Any], output_dir: Path, ns: str) -> dict[str, Any]:
    probe = json.loads(json.dumps(main))
    probe["job_id"] = f"bl-step-ruler-probe-{main['commit'][:8]}"
    probe["output_dir"] = str(output_dir)
    apply_stage(probe, "transport_probe")
    probe["env"].update(
        STEP_RULER_TRANSPORT_PROBE="1",
        STEP_RULER_SELF_CHECK_SEED=str(self_check_seed(ns)),
        SNAKE_COLAB_CANARY="1",
        SNAKE_COLAB_CANARY_PACKAGE=PACKAGE,
    )
    probe["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    probe.pop("resume")
    probe["resume_exemption"] = {
        "kind": "short-canary",
        "reason": "excluded step-ruler same-entrypoint transport probe; never enters a ledger",
        "timeout_minutes": probe["timeout_minutes"],
    }
    probe["step_ruler"]["stage"] = "EXCLUDED_TRANSPORT_PROBE"
    return probe


def prepare(
    commit: str, artifact_dir: Path, output_dir: Path, b8: Path, frp3: Path
) -> dict[str, Path]:
    """Create-only pack. Never dispatches, allocates or touches the Mac factory."""
    require_source_pin(commit)
    if _sha(JUDGMENT) != JUDGMENT_SHA256:
        raise ValueError("authority judgment bytes differ")
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    if artifact_dir.exists():
        raise FileExistsError(artifact_dir)
    if output_dir.exists() or output_dir.with_name(output_dir.name + "-probe").exists():
        raise FileExistsError(output_dir)
    ns = namespace(commit)
    rows = materialize_gate_plans(allocation(ns))
    seeds = [int(row["seed"]) for row in rows] + [self_check_seed(ns)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("internal dev seed collision")
    audit = seed_audit(ns, seeds)
    source = artifact_dir / "inputs"
    publish_json(source / "SEED_AUDIT.json", audit)
    control = {"schema": job.CONTROL_SCHEMA, "namespace": ns, "allocation": rows}
    publish_json(source / "control.json", control)
    publish_json(source / "verified-ledger.json", job.empty_ledger(ns, rows))
    preflight_record = artifact_dir / "clean-import-preflight.json"
    from tools.ops.import_preflight import check as import_check

    publish_json(
        preflight_record,
        import_check(ROOT, commit, source_paths(), REQUIREMENTS, list(RUNTIME_ROOTS)),
    )
    local = []
    for name in ("control.json", "verified-ledger.json", "SEED_AUDIT.json"):
        for suffix in ("", ".sha256", ".complete"):
            local.append(input_row(source / (name + suffix), f"inputs/{name}{suffix}"))
    local.append(input_row(ROOT / PROTOCOL, "inputs/PROTOCOL.md"))
    from research.bl_upload import assert_upload_budget, convert_public_checkpoints
    from research.metadata_archive import bundle_inputs

    draft = {"commit": commit, "inputs": [*local, *pool_inputs(b8, frp3)], "env": {}}
    draft["source_paths"] = source_paths()
    converted = convert_public_checkpoints(draft, repo=ROOT)
    if converted["env"].get("BL_PUBLIC_INPUTS") != "1":
        raise ValueError("all six public checkpoint identities are required")
    # Keep the pack independent of the preparing checkout: a byte-identical
    # copy of the registered public catalogue lives beside the pack.
    for row in converted["inputs"]:
        if row["dest"] == "public-inputs.json":
            copy = artifact_dir / "public-inputs.json"
            copy.write_bytes(Path(row["path"]).read_bytes())
            if _sha(copy) != row["sha256"]:
                raise ValueError("public catalogue copy differs")
            row["path"] = str(copy)
    inputs = bundle_inputs(converted["inputs"], artifact_dir / "metadata.tar")
    archive_sha = next(row["sha256"] for row in inputs if row["dest"] == "metadata.tar")
    main = make_main(commit, output_dir, inputs, archive_sha, ns)
    main["env"].update(
        BL_PUBLIC_INPUTS="1", BL_PUBLIC_INPUT_ALIASES=converted["env"]["BL_PUBLIC_INPUT_ALIASES"]
    )
    main["source_paths"] = list(dict.fromkeys([*main["source_paths"], *converted["source_paths"]]))
    assert_upload_budget(main, repo=ROOT)
    probe = make_probe(main, output_dir.with_name(output_dir.name + "-probe"), ns)
    from research.colab_cli.supervisor import validate

    validate(json.loads(json.dumps(probe)))
    paths = {
        "main_template": artifact_dir / "step-ruler-main-template.json",
        "probe": artifact_dir / "step-ruler-probe.json",
        "control": source / "control.json",
        "metadata": artifact_dir / "metadata.tar",
        "import_preflight": preflight_record,
    }
    publish_json(paths["main_template"], main)
    publish_json(paths["probe"], probe)
    return paths


def bind(template: Path, receipt: Path, out: Path) -> dict[str, Any]:
    """Bind only a verified, qualified family-probe receipt into a main manifest."""
    value = json.loads(template.read_text())
    if value.get("canary", {}).get("template") is not True:
        raise ValueError("step-ruler main template is malformed")
    value["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": _sha(receipt)}}
    from research.colab_cli.canary import check

    check(value, ROOT)
    publish_json(out, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--commit", required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--b8", type=Path, required=True)
    p.add_argument("--frp3", type=Path, required=True)
    b = sub.add_parser("bind")
    b.add_argument("--template", type=Path, required=True)
    b.add_argument("--receipt", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("probe-control")
    c.add_argument("--control", type=Path, required=True)
    c.add_argument("--ledger", type=Path, required=True)
    c.add_argument("--out-control", type=Path, required=True)
    c.add_argument("--out-ledger", type=Path, required=True)
    g = sub.add_parser("ledger")
    g.add_argument("--control", type=Path, required=True)
    g.add_argument("--exports", type=Path, action="append", required=True)
    g.add_argument("--prior", type=Path)
    g.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        paths = prepare(args.commit, args.artifact_dir, args.output_dir, args.b8, args.frp3)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "bind":
        bind(args.template, args.receipt, args.out)
        print(args.out)
    elif args.command == "probe-control":
        probe_control(args.control, args.ledger, args.out_control, args.out_ledger)
    else:
        control = json.loads(args.control.read_text())
        value = job.build_ledger(control, args.exports, args.out, args.prior)
        print(json.dumps({"units": len(value["units"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
