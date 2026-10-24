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
import io
import json
import math
import subprocess
import tarfile
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
RULER_PIN = "6ee41cdf372ed3daa80da4f05a0cdc44ddcaab7d"
COMPLETION_RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "fable-20261024/JUDGMENT_STEP_M1_COMPLETION.md"
)
COMPLETION_RULING_SHA256 = "1d0d4a98d12a0109077b4f2870b413e740e3667acc013b7896669082a936e5dd"
L4_RATE = 2.0
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
    if value.get("completion_ruling_sha256") == COMPLETION_RULING_SHA256:
        ledger = ledger.with_name("probe-ledger.json")
        job.verify_unit(ledger)
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


def check_collector_closure() -> dict[str, Any]:
    """Refuse any collector/ticks import drift from the original ruler pin."""
    from tools.ops.import_preflight import import_closure, module_path

    modules, _ = import_closure(
        ROOT, ["research.bl_step_ruler.job", "research.bl_step_ruler.ticks"]
    )
    paths = sorted({module_path(ROOT, name).relative_to(ROOT).as_posix() for name in modules})
    archive = subprocess.check_output(
        ["git", "archive", RULER_PIN, "--", *paths], cwd=ROOT, timeout=60
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for path in paths:
            reference = stream.extractfile(path)
            if reference is None or reference.read() != (ROOT / path).read_bytes():
                raise ValueError(f"collector/ticks closure differs from ruler pin: {path}")
    return {"pin": RULER_PIN, "files": len(paths), "status": "PASS"}


def completion_options(
    attempt: int, runtime_type: str, timeout_minutes: float | None, unit_guard: float | None
) -> bool:
    """Validate prospective completion envelopes before creating any artifacts."""
    if runtime_type not in {"cpu", "l4"}:
        raise ValueError("runtime_type must be cpu or l4")
    amended = runtime_type != "cpu" or timeout_minutes is not None or unit_guard is not None
    if not amended:
        return False
    if attempt not in (2, 3):
        raise ValueError("runtime/envelope amendments are completion-attempt only")
    timeout = 260 if timeout_minutes is None else timeout_minutes
    guard = 12 if unit_guard is None else unit_guard
    if not math.isfinite(timeout) or not 10 < timeout <= 260:
        raise ValueError("completion timeout must be finite, above 10 and at most 260 minutes")
    if not math.isfinite(guard) or not 0 < guard <= 12:
        raise ValueError("completion unit guard must be finite and within (0, 12]")
    rate = L4_RATE if runtime_type == "l4" else CPU_RATE
    if timeout * rate / 60 > guard:
        raise ValueError("completion unit guard does not cover the whole-session timeout")
    return True


def apply_completion(
    manifest: dict[str, Any], runtime_type: str, timeout_minutes: float, unit_guard: float
) -> None:
    """Bind the completion's own envelope; science still executes on CPU."""
    if runtime_type == "l4":
        manifest["requirements"].update(gpu="L4", allowed_gpus=["L4"], min_vcpu=12)
        manifest["budget"]["rate_units_per_hour"] = L4_RATE
    manifest["timeout_minutes"] = timeout_minutes
    manifest["requirements"]["est_minutes"] = min(130, timeout_minutes / 2)
    manifest["budget"].update(unit_cap=unit_guard, reserve_units=unit_guard / 5)
    manifest["env"]["F0_SESSION_SECONDS"] = str(int(timeout_minutes * 60))
    manifest["step_ruler"].pop("series_effective_units")
    manifest["step_ruler"].pop("series_guard_units")
    manifest["step_ruler"]["stages"]["main"].update(
        base_minutes=timeout_minutes / 2,
        effective_minutes=timeout_minutes,
        base_units=unit_guard / 2,
        effective_units=unit_guard,
        base_reserve_units=unit_guard / 10,
        effective_reserve_units=unit_guard / 5,
    )
    manifest["step_ruler"].update(
        runtime_type=runtime_type,
        completion_ruling=str(COMPLETION_RULING),
        completion_ruling_sha256=COMPLETION_RULING_SHA256,
        admission_note=f"Completion amendment sha256:{COMPLETION_RULING_SHA256}",
        completion_envelope={
            "base_minutes": timeout_minutes / 2,
            "effective_minutes": timeout_minutes,
            "effective_units": unit_guard,
            "owner_multiplier": 2,
            "scope": "OWN_ATTEMPT_NOT_SHARED",
        },
        completion_unit_guard=unit_guard,
        attempt_policy=(
            "One original plus at most two manual outcome-blind completions; "
            "attempts 2 and 3 each have their own 130/260-minute envelope. "
            "Verified stop and missing units required. No automatic retry."
        ),
        worker_policy="nproc (L4 minimum 12), capped by pending parents; one thread per worker",
        rate_use=("DESCRIPTIVE_ONLY_PLATFORM_CHANGED" if runtime_type == "l4" else "CPU"),
    )


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
    probe["job_id"] = main["job_id"].replace("bl-step-ruler-main-", "bl-step-ruler-probe-")
    probe["output_dir"] = str(output_dir)
    apply_stage(probe, "transport_probe")
    if main["step_ruler"].get("runtime_type") == "l4":
        # Same 10/20-minute envelope, priced at the L4 family's 2 u/h ceiling.
        cap = L4_RATE * probe["timeout_minutes"] / 60
        probe["budget"].update(unit_cap=cap, reserve_units=cap * 0.2)
        probe["step_ruler"]["stages"]["transport_probe"].update(
            base_units=cap / 2,
            effective_units=cap,
            base_reserve_units=cap * 0.1,
            effective_reserve_units=cap * 0.2,
        )
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
    commit: str,
    artifact_dir: Path,
    output_dir: Path,
    b8: Path,
    frp3: Path,
    *,
    ledger: Path | None = None,
    attempt: int = 1,
    runtime_type: str = "cpu",
    timeout_minutes: float | None = None,
    unit_guard: float | None = None,
) -> dict[str, Path]:
    """Create-only pack. Never dispatches, allocates or touches the Mac factory.

    ``attempt`` 1 ships an empty ledger. A completion attempt (2 or 3, i.e. at
    most two outcome-blind completions) ships the Mac-verified ledger union so
    verified parents are skipped, never recomputed.
    """
    if attempt not in (1, 2, 3) or (attempt == 1) != (ledger is None):
        raise ValueError("attempt 1 uses an empty ledger; attempts 2-3 need a verified ledger")
    amended = completion_options(attempt, runtime_type, timeout_minutes, unit_guard)
    require_source_pin(commit)
    if _sha(JUDGMENT) != JUDGMENT_SHA256:
        raise ValueError("authority judgment bytes differ")
    if amended and _sha(COMPLETION_RULING) != COMPLETION_RULING_SHA256:
        raise ValueError("completion ruling bytes differ")
    closure = check_collector_closure()
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    if artifact_dir.exists():
        raise FileExistsError(artifact_dir)
    if output_dir.exists() or output_dir.with_name(output_dir.name + "-probe").exists():
        raise FileExistsError(output_dir)
    ns = namespace(RULER_PIN if ledger is not None else commit)
    rows = materialize_gate_plans(allocation(ns))
    seeds = [int(row["seed"]) for row in rows] + [self_check_seed(ns)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("internal dev seed collision")
    if ledger is None:
        audit = seed_audit(ns, seeds)
    else:
        job.verify_unit(ledger)
        done = job.load_ledger(ledger, ns, rows)
        if not done or len(done) == len(rows):
            raise ValueError("a completion attempt needs some, but not all, parents verified")
        audit = {
            "schema": "bl-step-ruler-resume-allocation/v1",
            "namespace": ns,
            "seeds": seeds,
            "allocation_sha256": job.allocation_sha256(rows),
            "verified_ledger_sha256": _sha(ledger),
            "original_seed_audit": "original frozen allocation; no new seeds or seed scan",
            "excluded_unopened": [ROUND0_EXCLUDED],
            "outcome_blind": True,
        }
    source = artifact_dir / "inputs"
    publish_json(source / "SEED_AUDIT.json", audit)
    control = {"schema": job.CONTROL_SCHEMA, "namespace": ns, "allocation": rows}
    if amended:
        control.update(
            runtime_type=runtime_type,
            runtime_commit=commit,
            attempt=attempt,
            collector_pin=RULER_PIN,
            completion_ruling_sha256=COMPLETION_RULING_SHA256,
        )
        publish_json(source / "probe-ledger.json", job.empty_ledger(ns, rows))
    publish_json(source / "control.json", control)
    if ledger is None:
        publish_json(source / "verified-ledger.json", job.empty_ledger(ns, rows))
    else:
        job.verify_unit(ledger)
        done = job.load_ledger(ledger, ns, rows)
        if not done or len(done) == len(rows):
            raise ValueError("a completion attempt needs some, but not all, parents verified")
        for suffix in ("", ".sha256", ".complete"):
            target = source / ("verified-ledger.json" + suffix)
            with target.open("xb") as stream:
                stream.write(Path(str(ledger) + suffix).read_bytes())
        job.verify_unit(source / "verified-ledger.json")
    preflight_record = artifact_dir / "clean-import-preflight.json"
    from tools.ops.import_preflight import check as import_check

    publish_json(
        preflight_record,
        import_check(ROOT, commit, source_paths(), REQUIREMENTS, list(RUNTIME_ROOTS)),
    )
    local = []
    names = ["control.json", "verified-ledger.json", "SEED_AUDIT.json"]
    if amended:
        names.append("probe-ledger.json")
    for name in names:
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
    main["job_id"] += f"-a{attempt}"
    main["step_ruler"]["attempt"] = attempt
    if amended:
        apply_completion(
            main,
            runtime_type,
            260 if timeout_minutes is None else timeout_minutes,
            12 if unit_guard is None else unit_guard,
        )
        main["step_ruler"]["collector_closure"] = closure
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
    p.add_argument("--ledger", type=Path, help="Mac-verified ledger (completion attempts only)")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--runtime-type", choices=("cpu", "l4"), default="cpu")
    p.add_argument("--timeout-minutes", type=float, help="effective completion envelope (max 260)")
    p.add_argument("--unit-guard", type=float, help="completion runaway guard (max 12 units)")
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
        paths = prepare(
            args.commit,
            args.artifact_dir,
            args.output_dir,
            args.b8,
            args.frp3,
            ledger=args.ledger,
            attempt=args.attempt,
            runtime_type=args.runtime_type,
            timeout_minutes=args.timeout_minutes,
            unit_guard=args.unit_guard,
        )
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
