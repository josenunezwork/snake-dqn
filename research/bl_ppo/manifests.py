"""Prepare, but never dispatch, the BL E4/E2 Colab supervisor manifests.

The E2 manifests deliberately cannot be created until a complete, hash-bound
E4 receipt exists.  This keeps the throughput-dependent training length from
becoming an outcome-dependent runtime choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/bl_ppo"
SCRIPT = f"{PACKAGE}/colab.sh"
DEFAULT_ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2")
L4_RATE = 2.0
CRITIC_REFIT_TRANSITIONS = 2_000_000
# 128 worlds, 26 held out: ceil(2M/(102*128))*128*128 + 128*5000
# is 3,163,136 collected rows. Round up; refit and holdout are charged.
CRITIC_COLLECTION_FORECAST = 3_170_000
ORDINARY_BANK_TRANSITIONS = 10_000
E4_EXPORT_BYTES = 64 * 1024**2
E2_EXPORT_BYTES = 2 * 1024**3
ROSTER_DESTINATIONS = (
    "checkpoints/champion_a5_freespace_20260621.pth",
    "checkpoints/best_apex_fs.pth",
    "checkpoints/best_apex_stage1_fs.pth",
    "checkpoints/best_apex_pre_fs.pth",
    "checkpoints/frp3_m3_s12_u60000_20261005.pth",
)


class ManifestRefused(ValueError):
    """A manifest is not sufficiently bound to become a supervisor job."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_head(repo: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _commit(value: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")
    return value


def input_row(path: Path, dest: str) -> dict[str, str]:
    source = path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ManifestRefused(f"input is not a regular file: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "dest": dest}


def _complete_payload(path: Path) -> dict[str, Any]:
    """Verify the supervisor's immutable payload/sidecar/marker convention."""
    payload = path.expanduser().resolve(strict=True)
    digest = payload.with_name(payload.name + ".sha256")
    complete = payload.with_name(payload.name + ".complete")
    if not digest.is_file() or not complete.is_file():
        raise ManifestRefused("E4 receipt must have .sha256 and .complete sidecars")
    if complete.read_text(encoding="utf-8").strip() != "complete":
        raise ManifestRefused("E4 completion marker is not complete")
    expected = f"{sha256_file(payload)}  {payload.name}"
    if digest.read_text(encoding="utf-8").strip() != expected:
        raise ManifestRefused("E4 receipt SHA256 sidecar differs")
    try:
        row = json.loads(payload.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestRefused("E4 receipt is not JSON") from exc
    if not isinstance(row, dict):
        raise ManifestRefused("E4 receipt must be a JSON object")
    return row


def e4_rate(receipt: Path) -> float:
    """Return E4's allocated hero rate only after validating its complete receipt."""
    row = _complete_payload(receipt)
    if row.get("schema") != "bl-ppo-e4-ruler/v1" or row.get("status") != "COMPLETE":
        raise ManifestRefused("E4 receipt is not a complete BL E4 ruler result")
    if row.get("scientific") is not True or row.get("worlds") != 128:
        raise ManifestRefused("E2 requires the scientific 128-world E4 ruler, not a tiny preflight")
    if row.get("long_bodies") is not True:
        raise ManifestRefused("E2 requires E4 long-body fixtures")
    configs = row.get("configs")
    if not isinstance(configs, list) or {
        item.get("processes") for item in configs if isinstance(item, dict)
    } != {6, 12}:
        raise ManifestRefused("E4 receipt must report exactly 6 and 12 collector processes")
    rates = [item.get("allocated_hero_transitions_per_second") for item in configs]
    if any(
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
        for rate in rates
    ):
        raise ManifestRefused("E4 receipt has no positive allocated hero rates")
    selected_processes = row.get("selected_processes")
    if selected_processes not in {6, 12}:
        raise ManifestRefused("E4 receipt has no selected collector configuration")
    rate = next(
        item["allocated_hero_transitions_per_second"]
        for item in configs
        if item["processes"] == selected_processes
    )
    return float(rate)


def e2_transitions(receipt: Path, *, request_scale: bool = False) -> int:
    rate = e4_rate(receipt)
    if rate < 1000:
        return 4_000_000
    if request_scale and rate >= 4000:
        return 20_000_000
    return 8_000_000


def _budget(minutes: int, unit_cap: float) -> dict[str, Any]:
    # 20 percent of the cap remains unavailable to an allocation.  unit_cap is
    # the whole-session first cap, therefore timeout pricing must not exceed it.
    return {
        "unit_cap": unit_cap,
        "rate_units_per_hour": L4_RATE,
        "reserve_units": round(unit_cap * 0.20, 2),
        "other_assignments": {},
    }


def _bounded_source_paths(manifest: dict[str, Any]) -> list[str]:
    """Select executable closure and small source data, excluding historical dumps."""
    prefixes = [p for p in manifest["source_paths"] if p != "docs/research"]
    raw = subprocess.check_output(
        ["git", "ls-tree", "-r", "-l", manifest["commit"]], cwd=ROOT, text=True
    )
    selected = []
    for line in raw.splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, _oid, size = meta.split()
        if kind != "blob" or mode == "120000":
            continue
        if not any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes):
            continue
        if (
            path.startswith(("src/", "configs/", PACKAGE + "/", "research/colab_cli/"))
            or path in {"requirements.txt", "pyproject.toml"}
            or path.endswith((".py", ".yaml", ".yml", ".sh"))
            or path.endswith(".json")
            and int(size) <= 262144
        ):
            selected.append(path)
    return sorted(selected)


def _base(
    *, job_id: str, commit: str, output: Path, estimate: int, timeout: int, cap: float
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "commit": _commit(commit),
        "script": SCRIPT,
        "claim": "DEVELOPMENT_EXCLUDED_FROM_EVERY_REGISTERED_VERDICT_NOT_DISPATCHED",
        "transport": {"mode": "local_supervisor_upload", "dispatch_ready": False},
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output.resolve()),
        "budget": _budget(timeout, cap),
        "lease_seconds": 600,
        # Explicit package closure; ``research`` itself is rejected by the
        # sparse-source safety policy as an unbounded evidence-tree upload.
        "source_paths": [
            "src",
            "web/backend",
            "configs",
            "docs/research",
            "research/apex_safety_20260926",
            "research/fixed_search.py",
            PACKAGE,
            "research/colab_cli",
            "research/compute",
            "research/phase0_search_20261019",
            "research/b8_death_census_20261014",
            "research/redesign_m2_20261008",
            "research/redesign_scope_20261007",
            "research/redesign_t7_phase_r_20261016",
            "research/s1_mature_20261018/__init__.py",
            "research/s1_mature_20261018/admission.py",
            "research/s1_mature_20261018/analyze.py",
            "research/s1_mature_20261018/audit.py",
            "research/s1_mature_20261018/bank.py",
            "research/s1_mature_20261018/checkpoint_sizing.py",
            "research/s1_mature_20261018/evaluate.py",
            "research/s1_mature_20261018/final_artifacts.py",
            "research/s1_mature_20261018/ledger.py",
            "research/s1_mature_20261018/manifests.py",
            "research/s1_mature_20261018/price.py",
            "research/s1_mature_20261018/rerun.py",
            "research/s1_mature_20261018/review_jobs.py",
            "research/s1_mature_20261018/run.py",
            "research/s1_mature_20261018/train.py",
            "research/t3_screen_20261015",
            "research/t4e_return_20261016",
            "research/t4f_return_20261016",
            "research/t1_commitment_20261015",
            "research/t34_exit_b8",
            "research/t34_claim_a",
            "research/t34_exit_b8_fit",
            "research/t34_pathb",
            "research/t34_round0",
            "research/t34_round0_amend",
            "research/t34_round0d",
            "research/t34_shadow",
            "research/t34_speed_s2",
            "research/t3_rollout_labels_20261015",
            "requirements.txt",
            "pyproject.toml",
        ],
    }


def _roster_inputs(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    if len(paths) != len(ROSTER_DESTINATIONS):
        raise ManifestRefused("E4/E2 requires the five frozen vector-roster checkpoints")
    return [input_row(path, dest) for path, dest in zip(paths, ROSTER_DESTINATIONS)]


def _qualify_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    """Move registered checkpoints off ContentsClient and refuse oversized drafts."""
    from research.bl_ppo.upload_readiness import assert_upload_budget
    from research.bl_upload.qualify import convert_public_checkpoints

    manifest = convert_public_checkpoints(manifest, repo=ROOT)
    assert_upload_budget(manifest, repo=ROOT)
    return manifest


def make_e4_manifest(
    *, commit: str, output_dir: Path, b8_checkpoint: Path, roster_checkpoints: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """Create the sole 30-minute L4 E4 ruler job."""
    # The 30-minute allocation cap includes bootstrap/export.  Fifteen minutes
    # is the deliberately conservative timed-ruler estimate required by the
    # supervisor's 2x estimate invariant.
    manifest = _base(
        job_id=f"bl-e4-ruler-{commit[:8]}",
        commit=commit,
        output=output_dir,
        estimate=15,
        timeout=30,
        cap=2.0,
    )
    manifest.update(
        {
            "env": {
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "BL_PPO_MODE": "ruler",
                "BL_PPO_DEVICE": "cuda",
                "BL_PPO_WORLDS": "128",
                "BL_PPO_PROCESS_COUNTS": "6,12",
                "BL_PPO_LONG_BODY": "1",
                "BL_PPO_FROZEN_ROSTER": "1",
                "BL_PPO_NATIVE_V8": "auto",
                "BL_PPO_MAX_SECONDS": "1800",
                "BL_PPO_CHECKPOINT_EVERY": "0",
                "BL_PPO_COMMIT": commit,
                "BL_PPO_RATE_UNITS_PER_HOUR": str(L4_RATE),
                "BL_PPO_UNIT_CAP": "2.0",
                "BL_PPO_WARMUP_STEPS": "128",
                "BL_PPO_MEASURED_STEPS": "2048",
            },
            "inputs": [
                input_row(b8_checkpoint, "B8_s3_final.pth"),
                *_roster_inputs(roster_checkpoints),
            ],
            "max_export_bytes": E4_EXPORT_BYTES,
            "bl_ppo": {
                "experiment": "E4",
                "hero_rows_only": True,
                "report_rates": ["active", "allocated"],
                "native_v8": "S2-if-on-main-else-stock",
            },
        }
    )
    manifest["source_paths"] = _bounded_source_paths(manifest)
    return _qualify_inputs(manifest)


def make_e2_manifests(
    *,
    commit: str,
    output_root: Path,
    b8_checkpoint: Path,
    e4_receipt: Path,
    roster_checkpoints: tuple[Path, ...] = (),
    request_scale: bool = False,
    e9_receipt: Path | None = None,
) -> list[dict[str, Any]]:
    """Create R/F training drafts only after immutable E4 evidence is supplied."""
    transitions = e2_transitions(e4_receipt, request_scale=request_scale)
    if e9_receipt is None:
        raise ManifestRefused("E2 materialization requires an admitted immutable E9 receipt")
    from research.bl_ppo.oc import validate_admission

    try:
        e9_report = _complete_payload(e9_receipt)
        validate_admission(e9_report)
    except ValueError as exc:
        raise ManifestRefused(f"E9 blocks E2 materialization: {exc}") from exc
    # Both arms are independently bounded at 4.5 L4 hours/9 units.  The
    # separately reserved one-hour/two-unit screen makes the combined envelope
    # exactly 10 L4 h/20 units before the 20% per-job reserve.
    if request_scale and transitions != 20_000_000:
        raise ManifestRefused("20M requires a measured E4 rate >=4000 hero transitions/s")
    rate = e4_rate(e4_receipt)
    selected_processes = _complete_payload(e4_receipt)["selected_processes"]
    # Each training arm has 4.5 allocation hours (9 units).  The supervisor
    # also needs timeout >= 2*estimate, so a rate that cannot fit an 8M/4M arm
    # into a 135-minute estimate is a pre-data refusal, not a quiet overrun.
    # Price 25% learner/checkpoint overhead and 15 minutes of allocation/setup.
    # This is a prospective estimate, not a measurement of CUDA learning speed.
    monitor_reports = math.ceil((transitions + CRITIC_COLLECTION_FORECAST) / 1_000_000)
    monitor_steps = monitor_reports * 3 * 2 * 5000
    # Probes run serially, unlike the ruler. Until measured, charge a 12x
    # slowdown against the aggregate ruler rate, then reprice before admission.
    monitor_minutes = 12 * monitor_steps / rate / 60
    estimate = math.ceil(
        1.25 * (transitions + CRITIC_COLLECTION_FORECAST + ORDINARY_BANK_TRANSITIONS) / rate / 60.0
        + 15
        + monitor_minutes
    )
    if estimate > 135:
        raise ManifestRefused(
            "E4 rate cannot fit the selected E2 length within the per-arm 9-unit cap"
        )
    receipt_inputs = [
        input_row(Path(str(e4_receipt) + suffix), e4_receipt.name + suffix)
        for suffix in ("", ".sha256", ".complete")
    ]
    receipt_inputs.extend(
        input_row(Path(str(e9_receipt) + suffix), e9_receipt.name + suffix)
        for suffix in ("", ".sha256", ".complete")
    )
    b8_input = input_row(b8_checkpoint, "B8_s3_final.pth")
    roster_inputs = _roster_inputs(roster_checkpoints)
    result: list[dict[str, Any]] = []
    for arm in ("r", "f"):
        manifest = _base(
            job_id=f"bl-e2-{arm}-{commit[:8]}",
            commit=commit,
            output=output_root / f"e2-{arm}",
            estimate=estimate,
            timeout=270,
            cap=9.0,
        )
        manifest.update(
            {
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1",
                    "BL_PPO_MODE": "train",
                    "BL_PPO_DEVICE": "cuda",
                    "BL_PPO_ARM": arm.upper(),
                    "BL_PPO_TRANSITIONS": str(transitions),
                    # run.py interprets this as seconds, keeping a dropped L4 from
                    # losing more than about two minutes of completed rollouts.
                    "BL_PPO_E4_BASENAME": e4_receipt.name,
                    "BL_PPO_E9_BASENAME": e9_receipt.name,
                    "BL_PPO_CHECKPOINT_EVERY": "1800",
                    "BL_PPO_HORIZON": "5000",
                    "BL_PPO_PROCESSES": str(selected_processes),
                    "BL_PPO_COMMIT": commit,
                    "BL_PPO_RATE_UNITS_PER_HOUR": str(L4_RATE),
                    "BL_PPO_UNIT_CAP": "9.0",
                },
                "inputs": [b8_input, *roster_inputs, *receipt_inputs],
                "max_export_bytes": E2_EXPORT_BYTES,
                "bl_ppo": {
                    "experiment": "E2",
                    "arm": arm.upper(),
                    "e4_rate_gate": rate,
                    "transitions": transitions,
                    "forecast_minutes": estimate,
                    "monitor_reports_max": monitor_reports,
                    "monitor_simulator_steps_max": monitor_steps,
                    "monitor_minutes_assumed": monitor_minutes,
                    "critic_refit_transitions_min": CRITIC_REFIT_TRANSITIONS,
                    "critic_collection_forecast": CRITIC_COLLECTION_FORECAST,
                    "ordinary_bank_decisions": ORDINARY_BANK_TRANSITIONS,
                    "target_scope": "actor transitions; critic refit charged separately",
                    "e8_amendment_sha256": sha256_file(ROOT / PACKAGE / "E8_AMENDMENT.md"),
                    "forecast_assumption": (
                        "E4 collection rate including 3.17M refit/holdout/finish + 10k bank; "
                        "25% learner/export overhead "
                        "plus 15min setup; serial probes charged12x aggregate E4 rate; "
                        "CUDA learner and probe rates unmeasured"
                    ),
                    "screen_reservation": {"allocation_hours": 1, "unit_cap": 2},
                },
            }
        )
        manifest["source_paths"] = _bounded_source_paths(manifest)
        result.append(_qualify_inputs(manifest))
    return result


def make_e2_templates(
    *,
    commit: str,
    output_root: Path,
    b8_checkpoint: Path,
    roster_checkpoints: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    """Return locally-valid, deliberately non-dispatchable E2 8M templates.

    There is no E4 result before E4 runs.  A template consequently has no
    invented input path or digest; ``materialize_e2`` is the only route to a
    dispatchable job and requires the immutable result as an uploaded input.
    """
    result: list[dict[str, Any]] = []
    for arm in ("r", "f"):
        manifest = _base(
            job_id=f"bl-e2-template-{arm}-{commit[:8]}",
            commit=commit,
            output=output_root / f"e2-{arm}",
            estimate=135,
            timeout=270,
            cap=9.0,
        )
        manifest.update(
            {
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1",
                    "BL_PPO_MODE": "train",
                    "BL_PPO_DEVICE": "cuda",
                    "BL_PPO_ARM": arm.upper(),
                    "BL_PPO_TRANSITIONS": "8000000",
                    "BL_PPO_CHECKPOINT_EVERY": "1800",
                    "BL_PPO_HORIZON": "5000",
                    "BL_PPO_PROCESSES": "12",
                    "BL_PPO_COMMIT": commit,
                    "BL_PPO_RATE_UNITS_PER_HOUR": str(L4_RATE),
                    "BL_PPO_UNIT_CAP": "9.0",
                },
                "inputs": [
                    input_row(b8_checkpoint, "B8_s3_final.pth"),
                    *_roster_inputs(roster_checkpoints),
                ],
                "max_export_bytes": E2_EXPORT_BYTES,
                "bl_ppo": {
                    "experiment": "E2",
                    "template_only": True,
                    "e9_admission_required": True,
                    "dispatch_block": (
                        "materialize with admitted E9 and E4 payload + .sha256 + .complete; "
                        "choose 4M/8M/20M from frozen E4 rule"
                    ),
                    "arm": arm.upper(),
                    "default_transitions": 8_000_000,
                    "critic_refit_transitions_min": CRITIC_REFIT_TRANSITIONS,
                    "critic_collection_forecast": CRITIC_COLLECTION_FORECAST,
                    "ordinary_bank_decisions": ORDINARY_BANK_TRANSITIONS,
                    "e8_amendment_sha256": sha256_file(ROOT / PACKAGE / "E8_AMENDMENT.md"),
                    "screen_reservation": {"allocation_hours": 1, "unit_cap": 2},
                },
            }
        )
        manifest["source_paths"] = _bounded_source_paths(manifest)
        result.append(_qualify_inputs(manifest))
    return result


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    from research.bl_ppo.upload_readiness import assert_upload_budget
    from research.colab_cli.supervisor import validate

    assert_upload_budget(manifest, repo=repo)
    return validate(manifest, repo=repo)


def dispatch_text(e4_path: Path, e2_paths: list[Path] | None = None) -> str:
    e2 = (
        "\n".join(f"- `{path}`" for path in (e2_paths or []))
        or "- Not yet generated: requires E4 immutable receipt."
    )
    return f"""# BL E4 then E2 dispatch handoff

Prepared only; no Colab call has occurred. Validate each exact manifest locally.
Z-9 receipt: `{e4_path.parent / 'upload-preflight.json'}`. Missing or stale
receipt means NOT_DISPATCHABLE. Refresh it after any manifest edit, including
account inventory. It binds the exact manifest and pinned source fallback bytes.
The six public checkpoint identities are downloaded and verified on the runtime;
only the small public-inputs catalog crosses the supervisor upload path.
Do not dispatch E2 until E4 exports its payload, SHA sidecar and completion marker,
its selected allocated hero rate is bound, and the forecast fits 20 units/10 L4 hours.
Never auto-select 20M.

- E4: `{e4_path}`
{e2}

Local checks (no allocation):
    PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
    MANIFEST={e4_path}
    PREP={e4_path.parent}
    $PY -m research.bl_ppo.upload_readiness --manifest "$MANIFEST" \\
      --receipt "$PREP/upload-preflight-refresh.json"
    $PY -m research.colab_cli.supervisor --state-dir "$PREP/supervisor" \\
      validate "$MANIFEST"

E4 remains a 128-world, 6/12-process L4 ruler: 15-minute estimate, 30-minute
wall cap, 2-unit cap at 2 u/h, 0.4-unit reserve, 600-second lease. Account
assignments must be refreshed by the dispatcher; the draft is not live inventory.

Every exported payload requires matching `.sha256` and `.complete` sidecars.
A missing sidecar is INCOMPLETE.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--roster-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--e4-receipt", type=Path)
    parser.add_argument("--e9-receipt", type=Path)
    parser.add_argument("--scale-20m", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--e4-only", action="store_true", help="prepare only the E4 recovery")
    args = parser.parse_args()
    commit, artifacts = args.commit or git_head(), args.artifacts.expanduser().resolve()
    if not artifacts.is_absolute():
        raise ManifestRefused("artifacts must be absolute")
    artifacts.mkdir(parents=True, exist_ok=True)
    e4_path = artifacts / "e4-manifest.json"
    roster = tuple(args.roster_checkpoint)
    e4 = make_e4_manifest(
        commit=commit,
        output_dir=artifacts / "e4-output",
        b8_checkpoint=args.b8,
        roster_checkpoints=roster,
    )
    if args.validate:
        e4 = validate_manifest(e4)
    from research.bl_ppo.upload_readiness import qualify

    qualify(e4, repo=ROOT, receipt_path=artifacts / "upload-preflight.json")
    from research.bl_ppo.exports import export_json

    export_json(e4_path, e4)
    paths: list[Path] = []
    if args.e4_receipt:
        for manifest in make_e2_manifests(
            commit=commit,
            output_root=artifacts,
            b8_checkpoint=args.b8,
            e4_receipt=args.e4_receipt,
            roster_checkpoints=roster,
            request_scale=args.scale_20m,
            e9_receipt=args.e9_receipt,
        ):
            target = artifacts / f"{manifest['job_id']}.json"
            if args.validate:
                manifest = validate_manifest(manifest)
            qualify(
                manifest,
                repo=ROOT,
                receipt_path=artifacts / f"{manifest['job_id']}-upload-preflight.json",
            )
            export_json(target, manifest)
            paths.append(target)
    elif not args.e4_only:
        for manifest in make_e2_templates(
            commit=commit, output_root=artifacts, b8_checkpoint=args.b8, roster_checkpoints=roster
        ):
            target = artifacts / f"{manifest['job_id']}.json"
            if args.validate:
                manifest = validate_manifest(manifest)
            qualify(
                manifest,
                repo=ROOT,
                receipt_path=artifacts / f"{manifest['job_id']}-upload-preflight.json",
            )
            export_json(target, manifest)
            paths.append(target)
    (artifacts / "DISPATCH.txt").write_text(dispatch_text(e4_path, paths), encoding="utf-8")


if __name__ == "__main__":
    main()
