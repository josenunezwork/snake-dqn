"""Build and locally validate an undispatched E3 supervisor manifest."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from research.bl_info_value.io import sha256, verify_export, write_json
from research.colab_cli.source import prepare
from research.colab_cli.supervisor import validate

MAX_LOCAL_INPUT_BYTES = 1024**2


def forecast(timing: dict[str, Any], workers: int = 6) -> dict[str, Any]:
    """Extrapolate operational smoke costs, retaining explicit platform uncertainty."""
    metrics = timing["world_metrics"]
    # Sum worker costs, then scale base replay by worlds and branches by roots.
    # max(world factor, root factor) is conservative when replay dominates.
    factor = max(256 / timing["worlds"], 1536 / timing["roots"])
    worker_seconds = sum(m["elapsed_seconds"] for m in metrics)
    # Historical L4 host cores were about 3.7x slower than Mac cores; round to4.
    if timing.get("parity_seconds", 0) > 0:
        # The two-process qualification is the closest measured execution shape
        # to the six-process runtime. Initial serial includes cold Mac JIT;
        # runtime setup/cold JIT is separately covered by the20min allowance.
        parallel_seconds = (
            timing["parity_seconds"] if timing["workers"] == 1 else timing["reconstruction_seconds"]
        )
        reconstruct_seconds = parallel_seconds * 2 * factor * 4 / workers
    else:
        reconstruct_seconds = worker_seconds * factor * 4 / workers
    # Real batch32 optimizer updates and batch192 inference use duplicated tiny
    # TRAIN inputs only; this measures full-batch cost without new outcome looks.
    # No CUDA speedup is credited. At1152train roots, 36updates/epoch x60epochs.
    benchmark = timing["fits"]["benchmark"]
    fit_seconds = sum(
        row["batch32_update_seconds"] * 36 * 60 + row["batch192_inference_seconds"] * 61
        for row in benchmark.values()
    )
    overhead_seconds = 20 * 60
    hours = (reconstruct_seconds + fit_seconds + overhead_seconds) / 3600
    serial_hours = (worker_seconds * factor * 4 / workers + fit_seconds + overhead_seconds) / 3600
    return dict(
        reconstruction_hours=reconstruct_seconds / 3600,
        fit_hours_cpu_upper_projection=fit_seconds / 3600,
        setup_export_hours=overhead_seconds / 3600,
        total_hours=hours,
        serial_cold_projection_hours=serial_hours,
        serial_cold_projection_units=serial_hours * 2,
        units_at_2_per_hour=hours * 2,
        reserve_units=3.2,
        cap_units=16,
        usable_hours=6.4,
        workers=workers,
        host_slowdown_assumption=4,
        cuda_speedup_assumed=1,
        reconstruction_basis="measured two-process qualification; cold JIT in setup allowance",
        dispatch_within_forecast=hours <= 6.4,
        limitations=(
            "Three contended Mac worlds; no L4 timing measured. "
            "CPU batch timings without CUDA speedup; projection, not a measured L4 rate."
        ),
    )


def build(
    inputs: Path,
    timing_path: Path,
    destination: Path,
    *,
    assignments: dict[str, float] | None = None,
    checkpoint_url: str,
    prior_units: float = 0.2,
    attempt: str = "02",
) -> dict[str, Any]:
    """Bind a published source commit, verified inputs, forecast and fresh output."""
    if destination.exists():
        raise FileExistsError(destination)
    verify_export(inputs)
    verify_export(timing_path)
    timing = json.loads(timing_path.read_text())
    with tarfile.open(inputs, "r:gz") as archive:
        catalog = json.load(archive.extractfile("catalog.json"))
    if catalog["schema"] != "bl-e3-input/v2" or inputs.stat().st_size > MAX_LOCAL_INPUT_BYTES:
        raise ValueError("compact v2 catalog upload must be at most 1 MiB")
    if catalog["smoke"]:
        raise ValueError("production manifest requires the full frozen root catalog")
    if not math.isfinite(prior_units) or prior_units < 0:
        raise ValueError("failed-attempt charge must be finite and nonnegative")
    if not attempt.isdigit() or int(attempt) < 2:
        raise ValueError("recovery requires a fresh attempt identifier >= 02")
    if not timing["smoke"] or timing["parity_seconds"] <= 0:
        raise ValueError("real Mac preflight with serial/parallel proof required")
    plan = forecast(timing)
    plan.update(prior_attempt_units=prior_units, usable_hours=(12.8 - prior_units) / 2)
    plan["dispatch_within_forecast"] = plan["total_hours"] <= plan["usable_hours"]
    estimate = 60 * plan["total_hours"]
    timeout = max(3, math.ceil(2 * estimate))
    if timeout / 60 * 2 > 12.8 - prior_units:
        # Keep the full design and rule; never forge a lower estimate to fit cap.
        destination.mkdir(parents=True)
        write_json(
            destination / "DISPATCH_BLOCKED.json",
            plan | dict(reason="2x estimate exceeds remaining usable session envelope"),
        )
        return {"blocked": True, "forecast": plan}
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=repo, check=True
    )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "research/bl_info_value"], cwd=repo, text=True
    ).strip():
        raise ValueError("commit package implementation before manifest pinning")
    source_paths = [
        "src",
        "configs",
        "pyproject.toml",
        "research/colab_cli",
        "research/bl_info_value",
    ]
    source_paths += timing["research_sources"]
    # Generic supervisor validation does not automatically apply the T34
    # transport qualification fence to a new development package. Reuse its
    # read-only verifier so both local and pinned executable bytes are checked.
    from research.t34_exit_b8.qualification import require_qualified_transport

    qualification = require_qualified_transport(repo, commit)
    manifest = dict(
        job_id=f"bl-e3-info-value-{attempt}",
        commit=commit,
        script="research/bl_info_value/job.sh",
        source_paths=sorted(set(source_paths)),
        env={
            "COLAB_CLI_REQUIREMENTS": "research/bl_info_value/requirements.txt",
            "E3_INPUT_SHA256": sha256(inputs),
            "E3_WORKERS": "6",
        },
        requirements=dict(gpu="L4", allowed_gpus=["L4"], min_vcpu=6, est_minutes=estimate),
        timeout_minutes=timeout,
        lease_seconds=300,
        output_dir=str(destination.parent / f"colab-output-{attempt}"),
        inputs=[
            dict(path=str(inputs.resolve()), dest="inputs.tar.gz", sha256=sha256(inputs)),
            dict(url=checkpoint_url, dest="checkpoints.tar.gz", **catalog["checkpoint_archive"]),
        ],
        budget=dict(
            unit_cap=12.8 - prior_units,
            rate_units_per_hour=2,
            reserve_units=3.2,
            other_assignments=assignments or {},
        ),
        max_export_bytes=256 * 1024**2,
    )
    validated = validate(manifest, repo=repo)
    destination.mkdir(parents=True)
    source = prepare(repo, validated, destination / "source")
    write_json(destination / "job.json", validated)
    write_json(destination / "forecast.json", plan)
    write_json(
        destination / "validation.json",
        dict(
            status="VALIDATED_NOT_DISPATCHED",
            commit=commit,
            source_plan=source,
            forecast_within_usable_budget=plan["dispatch_within_forecast"],
            input_sha256=sha256(inputs),
            timing_sha256=sha256(timing_path),
            local_input_bytes=inputs.stat().st_size,
            local_input_limit_bytes=MAX_LOCAL_INPUT_BYTES,
            checkpoint_archive=catalog["checkpoint_archive"],
            prior_attempt_units=prior_units,
            transport_qualification=qualification,
        ),
    )
    return dict(manifest=str(destination / "job.json"), forecast=plan, commit=commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-url", required=True)
    parser.add_argument("--prior-units", type=float, default=0.2)
    parser.add_argument("--attempt", default="02")
    parser.add_argument(
        "--assignments", type=Path, help="Operator-reviewed endpoint-to-positive-rate JSON"
    )
    args = parser.parse_args()
    assignments = json.loads(args.assignments.read_text()) if args.assignments else None
    print(
        json.dumps(
            build(
                args.inputs,
                args.timing,
                args.out,
                assignments=assignments,
                checkpoint_url=args.checkpoint_url,
                prior_units=args.prior_units,
                attempt=args.attempt,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
