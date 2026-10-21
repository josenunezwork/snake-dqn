"""Prepare and locally validate E6/E7 manifests. This module never dispatches."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json
from research.bl_prefix.resource_limits import LIMITS, SESSION_LIMIT_MINUTES
from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.colab_cli.supervisor import validate

REPO = Path(__file__).resolve().parents[2]


def source_paths(repo: Path = REPO) -> list[str]:
    """Return the bounded local runtime-import closure for the prefix launcher."""
    pending = [repo / "research/bl_prefix/dispatch.py"]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for ancestor in path.parents:
            if ancestor == repo:
                break
            initializer = ancestor / "__init__.py"
            if initializer.is_file() and initializer not in visited:
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    package = list(path.relative_to(repo).parts[:-1])
                    prefix = ".".join(
                        package[: len(package) - node.level + 1] + ([prefix] if prefix else [])
                    )
                if prefix:
                    names = [prefix, *(f"{prefix}.{item.name}" for item in node.names)]
            for name in names:
                base = repo.joinpath(*name.split("."))
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if candidate.is_file() and candidate not in visited:
                        pending.append(candidate)
    return sorted(
        {str(path.relative_to(repo)) for path in visited}
        | {
            "research/bl_prefix/run.sh",
            "research/bl_prefix/requirements.txt",
            "research/colab_cli",
            "src",
            "configs",
            "pyproject.toml",
            "requirements.txt",
        }
    )


def sha256(path: Path) -> str:
    """Hash one immutable input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualified_transport(repo: Path, commit: str) -> dict[str, str]:
    """Compare local and pinned executable bytes to the accepted inventory."""
    document = (repo / "research/colab_cli/QUALIFICATION.md").read_text()
    records = document.split("```json\n")[1:]
    inventories = [json.loads(record.split("```", 1)[0]) for record in records]
    accepted = next(item for item in inventories if item.get("status") == "PASS")
    inventory = accepted["source_manifest"]
    for relative, expected in inventory.items():
        if sha256(repo / relative) != expected:
            raise ValueError(f"unqualified local transport bytes: {relative}")
        pinned = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
        if hashlib.sha256(pinned).hexdigest() != expected:
            raise ValueError(f"unqualified pinned transport bytes: {relative}")
    current = {
        str(path.relative_to(repo))
        for path in (repo / "research/colab_cli").rglob("*")
        if path.suffix in {".py", ".sh"}
    }
    if current != set(inventory):
        raise ValueError("qualified transport executable inventory changed")
    return {
        "qualified_commit": accepted["qualified_commit"],
        "receipt_sha256": accepted["receipt_sha256"],
    }


def forecast(experiment: str, seconds_per_world_step: float) -> dict[str, Any]:
    """Conservative all-phase bound; L4 CPU scaling is an unmeasured forecast."""
    if not math.isfinite(seconds_per_world_step) or seconds_per_world_step <= 0:
        raise ValueError("positive measured seconds/world-step required")
    # Full H5000 bounds include source-bank collection and all secondary rows.
    # E6: collection + selected replay + terminal-sealer replay + both protocols.
    # E7: 128 worlds, 3 ancestors, (8 train + 1 train B8 + 4 paired confirms).
    steps = (
        (4096 + 256 + 128) * 5000 + 256 * 3 * 2 * 2 * 5000
        if experiment == "e6"
        else 128 * 3 * (8 + 1 + 4) * 5000
    )
    central_minutes = 15 + steps * seconds_per_world_step * 3 / 12 / 60
    cap, reserve, timeout = LIMITS[experiment]
    return {
        "schema": "bl-prefix-forecast/v1",
        "experiment": experiment,
        "world_step_upper_bound": steps,
        "workers": 12,
        "mac_seconds_per_world_step": seconds_per_world_step,
        "l4_cpu_slowdown_assumption": 3.0,
        "setup_export_minutes": 15,
        "estimated_allocation_minutes": central_minutes,
        "estimated_units_at_2_per_hour": central_minutes / 30,
        "unit_cap": cap,
        "reserve_units": reserve,
        "usable_timeout_minutes": timeout,
        "fits_with_two_times_margin": 2 * central_minutes <= timeout <= SESSION_LIMIT_MINUTES,
        "minimum_unit_cap_with_reserve_and_two_times_margin": central_minutes / 12,
        "fits_single_24h_session": 2 * central_minutes <= SESSION_LIMIT_MINUTES,
        "status": "FORECAST_ONLY_NOT_MEASURED_ON_L4",
    }


def build(
    experiment: str,
    plan_path: Path,
    output_dir: Path,
    commit: str,
    *,
    input_files: list[tuple[str, Path]] | None = None,
    estimate_minutes: float | None = None,
    repo: Path = REPO,
) -> dict[str, Any]:
    """Bind actual local files and validate through the unchanged supervisor."""
    if experiment not in LIMITS:
        raise ValueError("experiment must be e6 or e7")
    authority = qualified_transport(repo, commit)
    plan = json.loads(plan_path.read_text())
    if plan.get("experiment") != experiment:
        raise ValueError("plan experiment mismatch")
    cap, reserve, timeout = LIMITS[experiment]
    estimate = timeout / 2 if estimate_minutes is None else estimate_minutes
    if (
        not math.isfinite(estimate)
        or estimate <= 0
        or 2 * estimate > timeout
        or timeout > SESSION_LIMIT_MINUTES
    ):
        raise ValueError("forecast does not fit the frozen cap with 2x margin")
    inputs = [{"path": str(plan_path.resolve()), "dest": "plan.json", "sha256": sha256(plan_path)}]
    for dest, path in input_files or []:
        inputs.append({"path": str(path.resolve()), "dest": dest, "sha256": sha256(path)})
        companions = [Path(str(path) + suffix) for suffix in (".sha256", ".complete")]
        if any(part.exists() for part in companions):
            from research.bl_ppo.exports import verify_export

            verify_export(path)
            if Path(dest).name != path.name:
                raise ValueError("completed input basename must be preserved with its sidecars")
            for suffix, companion in zip((".sha256", ".complete"), companions):
                inputs.append(
                    {
                        "path": str(companion.resolve()),
                        "dest": dest + suffix,
                        "sha256": sha256(companion),
                    }
                )
    from research.bl_prefix.readiness import validate_ready

    delivered = {item["dest"]: Path(item["path"]) for item in inputs}
    validate_ready(plan, delivered)
    resource_ready = False
    forecast_tag = plan.get("forecast", "")
    if isinstance(forecast_tag, str) and forecast_tag.startswith("input:"):
        bound = delivered.get(forecast_tag[6:])
        if bound is not None:
            from research.bl_ppo.exports import verify_export

            verify_export(bound)
            measured = json.loads(bound.read_text())
            recomputed = forecast(experiment, measured["mac_seconds_per_world_step"])
            if any(measured.get(key) != value for key, value in recomputed.items()):
                raise ValueError("resource forecast differs from measured work/cap calculation")
            resource_ready = recomputed["fits_with_two_times_margin"] and math.isclose(
                estimate, recomputed["estimated_allocation_minutes"], rel_tol=1e-12
            )
    manifest = {
        "job_id": f"bl-prefix-{experiment}-{output_dir.name}"[:64],
        "commit": commit,
        "script": "research/bl_prefix/run.sh",
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir),
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": 2.0,
            "reserve_units": reserve,
            "other_assignments": [],
        },
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_prefix/requirements.txt",
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "inputs": inputs,
        "source_paths": source_paths(repo),
        "lease_seconds": 300,
        "max_export_bytes": 64 * 1024**2,
        "bl_prefix": {
            "development_only": True,
            "excluded_from_registered_verdicts": True,
            "ready": plan.get("ready") is True,
            "dispatch_status": "NOT_DISPATCHED",
            "transport": authority,
            "resource_readiness": {
                "ready": resource_ready,
                "scope": "CODE_FORECAST_AND_PROPOSED_BUDGET_ONLY",
                "dispatch_ready": plan.get("ready") is True,
                "dependencies": list(plan.get("blocking_reasons", [])),
                "cap_authority": "research/bl_prefix/CAP_NOTE.md (proposal; not dispatch approval)",
            },
        },
    }
    manifest = convert_public_checkpoints(manifest, repo=repo)
    assert_upload_budget(manifest, repo=repo)
    return validate(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=LIMITS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--input", action="append", default=[], metavar="DEST=PATH")
    parser.add_argument("--estimate-minutes", type=float)
    args = parser.parse_args()
    inputs = [(value.split("=", 1)[0], Path(value.split("=", 1)[1])) for value in args.input]
    manifest = build(
        args.experiment,
        args.plan,
        args.output_dir,
        args.commit,
        input_files=inputs,
        estimate_minutes=args.estimate_minutes,
    )
    export_json(args.out, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.out),
                "supervisor_validation": "PASS",
                "ready": manifest["bl_prefix"]["ready"],
                "dispatch_status": "NOT_DISPATCHED",
            }
        )
    )


if __name__ == "__main__":
    main()
