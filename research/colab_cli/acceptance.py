#!/usr/bin/env python3
"""Prepare, and only after review execute, the bounded Colab CLI acceptance plan.

The default is intentionally dry: it writes pinned, fresh manifests and a
review record but performs no Colab command.  ``--execute-after-review`` is an
operator action after the acceptance plan has been reviewed; it still has no
automatic retry, purchase, or fallback allocation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from research.colab_cli import supervisor
from research.colab_cli.acceptance_faults import AcceptanceObserver

MAX_UNITS = 4.0
RESERVE_UNITS = 0.8
USABLE_UNITS = 3.2
L4_RATE_UNITS_PER_HOUR = 2.0
CPU_RATE_UNITS_PER_HOUR = 0.5
L4_MAX_MINUTES = 30
CPU_MAX_MINUTES = 20
REVIEW_TOKEN = "COLAB_CLI_ACCEPTANCE_REVIEWED"
BASELINE = {
    "supplied_at": "2026-10-18 18:38 PT",
    "balance_units": 1174.37,
    "aggregate_usage_rate_units_per_hour": 0.0,
    "active_notebook_assignments": 0,
    "note": (
        "Supplied context, not a new live measurement. Re-read usage and sessions before dispatch."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def create_json(path: Path, value: dict[str, Any]) -> None:
    """Create an immutable execution receipt without replacing plan evidence."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise ValueError(f"fresh output required; path already exists: {path}")
    path.mkdir(parents=True)
    return path


def manifest(
    *,
    job_id: str,
    commit: str,
    probe: Path,
    wrapper: Path,
    output: Path,
    gpu: str,
    minutes: int,
    priced_rate: float,
    other_assignments: dict[str, float],
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "commit": commit,
        "script": "research/colab_cli/acceptance_probe.sh",
        "env": {"PYTHONUNBUFFERED": "1", "COLAB_ACCEPTANCE": "1"},
        "requirements": {"gpu": gpu, "min_vcpu": 2, "est_minutes": minutes / 2},
        "timeout_minutes": minutes,
        "output_dir": str(output),
        "inputs": [
            {
                "path": str(probe),
                "sha256": sha256_file(probe),
                "dest": "acceptance_probe.py",
            },
            {"path": str(wrapper), "sha256": sha256_file(wrapper), "dest": "acceptance_probe.sh"},
        ],
        "budget": {
            # This is deliberately a conservative price ceiling.  It controls
            # admission, the liability reservation, and the session cap even
            # when the provider later reports a lower actual rate.
            "unit_cap": priced_rate * minutes / 60.0,
            "rate_units_per_hour": priced_rate,
            "reserve_units": RESERVE_UNITS,
            # Every assignment already present at admission must be priced so
            # an account-wide balance delta cannot silently consume this arm's
            # small unit cap.  The broker rejects an unmapped endpoint.
            "other_assignments": other_assignments,
        },
        "lease_seconds": 300 if gpu == "cpu" else 180,
    }


def review_gates() -> list[dict[str, str]]:
    """Every advice gate has a named required artifact; absence is INCOMPLETE."""
    return [
        {
            "id": "account-assignment-spend",
            "evidence": "all pre-existing endpoints and their rates are recorded at admission",
            "owner": "Mac supervisor",
        },
        {
            "id": "fresh-oauth-refresh",
            "evidence": "fresh terminal whoami/usage after refresh; no browser prompt",
            "owner": "Mac supervisor",
        },
        {
            "id": "busy-full-snapshot",
            "evidence": "remote snapshot.complete.json, downloaded hash, atomic local file",
            "owner": "probe + supervisor",
        },
        {
            "id": "silence-timeout",
            "evidence": "over-60s gap and explicit supervisor timeout record",
            "owner": "probe + supervisor",
        },
        {
            "id": "url-no-second-allocation",
            "evidence": "sessions before/after url show same single assignment",
            "owner": "Mac supervisor",
        },
        {
            "id": "client-loss-reconnect",
            "evidence": "generation fence and one remote execution identity after reconnect",
            "owner": "Mac supervisor",
        },
        {
            "id": "failure-and-corrupt-export",
            "evidence": "intentional child exit recorded; corrupt download rejected",
            "owner": "probe + supervisor",
        },
        {
            "id": "exact-resume",
            "evidence": "uninterrupted and resumed full-state hashes match",
            "owner": "probe + supervisor",
        },
        {
            "id": "mac-loss-watchdog-unassign",
            "evidence": "lease expiry plus backend assignment disappearance",
            "owner": "runtime watchdog + supervisor",
        },
        {
            "id": "checkpoint-overhead",
            "evidence": "timed checkpoint overhead <= 2 percent",
            "owner": "supervisor",
        },
        {
            "id": "backend-stop-disappearance",
            "evidence": "stop issued then sessions/status proves assignment gone",
            "owner": "Mac supervisor",
        },
    ]


def build_plan(
    root: Path,
    state_dir: Path,
    cpu_rate: float | None = CPU_RATE_UNITS_PER_HOUR,
    other_assignments: dict[str, float] | None = None,
    l4_rate: float = L4_RATE_UNITS_PER_HOUR,
) -> dict[str, Any]:
    if cpu_rate is None:
        cpu_rate = CPU_RATE_UNITS_PER_HOUR
    if not math.isfinite(cpu_rate) or cpu_rate <= 0:
        raise ValueError("--cpu-rate must be finite and positive")
    if not math.isfinite(l4_rate) or l4_rate <= 0:
        raise ValueError("--l4-rate must be finite and positive")
    other_assignments = other_assignments or {}
    if not all(
        isinstance(endpoint, str)
        and endpoint
        and isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and rate > 0
        for endpoint, rate in other_assignments.items()
    ):
        raise ValueError("other assignment rates must map nonempty endpoints to positive numbers")
    cpu_expected = cpu_rate * CPU_MAX_MINUTES / 60.0
    l4_expected = l4_rate * L4_MAX_MINUTES / 60.0
    if cpu_expected + l4_expected > USABLE_UNITS:
        raise ValueError(
            f"expected {cpu_expected + l4_expected:.3f} units exceeds usable {USABLE_UNITS:.1f}"
        )
    state_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"acceptance-{int(time.time())}"
    run_dir = fresh_dir(state_dir.resolve() / run_id)
    output_root = fresh_dir(run_dir / "outputs")
    manifests = fresh_dir(run_dir / "manifests")
    probe = root / "research/colab_cli/acceptance_probe.py"
    wrapper = root / "research/colab_cli/acceptance_probe.sh"
    if not probe.is_file() or not wrapper.is_file():
        raise ValueError("missing acceptance probe or wrapper")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    rows = [
        ("cpu", "cpu", CPU_MAX_MINUTES, cpu_rate),
        ("l4", "L4", L4_MAX_MINUTES, l4_rate),
    ]
    jobs: list[dict[str, Any]] = []
    for name, gpu, minutes, rate in rows:
        # The broker refuses an existing job output directory, so this is a
        # planned path only.  It creates it after manifest validation.
        output = output_root / name
        payload = manifest(
            job_id=f"{run_id}-{name}",
            commit=commit,
            probe=probe,
            wrapper=wrapper,
            output=output,
            gpu=gpu,
            minutes=minutes,
            priced_rate=rate,
            other_assignments={key: float(value) for key, value in other_assignments.items()},
        )
        target = manifests / f"{name}.json"
        atomic_json(target, payload)
        jobs.append(
            {
                "name": name,
                "manifest": str(target),
                "priced_rate_units_per_hour": rate,
                "expected_units": payload["budget"]["unit_cap"],
            }
        )
    plan = {
        "schema": "snake-dqn-colab-cli-acceptance/v1",
        "status": "NOT_RUN",
        "baseline_context": BASELINE,
        "limits": {
            "max_units_including_reserve": MAX_UNITS,
            "reserve_units": RESERVE_UNITS,
            "usable_units": USABLE_UNITS,
            "first_cap_wins": True,
        },
        "expected_costs": {
            "cpu_20_minutes_at_priced_rate": cpu_expected,
            "l4_30_minutes_at_priced_rate": l4_expected,
            "total": cpu_expected + l4_expected,
        },
        "jobs": jobs,
        "gates": [{**gate, "status": "INCOMPLETE"} for gate in review_gates()],
        "qualification": "INCOMPLETE until every gate has independently verified evidence",
    }
    atomic_json(run_dir / "acceptance-plan.json", plan)
    return {"run_dir": str(run_dir), "plan": plan}


def load_other_assignments(path: Path | None) -> dict[str, float]:
    """Load the reviewed endpoint-to-rate map without ever discovering endpoints by guesswork."""
    if path is None:
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("--other-assignments must contain a JSON object")
    result: dict[str, float] = {}
    for endpoint, rate in value.items():
        if not isinstance(endpoint, str) or not endpoint or isinstance(rate, bool):
            raise ValueError("--other-assignments has an invalid endpoint/rate")
        try:
            numeric = float(rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("--other-assignments has a nonnumeric rate") from exc
        if numeric <= 0:
            raise ValueError("--other-assignments rates must be positive")
        result[endpoint] = numeric
    return result


def execute(
    plan: dict[str, Any], *, state_dir: Path, colab: str, review_token: str, oauth_python: str
) -> int:
    if review_token != REVIEW_TOKEN:
        raise ValueError("execution requires the exact reviewed acceptance token")
    observer = AcceptanceObserver(oauth_python)
    results: list[dict[str, Any]] = []
    run_dir = Path(plan["jobs"][0]["manifest"]).parent.parent
    for job in plan["jobs"]:
        manifest_path = Path(job["manifest"])
        payload = json.loads(manifest_path.read_text())
        output = Path(payload["output_dir"])
        try:
            supervisor.Broker(state_dir.resolve() / "lanes", colab, observer=observer).run(payload)
            receipt = _clean_release_receipt(output)
            measurement = _calibration_evidence(receipt)
            observer._record("backend-stop-disappearance", receipt=str(output / "receipt.json"))
            results.append(
                {
                    "job": job["name"],
                    "status": "PASS" if measurement["status"] == "PASS" else "INCOMPLETE",
                    "receipt": str(output / "receipt.json"),
                    "priced_rate_units_per_hour": measurement["priced_rate_units_per_hour"],
                    "measured_rate_units_per_hour": measurement["observed_rate_units_per_hour"],
                    "rate_calibration": measurement,
                }
            )
        except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
            # The L4 arm intentionally stops renewals.  It only passes when the
            # runtime-side release was observed within the stated bound *before*
            # broker cleanup, and the broker receipt reports a clean release.
            watchdog = observer.records.get("mac-loss-watchdog-unassign", {})
            released = bool(watchdog.get("assignment_disappeared_within_bound"))
            try:
                receipt = _clean_release_receipt(output, require_success=False)
                measurement = _calibration_evidence(receipt)
                clean_cleanup = True
            except (OSError, ValueError, supervisor.Refused):
                clean_cleanup = False
                measurement = {
                    "status": "INCOMPLETE",
                    "reason": "missing clean release receipt or provider rate calibration",
                    "priced_rate_units_per_hour": None,
                    "observed_rate_units_per_hour": None,
                }
            results.append(
                {
                    "job": job["name"],
                    "status": (
                        "PASS"
                        if (
                            job["name"] == "l4"
                            and released
                            and clean_cleanup
                            and measurement["status"] == "PASS"
                        )
                        else "INCOMPLETE"
                    ),
                    "error": str(exc),
                    "priced_rate_units_per_hour": measurement["priced_rate_units_per_hour"],
                    "measured_rate_units_per_hour": measurement["observed_rate_units_per_hour"],
                    "rate_calibration": measurement,
                }
            )
        finally:
            if output.exists():
                observer.write(output)
        # A CPU admission/transport failure is not a reason to spend an L4
        # allocation. The watchdog fault is expected only after CPU passed.
        if job["name"] == "cpu" and results[-1]["status"] != "PASS":
            break
    gate_status = {gate["id"]: gate for gate in plan["gates"]}
    for gate, evidence in observer.records.items():
        if gate in gate_status:
            incomplete = str(evidence.get("status", "")).startswith("INCOMPLETE")
            gate_status[gate]["status"] = "INCOMPLETE" if incomplete else "OBSERVED"
            gate_status[gate]["evidence_record"] = evidence
    plan["execution"] = results
    plan["status"] = "OBSERVED_NOT_QUALIFIED"
    plan["qualification"] = (
        "INCOMPLETE pending independent review of every observer/receipt artifact"
    )
    qualified = (
        len(observer.records) == len(gate_status)
        and all(gate["status"] == "OBSERVED" for gate in plan["gates"])
        and all(result["status"] == "PASS" for result in results)
    )
    return_code = 0 if qualified else 1
    create_json(
        run_dir / "acceptance-result.json",
        {
            "schema": "snake-dqn-colab-cli-acceptance-result/v1",
            "return_code": return_code,
            "qualified": qualified,
            "final_plan": plan,
        },
    )
    return return_code


def _clean_release_receipt(output: Path, *, require_success: bool = True) -> dict[str, Any]:
    """Require broker cleanup evidence even when the watchdog ended execution.

    A release observation made by the L4 fault arm is not a waiver for a
    cleanup error.  In particular this prevents an earlier `stop` failure from
    being converted into a PASS merely because an observer record exists.
    """
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file():
        raise supervisor.Refused("missing broker receipt after acceptance release")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("cleanup_error"):
        raise supervisor.Refused("acceptance cleanup error: " + str(receipt["cleanup_error"]))
    if not receipt.get("stop_verified"):
        raise supervisor.Refused("acceptance release was not backend-verified")
    if require_success and receipt.get("status") not in {"PASS", "COMPLETED_SESSION_RETAINED"}:
        raise supervisor.Refused("acceptance receipt is incomplete")
    return receipt


def _calibration_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    """Require the bounded post-allocation rate calibration in every acceptance result.

    The priced rate remains the cap and reservation input.  This record is the
    provider-observed rate used by later polling drift checks, so an absent,
    incomplete, or malformed calibration cannot be represented as a pass.
    """
    accounting = receipt.get("accounting")
    calibration = accounting.get("calibration") if isinstance(accounting, dict) else None
    if not isinstance(calibration, dict):
        return {
            "status": "INCOMPLETE",
            "reason": "missing provider rate calibration in broker receipt",
            "priced_rate_units_per_hour": None,
            "observed_rate_units_per_hour": None,
        }
    priced = calibration.get("priced_rate_units_per_hour")
    observed = calibration.get("observed_rate_units_per_hour")
    if (
        calibration.get("status") != "PASS"
        or isinstance(priced, bool)
        or not isinstance(priced, (int, float))
        or isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(priced))
        or float(priced) <= 0
        or not math.isfinite(float(observed))
        or float(observed) <= 0
        or float(observed) > float(priced) + 0.05
    ):
        return {
            "status": "INCOMPLETE",
            "reason": str(calibration.get("reason", "provider rate calibration incomplete")),
            "priced_rate_units_per_hour": priced,
            "observed_rate_units_per_hour": observed,
            "samples": calibration.get("samples", []),
        }
    return {
        "status": "PASS",
        "priced_rate_units_per_hour": float(priced),
        "observed_rate_units_per_hour": float(observed),
        "samples": calibration.get("samples", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--state-dir", type=Path, required=True, help="absolute supervisor state parent"
    )
    parser.add_argument(
        "--cpu-rate",
        "--cpu-rate-units-per-hour",
        dest="cpu_rate",
        type=float,
        default=CPU_RATE_UNITS_PER_HOUR,
        help="conservative CPU price ceiling (default: 0.5 units/hour)",
    )
    parser.add_argument(
        "--l4-rate",
        "--l4-rate-units-per-hour",
        dest="l4_rate",
        type=float,
        default=L4_RATE_UNITS_PER_HOUR,
        help="conservative L4 price ceiling (default: 2.0 units/hour)",
    )
    parser.add_argument(
        "--other-assignments",
        type=Path,
        help="reviewed JSON endpoint-to-rate map for every pre-existing assignment",
    )
    parser.add_argument(
        "--colab", default="colab", help="official CLI executable, used only with execution"
    )
    parser.add_argument(
        "--oauth-python",
        default=str(Path.home() / ".venvs" / "colab-cli" / "bin" / "python"),
        help="official Colab CLI venv interpreter, used only after review",
    )
    parser.add_argument("--execute-after-review", action="store_true")
    parser.add_argument("--review-token", default="")
    args = parser.parse_args(argv)
    if not args.state_dir.is_absolute():
        parser.error("--state-dir must be absolute")
    try:
        record = build_plan(
            args.repo.resolve(),
            args.state_dir,
            args.cpu_rate,
            load_other_assignments(args.other_assignments),
            args.l4_rate,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        if args.execute_after_review:
            return execute(
                record["plan"],
                state_dir=args.state_dir,
                colab=args.colab,
                review_token=args.review_token,
                oauth_python=args.oauth_python,
            )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
