"""Outcome-blind reuse of retained L4 timings for operational shard envelopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.performance import MAX_BASE_MINUTES, SETUP_MINUTES
from tools.ops.package_spec import digest, publish_json, verify_immutable

BASE_PIN = "79668a17962606f12bb674352c5e3bc04aa1702f"
BASE_SCIENCE = "8ac54d173be3bdc85b2f280f03d35a4eeb69aa6307aa96d182ee85e14a3e47ff"
OPERATIONAL_PATHS = {
    "research/watch6_search_efficacy/costs.py",
    "research/watch6_search_efficacy/performance.py",
    "research/watch6_search_efficacy/prepare.py",
    "research/watch6_search_efficacy/admission.py",
    "research/watch6_search_efficacy/finalize.py",
    "research/watch6_search_efficacy/dispatch.py",
    "research/watch6_search_efficacy/README.md",
}
MARGIN = 1.5
REPLAY_HORIZON = 6000


def compatible_source() -> None:
    """Retain timing evidence only while execution/transport bytes are unchanged."""
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_PIN, "--", *p.source_paths()],
        cwd=p.ROOT,
        text=True,
    ).splitlines()
    # git diff omits untracked files; inspect those inside the bound source closure too.
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *p.source_paths()],
        cwd=p.ROOT,
        text=True,
    ).splitlines()
    if set(changed + untracked) - OPERATIONAL_PATHS:
        raise ValueError("retained timing source differs outside operational forecast/admission")
    if canary.executable_identity(p.ROOT, BASE_PIN) != canary.executable_identity(p.ROOT):
        raise ValueError("retained timing transport differs")


def verified_run(output: Path, *, measured_failure: bool = False) -> tuple[dict, dict]:
    """Verify immutable exported bytes and actual upload evidence, without outcomes."""
    receipt = json.loads((output / "receipt.json").read_text())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        digest(manifest_path) != receipt["manifest_sha256"]
        or receipt["job_id"] != manifest["job_id"]
        or receipt["commit"] != manifest["commit"]
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        raise ValueError("retained run identity/release differs")
    terminal = receipt["terminal"]
    if measured_failure:
        if (
            manifest["commit"] != BASE_PIN
            or receipt["status"] != "INCOMPLETE"
            or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
            or terminal.get("returncode") != 1
            or terminal.get("last_traceback_line")
            != "ValueError: NOT_READY measured forecast exceeds shard wall with margin"
        ):
            raise ValueError("only the retained post-measurement forecast refusal is reusable")
        canary.staging_measurement(
            manifest, json.loads((output / "upload-measurement.json").read_text())
        )
    elif receipt["status"] != "PASS" or receipt.get("error"):
        raise ValueError("replay timing run did not PASS")
    if (
        terminal.get("generation") != receipt["generation"]
        or terminal.get("job_id") != receipt["job_id"]
        or sorted(terminal.get("export_scanner", {}).get("inventory", []), key=lambda r: r["path"])
        != sorted(receipt["exports"], key=lambda r: r["path"])
    ):
        raise ValueError("retained terminal/scanner identity differs")
    for row in receipt["exports"]:
        checked = verify_immutable(output / "exports" / row["path"])
        if checked["sha256"] != row["sha256"] or checked["size"] != row["size"]:
            raise ValueError("retained exported bytes differ")
    for item in manifest["inputs"]:
        path = Path(item["path"])
        if digest(path) != item["sha256"] or not any(
            t.get("direction") == "upload"
            and t.get("status") == "verified"
            and t.get("remote", "").endswith("/inputs/" + item["dest"])
            and t.get("sha256") == item["sha256"]
            and t.get("size_bytes") == path.stat().st_size
            for t in receipt["transfers"]
        ):
            raise ValueError("retained upload evidence differs")
    hardware = json.loads((output / "hardware.json").read_text())["hardware"]["kind"]
    if hardware != "L4":
        raise ValueError("retained timing requires L4")
    return manifest, receipt


def collect(canary_output: Path, replay_outputs: list[Path]) -> dict:
    """Extract only cost/coverage fields from both completed, historical replay banks."""
    compatible_source()
    manifest, _ = verified_run(canary_output, measured_failure=True)
    report = json.loads((canary_output / "exports/performance.json").read_text())
    if manifest["watch6_efficacy"]["allocation"] != p.allocation():
        raise ValueError("retained canary allocation differs")
    banks = []
    if len(replay_outputs) != 2 or len(set(replay_outputs)) != 2:
        raise ValueError("requires two distinct replay banks")
    for output in replay_outputs:
        replay, receipt = verified_run(output)
        if replay["env"].get("WATCH_REPLAY_WORKERS") != "4":
            raise ValueError("replay timing requires measured W4")
        contract_path = output / "exports/sessions/contract.json"
        contract = json.loads(contract_path.read_text())
        operational_path = output / "exports/sessions/operational.json"
        operational = json.loads(operational_path.read_text())
        if (
            contract["horizon"] != REPLAY_HORIZON
            or contract["controller"] != "exact-stop-l2-prime"
            or contract["count"] != 12
            or operational["completed"] != 12
            or operational["total"] != 12
            or operational["skipped"]
            or operational["exclusions"]
            or len(operational["units"]) != 12
            or {u["index"] for u in operational["units"]} != set(range(12))
        ):
            raise ValueError("incomplete replay timing panel")
        banks.append(
            {
                "receipt": {
                    "path": str(output / "receipt.json"),
                    "sha256": digest(output / "receipt.json"),
                },
                "operational": {"path": str(operational_path), "sha256": digest(operational_path)},
                "contract": {"path": str(contract_path), "sha256": digest(contract_path)},
                "source_pin": receipt["commit"],
                "namespace": contract["namespace"],
                "workers": 4,
                "horizon": REPLAY_HORIZON,
                "units": [
                    {
                        "index": u["index"],
                        "seconds": u["seconds"],
                        "frames": u["stats"]["decisions"],
                        "search_roots": u["stats"]["search_roots"],
                    }
                    for u in operational["units"]
                ],
            }
        )
    return {
        "schema": "watch6-empirical-costs/v1",
        "science": p.identity(),
        "measurement_science": BASE_SCIENCE,
        "measurement_pin": BASE_PIN,
        "canary": report,
        "canary_receipt": {
            "path": str(canary_output / "receipt.json"),
            "sha256": digest(canary_output / "receipt.json"),
        },
        "banks": banks,
        "scope": (
            "Operational seconds only; replay totals include search, mirror, serialization "
            "and late main loops. No efficacy look."
        ),
    }


def verify_report_evidence(report: dict) -> None:
    """On the Mac, reconstruct every timing field from its verified original bytes."""
    actual = collect(
        Path(report["canary_receipt"]["path"]).parent,
        [Path(bank["receipt"]["path"]).parent for bank in report["banks"]],
    )
    if report != actual:
        raise ValueError("empirical evidence differs from verified historical costs")


def empirical_forecast(report: dict, frozen_layout: dict | None = None) -> dict:
    """Scale observed p95 full sessions plus bare main; retain independent 2x walls."""
    from research.watch6_search_efficacy.performance import validate_measurements

    if (
        report.get("science") != p.identity()
        or report.get("measurement_science") != BASE_SCIENCE
        or report.get("measurement_pin") != BASE_PIN
    ):
        raise ValueError("foreign empirical cost identity")
    validate_measurements(report["canary"], BASE_SCIENCE)
    banks = report["banks"]
    if len(banks) != 2 or len({b["namespace"] for b in banks}) != 2:
        raise ValueError("requires distinct historical replay banks")
    seconds = []
    for bank in banks:
        if bank["workers"] != 4 or bank["horizon"] != REPLAY_HORIZON or len(bank["units"]) != 12:
            raise ValueError("foreign empirical placement/coverage")
        if {u["index"] for u in bank["units"]} != set(range(12)):
            raise ValueError("duplicate/missing timing units")
        for unit in bank["units"]:
            value = unit["seconds"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                or unit["frames"] != REPLAY_HORIZON
                or type(unit["search_roots"]) is not int
                or not 0 <= unit["search_roots"] <= 1200
            ):
                raise ValueError("invalid empirical timing unit")
            seconds.append(value)
    ordered = sorted(seconds)
    position = (len(ordered) - 1) * 0.95  # linear p95, same convention as numpy percentile
    low = math.floor(position)
    p95 = ordered[low] + (position - low) * (ordered[math.ceil(position)] - ordered[low])
    profile = next(r for r in report["canary"]["profiles"] if r["workers"] == 4)
    bare_seconds = p.HORIZON * profile["main_seconds_per_frame"]["bare"]
    exact_seconds = p95 * p.HORIZON / REPLAY_HORIZON
    pair_seconds = bare_seconds + exact_seconds
    choices = []
    for shards in (3, 6, 12, 24, 48, 96):
        pairs = math.ceil(p.COUNT / shards)
        base = math.ceil(SETUP_MINUTES + MARGIN * math.ceil(pairs / 4) * pair_seconds / 60)
        if base <= MAX_BASE_MINUTES:
            choices.append((shards, base))
    if not choices:
        raise ValueError("NOT_READY empirical forecast exceeds shard wall with margin")
    shards, base = choices[0]
    if frozen_layout is not None:
        shards = frozen_layout["shards"]
        base = frozen_layout["base_minutes_per_shard"]
        required = dict(choices).get(shards)
        if (
            frozen_layout["workers"] != 4
            or type(base) is not int
            or required is None
            or not required <= base <= MAX_BASE_MINUTES
        ):
            raise ValueError("completion forecast exceeds frozen shard wall")
    return {
        "status": "EMPIRICAL_REPLAY_FORECAST",
        "workers": 4,
        "shards": shards,
        "pairs_per_shard": math.ceil(p.COUNT / shards),
        "observed_sessions": len(seconds),
        "observed_horizon": REPLAY_HORIZON,
        "observed_session_p95_seconds": p95,
        "observed_session_max_seconds": max(seconds),
        "bare_main_seconds": bare_seconds,
        "exact_stop_session_seconds_scaled": exact_seconds,
        "pair_seconds_upper_forecast": pair_seconds,
        "margin": MARGIN,
        "setup_minutes": SETUP_MINUTES,
        "base_minutes_per_shard": base,
        "effective_minutes": 2 * base,
        "aggregate_expected_runtime_hours": shards * base / 60,
        "calendar_hours_three_runtimes": math.ceil(shards / 3) * base / 60,
        "aggregate_effective_wall_hours": shards * 2 * base / 60,
        "calendar_wall_hours_three_runtimes": math.ceil(shards / 3) * 2 * base / 60,
        "evidence_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
        "limitations": (
            "Empirical forecast, not an all-state bound; H14400 may have heavier tails. "
            "Supervisor mechanically stops each shard at its doubled wall."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.canary_output, args.replay_output)
    publish_json(args.out, report)
    print(json.dumps(empirical_forecast(report), indent=2))


if __name__ == "__main__":
    main()
