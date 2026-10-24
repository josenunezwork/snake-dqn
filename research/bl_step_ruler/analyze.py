"""Step -1 one-look analyzer: STEP_M1_RULER.json (create-only).

Inputs are metadata only:
* dev units: verified counters-only step-ruler units (ledger union);
* fleet: Mac factory start/terminal JSON (completed parent IDs, elapsed,
  workers) and, for a live attempt, stat mtimes of session completion markers.
  Session/life payloads, sealed rate files and rescue counts are never opened;
* GPU: R2_ANALYSIS.json and, if verified at analysis time, R6_ANALYSIS.json.
The decision rule is frozen in PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from . import job

OLD_COLLECTOR = "61900af28932e47e22e332840fc4406559adb062"
# Start receipts name the admitted science pin; both run the OLD collector
# (D1024: research/bl_labels/D1024_RESUME_HANDOFF_20261024.md, "actual OLD collector").
START_PINS_RUNNING_OLD = {
    OLD_COLLECTOR: "D256 Mac runs: collector pin itself",
    "111c47818a097c57451386ef4f8a7009616f8f77": "D1024 science/admission pin; OLD collector, W6",
}
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
FLEET_STATE_DIRS = (
    ARTIFACTS / "bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01/d1024-state-mac-01",
    ARTIFACTS / "bl-labels-f0/mac-pack-02/d256-state-mac-01",
)
R2_ANALYSIS = ARTIFACTS / "big-lever/exp/gpu-r2-padding-20261024/R2_ANALYSIS.json"
R2_SHA256 = "0b253f33a0f59fd0992a8521bd1d7e79f5fea702f72f2ad3f79d91d14006158c"
R6_ANALYSIS = ARTIFACTS / "big-lever/exp/gpu-r6-escaling-20261024/R6_ANALYSIS.json"
R6_VERDICTS = {"KEEP_E4096", "DROP_E4096", "INCOMPLETE"}
GPU_E = {1024, 2048, 4096}
STREAMS = ("watch6", "gate-frozen", "gate-mixed", "gate-scripted")
STREAM_ROSTER = {"gate-frozen": "frozen", "gate-mixed": "mixed", "gate-scripted": "scripted"}
FACTORS = {"pessimistic": 0.35, "optimistic": 0.65}
BAR = 4.0
MIN_UNITS = 6
PRIMARY_WIDTH = 6
L4_PROXY_VCPU = 12
L4_SLOTS = 3
_PARENT = re.compile(r"-(watch6|gate-frozen|gate-mixed|gate-scripted)-\d+$")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_of(parent: str) -> str:
    match = _PARENT.search(parent)
    if match is None:
        raise ValueError(f"unrecognized parent id {parent}")
    return match.group(1)


def _sidecar_ok(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    marker = path.with_name(path.name + ".complete")
    if not (path.is_file() and sidecar.is_file() and marker.is_file()):
        return False
    return sidecar.read_text().split()[0] == sha256(path)


# --------------------------------------------------------------------- dev units


def dev_units(control: dict[str, Any], ledger: Path) -> dict[str, Any]:
    """Load verified counters-only units named by the Mac ledger union."""
    job.verify_unit(ledger)
    ids = job.load_ledger(ledger, control["namespace"], control["allocation"])
    rows = {row["id"]: row for row in json.loads(ledger.read_text())["units"]}
    by_id = {item["allocation_id"]: item for item in control["allocation"]}
    units = []
    for unit in sorted(ids):
        source = Path(rows[unit]["source"])
        if job.verify_unit(source) != rows[unit]["sha256"]:
            raise ValueError(f"unit bytes differ from ledger: {unit}")
        value = json.loads(source.read_text())
        item = by_id[unit]
        if (
            value.get("schema") != job.UNIT_SCHEMA
            or value.get("namespace") != control["namespace"]
            or value.get("allocation_id") != unit
            or value.get("stream") != item["stream"]
            or value.get("seed") != item["seed"]
        ):
            raise ValueError(f"foreign step-ruler unit {unit}")
        if any(key in value for key in job.SEALED_COUNTERS) or "rows" in value:
            raise ValueError(f"unit carries outcome-bearing fields: {unit}")
        ticks = value["tick_counters"]["ticks"]
        if value["world_steps"] != sum(ticks.values()):
            raise ValueError(f"unit world-step total differs: {unit}")
        units.append(value)
    per_stream: dict[str, dict[str, Any]] = {}
    for stream in STREAMS:
        chosen = [u for u in units if u["stream"] == stream and u["status"] == "COMPLETE"]
        if not chosen:
            continue
        steps = [u["world_steps"] for u in chosen]
        phases = {
            phase: sum(u["tick_counters"]["ticks"][phase] for u in chosen)
            for phase in ("play", "replay", "branch")
        }
        per_stream[stream] = {
            "n": len(chosen),
            "mean_world_steps_per_parent": statistics.fmean(steps),
            "min": min(steps),
            "max": max(steps),
            "phase_share": {k: v / max(1, sum(phases.values())) for k, v in phases.items()},
            "restores_per_parent": statistics.fmean(
                sum(u["tick_counters"]["restores"].values()) for u in chosen
            ),
            "roots_per_parent": statistics.fmean(u["work_counters"]["roots"] for u in chosen),
        }
    complete = [u for u in units if u["status"] == "COMPLETE"]
    worker_seconds = sum(u["elapsed_seconds"] for u in complete)
    return {
        "verified_units": len(units),
        "complete_units": len(complete),
        "missing_units": sorted(set(by_id) - ids),
        "per_stream": per_stream,
        "x86_worker_world_steps_per_second": (
            sum(u["world_steps"] for u in complete) / worker_seconds if worker_seconds else None
        ),
        "x86_platforms": sorted({u["provenance"]["platform"] for u in complete}),
    }


# ------------------------------------------------------------------------ fleet


def _attempt(start: Path) -> dict[str, Any]:
    if not _sidecar_ok(start):
        return {"start": str(start), "eligible": False, "reason": "unverified start receipt"}
    value = json.loads(start.read_text())
    base = {
        "start": str(start),
        "start_sha256": sha256(start),
        "workers": value.get("workers"),
        "output": value.get("output"),
    }
    attestation = value.get("platform_attestation", {})
    if (
        value.get("source_commit") not in START_PINS_RUNNING_OLD
        or attestation.get("platform") != "darwin-arm64"
    ):
        return {**base, "eligible": False, "reason": "foreign collector or platform"}
    terminal = start.with_name(start.name.replace(".start.json", ".terminal.json"))
    if terminal.exists():
        if not _sidecar_ok(terminal):
            return {**base, "eligible": False, "reason": "unverified terminal receipt"}
        end = json.loads(terminal.read_text())
        workers = end.get("workers")
        if end.get("collector_source_commit", OLD_COLLECTOR) != OLD_COLLECTOR:
            return {**base, "eligible": False, "reason": "terminal collector is not OLD"}
        if (
            workers != value.get("workers")
            or end.get("operational_width_fallback") is not None
            or end.get("admission_workers_final", workers) != workers
        ):
            return {**base, "eligible": False, "reason": "width changed within attempt"}
        parents = list(end.get("completed_parents") or [])
        elapsed = float(end["elapsed_seconds"])
        kind = "terminal"
        evidence = {"terminal": str(terminal), "terminal_sha256": sha256(terminal)}
    else:
        lives = Path(value["output"]) / "lives"
        started = float(value["started_epoch"])
        stamps = {}
        if lives.is_dir():
            for marker in lives.glob("*--session.json.complete"):
                mtime = marker.stat().st_mtime  # stat only; payloads never opened
                if mtime >= started:
                    stamps[marker.name.removesuffix("--session.json.complete")] = mtime
        parents = sorted(stamps)
        elapsed = (max(stamps.values()) - started) if stamps else 0.0
        kind = "live"
        evidence = {"session_markers": len(stamps), "cut_epoch": max(stamps.values(), default=None)}
    if len(parents) != len(set(parents)) or elapsed <= 0 or not parents:
        return {
            **base,
            "eligible": False,
            "reason": "no completed parents or zero elapsed",
            "kind": kind,
        }
    counts = {stream: 0 for stream in STREAMS}
    for parent in parents:
        counts[stream_of(parent)] += 1
    return {
        **base,
        **evidence,
        "eligible": True,
        "kind": kind,
        "completed_by_stream": counts,
        "elapsed_seconds": elapsed,
    }


def fleet(state_dirs: tuple[Path, ...] | list[Path]) -> list[dict[str, Any]]:
    attempts = []
    for directory in state_dirs:
        for start in sorted(Path(directory).glob("attempt-*.start.json")):
            attempts.append(_attempt(start))
    return attempts


def fleet_rate(attempts: list[dict[str, Any]], k: dict[str, float], width: int) -> dict[str, Any]:
    chosen = [a for a in attempts if a["eligible"] and a["workers"] == width]
    if not chosen:
        return {"width": width, "attempts": 0, "world_steps_per_second": None}
    missing = sorted(
        {s for a in chosen for s, n in a["completed_by_stream"].items() if n and s not in k}
    )
    if missing:
        return {
            "width": width,
            "attempts": len(chosen),
            "world_steps_per_second": None,
            "missing_k": missing,
        }
    steps = sum(n * k[s] for a in chosen for s, n in a["completed_by_stream"].items() if n)
    elapsed = sum(a["elapsed_seconds"] for a in chosen)
    parents = {s: sum(a["completed_by_stream"][s] for a in chosen) for s in STREAMS}
    return {
        "width": width,
        "attempts": len(chosen),
        "elapsed_seconds": elapsed,
        "completed_parents": parents,
        "parents_per_hour": sum(parents.values()) / elapsed * 3600,
        "world_steps_per_second": steps / elapsed,
    }


# -------------------------------------------------------------------------- GPU


def gpu_cells(r2: Path = R2_ANALYSIS, r6: Path = R6_ANALYSIS) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    if sha256(r2) != R2_SHA256:
        raise ValueError("R2_ANALYSIS.json bytes differ from the ruling's hash")
    value = json.loads(r2.read_text())
    if value.get("verdict") != "R2_ADVANCE":
        raise ValueError("R2 is not R2_ADVANCE")
    for roster, rates in value["e1024_i_prime_continuous_world_steps_per_second"].items():
        cells.append(
            {"source": "R2", "roster": roster, "E": 1024, "rate": statistics.median(rates)}
        )
    sources["R2"] = {"path": str(r2), "sha256": R2_SHA256, "verdict": value["verdict"]}
    if r6.exists():
        if not _sidecar_ok(r6):
            sources["R6"] = {"path": str(r6), "used": False, "reason": "unverified sidecars"}
        else:
            r6v = json.loads(r6.read_text())
            if r6v.get("verdict") not in R6_VERDICTS or r6v.get("mode") != "main":
                sources["R6"] = {"path": str(r6), "used": False, "verdict": r6v.get("verdict")}
            else:
                used = 0
                for roster, sizes in r6v.get("cells", {}).items():
                    for size, cell in sizes.items():
                        if int(size) in GPU_E and cell.get("exact") is True:
                            cells.append(
                                {
                                    "source": "R6",
                                    "roster": roster,
                                    "E": int(size),
                                    "rate": float(cell["median_rate"]),
                                }
                            )
                            used += 1
                sources["R6"] = {
                    "path": str(r6),
                    "sha256": sha256(r6),
                    "used": True,
                    "verdict": r6v["verdict"],
                    "cells_used": used,
                }
    else:
        sources["R6"] = {"path": str(r6), "used": False, "reason": "absent at analysis time"}
    return {"cells": cells, "sources": sources}


# ---------------------------------------------------------------------- decision


def decide(dev: dict[str, Any], attempts: list[dict[str, Any]], gpu: dict[str, Any]) -> dict:
    k = {s: v["mean_world_steps_per_parent"] for s, v in dev["per_stream"].items()}
    widths = {w: fleet_rate(attempts, k, w) for w in (6, 12)}
    reasons = []
    if dev["complete_units"] < MIN_UNITS:
        reasons.append(f"fewer than {MIN_UNITS} complete dev units")
    if set(k) != set(STREAMS):
        reasons.append("a stream has no complete dev unit")
    primary = widths[PRIMARY_WIDTH]["world_steps_per_second"]
    if not primary:
        reasons.append("no eligible Mac W6 fleet rate")
    if not gpu["cells"]:
        reasons.append("no eligible GPU cell")
    result: dict[str, Any] = {"fleet": widths, "steps_per_parent": k}
    if reasons:
        return {**result, "verdict": "INCOMPLETE", "reasons": reasons}
    best = max(gpu["cells"], key=lambda cell: cell["rate"])
    ratios = {name: f * best["rate"] / primary for name, f in FACTORS.items()}
    verdict = "S0_ADMITTED" if ratios["optimistic"] >= BAR else "CLOSE_NO_BOUND"
    x86 = dev["x86_worker_world_steps_per_second"]
    l4_slot = L4_PROXY_VCPU * x86 if x86 else None
    sensitivity: dict[str, Any] = {}
    for width, rate in widths.items():
        if not rate["world_steps_per_second"]:
            continue
        base = rate["world_steps_per_second"]
        sensitivity[f"mac_w{width}"] = {n: f * best["rate"] / base for n, f in FACTORS.items()}
        if l4_slot:
            plus = base + L4_SLOTS * l4_slot
            sensitivity[f"mac_w{width}_plus_3_l4_proxy"] = {
                n: f * best["rate"] / plus for n, f in FACTORS.items()
            }
    by_roster = {}
    for roster in ("frozen", "mixed", "scripted"):
        rates = [c["rate"] for c in gpu["cells"] if c["roster"] == roster]
        if rates:
            by_roster[roster] = {n: f * max(rates) / primary for n, f in FACTORS.items()}
    flags = []
    w12 = sensitivity.get("mac_w12")
    if w12 and w12["optimistic"] < BAR:
        flags.append("W12_BELOW_4")
    return {
        **result,
        "verdict": verdict,
        "gpu_best_cell": best,
        "t_gpu_eff": {n: f * best["rate"] for n, f in FACTORS.items()},
        "t_fleet_primary_w6": primary,
        "ratios_primary": ratios,
        "pessimistic": (
            "PESSIMISTIC_PASS" if ratios["pessimistic"] >= BAR else "PESSIMISTIC_BELOW_4"
        ),
        "sensitivity": sensitivity,
        "sensitivity_by_roster_gpu_rate": by_roster,
        "l4_slot_proxy_world_steps_per_second": l4_slot,
        "l4_proxy_basis": "PROXY: 12 x measured x86 per-worker rate; no measured L4 F0 rate",
        "flags": flags,
    }


def runtime_statuses(control: dict[str, Any], results: list[Path]) -> dict[str, Any]:
    """Every attempt's verified runtime-result; any INVALID parent invalidates the ruler."""
    rows = []
    for path in results:
        job.verify_unit(path)
        value = json.loads(Path(path).read_text())
        if value.get("namespace") != control["namespace"]:
            raise ValueError(f"foreign runtime result {path}")
        rows.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "status": value["status"],
                "invalid": value.get("invalid", []),
            }
        )
    return {
        "attempts": rows,
        "invalid": any(r["status"] == "INVALID" or r["invalid"] for r in rows),
    }


def analyze(
    control: Path, ledger: Path, out: Path, *, protocol: Path, results: list[Path]
) -> dict[str, Any]:
    from research.bl_labels.exports import publish_json

    if out.exists():
        raise FileExistsError(out)
    if not results:
        raise ValueError("every attempt's runtime-result.json is required")
    value = json.loads(control.read_text())
    runtime = runtime_statuses(value, results)
    dev = dev_units(value, ledger)
    attempts = fleet(FLEET_STATE_DIRS)
    gpu = gpu_cells()
    decision = decide(dev, attempts, gpu)
    if runtime["invalid"]:
        decision = {**decision, "verdict": "INVALID", "reasons": ["an attempt reported INVALID"]}
    result = {
        "schema": "bl-step-ruler-analysis/v1",
        "claim": "premise test only; no label, gate, fit, serving or release claim",
        "protocol": {"path": str(protocol), "sha256": sha256(protocol)},
        "inputs": {
            "control_sha256": sha256(control),
            "ledger_sha256": sha256(ledger),
            "fleet_state_dirs": [str(p) for p in FLEET_STATE_DIRS],
            "gpu": gpu["sources"],
        },
        "runtime_attempts": runtime,
        "dev_units": dev,
        "fleet_attempts": attempts,
        "gpu_cells": gpu["cells"],
        **decision,
    }
    publish_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-result", type=Path, action="append", required=True)
    args = parser.parse_args()
    value = analyze(
        args.control, args.ledger, args.out, protocol=args.protocol, results=args.runtime_result
    )
    print(
        json.dumps(
            {
                key: value.get(key)
                for key in ("verdict", "ratios_primary", "pessimistic", "flags", "reasons")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
