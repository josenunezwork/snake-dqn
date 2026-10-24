"""I0 sibling admission under WAIVE_V8_SELECTION_FOR_FROZEN_I0 and the small-bench path.

Host-only. The frozen runtime 968f0276, its qualified broker, the old
``gpu-integration-i0-20261024-03`` package, its canary receipt and review stay
byte-identical. This module never allocates, never queries the provider and never
edits a receipt. It replaces only two host-side admission inputs, as Astra's
2026-10-24 ruling authorizes:

* the V8 SELECT_G/SELECT_GV prerequisite becomes an explicit hash-bound waiver
  (never a synthetic selection receipt);
* the main sizing is re-derived from actual call counts, with the old 98-minute
  forecast reproduced beside it, and the I0 cleanup/envelope arithmetic is checked.

Both derivations are Mac-only admission evidence. Nothing is added to the runtime
upload inventory. When sizing or cleanup does not fit, ``prepare`` publishes the
audited HOLD record and writes no admissible manifest or config.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/Users/josenunez/Projects/ml")
RUN = BASE / "snake-dqn-gpuint-qualified-03.noindex"
PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"
PY = BASE / "snake-dqn/venv/bin/python"
EXP = BASE / "snake-dqn-artifacts/big-lever/exp"
OPS = BASE / "snake-dqn-artifacts/ops"
OLD = EXP / "gpu-integration-i0-20261024-03"
PACKET = OLD / "packet"
CANARY = PACKET / "canary-output"
RECEIPT = CANARY / "receipt.json"
REVIEW = CANARY / "CODEX_V_MAC_REVIEW.json"
REVIEW_SHA = "75e29939a36e1819c6f20b5de620afb791ad089d984373721af118a98a2b5f7d"
TEMPLATE = PACKET / "integration-main-template.json"
PREFLIGHT = OLD / "mac-preflight/preflight.json"
WAIVER = EXP / "JUDGMENT_I0_UNBLOCK_20261024.md"
WAIVER_SHA = "3dc6cb50ad36da031d189b70ab344441533109e9d99f1505cac17a81934830a7"
FAST_PATH = EXP / "JUDGMENT_SMALL_BENCH_ADMISSION_20261024.md"
FAST_PATH_SHA = "1441a0b81600eb5f55fa85336c72e491cac7ee82203daa4ab66c8a331b84ca86"
SPRINT = EXP / "small-bench-sprint-20261024"
REGISTER = SPRINT / "register.jsonl"
V8_CONFIG = SPRINT / "v8-config-release-tail-02.json"
V8_CLOSURE_ID = "v8-split-initial-main-closed-unneeded-for-i0-20261024"
ART = EXP / "gpu-integration-i0-main-waiver-20261024-01"
AUDIT = ART / "I0_SIZING_CLEANUP_AUDIT.json"
MANIFEST = ART / "integration-main.json"
CONFIG = ART / "i0-small-bench-config.json"
JOB = "gpu-integration-i0-main-968f02765f7c-01"
METADATA_SHA = "350f4de106285510fcb611fd974ac2b3801bc28b104cdddd851987937bdcb327"
SUPERVISOR_SHA = "263eeecc556ae02057d87c351e7c83b9fb3175f55f8b1d823982bf53c564e853"
FLAGS = {
    "GPU_SIM_SPIKE_COMMIT": PIN,
    "GPU_SIM_VECTOR_FLOOD": "static",
    "GPU_SIM_VECTOR_GEOMETRY": "native",
    "GPU_SIM_V8_INDEXED": "0",
    "INTEGRATION_BENCH_DEVICE": "cuda",
    "INTEGRATION_BENCH_SECONDS": "5400",
    "INTEGRATION_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
    "INTEGRATION_BENCH_SMOKE": "0",
    "INTEGRATION_BENCH_CANARY": "0",
}
# Registered limits; none is raised here.
ESTIMATE_CEILING_MINUTES = 60
MAIN_CAP = {"units": 20.0, "wall_minutes": 120.0}
I0_ENVELOPE = {"units": 24.0, "wall_minutes": 144.0}
SMALL_BENCH_SCOPE_UNITS = 25.0
PRODUCER_DEADLINE_SECONDS = 5400.0
# Explicit, unproved uncertainty margin for progressing-state growth across the
# 35 progressing E1024 ticks; the canary measured only post-reset single ticks.
PROGRESSING_STATE_MARGIN = 1.25
I0_HISTORY_IDS = (
    "gpu-integration-i0-canary-968f02765f7c-01",
    "native-compile-l4-968f0276-integration03",
)


class Hold(ValueError):
    """A concrete refusal; the package stays HOLD, never silently relaxed."""


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require_hash(path: Path, digest: str, label: str) -> dict[str, str]:
    if not Path(path).is_file() or Path(path).is_symlink() or sha(path) != digest:
        raise Hold(f"{label} identity differs: {path}")
    return {"path": str(path), "sha256": digest}


# --------------------------------------------------------------------------- sizing


def old_forecast(e64_pair: float, unit_seconds: list[float], elapsed: float) -> dict[str, Any]:
    """Reproduce the frozen ``integration_pack._main_forecast`` arithmetic exactly."""
    ordinary = 3 * 3 * sum(world / 64 for world in (6, 64, 256, 1024))
    continuity = 3 * (3 * 32 + 3) * (1024 / 64)
    setup = max(120.0, float(elapsed) - sum(unit_seconds))
    seconds = setup + e64_pair * (ordinary + continuity)
    return {
        "method": "frozen integration_pack._main_forecast (plain+diagnostic pair per tick)",
        "max_e64_pair_seconds": e64_pair,
        "ordinary_pair_equivalents": ordinary,
        "continuity_pair_equivalents": continuity,
        "setup_seconds": setup,
        "forecast_seconds": seconds,
        "forecast_minutes_exact": seconds / 60,
        "forecast_minutes": math.ceil(seconds / 60),
    }


def native_lower_bound(logs: list[str]) -> dict[str, Any]:
    """Earliest provable end of native setup from the producer's own ordered log.

    The heartbeat thread prints ``elapsed_seconds`` every 110 s. A pytest session
    whose header follows that line started no earlier than the heartbeat, so native
    setup ended no earlier than heartbeat + that session's reported duration.
    """
    best: dict[str, Any] | None = None
    for text in logs:
        for beat in re.finditer(r'"i0_heartbeat", "elapsed_seconds": ([0-9.]+)', text):
            after = text[beat.end() :]
            session = re.search(r"test session starts.*?passed.*? in ([0-9.]+)s", after, re.S)
            if session is None:
                continue
            bound = float(beat[1]) + float(session[1])
            if best is None or bound > best["seconds"]:
                best = {
                    "seconds": bound,
                    "heartbeat_seconds": float(beat[1]),
                    "following_session_seconds": float(session[1]),
                }
    if best is None:
        raise Hold("canary log lacks an ordered native-setup lower bound")
    return best


def revised_forecast(
    *,
    arms: dict[str, dict[str, dict[str, list[float]]]],
    receipt_elapsed: float,
    producer_elapsed: float,
    native_bound: float,
    download_bytes: float,
    download_seconds: float,
    max_export_bytes: float,
    warmups: int,
    repeats: int,
    continuous_ticks: int,
    margin: float = PROGRESSING_STATE_MARGIN,
) -> dict[str, Any]:
    """Charge the main's actual call counts from measured canary service times.

    ``arms[roster][arm]`` holds the canary's plain and diagnostic call seconds.
    A call is ``reset + full tick`` (both are inside the timed region of _tick).
    Every roster is charged at the slowest roster's cost; E scaling is linear
    above E64 and never below one E64 call at E6.
    """
    if len(arms) != 3 or any(set(value) != {"R", "I"} for value in arms.values()):
        raise Hold("canary timing must cover three rosters x R/I")
    pair_by_roster = {}
    for roster, value in arms.items():
        total = 0.0
        for arm in ("R", "I"):
            samples = value[arm]["plain"] + value[arm]["diagnostic"]
            if not samples or not all(math.isfinite(x) and x > 0 for x in samples):
                raise Hold("canary timing rows are malformed")
            total += max(samples)
        pair_by_roster[roster] = total
    call_pair = max(pair_by_roster.values())

    def scale(world: int) -> float:
        return max(1.0, world / 64)

    worlds = (1024, 256, 64, 6)
    calls_per_cell = warmups + repeats + repeats  # warmups + plain + diagnostic
    ordinary_pairs = 3 * sum(calls_per_cell * scale(world) for world in worlds)
    # Per roster and arm: reset + warmup ticks, then each repeat resets and runs
    # the progressing block. Every tick and every standalone reset is charged as
    # a full measured call (reset + tick).
    continuity_calls = 1 + warmups + repeats * (1 + continuous_ticks)
    continuity_pairs = 3 * continuity_calls * scale(1024)
    ordinary_seconds = call_pair * ordinary_pairs
    continuity_seconds = call_pair * continuity_pairs * margin
    # Non-tick per-cell work (construction, combined parity proof, digests,
    # publication, trace) is bounded by the canary producer time after native
    # setup, attributed to its three E64 cells and scaled linearly with E.
    post_native = producer_elapsed - native_bound
    if post_native <= 0:
        raise Hold("canary producer timing contradicts native lower bound")
    per_cell_overhead = post_native / 3
    overhead_seconds = 3 * sum(per_cell_overhead * scale(world) for world in worlds)
    fixed_producer = producer_elapsed  # native + register + three E64 cells, double counted
    setup_outside_producer = receipt_elapsed - producer_elapsed
    throughput = download_bytes / download_seconds
    export_seconds = max_export_bytes / throughput
    components = {
        "setup_outside_producer_seconds": setup_outside_producer,
        "fixed_canary_producer_seconds": fixed_producer,
        "ordinary_cell_call_seconds": ordinary_seconds,
        "continuity_call_seconds_with_margin": continuity_seconds,
        "per_cell_overhead_seconds": overhead_seconds,
        "export_allowance_seconds": export_seconds,
    }
    seconds = sum(components.values())
    floor = (
        setup_outside_producer + fixed_producer + ordinary_seconds + call_pair * continuity_pairs
    )
    return {
        "method": "actual call counts x slowest-roster measured call, linear E>=64, unit floor E6",
        "call_pair_seconds_by_roster": pair_by_roster,
        "charged_call_pair_seconds": call_pair,
        "calls_per_arm_per_ordinary_cell": calls_per_cell,
        "ordinary_call_pair_equivalents": ordinary_pairs,
        "continuity_calls_per_arm_per_roster": continuity_calls,
        "continuity_call_pair_equivalents": continuity_pairs,
        "progressing_state_margin": margin,
        "native_setup_lower_bound_seconds": native_bound,
        "post_native_canary_seconds_bound": post_native,
        "per_e64_cell_overhead_bound_seconds": per_cell_overhead,
        "download_throughput_bytes_per_second": throughput,
        "components": components,
        "forecast_seconds": seconds,
        "forecast_minutes_exact": seconds / 60,
        "forecast_minutes": math.ceil(seconds / 60),
        "floor_without_overhead_margin_or_exports_seconds": floor,
        "floor_minutes": math.ceil(floor / 60),
        "producer_deadline_seconds": PRODUCER_DEADLINE_SECONDS,
    }


def sizing_verdict(revised: dict[str, Any]) -> list[str]:
    reasons = []
    minutes = revised["forecast_minutes"]
    if minutes > ESTIMATE_CEILING_MINUTES:
        reasons.append(
            f"HOLD_SIZING: revised estimate {minutes} min exceeds registered "
            f"{ESTIMATE_CEILING_MINUTES} min (floor {revised['floor_minutes']} min)"
        )
    if 2 * minutes > MAIN_CAP["wall_minutes"]:
        reasons.append(
            f"HOLD_SUPERVISOR: timeout {MAIN_CAP['wall_minutes']:g} < 2 x estimate {minutes}"
        )
    if revised["forecast_seconds"] > PRODUCER_DEADLINE_SECONDS:
        reasons.append("HOLD_DEADLINE: estimate exceeds the frozen 5400 s producer deadline")
    return reasons


# ------------------------------------------------------------------------- envelope


def envelope(
    history: list[dict[str, Any]], tail: dict[str, float], cap: dict[str, float] = MAIN_CAP
) -> dict[str, Any]:
    spent = {
        key: sum(float(row["pair"][key]) for row in history) for key in ("units", "wall_minutes")
    }
    need = {key: spent[key] + cap[key] + tail[key] for key in ("units", "wall_minutes")}
    reasons = []
    for key in ("units", "wall_minutes"):
        if need[key] > I0_ENVELOPE[key] + 1e-9:
            reasons.append(
                f"HOLD_CLEANUP: I0 history {spent[key]:.6g} + main {cap[key]:g} + "
                f"outside-cap tail {tail[key]:g} = {need[key]:.6g} {key} > I0 "
                f"{I0_ENVELOPE[key]:g}"
            )
    if cap["units"] + tail["units"] + spent["units"] > SMALL_BENCH_SCOPE_UNITS + 1e-9:
        reasons.append("HOLD_SCOPE: whole logical I0 packet exceeds the 25-unit fast-path scope")
    return {
        "history": history,
        "history_total": spent,
        "main_cap": cap,
        "outside_tail": tail,
        "required_total": need,
        "i0_envelope": I0_ENVELOPE,
        "headroom_for_tail": {
            key: I0_ENVELOPE[key] - spent[key] - cap[key] for key in ("units", "wall_minutes")
        },
        "reasons": reasons,
    }


def i0_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    charges = {row["id"]: row for row in rows[0]["historical_charges"]}
    found = [charges[name] for name in I0_HISTORY_IDS if name in charges]
    if len(found) != len(I0_HISTORY_IDS):
        raise Hold("register lacks the I0 canary/compile historical charges")
    others = [
        row["id"]
        for row in rows[0]["historical_charges"]
        if ("integration-i0" in row["id"] or "968f0276" in row["id"])
        and row["id"] not in I0_HISTORY_IDS
    ]
    if others:
        raise Hold(f"unexplained I0 historical identities: {others}")
    return [{"id": row["id"], "pair": row["pair"], "basis": row["basis"]} for row in found]


def unrecorded_i0_identities(manifests: Path = OPS / "manifests") -> list[str]:
    known = {f"{name}.json" for name in ("gpu-integration-i0-canary-968f0276",)} | {
        "native-compile-l4-968f0276-integration03.json"
    }
    seen = []
    for path in sorted(manifests.glob("*.json")):
        if ".live." in path.name:
            continue
        if ("integration-i0" in path.name or "968f0276" in path.name) and path.name not in known:
            seen.append(path.name)
    return seen


# ------------------------------------------------------------------------ identity


def check_template(template: dict[str, Any], canary: dict[str, Any]) -> None:
    """Frozen flags/pin/upload inventory; any change needs new qualification."""
    if template.get("commit") != PIN or canary.get("commit") != PIN:
        raise Hold("runtime pin differs from 968f0276")
    if template.get("job_id") != JOB or template.get("scientific_admission") is not False:
        raise Hold("main template identity differs")
    for key, value in FLAGS.items():
        if template["env"].get(key) != value:
            raise Hold(f"executable flag differs: {key}")
    if template["env"].get("METADATA_ARCHIVE_SHA256") != METADATA_SHA:
        raise Hold("metadata archive identity differs")
    if any(key.startswith("SNAKE_COLAB_CANARY") for key in template["env"]):
        raise Hold("main template enables canary instrumentation")

    def inventory(value: dict[str, Any]) -> list[tuple[str, str]]:
        return sorted((row["dest"], row["sha256"]) for row in value["inputs"])

    if inventory(template) != inventory(canary):
        raise Hold("main upload inventory differs from the qualified canary")
    for row in template["inputs"]:
        if not Path(row["path"]).is_file() or sha(Path(row["path"])) != row["sha256"]:
            raise Hold(f"upload source bytes differ: {row['path']}")
    if (
        template["timeout_minutes"] != MAIN_CAP["wall_minutes"]
        or template["budget"]["unit_cap"] != MAIN_CAP["units"]
        or not 0 < template["budget"]["rate_units_per_hour"] <= 10
        or template["requirements"]["gpu"] != "A100"
        or template["requirements"]["allowed_gpus"] != ["A100"]
    ):
        raise Hold("main cap/rate/hardware identity differs")
    for key in ("source_paths", "transport_pin", "mac_preflight", "component_admission", "script"):
        if template.get(key) != canary.get(key):
            raise Hold(f"main {key} differs from qualified canary")


def no_main_executed(output_dir: str) -> None:
    names = [JOB, "gpu-integration-i0-main-968f0276"]
    paths = [Path(output_dir), PACKET / "main-output", PACKET / "state-main", ART / "main-output"]
    for name in names:
        paths += [
            OPS / "dispatch" / name,
            OPS / "logs" / f"{name}.log",
            OPS / "manifests" / f"{name}.json",
        ]
    used = [str(path) for path in paths if path.exists()]
    if used:
        raise Hold(f"I0 main identity already used or executed: {used}")


def check_review() -> dict[str, str]:
    bound = require_hash(REVIEW, REVIEW_SHA, "independent canary Mac review")
    review = json.loads(REVIEW.read_text())
    if review.get("verdict") != "PASS_CANARY" or review.get("runtime_pin") != PIN:
        raise Hold("independent canary review is not PASS_CANARY at the runtime pin")
    return bound


def check_v8_closed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("id") == V8_CLOSURE_ID:
            if (
                row.get("status") != "CLOSED_UNNEEDED_FOR_I0"
                or row.get("ruling", {}).get("sha256") != WAIVER_SHA
            ):
                raise Hold("V8 closure row does not bind this waiver")
            return {"revision": row["revision"], "row_hash": row["row_hash"]}
    raise Hold("V8 split main closure is not recorded in the standing register")


def check_tail() -> dict[str, Any]:
    """The I0 broker is the same qualified supervisor whose tail was audited for V8."""
    if sha(RUN / "research/colab_cli/supervisor.py") != SUPERVISOR_SHA:
        raise Hold("I0 supervisor differs from the audited broker")
    v8 = json.loads(V8_CONFIG.read_text())
    if v8["outside_tail_source"]["sha256"] != SUPERVISOR_SHA:
        raise Hold("audited outside-cap tail source differs")
    return {
        "pair": {key: float(v8["outside_tail"][key]) for key in ("units", "wall_minutes")},
        "basis": v8["outside_tail_basis"],
        "source": binding(V8_CONFIG),
        "supervisor_sha256": SUPERVISOR_SHA,
        "reduced_tail_evidence": None,
    }


# ------------------------------------------------------------------------- runtime


def runtime_modules() -> dict[str, Any]:
    """Import the frozen runtime's own modules; no module globals are patched."""
    if str(RUN) not in sys.path[:1]:
        raise Hold("run through tools/i0-small-bench.sh (runtime first on PYTHONPATH)")
    from research.gpu_sim_spike import integration_bench, integration_pack, integration_units

    if Path(integration_pack.__file__).resolve().parents[2] != RUN:
        raise Hold("runtime modules did not resolve from the qualified checkout")
    return {"pack": integration_pack, "bench": integration_bench, "units": integration_units}


def canary_timing(units_module: Any) -> dict[str, Any]:
    exports = CANARY / "exports"
    rows = units_module.load_ledger(
        exports / "verified-ledger.json",
        units_module.namespace(PIN, canary=True),
        units_module.allocation(canary=True),
        exports=exports,
    )
    arms, unit_seconds = {}, []
    for unit, row in rows.items():
        roster = unit.split("-")[1]
        arms[roster] = {}
        total = 0.0
        for arm in ("R", "I"):
            value = row.get("arms", {}).get(arm, {})
            plain = list(value.get("plain_seconds", []))
            diagnostic = [stage.get("wall_seconds", 0.0) for stage in value.get("stages", [])]
            arms[roster][arm] = {"plain": plain, "diagnostic": diagnostic}
            total += sum(plain) + sum(diagnostic)
        unit_seconds.append(total)
    return {"arms": arms, "unit_seconds": unit_seconds}


def sizing(modules: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    receipt = json.loads(RECEIPT.read_text())
    producer = json.loads((CANARY / "exports/integration-bench.json").read_text())
    timing = canary_timing(modules["units"])
    old = old_forecast(
        max(timing["unit_seconds"]), timing["unit_seconds"], receipt["elapsed_seconds"]
    )
    logs = [
        json.loads(path.read_text()).get("job_log_tail", "")
        for path in sorted(CANARY.glob("poll-*.json"))
    ]
    native = native_lower_bound(logs)
    downloads = [
        row
        for row in receipt["transfers"]
        if row.get("direction") == "download" and row.get("status") == "verified"
    ]
    bench = modules["bench"]
    revised = revised_forecast(
        arms=timing["arms"],
        receipt_elapsed=receipt["elapsed_seconds"],
        producer_elapsed=producer["elapsed_seconds"],
        native_bound=native["seconds"],
        download_bytes=sum(row["size_bytes"] for row in downloads),
        download_seconds=sum(row["elapsed_seconds"] for row in downloads),
        max_export_bytes=template["max_export_bytes"],
        warmups=bench.WARMUPS,
        repeats=bench.REPEATS,
        continuous_ticks=bench.CONTINUOUS_TICKS,
    )
    revised["native_lower_bound_derivation"] = native
    return {"old": old, "revised": revised, "reasons": sizing_verdict(revised)}


def candidate_manifest(
    template: dict[str, Any], canary_binding: dict[str, Any], estimate: int, output: Path
) -> dict[str, Any]:
    """Template plus the same-pin canary receipt binding and honest estimate only."""
    value = copy.deepcopy(template)
    value.pop("admission_reason", None)
    value["scientific_admission"] = True
    value["canary"] = {"receipt": canary_binding}
    value["requirements"]["est_minutes"] = estimate
    value["output_dir"] = str(output)
    return value


def offline() -> dict[str, Any]:
    """Every host-side check; returns ADMISSIBLE or HOLD with concrete reasons."""
    from tools import small_bench_register as register

    checks: dict[str, Any] = {
        "waiver": require_hash(WAIVER, WAIVER_SHA, "I0 unblock waiver ruling"),
        "fast_path": require_hash(FAST_PATH, FAST_PATH_SHA, "small-bench ruling"),
        "canary_review": check_review(),
    }
    modules = runtime_modules()
    pack = modules["pack"]
    pack.require_source_pin(PIN)
    preflight = pack.preflight_binding(PREFLIGHT, PIN)
    admission = pack._admission(PIN)  # retains historical v8_selection: PENDING
    checks["tracked_admission_v8_selection"] = admission["v8_selection"]
    canary_binding = pack._canary_binding(RECEIPT, PIN, preflight)
    if canary_binding["analysis"].get("verdict") != "PASS_CANARY":
        raise Hold("independent canary reducer did not pass")
    checks["canary_receipt"] = {
        "path": canary_binding["path"],
        "sha256": canary_binding["sha256"],
        "reducer_verdict": "PASS_CANARY",
    }
    template = json.loads(TEMPLATE.read_text())
    canary = json.loads((PACKET / "integration-canary.json").read_text())
    check_template(template, canary)
    if template["mac_preflight"] != preflight:
        raise Hold("main template preflight binding differs")
    checks["template"] = binding(TEMPLATE)
    ledger = modules["units"].load_ledger(
        PACKET / "verified-ledger.json",
        modules["units"].namespace(PIN),
        modules["units"].allocation(),
    )
    if ledger:
        raise Hold("main ledger is not empty")
    checks["main_ledger_units"] = 0
    no_main_executed(str(ART / "main-output"))
    unknown = unrecorded_i0_identities()
    if unknown:
        raise Hold(f"unrecorded I0 operational identities: {unknown}")
    rows = register.read(REGISTER)
    checks["v8_closure"] = check_v8_closed(rows)
    sized = sizing(modules, template)
    cleanup = envelope(i0_history(rows), check_tail()["pair"])
    cleanup["tail"] = check_tail()
    reasons = sized["reasons"] + cleanup["reasons"]
    return {
        "schema": "i0-waiver-admission-audit/v1",
        "status": "ADMISSIBLE" if not reasons else "HOLD",
        "reasons": reasons,
        "runtime_pin": PIN,
        "job_id": JOB,
        "checks": checks,
        "sizing": sized,
        "cleanup": cleanup,
        "register": {"revision": rows[-1]["revision"], "row_hash": rows[-1]["row_hash"]},
        "allocations": 0,
        "provider_queries": 0,
        "_canary_binding": canary_binding,
        "_template": template,
    }


def governed_dry_run(manifest: dict[str, Any], label: str) -> dict[str, Any]:
    """Run the unchanged governed wrapper's dry-run on a throwaway manifest copy."""
    with tempfile.TemporaryDirectory(prefix=f"i0-{label}-") as directory:
        folder = Path(directory)
        value = dict(manifest, output_dir=str(folder / "main-output"))
        path = folder / "integration-main.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result = subprocess.run(
            [
                str(ROOT / "tools/ops/dispatch.sh"),
                JOB,
                str(path),
                "--broker",
                str(RUN),
                "--cwd",
                str(RUN),
                "--state-dir",
                str(folder / "state-main"),
                "--dispatch-txt",
                str(folder / "DISPATCH.txt"),
                "--dry-run",
            ],
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        # Same qualified validation in-process, only to name the refusal reason.
        from research.colab_cli import supervisor

        try:
            supervisor.validate(json.loads(path.read_text()), repo=RUN)
            reason = None
        except Exception as error:  # noqa: BLE001 - report the qualified refusal text
            reason = f"{type(error).__name__}: {error}"
    output = (result.stdout + result.stderr).strip().splitlines()
    return {
        "label": label,
        "estimate_minutes": manifest["requirements"]["est_minutes"],
        "returncode": result.returncode,
        "status": "PASS_DRY_RUN" if result.returncode == 0 else "REFUSED",
        "supervisor_validate_reason": reason,
        "tail": output[-3:],
    }


def dry_run() -> dict[str, Any]:
    """Full offline prepare path plus governed dispatch dry-runs; zero allocations."""
    audit = offline()
    canary_binding, template = audit.pop("_canary_binding"), audit.pop("_template")
    estimate = audit["sizing"]["revised"]["forecast_minutes"]
    honest = candidate_manifest(template, canary_binding, estimate, ART / "main-output")
    runs = [governed_dry_run(honest, "revised-estimate")]
    if audit["status"] == "HOLD":
        # Mechanics only: the registered template estimate shows the remaining
        # wrapper/canary/upload path is sound. Never published or dispatchable.
        mechanics = candidate_manifest(
            template, canary_binding, template["requirements"]["est_minutes"], ART / "main-output"
        )
        runs.append(governed_dry_run(mechanics, "mechanics-only-registered-template-estimate"))
    audit["governed_dry_runs"] = runs
    return audit


def prepare() -> dict[str, Any]:
    """Publish the audited derivation; admissible artifacts only when everything fits."""
    from tools import small_bench_operator as operator

    audit = dry_run()
    audit["commands"] = {
        "dry_run": f"nice -n 15 bash {ROOT}/tools/i0-small-bench.sh dry-run",
        "admit": f"bash {ROOT}/tools/i0-small-bench.sh admit",
        "dispatch": f"nice -n 15 bash {ROOT}/tools/i0-small-bench.sh dispatch",
        "superseded_not_usable": [
            str(OLD / "prepare-main-after-canary.sh"),
            str(OLD / "dispatch-main.sh"),
        ],
    }
    audit["implementation"] = [
        binding(ROOT / "tools/i0_small_bench.py"),
        binding(ROOT / "tools/i0-small-bench.sh"),
    ]
    if AUDIT.exists():
        raise Hold(f"audit already published (create-only): {AUDIT}")
    operator.publish_json(AUDIT, audit)
    if audit["status"] != "ADMISSIBLE":
        return {"status": "HOLD", "audit": str(AUDIT), "reasons": audit["reasons"]}
    raise Hold(
        "ADMISSIBLE sizing/cleanup requires the independent offline review of this "
        "derivation before an admissible manifest and small-bench config are frozen"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("offline", "dry-run", "prepare", "admit", "dispatch"))
    args = parser.parse_args()
    try:
        if args.operation in ("admit", "dispatch"):
            if not CONFIG.is_file():
                raise Hold(
                    "no admissible I0 small-bench config: sizing/cleanup HOLD or independent "
                    f"acceptance absent ({AUDIT})"
                )
            from tools import small_bench_operator as operator

            result = (
                operator.admit(CONFIG)
                if args.operation == "admit"
                else operator.dispatch(CONFIG, False)
            )
        elif args.operation == "prepare":
            result = prepare()
        else:
            result = dry_run() if args.operation == "dry-run" else offline()
            result.pop("_canary_binding", None)
            result.pop("_template", None)
    except Hold as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error)}, sort_keys=True))
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if result.get("status") == "HOLD":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
