"""Amendment 2 pre-check and paced-retention execution harness.

The live adapter is intentionally the only place that constructs a Watch6 or
terminal world.  This module owns registration guards, pacing, create-only
receipts and metric collection; it never imports retired v1 ``run.py`` or
``web.backend.pathb``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping

# This module is also used with ``python -m``.  Fix the native pool caps before
# importing either the adapter or its NumPy/Torch dependencies.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

from research.t34_pathb import timing_v2  # noqa: E402
from research.t34_pathb.analyze_v2 import ACTIVE_ARMS, MIXES, analyze_path  # noqa: E402

ARMS = ("bare", *ACTIVE_ARMS)
PRECHECK_WORLDS = 64
PRECHECK_MIN_SURVIVORS = 50
TERMINAL_PER_MIX = 32
PRODUCT_SESSIONS = 16
MAX_ELAPSED_HOURS = 32.0
MAX_CPU_HOURS = 48.0


class RegistrationError(RuntimeError):
    """A non-smoke run is not authorized or does not match Amendment 2."""


class UnconditionalFault(RuntimeError):
    """A registered controller tripwire whose partial evidence is retained."""


class ActionWindow:
    """Per-arm rolling 100k delivered-action tripwire across all worlds."""

    def __init__(self) -> None:
        self.values: deque[int] = deque(maxlen=100_000)
        self.counts = [0] * 6

    def add(self, action: int) -> None:
        if not 0 <= action < 6:
            raise UnconditionalFault("illegal_action")
        if len(self.values) == self.values.maxlen:
            self.counts[self.values[0]] -= 1
        self.values.append(action)
        self.counts[action] += 1
        if len(self.values) == self.values.maxlen and max(self.counts) / len(self.values) > 0.90:
            raise UnconditionalFault("one_action_share_100k")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(repo: Path) -> dict[str, str]:
    """Hash all tracked code/config inputs plus binding documents and serving pin."""
    names = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    selected = {
        name
        for name in names
        if (name.startswith(("src/", "research/", "web/backend/")) and name.endswith(".py"))
        or (name.startswith("configs/") and name.endswith((".yaml", ".yml")))
    }
    selected.update(
        {
            "research/t34_pathb/PROTOCOL_V2.md",
            "research/t34_pathb/PROTOCOL_V2_BUILD_RULES.md",
            "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
            "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
            "docs/research/final_run_20261019/P7_STATISTICS.md",
            "web/backend/served_checkpoint_pins.json",
        }
    )
    return {name: _sha(repo / name) for name in sorted(selected) if (repo / name).is_file()}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RegistrationError(f"{path}: expected an object")
    return value


def precheck_receipt(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the T3-bank development gate; it deliberately decides no claim."""
    if len(results) != PRECHECK_WORLDS:
        raise RegistrationError("pre-check must contain exactly 64 reused T3 worlds")
    cases = {(row.get("t3_source_arm"), row.get("t3_env")) for row in results}
    if cases != {(arm, env) for arm in ("b8_s0", "b8_s3") for env in range(32)}:
        raise RegistrationError("pre-check must cover b8_s0/b8_s3 x env0..31")
    seeds_by_env: dict[int, set[int]] = {}
    for row in results:
        env, seed = row.get("t3_env"), row.get("seed")
        if not isinstance(env, int) or not isinstance(seed, int):
            raise RegistrationError("pre-check environment/seed metadata is invalid")
        seeds_by_env.setdefault(env, set()).add(seed)
    if set(seeds_by_env) != set(range(32)) or any(
        len(values) != 1 for values in seeds_by_env.values()
    ):
        raise RegistrationError("T3 source arms must share one seed per env")
    if len({next(iter(values)) for values in seeds_by_env.values()}) != 32:
        raise RegistrationError("T3 physical seeds must be distinct across environments")
    if any("bare_survives" not in row or "paced_survives" not in row for row in results):
        raise RegistrationError("pre-check requires paired bare and paced outcomes")
    if any(
        row.get("horizon") != 5000
        or row.get("t3_namespace") != "t3-dense-ceiling/v1"
        or row.get("t3_world") != f"{row.get('t3_source_arm')}_{row.get('t3_env')}"
        or not isinstance(row.get("t3_receipt"), Mapping)
        or not isinstance(row.get("t3_checkpoint"), Mapping)
        or not isinstance(row.get("t3_materialized_sha256"), str)
        or len(row["t3_materialized_sha256"]) != 64
        for row in results
    ):
        raise RegistrationError("pre-check must bind materialized T3 dense H5000 provenance")
    survivors = sum(bool(row["paced_survives"]) for row in results)
    raw = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    repo = Path(__file__).resolve().parents[2]
    return {
        "schema": "t34-pathb-v2-precheck/v1",
        "development_not_evidence": True,
        "bank": "reused-t3-dense",
        "worlds": PRECHECK_WORLDS,
        "trajectory_cases": PRECHECK_WORLDS,
        "unique_physical_seeds": 32,
        "survivors": survivors,
        "threshold": PRECHECK_MIN_SURVIVORS,
        "raw_records": results,
        "raw_records_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_hashes": source_inventory(repo),
        "proceed": survivors >= PRECHECK_MIN_SURVIVORS,
    }


def _execution_plan(row: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    """Return the exact plan an adapter will execute, with outer identity fenced."""
    if not isinstance(row, Mapping):
        raise RegistrationError(f"{label} row is not a mapping")
    plan = row.get("roster", row)
    if not isinstance(plan, Mapping):
        raise RegistrationError(f"{label} roster is not a mapping")
    for field in ("seed", "mix"):
        if field not in row or field not in plan or row[field] != plan[field]:
            raise RegistrationError(f"{label} roster {field} differs from its registered row")
    return plan


def _validate_checkpoint_coverage(registration: Mapping[str, Any]) -> None:
    """Bind every checkpoint an adapter can load before the first world starts."""
    checkpoints = registration.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RegistrationError("checkpoint bytes and deployment profile hashes are required")
    by_path: dict[str, str] = {}
    for item in checkpoints:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
        ):
            raise RegistrationError("checkpoint inventory is malformed")
        path = Path(item["path"])
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise RegistrationError("checkpoint bytes differ from registration")
        resolved = str(path.resolve())
        if resolved in by_path and by_path[resolved] != item["sha256"]:
            raise RegistrationError("checkpoint inventory binds one path to conflicting hashes")
        by_path[resolved] = item["sha256"]

    # These are the two fixed serving identities loaded by RunSession/open_watch;
    # use the registry constants, rather than constructing a Watch episode.
    from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

    inventory_hashes = set(by_path.values())
    if B8_S3_SHA256 not in inventory_hashes:
        raise RegistrationError("released B8 s3 is absent from checkpoint inventory")
    if FRP3_S12_SHA256 not in inventory_hashes:
        raise RegistrationError("Watch FRP3 s12 is absent from checkpoint inventory")

    terminal_rows = registration.get("terminal_rows")
    if not isinstance(terminal_rows, list):
        raise RegistrationError("terminal rows are required for checkpoint coverage")
    for index, row in enumerate(terminal_rows):
        plan = _execution_plan(row, f"terminal[{index}]")
        specs = plan.get("specs")
        if not isinstance(specs, list):
            raise RegistrationError(f"terminal[{index}] roster lacks materialized specs")
        for slot, spec in enumerate(specs):
            if not isinstance(spec, (list, tuple)) or not spec:
                raise RegistrationError(f"terminal[{index}] spec {slot} is malformed")
            if spec[0] != "checkpoint":
                continue
            if len(spec) < 2 or not isinstance(spec[1], str):
                raise RegistrationError(f"terminal[{index}] checkpoint spec {slot} is malformed")
            path = str(Path(spec[1]).resolve())
            if path not in by_path:
                raise RegistrationError(
                    f"terminal[{index}] checkpoint spec {slot} is absent from inventory"
                )

    product_rows = registration.get("product_rows")
    if not isinstance(product_rows, list):
        raise RegistrationError("product rows are required for identity coverage")
    for index, row in enumerate(product_rows):
        if _execution_plan(row, f"product[{index}]").get("mix") != "mixed":
            raise RegistrationError(f"product[{index}] is not a mixed Watch6 row")
    timing_row = registration.get("timing_row")
    if not isinstance(timing_row, Mapping):
        raise RegistrationError("timing row is required for identity coverage")
    if _execution_plan(timing_row, "timing").get("mix") != "mixed":
        raise RegistrationError("timing row is not a mixed Watch6 row")


def _registered_seed_union(registration: Mapping[str, Any]) -> set[Any]:
    """Return the exact seed union and reject timing reuse before any run."""
    terminal_rows = registration.get("terminal_rows", [])
    product_rows = registration.get("product_rows", [])
    timing_row = registration.get("timing_row", {})
    if not isinstance(terminal_rows, list) or not isinstance(product_rows, list):
        raise RegistrationError("registered seed rows are malformed")
    terminal = {row.get("seed") for row in terminal_rows if isinstance(row, Mapping)}
    product = {row.get("seed") for row in product_rows if isinstance(row, Mapping)}
    timing_seed = timing_row.get("seed") if isinstance(timing_row, Mapping) else None
    if None in terminal or None in product or timing_seed is None:
        raise RegistrationError("seed audit is incomplete or failed")
    if timing_seed in terminal or timing_seed in product:
        raise RegistrationError("timing seed must be disjoint from terminal and product banks")
    return terminal | product | {timing_seed}


def require_registration(registration: Mapping[str, Any], precheck: Mapping[str, Any]) -> None:
    """Enforce the frozen full-test design before the first efficacy world."""
    if registration.get("schema") != "t34-pathb-v2-registration/v1":
        raise RegistrationError("missing v2 registration")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_commit = registration.get("source_commit")
    if registration.get("ratified") is not True or not isinstance(source_commit, str):
        raise RegistrationError("non-smoke registration must be ratified and source-pinned")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, current_commit],
        cwd=Path(__file__).resolve().parents[2],
    ).returncode:
        raise RegistrationError("registered source revision is not an ancestor of current HEAD")
    if registration.get("arms") != list(ARMS) or registration.get("paced_fps") != 24:
        raise RegistrationError("registration arms/pacing differ from Amendment 2")
    if (
        registration.get("terminal_per_mix") != TERMINAL_PER_MIX
        or registration.get("product_sessions") != PRODUCT_SESSIONS
    ):
        raise RegistrationError("registration sample size differs from Amendment 2")
    if registration.get("horizon") != 5000 or registration.get("workers") != 1:
        raise RegistrationError("registration must be H5000 with one assistance worker")
    load_profile = registration.get("load_profile")
    if not isinstance(load_profile, Mapping) or load_profile.get("quiet_ratified") is not True:
        raise RegistrationError("registered quiet load profile is required for final timing")
    if (
        registration.get("max_elapsed_hours") != MAX_ELAPSED_HOURS
        or registration.get("max_cpu_hours") != MAX_CPU_HOURS
    ):
        raise RegistrationError("registration cap differs from Amendment 2")
    sources = registration.get("source_hashes")
    if (
        registration.get("fresh_disjoint_banks") is not True
        or not isinstance(sources, Mapping)
        or not sources
    ):
        raise RegistrationError("fresh collision-audited banks and source hashes are required")
    repo = Path(__file__).resolve().parents[2]
    if dict(sources) != source_inventory(repo):
        raise RegistrationError("source hash inventory does not bind the full v2 closure")
    for name, digest in sources.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RegistrationError("source hash inventory is malformed")
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or _sha(path) != digest:
            raise RegistrationError(f"source hash differs: {name}")
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{name}"], cwd=repo
            )
        except subprocess.CalledProcessError as exc:
            raise RegistrationError(f"source is absent from registered commit: {name}") from exc
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RegistrationError(f"registered commit hash differs: {name}")
    raw = precheck.get("raw_records")
    if (
        not isinstance(raw, list)
        or precheck.get("raw_records_sha256")
        != hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise RegistrationError("pre-check raw evidence is missing or hash-mismatched")
    if precheck.get("schema") != "t34-pathb-v2-precheck/v1" or not precheck_receipt(raw)["proceed"]:
        raise RegistrationError("pre-check did not reach 50/64; fresh test is blocked")
    if precheck.get("source_commit") != source_commit or precheck.get(
        "source_hashes"
    ) != source_inventory(repo):
        raise RegistrationError("pre-check source identity differs from registered source revision")
    _validate_checkpoint_coverage(registration)
    for label in ("profile", "bank", "seed_audit", "preregistration"):
        item = registration.get(label)
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            raise RegistrationError(f"{label} path and hash are required")
        path = Path(item["path"])
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise RegistrationError(f"{label} bytes differ from registration")
    prereg = registration["preregistration"]
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{prereg['path']}"], cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise RegistrationError("preregistration is not committed before runtime") from exc
    if hashlib.sha256(committed).hexdigest() != prereg["sha256"]:
        raise RegistrationError("committed preregistration hash differs")
    for label in ("trace_receipt", "operating_characteristic"):
        item = registration.get(label)
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            raise RegistrationError(f"{label} path and hash are required")
        path = Path(item["path"])
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise RegistrationError(f"{label} bytes differ from registration")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RegistrationError(f"{label} is not JSON") from exc
        if payload.get("status") != "PASS" or payload.get("source_hashes") != sources:
            raise RegistrationError(f"{label} does not bind passing source bytes")
    oc = json.loads(Path(registration["operating_characteristic"]["path"]).read_text())
    if not isinstance(oc.get("simulator_hash"), str) or not isinstance(
        oc.get("world_pool_sha256"), str
    ):
        raise RegistrationError("operating characteristic lacks simulator/world-pool binding")
    try:
        bank_payload = json.loads(Path(registration["bank"]["path"]).read_text())
        audit_payload = json.loads(Path(registration["seed_audit"]["path"]).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationError("bank or seed audit is not readable JSON") from exc
    for key in ("terminal_rows", "product_rows", "timing_row"):
        if bank_payload.get(key) != registration.get(key):
            raise RegistrationError(f"materialized bank differs for {key}")
    expected_seeds = _registered_seed_union(registration)
    if audit_payload.get("pass") is not True:
        raise RegistrationError("seed audit is incomplete or failed")
    audit_seeds = set(audit_payload.get("seeds", audit_payload.get("candidate_seeds", [])))
    if audit_seeds != expected_seeds or audit_payload.get(
        "collisions", audit_payload.get("hits", [])
    ):
        raise RegistrationError("seed audit does not bind the exact collision-free bank")


def _adapter() -> Any:
    try:
        from research.t34_pathb import adapter_v2
    except ImportError as exc:  # pragma: no cover - integration failure is deliberately loud
        raise RegistrationError("Path B v2 live adapter is unavailable") from exc
    return adapter_v2


def _execution_guard() -> None:
    """Refuse a real Mac run without the required resource and power constraints."""
    if sys.platform != "darwin":
        raise RegistrationError("T34 Path B execution is Mac-only")
    if any(
        os.environ.get(key) not in ("1", None)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    ):
        raise RegistrationError("all numerical thread pools must be capped at one")
    from research.compute.thermal_guard import parse_pmset_therm, read_pmset_therm, thermal_reasons
    from research.perf_sim_20261007 import with_slots

    reading = read_pmset_therm()
    if not with_slots.on_ac() or not with_slots.lid_open() or not reading.get("available"):
        raise RegistrationError("real Path B run requires AC, open lid, and thermal telemetry")
    if thermal_reasons(parse_pmset_therm(reading.get("text") or "")):
        raise RegistrationError("thermal admission refused")


def _cap(started: float, cpu_started: float) -> None:
    if time.monotonic() - started >= MAX_ELAPSED_HOURS * 3600:
        raise TimeoutError("32-hour elapsed cap reached")
    if time.process_time() - cpu_started >= MAX_CPU_HOURS * 3600:
        raise TimeoutError("48 CPU-hour cap reached")


def _collect(
    session: Any,
    *,
    horizon: int,
    fps: int,
    started: float,
    cpu_started: float,
    action_window: ActionWindow | None = None,
) -> dict[str, Any]:
    masses: list[int] = []
    deaths = boosts = decisions = overruns = large = late = overlap = 0
    action_counts = [0] * 6
    for index in range(horizon):
        _cap(started, cpu_started)
        if fps == 0 and hasattr(session, "settle_between_ticks"):
            session.settle_between_ticks(lambda: _cap(started, cpu_started))
            settled = session.controller_state()
            faults = settled.get("unconditional_faults") if isinstance(settled, Mapping) else None
            if faults:
                marker = faults[0]
                raise UnconditionalFault(
                    str(marker.get("marker", marker))
                    if isinstance(marker, Mapping)
                    else str(marker)
                )
        tick_started = time.monotonic()
        try:
            session.step()
        except ValueError as exc:
            # The engine exposes NonFiniteQ as a typed ValueError without
            # forcing this harness to import numerical modules at startup.
            if type(exc).__name__ == "NonFiniteQ":
                raise UnconditionalFault("nonfinite_q") from exc
            raise
        stepped = session.controller_state()
        faults = stepped.get("unconditional_faults") if isinstance(stepped, Mapping) else None
        if faults:
            marker = faults[0]
            raise UnconditionalFault(
                str(marker.get("marker", marker)) if isinstance(marker, Mapping) else str(marker)
            )
        masses.append(int(session.hero_length) if session.hero_alive else 0)
        deaths += int(session.hero_death)
        if session.hero_death:
            death_length = int(
                getattr(session, "death_length", getattr(session, "last_alive_length", 0))
            )
            is_large = death_length >= 800
            is_late = index + 1 >= 2000
            large += int(is_large)
            late += int(is_late)
            overlap += int(is_large and is_late)
        decisions += int(session.decisions_delta)
        boosts += int(session.boost_decisions_delta)
        action = int(getattr(session, "executed_action", getattr(session, "bare_action", 1)))
        if not 0 <= action < 6:
            raise UnconditionalFault("illegal_action")
        if action_window is not None and session.decisions_delta:
            action_window.add(action)
        action_counts[action] += int(session.decisions_delta)
        elapsed = time.monotonic() - tick_started
        overruns += int(bool(fps) and elapsed >= 1 / fps)
        if fps:
            time.sleep(max(0.0, 1 / fps - elapsed))
        # Terminal heroes are scored as zero after death.  Do not waste paced
        # wall-clock time advancing an already-dead terminal world.
        if not session.hero_alive and getattr(session, "live", None) is None:
            masses.extend([0] * (horizon - index - 1))
            break
    if decisions < 1:
        raise RegistrationError("session exposed no hero decisions")
    ordered = sorted(masses)
    controller = session.controller_state()
    counts = dict(controller.get("counts") or {}) if isinstance(controller, Mapping) else {}
    declared_faults = (
        list(controller.get("unconditional_faults") or [])
        if isinstance(controller, Mapping)
        else []
    )
    if declared_faults:
        marker = declared_faults[0]
        raise UnconditionalFault(
            str(marker.get("marker", marker)) if isinstance(marker, Mapping) else str(marker)
        )
    faults = [
        name for name in ("illegal_action", "worker_error", "fault") if int(counts.get(name, 0))
    ]
    invalid_controller = any(
        int(counts.get(name, 0)) for name in ("worker_errors", "invalid_results")
    )
    return {
        "mass_integral": sum(masses) / horizon,
        "survival_fraction": sum(x > 0 for x in masses) / horizon,
        "no_death": int(deaths == 0),
        "deaths": deaths,
        "decisions": decisions,
        "deaths_per_1k": deaths * 1000 / decisions,
        "boost_decisions": boosts,
        "boost_share": boosts / decisions,
        "median_length": ordered[len(ordered) // 2],
        "p95_length": ordered[int(0.95 * (len(ordered) - 1))],
        "overruns": overruns,
        "action_counts": action_counts,
        "one_action_share": max(action_counts) / decisions,
        "unconditional_faults": faults,
        "controller": controller,
        "deaths_length_ge_800": large,
        "late_deaths": late,
        "late_large_overlap": overlap,
        "unpaced_clock": "held_between_ticks_until_worker_completes" if fps == 0 else None,
        "identity_status": (
            "NOT_VERIFIED"
            if invalid_controller
            else getattr(session, "identity_status", "NOT_VERIFIED")
        ),
        "parity_status": getattr(session, "parity_status", "NOT_VERIFIED"),
    }


def run_precheck(
    rows: list[Mapping[str, Any]],
    open_terminal: Callable[[Mapping[str, Any], int, str], Any] | None = None,
) -> dict[str, Any]:
    """Execute the paced T3 development pre-check through the real adapter seam."""
    _execution_guard()
    from research.t34_pathb.precheck_v2 import validate_t3_row

    if open_terminal is None:
        open_terminal = _adapter().open_precheck
    if len(rows) != PRECHECK_WORLDS:
        raise RegistrationError("pre-check input must have 64 T3 rows")
    if {(row.get("t3_source_arm"), row.get("t3_env")) for row in rows} != {
        (arm, env) for arm in ("b8_s0", "b8_s3") for env in range(32)
    }:
        raise RegistrationError("pre-check input must contain the complete original 2x32 inventory")
    for row in rows:
        validate_t3_row(row)
    started, cpu_started = time.monotonic(), time.process_time()
    output = []
    for row in rows:
        paired = {
            **dict(row),
            "seed": int(row["seed"]),
            "horizon": 5000,
            "t3_namespace": row.get("t3_namespace"),
            "t3_world": row.get("t3_world"),
            "t3_source_arm": row.get("t3_source_arm"),
            "t3_env": row.get("t3_env"),
            "t3_receipt": row.get("t3_receipt"),
            "t3_checkpoint": row.get("t3_checkpoint"),
            "t3_materialized_sha256": row.get("t3_materialized_sha256"),
        }
        for arm in ("bare", "paced"):
            session = open_terminal(row, 24, arm)
            try:
                result = _collect(
                    session,
                    horizon=5000,
                    fps=24,
                    started=started,
                    cpu_started=cpu_started,
                )
                paired[f"{arm}_survives"] = result["no_death"] == 1
            finally:
                session.close(wait=True)
        output.append(paired)
    return precheck_receipt(output)


def run_registered(
    out: Path,
    registration: Mapping[str, Any],
    precheck: Mapping[str, Any],
    *,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Run one fixed-N real session set.  This never resumes or fills missing rows."""
    started, cpu_started = time.monotonic(), time.process_time()
    require_registration(registration, precheck)
    _execution_guard()
    if out.exists():
        raise RegistrationError("output root must be fresh")
    if adapter is None:
        adapter = _adapter()
    terminal_rows = list(registration.get("terminal_rows", ()))
    product_rows = list(registration.get("product_rows", ()))
    if len(terminal_rows) != TERMINAL_PER_MIX * len(MIXES) or len(product_rows) != PRODUCT_SESSIONS:
        raise RegistrationError("registered bank rows do not match fixed N")
    if {row.get("mix") for row in terminal_rows} != set(MIXES):
        raise RegistrationError("terminal bank lacks a mix")
    if any(
        sum(row.get("mix") == mix for row in terminal_rows) != TERMINAL_PER_MIX for mix in MIXES
    ):
        raise RegistrationError("terminal bank must hold 32 rows in each mix")
    terminal_seeds = [row.get("seed") for row in terminal_rows]
    product_seeds = [row.get("seed") for row in product_rows]
    if len(set(terminal_seeds)) != len(terminal_seeds) or len(set(product_seeds)) != len(
        product_seeds
    ):
        raise RegistrationError("registered bank rows must not repeat seeds")
    if set(terminal_seeds) & set(product_seeds):
        raise RegistrationError("terminal and continuous banks must be disjoint")
    _write_new(out / "registration.json", dict(registration))
    _write_new(out / "precheck.json", dict(precheck))
    windows = {arm: ActionWindow() for arm in ARMS}
    try:
        for kind, rows, opener in (
            ("terminal", terminal_rows, adapter.open_terminal),
            ("product", product_rows, adapter.open_watch),
        ):
            for index, row in enumerate(rows):
                arms: dict[str, Any] = {}
                for arm in ARMS:
                    _cap(started, cpu_started)
                    arm_fps = 0 if arm == "unpaced" else 24
                    session = opener(row, arm_fps, arm)
                    try:
                        arms[arm] = _collect(
                            session,
                            horizon=5000,
                            fps=arm_fps,
                            started=started,
                            cpu_started=cpu_started,
                            action_window=windows[arm],
                        )
                    finally:
                        session.close(wait=True)
                _write_new(
                    out / "records" / f"{kind}-{index:03d}.json",
                    {
                        "kind": kind,
                        "horizon": 5000,
                        "mix": row.get("mix", "mixed"),
                        "seed": int(row["seed"]),
                        "identity_status": (
                            "PASS"
                            if all(metric["identity_status"] == "PASS" for metric in arms.values())
                            else "NOT_VERIFIED"
                        ),
                        "parity_status": (
                            "PASS"
                            if all(metric["parity_status"] == "PASS" for metric in arms.values())
                            else "NOT_VERIFIED"
                        ),
                        "world": dict(row),
                        "arms": arms,
                        "registration_sha256": _sha(out / "registration.json"),
                    },
                )
        timing = timing_v2.run_probes(
            lambda arm, fps: adapter.open_watch(registration["timing_row"], fps, arm),
            ticks=int(registration["timing_ticks"]),
            warmup=int(registration["timing_warmup"]),
            contended=bool(registration.get("contended", True)),
            check_cap=lambda: _cap(started, cpu_started),
        )
        _write_new(out / "timing.json", timing)
        record_statuses = [
            json.loads(path.read_text()) for path in (out / "records").glob("*.json")
        ]
        _write_new(
            out / "complete.json",
            {
                "status": "complete",
                "registration_sha256": _sha(out / "registration.json"),
                "elapsed_seconds": time.monotonic() - started,
                "cpu_seconds": time.process_time() - cpu_started,
                "identity_status": (
                    "PASS"
                    if all(row["identity_status"] == "PASS" for row in record_statuses)
                    else "NOT_VERIFIED"
                ),
                "parity_status": (
                    "PASS"
                    if all(row["parity_status"] == "PASS" for row in record_statuses)
                    else "NOT_VERIFIED"
                ),
            },
        )
    except Exception as exc:
        # Timing probes use the same adapter hooks as collection. A nonfinite
        # Q there is the same unconditional fault, even before a row is saved.
        faults = (
            ["nonfinite_q"]
            if type(exc).__name__ == "NonFiniteQ"
            else [str(exc)] if isinstance(exc, (UnconditionalFault, timing_v2.ProbeFault)) else []
        )
        _write_new(out / "fault.json", {"unconditional_faults": faults})
        _write_new(
            out / "incomplete.json",
            {
                "status": "INCOMPLETE",
                "reason": f"{type(exc).__name__}: {exc}",
                "unconditional_faults": faults,
            },
        )
        raise
    result = analyze_path(out)
    _write_new(out / "decision.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("validate-precheck", "precheck", "smoke", "analyze", "run")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--precheck", type=Path)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    if args.action == "validate-precheck":
        if args.input is None:
            parser.error("validate-precheck needs --input JSON list")
        print(json.dumps(precheck_receipt(json.loads(args.input.read_text())), sort_keys=True))
    elif args.action == "precheck":
        if args.input is None or args.out.exists():
            parser.error("precheck needs --input rows and a fresh --out directory")
        rows = json.loads(args.input.read_text())
        if not isinstance(rows, list):
            parser.error("precheck input must be a JSON list")
        receipt = run_precheck(rows)
        _write_new(args.out / "precheck.json", receipt)
        print(f"{receipt['survivors']}/64 survive (64 source trajectories, 32 physical seeds)")
        print(json.dumps(receipt, sort_keys=True))
    elif args.action == "smoke":
        if args.input is None or args.out.exists() or not 1 <= args.frames <= 4:
            parser.error("smoke needs --input rows, fresh --out, and --frames 1..4")
        _execution_guard()
        rows = json.loads(args.input.read_text())
        if not isinstance(rows, list) or not rows:
            parser.error("smoke input must provide one materialized terminal row")
        adapter, started, cpu_started = _adapter(), time.monotonic(), time.process_time()
        artifacts: dict[str, Any] = {
            "schema": "t34-pathb-v2-smoke/v1",
            "development_only": True,
            "frames": args.frames,
            "runs": [],
        }
        for kind, opener, row in (
            ("terminal", adapter.open_terminal, rows[0]),
            ("watch", adapter.open_watch, rows[0]),
        ):
            for arm in ("bare", "paced"):
                session = opener(row, 24, arm)
                try:
                    artifacts["runs"].append(
                        {
                            "kind": kind,
                            "arm": arm,
                            "metrics": _collect(
                                session,
                                horizon=args.frames,
                                fps=24,
                                started=started,
                                cpu_started=cpu_started,
                            ),
                        }
                    )
                finally:
                    session.close(wait=True)
        _write_new(args.out / "smoke.json", artifacts)
        print(json.dumps({"smoke": "complete", "artifact_root": str(args.out)}, sort_keys=True))
    elif args.action == "analyze":
        print(json.dumps(analyze_path(args.out), sort_keys=True))
    else:
        if args.registration is None or args.precheck is None:
            parser.error("run needs --registration and --precheck")
        result = run_registered(args.out, _load(args.registration), _load(args.precheck))
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
