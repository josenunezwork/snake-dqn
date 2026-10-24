"""I0S single-attempt A100 integration speed benchmark: adapter, admission, settlement.

Authority: ``JUDGMENT_I0R_PATH_20261024.md`` (VERDICT REPLACE_I0R_WITH_I0S, SHA256
``ed679b6e…2969``), §3 items 1, 2 and 5. Host-only. Nothing here allocates by
itself: ``prepare``/``offline``/``project`` make zero provider calls. ``admit``
captures live ``colab usage``/``sessions`` (read-only) and appends one register
reservation; ``dispatch`` hands the exact manifest to the governed
``tools/ops/dispatch.sh``; ``settle`` closes one attempt conservatively.

The frozen runtime 968f0276, its qualified broker, the canary receipt, the main
template and every earlier I0/I0R record stay byte-identical. The attempt manifest is
the frozen template plus the frozen main-path transformation (canary receipt binding,
``scientific_admission``, no ``admission_reason``, fresh ``output_dir``) and the
ruling's fields only: ``requirements.est_minutes`` 110, ``timeout_minutes`` 220,
``budget.unit_cap`` 37, ``budget.reserve_units`` 37/6 (frozen packer ratio cap/6),
``job_id`` and the recorded ``i0s_envelope``. ``INTEGRATION_BENCH_SECONDS`` stays 5400
and the main ledger stays empty (metadata ``350f4de1…``).

Attempts: exactly one, plus at most one identical-manifest retry under a new job ID
suffix, admissible only when attempt 1 is settled, verified zero units and ended for a
supervisor-classified mechanical cause. Latest dispatch 2026-10-26 20:00 UTC.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import small_bench_operator as operator
from tools import small_bench_register as register

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/Users/josenunez/Projects/ml")
RUN = BASE / "snake-dqn-gpuint-qualified-03.noindex"
PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
EXP = BASE / "snake-dqn-artifacts/big-lever/exp"
OPS = BASE / "snake-dqn-artifacts/ops"
RULING = EXP / "JUDGMENT_I0R_PATH_20261024.md"
RULING_SHA = "ed679b6ebeee861e75900763c535cc6a68ea541ef92a9b06b8537be6bbfa2969"
ROUND36 = BASE / "snake-dqn-artifacts/big-lever/round-36/JUDGMENT.md"
ROUND36_SHA = "156c80b412064defd88281424e4c84e6a732d9cb8bc6e163561ef352ffb5fe3b"
OLD = EXP / "gpu-integration-i0-20261024-03"
PACKET_OLD = OLD / "packet"
TEMPLATE = PACKET_OLD / "integration-main-template.json"
TEMPLATE_SHA = "acea317dbffd14b03df58124d470a7b8029ab04b7336b5ad6aad787ebde5d19b"
CANARY_MANIFEST = PACKET_OLD / "integration-canary.json"
RECEIPT = PACKET_OLD / "canary-output/receipt.json"
RECEIPT_SHA = "d230135d78fdf51f8647659d4926b19677bc47136903172d3cff3fcc91157ad0"
PREFLIGHT = OLD / "mac-preflight/preflight.json"
MAIN_LEDGER = PACKET_OLD / "verified-ledger.json"
METADATA_SHA = "350f4de106285510fcb611fd974ac2b3801bc28b104cdddd851987937bdcb327"
PUBLIC_INPUTS_SHA = "736f7b4e7730f760b8b81db00772d14abcf84c74b261a8dc7630732fdd383a10"
SUPERVISOR_SHA = "263eeecc556ae02057d87c351e7c83b9fb3175f55f8b1d823982bf53c564e853"
SPRINT = EXP / "small-bench-sprint-20261024"
REGISTER = SPRINT / "register.jsonl"
RESERVATION_LOCK = OPS / "sprint-reservations.lock"
ACCOUNT_ROOT = Path("/Users/josenunez/.config/colab-cli/snake")
COLAB = Path("/Users/josenunez/.local/bin/colab")
CAMPAIGN_LOG = ROOT / "docs/research/final_run_20261019/CAMPAIGN_LOG.md"
PACKET = EXP / "gpu-integration-i0s-20261024"
REVIEW = PACKET / "CLAUDE_V_STANDIN_I0S_REVIEW.json"
PACKET_ID = "I0S_SINGLE_ATTEMPT_20261024"

# Ruling §2 attempt parameters. Base = estimate; effective = 2x once in timeout/cap.
PRODUCER_SECONDS = "5400"
EST_MINUTES = 110
TIMEOUT_MINUTES = 220
UNIT_CAP = 37.0
RATE_CEILING = 10.0
RESERVE_UNITS = UNIT_CAP / 6  # frozen packer ratio (20 -> 10/3, 4 -> 2/3); not doubled
OUTSIDE_TAIL = {"units": 5.0, "wall_minutes": 30.0}
RESERVATION = {"units": UNIT_CAP + 5.0, "wall_minutes": TIMEOUT_MINUTES + 30.0}  # 42 / 250
I0_CEILING = {"units": 66.0, "wall_minutes": 396.0}
FIRST_WAVE = {"units": 88.8, "wall_minutes": 648.0}
RULING_HISTORY = {"units": 2.17994, "wall_minutes": 22.7094}
I0_HISTORY_IDS = (
    "gpu-integration-i0-canary-968f02765f7c-01",
    "native-compile-l4-968f0276-integration03",
)
SCOPES = ["global", "q4", "first_wave", "i0"]
SPEED_DEADLINE = "2026-10-27T01:00:00+00:00"
LATEST_DISPATCH = "2026-10-26T20:00:00+00:00"
ALLOCATION_WAIT_MINUTES = 10
MAX_ATTEMPTS = 2
JOB_PREFIX = "gpu-integration-i0s-main-968f02765f7c"
OLD_MAIN_JOBS = ("gpu-integration-i0-main-968f02765f7c-01", "gpu-integration-i0-main-968f0276")
FLAGS = {
    "GPU_SIM_SPIKE_COMMIT": PIN,
    "GPU_SIM_VECTOR_FLOOD": "static",
    "GPU_SIM_VECTOR_GEOMETRY": "native",
    "GPU_SIM_V8_INDEXED": "0",
    "INTEGRATION_BENCH_DEVICE": "cuda",
    "INTEGRATION_BENCH_SECONDS": PRODUCER_SECONDS,
    "INTEGRATION_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
    "INTEGRATION_BENCH_SMOKE": "0",
    "INTEGRATION_BENCH_CANARY": "0",
    "METADATA_ARCHIVE_SHA256": METADATA_SHA,
}
# Template paths the attempt manifest may differ in (dotted). Everything else is frozen.
ALLOWED_DELTA = frozenset(
    {
        "job_id",
        "output_dir",
        "canary",
        "scientific_admission",
        "admission_reason",
        "requirements.est_minutes",
        "timeout_minutes",
        "budget.unit_cap",
        "budget.reserve_units",
        "i0s_envelope",
    }
)
# Supervisor-classified causes the ruling calls mechanical (allocation, transport or
# source delivery failure, preemption, supervisor fault). CHILD_NONZERO_EXIT is never
# mechanical here: a producer TimeoutError and a native/runtime crash both surface as it.
MECHANICAL_FAILURE_CLASSES = frozenset(
    {"UPLOAD_CAP", "RPC_EXEC_POLL_TIMEOUT", "ACCOUNTING", "SUPERVISOR", "FOREIGN_GENERATION"}
)
PRODUCER_STATUS_FILES = ("integration-bench.json", "integration-failure.json")


class Hold(ValueError):
    """A concrete refusal; nothing is relaxed or retried automatically."""


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require(path: Path, digest: str, label: str) -> dict[str, str]:
    path = Path(path)
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise Hold(f"{label} identity differs: {path}")
    return {"path": str(path), "sha256": digest}


def utc(value: str | datetime) -> datetime:
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if moment.tzinfo is None:
        raise Hold("timezone-aware time required")
    return moment.astimezone(timezone.utc)


def check_attempt_number(attempt: int) -> int:
    if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise Hold(f"I0S admits attempt 1 and at most one retry; refused attempt {attempt}")
    return attempt


def job_id(attempt: int) -> str:
    return f"{JOB_PREFIX}-a{check_attempt_number(attempt)}"


def reservation_id(attempt: int) -> str:
    return f"i0s-968f0276-attempt-{check_attempt_number(attempt)}"


def attempt_dir(attempt: int, packet: Path = PACKET) -> Path:
    return Path(packet) / f"attempt-{check_attempt_number(attempt):02d}"


def envelope_record() -> dict[str, Any]:
    """Base/effective envelope metadata recorded in the manifest (owner 2x applied once)."""
    return {
        "ruling": {"path": str(RULING), "sha256": RULING_SHA},
        "verdict": "REPLACE_I0R_WITH_I0S",
        "producer_deadline_seconds": int(PRODUCER_SECONDS),
        "base": {"est_minutes": EST_MINUTES, "timeout_minutes": EST_MINUTES, "unit_cap": 18.3},
        "effective": {
            "timeout_minutes": TIMEOUT_MINUTES,
            "unit_cap": UNIT_CAP,
            "rate_units_per_hour_ceiling": RATE_CEILING,
        },
        "owner_2x": "applied once to timeout and cap; never to estimate, rate or reserve",
        "reserve_units": RESERVE_UNITS,
        "reserve_basis": "frozen integration_pack ratio cap/6, not doubled again",
        "outside_cap_tail": OUTSIDE_TAIL,
        "per_attempt_reservation": RESERVATION,
        "i0_ceiling_including_history": I0_CEILING,
        "max_attempts": MAX_ATTEMPTS,
        "latest_dispatch_utc": LATEST_DISPATCH,
        "speed_decision_utc": SPEED_DEADLINE,
        "gating_cells": ["cell-frozen-e1024", "cell-mixed-e1024", "cell-scripted-e1024"],
    }


# --------------------------------------------------------------------------- manifest


def delta_paths(template: Any, manifest: Any, prefix: str = "") -> set[str]:
    """Dotted paths where two JSON values differ (dicts recursed, other values compared)."""
    if isinstance(template, dict) and isinstance(manifest, dict):
        paths: set[str] = set()
        for key in set(template) | set(manifest):
            path = f"{prefix}{key}"
            if key not in template or key not in manifest:
                paths.add(path)
            else:
                paths |= delta_paths(template[key], manifest[key], f"{path}.")
        return paths
    return set() if template == manifest else {prefix.rstrip(".")}


def outside_ruling(template: dict[str, Any], manifest: dict[str, Any]) -> set[str]:
    """Changed paths not covered by an allowed field (or a field nested inside one)."""
    return {
        path
        for path in delta_paths(template, manifest)
        if not any(path == key or path.startswith(f"{key}.") for key in ALLOWED_DELTA)
    }


def build_manifest(
    template: dict[str, Any], canary_binding: dict[str, Any], attempt: int, output: Path
) -> dict[str, Any]:
    """Frozen template + frozen main-path transformation + the ruling's fields only."""
    value = copy.deepcopy(template)
    value.pop("admission_reason", None)
    value["scientific_admission"] = True
    value["canary"] = {"receipt": canary_binding}
    value["job_id"] = job_id(attempt)
    value["output_dir"] = str(output)
    value["requirements"]["est_minutes"] = EST_MINUTES
    value["timeout_minutes"] = TIMEOUT_MINUTES
    value["budget"]["unit_cap"] = UNIT_CAP
    value["budget"]["reserve_units"] = RESERVE_UNITS
    value["i0s_envelope"] = envelope_record()
    check_manifest(template, value, attempt)
    return value


def check_manifest(
    template: dict[str, Any],
    manifest: dict[str, Any],
    attempt: int,
    first: dict[str, Any] | None = None,
) -> None:
    """Refuse anything but the exact I0S shape; attempt 2 must equal attempt 1."""
    check_attempt_number(attempt)
    extra = outside_ruling(template, manifest)
    if extra:
        raise Hold(f"manifest differs from the frozen template outside the ruling: {sorted(extra)}")
    env = manifest.get("env", {})
    for key, value in FLAGS.items():
        if env.get(key) != value:
            raise Hold(f"frozen flag/deadline differs: {key}={env.get(key)!r}")
    if manifest.get("commit") != PIN or manifest.get("transport_pin") != TRANSPORT_PIN:
        raise Hold("runtime or transport pin differs")
    if manifest.get("job_id") != job_id(attempt):
        raise Hold("attempt job identity differs")
    if manifest.get("scientific_admission") is not True or "admission_reason" in manifest:
        raise Hold("attempt manifest is not a scientific main")
    inventory = sorted((row["dest"], row["sha256"]) for row in manifest.get("inputs", []))
    if inventory != [("metadata.tar", METADATA_SHA), ("public-inputs.json", PUBLIC_INPUTS_SHA)]:
        raise Hold("upload inventory differs from the canary-proved empty-ledger inputs")
    numbers = (
        manifest["requirements"].get("est_minutes"),
        manifest.get("timeout_minutes"),
        manifest["budget"].get("unit_cap"),
        manifest["budget"].get("rate_units_per_hour"),
    )
    if numbers != (EST_MINUTES, TIMEOUT_MINUTES, UNIT_CAP, RATE_CEILING):
        raise Hold(f"est/timeout/cap/rate differ from ruling 110/220/37/10: {numbers}")
    if not math.isclose(manifest["budget"].get("reserve_units", -1), RESERVE_UNITS):
        raise Hold("reserve differs from the frozen cap/6 formula")
    if manifest.get("i0s_envelope") != envelope_record():
        raise Hold("recorded I0S envelope differs")
    receipt = manifest.get("canary", {}).get("receipt", {})
    if receipt.get("path") != str(RECEIPT) or receipt.get("sha256") != RECEIPT_SHA:
        raise Hold("canary receipt binding differs from the I0 canary")
    if not str(manifest.get("output_dir", "")).startswith(str(PACKET) + "/"):
        raise Hold("output_dir must be a fresh directory inside the I0S packet")
    if first is not None:
        if attempt != 2:
            raise Hold("only attempt 2 is compared against attempt 1")
        if delta_paths(first, manifest) - {"job_id", "output_dir"}:
            raise Hold("retry manifest is not identical to attempt 1 apart from job/output")


# ---------------------------------------------------------------------- identities


def runtime() -> dict[str, Any]:
    """Qualified runtime/broker modules; refuse any other checkout."""
    if str(RUN) not in sys.path[:1]:
        raise Hold("run through tools/i0s.sh (qualified runtime first on PYTHONPATH)")
    from research.colab_cli import canary, supervisor
    from research.gpu_sim_spike import integration_pack, integration_units

    for module in (canary, supervisor, integration_pack, integration_units):
        if not Path(module.__file__).resolve().is_relative_to(RUN):
            raise Hold(f"{module.__name__} did not resolve from the qualified checkout")
    return {
        "canary": canary,
        "supervisor": supervisor,
        "pack": integration_pack,
        "units": integration_units,
    }


def used_identities(names: list[str], outputs: list[Path]) -> list[str]:
    paths = list(outputs)
    for name in names:
        paths += [
            OPS / "dispatch" / name,
            OPS / "logs" / f"{name}.log",
            OPS / "manifests" / f"{name}.json",
            OPS / "state" / name,
        ]
    return [str(path) for path in paths if path.exists()]


def no_i0_main_executed(attempt: int) -> None:
    """No old I0 main and no I0S attempt >= this one has run."""
    later = [job_id(n) for n in range(attempt, MAX_ATTEMPTS + 1)]
    outputs = [PACKET_OLD / "main-output", PACKET_OLD / "state-main"]
    outputs += [attempt_dir(n) / "main-output" for n in range(attempt, MAX_ATTEMPTS + 1)]
    used = used_identities([*OLD_MAIN_JOBS, *later], outputs)
    if used:
        raise Hold(f"I0 main identity already used or executed: {used}")


def static_bindings(mods: dict[str, Any]) -> dict[str, Any]:
    """Every frozen identity the attempt depends on; returns the canary binding."""
    from tools import i0_small_bench as i0

    checks = {
        "ruling": require(RULING, RULING_SHA, "I0S ruling"),
        "template": require(TEMPLATE, TEMPLATE_SHA, "frozen main template"),
        "canary_receipt": require(RECEIPT, RECEIPT_SHA, "I0 canary receipt"),
        "supervisor": require(
            RUN / "research/colab_cli/supervisor.py", SUPERVISOR_SHA, "qualified supervisor"
        ),
        "canary_review": i0.check_review(),
    }
    pack = mods["pack"]
    pack.require_source_pin(PIN)
    preflight = pack.preflight_binding(PREFLIGHT, PIN)
    canary_binding = pack._canary_binding(RECEIPT, PIN, preflight)
    if canary_binding["analysis"].get("verdict") != "PASS_CANARY":
        raise Hold("independent canary reducer did not pass")
    if canary_binding["sha256"] != RECEIPT_SHA:
        raise Hold("canary binding differs from the I0 canary receipt")
    template = json.loads(TEMPLATE.read_text())
    i0.check_template(template, json.loads(CANARY_MANIFEST.read_text()))
    if template["mac_preflight"] != preflight:
        raise Hold("main template preflight binding differs")
    ledger = mods["units"].load_ledger(
        MAIN_LEDGER, mods["units"].namespace(PIN), mods["units"].allocation()
    )
    if ledger:
        raise Hold("main ledger is not empty; I0S never binds a non-empty ledger")
    checks["main_ledger_units"] = 0
    return {"checks": checks, "canary_binding": canary_binding, "template": template}


# ------------------------------------------------------------------ retry predicate


def attempt_outputs(attempt: int, packet: Path = PACKET) -> list[Path]:
    """The attempt's output plus dispatch.sh's built-in allocation retry, if any."""
    base = attempt_dir(attempt, packet) / "main-output"
    return [path for path in (base, Path(f"{base}-retry-1")) if path.exists()]


def exported_names(output: Path) -> list[str]:
    root = output / "exports"
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith((".sha256", ".complete"))
    )


def classify_attempt(outputs: list[Path]) -> dict[str, Any]:
    """Path/receipt-only classification; never reads a unit payload or a rate."""
    if not outputs:
        return {"terminal": False, "mechanical_zero_unit": False, "reasons": ["no output"]}
    receipts = []
    for output in outputs:
        path = output / "receipt.json"
        if not path.is_file():
            return {
                "terminal": False,
                "mechanical_zero_unit": False,
                "reasons": [f"receipt absent: {path}"],
            }
        receipts.append(json.loads(path.read_text()))
    names = [name for output in outputs for name in exported_names(output)]
    units = [name for name in names if name.startswith("integration-units/")]
    producer = [name for name in names if name in PRODUCER_STATUS_FILES]
    final = receipts[-1]
    reasons = []
    if units:
        reasons.append(f"{len(units)} unit export(s) present")
    if producer:
        reasons.append(f"producer published a terminal status: {producer}")
    if final.get("status") == "PASS":
        reasons.append("receipt status PASS")
    if final.get("failure_class") not in MECHANICAL_FAILURE_CLASSES:
        reasons.append(f"failure_class {final.get('failure_class')!r} is not mechanical")
    released = all(
        receipt.get("stop_verified") is True
        or (receipt.get("endpoint") is None and receipt.get("generation") is None)
        for receipt in receipts
    )
    return {
        "terminal": True,
        "released": released,
        "failure_class": final.get("failure_class"),
        "receipt_status": final.get("status"),
        "unit_exports": len(units),
        "producer_status_files": producer,
        "mechanical_zero_unit": not reasons,
        "reasons": reasons,
    }


def retry_predicate(
    rows: list[dict[str, Any]],
    packet: Path = PACKET,
    *,
    now: datetime | None = None,
    prepared: bool = False,
) -> dict[str, Any]:
    """Ruling §2: retry iff attempt 1 is settled, zero-unit and mechanical.

    ``prepared`` is set when attempt 2's create-only directory is expected to exist
    (its admission); otherwise an existing attempt-2 directory refuses.
    """
    reasons = []
    current = register.state(rows)
    first = current.get("reservations", {}).get(reservation_id(1))
    if first is None:
        reasons.append("attempt 1 was never reserved")
    elif not first["settled"]:
        reasons.append("attempt 1 is not settled")
    if attempt_dir(2, packet).exists() != prepared:
        reasons.append("attempt 2 already prepared" if not prepared else "attempt 2 not prepared")
    classification = classify_attempt(attempt_outputs(1, packet))
    if not classification["terminal"]:
        reasons.append("attempt 1 is not terminal")
    reasons += classification["reasons"]
    moment = now or datetime.now(timezone.utc)
    if moment >= utc(LATEST_DISPATCH):
        reasons.append("latest dispatch 2026-10-26 20:00 UTC has passed")
    return {"eligible": not reasons, "reasons": reasons, "attempt_1": classification}


# ---------------------------------------------------------------- register / parents


def i0_history(rows: list[dict[str, Any]]) -> dict[str, float]:
    charges = {row["id"]: row for row in rows[0]["historical_charges"]}
    if any(name not in charges for name in I0_HISTORY_IDS):
        raise Hold("register lacks the I0 canary/compile historical charges")
    total = {
        key: sum(float(charges[name]["pair"][key]) for name in I0_HISTORY_IDS)
        for key in register.PAIR_KEYS
    }
    if any(abs(total[key] - RULING_HISTORY[key]) > 1e-4 for key in register.PAIR_KEYS):
        raise Hold(f"I0 history {total} differs from the ruling's {RULING_HISTORY}")
    return total


def scopes_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Add the I0S envelope and round-36 first-wave parents to the sprint register.

    First-wave (88.8 u / 648 min = I0 + R1 + O1 + first L4 hour) also carries O1's
    existing protected hold, scoped to first_wave only so the account and the global
    scope still count it once. R1 and the L4 hour have no register hold or charge
    (optional/untriggered); the operator adds them with sources if they are admitted.
    """
    current = register.state(rows)
    if not current["initialized"]:
        raise Hold("sprint register is not initialized")
    present = [name for name in ("i0", "first_wave") if name in current["scopes"]]
    if present:
        raise Hold(f"I0S parent scopes already recorded: {present}")
    history = i0_history(rows)
    o1 = current["holds"].get("O1")
    if o1 is None:
        raise Hold("register lacks the O1 protected hold")
    return {
        "type": "delta",
        "id": "i0s-parent-scopes-20261024",
        "sources": [
            require(RULING, RULING_SHA, "I0S ruling"),
            require(ROUND36, ROUND36_SHA, "round-36 judgment (first wave)"),
        ],
        "scopes_add": {
            "i0": {"ceiling": dict(I0_CEILING), "spent": history},
            "first_wave": {"ceiling": dict(FIRST_WAVE), "spent": history},
        },
        "holds_add": [
            {
                "id": "first-wave-O1",
                "scopes": ["first_wave"],
                "pair": dict(o1["pair"]),
                "account_units": 0,
                "kind": "protected_required_work",
                "mirrors": "O1 (global hold already counted on the account)",
            }
        ],
        "note": (
            "I0S parents per JUDGMENT_I0R_PATH §2: i0 = 66 u / 396 min including the canary "
            "and L4 compile history; first_wave = round-36 88.8 u / 648 min. The pre-existing "
            "I0 protected hold (global, q4) is left in place (counted conservatively)."
        ),
    }


def ruling_arithmetic(rows: list[dict[str, Any]], rid: str) -> dict[str, Any]:
    """Per-parent headroom before this reservation, excluding only its own refresh."""
    current = register.state(rows)
    result = {}
    for name in SCOPES:
        if name not in current["scopes"]:
            raise Hold(f"register lacks parent scope {name}; run register-scopes first")
        scope = current["scopes"][name]
        held = {key: 0.0 for key in register.PAIR_KEYS}
        for held_id, entry in current["outstanding"].items():
            if name in entry["scopes"] and held_id != rid:
                for key in register.PAIR_KEYS:
                    held[key] += entry["pair"][key]
        available = {
            key: scope["ceiling"][key] - scope["spent"][key] - held[key]
            for key in register.PAIR_KEYS
        }
        result[name] = {
            "ceiling": scope["ceiling"],
            "spent": scope["spent"],
            "outstanding_holds": held,
            "available": available,
            "fits_reservation": all(
                available[key] + 1e-9 >= RESERVATION[key] for key in register.PAIR_KEYS
            ),
        }
    return result


def admission_event(
    rows: list[dict[str, Any]],
    package: dict[str, Any],
    attempt: int,
    now: datetime,
    *,
    packet: Path = PACKET,
) -> dict[str, Any]:
    """Ruling checks first, then the shared register's fail-closed reservation."""
    check_attempt_number(attempt)
    if utc(now) >= utc(LATEST_DISPATCH):
        raise Hold("latest dispatch 2026-10-26 20:00 UTC has passed")
    rid = reservation_id(attempt)
    if package.get("reservation_id") != rid or package.get("job_id") != job_id(attempt):
        raise Hold("package reservation/job identity differs from the attempt")
    current = register.state(rows)
    others = [
        key
        for key in current.get("reservations", {})
        if key.startswith("i0s-") and key not in (reservation_id(n) for n in (1, 2))
    ]
    if others:
        raise Hold(f"foreign I0S reservations present: {others}")
    if attempt == 2:
        if current["reservations"].get(rid) is None:
            verdict = retry_predicate(rows, packet, now=now, prepared=True)
            if not verdict["eligible"]:
                raise Hold(f"retry predicate fails: {verdict['reasons']}")
    elif current["reservations"].get(reservation_id(2)) is not None:
        raise Hold("attempt 1 cannot be admitted after attempt 2")
    existing = current["reservations"].get(rid)
    if existing is not None and existing["settled"]:
        raise Hold("this attempt is already settled; a third attempt is never admitted")
    for key in register.PAIR_KEYS:
        if package["cap"][key] + package["outside_tail"][key] != RESERVATION[key]:
            raise Hold("reservation differs from 42 u / 250 min")
    if package["scopes"] != SCOPES or package["deadline_utc"] != SPEED_DEADLINE:
        raise Hold("parent scopes or speed deadline differ from the ruling")
    parents = ruling_arithmetic(rows, rid)
    short = [name for name, row in parents.items() if not row["fits_reservation"]]
    if short:
        raise Hold(f"insufficient parent headroom for 42 u / 250 min: {short}")
    event = register.reserve(rows, package, rid, utc(now))
    event["i0s"] = {"attempt": attempt, "parents_before": parents}
    return event


# ------------------------------------------------------------------------- config


def attempt_config(attempt: int, manifest_path: Path) -> dict[str, Any]:
    folder = attempt_dir(attempt)
    name = job_id(attempt)
    releases = [{"path": str(RECEIPT), "sha256": RECEIPT_SHA}]
    if attempt == 2:
        releases += [binding(output / "receipt.json") for output in attempt_outputs(1)]
    return {
        "schema": "i0s-attempt-config/v1",
        "attempt": attempt,
        "job_id": name,
        "packet_id": PACKET_ID,
        "reservation_id": reservation_id(attempt),
        "purpose": "timing",
        "manifest": binding(manifest_path),
        "ruling": {"path": str(RULING), "sha256": RULING_SHA},
        "bindings": [
            {"path": str(RULING), "sha256": RULING_SHA},
            binding(manifest_path),
            {"path": str(TEMPLATE), "sha256": TEMPLATE_SHA},
            {"path": str(RECEIPT), "sha256": RECEIPT_SHA},
        ],
        "cap": {"units": UNIT_CAP, "wall_minutes": float(TIMEOUT_MINUTES)},
        "outside_tail": dict(OUTSIDE_TAIL),
        "outside_tail_source": {
            "path": str(RUN / "research/colab_cli/supervisor.py"),
            "sha256": SUPERVISOR_SHA,
        },
        "scopes": list(SCOPES),
        "deadline_utc": SPEED_DEADLINE,
        "latest_dispatch_utc": LATEST_DISPATCH,
        "allocation_wait_minutes": ALLOCATION_WAIT_MINUTES,
        "register": str(REGISTER),
        "reservation_lock": str(RESERVATION_LOCK),
        "account_root": str(ACCOUNT_ROOT),
        "colab": str(COLAB),
        "admissions_dir": str(folder / "admissions"),
        "admission_pointer": str(folder / "OPERATOR_ADMISSION.current.json"),
        "previous_releases": releases,
        "watch_globs": [
            {"pattern": "active-*.json", "root": str(ACCOUNT_ROOT)},
            {"pattern": "pending-*.json", "root": str(ACCOUNT_ROOT)},
            {"pattern": "*.json", "root": str(OPS / "manifests")},
        ],
        "campaign_log": str(CAMPAIGN_LOG),
        "review": str(REVIEW),
        "broker": str(RUN),
        "dispatch_command": [
            str(ROOT / "tools/ops/dispatch.sh"),
            name,
            str(manifest_path),
            "--broker",
            str(RUN),
            "--cwd",
            str(RUN),
        ],
        "dispatch_cwd": str(ROOT),
        "dispatch_pythonpath": str(ROOT),
        "implementation": implementation(),
    }


def implementation() -> list[dict[str, str]]:
    return [
        binding(ROOT / name)
        for name in (
            "tools/i0s_attempt.py",
            "tools/i0s_reduce.py",
            "tools/i0s.sh",
            "tools/small_bench_operator.py",
            "tools/small_bench_register.py",
            "tools/ops/dispatch.py",
            "tools/ops/dispatch.sh",
        )
    ]


def load_config(attempt: int) -> tuple[dict[str, Any], Path]:
    path = attempt_dir(attempt) / "i0s-attempt-config.json"
    config = operator.verified(path)
    for source in config["bindings"]:
        operator.bound(source)
    operator.bound(config["outside_tail_source"])
    for source in config["implementation"]:
        operator.bound(source)
    manifest = json.loads(operator.bound(config["manifest"]).read_text())
    first = None
    if attempt == 2:
        first = json.loads((attempt_dir(1) / "integration-main.json").read_text())
    check_manifest(json.loads(TEMPLATE.read_text()), manifest, attempt, first)
    if config["job_id"] != job_id(attempt) or config["reservation_id"] != reservation_id(attempt):
        raise Hold("attempt config identity differs")
    return config, path


def require_review(config: dict[str, Any]) -> dict[str, Any]:
    """The Claude-V stand-in ACCEPT must bind attempt 1's manifest and this code."""
    review = operator.verified(Path(config["review"]))
    if review.get("verdict") != "ACCEPT":
        raise Hold("I0S requires the Claude-V stand-in ACCEPT before admission")
    first = binding(attempt_dir(1) / "integration-main.json")
    if review.get("attempt_01_manifest_sha256") != first["sha256"]:
        raise Hold("review does not bind the attempt-1 manifest")
    for source in review.get("implementation", []):
        operator.bound(source)
    reviewed = {row["path"]: row["sha256"] for row in review.get("implementation", [])}
    for row in config["implementation"]:
        if row["path"].endswith(("i0s_attempt.py", "i0s_reduce.py", "i0s.sh")) and (
            reviewed.get(row["path"]) != row["sha256"]
        ):
            raise Hold(f"review does not bind current implementation: {row['path']}")
    return review


# ---------------------------------------------------------------------- operations


def prepare(attempt: int) -> dict[str, Any]:
    """Create-only attempt manifest + admission config (+ sidecars). Zero provider calls."""
    mods = runtime()
    folder = attempt_dir(attempt)
    if folder.exists():
        raise Hold(f"attempt directory already exists (create-only): {folder}")
    no_i0_main_executed(attempt)
    first = None
    if attempt == 2:
        verdict = retry_predicate(register.read(REGISTER))
        if not verdict["eligible"]:
            raise Hold(f"retry predicate fails: {verdict['reasons']}")
        first = operator.verified(attempt_dir(1) / "integration-main.json")
    frozen = static_bindings(mods)
    manifest = build_manifest(
        frozen["template"], frozen["canary_binding"], attempt, folder / "main-output"
    )
    if first is not None:
        check_manifest(frozen["template"], manifest, attempt, first)
    path = folder / "integration-main.json"
    operator.publish_json(path, manifest)
    config_path = folder / "i0s-attempt-config.json"
    operator.publish_json(config_path, attempt_config(attempt, path))
    return {
        "status": "PREPARED",
        "attempt": attempt,
        "manifest": binding(path),
        "config": binding(config_path),
        "checks": frozen["checks"],
        "allocations": 0,
        "provider_queries": 0,
    }


def offline(attempt: int, *, dispatch_dry_run: bool = True) -> dict[str, Any]:
    """Qualified canary.check + supervisor.validate + governed dry-run on the exact file."""
    mods = runtime()
    config, _ = load_config(attempt)
    no_i0_main_executed(attempt)
    frozen = static_bindings(mods)
    path = Path(config["manifest"]["path"])
    manifest = json.loads(path.read_text())
    if manifest["canary"]["receipt"] != frozen["canary_binding"]:
        raise Hold("manifest canary binding differs from the re-derived frozen binding")
    result: dict[str, Any] = {"manifest": binding(path)}
    for name, call in (
        ("canary_check", lambda: mods["canary"].check(copy.deepcopy(manifest), RUN)),
        (
            "supervisor_validate",
            lambda: mods["supervisor"].validate(copy.deepcopy(manifest), repo=RUN),
        ),
    ):
        try:
            call()
            result[name] = "ACCEPT"
        except Exception as error:  # noqa: BLE001 - record the qualified refusal text
            result[name] = f"REFUSE {type(error).__name__}: {error}"
    if dispatch_dry_run:
        command = [*config["dispatch_command"], "--dry-run"]
        run = subprocess.run(
            command,
            cwd=config["dispatch_cwd"],
            env=dict(
                os.environ, PYTHONPATH=config["dispatch_pythonpath"], PYTHONDONTWRITEBYTECODE="1"
            ),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        lines = (run.stdout + run.stderr).strip().splitlines()
        result["governed_dry_run"] = {
            "command": command,
            "returncode": run.returncode,
            "tail": lines[-2:],
        }
    passed = result["canary_check"] == result["supervisor_validate"] == "ACCEPT" and (
        not dispatch_dry_run
        or (
            result["governed_dry_run"]["returncode"] == 0
            and any(
                line.startswith(f"PASS_DRY_RUN {config['job_id']} allocations=0")
                for line in result["governed_dry_run"]["tail"]
            )
        )
    )
    return {
        "schema": "i0s-offline-proof/v1",
        "status": "PASS_OFFLINE" if passed else "HOLD",
        "attempt": attempt,
        "checks": frozen["checks"],
        "results": result,
        "implementation": config["implementation"],
        "allocations": 0,
        "provider_queries": 0,
    }


def project() -> dict[str, Any]:
    """Read-only parent headroom projection for attempt 1 against the live register."""
    rows = register.read(REGISTER)
    simulated = list(rows)
    if "i0" not in register.state(rows)["scopes"]:
        delta = scopes_delta(rows)
        delta.update(revision=len(rows), prev_hash=rows[-1]["row_hash"])
        delta["row_hash"] = register._event_hash(delta)
        simulated.append(delta)
    parents = ruling_arithmetic(simulated, reservation_id(1))
    short = [name for name, row in parents.items() if not row["fits_reservation"]]
    return {
        "schema": "i0s-admission-projection/v1",
        "status": "FITS" if not short else "HOLD_PARENT_HEADROOM",
        "short_parents": short,
        "reservation": RESERVATION,
        "register_revision": rows[-1]["revision"],
        "register_row_hash": rows[-1]["row_hash"],
        "scopes_delta_simulated": "i0" not in register.state(rows)["scopes"],
        "parents": parents,
        "live_balance_checked": False,
        "allocations": 0,
        "provider_queries": 0,
    }


def register_scopes() -> dict[str, Any]:
    with register.reservation_lock(RESERVATION_LOCK):
        rows = register.read(REGISTER)
        row = register.append(REGISTER, scopes_delta(rows))
    return {"status": "RECORDED", "revision": row["revision"], "row_hash": row["row_hash"]}


def admit(attempt: int) -> dict[str, Any]:
    """Fast-path steps 2-3 under the I0S ruling: capture, check, reserve once."""
    config, config_path = load_config(attempt)
    require_review(config)
    no_i0_main_executed(attempt)
    with (
        register.reservation_lock(RESERVATION_LOCK),
        register.reservation_lock(ACCOUNT_ROOT / "broker.lock"),
    ):
        rows = register.read(REGISTER)
        pointer = Path(config["admission_pointer"])
        pointer.unlink(missing_ok=True)
        before = operator.fingerprint(config)
        baseline = rows[0].get("operational_baseline")
        for delta in rows[1:]:
            if "operational_baseline" in delta:
                baseline = delta["operational_baseline"]
        if baseline is not None and before != baseline:
            raise Hold("unrecorded operational liability delta; append supported register delta")
        evidence = operator.capture(config, Path(config["admissions_dir"]) / uuid.uuid4().hex)
        live = operator.live_checks(config, evidence)
        finances = register.state(rows)
        operator.assignment_coverage(finances, live)
        if before != operator.fingerprint(config):
            raise Hold("assignment or liability changed during capture; recapture")
        package = dict(
            config,
            manifest_sha256=config["manifest"]["sha256"],
            capture_utc=evidence["started_utc"],
            live_balance_units=live["balance"],
            operator=os.environ.get("USER", "unknown"),
        )
        event = admission_event(rows, package, attempt, datetime.now(timezone.utc))
        event.update(
            capture=evidence,
            live_checks=live,
            config=binding(config_path),
            operational_fingerprint=before,
            financial_before=operator.financial_snapshot(finances, config, live["balance"]),
        )
        row = register.append(REGISTER, event)
        admission = {
            "status": "ADMIT_I0S_ATTEMPT",
            "attempt": attempt,
            "reservation_id": config["reservation_id"],
            "register_revision": row["revision"],
            "register_row_hash": row["row_hash"],
            "config": binding(config_path),
            "capture": evidence,
            "live_checks": live,
            "operational_fingerprint": before,
            "row": row,
        }
        target = Path(evidence["usage"]["path"]).parent / "OPERATOR_ADMISSION.json"
        operator.publish_json(target, admission)
        temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(binding(target), sort_keys=True))
        os.replace(temporary, pointer)
        with Path(config["campaign_log"]).open("a") as stream:
            stream.write(
                f"\nI0S admission attempt {attempt} `{config['job_id']}`: ruling `{RULING_SHA}`, "
                f"reservation `{config['reservation_id']}` "
                f"({RESERVATION['units']:g} u / {RESERVATION['wall_minutes']:g} min), register "
                f"revision {row['revision']} SHA256 `{row['row_hash']}`, admission `{target}` "
                f"SHA256 `{sha(target)}`. Dispatch not implied.\n"
            )
    return {"status": admission["status"], "admission": str(target), "allocations": 0}


def check_admission(config: dict[str, Any], now: datetime) -> dict[str, Any]:
    pointer = json.loads(Path(config["admission_pointer"]).read_text())
    admission = operator.verified(operator.bound(pointer))
    rows = register.read(REGISTER)
    if rows[-1]["row_hash"] != admission["register_row_hash"]:
        raise Hold("register changed after admission; rerun admit (same reservation ID)")
    if admission["reservation_id"] != config["reservation_id"]:
        raise Hold("admission reservation differs from the attempt")
    age = (now - utc(admission["capture"]["started_utc"])).total_seconds()
    if not 0 <= age < 600:
        raise Hold("stale live capture (>10 min); rerun admit with the same reservation ID")
    if operator.fingerprint(config) != admission["operational_fingerprint"]:
        raise Hold("assignment/liability identity changed; rerun admit")
    package = dict(
        config,
        manifest_sha256=config["manifest"]["sha256"],
        capture_utc=admission["capture"]["started_utc"],
        live_balance_units=admission["live_checks"]["balance"],
        operator=admission["row"]["package"]["operator"],
    )
    admission_event(rows, package, config["attempt"], now)
    return admission


def dispatch(attempt: int) -> dict[str, Any]:
    config, _ = load_config(attempt)
    require_review(config)
    no_i0_main_executed(attempt)
    with register.reservation_lock(RESERVATION_LOCK):
        now = datetime.now(timezone.utc)
        admission = check_admission(config, now)
        remaining = 600 - (now - utc(admission["capture"]["started_utc"])).total_seconds()
        if remaining < 60:
            raise Hold("insufficient live window; rerun admit")
        env = dict(
            os.environ,
            OPS_ALLOC_LOCK_TIMEOUT_SECONDS=str(min(remaining, 600)),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=config["dispatch_pythonpath"],
        )
        subprocess.run(config["dispatch_command"], cwd=config["dispatch_cwd"], env=env, check=True)
    return {"status": "DISPATCHED", "attempt": attempt, "job_id": config["job_id"]}


def conservative_charge(receipts: list[dict[str, Any]]) -> dict[str, float]:
    """Elapsed at the 10 u/h ceiling, or the observed balance drop if larger."""
    units = wall = 0.0
    for receipt in receipts:
        elapsed = receipt.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise Hold("receipt elapsed time unknown; unknown never means zero")
        priced = elapsed / 3600 * RATE_CEILING
        before = (receipt.get("usage_before") or {}).get("balance")
        after = (receipt.get("usage_after") or {}).get("balance")
        drop = before - after if isinstance(before, float) and isinstance(after, float) else 0.0
        units += max(priced, drop)
        wall += elapsed / 60
    if units > RESERVATION["units"] + 1e-9 or wall > RESERVATION["wall_minutes"] + 1e-9:
        raise Hold("conservative charge exceeds the attempt reservation; escalate, do not settle")
    return {"units": units, "wall_minutes": wall}


def settle(attempt: int) -> dict[str, Any]:
    """Per-attempt settlement: receipt + export integrity + release capture -> register."""
    from tools import i0s_reduce

    config, _ = load_config(attempt)
    folder = attempt_dir(attempt)
    outputs = attempt_outputs(attempt)
    if not outputs:
        raise Hold("attempt has no output directory to settle")
    receipts = [json.loads((output / "receipt.json").read_text()) for output in outputs]
    if receipts[-1].get("job_id") not in (config["job_id"], f"{config['job_id']}-retry-1"):
        raise Hold("receipt job identity differs from the attempt")
    integrity = i0s_reduce.integrity(outputs)
    integrity_path = folder / "EXPORT_INTEGRITY.json"
    operator.publish_json(integrity_path, integrity)
    release = operator.capture(config, folder / f"release-{uuid.uuid4().hex}")
    release_path = Path(release["usage"]["path"]).parent / "capture.json"
    charged = conservative_charge(receipts)
    event = {
        "type": "settlement",
        "id": f"settlement:{config['reservation_id']}",
        "reservation_id": config["reservation_id"],
        "sources": [binding(output / "receipt.json") for output in outputs],
        "terminal_receipt": binding(outputs[-1] / "receipt.json"),
        "verified_exports": binding(integrity_path),
        "release_disappearance": binding(release_path),
        "charged": charged,
        "remaining_hold": {"units": 0.0, "wall_minutes": 0.0},
        "remaining_account_units": charged["units"],
        "basis": "elapsed at the 10 u/h ceiling or observed balance drop, whichever is larger; "
        "debit stays held on the account until provider reconciliation",
        "settled_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classify_attempt(outputs),
    }
    operator.settlement_evidence(event)
    record = folder / "SETTLEMENT.json"
    operator.publish_json(record, event)
    with register.reservation_lock(RESERVATION_LOCK):
        row = register.append(REGISTER, event)
    return {"status": "SETTLED", "record": str(record), "revision": row["revision"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "prepare",
            "offline",
            "project",
            "register-scopes",
            "admit",
            "dispatch",
            "settle",
            "retry-predicate",
        ),
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--no-dispatch-dry-run", action="store_true")
    parser.add_argument("--publish", type=Path, help="create-only record (+ sidecars)")
    args = parser.parse_args()
    try:
        check_attempt_number(args.attempt)
        if args.operation == "prepare":
            result = prepare(args.attempt)
        elif args.operation == "offline":
            result = offline(args.attempt, dispatch_dry_run=not args.no_dispatch_dry_run)
        elif args.operation == "project":
            result = project()
        elif args.operation == "register-scopes":
            result = register_scopes()
        elif args.operation == "admit":
            result = admit(args.attempt)
        elif args.operation == "dispatch":
            result = dispatch(args.attempt)
        elif args.operation == "settle":
            result = settle(args.attempt)
        else:
            result = retry_predicate(register.read(REGISTER))
    except (Hold, register.Refused, ValueError, OSError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error)}, sort_keys=True))
        raise SystemExit(2) from error
    if args.publish:
        if args.publish.exists():
            raise SystemExit(f"record already published (create-only): {args.publish}")
        operator.publish_json(args.publish, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if result.get("status") in ("HOLD", "HOLD_PARENT_HEADROOM"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
