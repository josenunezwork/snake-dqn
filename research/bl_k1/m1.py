"""PLAN_v3 M1: native exact-stop co-residency, operated only by Claude on a lease.

A completed *cell* is the immutable resume unit. Interrupted cells are retained
as diagnostics and rerun whole, never stitched across paced clock epochs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from research.bl_k1.carry import carried_violation
from research.bl_mac_latency import l1
from research.bl_served_search import native_ruler, ruler
from research.bl_served_search.exports import digest, export_json, verify_export

SCHEMA = "bl-plan-v3-m1/v1"
BASE_SECONDS = 7200
CAP_SECONDS = 14400  # PLAN_v3 already doubles the 2h forecast; never double again.
REPEATS = 30
CONTENTION_SECONDS = 30
Y1_FIXTURES_SHA256 = "bbe6975642512091ba279d9ff7c870000d68faa18e30063807f47af1352e8574"
Y1_REFERENCE_SHA256 = "b97f73d2384a95455beb3c0d243caba513f5f307edc85d0a8da4b81f5e1ac791"
# The single historical pre-pool failure audited against frozen 3708063b.
PRECELL_RECEIPT_SHA256 = "5ee5b1629ad11b7e4c181455846f9591ec2be67dc71549501a3f4336fba6a29c"
PRECELL_ATTEMPT_SHA256 = "d917e555b79d54b245733b66db309546dbd5c6e91ed9b3d916b7e162612b8bc0"
PRECELL_LOG_SHA256 = "64b160641243f014550c86650092077f8f2aefc41e8401f5059fdfa3555d062c"
PRECELL_SOURCE_SHA256 = "3aff0cf6c037fb5b4597bddbc5707b5e23b36146bc2a7e34b0a07fcf579d0e57"
# Fable's final, outcome-blind completion admission for the attempt-03 stop.
JUDGED_RECEIPT_SHA256 = "53e45bff4145c26c7b507037d960baf47e5e293c9bca7b59c8a3180694562063"
JUDGED_ATTEMPT_SHA256 = "40372be149d99b7c8d34010bf0a7d569541819de3bf85af883de83e9d3113604"
JUDGED_LOG_SHA256 = "c7d0780e55a21b002ed4648069bc569f8e3f39de736ec9c89bcf6f156115e8de"
JUDGED_CELL_SHA256 = {
    "cpu-w1-k1": "c76921b70835d1c6dc24b543d26d86721ac389b99b042f547b3bed9988a874e1"
}
JUDGED_SOURCE_SHA256 = "78ae9369780d0f2b9c4f4e3fb2946a743589f625a8afafa622b545d1c4f27c7a"
JUDGED_PRECELL_SHA256 = "4ebca231284aac36426f5d7aebe0f2e2e5029f2ed67586caa95f55e69d75d9f7"
JUDGED_CHARGED_SECONDS = 5.999214000010397

# Fable's second pinned admission; no cell evidence crosses the namespace.
JUDGED2_RECEIPT_SHA256 = "77e3d9a9274fd1a55697231c4913d100515af7a420d67622bbf46534310944ba"
JUDGED2_ATTEMPT_SHA256 = "3562923b76f91755acba0c7ff9a5382e73595f4beec882f089537887ef8be79e"
JUDGED2_LOG_SHA256 = "6b31913c90fd7c493067c66f48632559f1304f537a6dd1854fea35660d8cb683"
JUDGED2_CELL_SHA256 = {
    "cpu-w1-k1": "47d73fc1e2836d92299da5ae3221a5d670f421b6196b01fac4f0c9f7f43e7e9f"
}
JUDGED2_SOURCE_SHA256 = "ee619dff6ffbb92f3dfd3be5bf5bb336e506e29be23fc80ba7fb69248ee89022"
JUDGED2_PREVIOUS_JUDGED_SHA256 = "7eae5ceeca340e082e92f78bcec769d97693db452dc6548c437cfe0ed2bb8169"
JUDGED2_CHARGED_SECONDS = 36.47473570803413
JUDGED2_JUDGMENT_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_M1_ATTEMPT5.md"
)
# Third pinned admission: attempt-05 debit only, with in-harness R1 required.
JUDGED3_RECEIPT_SHA256 = "48dcd2ad67be5a44d563d95af6737be275bb73562e666fcc0080dd2544453f05"
JUDGED3_ATTEMPT_SHA256 = "91baa2ea072279b3d741b9db08c9a98cf78406eff77dd9996382a706836f305e"
JUDGED3_LOG_SHA256 = "10adb4136c49b6326631cfd550a572282440016ab03900a44b29d589ce286f0b"
JUDGED3_IDLE_SHA256 = "241da7f0874bc5f703aafcf2594a3e9d7293978a65a0eed2128efab95cc1f9c9"
JUDGED3_CELL_SHA256 = {
    "cpu-w1-k1": "d898970d58943ca52eb1f2dc56c5095e005db15540ca511f52934336982761aa"
}
JUDGED3_DISCARDED_SHA256 = {
    "cpu-w1-k1.discarded-1.json": (
        "a0bbca217e3ea2edd69e54446410803c06295b11c8adb6e139c74c309668bfbe"
    ),
    "cpu-w1-k1.discarded-2.json": (
        "193fa08922aa7b4101c8ba8dbd5164a3a5e7d3e826e1cd9d124443c1da8f6a11"
    ),
    "cpu-w1-k1.discarded-3.json": (
        "f8358cf9cba573ad72fd6da9969a29dad88c427d33344f8af42d29012b7c83ab"
    ),
}
JUDGED3_SOURCE_SHA256 = "b110cb5b364a7045b698d987e60476a18a76432910c38762d2071a5b6ea21a28"
JUDGED3_PREVIOUS_JUDGED_SHA256 = "c515d0cac27f0effc365d113bd990d3d4d3a7b2405fc090856e9eb031a814e4a"
JUDGED3_PREVIOUS_NAMESPACE = "47d211b9fbf1b2313e7d11c36e5966faf6148b7a6509a9de15ec760ce13af3f2"
JUDGED3_CHARGED_SECONDS = 771.5500928750262
JUDGED3_JUDGMENT_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_M1_ATTEMPT6.md"
)
MAX_CELL_RESTARTS = 3
MAX_DISCARDED_RUNS = 8
MIN_CELL_BUDGET_SECONDS = 600
RESTART_POLICY = {
    "trigger": "foreign heavy load during M1",
    "max_per_cell": MAX_CELL_RESTARTS,
    "max_per_attempt": MAX_DISCARDED_RUNS,
    "quiet": {
        "threshold_percent": 20.0,
        "consecutive": 3,
        "interval_seconds": 10.0,
        "max_wait_seconds": 900.0,
    },
}


def refuse_consumed(prior_path: Path, field: str, attempt_number: int) -> None:
    """A historical recovery proof is dispatch-once, including at --check."""
    roots = {prior_path.parent.parent, JUDGED2_JUDGMENT_PATH.parent}
    for root in roots:
        for candidate in root.glob("bl-m1-attempt-*/receipt.json"):
            newer, _ = l1._read_completed_json(candidate)
            if (
                newer.get("schema") == SCHEMA
                and newer.get(field)
                and newer.get("attempt_number", 0) > attempt_number
            ):
                raise ValueError("M1 recovery proof already consumed by a newer receipt")


def cells() -> list[dict[str, Any]]:
    """The four frozen CPU cells, in conservative-to-saturated order."""
    return [
        {"cell_id": f"cpu-w{w}-k{k}", "device": "cpu", "workers": w, "max_pending": k}
        for w, k in ((1, 1), (2, 2), (4, 4), (4, 8))
    ]


def schedule(fixtures: list[dict]) -> list[tuple[int, dict]]:
    """Thirty rounds, rotating the nine roots to interleave fixture/queue phase."""
    return [
        (repeat, fixtures[(index + repeat) % len(fixtures)])
        for repeat in range(REPEATS)
        for index in range(len(fixtures))
    ]


def sources() -> dict[str, str]:
    """Bind math, real Watch path, timing, lease and export implementations."""
    from research.bl_k1.prepare import sources as k1_sources

    root = Path(__file__).resolve().parents[2]
    names = [root / "research/bl_mac_latency/l1.py", root / "research/mac_lease.py"]
    names += list((root / "web/backend").glob("*.py"))
    names += list((root / "research/bl_evidence").glob("*.py"))
    names += [root / "web/backend/served_checkpoint_pins.json"]
    names += list((root / "research/t15_mixed_roster_20261017").glob("*.py"))
    names += list((root / "research/apex_veto_serving_20261001").glob("*.py"))
    names += [root / "research/bl_k1/run_m1.sh", root / "research/bl_served_search/native_ruler.py"]
    return {**k1_sources(), **{str(p.relative_to(root)): digest(p) for p in names}}


def inputs(fixtures_path: Path, reference_path: Path) -> dict[str, Any]:
    """Open immutable Y1 evidence and validate every all/stop batch, untimed."""
    bank, bank_sha = l1._read_completed_json(fixtures_path)
    fixtures = ruler.validate_fixtures(bank)
    reference, ref_sha = l1._read_completed_json(reference_path)
    # Y1 predates additive Mac-L2 source inventories and the full_depth seam.
    # Accept exactly these immutable historical bytes as an ACTION oracle;
    # no old timing or PASS is transferred to the new source closure.
    if bank_sha != Y1_FIXTURES_SHA256 or ref_sha != Y1_REFERENCE_SHA256:
        raise ValueError("M1 requires the hash-pinned historical Y1 oracle")
    for mode in ("all", "exact_stop"):
        records = reference["cells"][mode]["records"]
        if {r["fixture_id"] for r in records} != {f["fixture_id"] for f in fixtures}:
            raise ValueError("Y1 oracle fixture coverage mismatch")
        for record in records:
            fixture = next(f for f in fixtures if f["fixture_id"] == record["fixture_id"])
            if len(record["batches"]) != REPEATS:
                raise ValueError("Y1 oracle batch count mismatch")
            for batch in record["batches"]:
                native_ruler.validate_batch(batch, fixture, mode)
    if reference["fixtures"] != bank or reference["platform"]["label"] != "mac":
        raise ValueError("M1 requires the same nine Y1 fixtures and Mac all-branch oracle")
    actions = {
        row["fixture_id"]: row["batches"][0]["semantic"]["chosen"]
        for row in reference["cells"]["all"]["records"]
    }
    for row in reference["cells"]["all"]["records"]:
        if any(
            batch["semantic"]["chosen"] != actions[row["fixture_id"]] for batch in row["batches"]
        ):
            raise ValueError("Y1 all-branch reference actions drift")
    identity = {
        "sources": sources(),
        "fixtures_sha256": bank_sha,
        "reference_sha256": ref_sha,
        "cells": cells(),
        "repeats": REPEATS,
        "cap_seconds": CAP_SECONDS,
        "contention_seconds": CONTENTION_SECONDS,
    }
    return {
        **identity,
        "namespace": ruler.sha256_bytes(ruler.canonical(identity).encode()),
        "fixtures": fixtures,
        "actions": actions,
    }


def verify_ledger(path: Path, namespace: str) -> dict[str, dict]:
    """Verify paths/sizes/hashes first; refuse foreign/duplicate cell allocations."""
    value, _ = l1._read_completed_json(path)
    if value.get("schema") != SCHEMA + "/ledger" or value.get("namespace") != namespace:
        raise ValueError("foreign M1 ledger namespace")
    result = {}
    allowed = {cell["cell_id"] for cell in cells()}
    for row in value["units"]:
        unit = row["unit_id"]
        if unit not in allowed or unit in result:
            raise ValueError("foreign or duplicate M1 cell allocation")
        candidate = Path(row["path"])
        if (
            not candidate.is_absolute()
            or verify_export(candidate) != row["sha256"]
            or candidate.stat().st_size != row["size"]
        ):
            raise ValueError("M1 ledger size/hash mismatch")
        result[unit] = row
    return result


def make_ledger(out: Path, namespace: str, paths: list[Path]) -> None:
    """Outcome-blind path/size/hash inventory; content is checked by the reducer."""
    units = []
    for path in paths:
        unit_id = path.stem
        if unit_id not in {c["cell_id"] for c in cells()}:
            raise ValueError("foreign M1 allocation filename")
        units.append(
            {
                "unit_id": unit_id,
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": verify_export(path),
            }
        )
    if len({row["unit_id"] for row in units}) != len(units):
        raise ValueError("duplicate M1 ledger allocations")
    export_json(out, {"schema": SCHEMA + "/ledger", "namespace": namespace, "units": units})


def reconciled_precell_stop(path: Path, data: dict) -> dict:
    """Admit only the independently audited 3708063b pre-pool exception.

    This carries debit/retry count, never cell evidence, across the bug-fix
    namespace. It cannot reconcile an arbitrary unverified stop.
    """
    proof, _ = l1._read_completed_json(path)
    prior_path = Path(proof["previous_receipt_path"])
    previous, receipt_sha = l1._read_completed_json(prior_path)
    attempt, attempt_sha = l1._read_completed_json(prior_path.parent / "attempt.json")
    if (
        proof.get("schema") != SCHEMA + "/precell-reconciliation"
        or proof.get("reviewer") != "Codex-V"
        or proof.get("status") != "VERIFIED_PRECELL_MECHANICAL_STOP"
        or proof.get("target_namespace") != data["namespace"]
        or proof.get("previous_receipt_sha256") != receipt_sha
        or receipt_sha != PRECELL_RECEIPT_SHA256
        or attempt_sha != PRECELL_ATTEMPT_SHA256
        or proof.get("previous_attempt_sha256") != attempt_sha
        or digest(prior_path.parent.with_suffix(".log")) != PRECELL_LOG_SHA256
        or proof.get("previous_log_sha256") != PRECELL_LOG_SHA256
        or attempt.get("sources", {}).get("research/bl_k1/m1.py") != PRECELL_SOURCE_SHA256
        or previous.get("schema") != SCHEMA
        or previous.get("namespace") != attempt.get("namespace")
        or previous.get("attempt_number") != 1
        or previous.get("verdict") != "INCOMPLETE"
        or previous.get("verified_stop") is not False
        or previous.get("mechanical_failure") is not True
        or previous.get("reason")
        != "AttributeError: 'AISnake' object has no attribute 'auto_respawn'"
        or previous.get("cells") != []
        or previous.get("completed_unit_exports") != {}
        or proof.get("charged_seconds") != previous.get("charged_seconds")
    ):
        raise ValueError("M1 reconciliation requires the exact audited empty pre-pool failure")
    refuse_consumed(prior_path, "reconciled_precell_stop_sha256", 1)
    return previous


def judged_recovery(path: Path, data: dict) -> dict:
    """Carry only debit and attempt count from an exact judge-pinned stop."""
    error = "M1 judged recovery requires the exact audited attempt-03 foreign-load stop"
    try:
        proof, _ = l1._read_completed_json(path)
        second = proof.get("judge") == "Fable-M1-attempt5"
        third = proof.get("judge") == "Fable-M1-attempt6"
        if second:
            error = "M1 judged recovery requires the exact audited attempt-04 foreign-load stop"
        if third:
            error = "M1 judged recovery requires the exact audited attempt-05 foreign-load stop"
        pins = (
            (
                JUDGED3_RECEIPT_SHA256,
                JUDGED3_ATTEMPT_SHA256,
                JUDGED3_LOG_SHA256,
                JUDGED3_CELL_SHA256,
                JUDGED3_SOURCE_SHA256,
                JUDGED3_CHARGED_SECONDS,
                4,
            )
            if third
            else (
                (
                    JUDGED2_RECEIPT_SHA256,
                    JUDGED2_ATTEMPT_SHA256,
                    JUDGED2_LOG_SHA256,
                    JUDGED2_CELL_SHA256,
                    JUDGED2_SOURCE_SHA256,
                    JUDGED2_CHARGED_SECONDS,
                    3,
                )
                if second
                else (
                    JUDGED_RECEIPT_SHA256,
                    JUDGED_ATTEMPT_SHA256,
                    JUDGED_LOG_SHA256,
                    JUDGED_CELL_SHA256,
                    JUDGED_SOURCE_SHA256,
                    JUDGED_CHARGED_SECONDS,
                    2,
                )
            )
        )
        receipt_pin, attempt_pin, log_pin, cell_pins, source_pin, debit, number = pins
        prior_path = Path(proof["previous_receipt_path"])
        previous, receipt_sha = l1._read_completed_json(prior_path)
        attempt, attempt_sha = l1._read_completed_json(prior_path.parent / "attempt.json")
        cell, cell_sha = l1._read_completed_json(prior_path.parent / "cpu-w1-k1.json")
        log_path = (
            Path(proof["previous_log_path"])
            if second or third
            else prior_path.parent.with_suffix(".log")
        )
        if (
            proof.get("schema") != SCHEMA + "/judged-recovery"
            or proof.get("judge")
            != (
                "Fable-M1-attempt6"
                if third
                else "Fable-M1-attempt5" if second else "Fable-M1-attempt4"
            )
            or proof.get("status")
            != (
                "JUDGED_FOREIGN_LOAD_STOP_3"
                if third
                else "JUDGED_FOREIGN_LOAD_STOP_2" if second else "JUDGED_FOREIGN_LOAD_STOP"
            )
            or proof.get("target_namespace") != data["namespace"]
            or proof.get("judgment_sha256") != digest(Path(proof["judgment_path"]))
            or proof.get("next_attempt_number") != number + 1
            or proof.get("final_attempt") is not True
            or receipt_sha != receipt_pin
            or proof.get("previous_receipt_sha256") != receipt_sha
            or attempt_sha != attempt_pin
            or proof.get("previous_attempt_sha256") != attempt_sha
            or digest(log_path) != log_pin
            or proof.get("previous_log_sha256") != log_pin
            or cell_sha != cell_pins["cpu-w1-k1"]
            or proof.get("previous_cell_sha256") != cell_pins
            or attempt.get("sources", {}).get("research/bl_k1/m1.py") != source_pin
            or previous.get("schema") != SCHEMA
            or previous.get("namespace") != attempt.get("namespace")
            or previous.get("attempt_number") != number
            or previous.get("verdict") != "INCOMPLETE"
            or previous.get("verified_stop") is not True
            or previous.get("mechanical_failure") is not True
            or previous.get("completed_unit_exports") != {}
            or previous.get("charged_seconds") != debit
            or proof.get("charged_seconds") != debit
            or len(previous["cells"]) != 1
            or previous["cells"][0].get("cell_id") != "cpu-w1-k1"
            or previous["cells"][0].get("complete") is not False
            or cell.get("stop_reason") != "foreign heavy load during M1"
            or cell["summary"].get("complete") is not False
        ):
            raise ValueError(error)
        if second or third:
            prior_judged = (
                JUDGED3_PREVIOUS_JUDGED_SHA256 if third else JUDGED2_PREVIOUS_JUDGED_SHA256
            )
            if (
                Path(proof["judgment_path"])
                != (JUDGED3_JUDGMENT_PATH if third else JUDGED2_JUDGMENT_PATH)
                or proof.get("restart_policy_required") is not True
                or not RESTART_POLICY
                or data.get("sources", {}).get("research/bl_k1/m1.py") != digest(Path(__file__))
                or previous.get("final_attempt") is not True
                or previous.get("judged_recovery_sha256") != prior_judged
                or proof.get("previous_judged_recovery_sha256") != prior_judged
                or proof.get("previous_namespace") != previous["namespace"]
                or proof.get("remaining_seconds") != CAP_SECONDS - debit
                or proof.get("transferred_cell_or_timing_evidence") is not False
                or proof.get("operator_dispatch_once") is not True
            ):
                raise ValueError(error)
        elif previous.get("reconciled_precell_stop_sha256") != JUDGED_PRECELL_SHA256:
            raise ValueError(error)
        if third:
            _, idle_sha = l1._read_completed_json(prior_path.parent / "idle.json")
            discarded = {
                name: l1._read_completed_json(prior_path.parent / name)[1]
                for name in JUDGED3_DISCARDED_SHA256
            }
            if (
                idle_sha != JUDGED3_IDLE_SHA256
                or proof.get("previous_idle_sha256") != idle_sha
                or discarded != JUDGED3_DISCARDED_SHA256
                or proof.get("previous_discarded_sha256") != discarded
                or previous.get("namespace") != JUDGED3_PREVIOUS_NAMESPACE
                or previous.get("failure") is not None
                or len(previous.get("discarded_runs", [])) != 3
                or any(
                    r.get("stop_reason") != RESTART_POLICY["trigger"]
                    for r in previous["discarded_runs"]
                )
                or len(previous.get("quiet_waits", [])) != 4
                or any(q.get("quiet") is not True for q in previous["quiet_waits"])
                or proof.get("carry_forward_required") is not True
                or "carried_violation" not in Path(__file__).read_text()
                or not callable(carried_violation)
            ):
                raise ValueError(error)
        refuse_consumed(prior_path, "judged_recovery_sha256", number)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(error) from exc
    return previous


def validate_sample(payload: dict, fixture: dict, action: int) -> None:
    """Counters from executed branches, plus chosen-action identity, fail closed."""
    native_ruler.validate_batch(payload, fixture, "exact_stop")
    if payload["semantic"]["chosen"] != action:
        raise ValueError("INVALID chosen action differs from Y1 all-branch reference")
    if payload["execution_environment"]["torch_num_interop_threads"] != 1:
        raise ValueError("INVALID worker interop thread count")


def summarize(cell: dict, samples: list[dict], ticks: list[dict], complete: bool) -> dict:
    """Frozen M1 bars; dense caps are descriptive, never the admission rule."""
    errors = [s["error"] for s in samples if "error" in s]
    warm = [s for s in samples if s["phase"] == "warm" and "error" not in s and "S_seconds" in s]
    cold = [s for s in samples if s["phase"] == "cold"]
    by_fixture = {
        s["fixture_id"]: l1.distribution(
            row["S_seconds"] for row in warm if row["fixture_id"] == s["fixture_id"]
        )
        for s in warm
    }
    doom = by_fixture.get("frozen-800", {}).get("p99_seconds")
    update = l1.distribution(row["update_seconds"] for row in ticks)
    overrun = sum(row["overrun"] for row in ticks)
    failed = overrun > 0 or (update["p95_seconds"] is not None and update["p95_seconds"] > 0.011688)
    if cell["workers"] == 4 and doom is not None and doom > 5.0:
        failed = True
    counts_complete = len(by_fixture) == 9 and all(
        v["count"] == REPEATS for v in by_fixture.values()
    )
    completed = complete and counts_complete and len(cold) == cell["max_pending"] and bool(ticks)
    native_totals = {
        key: sum(s.get("payload", {}).get("native_v8", {}).get(key, 0) for s in warm)
        for key in ("kernel_rows", "stock_rows", "fallback_rows", "calls")
    }
    return {
        "native_v8_totals_warm": native_totals,
        "verdict": (
            "INVALID"
            if errors
            else ("INCOMPLETE" if not completed else ("BLOCKED" if failed else "PASS"))
        ),
        "complete": completed,
        "errors": errors,
        "cold_process_wave": cold,
        "S_seconds": l1.distribution(s["S_seconds"] for s in warm),
        "per_fixture": by_fixture,
        "doom_fixture_id": "frozen-800",
        "doom_p99_seconds": doom,
        "tick_overruns": overrun,
        "update": update,
        "dense_caps_seconds_report_only": {"W2": 0.830, "W3": 1.250, "W4": 1.458},
        "scope": "M1 admits construction only; no L1 qualification or release",
    }


def _contention(stop: Any, seconds: float, output: Any) -> None:
    """One declared 100%-of-one-core busy block; cooperative exit only."""
    started, cpu = time.monotonic_ns(), time.process_time()
    limit = time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < limit:
        pass
    output.put(
        {
            "started_ns": started,
            "finished_ns": time.monotonic_ns(),
            "cpu_seconds": time.process_time() - cpu,
            "pid": os.getpid(),
        }
    )


def foreign_load(screen: dict, coordinator_pid: int) -> list[dict]:
    """Use the complete ps ancestry to distinguish owned load from foreign jobs."""
    parents = {}
    for line in screen.get("processes", "").splitlines()[1:]:
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            parents[int(fields[0])] = int(fields[1])
    owned = {coordinator_pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in owned}
        if children <= owned:
            break
        owned |= children
    return [row for row in screen.get("other_heavy_processes", []) if row["pid"] not in owned]


class HostMonitor:
    """Sample guard/process/thermal state off the tick thread; fail closed."""

    def __init__(self, observe: Callable[[], dict]) -> None:
        self.stop = threading.Event()
        self.rows: list[dict] = []
        self.error: str | None = None
        self.observe = observe
        self.thread = threading.Thread(target=self._loop, daemon=False)

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                row = self.observe()
                row["coordinator_pid"] = os.getpid()
                row["foreign_heavy_processes"] = foreign_load(row["process_screen"], os.getpid())
                self.rows.append(row)
                if row["foreign_heavy_processes"]:
                    self.error = "foreign heavy load during M1"
                    break
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                break
            self.stop.wait(2.0)

    def close(self) -> None:
        self.stop.set()
        self.thread.join()


def new_watch(seed: int) -> tuple[Any, dict]:
    """Use the actual product Watch6 session; never a terminal evaluator clone."""
    from research.t15_mixed_roster_20261017.episodes import new_session, roster_identity

    # Explicitly disable the optional assisted Path B before constructing Watch.
    previous_pathb = os.environ.get("SNAKE_SERVE_PATHB")
    os.environ["SNAKE_SERVE_PATHB"] = "0"
    try:
        session = new_session(seed, 6)
    finally:
        if previous_pathb is None:
            os.environ.pop("SNAKE_SERVE_PATHB", None)
        else:
            os.environ["SNAKE_SERVE_PATHB"] = previous_pathb
    if getattr(session, "pathb", None) is not None or any(
        getattr(snake, "action_transform", None) is not None for snake in session.game.snakes
    ):
        raise ValueError("M1 refuses assisted Watch or an action transform")
    session.set_speed(24)
    # Watch.step passes allow_respawn=True. GameState's per-snake hook is
    # opt-out: ordinary AISnake objects have no auto_respawn attribute.
    from web.backend.session import MODE_WATCH

    if (
        session.mode != MODE_WATCH
        or not session.playing
        or session.speed != 24
        or len(session.game.snakes) != 6
        or any(not getattr(snake, "auto_respawn", True) for snake in session.game.snakes)
    ):
        raise ValueError("M1 requires real continuous-respawn mixed Watch6 at 24fps")
    return session, roster_identity(session)


def smoke(data: dict) -> dict:
    """Real, unpaced API/IPC check; never a cell, lease, or timing export."""
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    session, roster = new_watch(2026102300)

    def tick() -> None:
        before = int(session.game.frame)
        session.step()
        snapshot = session.snapshot()
        if int(session.game.frame) != before + 1 or session.last_error or not snapshot:
            raise RuntimeError(f"M1 smoke Watch tick failed: {session.last_error!r}")

    for _ in range(3):
        tick()
    # Exercise the actual death/respawn lifecycle for both policy classes,
    # including the optional-attribute default that the former mocks hid.
    targets = session.game.snakes[:2]
    for snake in targets:
        snake.die()
        snake.respawn_timer = 1
    tick()
    if any(not snake.is_alive or snake.respawn_timer != 0 for snake in targets):
        raise RuntimeError("M1 smoke did not respawn hero and opponent through Watch.step")
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    fixture = next(f for f in data["fixtures"] if f["fixture_id"] == "frozen-800")
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=l1._worker_init,
        initargs=("cpu", stop, time.monotonic() + 120),
    )
    try:
        future = pool.submit(l1._worker_batch, fixture)
        for _ in range(3):
            tick()
        payload = future.result(timeout=120)
        validate_sample(payload, fixture, data["actions"][fixture["fixture_id"]])
        if payload["worker_pid"] == os.getpid() or not payload["placement"]:
            raise RuntimeError("M1 smoke requires real spawned policy placement")
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    # Worker internals include diagnostic clocks. Discard all of them: this
    # output has no S/update distributions, deadlines, bars or cell payloads.
    return {
        "status": "SMOKE_OK",
        "timing_evidence": False,
        "timed_cells_run": False,
        "lease_acquired": False,
        "watch_ticks": 7,
        "watch_roster": roster["watch_roster"],
        "snakes": len(session.game.snakes),
        "respawned_ids": [snake.id for snake in targets],
        "workers": 1,
        "max_pending": 1,
        "fixture_id": fixture["fixture_id"],
        "chosen_action": payload["semantic"]["chosen"],
        "native_v8": payload["native_v8"],
        "worker_threads": payload["worker_threads"],
        "verified_stop": True,
    }


def timed_tick(session: Any) -> float:
    """Time the ordinary game update, retaining the full real step/snapshot load."""
    from research.t15_mixed_roster_20261017.measurements import _timed_tick

    update_ms, _ = _timed_tick(session)
    return update_ms / 1000


def run_cell(
    cell: dict,
    data: dict,
    deadline: float,
    observe: Callable[[], dict],
    stop_state: dict | None = None,
) -> dict:
    """Cold pool start and 270 warm submissions under one absolute Watch clock."""
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    session, roster = new_watch(2026102300 + cells().index(cell))
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    monitor = HostMonitor(observe)
    if stop_state is not None:
        stop_state["verified_stop"] = False
    pool = None
    monitor_started = False
    pending: dict[Any, tuple] = {}
    samples: list[dict] = []
    ticks: list[dict] = []
    fixtures = data["fixtures"]
    cold_jobs = [(i, fixtures[i % 9]) for i in range(cell["max_pending"])]
    warm_jobs = schedule(fixtures)
    jobs = iter(cold_jobs)
    phase, admitted, max_pending = "cold", 0, 0
    reason = None
    contention = None
    contention_rows = []
    load_stop, load_queue = context.Event(), None
    epoch = time.monotonic_ns()
    try:
        # Keep resource setup inside cleanup coverage, including partial setup.
        # Process construction is cold; no worker starts until the clock is live.
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=cell["workers"],
            mp_context=context,
            initializer=l1._worker_init,
            initargs=("cpu", stop, deadline),
        )
        load_queue = context.Queue()
        epoch = time.monotonic_ns()  # preserve the original successful-cell clock placement
        monitor.thread.start()
        monitor_started = True
        while True:
            tick_index = len(ticks)
            target = epoch + tick_index * 1_000_000_000 // 24
            if time.monotonic() >= deadline:
                reason = "mechanical wall cap"
                break
            if monitor.error:
                reason = monitor.error
                break
            now = time.monotonic_ns()
            if now < target:
                time.sleep((target - now) / 1e9)
            tick_started = time.monotonic_ns()
            update_seconds = timed_tick(session)
            for future in list(pending):
                if not future.done():
                    continue
                repeat, fixture, admission, sample_phase = pending.pop(future)
                sample = {
                    "phase": sample_phase,
                    "repeat": repeat,
                    "fixture_id": fixture["fixture_id"],
                    "admitted_ns": admission,
                }
                try:
                    payload = future.result()
                    visible = time.monotonic_ns()  # decoded, first coordinator observation
                    sample.update(
                        payload=payload, visible_ns=visible, S_seconds=(visible - admission) / 1e9
                    )
                    validate_sample(payload, fixture, data["actions"][fixture["fixture_id"]])
                except Exception as exc:
                    if str(exc) == "measurement stopped before next rollout step":
                        sample["mechanical_error"] = str(exc)
                        reason = "mechanical wall cap in worker"
                    else:
                        sample["error"] = f"{type(exc).__name__}: {exc}"
                samples.append(sample)
            if any("error" in s for s in samples):
                reason = "invalid worker execution"
            if not pending and admitted == len(cold_jobs) and phase == "cold":
                phase, jobs, admitted = "warm", iter(warm_jobs), 0
                if cell["cell_id"] == "cpu-w4-k4":
                    contention = context.Process(
                        target=_contention, args=(load_stop, CONTENTION_SECONDS, load_queue)
                    )
                    contention.start()
            total = len(cold_jobs) if phase == "cold" else len(warm_jobs)
            if reason is None:
                while len(pending) < cell["max_pending"] and admitted < total:
                    repeat, fixture = next(jobs)
                    admission = time.monotonic_ns()
                    pending[pool.submit(l1._worker_batch, fixture)] = (
                        repeat,
                        fixture,
                        admission,
                        phase,
                    )
                    admitted += 1
                    max_pending = max(max_pending, len(pending))
            finished = time.monotonic_ns()
            ticks.append(
                {
                    "target_ns": target,
                    "tick_started_ns": tick_started,
                    "finished_ns": finished,
                    "update_seconds": update_seconds,
                    "overrun": finished >= epoch + (tick_index + 1) * 1_000_000_000 // 24,
                }
            )
            if tick_index % (24 * 60) == 0:
                print(
                    json.dumps(
                        {
                            "cell": cell["cell_id"],
                            "phase": phase,
                            "samples": len(samples),
                            "ticks": len(ticks),
                        }
                    ),
                    flush=True,
                )
            if reason or (
                phase == "warm"
                and admitted == total
                and not pending
                and (contention is None or not contention.is_alive())
            ):
                break
    except (Exception, KeyboardInterrupt) as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        load_stop.set()
        try:
            try:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
            finally:
                if contention is not None:
                    contention.join()
                    try:
                        contention_rows.append(load_queue.get(timeout=5))
                    except queue.Empty:
                        reason = reason or "missing contention process telemetry"
        finally:
            try:
                if load_queue is not None:
                    load_queue.close()
            finally:
                if monitor_started:
                    monitor.close()
    if stop_state is not None:
        stop_state["verified_stop"] = True
    reason = reason or monitor.error
    if not monitor.rows:
        reason = reason or "missing host telemetry"
    result = summarize(cell, samples, ticks, reason is None)

    def mark_incomplete() -> None:
        result["complete"] = False
        if result["verdict"] != "INVALID":
            result["verdict"] = "INCOMPLETE"

    if contention_rows:
        block = contention_rows[0]
        if block["cpu_seconds"] < 0.5 * (block["finished_ns"] - block["started_ns"]) / 1e9:
            mark_incomplete()
    pids = {s["payload"]["worker_pid"] for s in samples if "payload" in s}
    if len(pids) != cell["workers"]:
        mark_incomplete()
    if cell["cell_id"] == "cpu-w4-k4" and (
        not contention_rows
        or contention_rows[0]["finished_ns"] - contention_rows[0]["started_ns"]
        < CONTENTION_SECONDS * 1_000_000_000
    ):
        mark_incomplete()
    return {
        "schema": SCHEMA + "/cell",
        "namespace": data["namespace"],
        **cell,
        "summary": result,
        "stop_reason": reason,
        "samples": samples,
        "ticks": ticks,
        "epoch_ns": epoch,
        "clock": "T(n)=t0+floor(n*1e9/24)",
        "actual_max_pending": max_pending,
        "roster": roster,
        "host": monitor.rows,
        "contention": contention_rows,
        "pending_at_stop": len(pending),
    }


def audit_unit(row: dict, data: dict) -> None:
    """Recompute a resumed cell from raw payloads/ticks instead of trusting PASS."""
    cell = next((c for c in cells() if c["cell_id"] == row.get("cell_id")), None)
    if (
        cell is None
        or row.get("namespace") != data["namespace"]
        or any(row.get(key) != value for key, value in cell.items())
    ):
        raise ValueError("foreign M1 cell content")
    if not row.get("mac_lease", {}).get("id"):
        raise ValueError("missing M1 cell lease")
    fixture_map = {f["fixture_id"]: f for f in data["fixtures"]}
    seen = set()
    events = []
    ordered_allocations = []
    for sample in row["samples"]:
        key = (sample["phase"], sample["repeat"], sample["fixture_id"])
        if key in seen or sample["phase"] not in ("cold", "warm"):
            raise ValueError("duplicate/foreign M1 sample")
        seen.add(key)
        fixture = fixture_map[sample["fixture_id"]]
        validate_sample(sample["payload"], fixture, data["actions"][fixture["fixture_id"]])
        start, end = sample["admitted_ns"], sample["visible_ns"]
        if type(start) is not int or type(end) is not int or start < row["epoch_ns"]:
            raise ValueError("M1 sample outside Watch clock")
        events.extend(((start, 1), (end, -1)))
        ordered_allocations.append((start, key))
        payload = sample["payload"]
        if (
            not start <= payload["worker_started_ns"] <= payload["worker_completed_ns"] <= end
            or sample["S_seconds"] != (end - start) / 1e9
        ):
            raise ValueError("M1 visibility timestamps disagree")
    warm_expected = {("warm", repeat, f["fixture_id"]) for repeat, f in schedule(data["fixtures"])}
    cold_expected = {
        ("cold", i, data["fixtures"][i % 9]["fixture_id"]) for i in range(cell["max_pending"])
    }
    if seen != warm_expected | cold_expected:
        raise ValueError("incomplete M1 sample allocations")
    expected_order = [
        ("cold", i, data["fixtures"][i % 9]["fixture_id"]) for i in range(cell["max_pending"])
    ]
    expected_order += [
        ("warm", repeat, f["fixture_id"]) for repeat, f in schedule(data["fixtures"])
    ]
    if [key for _, key in sorted(ordered_allocations)] != expected_order:
        raise ValueError("M1 interleaved admission order changed")
    pending = high_water = 0
    for _, delta in sorted(events):
        pending += delta
        if pending < 0:
            raise ValueError("M1 visibility without an outstanding admission")
        high_water = max(high_water, pending)
    if pending or high_water != cell["max_pending"]:
        raise ValueError("M1 outstanding admission high-water differs from K")
    if (
        not row["ticks"]
        or max(s["visible_ns"] for s in row["samples"]) > row["ticks"][-1]["finished_ns"]
    ):
        raise ValueError("M1 ticks do not cover all result residence")
    previous_finish = row["epoch_ns"]
    for index, tick in enumerate(row["ticks"]):
        target = row["epoch_ns"] + index * 1_000_000_000 // 24
        next_target = row["epoch_ns"] + (index + 1) * 1_000_000_000 // 24
        if (
            tick["target_ns"] != target
            or not max(target, previous_finish) <= tick["tick_started_ns"] <= tick["finished_ns"]
            or tick["overrun"] != (tick["finished_ns"] >= next_target)
            or not 0
            <= tick["update_seconds"]
            <= (tick["finished_ns"] - tick["tick_started_ns"]) / 1e9
        ):
            raise ValueError("M1 tick clock/overrun mismatch")
        previous_finish = tick["finished_ns"]
    recomputed = summarize(cell, row["samples"], row["ticks"], row["stop_reason"] is None)
    if recomputed != row["summary"] or not recomputed["complete"]:
        raise ValueError("incomplete or changed M1 cell summary")
    pids = {sample["payload"]["worker_pid"] for sample in row["samples"]}
    if any(
        foreign_load(host["process_screen"], host["coordinator_pid"])
        for host in row.get("host", [])
    ):
        raise ValueError("M1 foreign heavy load attested")
    if len(pids) != cell["workers"] or not row.get("host"):
        raise ValueError("M1 worker/host attestation missing")
    if cell["cell_id"] == "cpu-w4-k4":
        if len(row.get("contention", [])) != 1:
            raise ValueError("M1 contention block missing")
        block = row["contention"][0]
        elapsed = (block["finished_ns"] - block["started_ns"]) / 1e9
        if (
            elapsed < CONTENTION_SECONDS
            or block["cpu_seconds"] < 0.5 * elapsed
            or not row["epoch_ns"]
            <= block["started_ns"]
            < block["finished_ns"]
            <= row["ticks"][-1]["finished_ns"]
        ):
            raise ValueError("M1 contention load below 50% of one core")
    elif row.get("contention"):
        raise ValueError("undeclared M1 contention block")
    if row["actual_max_pending"] != cell["max_pending"]:
        raise ValueError("M1 pending wave was not saturated")


def reduce_units(units: list[dict], namespace: str, data: dict | None = None) -> dict:
    """One final look over four same-namespace, complete, distinct cells."""
    ids = [row["cell_id"] for row in units]
    if len(ids) != len(set(ids)) or any(row["namespace"] != namespace for row in units):
        raise ValueError("duplicate or foreign M1 cell")
    if set(ids) - {c["cell_id"] for c in cells()}:
        raise ValueError("foreign M1 cell")
    for row in units:
        if row["summary"].get("decided") is not True and row["summary"]["complete"]:
            if data is None:
                raise ValueError("M1 reduction requires raw evidence inputs")
            audit_unit(row, data)
    statuses = [
        (
            "BLOCKED"
            if row["summary"].get("decided") is True
            else (
                row["summary"]["verdict"]
                if row["summary"]["complete"] or row["summary"]["verdict"] == "INVALID"
                else "INCOMPLETE"
            )
        )
        for row in units
    ]
    verdict = (
        "INVALID"
        if "INVALID" in statuses
        else (
            "INCOMPLETE"
            if len(units) != 4 or "INCOMPLETE" in statuses
            else ("BLOCKED" if "BLOCKED" in statuses else "PASS")
        )
    )
    return {
        "schema": SCHEMA,
        "namespace": namespace,
        "verdict": verdict,
        "cells": [{"cell_id": row["cell_id"], **row["summary"]} for row in units],
        "lead_source_cell_id": "cpu-w4-k4",
        "scope": "construction admission only; B8 serving unchanged",
    }


def wait_quiet(
    observe: Callable[[], dict],
    threshold_percent: float = 20.0,
    consecutive: int = 3,
    interval_seconds: float = 10.0,
    max_wait_seconds: float = 900.0,
) -> dict:
    """Record consecutive quiet process screens using the unchanged ownership walk."""
    started = time.monotonic_ns()
    deadline = time.monotonic() + max_wait_seconds
    samples = []
    streak = 0
    while time.monotonic() < deadline:
        row = observe()
        screen = dict(row["process_screen"])
        heavy = []
        for line in screen["processes"].splitlines()[1:]:
            fields = line.strip().split(None, 3)
            if len(fields) == 4 and float(fields[2]) >= threshold_percent:
                heavy.append(
                    {
                        "pid": int(fields[0]),
                        "ppid": int(fields[1]),
                        "cpu_percent": float(fields[2]),
                        "command": fields[3],
                    }
                )
        screen["other_heavy_processes"] = heavy
        foreign = foreign_load(screen, os.getpid())
        samples.append(
            {**row, "sampled_ns": time.monotonic_ns(), "foreign_heavy_processes": foreign}
        )
        streak = 0 if foreign else streak + 1
        if streak >= consecutive and time.monotonic() < deadline:
            return {
                "started_ns": started,
                "finished_ns": time.monotonic_ns(),
                "samples": samples,
                "quiet": True,
            }
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval_seconds, remaining))
    return {
        "started_ns": started,
        "finished_ns": time.monotonic_ns(),
        "samples": samples,
        "quiet": False,
    }


def audit_discarded(out_dir: Path, receipt: dict, units: list[dict]) -> None:
    """Re-audit every immutable discard; no outcome-based exclusions are legal."""
    runs = receipt["discarded_runs"]
    expected = {f"{r['cell_id']}.discarded-{r['n']}.json" for r in runs}
    if len(expected) != len(runs) or expected != {
        p.name for p in out_dir.glob("*.discarded-*.json")
    }:
        raise ValueError("missing, duplicate or unrecorded M1 discard")
    counts = {}
    retained_epochs = {row["epoch_ns"] for row in units}
    if len(retained_epochs) != len(units):
        raise ValueError("shared M1 retained epoch")
    for row in units:
        if row["summary"].get("decided") is True:
            saved, _ = l1._read_completed_json(out_dir / f"{row['cell_id']}.json")
            carry = carried_violation(saved)
            expected_carry = {
                **carry,
                "run_n": 1 + sum(r["cell_id"] == row["cell_id"] for r in runs),
                "rule": "R1",
            }
            if (
                saved != row
                or row["namespace"] != receipt["namespace"]
                or row["stop_reason"] != RESTART_POLICY["trigger"]
                or row["summary"].get("complete") is not False
                or row["summary"].get("verdict") != "BLOCKED"
                or row["summary"].get("carried_violation") != expected_carry
                or not (carry["carried_overruns"] or carry["carried_doom"])
                or not row["host"][-1]["foreign_heavy_processes"]
            ):
                raise ValueError("invalid M1 decided-row provenance")
    for record in runs:
        unit_id, n = record["cell_id"], record["n"]
        row, _ = l1._read_completed_json(out_dir / f"{unit_id}.discarded-{n}.json")
        if (
            row.get("stop_reason") != RESTART_POLICY["trigger"]
            or record.get("stop_reason") != RESTART_POLICY["trigger"]
        ):
            raise ValueError("non-foreign discard")
        counts[unit_id] = counts.get(unit_id, 0) + 1
        if (
            unit_id not in {c["cell_id"] for c in cells()}
            or n != counts[unit_id]
            or counts[unit_id] > MAX_CELL_RESTARTS
            or len(runs) > MAX_DISCARDED_RUNS
            or row.get("namespace") != receipt["namespace"]
            or row.get("cell_id") != unit_id
            or row["summary"].get("complete") is not False
            or row["summary"].get("verdict") == "INVALID"
            or not row.get("host")
            or not row["host"][-1].get("foreign_heavy_processes")
            or row["host"][-1]["foreign_heavy_processes"] != record["foreign_heavy_processes"]
            or row["epoch_ns"] in retained_epochs
            or record["ticks"] != len(row["ticks"])
            or record["samples_count"] != len(row["samples"])
            or not record["started_ns"] <= row["epoch_ns"] <= record["finished_ns"]
        ):
            raise ValueError("invalid M1 discard or shared epoch")
        retained_epochs.add(row["epoch_ns"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ledger", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-ledger", type=Path)
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--unit", type=Path, action="append", default=[])
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--previous-attempt", type=Path)
    recovery.add_argument("--reconciled-precell-stop", type=Path)
    recovery.add_argument("--judged-recovery", type=Path)
    args = parser.parse_args()
    data = inputs(args.fixtures, args.reference)
    if args.smoke:
        if any(
            (
                args.out,
                args.ledger,
                args.previous_attempt,
                args.reconciled_precell_stop,
                args.judged_recovery,
                args.unit,
            )
        ):
            parser.error("--smoke is standalone; no attempt, ledger or recovery options")
        print(json.dumps(smoke(data)))
        return
    if args.make_ledger:
        make_ledger(args.make_ledger, data["namespace"], args.unit)
        return
    ledger = verify_ledger(args.ledger, data["namespace"]) if args.ledger else {}
    previous = None
    if args.previous_attempt:
        previous, _ = l1._read_completed_json(args.previous_attempt)
        if (
            previous.get("schema") != SCHEMA
            or previous.get("namespace") != data["namespace"]
            or previous.get("attempt_number") != 1
            or previous.get("verdict") != "INCOMPLETE"
            or previous.get("verified_stop") is not True
            or previous.get("mechanical_failure") is not True
        ):
            raise ValueError("M1 recovery requires one verified mechanical stop in this namespace")
        if ledger != previous.get("completed_unit_exports"):
            raise ValueError("recovery ledger must skip exactly all verified completed cells")
    elif args.reconciled_precell_stop:
        previous = reconciled_precell_stop(args.reconciled_precell_stop, data)
        if ledger:
            raise ValueError("reconciled pre-cell stop requires an empty new ledger")
    elif args.judged_recovery:
        previous = judged_recovery(args.judged_recovery, data)
        if ledger:
            raise ValueError("judged recovery requires an empty new ledger")
    elif ledger:
        raise ValueError("nonempty recovery ledger requires --previous-attempt")
    charged = 0.0 if previous is None else previous["charged_seconds"]
    if type(charged) not in (int, float) or not 0 <= charged < CAP_SECONDS:
        raise ValueError("M1 cumulative cap exhausted or invalid")
    if args.check:
        print(
            json.dumps(
                {
                    "status": "CHECK_ONLY",
                    "namespace": data["namespace"],
                    "cells": cells(),
                    "skipped": list(ledger),
                    "base_seconds": BASE_SECONDS,
                    "cap_seconds": CAP_SECONDS,
                    "remaining_seconds": CAP_SECONDS - charged,
                    "attempt_number": 1 if previous is None else previous["attempt_number"] + 1,
                    "timed_cells_run": False,
                }
            )
        )
        return
    if args.out is None or args.ledger is None:
        parser.error("dispatch requires --out and an explicit ledger (empty on attempt 1)")
    if not args.out.is_absolute():
        raise ValueError("M1 attempt output must be absolute")
    if args.out.exists():
        raise FileExistsError("M1 requires a fresh attempt directory")
    from research.bl_mac_latency.host import check_guard, process_screen, snapshot
    from research.mac_lease import held

    def observe() -> dict:
        return {
            **check_guard(),
            "process_screen": process_screen(),
            "load_average": list(os.getloadavg()),
        }

    with held(
        "claude-m1", "PLAN_v3 M1 native co-residency", seconds=CAP_SECONDS - charged
    ) as lease:
        started = time.monotonic()
        deadline = started + CAP_SECONDS - charged
        idle = snapshot()
        if not idle["quiet"]:
            raise RuntimeError("M1 requires an idle Mac receipt; foreign load observed")
        quiet_waits = [
            wait_quiet(
                observe,
                **{
                    **RESTART_POLICY["quiet"],
                    "max_wait_seconds": min(900.0, max(0.0, deadline - time.monotonic())),
                },
            )
        ]
        if not quiet_waits[0]["quiet"]:
            raise RuntimeError("host not quiet for M1 restart")
        args.out.mkdir(parents=True)
        export_json(args.out / "idle.json", idle)
        export_json(
            args.out / "attempt.json",
            {
                "namespace": data["namespace"],
                "sources": data["sources"],
                "mac_lease": lease.receipt,
                "ledger_sha256": digest(args.ledger),
                "base_seconds": BASE_SECONDS,
                "cap_seconds": CAP_SECONDS,
                "mechanical_retry": "bounded whole-cell restart on foreign load only",
                "restart_policy": RESTART_POLICY,
                "reconciled_precell_stop_sha256": (
                    digest(args.reconciled_precell_stop) if args.reconciled_precell_stop else None
                ),
                "judged_recovery_sha256": (
                    digest(args.judged_recovery) if args.judged_recovery else None
                ),
            },
        )
        units = []
        discarded_runs = []
        completed_exports = dict(ledger)
        stop_state = {"verified_stop": True}
        failure = None
        invalid = False
        try:
            for cell in cells():
                unit_id = cell["cell_id"]
                if unit_id in ledger:
                    row, _ = l1._read_completed_json(Path(ledger[unit_id]["path"]))
                    audit_unit(row, data)
                    if row["namespace"] != data["namespace"] or row["cell_id"] != unit_id:
                        raise ValueError("foreign M1 unit content")
                    if not row["summary"]["complete"]:
                        raise ValueError("cannot skip an incomplete M1 cell")
                else:
                    restarts = 0
                    while True:
                        run_started = time.monotonic_ns()
                        row = run_cell(cell, data, deadline, observe, stop_state)
                        run_finished = time.monotonic_ns()
                        row["mac_lease"] = lease.receipt
                        if (
                            row["stop_reason"] == RESTART_POLICY["trigger"]
                            and row["summary"]["complete"] is False
                            and row["summary"]["verdict"] != "INVALID"
                            and stop_state["verified_stop"]
                        ):
                            carry = carried_violation(row)
                            if carry["carried_overruns"] or carry["carried_doom"]:
                                row["summary"].update(
                                    carried_violation={
                                        **carry,
                                        "run_n": restarts + 1,
                                        "rule": "R1",
                                    },
                                    verdict="BLOCKED",
                                    decided=True,
                                )
                                break
                        if not (
                            row["stop_reason"] == RESTART_POLICY["trigger"]
                            and row["summary"]["complete"] is False
                            and row["summary"]["verdict"] != "INVALID"
                            and stop_state["verified_stop"]
                            and restarts < MAX_CELL_RESTARTS
                            and len(discarded_runs) < MAX_DISCARDED_RUNS
                            and time.monotonic() < deadline - MIN_CELL_BUDGET_SECONDS
                        ):
                            break
                        restarts += 1
                        export_json(args.out / f"{unit_id}.discarded-{restarts}.json", row)
                        discarded_runs.append(
                            {
                                "cell_id": unit_id,
                                "n": restarts,
                                "stop_reason": row["stop_reason"],
                                "foreign_heavy_processes": row["host"][-1][
                                    "foreign_heavy_processes"
                                ],
                                "started_ns": run_started,
                                "finished_ns": run_finished,
                                "ticks": len(row["ticks"]),
                                "samples_count": len(row["samples"]),
                            }
                        )
                        quiet = wait_quiet(
                            observe,
                            **{
                                **RESTART_POLICY["quiet"],
                                "max_wait_seconds": min(
                                    900.0, max(0.0, deadline - time.monotonic())
                                ),
                            },
                        )
                        quiet_waits.append(quiet)
                        if not quiet["quiet"]:
                            failure = "host not quiet for M1 restart"
                            break
                    if failure:
                        break
                    if row["summary"]["complete"]:
                        try:
                            audit_unit(row, data)
                        except ValueError as exc:
                            row["summary"].update(verdict="INVALID", complete=False)
                            row["validation_error"] = str(exc)
                    exported = export_json(args.out / f"{unit_id}.json", row)
                    if row["summary"]["complete"]:
                        completed_exports[unit_id] = {"unit_id": unit_id, **exported}
                units.append(row)
                if not row["summary"]["complete"] and not row["summary"].get("decided"):
                    break
        except (Exception, KeyboardInterrupt) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            invalid = isinstance(exc, ValueError)
        receipt = reduce_units(units, data["namespace"], data)
        receipt.update(
            discarded_runs=discarded_runs, quiet_waits=quiet_waits, restart_policy=RESTART_POLICY
        )
        try:
            audit_discarded(args.out, receipt, units)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failure, invalid = str(exc), True
        if failure:
            receipt["verdict"] = "INVALID" if invalid else "INCOMPLETE"
            receipt["reason"] = failure
        if sources() != data["sources"]:
            receipt["verdict"] = "INVALID"
            receipt["reason"] = "source drift during measurement"
        receipt["mac_lease"] = lease.receipt
        receipt["attempt_number"] = 1 if previous is None else previous["attempt_number"] + 1
        receipt["charged_seconds"] = charged + time.monotonic() - started
        receipt["completed_unit_exports"] = completed_exports
        receipt["verified_stop"] = stop_state["verified_stop"]
        if args.reconciled_precell_stop:
            receipt["reconciled_precell_stop_sha256"] = digest(args.reconciled_precell_stop)
        if args.judged_recovery:
            receipt["judged_recovery_sha256"] = digest(args.judged_recovery)
            receipt["final_attempt"] = True
        receipt["mechanical_failure"] = receipt["verdict"] == "INCOMPLETE" and (
            bool(failure) or any(row.get("stop_reason") for row in units)
        )
        if receipt["charged_seconds"] >= CAP_SECONDS and receipt["verdict"] != "INVALID":
            receipt["verdict"] = "INCOMPLETE"
            receipt["reason"] = "cumulative wall cap exhausted"
        export_json(args.out / "receipt.json", receipt)
        print(json.dumps({"verdict": receipt["verdict"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
