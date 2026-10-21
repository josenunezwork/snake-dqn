"""One-look O2 analysis: paired worlds, two Holm contrasts and contract guards."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Any

from scipy.stats import t

from research.t34_round0.analyze import InvalidEvidence, _ci, _stratified, _validate_arm
from research.t34_round0.common import B8_SHA, read_json, sha256, write_once

MIXES = ("scripted", "frozen", "mixed")
ARMS = ("bare", "d", "do")
COUNTS = dict.fromkeys(MIXES, 32)
MAX_DECOMPRESSED_BYTES = 2 * 1024**3
MAX_BUNDLE_MEMBERS = 1000
SCIENTIFIC_FAULT_CLASSES = frozenset(
    {"nonfinite_q", "nonfinite_loss", "nonfinite_grad", "illegal_action", "action_collapse"}
)


class ScientificFaultEvidence(Exception):
    """An admitted, hash-bound unconditional fault; never a transport error."""

    def __init__(self, fault: dict[str, Any]) -> None:
        self.fault = fault
        super().__init__(fault["class"])


def _check_bundle_limits(total: int, members: int) -> None:
    """Allow the registered full provenance volume while bounding decompression."""
    if total > MAX_DECOMPRESSED_BYTES or members > MAX_BUNDLE_MEMBERS:
        raise InvalidEvidence("unbounded decompressed export")


def _validate_scientific_fault(
    fault: dict[str, Any], registration: dict[str, Any], registration_sha: str
) -> None:
    """Admit only the registered study's explicitly typed unconditional faults."""
    expected = {
        "schema": "t34-o2-scientific-fault/v1",
        "registration_sha256": registration_sha,
        "source_commit": registration["source_commit"],
        "checkpoint_sha256": B8_SHA,
        "bank_sha256": registration["bank_sha256"],
    }
    if any(fault.get(k) != v for k, v in expected.items()):
        raise InvalidEvidence("scientific fault source/registration identity differs")
    if fault.get("phase") != "study":
        raise InvalidEvidence("excluded-fixture fault is not a study scientific result")
    if fault.get("world") not in registration["bank"]["rows"]:
        raise InvalidEvidence("scientific fault is outside the registered study worlds")
    if fault.get("arm") not in ARMS or fault.get("class") not in SCIENTIFIC_FAULT_CLASSES:
        raise InvalidEvidence("unknown scientific fault arm or class")
    if type(fault.get("decision")) is not int or not 0 <= fault["decision"] < 5000:
        raise InvalidEvidence("scientific fault decision outside H5000")
    if not isinstance(fault.get("detail"), str) or not fault["detail"].strip():
        raise InvalidEvidence("scientific fault lacks detection detail")


def holm(contrasts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Step-down Holm family alpha .10; fixed name order breaks equal-p ties."""
    result = {}
    for name, summary in contrasts.items():
        mu, se, df = summary["mean"], summary["se"], summary["df"]
        p = float(t.sf(mu / se, df)) if se else (0.0 if mu > 0 else 1.0)
        result[name] = {**summary, "p_one_sided": p}
    ordered = sorted(result, key=lambda k: (result[k]["p_one_sided"], k))
    previous_pass = True
    for j, name in enumerate(ordered):
        entry = result[name]
        alpha = 0.10 / (2 - j)
        lower = entry["mean"] - float(t.isf(alpha, entry["df"])) * entry["se"]
        passed = previous_pass and entry["p_one_sided"] < alpha and lower > 0
        entry.update(holm_rank=j + 1, holm_tail_alpha=alpha, holm_lower=lower, holm_pass=passed)
        previous_pass = passed
    return result


def _validate_controller(arm: dict[str, Any], name: str, *, schedule: Any = None) -> None:
    """Recompute the online schedule, O1 ledger and retained D/O selection."""
    from dataclasses import asdict

    from research.t34_o1_harvest.analyze import qualifying_root, validate_ledger
    from research.t34_o2_harvest.controller import (
        HarvestScore,
        OpportunitySchedule,
        choose_shared_tree,
    )

    schedule = schedule or OpportunitySchedule()
    controller = arm["controller"]
    roots = controller["roots"]
    if name == "bare":
        if roots:
            raise InvalidEvidence("bare arm spent search work")
        return
    expected_budget = {
        "max_roots": len(schedule.time_roots) + schedule.max_event_roots,
        "max_actions": 6,
        "horizon": 500,
        "harvest_added_branches": 0,
    }
    if controller.get("branch_budget") != expected_budget:
        raise InvalidEvidence("D/O shared total branch budget differs")
    expected_rule = "rescue_only" if name == "d" else "rescue_plus_harvest"
    if controller.get("rule_id") != expected_rule:
        raise InvalidEvidence("controller rule differs")
    events = controller["nearby_death_decisions"]
    if (
        not isinstance(events, list)
        or events != sorted(set(events))
        or any(type(i) is not int or not 0 <= i <= arm["decisions"] for i in events)
    ):
        raise InvalidEvidence("nearby-death event chronology invalid")
    expected = []
    used = 0
    last = -64
    for decision in range(arm["decisions"]):
        if schedule.due(decision, used, nearby_death=decision in events, last_event=last):
            expected.append(decision)
            if decision not in schedule.time_roots:
                used += 1
                last = decision
    if [event["decision"] for event in roots] != expected:
        raise InvalidEvidence("online root schedule incomplete or altered")
    rows = 0
    for event in roots:
        root = event["root"]
        qualifying_root(root)  # independently rechecks every provenance step
        branches = root["branches"]
        bare = str(root["bare_action"])
        integrals = {
            action: validate_ledger(branch["ledger"], branch["row_steps"])
            for action, branch in branches.items()
        }
        scores = [
            HarvestScore(
                int(action),
                branch["delta_mass_sum"],
                integrals[action]["death_drop"] - integrals[bare]["death_drop"],
                branch["survives"],
            )
            for action, branch in branches.items()
        ]
        if event["scores"] != [asdict(score) for score in scores]:
            raise InvalidEvidence("harvest provenance scores differ")
        d, do = choose_shared_tree(
            branches, q=root["q"], bare_action=root["bare_action"], scores=scores
        )
        choice = d if name == "d" else do
        if json.loads(json.dumps(choice.record())) != event["choice"]:
            raise InvalidEvidence("delivered controller choice/ranking differs")
        if arm["action_trace"][event["decision"]] != choice.action:
            raise InvalidEvidence("proposed and executed root action differ")
        time_root = event["decision"] in schedule.time_roots
        if event["time_root"] != time_root or event["event_root"] == time_root:
            raise InvalidEvidence("root trigger provenance differs")
        rows += sum(branch["row_steps"] for branch in branches.values())
    if len(roots) > expected_budget["max_roots"] or rows > expected_budget["max_roots"] * 6 * 500:
        raise InvalidEvidence("shared branch budget exceeded")
    if controller.get("search_rows_total") != rows:
        raise InvalidEvidence("search row counter does not reconcile")


def _records(records: list[dict[str, Any]], bank_rows: list[dict[str, Any]]) -> None:
    if len(records) != 96:
        raise EOFError("single look requires all 96 paired worlds")
    if [r.get("world") for r in records] != bank_rows:
        raise InvalidEvidence("wrong, duplicated, missing or reordered bank world")
    if Counter(r["world"]["mix"] for r in records) != COUNTS:
        raise InvalidEvidence("one-look mix counts differ from 32/32/32")
    for row in records:
        if row.get("schema") != "t34-o2-harvest-world/v1" or row.get("horizon") != 5000:
            raise InvalidEvidence("wrong O2 world schema or horizon")
        if (
            row.get("checkpoint_sha256") != B8_SHA
            or row.get("cfg", {}).get("mechanics_version") != 2
        ):
            raise InvalidEvidence("wrong checkpoint or executable mechanics")
        if set(row.get("arms", {})) != set(ARMS):
            raise InvalidEvidence("missing or extra controller arm")
        if (
            row.get("operational", {}).get("device") != "cpu"
            or row["operational"].get("threads") != 1
        ):
            raise InvalidEvidence("CUDA-free CPU single-thread execution required")
        initial = [row["arms"][name].get("initial_digest") for name in ARMS]
        if (
            any(
                not isinstance(d, str)
                or len(d) != 64
                or any(c not in "0123456789abcdef" for c in d)
                for d in initial
            )
            or len(set(initial)) != 1
        ):
            raise InvalidEvidence("paired controller initial states differ or lack identity")
        for name, arm in row["arms"].items():
            _validate_arm(arm)
            trace = arm.get("action_trace")
            if not isinstance(trace, list) or any(
                type(a) is not int or not 0 <= a < 6 for a in trace
            ):
                raise InvalidEvidence("missing/illegal executed action trace")
            if (
                len(trace) != arm["decisions"]
                or [trace.count(a) for a in range(6)] != arm["action_counts"]
            ):
                raise InvalidEvidence("executed action trace/counters disagree")
            digest = arm.get("final_digest")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise InvalidEvidence("missing final trajectory digest")
            _validate_controller(arm, name)
            mass_trace = arm.get("mass_trace")
            if (
                not isinstance(mass_trace, list)
                or len(mass_trace) != arm["decisions"]
                or any(type(v) is not int or v < 0 for v in mass_trace)
            ):
                raise InvalidEvidence("missing or malformed post-step mass trace")
            if (
                sum(mass_trace) / 5000 != arm["mass_integral"]
                or (arm["decisions"] - arm["deaths"]) / 5000 != arm["survival_fraction"]
            ):
                raise InvalidEvidence("full-horizon mass/alive metrics do not reconcile")


def _collapse(records: list[dict[str, Any]], name: str) -> bool:
    window: deque[int] = deque()
    counts = [0] * 6
    for row in records:
        for action in row["arms"][name]["action_trace"]:
            window.append(action)
            counts[action] += 1
            if len(window) > 100000:
                counts[window.popleft()] -= 1
            if len(window) == 100000 and max(counts) > 90000:
                return True
    return False


def decision(records: list[dict[str, Any]], bank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure arithmetic; artifact admission is required before evidence authority."""
    try:
        _records(records, bank_rows)
        groups = {m: [r for r in records if r["world"]["mix"] == m] for m in MIXES}
        contrasts = {
            name: _stratified(
                {
                    m: [
                        r["arms"]["do"]["mass_integral"] - r["arms"][ref]["mass_integral"]
                        for r in groups[m]
                    ]
                    for m in MIXES
                },
                0.10,
            )
            for name, ref in (("do_minus_bare", "bare"), ("do_minus_d", "d"))
        }
        masses = holm(contrasts)
        alive = {
            m: [
                r["arms"]["do"]["survival_fraction"] - r["arms"]["bare"]["survival_fraction"]
                for r in groups[m]
            ]
            for m in MIXES
        }
        pooled = _stratified(alive, 0.05)
        per_mix = {
            m: {
                "alive": _ci(alive[m], 0.05),
                "absolute_alive": sum(r["arms"]["do"]["survival_fraction"] for r in groups[m]) / 32,
                "no_death_gain": sum(
                    r["arms"]["do"]["no_death"] - r["arms"]["bare"]["no_death"] for r in groups[m]
                )
                / 32,
            }
            for m in MIXES
        }
        boosts = sum(r["arms"]["do"]["boost_decisions"] for r in records)
        decisions = sum(r["arms"]["do"]["decisions"] for r in records)
        faults = [
            {"world_id": r["world"]["world_id"], "arm": name, "fault": fault}
            for r in records
            for name in ARMS
            for fault in r["arms"][name]["unconditional_faults"]
        ]
        faults += [
            {
                "arm": name,
                "fault": "action collapse >90% in a 100k executed-decision sliding window",
            }
            for name in ARMS
            if _collapse(records, name)
        ]
        clauses = {
            "incremental_holm_lb_positive": masses["do_minus_d"]["holm_pass"],
            "vs_b8_holm_lb_positive": masses["do_minus_bare"]["holm_pass"],
            "vs_b8_gain_ge40": masses["do_minus_bare"]["mean"] >= 40,
            "guard_a_pooled": pooled["lower"] > -0.05,
            "guard_a_per_mix": all(per_mix[m]["alive"]["lower"] > -0.075 for m in MIXES),
            "guard_a_absolute_floor": all(per_mix[m]["absolute_alive"] >= 0.30 for m in MIXES),
            "guard_b_scripted_no_death_gain": per_mix["scripted"]["no_death_gain"] >= 0.03,
            "delivered_boost_share": 0.005 <= boosts / decisions <= 0.10,
        }
        clear_stop = (
            masses["do_minus_d"]["upper"] <= 0
            or masses["do_minus_bare"]["upper"] <= 0
            or pooled["upper"] <= -0.05
            or any(per_mix[m]["alive"]["upper"] <= -0.075 for m in MIXES)
        )
        point_fail = any(
            not clauses[k]
            for k in (
                "vs_b8_gain_ge40",
                "guard_a_absolute_floor",
                "guard_b_scripted_no_death_gain",
                "delivered_boost_share",
            )
        )
        status = (
            "KILL"
            if faults or clear_stop or point_fail
            else "GO" if all(clauses.values()) else "INCONCLUSIVE"
        )
        return {
            "schema": "t34-o2-decision/v1",
            "status": status,
            "independent_unit": "paired_world",
            "look": 1,
            "evidence_authority": "UNVERIFIED_NUMERICAL_SUMMARY",
            "contrasts": masses,
            "pooled_alive": pooled,
            "per_mix": per_mix,
            "boost_share": boosts / decisions,
            "clauses": clauses,
            "unconditional_faults": faults,
            "release_authorized": False,
            "reason": (
                "all development bars pass; Mac paced test remains required"
                if status == "GO"
                else (
                    "this controller failed its registered bar"
                    if status == "KILL"
                    else "confidence bounds inconclusive; no extension"
                )
            ),
        }
    except EOFError as exc:
        return {"status": "INCOMPLETE", "reason": str(exc), "release_authorized": False}
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"status": "INVALID", "reason": str(exc), "release_authorized": False}


def _bound_file(root: Path, entry: dict[str, Any]) -> Path:
    name = Path(entry["path"])
    if name.is_absolute() or ".." in name.parts:
        raise InvalidEvidence("artifact path escapes export")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise InvalidEvidence("missing or escaping artifact")
    if sha256(path) != entry["sha256"]:
        raise InvalidEvidence("artifact SHA mismatch")
    return path


def _payload(payloads: dict[str, Path], entry: dict[str, Any]) -> dict[str, Any]:
    """Read only a digest-verified payload with its immutable inner sidecars."""
    name = entry["path"]
    path = payloads[name]
    if sha256(path) != entry["sha256"]:
        raise InvalidEvidence("payload SHA mismatch")
    checksum = payloads[name + ".sha256"].read_text().split()
    if (
        not checksum
        or checksum[0] != entry["sha256"]
        or payloads[name + ".complete"].read_text().strip() != "complete"
    ):
        raise InvalidEvidence("payload sidecars incomplete or mismatched")
    return read_json(path)


def _validate_admissions(
    run: dict[str, Any], payloads: dict[str, Path], registration: dict[str, Any]
) -> None:
    """Recompute fixture integrity and the prospective timing/cap admission."""
    import math

    from research.t34_o1_harvest.analyze import validate_ledger
    from research.t34_o2_harvest.controller import OpportunitySchedule

    parity = [_payload(payloads, e) for e in run["parity_worlds"]]
    if [w.get("row") for w in parity] != registration["bank"]["excluded_parity_rows"]:
        raise InvalidEvidence("wrong or incomplete three-world parity fixture")
    for world in parity:
        steps = world["base_steps"]
        evidence = world["parity"]
        if (
            type(steps) is not int
            or not 1 <= steps <= 500
            or world.get("schema") != "t34-o1-world/v1"
            or world.get("frames") != 500
            or world.get("horizon") != 500
            or world.get("checkpoint_sha256") != B8_SHA
            or world.get("cfg", {}).get("mechanics_version") != 2
            or type(world.get("base_survives")) is not bool
            or world.get("complete") is not True
            or world.get("accounting") is not True
            or evidence.get("pass") is not True
            or evidence.get("frames") != steps
            or not isinstance(evidence.get("chain"), str)
            or len(evidence["chain"]) != 64
            or any(c not in "0123456789abcdef" for c in evidence["chain"])
            or world.get("roots") != []
            or (world.get("base_survives") is True and steps != 500)
            or (world.get("base_survives") is False and world.get("base_death_cause", 0) == 0)
        ):
            raise InvalidEvidence("parity content/counters or executable profile differs")
        validate_ledger(world["base_ledger"], steps)
    timing = [_payload(payloads, e) for e in run["timing_worlds"]]
    if [w.get("world") for w in timing] != registration["bank"]["excluded_timing_rows"]:
        raise InvalidEvidence("wrong or incomplete three-world timing fixture")
    base_rates = []
    branch_rates = []
    schedule = OpportunitySchedule(time_roots=(10, 20), max_event_roots=0)
    for world in timing:
        if (
            world.get("schema") != "t34-o2-harvest-world/v1"
            or world.get("horizon") != 5000
            or world.get("checkpoint_sha256") != B8_SHA
            or world.get("cfg", {}).get("mechanics_version") != 2
            or set(world.get("arms", {})) != set(ARMS)
        ):
            raise InvalidEvidence("timing executable profile differs")
        initials = [world["arms"][name].get("initial_digest") for name in ARMS]
        if (
            len(set(initials)) != 1
            or not isinstance(initials[0], str)
            or len(initials[0]) != 64
            or world.get("operational", {}).get("device") != "cpu"
            or world.get("operational", {}).get("threads") != 1
        ):
            raise InvalidEvidence("timing paired source or CPU identity differs")
        for name, arm in world["arms"].items():
            _validate_arm(arm)
            _validate_controller(arm, name, schedule=schedule)
            if arm.get("executed_cfg", {}).get("mechanics_version") != 2:
                raise InvalidEvidence("timing executable config differs")
            trace = arm["action_trace"]
            mass = arm["mass_trace"]
            if (
                len(trace) != arm["decisions"]
                or [trace.count(a) for a in range(6)] != arm["action_counts"]
                or len(mass) != arm["decisions"]
                or sum(mass) / 5000 != arm["mass_integral"]
                or (arm["decisions"] - arm["deaths"]) / 5000 != arm["survival_fraction"]
            ):
                raise InvalidEvidence("timing counters do not reconcile")
            search = arm["search_elapsed_seconds"]
            elapsed = arm["elapsed_seconds"]
            base_steps = arm["base_row_steps"]
            branch_steps = arm["branch_row_steps"]
            if (
                any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    or x < 0
                    for x in (search, elapsed, base_steps, branch_steps)
                )
                or elapsed < search
                or base_steps != arm["decisions"]
            ):
                raise InvalidEvidence("invalid timing counters")
            base_rates.append((elapsed - search) / max(1, base_steps))
            actual_branch = sum(
                b["row_steps"]
                for r in arm["controller"]["roots"]
                for b in r["root"]["branches"].values()
            )
            if actual_branch != branch_steps:
                raise InvalidEvidence("timing search row counter differs")
            if branch_steps:
                branch_rates.append(search / branch_steps)
    if not branch_rates:
        raise InvalidEvidence("timing fixture has no complete branch work")
    admission = run["timing_admission"]
    workers = admission["workers"]
    count = admission["study_worlds"]
    if (
        type(workers) is not int
        or not 1 <= workers <= registration["caps"]["workers_per_job"]
        or type(count) is not int
        or count != 32
        or admission.get("passed") is not True
        or admission.get("formula") != "max-rates-2x-plus600/v1"
    ):
        raise InvalidEvidence("timing admission design differs")
    per_world = 5000 * 3 * max(base_rates) + 8 * 6 * 500 * 2 * max(branch_rates)
    forecast = (
        math.ceil(count / workers) * per_world * 2 + max(w["elapsed_seconds"] for w in timing) + 600
    )
    cap = admission["cap_seconds"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in (
            cap,
            admission["forecast_seconds"],
            admission["pre_admission_elapsed_seconds"],
        )
    ):
        raise InvalidEvidence("non-finite timing admission")
    if (
        admission.get("base_seconds_per_step") != max(base_rates)
        or admission.get("branch_seconds_per_step") != max(branch_rates)
        or admission.get("ordinary_deadline_seconds")
        != registration["caps"]["ordinary_job_seconds"]
    ):
        raise InvalidEvidence("timing rate/deadline identity differs")
    if (
        not math.isclose(admission["forecast_seconds"], forecast, rel_tol=1e-12, abs_tol=1e-9)
        or cap > registration["caps"]["ordinary_job_seconds"]
        or not 0 < forecast <= cap
        or admission["pre_admission_elapsed_seconds"] + forecast
        >= registration["caps"]["ordinary_job_seconds"]
    ):
        raise InvalidEvidence("timing forecast/cap admission differs")


def _partial_identity(row: dict[str, Any], registration: dict[str, Any]) -> None:
    """Check a completed row's identity even when a fault precludes full-N metrics."""
    if (
        row.get("world") not in registration["bank"]["rows"]
        or row.get("schema") != "t34-o2-harvest-world/v1"
        or row.get("horizon") != 5000
        or row.get("checkpoint_sha256") != B8_SHA
        or row.get("cfg", {}).get("mechanics_version") != 2
    ):
        raise InvalidEvidence("completed world/source identity differs")


def _admit_job(
    job: Path, registration: dict[str, Any], registration_sha: str
) -> list[dict[str, Any]]:
    """Verify the supervisor publication, stopped runtime and immutable bundle."""
    receipt, manifest, terminal = (
        read_json(job / name) for name in ("receipt.json", "manifest.json", "terminal.json")
    )
    if receipt.get("error") or receipt.get("cleanup_error") or receipt.get("terminal") != terminal:
        raise InvalidEvidence("supervisor terminal identity or cleanup differs")
    if terminal.get("generation") != receipt.get("generation"):
        raise InvalidEvidence("foreign runtime generation")
    if (
        receipt.get("manifest_sha256") != sha256(job / "manifest.json")
        or receipt.get("commit") != registration["source_commit"]
        or manifest.get("commit") != registration["source_commit"]
    ):
        raise InvalidEvidence("transport manifest/source mismatch")
    if not any(
        i.get("dest") == "registration.json" and i.get("sha256") == registration_sha
        for i in manifest.get("inputs", [])
    ):
        raise InvalidEvidence("transport did not upload governing registration")
    if receipt.get("status") != "PASS" or receipt.get("stop_verified") is not True:
        raise EOFError("supervisor terminal/cleanup incomplete")
    if terminal.get("returncode") != 0 or terminal.get("status") not in {
        "completed",
        "success",
        "succeeded",
        "PASS",
    }:
        raise EOFError("runtime did not terminate successfully")
    budget = manifest.get("budget", {})
    expected_caps = registration["caps"]
    if (
        budget.get("unit_cap") != expected_caps["units_per_job"]
        or budget.get("rate_units_per_hour") != expected_caps["l4_rate_units_per_hour"]
        or budget.get("reserve_units") != expected_caps["reserve_units_per_job"]
    ):
        raise InvalidEvidence("job budget differs from registered tranche")
    if (
        manifest.get("timeout_minutes", float("inf")) * 60
        > expected_caps["units_per_job"] / expected_caps["l4_rate_units_per_hour"] * 3600
    ):
        raise InvalidEvidence("job time cap exceeds registered unit envelope")
    if (
        manifest.get("requirements", {}).get("gpu") != "L4"
        or manifest.get("env", {}).get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise InvalidEvidence("job execution placement differs")
    exports = receipt.get("exports", [])
    if len(exports) != 1 or exports != terminal.get("exports"):
        raise InvalidEvidence("expected one terminal-bound immutable export")
    entry = exports[0]
    archive_path = _bound_file(job / "exports", entry)
    if archive_path.stat().st_size != entry["size"] or entry["size"] > 67108864:
        raise InvalidEvidence("export size differs/exceeds cap")
    sidecar = Path(str(archive_path) + ".sha256")
    if (
        not Path(str(archive_path) + ".complete").is_file()
        or sidecar.read_text().split()[0] != entry["sha256"]
    ):
        raise InvalidEvidence("export sidecars incomplete or mismatched")
    # Stream raw decompressed bytes to bounded temporary files: a legal shard
    # can exceed 256 MiB of provenance without retaining every raw JSON blob in RAM.
    with tempfile.TemporaryDirectory(prefix="o2-analysis-") as staging_name:
        staging = Path(staging_name)
        payloads: dict[str, Path] = {}
        total = 0
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                name = Path(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or member.name in payloads
                    or name.as_posix() != member.name
                    or name == Path(".")
                ):
                    raise InvalidEvidence("unsafe or duplicate export member")
                total += member.size
                _check_bundle_limits(total, len(payloads) + 1)
                stream = archive.extractfile(member)
                if stream is None:
                    raise InvalidEvidence("missing export member body")
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as output:
                    shutil.copyfileobj(stream, output, length=1024**2)
                if path.stat().st_size != member.size:
                    raise InvalidEvidence("truncated export member")
                payloads[member.name] = path
        run = read_json(payloads["receipt.json"])
        # Identity checks precede status so a foreign fault never becomes KILL.
        if run.get("mode") != "campaign" or run.get("registration_sha256") != registration_sha:
            raise InvalidEvidence("fixture or mismatched registration receipt")
        if (
            run.get("checkpoint_sha256") != B8_SHA
            or run.get("source_commit") != registration["source_commit"]
        ):
            raise InvalidEvidence("checkpoint/source identity mismatch")
        if (
            run.get("bank_sha256") != registration["bank_sha256"]
            or run.get("parity_pass") is not True
        ):
            raise InvalidEvidence("bank/parity admission mismatch")
        _validate_admissions(run, payloads, registration)
        records = [_payload(payloads, entry) for entry in run["worlds"]]
        if run.get("status") == "SCIENTIFIC_FAULT":
            entry = run["scientific_fault"]
            fault = _payload(payloads, entry)
            _validate_scientific_fault(fault, registration, registration_sha)
            # Even a fault receipt cannot hide a foreign completed world.
            for row in records:
                _partial_identity(row, registration)
            raise ScientificFaultEvidence(fault)
        if run.get("status") != "COMPLETE":
            raise EOFError("incomplete runtime receipt")
        return records


def analyze(registration_path: Path, run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Claim the sole look; run_dirs are original supervisor job directories."""
    from research.t34_o2_harvest.bank import PACKAGE
    from research.t34_o2_harvest.registration import require_registration

    if registration_path.resolve() != (PACKAGE / "REGISTRATION_O2.json").resolve():
        raise InvalidEvidence("analysis requires the canonical frozen package registration")
    registration = require_registration(registration_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = registration_path.with_name("LOOK_O2.json")
    write_once(
        marker,
        {"schema": "t34-o2-look/v1", "look": 1, "registration_sha256": sha256(registration_path)},
    )
    try:
        if not 1 <= len(run_dirs) <= registration["caps"]["jobs"] or len(
            {p.resolve() for p in run_dirs}
        ) != len(run_dirs):
            raise InvalidEvidence("expected one to three distinct supervisor job publications")
        records = []
        faults = []
        incomplete = []
        seen = set()
        for directory in run_dirs:
            try:
                rows = _admit_job(directory, registration, sha256(registration_path))
            except ScientificFaultEvidence as exc:
                faults.append(exc.fault)
                continue
            except (FileNotFoundError, EOFError) as exc:
                incomplete.append(str(exc))
                continue
            for row in rows:
                world_id = row["world"]["world_id"]
                if world_id in seen:
                    raise InvalidEvidence("duplicate world across runtime shards")
                _partial_identity(row, registration)
                seen.add(world_id)
                records.append(row)
        # Every supplied publication has been inspected for invalid identities.
        # A genuine unconditional fault stops the controller before full N;
        # missing remaining worlds are expected and no metrics are fabricated.
        if faults:
            result = {
                "status": "KILL",
                "reason": "registered unconditional scientific fault",
                "evidence_authority": "TIER_1_DEVELOPMENT",
                "unconditional_faults": faults,
                "missing_publications": incomplete,
                "completed_worlds": len(records),
                "early_stop": "unconditional_fault",
                "release_authorized": False,
            }
            result.update(
                schema="t34-o2-decision/v1", look=1, registration_sha256=sha256(registration_path)
            )
            write_once(output_dir / "decision.json", result)
            return result
        if incomplete or len(run_dirs) != registration["caps"]["jobs"]:
            raise EOFError("incomplete supervisor publications: " + "; ".join(incomplete))
        index = {r["world"]["world_id"]: r for r in records}
        expected = registration["bank"]["rows"]
        if not set(index).issubset({r["world_id"] for r in expected}):
            raise InvalidEvidence("unexpected world outside frozen bank")
        if len(index) != 96:
            raise EOFError("incomplete 96-world set")
        result = decision([index[r["world_id"]] for r in expected], expected)
        if result.get("evidence_authority") == "UNVERIFIED_NUMERICAL_SUMMARY":
            result["evidence_authority"] = "TIER_1_DEVELOPMENT"
    except (FileNotFoundError, EOFError) as exc:
        result = {"status": "INCOMPLETE", "reason": str(exc)}
    except (KeyError, ValueError, TypeError, tarfile.TarError, IndexError) as exc:
        result = {"status": "INVALID", "reason": str(exc)}
    result.update(
        schema="t34-o2-decision/v1",
        look=1,
        registration_sha256=sha256(registration_path),
        release_authorized=False,
    )
    write_once(output_dir / "decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.registration, args.run_dir, args.out_dir)["status"])


if __name__ == "__main__":
    main()
