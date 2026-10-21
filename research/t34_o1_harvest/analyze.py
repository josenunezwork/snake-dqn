"""One-look O1 analyzer; validate identities/accounting before any KILL/GO."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from research.t34_o1_harvest.registration import (
    B8_SHA,
    read_json,
    require_registration,
    sha256,
    write_once,
)

SOURCES = ("ambient", "death_drop", "boost_pellet")


def validate_ledger(ledger: dict[str, Any], rows: int) -> dict[str, int]:
    """Recompute source integrals and stored mass independently of producer."""
    envs = ledger["per_env"]
    events = ledger["events"]
    if len(envs) != 1 or len(events) != rows:
        raise ValueError("incomplete branch provenance steps")
    env = envs[0]
    cumulative = dict.fromkeys(SOURCES, 0)
    weighted = dict.fromkeys(SOURCES, 0)
    burn, weighted_burn = 0, 0
    for event in events:
        if event["env"] != 0:
            raise ValueError("wrong provenance environment")
        for source in SOURCES:
            count = event["intake"][source]
            if type(count) is not int or count < 0:
                raise ValueError("invalid source intake")
            cumulative[source] += count
            if cumulative[source] != event["cumulative_intake"][source]:
                raise ValueError("source cumulative counter mismatch")
            weighted[source] += cumulative[source]
        if type(event["boost_cost"]) is not int or event["boost_cost"] < 0:
            raise ValueError("invalid boost cost")
        burn += event["boost_cost"]
        if burn != event["cumulative_boost_cost"]:
            raise ValueError("boost cumulative counter mismatch")
        weighted_burn += burn
    if cumulative != env["sources"] or cumulative != env["cumulative_intake"]:
        raise ValueError("source totals mismatch")
    if (
        weighted != env["weighted_cumulative_intake"]
        or weighted_burn != env["weighted_cumulative_boost_cost"]
    ):
        raise ValueError("weighted attribution mismatch")
    if (
        burn != env["boost_cost"]
        or env["initial_stored_length"] + sum(cumulative.values()) - burn
        != env["final_stored_length"]
    ):
        raise ValueError("stored length mass balance failed")
    return {
        **weighted,
        "boost_cost": weighted_burn,
        "initial_stored_length": env["initial_stored_length"],
    }


def qualifying_root(root: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Validate every branch and recompute the surviving death-food mechanism."""
    if root["horizon"] != 500 or root["source_preserved"] is not True:
        raise ValueError("wrong horizon or mutable source")
    branches = root["branches"]
    q = root["q"]
    if len(q) != 6 or any(not math.isfinite(x) for x in q):
        raise ValueError("missing/nonfinite Q census")
    bare = str(root["bare_action"])
    mask = root["advisory_legal"]
    if (
        len(mask) != 6
        or set(branches) != {str(a) for a in range(6) if mask[a]}
        or bare not in branches
    ):
        raise ValueError("missing or illegal first-action branch")
    selected = min((a for a in range(6) if mask[a]), key=lambda a: (-q[a], a))
    if selected != int(bare):
        raise ValueError("bare action differs from B8 masked greedy Q")
    if root["illegal_actions"] != [a for a in range(6) if not mask[a]]:
        raise ValueError("illegal-action census mismatch")
    integrals = {}
    for action, branch in branches.items():
        rows, alive = branch["row_steps"], branch["alive_steps"]
        survives = branch["survives"]
        if (
            type(rows) is not int
            or not 1 <= rows <= 500
            or alive != rows - (not survives)
            or survives != (alive == 500)
        ):
            raise ValueError("branch survival/row count mismatch")
        if (
            int(action) != branch["action"]
            or type(branch["mass_sum"]) is not int
            or branch["mass_sum"] < 0
        ):
            raise ValueError("branch identity/mass mismatch")
        components = validate_ledger(branch["ledger"], rows)
        if survives:
            expected = (
                500 * components["initial_stored_length"]
                + sum(components[s] for s in SOURCES)
                - components["boost_cost"]
            )
            if branch["mass_sum"] != expected:
                raise ValueError("mass integral provenance decomposition failed")
        integrals[action] = components
    ranked = sorted(
        (int(a) for a in branches), key=lambda a: (-branches[str(a)]["mass_sum"], str(a) != bare, a)
    )
    if ranked != root["ranked_actions"]:
        raise ValueError("rank/tie rule mismatch")
    details = []
    for action, branch in branches.items():
        delta = branch["mass_sum"] - branches[bare]["mass_sum"]
        death_delta = integrals[action]["death_drop"] - integrals[bare]["death_drop"]
        if branch["delta_mass_sum"] != delta or branch["delta_A500"] != delta / 500:
            raise ValueError("branch delta mismatch")
        qualifies = (
            action != bare
            and branch["survives"]
            and branches[bare]["survives"]
            and death_delta >= delta >= 2500
        )
        details.append(
            {
                "action": int(action),
                "delta_A500": delta / 500,
                "death_delta_A500": death_delta / 500,
                "qualifies": bool(qualifies),
            }
        )
    return any(item["qualifies"] for item in details), details


def decision(worlds: list[dict[str, Any]], bank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count worlds once, respecting INVALID/INCOMPLETE before scientific rules."""
    if len(worlds) != 48:
        return {"status": "INCOMPLETE", "reason": "missing worlds"}
    if [w.get("row") for w in worlds] != bank_rows:
        return {"status": "INVALID", "reason": "wrong/reordered world identities"}
    qualified, per_world = Counter(), []
    try:
        for world in worlds:
            steps = world["base_steps"]
            if type(steps) is not int or not 1 <= steps <= 5000:
                return {"status": "INCOMPLETE", "reason": "missing/truncated base trajectory"}
            if world["base_survives"] is not False and steps < 5000:
                return {"status": "INCOMPLETE", "reason": "base stopped before terminal/horizon"}
            if world["base_survives"] is False and world["base_death_cause"] == 0:
                raise ValueError("terminal base lacks a hero death event")
            if world["checkpoint_sha256"] != B8_SHA:
                raise ValueError("wrong base checkpoint identity")
            validate_ledger(world["base_ledger"], steps)
            if (
                world["complete"] is not True
                or world["accounting"] is not True
                or world["cfg"]["mechanics_version"] != 2
                or world["frames"] != 5000
                or world["horizon"] != 500
                or len(world["roots"]) > 8
            ):
                raise ValueError("world incomplete or wrong profile/design")
            roots = world["roots"]
            from research.t34_o1_harvest.runner import reservoir_priority

            last_event, previous_event, eligible = -64, -1, []
            for event in world["events"]:
                frame = event["decision"]
                if not previous_event < frame <= world["base_steps"] or not event["slots"]:
                    raise ValueError("malformed opponent-death event")
                previous_event = frame
                admitted = frame - last_event >= 64
                if event["eligible"] != admitted:
                    raise ValueError("event dedupe mismatch")
                if admitted:
                    eligible.append(frame)
                    last_event = frame
            chosen = set(
                sorted(eligible, key=lambda d: reservoir_priority(world["row"]["world_seed"], d))[
                    :4
                ]
            )
            for event in world["events"]:
                probability = min(1.0, 4 / len(eligible)) if event["eligible"] and eligible else 0.0
                if (
                    event["selected"] != (event["decision"] in chosen)
                    or event["inclusion_probability"] != probability
                ):
                    raise ValueError("event reservoir selection mismatch")
            expected_roots = chosen | {
                t for t in (1000, 2000, 3000, 4000) if t < world["base_steps"]
            }
            if {r["decision"] for r in roots} != expected_roots:
                raise ValueError("root schedule differs from frozen selection")
            decisions = [r["decision"] for r in roots]
            if len(set(decisions)) != len(decisions):
                raise ValueError("duplicate roots")
            for frame in (1000, 2000, 3000, 4000):
                if frame < world["base_steps"] and frame not in decisions:
                    raise ValueError("missing live time root")
            if sum(r["event_root"] for r in roots) > 4:
                raise ValueError("event root cap exceeded")
            details, qualifies = [], False
            for root in roots:
                found, branches = qualifying_root(root)
                qualifies |= found
                details.append({"decision": root["decision"], "branches": branches})
            if qualifies:
                qualified[world["row"]["mix"]] += 1
            per_world.append(
                {"world_id": world["row"]["world_id"], "qualifies": qualifies, "roots": details}
            )
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "INVALID", "reason": str(exc)}
    fm = qualified["frozen"] + qualified["mixed"]
    return {
        "status": "GO" if sum(qualified.values()) >= 8 and fm >= 4 else "KILL",
        "qualifying_worlds": sum(qualified.values()),
        "frozen_mixed": fm,
        "per_mix": dict(qualified),
        "worlds": per_world,
        "scope": (
            "single-action H500 harvesting mechanism only; "
            "GO to separately registered O2; no release"
        ),
    }


def analyze(registration_path: Path, run_dir: Path) -> dict[str, Any]:
    """Claim the sole look, verify complete immutable inputs, then emit a verdict."""
    registration = require_registration(registration_path)
    write_once(
        run_dir / "LOOK_O1.json",
        {"schema": "t34-o1-look/v1", "look": 1, "registration_sha256": sha256(registration_path)},
    )
    try:
        receipt = read_json(run_dir / "receipt.json")
        if receipt["mode"] != "campaign":
            raise ValueError("excluded fixtures cannot be analyzed as study data")
        if receipt["status"] != "COMPLETE":
            result = {"status": "INCOMPLETE", "reason": "run incomplete"}
        elif (
            receipt["registration_sha256"] != sha256(registration_path)
            or receipt["parity_pass"] is not True
            or receipt["timing_admitted"] is not True
        ):
            result = {"status": "INVALID", "reason": "run identity/parity/admission"}
        else:
            for name in ("parity", "timing"):
                if sha256(run_dir / f"{name}.json") != receipt[f"{name}_sha256"]:
                    raise ValueError(f"{name} receipt identity differs")
            import json

            parity = json.loads((run_dir / "parity.json").read_text())
            if (
                not isinstance(parity, list)
                or len(parity) != 3
                or [w["row"] for w in parity] != registration["bank"]["excluded_parity_rows"]
                or any(
                    w["parity"]["pass"] is not True
                    or w["parity"]["frames"] != w["base_steps"]
                    or w["frames"] != 5000
                    or w["cfg"]["mechanics_version"] != 2
                    or not isinstance(w["parity"]["chain"], str)
                    or len(w["parity"]["chain"]) != 64
                    or w["complete"] is not True
                    or w["accounting"] is not True
                    for w in parity
                )
            ):
                raise ValueError("excluded three-world parity content differs")
            timing = read_json(run_dir / "timing.json")
            from research.t34_o1_harvest.timing import validate_timing

            validate_timing(timing["worlds"], registration["bank"]["excluded_timing_rows"])
            if (
                not 0 < timing["timing_seconds"] <= 1800
                or receipt["elapsed_seconds"] > registration["caps"]["runner_seconds"]
            ):
                raise ValueError("timing fixture/cap content differs")
            from research.t34_o1_harvest.runner import fixture_forecast

            expected = fixture_forecast(timing["worlds"], registration["caps"]["workers"])
            if (
                timing["forecast"] != expected
                or receipt["forecast"] != expected
                or expected["remaining_work_seconds"] + timing["timing_seconds"]
                > registration["caps"]["runner_seconds"]
            ):
                raise ValueError("timing admission forecast differs/exceeds cap")
            worlds = []
            for i, entry in enumerate(receipt["worlds"]):
                if (
                    entry["path"] != f"world-{i:03d}.json"
                    or sha256(run_dir / entry["path"]) != entry["sha256"]
                ):
                    raise ValueError("payload identity mismatch")
                worlds.append(read_json(run_dir / entry["path"]))
            result = decision(worlds, registration["bank"]["rows"])
    except FileNotFoundError as exc:
        result = {"status": "INCOMPLETE", "reason": str(exc)}
    except (ValueError, TypeError, KeyError) as exc:
        result = {"status": "INVALID", "reason": str(exc)}
    result.update(
        schema="t34-o1-decision/v1", look=1, registration_sha256=sha256(registration_path)
    )
    write_once(run_dir / "decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.registration, args.run_dir)
    print(result["status"])


if __name__ == "__main__":
    main()
