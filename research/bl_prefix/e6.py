"""E6 counterfactual runner.  It never dispatches or issues a verdict for preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix import seeds
from research.bl_prefix.analysis import IncompleteResult, e6_summary
from research.bl_prefix.controllers import B8Controller, E2Controller
from research.bl_prefix.engine import HORIZON, load_root, mass_ledger, relocate_root, rollout


def _roots(
    bank: Mapping[str, Any], base: Path, relocations: Mapping[str, str], allowed_seeds: set[int]
) -> list[dict[str, Any]]:
    rows = bank.get("roots")
    if not isinstance(rows, list) or not rows:
        raise IncompleteResult("E6 bank has no roots")
    required = 3 if bank.get("preflight") else 256
    if len(rows) != required:
        raise IncompleteResult(f"E6 requires exactly {required} roots")
    if not bank.get("preflight"):
        for stratum, count in (("early", 128), ("ordinary", 128)):
            if sum(row.get("stratum") == stratum for row in rows) != count:
                raise IncompleteResult("E6 bank stratum count mismatch")
        for stratum in ("early", "ordinary"):
            quotas = {"scripted": 44, "frozen": 42, "mixed": 42}
            if {
                mix: sum(row.get("stratum") == stratum and row.get("mix") == mix for row in rows)
                for mix in quotas
            } != quotas:
                raise IncompleteResult("E6 mix quota mismatch")
        if (
            len({str(row.get("source_world")) for row in rows if row.get("stratum") == "early"})
            != 128
        ):
            raise IncompleteResult("E6 early source worlds are not distinct")
        ordinary = [row.get("ordinary_world") for row in rows if row.get("stratum") == "ordinary"]
        if len(ordinary) != 128 or None in ordinary or len(set(map(str, ordinary))) != 128:
            raise IncompleteResult("E6 ordinary matching worlds are not distinct")
    for row in rows:
        if row.get("stratum") not in {"early", "ordinary"} or not isinstance(row.get("root"), str):
            raise IncompleteResult("malformed E6 root")
        path = base / row["root"]
        verify_export(path)
        root = relocate_root(load_root(path), relocations)
        if (
            root.seed not in allowed_seeds
            or str(root.seed) != str(row.get("source_world"))
            or root.mix != row["mix"]
            or ("frame" in row and root.frame != row["frame"])
        ):
            raise IncompleteResult("root identity metadata differs from bank")
        if row["stratum"] == "ordinary" and str(row.get("ordinary_world")) != str(root.seed):
            raise IncompleteResult("ordinary world label does not identify the loaded root")
        row["root_path"] = str(path)
        row["loaded_root"] = root
    actual = [row["loaded_root"].seed for row in rows]
    if len(actual) != len(set(actual)):
        raise IncompleteResult("duplicated actual source world in root bank")
    return rows


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _critic_ok(plan: Mapping[str, Any], checkpoints: Mapping[str, Any]) -> bool:
    if plan.get("preflight"):
        return True
    receipts = plan.get("critic_receipts")
    if not isinstance(receipts, Mapping) or set(receipts) != {"R", "F"}:
        return False
    for arm, path in receipts.items():
        if not isinstance(path, str):
            return False
        receipt_path = Path(path)
        try:
            verify_export(receipt_path)
        except (OSError, ValueError):
            return False
        item = json.loads(receipt_path.read_text())
        if (
            item.get("complete") is not True
            or item.get("training_complete") is not True
            or item.get("checkpoint_role") != "final"
            or item.get("arm") != arm
        ):
            return False
        if (
            item.get("checkpoint_sha256") != _digest(checkpoints[arm])
            or int(item.get("refit_transitions", 0)) < 2_000_000
        ):
            return False
        try:
            ev = float(item.get("late_body_ev"))
            mse = float(item.get("late_body_mse"))
            baseline = float(item.get("baseline_mse"))
        except (TypeError, ValueError):
            return False
        if (
            not all(__import__("math").isfinite(x) for x in (ev, mse, baseline))
            or ev < 0.1
            or not mse < baseline
        ):
            return False
    return True


def _root_rows(
    task: tuple[str, dict[str, Any], dict[str, str], tuple[int, int], bool, int],
) -> list[dict[str, Any]]:
    """Picklable, deterministic one-root worker; executor.map preserves bank order."""
    root_path, item, checkpoints, future_values, preflight, max_steps = task
    root = load_root(root_path)
    root = relocate_root(root, item["relocations"])
    b8 = B8Controller(checkpoints["B8"])
    controllers: dict[str, Any] = {
        "B8": b8,
        "R": E2Controller(checkpoints["R"], expected_arm="R"),
        "F": E2Controller(checkpoints["F"], expected_arm="F"),
    }
    rows = []
    for stream, seed in enumerate(future_values):
        for phase, prefix_steps in (("primary", 64), ("secondary", HORIZON - root.frame)):
            results = {}
            for name, controller in controllers.items():
                results[name] = rollout(
                    root,
                    controller,
                    b8,
                    prefix_steps=prefix_steps,
                    future_seed=seed,
                    **({"max_steps": max_steps, "preflight": True} if preflight else {}),
                )
            baseline = results["B8"]
            for name, result in results.items():
                row = {
                    "stratum": item["stratum"],
                    "mix": root.mix,
                    "source_world": item["source_world"],
                    "stream": stream,
                    "controller": name,
                    "phase": phase,
                    "score": result.mass_sum / HORIZON,
                    "terminal_survival": result.terminal_survival,
                    "steps": result.steps,
                    "truncated": bool(getattr(result, "truncated", False)),
                }
                if name != "B8":
                    row["ledger"] = mass_ledger(result, baseline, prefix_steps=64)
                rows.append(row)
    return rows


def run(
    plan: Mapping[str, Any], out: Path, *, preflight: bool = False, compute_analysis: bool = False
) -> dict[str, Any]:
    """Run only a fully-bound E6 plan; any missing scientific guard is INCOMPLETE."""
    begun = time.perf_counter()
    preflight = bool(preflight or plan.get("preflight"))
    from research.bl_prefix.resource_limits import usable_seconds

    cap_seconds = usable_seconds("e6")
    try:
        if plan.get("experiment") != "e6" or bool(plan.get("preflight")) != preflight:
            raise IncompleteResult("E6 plan/preflight identity mismatch")
        workers = plan.get("workers", 2 if preflight else 12)
        if type(workers) is not int or not 1 <= workers <= (2 if preflight else 12):
            raise IncompleteResult("E6 worker cap differs from frozen protocol")
        if not preflight and workers != 12:
            raise IncompleteResult("scientific runs require exactly twelve workers")
        checkpoints = plan.get("controllers", {})
        if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"R", "F", "B8"}:
            raise IncompleteResult("E6 requires exact R/F/B8 controller identities")
        if not _critic_ok(plan, checkpoints):
            raise IncompleteResult("critic preparation receipt is missing or failed")
        audit_path = Path(str(plan["seed_audit"]))
        verify_export(audit_path)
        audit = json.loads(audit_path.read_text())
        seeds.verify_receipt(audit, round_index=int(audit.get("round_index", -1)))
        bank_path = Path(str(plan["bank"]))
        verify_export(bank_path)
        bank = json.loads(bank_path.read_text())
        if bool(bank.get("preflight")) != preflight or bank.get("audit_sha256") != _digest(
            audit_path
        ):
            raise IncompleteResult("bank preflight/audit binding mismatch")
        relocations = plan.get("checkpoint_relocations", {})
        if not isinstance(relocations, Mapping):
            raise IncompleteResult("missing verified checkpoint relocations")
        roots = _roots(
            bank,
            bank_path.parent,
            relocations,
            set(audit["allocation"]["preflight" if preflight else "source"]["seeds"]),
        )
        if bank.get("b8", {}).get("sha256") != _digest(checkpoints["B8"]):
            raise IncompleteResult("bank B8 identity differs from comparator")
        future_values = list(plan.get("future_seeds", []))
        allowed = set(audit["allocation"]["preflight" if preflight else "future"]["seeds"])
        if (
            len(future_values) != 2
            or len(set(future_values)) != 2
            or not set(future_values) <= allowed
        ):
            raise IncompleteResult("two future RNG seeds are required")
        rows: list[dict[str, Any]] = []
        tasks = [
            (
                item["root_path"],
                {
                    "stratum": item["stratum"],
                    "source_world": str(item["source_world"]),
                    "relocations": dict(relocations),
                },
                dict(checkpoints),
                tuple(map(int, future_values)),
                preflight,
                int(plan.get("preflight_max_steps", 16)),
            )
            for item in roots
        ]
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for index, value in enumerate(executor.map(_root_rows, tasks)):
                if time.perf_counter() - begun > cap_seconds:
                    raise IncompleteResult("E6 allocation cap expired")
                rows.extend(value)
                if index and index % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "progress": [index, len(tasks)],
                                "elapsed_seconds": time.perf_counter() - begun,
                            }
                        ),
                        flush=True,
                    )
        analyses = (
            {"decision": "NOT_SCORED_PREFLIGHT", "reason": "tiny operational fixture"}
            if preflight
            else (
                {"status": "UNOPENED_PENDING_PUBLISHED_LEDGER"}
                if not compute_analysis
                else {
                    name: e6_summary(
                        rows,
                        name,
                        expected_early=[
                            str(x.get("source_world", i))
                            for i, x in enumerate(roots)
                            if x["stratum"] == "early"
                        ],
                        expected_ordinary=[
                            str(x.get("source_world", i))
                            for i, x in enumerate(roots)
                            if x["stratum"] == "ordinary"
                        ],
                    )
                    for name in ("R", "F")
                }
            )
        )
        if time.perf_counter() - begun > cap_seconds:
            raise IncompleteResult("E6 allocation cap expired")
        value = {
            "schema": "bl-prefix-e6-results/v1",
            "preflight": preflight,
            "scientific": not preflight,
            "claim_status": "COMPUTE_ONLY_UNCLAIMED",
            "status": "PREFLIGHT_ONLY" if preflight else "COMPLETE",
            "workers": workers,
            "rows": rows,
            "analysis": analyses,
            "bank_sha256": _digest(bank_path),
            "audit_sha256": _digest(audit_path),
            "b8_sha256": _digest(checkpoints["B8"]),
            "controller_sha256": {name: _digest(path) for name, path in checkpoints.items()},
            "elapsed_seconds": time.perf_counter() - begun,
        }
    except (KeyError, OSError, ValueError, IncompleteResult, RuntimeError) as exc:
        value = {
            "schema": "bl-prefix-e6-results/v1",
            "preflight": preflight,
            "scientific": False,
            "claim_status": "COMPUTE_ONLY_UNCLAIMED",
            "status": "INCOMPLETE",
            "reason": str(exc),
            "elapsed_seconds": time.perf_counter() - begun,
        }
    export_json(Path(out) / "e6-results.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(
        json.loads(args.plan.read_text()),
        args.out,
        preflight=args.preflight,
        compute_analysis=args.preflight,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == ("PREFLIGHT_ONLY" if args.preflight else "COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
