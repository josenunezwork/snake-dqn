"""E7 B8-local frontier probe, with training selection isolated from confirmation."""

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
from research.bl_prefix.analysis import IncompleteResult, e7_summary
from research.bl_prefix.controllers import (
    B8Controller,
    StochasticB8Controller,
    calibrate_b8_temperature,
)
from research.bl_prefix.engine import HORIZON, load_root, mass_ledger, relocate_root, rollout


def _tail_mean_length(candidate: Any, baseline: Any, *, horizon_steps: int) -> float | None:
    """Mean candidate length over frames where the paired B8 continuation is dead."""
    if baseline.death_frame is None:
        return None
    # The terminal B8 step is recorded as zero mass, so it is the first dead
    # tail frame and must be included rather than dropped with ``len(trace)``.
    start = max(0, len(baseline.mass_trace) - 1)
    if not start < horizon_steps:
        return None
    candidate_trace = candidate.mass_trace + (0.0,) * max(
        0, horizon_steps - len(candidate.mass_trace)
    )
    tail = candidate_trace[start:horizon_steps]
    return sum(tail) / len(tail)


def _load_bank(
    path: Path,
    preflight: bool,
    b8_path: str | Path,
    audit_sha: str,
    relocations: Mapping[str, str],
    allowed_seeds: set[int],
) -> list[dict[str, Any]]:
    verify_export(path)
    bank = json.loads(path.read_text())
    rows = bank.get("early_roots")
    digest = hashlib.sha256(Path(b8_path).read_bytes()).hexdigest()
    if (
        bool(bank.get("preflight")) != preflight
        or bank.get("audit_sha256") != audit_sha
        or bank.get("b8", {}).get("sha256") != digest
    ):
        raise IncompleteResult("E7 bank audit/preflight/B8 identity mismatch")
    if not isinstance(rows, list) or len(rows) != (3 if preflight else 128):
        raise IncompleteResult("E7 needs the exact E6 early-source bank")
    identities = set()
    for item in rows:
        source = str(item.get("source_world"))
        if not source or source in identities or not isinstance(item.get("ancestors"), Mapping):
            raise IncompleteResult("malformed E7 source bank")
        identities.add(source)
        item["roots"] = {}
        for ancestor in (100, 150, 200):
            rel = item["ancestors"].get(str(ancestor), item["ancestors"].get(ancestor))
            if not isinstance(rel, str):
                raise IncompleteResult("missing E7 ancestor")
            root_path = path.parent / rel
            verify_export(root_path)
            root = relocate_root(load_root(root_path), relocations)
            if (
                root.seed not in allowed_seeds
                or str(root.seed) != source
                or root.mix != item["mix"]
            ):
                raise IncompleteResult("E7 root identity differs from bank")
            item["roots"][ancestor] = root
        frames = {ancestor: root.frame for ancestor, root in item["roots"].items()}
        if frames[100] - frames[150] != 50 or frames[150] - frames[200] != 50:
            raise IncompleteResult("E7 ancestor frames do not share one recorded death")
    return rows


def _source_rows(
    task: tuple[
        dict[str, Any], str, dict[str, str], tuple[int, int, int], tuple[int, ...], bool, int
    ],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Picklable deterministic E7 source worker; no selection outcome crosses workers."""
    source, b8_path, relocations, future, policy_seeds, preflight, max_steps = task
    b8 = B8Controller(b8_path)
    confirmations: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    offset = 0
    for ancestor, original in source["roots"].items():
        root = relocate_root(original, relocations)
        temperature = calibrate_b8_temperature(
            b8, [p.observation for p in root.prefix], [p.legal for p in root.prefix], 0.05
        )
        training, sampled = [], []
        for training_index in range(8):
            stochastic = StochasticB8Controller(
                b8_path, net=b8.net, temperature=temperature, seed=policy_seeds[offset]
            )
            offset += 1
            training.append(
                rollout(
                    root,
                    stochastic,
                    b8,
                    prefix_steps=64,
                    future_seed=future[0],
                    **({"max_steps": max_steps, "preflight": True} if preflight else {}),
                )
            )
            sampled.append(stochastic)
        train_b8 = rollout(
            root,
            b8,
            b8,
            prefix_steps=0,
            future_seed=future[0],
            **({"max_steps": max_steps, "preflight": True} if preflight else {}),
        )
        training_outcomes = []
        for training_index, (result, stochastic) in enumerate(zip(training, sampled)):
            ledger = mass_ledger(result, train_b8)
            training_outcomes.append(
                {
                    "source_world": source["source_world"],
                    "mix": root.mix,
                    "ancestor": ancestor,
                    "training_index": training_index,
                    "paired_delta": result.mass_sum / HORIZON - train_b8.mass_sum / HORIZON,
                    "candidate_survival": result.terminal_survival,
                    "b8_survival": train_b8.terminal_survival,
                    "realized_non_greedy": stochastic.realized_non_greedy_rate,
                    **ledger,
                }
            )
        selected_index = max(range(8), key=lambda i: training[i].mass_sum)
        selected = training[selected_index]
        selection.append(
            {
                "source_world": source["source_world"],
                "ancestor": ancestor,
                "temperature": temperature,
                "selected_prefix": selected_index,
                "training_scores": [x.mass_sum / HORIZON for x in training],
                "training_b8_score": train_b8.mass_sum / HORIZON,
                "actual_prefix_decisions": min(64, len(selected.action_tape)),
                "training_terminal_before64": len(selected.action_tape) < 64,
                "expected_non_greedy": 0.05,
                "realized_non_greedy": [x.realized_non_greedy_rate for x in sampled],
                "stochastic_decisions": [x.draw_count for x in sampled],
                "training_outcomes": training_outcomes,
            }
        )
        for stream, seed in enumerate(future[1:3]):
            candidate = rollout(
                root,
                b8,
                b8,
                prefix_steps=64,
                future_seed=seed,
                forced_actions=selected.action_tape[:64],
                **({"max_steps": max_steps, "preflight": True} if preflight else {}),
            )
            baseline = rollout(
                root,
                b8,
                b8,
                prefix_steps=0,
                future_seed=seed,
                **({"max_steps": max_steps, "preflight": True} if preflight else {}),
            )
            tail_mean = _tail_mean_length(candidate, baseline, horizon_steps=HORIZON - root.frame)
            confirmations.append(
                {
                    "source_world": source["source_world"],
                    "mix": root.mix,
                    "ancestor": ancestor,
                    "stream": stream,
                    "candidate_score": candidate.mass_sum / HORIZON,
                    "b8_score": baseline.mass_sum / HORIZON,
                    "candidate_survival": candidate.terminal_survival,
                    "b8_survival": baseline.terminal_survival,
                    "fallback_count": candidate.fallback_count,
                    "b8_death_frame": baseline.death_frame,
                    "b8_last_living_length": baseline.last_living_length,
                    "candidate_tail_mean_length": tail_mean,
                    "productive": bool(
                        candidate.mass_sum / HORIZON - baseline.mass_sum / HORIZON >= 20.0
                        and not baseline.terminal_survival
                        and candidate.terminal_survival
                        and baseline.last_living_length is not None
                        and tail_mean is not None
                        and tail_mean >= 0.9 * baseline.last_living_length
                    ),
                }
            )
    return selection, confirmations


def run(plan: Mapping[str, Any], out: Path, *, preflight: bool = False) -> dict[str, Any]:
    begun = time.perf_counter()
    preflight = bool(preflight or plan.get("preflight"))
    from research.bl_prefix.resource_limits import usable_seconds

    cap_seconds = usable_seconds("e7")
    try:
        if plan.get("experiment") != "e7" or bool(plan.get("preflight")) != preflight:
            raise IncompleteResult("E7 plan/preflight identity mismatch")
        workers = plan.get("workers", 2 if preflight else 12)
        if type(workers) is not int or not 1 <= workers <= (2 if preflight else 12):
            raise IncompleteResult("E7 worker cap differs from frozen protocol")
        if not preflight and workers != 12:
            raise IncompleteResult("scientific runs require exactly twelve workers")
        b8_path = plan["b8_checkpoint"]
        B8Controller(b8_path)  # Verify the pinned comparator before worker dispatch.
        audit_path = Path(str(plan["seed_audit"]))
        verify_export(audit_path)
        audit = json.loads(audit_path.read_text())
        seeds.verify_receipt(audit, round_index=int(audit.get("round_index", -1)))
        relocations = plan.get("checkpoint_relocations", {})
        if not isinstance(relocations, Mapping):
            raise IncompleteResult("missing verified checkpoint relocations")
        bank_path = Path(str(plan["bank"]))
        bank = _load_bank(
            bank_path,
            preflight,
            b8_path,
            hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            relocations,
            set(audit["allocation"]["preflight" if preflight else "source"]["seeds"]),
        )
        future = list(plan.get("future_seeds", []))
        policy_seeds = list(plan.get("exploration_seeds", []))
        allowed_future = set(audit["allocation"]["preflight" if preflight else "future"]["seeds"])
        allowed_explore = set(
            audit["allocation"]["preflight" if preflight else "exploration"]["seeds"]
        )
        if (
            len(future) != 3
            or len(set(future)) != 3
            or not set(future) <= allowed_future
            or len(policy_seeds) != len(bank) * 3 * 8
            or len(set(policy_seeds)) != len(policy_seeds)
            or not set(policy_seeds) <= allowed_explore
            or set(future) & set(policy_seeds)
        ):
            raise IncompleteResult("E7 lacks independent training/confirmation RNG seeds")
        if not preflight:
            from research.bl_prefix.operational_burns import require_unburned

            require_unburned([*future, *policy_seeds])
        confirmations: list[dict[str, Any]] = []
        selection: list[dict[str, Any]] = []
        tasks = []
        for index, source in enumerate(bank):
            start = index * 24
            tasks.append(
                (
                    source,
                    str(b8_path),
                    dict(relocations),
                    tuple(map(int, future)),
                    tuple(map(int, policy_seeds[start : start + 24])),
                    preflight,
                    int(plan.get("preflight_max_steps", 16)),
                )
            )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for index, (selected_rows, confirmation_rows) in enumerate(
                executor.map(_source_rows, tasks)
            ):
                if time.perf_counter() - begun > cap_seconds:
                    raise IncompleteResult("E7 allocation cap expired")
                selection.extend(selected_rows)
                confirmations.extend(confirmation_rows)
                if index and index % 4 == 0:
                    print(
                        json.dumps(
                            {
                                "progress": [index, len(tasks)],
                                "elapsed_seconds": time.perf_counter() - begun,
                            }
                        ),
                        flush=True,
                    )
        summary = (
            {"decision": "NOT_SCORED_PREFLIGHT", "reason": "tiny operational fixture"}
            if preflight
            else e7_summary(confirmations, [str(x["source_world"]) for x in bank])
        )
        if time.perf_counter() - begun > cap_seconds:
            raise IncompleteResult("E7 allocation cap expired")
        value = {
            "schema": "bl-prefix-e7-results/v1",
            "preflight": preflight,
            "scientific": not preflight,
            "status": "PREFLIGHT_ONLY" if preflight else "COMPLETE",
            "workers": workers,
            "selection": selection,
            "training_outcomes": [
                row for selection_row in selection for row in selection_row["training_outcomes"]
            ],
            "confirmations": confirmations,
            "analysis": summary,
            "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "b8_sha256": hashlib.sha256(Path(b8_path).read_bytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - begun,
        }
    except (KeyError, OSError, ValueError, IncompleteResult, RuntimeError) as exc:
        value = {
            "schema": "bl-prefix-e7-results/v1",
            "preflight": preflight,
            "scientific": False,
            "status": "INCOMPLETE",
            "reason": str(exc),
            "elapsed_seconds": time.perf_counter() - begun,
        }
    export_json(Path(out) / "e7-results.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.out, preflight=args.preflight)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == ("PREFLIGHT_ONLY" if args.preflight else "COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
