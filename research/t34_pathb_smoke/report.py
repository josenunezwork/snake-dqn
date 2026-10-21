"""Audit and render the excluded development smoke's immutable receipts."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from research.t34_pathb_smoke.run import OUT, ROOT, delivered_search, load, sha, summarize, write


def audit(out: Path) -> tuple[dict, list[dict]]:
    """Reconcile fixed coverage, hashes, actual applications and controller faults."""
    receipt = load(out / "receipt.json")
    start = load(out / "start.json")
    assert receipt["status"] == "COMPLETE", receipt.get("error")
    assert receipt["inputs_unchanged"] and receipt["source_unchanged"]
    assert sha(out / "start.json") == receipt["start_sha256"]
    rows = []
    for name, digest in receipt["rows"].items():
        assert sha(out / name) == digest, name
        row = load(out / name)
        assert row["status"] == "COMPLETE"
        assert all(
            row["controller"]["counts"][k] == 0
            for k in ("worker_errors", "invalid_results", "unconditional_faults")
        )
        assert row["v2_lad"] == load(out / f'lad-{row["id"]}.json')
        assert row["detections"] and row["detections"][0]["actual_live_frame"] is not None
        assert row["lead_frames_to_last_avoidable_decision"] == (
            row["v2_lad"]["lad"] - row["detections"][0]["actual_live_frame"]
        )
        if row["applied"]:
            first = next(s for s in row["searches"] if s["status"] == "IMPROVING")
            initial_wall = first["finish_wall"] - row["first_improving_seconds_from_start"]
            matched = delivered_search(row["searches"], row["applied"], row["start"], initial_wall)
            assert bool(matched) == row["delivered_before_its_frame"]
            event = row["applied"]
            assert event["accepted"] and event["applied"] and not event["deadline_miss"]
            assert not event["fallback_reason"]
            if matched:
                outcomes = matched["outcomes"]
                chosen = outcomes[str(matched["action"])]
                # The selected branch is a complete recorded outcome; delay
                # versus rescue follows the producer's exact survival flag.
                assert (matched["kind"] == "rescue") == chosen["survives"]
                assert not chosen["survives"] or chosen["alive_steps"] == chosen["horizon"]
        else:
            assert not row["delivered_before_its_frame"]
        rows.append(row)
    assert len(rows) == 160
    assert summarize(rows) == receipt["summary"]
    for workers in (1, 6):
        subset = [r for r in rows if r["workers"] == workers]
        t3 = [r for r in subset if r["group"] == "T3"]
        assert len(t3) == 75 and len(subset) == 80
        assert len({r["id"].rsplit("-c", 1)[0] for r in t3}) == 22
        assert sum(r["historical_survival_rescue"] for r in t3) == 34
    for key in {r["id"] for r in rows}:
        paired = [r for r in rows if r["id"] == key]
        assert len(paired) == 2
        assert {r["workers"] for r in paired} == {1, 6}
        assert paired[0]["group"] == paired[1]["group"]
        assert all(
            paired[0][field] == paired[1][field]
            for field in ("start", "doom", "v2_lad", "historical_lad_d300")
        )
    assert all(sha(Path(p)) == digest for p, digest in start["inputs"].items())
    assert all(
        sha(ROOT / p) == digest for p, digest in start["provenance"]["source_sha256"].items()
    )
    result = {
        "status": "PASS",
        "rows": len(rows),
        "paired_dooms": 80,
        "source_files_verified": len(start["provenance"]["source_sha256"]),
        "input_files_verified": len(start["inputs"]),
        "receipt_sha256": sha(out / "receipt.json"),
        "checks": [
            "fixed inventory",
            "row/start/input/source hashes",
            "LAD diagnostic equality",
            "actual execution and ideal frame deadline",
            "controller faults",
            "paired roots and LADs",
            "summary and stopping rule",
        ],
        "source_commit": receipt["source_commit"],
    }
    return result, rows


def render(out: Path, audit_result: dict, rows: list[dict]) -> str:
    """Write a reviewable result with all eighty paired doom measurements."""
    receipt = load(out / "receipt.json")
    verdict = receipt["summary"]["stopping_rule_verdict"]
    recommendation = {
        "STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY": (
            "Do not start operator-02. Both configurations fall below 50%; do not run "
            "the paced study."
        ),
        "STOP_OPERATOR02_AMEND_SECTION4_AND_RETIME": (
            "Do not start operator-02. Amend section 4 to the worker count and re-time "
            "before reconsidering the chain."
        ),
        "N1_DELIVERABLE_DEVELOPMENT_ONLY": (
            "N=1 meets the development delivery threshold. Registered timing, efficacy "
            "and serving requirements remain separate."
        ),
    }[verdict]
    lines = [
        "# Path B v2 deliverability smoke — 2026-10-21",
        "",
        f'**Stopping-rule verdict: {receipt["summary"]["stopping_rule_verdict"]}.**',
        recommendation,
        "",
        (
            "DEVELOPMENT ONLY — excluded from every registered verdict. This result "
            "changes no registration,"
        ),
        (
            "product thread contract, serving configuration, or release. Next steps "
            "follow the precommitted"
        ),
        "stopping rule; this report does not perform a section 4 amendment or admit operator-02.",
        "",
        (
            "| Configuration | Delivered improvements / 75 T3 changes | "
            "Horizon-surviving rescues delivered | Historical rescue intervals rescued / "
            "34 | M3 delivered / 5 |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for workers in (1, 6):
        t3 = [r for r in rows if r["workers"] == workers and r["group"] == "T3"]
        delivered = [r for r in t3 if r["delivered_before_its_frame"]]
        rescues = [r for r in delivered if r["applied"]["intervention_kind"] == "rescue"]
        historic = sum(r["historical_survival_rescue"] for r in rescues)
        m3 = sum(
            r["delivered_before_its_frame"]
            for r in rows
            if r["workers"] == workers and r["group"] == "M3"
        )
        lines.append(
            f"| N={workers} | {len(delivered)}/75 = {len(delivered)/75:.6%} | "
            f"{len(rescues)} | {historic}/34 | {m3}/5 |"
        )
    lines += [
        "",
        "The 75 changes occupy 22 original arm/environment trajectories. Of the 75 original chosen",
        (
            "actions, 34 survived the historical D300 horizon and 41 were delay-only. "
            "The historical 54/64"
        ),
        (
            "number is episode survival, not the delivery denominator. "
            "Controller-labelled rescue horizons"
        ),
        "are up to H500 and truncate at terminal frame 5,000, as in the unchanged adapter.",
        "",
        "## Execution and interpretation",
        "",
        f'- Run source: `{receipt["source_commit"]}`; '
        "stopping-rule commit: `7622d62e` (committed/pushed before data).",
        f'- Wall time: {receipt["wall_seconds"]:.3f} seconds '
        f'({receipt["wall_seconds"]/60:.3f} minutes), cap 6,900 seconds.',
        "- Mac CPU, 24 fps absolute deadlines, N=1 followed by N=6. Numerical pools capped at one.",
        (
            "- N=6 uses one existing controller worker plus five helpers; main tick "
            "thread makes seven active compute threads at most."
        ),
        (
            "- All periodic AC, open-lid and thermal checks passed. Phase 0 and other "
            "owner jobs remained active; this is a contended development measurement."
        ),
        (
            "- Only the new smoke package changed. Path B source, registrations, frozen "
            "worktrees, existing processes, tmux sessions, checkpoints and serving pins "
            "were untouched. The sealed Round 0 study directory was never opened."
        ),
        "",
        (
            "Each interval reconstructs earlier original T3 overrides, then starts a "
            "fresh actual v2 adapter"
        ),
        "at max(previous change + 1, doom - 499). Cold shadow fill is charged from before adapter",
        (
            "construction. The original controller runs without a search wait at any "
            "tick; the interval ends"
        ),
        (
            "at its first applied change or original greedy doom. An actual application "
            "also needs a result"
        ),
        (
            "completed before its absolute 24 fps frame deadline to count as delivered. "
            "This prevents tick"
        ),
        (
            "overruns donating extra search time. These conditional cold-root replays "
            "measure opportunity"
        ),
        (
            "delivery; they do not establish continuous-trajectory efficacy or "
            "quiet-host serving timing."
        ),
        "",
        "The six-thread wrapper partitions chronological branch points, calls the unchanged native",
        (
            "evaluator, latches the first complete winner, and cooperatively drains "
            "sibling chunks. Drain"
        ),
        "cost is charged to publication. Results apply to this wrapper and the recorded load.",
        "",
        "The operational LAD below is the exact backward-scanned last H500 **improving** decision:",
        (
            "either a surviving alternative or a strict survival delay. It is separate "
            "from the preserved"
        ),
        (
            "historical D300 survival-only LAD. Lead uses the live frame observed when "
            "worker doom detection"
        ),
        (
            "actually finishes, rather than its older source-frame metadata. Bracketed "
            "lead also uses the"
        ),
        (
            "ideal 24 fps wall-clock frame at that moment; negative means its ideal "
            "deadline already passed."
        ),
        "Diagnostic LAD scans run outside",
        "paced windows and provide no search-result cache.",
        "",
        "| Timing/accounting | N=1 | N=6 |",
        "|---|---:|---:|",
    ]
    for label, fn in (
        ("Completed improving evaluations already late", lambda r: r["evaluated_late_count"]),
        ("Controller late discards", lambda r: r["late_count"]),
        ("Controller stale discards", lambda r: r["stale_count"]),
        ("Tick deadline overruns", lambda r: r["tick_overruns"]),
    ):
        values = [sum(fn(r) for r in rows if r["workers"] == w) for w in (1, 6)]
        lines.append(f"| {label} (all 80 dooms) | {values[0]} | {values[1]} |")
    kinds = {
        w: dict(
            collections.Counter(
                r["applied"]["intervention_kind"]
                for r in rows
                if r["workers"] == w and r["delivered_before_its_frame"]
            )
        )
        for w in (1, 6)
    }
    lines += [
        "",
        f"Delivered intervention kinds: `{json.dumps(kinds, sort_keys=True)}`.",
        (
            "All scored controller states have zero worker, invalid-result and "
            "unconditional-fault counts."
        ),
        (
            "Searches still active at the original doom are cooperatively cancelled by "
            "adapter close. The"
        ),
        (
            "engine records that drain as INVALID; these terminal shutdown records are "
            "retained separately"
        ),
        "from the scored controller state. No failed unit was retried.",
        "",
        "## M3 pacing",
        "",
        (
            "The target was real 24 fps pacing. The achieved rates below expose failures "
            "to sustain that"
        ),
        "pace; absolute 24 fps deadlines still govern delivery even when scored ticks fall behind.",
        "",
        "| M3 doom | Achieved fps N=1 / N=6 | Tick overruns N=1 / N=6 |",
        "|---|---:|---:|",
    ]
    for key in sorted({r["id"] for r in rows if r["group"] == "M3"}):
        pair = sorted((r for r in rows if r["id"] == key), key=lambda r: r["workers"])
        rates = " / ".join(f'{r["achieved_fps"]:.3f}' for r in pair)
        overruns = " / ".join(str(r["tick_overruns"]) for r in pair)
        lines.append(f"| {key} | {rates} | {overruns} |")
    lines += [
        "",
        "## Per-doom measurements",
        "",
        (
            "Pairs are N=1 / N=6. First improving time is seconds from observed doom "
            "detection to the first"
        ),
        (
            "fully evaluated improving candidate, including candidates whose frame has "
            "passed. `—` means none"
        ),
        (
            "observed through cooperative shutdown/drain. First-improvement timing can "
            "include a drain completion. Delivery is `D` for executed delay, `R` for "
            "executed rescue, `—` for none."
        ),
        "Late evaluations and controller late/stale discards are distinct counters.",
        "",
        (
            "| Doom ID | Original doom | Observed [ideal-24fps] lead | First full "
            "improvement, s | Delivered | Late evaluations | Controller late/stale |"
        ),
        "|---|---:|---:|---:|---|---:|---|",
    ]

    def number(value: Any) -> str:
        return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    def lead_text(row: dict) -> str:
        observed = number(row["lead_frames_to_last_avoidable_decision"])
        ideal = number(row["v2_lad"]["lad"] - row["detection_live_frame_at_24fps"])
        return f"{observed} [{ideal}]"

    for key in sorted({r["id"] for r in rows}):
        pair = sorted((r for r in rows if r["id"] == key), key=lambda r: r["workers"])
        lead = " / ".join(lead_text(r) for r in pair)
        latency = " / ".join(
            number(r["time_to_first_fully_evaluated_improving_action_seconds"]) for r in pair
        )
        delivered = " / ".join(
            (
                ("R" if r["applied"]["intervention_kind"] == "rescue" else "D")
                if r["delivered_before_its_frame"]
                else "—"
            )
            for r in pair
        )
        late = " / ".join(str(r["evaluated_late_count"]) for r in pair)
        stale = " ; ".join(f'{r["late_count"]}/{r["stale_count"]}' for r in pair)
        lines.append(
            f'| {key} | {pair[0]["doom"]} | {lead} | {latency} | {delivered} | {late} | {stale} |'
        )
    lines += [
        "",
        "## Evidence and verification",
        "",
        f"- Artifact root: `{out}`.",
        f'- `receipt.json` SHA-256: `{audit_result["receipt_sha256"]}`.',
        f'- `start.json` SHA-256: `{receipt["start_sha256"]}`.',
        f"- Audit: PASS, 160/160 rows, 80 paired dooms, "
        f'{audit_result["source_files_verified"]} source hashes and '
        f'{audit_result["input_files_verified"]} input hashes verified.',
        (
            "- Targeted package tests: 5 passed. Native serial/six-thread outcomes "
            "matched; first-winner cancellation, fixed denominator and later on-time "
            "application checks passed."
        ),
        "- Black, isort and flake8 passed on the new package. No broader tests were run.",
        (
            "- Raw `n1-*.json`, `n6-*.json` and `lad-*.json` retain all per-doom fields "
            "and outcome details; raw artifacts remain outside Git."
        ),
        "",
        (
            "The current operator-02 chain remains unstarted by this task. Any future "
            "engineering revision"
        ),
        "needs its own development measurement and prospective product/registration process.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("audit requires Python assertions enabled")
    result, rows = audit(args.out)
    text = render(args.out, result, rows)
    if not args.check:
        write(args.out / "audit.json", result)
        with (args.out / "RESULT.md").open("x") as stream:
            stream.write(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
