"""Auditable T1 routing summaries, artifact hashes and results document."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.parity import write_json
from research.t1_commitment_20261015.replay import ARTIFACTS, INPUTS

REPO = Path(__file__).resolve().parents[2]
DOCUMENT = REPO / "docs/research/t1_commitment_study_2026-10-15.md"


def quartiles(values: Sequence[float]) -> Any:
    return [float(v) for v in np.percentile(values, [25, 50, 75])] if len(values) else None


def fraction(numerator: int, denominator: int) -> dict:
    return {
        "n": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def censored_quartiles(rows: Sequence[dict]) -> list:
    """Empirical inverse-CDF ranks, retaining missing horizons above 300."""
    known = sorted(r["H_LAD"] for r in rows if r["H_LAD"] is not None)
    result = []
    for p in (0.25, 0.5, 0.75):
        rank = int(np.ceil(p * len(rows)))
        result.append(known[rank - 1] if 0 < rank <= len(known) else ">300 (censored)")
    return result


def topology_diff(lad: dict) -> bool:
    chosen, survivor = lad["chosen_geometry"], lad["best_surviving_geometry"]
    return (
        chosen["tail_in_or_on_boundary"] != survivor["tail_in_or_on_boundary"]
        or chosen["min_own_boundary_ttl"] != survivor["min_own_boundary_ttl"]
    )


def summarize_group(rows: Sequence[dict]) -> dict:
    n = len(rows)
    lads = [r["lad"] for r in rows if r["lad"] is not None]
    onsets = [r["onset"] for r in rows if r["onset"] is not None]
    return {
        "deaths": n,
        "lad_available": len(lads),
        "none_within_300": n - len(lads),
        "H_LAD_quartiles_censored": censored_quartiles(rows),
        "H_LAD_quartiles_observed": quartiles([r["H_LAD"] for r in rows if r["H_LAD"] is not None]),
        "survivor_within": {
            str(h): fraction(sum(r["H_LAD"] is not None and r["H_LAD"] <= h for r in rows), n)
            for h in (30, 100, 150, 300)
        },
        "Q_gap_quartiles": quartiles([m["q_gap"] for m in lads]),
        "Q_gap_below_2": fraction(sum(m["q_gap"] < 2 for m in lads), len(lads)),
        "Q_gap_above_5": fraction(sum(m["q_gap"] > 5 for m in lads), len(lads)),
        "tail_or_boundary_TTL_diff": fraction(sum(topology_diff(m) for m in lads), len(lads)),
        "boost_survives_at_LAD": fraction(sum(m["any_surviving_boost"] for m in lads), n),
        "chosen_boost_at_LAD": fraction(sum(m["chosen_is_boost"] for m in lads), len(lads)),
        "sealer_owner_counts": dict(
            Counter((r["sealer"] or {}).get("owner", "no_t0") for r in rows)
        ),
        "sealer_opponent_kind_counts": dict(
            Counter(
                r["sealer"]["kind"]
                for r in rows
                if r["sealer"] and r["sealer"].get("owner") == "opponent"
            )
        ),
        "onsets_available": len(onsets),
        "head_side_region_at_t0_quartiles": quartiles(
            [m["head_region"]["component_size"] for m in onsets]
        ),
        "chosen_next_cell_region_at_t0_quartiles": quartiles(
            [m["chosen_geometry"]["component_size"] for m in onsets]
        ),
        "opponent_head_inside_at_t0": fraction(
            sum(m["chosen_geometry"]["opponent_head_inside"] for m in onsets), len(onsets)
        ),
        "opponent_head_in_or_on_boundary_at_t0": fraction(
            sum(m["chosen_geometry"]["opponent_head_in_or_on_boundary"] for m in onsets),
            len(onsets),
        ),
        "recorded_death_controls_passed": sum(r["recorded_death_reproduced"] for r in rows),
        "rollout_env_frames": sum(r["rollout_env_frames"] for r in rows),
        "all_step1_env_frames": sum(r["env_frames"] for r in rows),
        "summed_worker_seconds": sum(r["wall_seconds"] for r in rows),
    }


def route(groups: dict) -> dict:
    all_rows, late = groups["all"], groups["late"]
    censored_median = late["H_LAD_quartiles_censored"][1]
    median = censored_median if isinstance(censored_median, (int, float)) else None
    r1 = late["survivor_within"]["150"]["fraction"] >= 0.6 and median is not None and median <= 100
    r1_off = late["survivor_within"]["300"]["fraction"] < 0.3 or median is None
    r2 = (all_rows["Q_gap_below_2"]["fraction"] or 0) > 0.5
    r2_prime = (all_rows["Q_gap_above_5"]["fraction"] or 0) > 0.5
    topology = all_rows["tail_or_boundary_TTL_diff"]["fraction"]
    r3 = (
        "GO"
        if topology is not None and topology >= 0.7
        else ("NO_GO" if topology is not None and topology < 0.5 else "unresolved")
    )
    return {
        "R1": {"fires": r1, "verdict": "T3 eligible" if r1 else "does not fire"},
        "R1_prime": {"fires": r1_off, "verdict": "T3 off" if r1_off else "does not fire"},
        "R2": {
            "fires": r2,
            "verdict": "value learning is the limit" if r2 else "does not fire",
            "part_c": "not run here; joint interpretation pending",
        },
        "R2_prime": {
            "condition_met": r2_prime,
            "verdict": (
                "Q-gap condition met; perception verdict pending Part C"
                if r2_prime
                else "does not fire"
            ),
        },
        "R3": {"verdict": r3},
        "R4": {
            "fires": late["boost_survives_at_LAD"]["fraction"] >= 0.2,
            "verdict": (
                "include boost labels in T3"
                if late["boost_survives_at_LAD"]["fraction"] >= 0.2
                else "does not fire"
            ),
        },
    }


def load_rows(out: Path) -> list:
    rows = []
    for arm in census.B8_ARMS:
        status = json.loads((out / "rows" / f"{arm}.status.json").read_text())
        assert status["status"] == "PASS", status
        expected = json.loads((INPUTS / f"{arm}.json").read_text())["deaths"]
        actual = [
            json.loads(line) for line in (out / "rows" / f"{arm}.jsonl").read_text().splitlines()
        ]
        assert {(r["env"], r["fatal_t"]) for r in actual} == {
            (r["env"], r["fatal_t"]) for r in expected
        }
        assert len(actual) == len(expected)
        for row in actual:
            assert row["recorded_death_reproduced"]
            assert len(row["region_series"]) == min(60, row["fatal_t"] + 1)
            assert row["region_series"][-1]["decision"] == row["fatal_t"]
            assert all(
                m["outcomes"][str(m["chosen_action"])]["control_reproduced"] for m in row["scans"]
            )
            observed = [
                m["decision"]
                for m in row["scans"]
                if m["surviving_actions"]
                and max(0, row["fatal_t"] - 300) <= m["decision"] < row["fatal_t"]
            ]
            assert (max(observed) if observed else None) == (
                row["lad"]["decision"] if row["lad"] else None
            )
        rows.extend(actual)
    assert len(rows) == 75
    assert sum(r["stratum"] == "late" for r in rows) == 49
    assert sum(r["stratum"] == "early" for r in rows) == 26
    return rows


def manifest(out: Path) -> dict:
    """Pin inputs/checkpoints and all Python implementation dependencies."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m2_20261008 import ni_spec

    inputs = sorted(INPUTS.glob("*.json")) + sorted(INPUTS.glob("*.npz"))
    inputs += [census.b8_path(arm) for arm in census.B8_ARMS]
    inputs += [dev_screen.DEFAULT_CONFIG]
    inputs += [dev_screen.DEFAULT_CHECKPOINT_DIR / name for name, _ in dev_screen.POOL]
    inputs += [dev_screen.DEFAULT_CHECKPOINT_DIR / ni_spec.FRP3_S12[0]]
    inputs += [
        REPO / p
        for p in (
            "AGENTS.md",
            "docs/STATUS.md",
            "docs/research/specs/t1a_commitment_spec_2026-10-15.md",
            "docs/research/panel_review_summary_2026-10-14.md",
            "docs/research/panel_mac_answers_2026-10-14.md",
            "docs/research/panel_2026-10-15/PANEL_DECISION_MEMO.md",
            "configs/eval_free_space.yaml",
        )
    ]
    sources = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "research").rglob("*.py"))
    sources += sorted((REPO / "tests").glob("test_t1*.py"))
    result = {
        "inputs": {str(p): census.sha256_file(p) for p in inputs},
        "source_files": {str(p.relative_to(REPO)): census.sha256_file(p) for p in sources},
        "git": census._git(),
    }
    write_json(out / "manifest.json", result)
    return result


def present_fraction(value: dict) -> str:
    p = value["fraction"]
    return f"{value['n']}/{value['denominator']} ({p:.1%})" if p is not None else "unavailable"


def build_report(out: Path) -> dict:
    rows = load_rows(out)
    run = json.loads((out / "run.json").read_text())
    parity = json.loads((out / "parity.json").read_text())
    sealer_review = json.loads((out / "sealer-review.json").read_text())
    assert parity["status"] == run["status"] == "PASS"
    assert sealer_review["status"] == "PASS"
    assert not run["partial_pilot"]
    groups = {
        "all": summarize_group(rows),
        "late": summarize_group([r for r in rows if r["stratum"] == "late"]),
        "early": summarize_group([r for r in rows if r["stratum"] == "early"]),
    }
    summary = {
        "schema": "t1a-summary/v1",
        "authority": "development diagnostic, not gate evidence",
        "groups": groups,
        "rules": route(groups),
        "wall_seconds": run["wall_seconds"] + sealer_review["wall_seconds"],
        "rollout_phase_wall_seconds": run["wall_seconds"],
        "sealer_review_wall_seconds": sealer_review["wall_seconds"],
        "sealer_review_env_frames": sealer_review["env_frames"],
        "sealer_review_classifications_changed": sum(
            c["classification_changed"] for arm in sealer_review["arms"] for c in arm["corrections"]
        ),
        "references_run": False,
        "part_c_run": False,
    }
    write_json(out / "summary.json", summary)
    with (out / "rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest(out)
    initial = DOCUMENT.read_text()
    rule_section = initial.split("## Method and status")[0].split("## Method and results")[0]
    late, early = groups["late"], groups["early"]
    lines = [
        rule_section.rstrip(),
        "",
        "## Method and results",
        "",
        "All 75 B8 census deaths were completed: 49 late (L >= 800) and 26 early.",
        "Each arm uses its original final checkpoint and census world seed/roster.",
        "The executor captures the actual census evaluator's initial world and policies:",
        "grid engine, terminal hero, respawning scripted opponents, fast anchors,",
        "rowwise ego2s inference, advisory mask, eps 0, no B8 veto.",
        "",
        "Recorded actions fast-forward the original trajectory. Every pre-action hero",
        "length is checked against the census series. All advisory-legal actions, boosts",
        "included, are followed by the same hero's own greedy continuation for 300",
        "transitions or to decision 5000. The recorded action is a separate control",
        "when advisory-unsafe; every control must reproduce the original death decision.",
        "",
        "Indices are zero-based `fatal_t`/`t0`; census `frame` is one-based after preparation.",
        "At `d = fatal_t - 300`, the control continues one extra transition beyond the",
        "300-transition endpoint to verify death. Controls never count as alternatives.",
        "The stride-10 scan includes f-10; the latest positive window is refined at stride 1.",
        "The separately measured t0 can also establish a later positive inside the scan window.",
        "All outcomes, controls, Q vectors and chosen/survivor geometry are retained in rows.",
        "",
        "| Quantity | Late (49) | Early (26) |",
        "|---|---|---|",
    ]
    for label, key in (
        ("LAD observed", "lad_available"),
        ("None within 300", "none_within_300"),
        ("H_LAD quartiles (all deaths, censored ranks)", "H_LAD_quartiles_censored"),
        ("H_LAD quartiles (observed)", "H_LAD_quartiles_observed"),
        ("Q-gap quartiles at LAD", "Q_gap_quartiles"),
        ("Head-side region at t0 quartiles", "head_side_region_at_t0_quartiles"),
        ("Chosen next-cell region at t0 quartiles", "chosen_next_cell_region_at_t0_quartiles"),
    ):
        lines.append(f"| {label} | {late[key]} | {early[key]} |")
    for h in (30, 100, 150, 300):
        lines.append(
            f"| Survivor within {h} decisions | "
            f"{present_fraction(late['survivor_within'][str(h)])} | "
            f"{present_fraction(early['survivor_within'][str(h)])} |"
        )
    for label, key in (
        ("Q-gap <2 at LAD", "Q_gap_below_2"),
        ("Q-gap >5 at LAD", "Q_gap_above_5"),
        ("Tail/boundary-TTL contrast", "tail_or_boundary_TTL_diff"),
        ("Surviving boost at LAD / deaths", "boost_survives_at_LAD"),
        ("Opponent head literal inside at t0", "opponent_head_inside_at_t0"),
        ("Opponent head in/on component boundary at t0", "opponent_head_in_or_on_boundary_at_t0"),
    ):
        lines.append(
            f"| {label} | {present_fraction(late[key])} | {present_fraction(early[key])} |"
        )
    lines += ["", "| Rule | Verdict |", "|---|---|"]
    lines += [f"| {rule} | {verdict['verdict']} |" for rule, verdict in summary["rules"].items()]
    lines += [
        "",
        "R1 uses the full late-death median (rank 25 of 49) and the full 49-death denominator",
        "for its survivor fraction. Censored quartiles use inverse empirical-CDF ranks",
        "ceil(p*N), with missing horizons retained above 300 under the prescribed search.",
        "Observed-only quartiles use NumPy's linear interpolation and are shown separately.",
        "Missing LADs' exact horizons are unknown. R2/R2' use all observed LADs and retain the",
        "Part C qualification. R3 compares the highest-Q surviving alternative with the",
        "chosen action; empty boundary TTL is represented by null, not zero.",
        "",
        f"Sealer owners, late: `{late['sealer_owner_counts']}`; "
        f"early: `{early['sealer_owner_counts']}`.",
        f"Resolved opponent kinds, late: `{late['sealer_opponent_kind_counts']}`; "
        f"early: `{early['sealer_opponent_kind_counts']}`.",
        "",
        "## Geometry conventions",
        "",
        "Components are exact four-connected free regions, uncapped. The explicit spec",
        "rule is used: empty cells and own-body TTL <=1 cells are free; opponent segments",
        "block. This differs from ego_raster_b's all-owner tail-release rule. TTL is",
        "logical length minus head-first segment index. Normal/boost variants use the",
        "same first next cell. Tail membership includes the component and its boundary.",
        "For the head-side region, only the current hero head is temporarily opened as",
        "the flood seed. The region series covers fatal_t-59 through fatal_t inclusive.",
        "A live opponent head is ordinarily blocked: literal component membership and",
        "four-adjacent boundary membership are reported separately to avoid conflating them.",
        "Sealers require a newly occupied cell with a demonstrated split; ambiguous",
        "multi-cell/multi-owner cases remain unresolved. No proximity guess is used.",
        "",
        "## Parity, timing and verification",
        "",
        "The unmodified census clone passed two B8_s0 death worlds (source envs 3 and 8):",
        "2,500 recorded warmup decisions followed by 1,000 paired clone decisions each.",
        "All 9,000 checked length values and 2,002 full-state comparisons matched,",
        "including raw body storage, grids/pads, food order/sets, RNG, policy state, masks,",
        "timers and event/scratch arrays. Clone storage independence and restored pad",
        "views passed. Prepared callback clones normalize only lifecycle guards to the",
        "census pre_step phase, then apply their first action with step(), without preparation.",
        "",
        "Early-state benchmark at decision 100 (warm paths, one CPU thread per worker):",
        "",
        "| E | Clone (ms) | Step without hero (ms) | Step with hero (ms) | 300 decisions (s) |",
        "|---|---|---|---|---|",
    ]
    for timing in parity["timing"]:
        lines.append(
            f"| {timing['world_count']} | {1000*timing['clone_seconds']:.3f} | "
            f"{1000*timing['step_without_hero_forward_seconds']:.3f} | "
            f"{1000*timing['step_with_hero_forward_seconds']:.3f} | "
            f"{timing['rollout_300_seconds']:.3f} |"
        )
    lines += [
        "",
        f"Step 1 wall time: {run['wall_seconds']:.2f} s with {run['workers']} worker processes.",
        f"Separately timed sealer review: {sealer_review['wall_seconds']:.2f} s, "
        f"{sealer_review['env_frames']:,} recorded replay env-frames; "
        f"{summary['sealer_review_classifications_changed']} owner classifications changed.",
        f"Combined study and sealer-review wall time: {summary['wall_seconds']:.2f} s.",
        f"Counterfactual/control rollout env-frames: {groups['all']['rollout_env_frames']:,}.",
        "All Step 1 env-frames, including recorded/refinement replay: "
        f"{groups['all']['all_step1_env_frames']:,}.",
        "Parity/timing and the preliminary pilot are separate from those totals.",
        "Rollout workers used the frozen launch commit in launch-head.txt. A reviewed",
        "joint-cell seal correction was applied in a separate recorded-action replay only:",
        "no counterfactual outcome, LAD, Q-gap or action geometry was changed. Original",
        "rows remain in rows-before-sealer-review; sealer-review.json pins source and costs.",
        "Raw verification output is in `verification.log` under the artifact directory.",
        "",
        "## Caveats and artifacts",
        "",
        "Small counts limit interpretation. 'No survivor' means 'no survivor under B8's own",
        "greedy continuation' at the sampled decisions, not impossibility under other",
        "policies or multi-action interventions. The stride-10 search can miss narrow",
        "positive windows outside its final positive refinement window; LAD is measured",
        "under that prescribed search, not exhaustive over the entire 300 decisions.",
        "R2/R2' are descriptive Part A routing signals; Part C's representation probe is",
        "not part of this run. Optional 11 incumbent and 10 champion reference deaths were",
        "not run, to keep this pass within the mandatory B8 scope and two-process budget.",
        "No training, gate evaluation, serving change, remote upload or push occurred.",
        "",
        f"Artifacts: `{out}`. `rows.jsonl` (75 deaths), `rows/*.jsonl`, `summary.json`,",
        "`parity.json`, `timing.json`, `run.json`, `manifest.json`, README, detached-run logs",
        "and verification receipts retain the evidence. The manifest pins input/checkpoint",
        "SHA256s and implementation/dependency source hashes.",
        "",
    ]
    DOCUMENT.write_text("\n".join(lines))
    readme = [
        "# T1 Part A artifacts",
        "",
        "development diagnostic, not gate evidence",
        "",
        "Completed mandatory 75 B8 deaths; optional reference arms not run.",
        "`summary.json`: rules and all/late/early denominators; `rows.jsonl`: complete rows.",
        "`manifest.json`: exact SHA256s of inputs/checkpoints and Python dependencies.",
        "`parity.json` / `timing.json`: mandatory precondition and early-state benchmarks.",
        "`run.json` / `rows/*.status.json`: bounded workers, completion and PIDs.",
        "`rows-before-sealer-review/`: original frozen rollout rows, retained verbatim.",
        "`sealer-review.json` / `sealer-review/`: reviewed sealer replay provenance and costs.",
        "`pilot/` is an implementation pilot, excluded from final analysis totals.",
        "`*.exit` and `*.log`: detached tmux t1a-run receipts; no t2 session was touched.",
        "",
        "Reproduce from repository root:",
        "",
        "```sh",
        'export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        "export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu",
        "./venv/bin/python -m research.t1_commitment_20261015.parity",
        "./venv/bin/python -m research.t1_commitment_20261015.study --workers 2",
        "./venv/bin/python -m research.t1_commitment_20261015.sealer_refresh",
        "./venv/bin/python -m research.t1_commitment_20261015.report",
        "```",
        "",
        "Input SHA256s (also machine-readable in manifest.json):",
        "",
    ]
    hashes = json.loads((out / "manifest.json").read_text())["inputs"]
    readme += [f"- `{digest}`  `{path}`" for path, digest in hashes.items()]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        manifest(args.out)
    else:
        print(json.dumps(build_report(args.out), indent=2))


if __name__ == "__main__":
    main()
