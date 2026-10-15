"""Render the completed development ceiling diagnostic and per-world pairing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research.t1_commitment_20261015.parity import write_json
from research.t3_rollout_labels_20261015.ceiling import ARMS
from research.t3_rollout_labels_20261015.trigger import ROOT

REPO = Path(__file__).resolve().parents[2]


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def render_ceiling(root: Path = ROOT) -> dict[str, Any]:
    """Require all 64 receipts and report episode and alive-frame endpoints."""
    summary = json.loads((root / "ceiling" / "summary.json").read_text())
    if summary["status"] != "PASS" or summary["episodes"] != 64:
        raise ValueError("the ceiling document requires all 64 required episodes")
    trigger = json.loads((root / "trigger.json").read_text())
    fidelity = json.loads((root / "fidelity.json").read_text())
    rows = [
        json.loads((root / "ceiling" / f"{arm}_{env}.json").read_text())
        for arm in ARMS
        for env in range(32)
    ]
    if any(row["status"] != "PASS" or not row["prefix_identity"] for row in rows):
        raise ValueError("ceiling identity check failed")
    verdict = "GO" if sum(row["episode_survival"] for row in rows) / 64 >= 0.65 else "NO_GO"
    if summary["go"] != (verdict == "GO"):
        raise ValueError("ceiling summary verdict mismatch")
    groups = {}
    for arm in (*ARMS, "pooled"):
        selected = rows if arm == "pooled" else [r for r in rows if r["arm"] == arm]
        base = [r["baseline"] for r in selected]
        groups[arm] = {
            "episodes": len(selected),
            "baseline_survivors": sum(r["survived"] for r in base),
            "shadow_survivors": sum(r["episode_survival"] for r in selected),
            "baseline_episode_survival": mean(base, "survived"),
            "shadow_episode_survival": mean(selected, "episode_survival"),
            "baseline_alive_frame_survival": mean(base, "survival_fraction"),
            "shadow_alive_frame_survival": mean(selected, "alive_frame_fraction"),
            "baseline_mass": mean(base, "mass_integral"),
            "shadow_mass": mean(selected, "mass_integral"),
        }
    total_decisions = sum(r["decisions"] for r in rows)
    scoped = sum(r["trigger_scope_decisions"] for r in rows)
    raw = sum(r["raw_trigger_count"] for r in rows)
    batches = sum(r["batch_count"] for r in rows)
    result = {
        "verdict": verdict,
        "groups": groups,
        "total_decisions": total_decisions,
        "length_eligible_decisions": scoped,
        "raw_triggers": raw,
        "batches": batches,
        "raw_rate_all_decisions": raw / total_decisions,
        "capped_rate_all_decisions": batches / total_decisions,
        "raw_rate_length_eligible": raw / scoped,
        "capped_rate_length_eligible": batches / scoped,
    }
    write_json(root / "ceiling-result.json", result)
    pairs = []
    for row in rows:
        pairs.append(
            {
                "arm": row["arm"],
                "env": row["env"],
                "world_seed": row["world_seed"],
                "baseline_episode_survival": int(row["baseline"]["survived"]),
                "shadow_episode_survival": int(row["episode_survival"]),
                **{f"delta_{k}": v for k, v in row["paired_delta"].items()},
                "shadow_death_decision": row["death_t"],
                "shadow_death_length": row["death_length"],
            }
        )
    with (root / "ceiling-pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = [
        "# T3 rollout-label ceiling test (2026-10-15)",
        "",
        "Development diagnostic, not promotion or serving evidence. The shadow uses "
        "simulator rollouts; any future training labels would also come from simulator "
        "rollouts. The intended served artifact remains a bare network with no lookahead.",
        "",
        f"**Verdict: {verdict}.** The fixed rule is pooled scripted episode survival "
        ">=0.65 on the 64 required episodes: B8_s0 and B8_s3, each on the 32 original "
        "round-52 census worlds. The optional extension was omitted to bound CPU work "
        "while T2 lanes/probes were active. No outcome-dependent threshold tuning occurred.",
        "",
        "## Rollout fidelity and throughput",
        "",
        f"The fixed random sample contains {fidelity['samples']} T1 world/decision pairs "
        f"and {sum(len(r['actions']) for r in fidelity['comparisons'])} legal continuations. "
        "Every continuation matched a fresh T1 single-world rollout on survival, death "
        "decision, transitions executed and alive transitions. Prepared-phase cloning, "
        "RNG independence, rowwise B8 inference and advisory masks preserve the census path.",
        "",
        f"Measured batched throughput: **{fidelity['env_frames_per_second']:.1f} env-frames/s** "
        f"and **{fidelity['batches_per_second']:.4f} decision batches/s**, including clone "
        f"construction ({fidelity['env_frames']:,} valid continuation env-frames in "
        f"{fidelity['wall_seconds']:.3f}s). Replay and single-world comparison costs are "
        "outside this throughput denominator. Single-thread CPU workers; at most three.",
        "",
        "## Trigger and ceiling procedure",
        "",
        f"Trigger: length >=600 and **-max(raw Q) > {trigger['threshold']!r}**. Chosen "
        "prospectively as the most selective observed per-death maximum cutoff meeting "
        "90% coverage after the five-decision cooldown. All six raw Q outputs enter "
        "the score. On the five B8 T1/M7 captures, "
        f"{trigger['covered_late_deaths']}/{trigger['late_deaths']} late deaths "
        f"({trigger['late_death_coverage']:.2%}) have a trigger in the last 150 decisions. "
        f"Raw trigger rate {trigger['raw_trigger_rate']:.4%}; capped batch rate "
        f"{trigger['capped_trigger_rate']:.4%}, both over all recorded alive decisions.",
        "",
        "Each triggered batch tests every advisory-legal action, boosts included, for "
        "300 transitions or the remaining H5000 horizon with the same B8 greedy "
        "continuation, eps 0, no veto. Select maximum alive transitions; ties prefer "
        "B8's current argmax, then highest Q and lowest index. There is at most one "
        "batch per five decisions; a state with no advisory-legal action retains "
        "B8's normal-only fallback. Before the first actual intervention each world "
        "must match recorded actions and lengths; unchanged episodes must reproduce "
        "census endpoint metrics exactly.",
        "",
        "## Paired results",
        "",
        "Episode survival is alive at H5000. Alive-frame survival is the mean post-step "
        "alive indicator over all 5000 frames. Mass is the mean post-step length over "
        "the same horizon with dead frames zero. Both runs use identical world seeds, "
        "rosters and checkpoint bytes. Baseline values are the census JSON records.",
        "",
        "| Arm | Baseline episodes | Shadow episodes | Baseline alive-frame | Shadow "
        "alive-frame | Baseline mass | Shadow mass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, group in groups.items():
        lines.append(
            f"| {arm} | {group['baseline_survivors']}/{group['episodes']} "
            f"({group['baseline_episode_survival']:.6f}) | "
            f"{group['shadow_survivors']}/{group['episodes']} "
            f"({group['shadow_episode_survival']:.6f}) | "
            f"{group['baseline_alive_frame_survival']:.6f} | "
            f"{group['shadow_alive_frame_survival']:.6f} | "
            f"{group['baseline_mass']:.3f} | {group['shadow_mass']:.3f} |"
        )
    lines += [
        "",
        f"Ceiling wall time: {summary['wall_seconds']:.2f}s. "
        f"{batches} rollout batches, {summary['action_changes']} actual changes from "
        f"B8's current greedy action, {summary['core_rollout_env_frames']:,} continuation "
        f"env-frames. On {total_decisions:,} decisions ({scoped:,} at length >=600), "
        f"raw trigger rate was {raw/total_decisions:.4%} overall / {raw/scoped:.4%} "
        f"among length-eligible states; capped rate was {batches/total_decisions:.4%} / "
        f"{batches/scoped:.4%} respectively.",
        "",
        "Death lengths below are actual pre-action lengths on the shadow trajectory. "
        "Complete death times, baseline identities, trigger decisions, candidate outcomes "
        "and paired endpoint deltas are retained per world.",
        "",
    ]
    for arm in ARMS:
        lengths = [r["death_length"] for r in rows if r["arm"] == arm and not r["episode_survival"]]
        lines.append(f"- {arm}: {len(lengths)} deaths; lengths {lengths}.")
    lines += [
        "",
        "## Interpretation and next step",
        "",
        (
            "Part D is eligible under the fixed ceiling rule. The label-bank document "
            "records its separate generation evidence."
            if verdict == "GO"
            else "Part D did not run: no rollout label bank was generated. Part C is built "
            "and tested regardless of the ceiling verdict, as explicitly required. "
            "Part E supplies the unexecuted template requested by the owner; label "
            "arms remain unavailable without a separately eligible bank."
        ),
        "",
        "Coverage of a trigger in the last 150 decisions does not prove that it fired "
        "before the last avoidable decision. This result applies to the fixed trigger "
        "and greedy continuations on a reused development bank. It is not an "
        "impossibility result for other triggers, depths or continuation policies. "
        "No training experiment, promotion, serving change, upload or push occurred.",
        "",
        f"Artifact root: `{root}`. Receipts: `fidelity.json`, `trigger.json`, "
        "`ceiling/summary.json`, `ceiling-result.json`, `ceiling-pairs.csv`, "
        "`ceiling/<arm>_<env>.json`. Launch commands, logs, exit statuses and input "
        "hashes are retained alongside these files.",
        "",
        "## Per-world paired changes",
        "",
        "| Arm | Census env | World seed | Baseline alive at H | Shadow alive at H | "
        "Delta alive-frame | Delta mass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['arm']} | {row['env']} | {row['world_seed']} | "
            f"{row['baseline_episode_survival']} | {row['shadow_episode_survival']} | "
            f"{row['delta_alive_frame_fraction']:+.6f} | {row['delta_mass_integral']:+.3f} |"
        )
    (REPO / "docs/research/t3_ceiling_test_2026-10-15.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    render_ceiling(parser.parse_args().root)


if __name__ == "__main__":
    main()
