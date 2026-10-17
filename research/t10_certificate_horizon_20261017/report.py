"""Render complete T10 numeric tables from audited, saved diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from research.t5b_risk_and_fuse_20261016.fuse import sha256
from research.t10_certificate_horizon_20261017.audit import verify_hashes
from research.t10_certificate_horizon_20261017.certificate import KS, ROOT
from research.t10_certificate_horizon_20261017.horizon import HORIZONS

DOC = Path("docs/research/t10_certificate_horizon_2026-10-17.md")
EVIDENCE = Path("docs/research/t10_certificate_horizon_2026-10-17_evidence")


def percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def auc(metric: dict | None) -> str:
    if metric is None:
        return "skipped / unavailable"
    lo, hi = metric["ci95"]
    return f"{metric['auc']:.4f} [{lo:.4f}, {hi:.4f}]"


def render(certificate: dict, horizon: dict, audit: dict) -> str:
    """Render all specified counts, death rows, conditional survival and AUCs."""
    c, h = certificate["summary"], horizon["results"]
    have_arrival = "60+24+18_arrival" in horizon["feature_sets"]
    fuse_count = sum(name.startswith("fuse_q_") for name in horizon["feature_sets"])
    lines = [
        "# T10 certificate audit and horizon-stratified AUC — 2026-10-17",
        "",
        "**Development diagnostics only.** No policy training, observation/reward change,",
        "promotion evaluation, serving change, GPU or external compute.",
        "",
        "## Verdicts",
        "",
        "<!-- verdicts:start -->",
        "VERDICTS_PENDING_NUMERICAL_REVIEW",
        "<!-- verdicts:end -->",
        "",
        "## Certificate definition and replay",
        "",
        "The spec does not assign a numeric deadline to “before it would be needed.”",
        "The disclosed operational convention is arrival at the **actual current physical",
        "own tail within L unit-speed cell steps**, with **128 steps** as a sensitivity check.",
        "This convention was raised with the user before the full run; no alternative",
        "deadline was supplied at launch. The flood reuses T9’s TTL-release relaxation",
        "`max(previous + 1, destination_TTL)` on four-neighbor cells. All owners release",
        "their existing body at its current TTL. Sources seed time zero. Race-128 counts",
        "hero arrivals within 128 steps strictly before all live opponents; ties lose.",
        "",
        "Waiting, reverse moves and routes through future trail are optimistic: growth,",
        "new trails, respawns, heading restrictions and movement feasibility during waiting",
        "are absent. **A positive flag is an access proxy, not a guaranteed safe Snake path.**",
        "The tail is taken from physical `seg_count`, even when logical L exceeds it.",
        "",
        "All 64 episode identities (B8_s0 and B8_s3, 32 worlds each) are replayed under",
        "both B8 and the recorded dense lookahead. The two arms share the same 32 world",
        "seeds. B8 uses recorded census actions; dense uses its recorded trigger choices",
        "and original B8 greedy continuation otherwise. No search is rerun. Every B8",
        "predecision length, dense trigger length/greedy action, intervention count,",
        "world/checkpoint identity, survival, alive-frame fraction and mass integral",
        "matches its source receipt; dense death decisions and lengths also match.",
        "",
        "Measurements include every prepared alive decision at L >= 800. The strict area",
        "thresholds are race-128 > L/2 and race-128 > L, not >=.",
        "Fractions pool decisions, not per-world means; the policies induce different",
        "state exposures and trajectory lengths. These are descriptive policy comparisons.",
        "",
        "| Policy | H5000 survivors | L >= 800 decisions | Tail by L | Tail by 128 | "
        "Race > L/2 | Race > L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("b8", "dense"):
        row = c[policy]
        cells = [
            f"{row['counts'][key]:,} ({percent(row['fractions'][key])})"
            for key in ("tail_L", "tail_128", "race_above_L2", "race_above_L")
        ]
        lines.append(
            f"| {policy} | {row['survivors']}/{row['episodes']} | "
            f"{row['eligible_decisions']:,} | " + " | ".join(cells) + " |"
        )
    lines += ["", "Deadline-equality sensitivity from the same saved arrivals:"]
    for policy in ("b8", "dense"):
        row = audit["certificate"]["deadline_sensitivity"][policy]
        lines.append(
            f"- {policy}: {row['strictly_before_L']:,}/{row['eligible_decisions']:,} "
            f"arrive strictly before L; {row['exactly_at_L']:,} arrive exactly at L."
        )
    lines += [
        "",
        "## Last certificate and loss duration before every death",
        "",
        "For fatal decision D and last eligible certified decision d, duration is D−d",
        "recorded decision transitions. Zero means the flag remains true on the fatal",
        "decision. These durations are time since the last **observed eligible** true",
        "state, not proven continuously false time across a later length dip below 800.",
        "“Never” means no true flag in eligible states; “no L800” means no eligible",
        "state at all. Both are censored/unobserved values, not zero.",
        "",
        "| Policy | Arm | Env | Death decision | Death L | Last tail-by-L | Duration | "
        "Last tail-by-128 | Duration |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("b8", "dense"):
        for row in c[policy]["deaths"]:
            cells = []
            for key in ("tail_L", "tail_128"):
                value = row[key]
                if value["status"] == "observed":
                    cells += [
                        str(value["last_true_decision"]),
                        str(value["decisions_before_death"]),
                    ]
                else:
                    cells += ["never" if value["status"].startswith("never") else "no L800", "—"]
            lines.append(
                f"| {policy} | {row['arm']} | {row['env']} | {row['death_decision']} | "
                f"{row['death_length']} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "| Policy | Certificate | Observed deaths | Unobserved deaths | "
        "Min / Q1 / median / Q3 / max duration |",
        "|---|---|---:|---:|---|",
    ]
    for policy in ("b8", "dense"):
        for key, row in c[policy]["loss_duration"].items():
            quantiles = " / ".join(f"{v:g}" for v in row["quartiles"]) if row["quartiles"] else "—"
            lines.append(
                f"| {policy} | {key} | {row['observed']} | {row['unobserved']} | {quantiles} |"
            )
    lines += [
        "",
        "## Survival conditional on consecutive certified decisions",
        "",
        "Each cell is H5000 survivors / episodes that **ever** held the flag at",
        "L >= 800 for k consecutive decisions. Each episode enters once. False flags",
        "and ineligible gaps reset the streak. This is a retrospective association",
        "conditioned on surviving long enough to enter; it is not causal or a safety",
        "guarantee. Decision-landmark counts are retained in the evidence JSON.",
        "",
        "| k | B8 tail by L | Dense tail by L | B8 tail by 128 | Dense tail by 128 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in KS:
        cells = []
        for field, policy in (
            ("tail_L", "b8"),
            ("tail_L", "dense"),
            ("tail_128", "b8"),
            ("tail_128", "dense"),
        ):
            row = c[policy]["conditional_survival"][field][str(k)]
            cells.append(f"{row['survivors']}/{row['episodes']} ({percent(row['survival'])})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Horizon-stratified AUC",
        "",
        "The immutable T5 bank contributes 19,976 states, 103,264 legal branches and",
        "342 worlds. Death on or before H transitions is positive, including exactly",
        "H; survival through D300 is negative at every requested H. Illegal branches",
        "are excluded. All horizons and inputs reuse T5’s saved five world-grouped",
        "folds, fixed GBM (200 iterations, 31 leaves, learning rate 0.1, no early",
        "stopping) and 2,000 whole-world bootstrap draws, seed 20261017. Intervals are",
        "95% percentile intervals of fixed out-of-fold predictions, not model-refit",
        "uncertainty. Six action one-hot columns are appended to every input design.",
        "Changing the horizon changes prevalence and the composition of positive outcomes;",
        "AUC is not required to decline monotonically across these different targets.",
        "",
        (
            "T8 arrival features are available and included: 60+24+18."
            if have_arrival
            else "T8 arrival features were skipped for this run; no arrival result is available."
        ),
        f"T5b contributes 256 fuse activations plus six Q values for {fuse_count} B8 finals;",
        "no best final is selected. No raster CNN is refit, so conclusions below",
        "apply only to these fixed GBM input sets and this continuation-policy bank.",
        "",
        "| Death within | Positive branches | 60 scalars AUC [95% CI] | "
        "60+24 AUC [95% CI] | 60+24+18 arrival AUC [95% CI] |",
        "|---:|---:|---|---|---|",
    ]
    for value in HORIZONS:
        row = h[f"D{value}"]
        cells = [auc(row["metrics"].get(name)) for name in ("60", "60+24", "60+24+18_arrival")]
        lines.append(f"| {value} | {row['positive']:,} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Fuse+Q source | D30 AUC [95% CI] | D60 | D100 | D200 | D300 |",
        "|---|---|---|---|---|---|",
    ]
    for seed in range(5):
        cells = [auc(h[f"D{value}"]["metrics"].get(f"fuse_q_B8_s{seed}")) for value in HORIZONS]
        lines.append(f"| B8_s{seed} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Horizon | Arrival − 60 AUC [95% CI] | Arrival − 60+24 AUC [95% CI] |",
        "|---:|---|---|",
    ]
    for value in HORIZONS:
        row = audit["horizon"]["results"][f"D{value}"]["arrival_comparisons"]
        cells = []
        for name in ("60", "60+24"):
            if f"arrival_minus_{name}" not in row:
                cells.append("skipped / unavailable")
                continue
            item = row[f"arrival_minus_{name}"]
            cells.append(
                f"{item['auc_delta']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"
            )
        lines.append(f"| {value} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The paired arrival differences use the same whole-world draws. Similar",
        "decay is descriptive evidence, not proof of a universal horizon ceiling,",
        "chaos, observation sufficiency or impossibility for other policies/models.",
        "",
        "## Verification and provenance",
        "",
        f"- Certificate replay: {certificate['wall_seconds']:.2f} seconds, 128/128 episodes.",
        f"- Horizon refits: {25 * len(horizon['feature_sets'])} fold fits, "
        f"{len(horizon['feature_sets'])} input sets.",
        f"- {'Late D200/D300 shard' if 'provenance' in horizon else 'Fit worker'} elapsed: "
        f"{horizon['elapsed_s']:.2f} seconds (not total parallel study wall time).",
        "- Both workers used CPU only, one numerical thread each, at most two concurrent",
        "  numerical processes. No tmux session or process not started by this task",
        "  was controlled.",
        "- Saved-artifact audit PASS: complete cohorts, source/output hashes, every label",
        "  and fold identity, all AUCs and all bootstrap intervals recomputed from saved",
        "  predictions using the established T5 routines. This is saved-artifact verification,",
        "  not a second statistical implementation. D300 T5/T8/T5b predictions match exactly.",
        "- Execution source: implementation `86458528`, merged/pushed launch HEAD `c3558167`.",
        "- After certificate completion, D200/D300 ran in the free second slot under",
        "  shard driver `a739094e`, using the unchanged fit implementation. The original",
        "  worker was intentionally stopped only after saving D30/D60/D100; its raw",
        "  receipt retains that partial status. The combined five-horizon receipt passes",
        "  source/settings/branch/copy-hash checks and the complete saved-artifact audit.",
        "- Focused tests and flake8 results are recorded in `verification.json` in the",
        "  evidence directory. Read-only independent review found no blocking issue.",
        "",
        "Raw artifacts are in",
        "`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t10-certificate-horizon-20261017/`.",
        "Committed numeric receipts: "
        "[certificate](t10_certificate_horizon_2026-10-17_evidence/certificate.json),",
        "[horizon fits](t10_certificate_horizon_2026-10-17_evidence/horizon.json),",
        "[saved-artifact audit](t10_certificate_horizon_2026-10-17_evidence/audit.json).",
        "Source/input/output SHA256s are embedded in those receipts. Reproduction commands",
        "and complete measurement definitions are in",
        "[the T10 README](../../research/t10_certificate_horizon_20261017/README.md).",
        "",
        f"Certificate episode trace SHA256: `{certificate['episodes_sha256']}`.",
        "",
        f"Canonical T5 bank SHA256: `{horizon['data_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--horizon-dir", type=Path)
    args = parser.parse_args()
    horizon_dir = args.horizon_dir or args.root / "horizon"
    certificate = json.loads((args.root / "certificate/results.json").read_text())
    horizon = json.loads((horizon_dir / "horizon_fits.json").read_text())
    audit = json.loads((args.root / "audit.json").read_text())
    assert audit["status"] == "PASS"
    verify_hashes(audit["certificate"]["source_sha256"])
    verify_hashes(audit["horizon"]["source_sha256"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for source, dest in [
        (args.root / "certificate/results.json", "certificate.json"),
        (args.root / "certificate/manifest.json", "certificate-manifest.json"),
        (horizon_dir / "horizon_fits.json", "horizon.json"),
        (args.root / "audit.json", "audit.json"),
    ]:
        shutil.copyfile(source, EVIDENCE / dest)
        assert sha256(source) == sha256(EVIDENCE / dest)
    content = render(certificate, horizon, audit)
    if DOC.exists():
        previous = DOC.read_text()
        start, end = "<!-- verdicts:start -->", "<!-- verdicts:end -->"
        if start in previous and end in previous:
            verdict = previous.split(start, 1)[1].split(end, 1)[0]
            content = content.replace("\nVERDICTS_PENDING_NUMERICAL_REVIEW\n", verdict)
    DOC.write_text(content)


if __name__ == "__main__":
    main()
