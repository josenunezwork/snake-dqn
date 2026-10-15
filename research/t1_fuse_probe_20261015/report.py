"""Render the T1C diagnostic report and artifact provenance from completed measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.t1_fuse_probe_20261015.collect import ARMS, M7_ROOT, ROOT, checkpoint_for

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs/research/t1_fuse_probe_2026-10-15.md"


def sha256(path: Path) -> str:
    """Hash inputs and results without loading checkpoints into torch."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(value: dict[str, Any]) -> str:
    """Format a point AUC and percentile 95% confidence interval."""
    if value.get("auc") is None:
        return "undefined"
    return f"{value['auc']:.3f} [{value['ci95_low']:.3f}, {value['ci95_high']:.3f}]"


def render(root: Path = ROOT) -> None:
    """Write the results doc and README only after all included measurements complete."""
    metrics_path = root / "runs/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    scopes = metrics["scopes"]
    pooled = scopes["pooled_B8"]["by_horizon"]
    point = pooled["100"]["models"]["lr_nested"]["auc"]
    if point >= 0.75:
        verdict = "value learning is the limit"
    elif point < 0.65:
        verdict = "perception is the limit"
    else:
        verdict = "a calibrated death head is worth adding"
    arms = [arm for arm in ARMS if arm in scopes]
    metas = {arm: json.loads((root / "runs" / f"{arm}.json").read_text()) for arm in arms}
    capture_seconds = sum(meta["total_seconds"] for meta in metas.values())
    analysis_seconds = metrics["wall_seconds"]
    lines = [
        "# T1 Part C: B8 fuse representation probe (2026-10-15)",
        "",
        "**Development diagnostic, not gate evidence.** No training or promotion decision.",
        "",
        f"R2/R2' verdict: **{verdict}**, using the pooled-five-B8 nested LR H100 point AUC ",
        f"**{auc(pooled['100']['models']['lr_nested'])}**. The rule uses the point estimate ",
        "(>=0.75 value learning, <0.65 perception, otherwise calibrated death head). ",
        "This is the panel's diagnostic routing rule; predictive association does not establish ",
        "a causal learning bottleneck or demonstrate that a learner intervention will work.",
        "",
        "## Method",
        "",
        f"Captured {len(arms)} own-checkpoint arms on the same 32 scripted development worlds ",
        "as M7: round 52, promotion-v2-watch-rect, GridBatchSim, H5000, rowwise ego2s ",
        "forward, advisory mask, normal-only fallback when no action is safe, no veto, ",
        "numba vector61 backend and fast scripted anchors. Recorded L>=800 at frame%3==0.",
        "The hook reads fuse[2]'s **post-ReLU output**, 256 dimensions, stored as float16; ",
        "Q vectors remain float32. Each NPZ includes arm, numeric world seed, frame, length, ",
        "chosen action, all labels and eligibility masks. All inference stays on CPU.",
        "",
        "A positive label means self/enemy-body death with death_frame-frame < H; the fatal ",
        "decision counts. Wall/head-on deaths within H are excluded. A surviving row is ",
        "excluded when frame > 5000-H. These are exactly M7's label and censoring formulas.",
        "",
        "LR uses train-only standardization, L2 regularization and C in {0.01,0.1,1}. ",
        "For every outer training set, C is selected by three inner folds grouped by world, ",
        "maximizing concatenated inner out-of-fold AUC. Five outer world folds produce every ",
        "reported prediction. Numeric worlds stay together across all five B8 arms in the ",
        "pooled model. HistGradientBoostingClassifier is a fixed-parameter comparison; exact ",
        "Settings, selected C values, warnings and timings are in metrics.json.",
        "",
        "CIs are percentile 95% intervals from 2,000 bootstrap resamples of numeric worlds, ",
        "with replacement, carrying all rows and arms for each sampled world together. They ",
        "condition on the fitted out-of-fold predictions; models are not refit in each bootstrap. ",
        "Rows are decision weighted, and world clustering accounts for temporal dependence.",
        "",
        "## Capture compatibility and integrity",
        "",
        "All nine M7 raw NPZs exist. They have 60 scalars and 10 plane summaries, but lack ",
        "full local/global rasters and fuse activations. Summaries cannot reconstruct the ",
        "representation. Instead, recaptured each included arm with its own checkpoint.",
        "For every full capture, compared selected row coordinates, lengths, actions, all six Q ",
        "values, world seeds, death frames and death causes bitwise against M7, with matching ",
        "checkpoint sha256. Observer checks also verify the actions actually executed.",
    ]
    for arm, meta in metas.items():
        lines.append(
            f"- {arm}: {meta['rows']:,} rows; M7 identity "
            f"`{meta['m7_identity']['status']}`; observer checks `{meta['checks']}`."
        )
    if len(arms) < 9:
        lines.extend(
            [
                "",
                "CTRL arms omitted under the spec's slow-capture fallback; ",
                "the five B8 arms are complete. See capture.log for the recorded decision.",
            ]
        )
    lines.extend(
        [
            "",
            "## Probe AUCs",
            "",
            "Every entry is AUC [95% CI].",
            "",
            "| Scope | H | Kept rows | Death-positive rows | Nested LR | HGB |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for name in arms + ["pooled_B8"]:
        for h, result in scopes[name]["by_horizon"].items():
            model = result["models"]
            lines.append(
                f"| {name} | {h} | {result['rows']:,} | {result['positives']:,} | "
                f"{auc(model['lr_nested'])} | {auc(model['hgb_fixed'])} |"
            )
    baseline_path = M7_ROOT / "results.json"
    baselines = json.loads(baseline_path.read_text())
    lines.extend(
        [
            "",
            "## M7 existing-input baselines",
            "",
            "These historical fits pooled all nine arms, whereas this report's pooled ",
            "probe uses five B8 arms. They use the same world bank and label definitions, ",
            "but are contextual baselines, not a paired improvement estimate. M7 LR used ",
            "fixed C=0.1, rather than nested selection. Its existing-input set includes ",
            "scalars, plane summaries and derived aggregates, not full raster tensors.",
            "",
            "| H | M7 LR | M7 GBM |",
            "|---:|---|---|",
        ]
    )
    for h, result in baselines["by_H"].items():
        values = []
        for kind in ("lr", "gbm"):
            model = result["models"][kind]["existing"]
            values.append(
                auc(
                    {
                        "auc": model["auc"],
                        "ci95_low": model["ci95"][0],
                        "ci95_high": model["ci95"][1],
                    }
                )
            )
        lines.append(f"| {h} | {values[0]} | {values[1]} |")
    lines.extend(
        [
            "",
            "## Hero Q signals",
            "",
            "Raw-score orientation: larger max Q or top-two gap predicts death. ",
            "The negative-score AUC is 1 minus this AUC (CI endpoints reversed); both ",
            "orientations and their CIs are preserved in metrics.json. No sign was chosen ",
            "by maximizing held-out performance, and no probe was fitted for these scores.",
            "",
            "| Scope | H | max Q raw | Top-two gap raw |",
            "|---|---:|---|---|",
        ]
    )
    for name in arms + ["pooled_B8"]:
        for h, result in scopes[name]["by_horizon"].items():
            signals = result["q_signals"]
            lines.append(
                f"| {name} | {h} | {auc(signals['max_q']['auc_raw_score'])} | "
                f"{auc(signals['top2_gap']['auc_raw_score'])} |"
            )
    lines.extend(
        [
            "",
            "## H100 LR reliability",
            "",
            "Fixed-width probability bins, using outer held-world predictions; no ",
            "additional calibrator was fit. Empty bins show —. This describes calibration ",
            "under the captured late-game decision distribution.",
            "",
            "| Scope | Probability bin | Rows | Mean predicted | Observed death rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name in arms + ["pooled_B8"]:
        for row in scopes[name]["by_horizon"]["100"]["lr_reliability_10bin"]:
            pred = "—" if row["mean_prediction"] is None else f"{row['mean_prediction']:.4f}"
            obs = "—" if row["observed_rate"] is None else f"{row['observed_rate']:.4f}"
            lines.append(
                f"| {name} | {row['lo']:.1f}–{row['hi']:.1f} | {row['n']:,} | {pred} | {obs} |"
            )
    lines.extend(
        [
            "",
            "## Timing and limits",
            "",
            "| Arm | Evaluation seconds | Total capture seconds |",
            "|---|---:|---:|",
        ]
    )
    for arm, meta in metas.items():
        lines.append(f"| {arm} | {meta['eval_seconds']:.1f} | {meta['total_seconds']:.1f} |")
    lines.extend(
        [
            "",
            f"Serial capture total: {capture_seconds:.1f}s. Analysis wall time: ",
            f"{analysis_seconds:.1f}s. Torch and native NumPy/BLAS/OpenMP pools were ",
            "limited to two threads. Captures, full tests and fits ran as serial numerical ",
            "steps in detached tmux session `t1c-run`, with logs and exit receipts.",
            "",
            "Only 32 shared development worlds support this diagnostic. Death-positive ",
            "episodes are few, rows are correlated, CIs exclude fit/model-selection and ",
            "training-seed uncertainty, and the bank was already used for diagnosis. ",
            "This is neither a fresh-world gate nor a calibrated deployment risk model. ",
            "Each arm has its own learned latent coordinates: pooled feature coordinates ",
            "are not guaranteed to align across checkpoints. Read the per-arm results ",
            "alongside the pooled point estimate; the 5-arm pool is not a seed bootstrap. ",
            "AUC measures ranking rather than intervention benefit or calibration quality.",
            "",
            "## Reproduction and artifacts",
            "",
            f"Artifact root: `{root}`. Input SHA256s: `inputs.sha256`; result SHA256s: ",
            "`outputs.sha256`. See `README.md` for commands, dependencies and verification.",
        ]
    )
    DOC.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    inputs = [
        REPO / "AGENTS.md",
        REPO / "docs/research/specs/t1c_fuse_probe_spec_2026-10-15.md",
        baseline_path,
    ]
    inputs += [
        REPO / p
        for p in (
            "research/m7_feature_auc_20261014/collect.py",
            "research/m7_feature_auc_20261014/analyze.py",
            "src/model/ego2s_network.py",
            "src/simd_env/ego2s_policy.py",
            "research/b8_death_census_20261014/census.py",
        )
    ]
    for arm in arms:
        inputs += [
            checkpoint_for(arm),
            M7_ROOT / "runs" / f"{arm}.npz",
            M7_ROOT / "runs" / f"{arm}.json",
        ]
    (root / "inputs.sha256").write_text("".join(f"{sha256(p)}  {p}\n" for p in inputs))
    readme = [
        "# T1C fuse probe artifacts (2026-10-15)",
        "",
        "**Development diagnostic, not gate evidence.**",
        "",
        f"Verdict by pooled B8 nested LR H100: **{verdict}**; ",
        f"AUC [95% CI] {auc(pooled['100']['models']['lr_nested'])}.",
        "",
        f"Source worktree: `{REPO}`, branch `t1c-20261015` (local commits only).",
        f"Results document: `{DOC}`.",
        "",
        "## Contents",
        "",
        "- `runs/<arm>.npz`: float16 post-ReLU fuse features plus aligned row metadata, ",
        "  Q vectors, labels/eligibility masks and per-world death metadata.",
        "- `runs/<arm>.json`: capture checkpoint SHA, timing, episodes and M7 identity receipt.",
        "- `runs/metrics.json`: all per-arm and pooled metrics, CIs, hyperparameters, ",
        "  Q-score orientations, calibration, warnings and timings.",
        "- `runs/oof_*.npz`: held-world predictions and row alignment for audit.",
        "- `capture.log`, `capture.exit`, `verification.log`, `analysis.log`, ",
        "  `pipeline.exit`: detached-run logs and exit receipts; 0 indicates success.",
        "- `python-deps/`: isolated analysis dependencies; shared project venv is unchanged.",
        "- `inputs.sha256`, `outputs.sha256`: complete input/result hash manifests.",
        "",
        "## Reproduce",
        "",
        "Use the project venv and run commands from the source worktree. Install analysis ",
        "dependencies into an isolated directory with `--no-deps`; NumPy 1.26.4 and SciPy ",
        "1.14.1 come from the existing project venv.",
        "",
        "```sh",
        f"cd '{REPO}'",
        f"T1C_ART='{root}'",
        "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2",
        "export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2",
        './venv/bin/python -m pip install --no-deps --target "$T1C_ART/python-deps" \\',
        "  scikit-learn==1.5.2 joblib==1.4.2 threadpoolctl==3.5.0",
        "PYTHONPATH=$PWD ./venv/bin/python -m research.t1_fuse_probe_20261015.collect",
        "PYTHONPATH=$PWD ./venv/bin/python -m pytest -q \\",
        "  tests/test_t1_fuse_probe.py tests/test_td_huber_delta.py",
        "./venv/bin/python -m flake8 research/t1_fuse_probe_20261015 \\",
        "  tests/test_t1_fuse_probe.py tests/test_td_huber_delta.py",
        'PYTHONPATH="$PWD:$T1C_ART/python-deps" ./venv/bin/python -m \\',
        '  research.t1_fuse_probe_20261015.analyze --root "$T1C_ART/runs" --boot 2000',
        "PYTHONPATH=$PWD ./venv/bin/python -m research.t1_fuse_probe_20261015.report",
        "```",
        "",
        "The actual numerical steps ran serially in detached tmux session `t1c-run`. ",
        "Do not reuse or modify other sessions; this study never touched `t2-lane0` or `t2-lane1`.",
        "",
        "## M7 compatibility",
        "",
        "M7 raw rows exist for all nine arms but lack the full local/global observation ",
        "rasters and fuse activations. Recaptured observations using each arm's own checkpoint. ",
        "All included captures passed exact M7 selected-row/Q/action/death identity checks.",
        "",
        "## Input SHA256s",
        "",
        "Checkpoint and M7 raw-input hashes below; additional metadata and source inputs ",
        "are listed in `inputs.sha256`.",
        "",
        "| Arm | Final checkpoint SHA256 | M7 raw rows SHA256 |",
        "|---|---|---|",
    ]
    for arm, meta in metas.items():
        readme.append(
            f"| {arm} | `{meta['checkpoint']['sha256']}` | " f"`{meta['m7_identity']['sha256']}` |"
        )
    readme.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The pooled probe uses five B8 checkpoints on one shared development bank. ",
            "Latent coordinates need not align across checkpoints. Read individual-arm ",
            "results too. World-bootstrap intervals condition on OOF fits and do not ",
            "measure training-seed or model-selection uncertainty. Historical M7 scalar ",
            "baselines used all nine arms, so comparisons are contextual rather than paired.",
            "",
            f"Capture total {capture_seconds:.1f}s; analysis {analysis_seconds:.1f}s.",
        ]
    )
    (root / "README.md").write_text("\n".join(line.rstrip() for line in readme) + "\n")
    outputs = sorted((root / "runs").glob("*")) + [DOC]
    (root / "outputs.sha256").write_text(
        "".join(f"{sha256(p)}  {p}\n" for p in outputs if p.is_file())
    )
    print(f"wrote {DOC}; verdict: {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    render(parser.parse_args().root)
