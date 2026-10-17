"""Saved-artifact T10 audit: complete cohort, source hashes, labels, folds and AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from research.t1_commitment_20261015.replay import INPUTS
from research.t5_separability_20261016.fit import bootstrap_auc
from research.t5b_risk_and_fuse_20261016.fuse import T5_DATA, load_t5_data, saved_folds, sha256
from research.t10_certificate_horizon_20261017.certificate import ARMS, ROOT, summarize
from research.t10_certificate_horizon_20261017.horizon import HORIZONS, T5B_ROOT, death_within


def verify_hashes(hashes: dict[str, str], base: Path = Path(".")) -> int:
    """Reject changed inputs/outputs rather than trusting the completion marker."""
    for name, expected in hashes.items():
        assert sha256(base / name) == expected, f"hash mismatch: {name}"
    return len(hashes)


def audit_certificate(root: Path) -> dict:
    """Recompute summaries and verify complete, distinct 64-world policy cohorts."""
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "results.json").read_text())
    assert result["status"] == "PASS" and result["full_cohort"]
    assert sha256(root / "manifest.json") == result["manifest_sha256"]
    assert sha256(root / "episodes.jsonl") == result["episodes_sha256"]
    hashes = verify_hashes(manifest["input_source_sha256"])
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    keys = [(e["policy"], e["arm"], e["env"]) for e in episodes]
    expected = {(p, a, e) for p in ("b8", "dense") for a in ARMS for e in range(32)}
    assert len(keys) == 128 and set(keys) == expected
    census = {arm: json.loads((INPUTS / f"{arm}.json").read_text()) for arm in ARMS}
    for e in episodes:
        source = census[e["arm"]]
        assert e["world_seed"] == source["seeds"][e["env"]]
        if e["policy"] == "b8":
            baseline = next(
                row for row in source["episodes"] if row["world_seed"] == e["world_seed"]
            )
            assert e["survived"] == baseline["survived"]
            assert e["mass_integral"] == baseline["mass_integral"]
            assert e["alive_frame_fraction"] == baseline["survival_fraction"]
        trace = e["trace"]
        decisions = [r["decision"] for r in trace]
        assert decisions == sorted(set(decisions))
        for r in trace:
            assert 0 <= r["decision"] <= (4999 if e["survived"] else e["death_decision"])
            assert r["length"] >= 800 and r["race_128"] >= 0
            arrival = r["tail_arrival"]
            assert r["tail_L"] == (arrival is not None and 0 <= arrival <= r["length"])
            assert r["tail_128"] == (arrival is not None and 0 <= arrival <= 128)
    assert summarize(episodes) == result["summary"]
    assert result["summary"]["b8"]["survivors"] == 38
    assert result["summary"]["dense"]["survivors"] == 54
    deadline_sensitivity = {}
    for policy in ("b8", "dense"):
        rows = [r for e in episodes if e["policy"] == policy for r in e["trace"]]
        deadline_sensitivity[policy] = {
            "strictly_before_L": sum(
                r["tail_arrival"] is not None and r["tail_arrival"] < r["length"] for r in rows
            ),
            "exactly_at_L": sum(r["tail_arrival"] == r["length"] for r in rows),
            "eligible_decisions": len(rows),
        }
    return {
        "status": "PASS",
        "episodes": len(episodes),
        "input_source_hashes": hashes,
        "deadline_sensitivity": deadline_sensitivity,
        "eligible_decisions": sum(len(e["trace"]) for e in episodes),
        "source_sha256": {
            str(root / name): sha256(root / name)
            for name in ("manifest.json", "episodes.jsonl", "results.json")
        },
    }


def audit_horizon(root: Path) -> dict:
    """Recompute saved labels, group membership, AUC and full bootstrap intervals."""
    report = json.loads((root / "horizon_fits.json").read_text())
    assert report["status"] == "PASS" and report["bootstrap_world_draws"] == 2000
    assert report["source_sha256_before"] == report["source_sha256_after"]
    hashes = verify_hashes(report["source_sha256_before"])
    verify_hashes(report["artifact_sha256"], root)
    if "provenance" in report:
        provenance = report["provenance"]
        for name in ("prefix", "late"):
            directory = Path(provenance[name])
            assert sha256(directory / "horizon_fits.json") == provenance[f"{name}_report_sha256"]
            horizons = (30, 60, 100) if name == "prefix" else (200, 300)
            verify_hashes(
                {
                    f"D{h}.predictions.npz": report["artifact_sha256"][f"D{h}.predictions.npz"]
                    for h in horizons
                },
                directory,
            )
        boundary = provenance["prefix_boundary"]
        assert boundary["report_sha256"] == provenance["prefix_report_sha256"]
        assert boundary["status"] == "INTENTIONALLY_INTERRUPTED_AFTER_D100"
        launch = provenance["late_launch"]
        verify_hashes(
            {
                launch["driver"]: launch["driver_sha256"],
                launch["implementation"]: launch["implementation_sha256"],
            }
        )
    data, results = load_t5_data(T5_DATA), {}
    folds = saved_folds(data)
    state, action = np.nonzero(data["legal"])
    prior_labels = np.zeros(len(state), np.uint8)
    for h in HORIZONS:
        path = root / f"D{h}.predictions.npz"
        with np.load(path, allow_pickle=False) as z:
            labels = death_within(data, h)[state, action]
            for field, value in (
                ("y", labels),
                ("state", state),
                ("action", action),
                ("world", data["world"][state]),
                ("fold", folds[state]),
            ):
                np.testing.assert_array_equal(z[field], value)
            assert np.all(labels >= prior_labels)
            prior_labels = labels
            for world in np.unique(z["world"]):
                assert len(np.unique(z["fold"][z["world"] == world])) == 1
            predictions = {k[4:]: z[k].copy() for k in z.files if k.startswith("oof_")}
        assert set(predictions) == set(report["feature_sets"])
        for name, scores in predictions.items():
            assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
            assert (
                roc_auc_score(labels, scores) == report["results"][f"D{h}"]["metrics"][name]["auc"]
            )
        summary = bootstrap_auc(labels, predictions, data["world"][state], 2000)
        for key, value in summary.items():
            assert value == report["results"][f"D{h}"][key], (h, key)
        # Same draws, paired comparison of arrival against both scalar designs.
        comparisons = {}
        for name in (("60", "60+24") if "60+24+18_arrival" in predictions else ()):
            comparison = bootstrap_auc(
                labels,
                {"60": predictions[name], "60+24": predictions["60+24+18_arrival"]},
                data["world"][state],
                2000,
            )
            comparisons[f"arrival_minus_{name}"] = comparison["metrics"]["delta_60+24_minus_60"]
        results[f"D{h}"] = {
            "labels": len(labels),
            "deaths": int(labels.sum()),
            "arrival_comparisons": comparisons,
        }
        print(json.dumps({"audited_horizon": h}), flush=True)
    parity = {}
    with np.load(root / "D300.predictions.npz") as current:
        paths = [
            (
                T5_DATA.parent / "fits/death_d300.hist_gradient_boosting.predictions.npz",
                {"oof_60": "oof_60", "oof_60+24": "oof_60+24"},
            )
        ]
        if "oof_60+24+18_arrival" in current:
            paths.append(
                (
                    ROOT.parent / "t8-probes-20261017/arrival-fit/"
                    "hist_gradient_boosting.predictions.npz",
                    {"arrival": "oof_60+24+18_arrival"},
                )
            )
        for seed in range(5):
            key = f"oof_fuse_q_B8_s{seed}"
            if key in current:
                paths.append(
                    (
                        T5B_ROOT
                        / "runs"
                        / f"B8_s{seed}"
                        / f"B8_s{seed}"
                        / "fuse_q.predictions.npz",
                        {"hist_gradient_boosting": key},
                    )
                )
        for path, fields in paths:
            with np.load(path) as old:
                for prior, new in fields.items():
                    np.testing.assert_array_equal(current[new], old[prior])
                    parity[new] = {"max_abs_difference": 0.0, "source_sha256": sha256(path)}
    return {
        "status": "PASS",
        "input_source_hashes": hashes,
        "results": results,
        "D300_prediction_parity": parity,
        "source_sha256": {str(root / "horizon_fits.json"): sha256(root / "horizon_fits.json")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--horizon-dir", type=Path)
    args = parser.parse_args()
    result = {
        "certificate": audit_certificate(args.root / "certificate"),
        "horizon": audit_horizon(args.horizon_dir or args.root / "horizon"),
        "status": "PASS",
    }
    (args.root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
