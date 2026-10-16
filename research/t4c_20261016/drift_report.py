"""Aggregate hash-bound T4c drift captures on the exact T4a `(c)` contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.t4a_posthoc_20261016 import report, spillover
from research.t4c_20261016.drift import (
    ARMS,
    BANK,
    DEFAULT_OUT,
    POSTHOC,
    SEEDS,
    atomic_json,
    checkpoint,
    expected_row,
    raw_rates,
    sha256,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _concat(items: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([item[key] for item in items]) for key in items[0]}


def bound(path: Path, digest: str, evidence: dict[str, str]) -> None:
    """Fail on changed evidence rather than recomputing its expected identity."""
    if sha256(path) != digest:
        raise ValueError(f"diagnostic input hash mismatch: {path}")
    evidence[str(path)] = digest


def receipt(path: Path, schema: str, evidence: dict[str, str]) -> dict:
    """Verify the receipt schema and every originally bound input."""
    value = json.loads(path.read_text())
    if value.get("schema") != schema or not value.get("inputs_sha256"):
        raise ValueError(f"invalid diagnostic receipt: {path}")
    for source, digest in value["inputs_sha256"].items():
        bound(Path(source), digest, evidence)
    evidence[str(path)] = sha256(path)
    return value


def validated_capture(
    out: Path, seed: int, round_index: int, evidence: dict[str, str]
) -> dict[str, np.ndarray]:
    """Verify identity, original receipt, historical Q joins and finite row arrays."""
    path = out / "captures" / f"CTRL_s{seed}__r{round_index}__scripted.npz"
    item = receipt(path.with_suffix(".receipt.json"), "t4c-drift-capture/v1", evidence)
    if (item["seed"], item["round_index"], item["mix"], item["capture"]) != (
        seed,
        round_index,
        "scripted",
        str(path),
    ):
        raise ValueError("diagnostic capture identity mismatch")
    required = {str(expected_row(seed, round_index)), str(BANK / "manifest.json")}
    required.update(str(checkpoint(arm, seed)) for arm in ("CTRL", *ARMS))
    if set(item["inputs_sha256"]) != required:
        raise ValueError("diagnostic capture has unexpected input identities")
    historical = expected_row(seed, round_index)
    if item["historical_row"] != str(historical):
        raise ValueError("historical row identity mismatch")
    bound(historical, item["historical_row_sha256"], evidence)
    bound(path, item["capture_sha256"], evidence)
    arrays = _load(path)
    count = item["rows"]
    if any(len(values) != count for values in arrays.values()) or count <= 0:
        raise ValueError("diagnostic row count mismatch")
    if not np.isfinite(arrays["length"]).all() or not np.all(arrays["length"] >= 0):
        raise ValueError("invalid captured length")
    for key, values in arrays.items():
        if key.endswith("_q") and (values.shape != (count, 6) or not np.isfinite(values).all()):
            raise ValueError("invalid diagnostic Q array")
    if round_index == 62:
        prior_path = POSTHOC.parent / "receipts" / f"CTRL_s{seed}.json"
        prior = receipt(prior_path, "t4a-posthoc-ctrl-capture/v1", evidence)
        for arm in ARMS[1:]:
            old_path = POSTHOC / f"{arm}_s{seed}.npz"
            binding = prior["trajectory"][arm]
            if binding["path"] != str(old_path):
                raise ValueError("historical trajectory identity mismatch")
            bound(old_path, binding["sha256"], evidence)
            old = _load(old_path)
            for key in ("length", "fingerprint", "ctrl_q"):
                if not np.array_equal(old[key], arrays[key]):
                    raise ValueError(f"historical CTRL state alignment failed: {key}")
            if not np.array_equal(old["candidate_q"], arrays[f"{arm}_q"]):
                raise ValueError("historical candidate Q alignment failed")
    return arrays


def seed_rates(
    capture: Mapping[str, np.ndarray],
    arm: str,
    cuts: np.ndarray,
    weights: Mapping[str, float],
    bank_fingerprints: np.ndarray,
) -> dict:
    """Use pooled cutpoints/weights, with null instead of unsupported extrapolation."""
    labels = spillover._strata(capture["length"], capture["ctrl_q"], cuts)
    outside = ~np.isin(capture["fingerprint"], bank_fingerprints)
    strata = {}
    missing = []
    for key, weight in weights.items():
        take = outside & (labels == key)
        if not take.any():
            missing.append(key)
            continue
        strata[key] = {
            **raw_rates(capture["ctrl_q"][take], capture[f"{arm}_q"][take]),
            "weight": weight,
        }
    return {
        "raw_change_rate": (
            None
            if missing
            else sum(row["weight"] * row["raw_change_rate"] for row in strata.values())
        ),
        "consequential_change_rate": (
            None
            if missing
            else sum(row["weight"] * row["consequential_change_rate"] for row in strata.values())
        ),
        "missing_strata": missing,
        "strata": strata,
    }


def _consequential_standardized(
    summary: Mapping[str, Any], trajectory: Mapping[str, np.ndarray], arm: str
) -> dict[str, Any]:
    """Apply report.standardize's pooled-bank weights to the strict >0.3 event."""
    standard = report.standardize(summary)["overall"]
    weights = standard["stratum_weights"]
    ctrl, candidate = trajectory["ctrl_q"], trajectory[f"{arm}_q"]
    cuts = np.asarray(summary["global_decile_cutpoints"])
    labels = spillover._strata(trajectory["length"], ctrl, cuts)
    outside = ~np.isin(trajectory["fingerprint"], np.unique(summary["_bank_fingerprint"]))
    values = {}
    for stratum, weight in weights.items():
        take = outside & (labels == stratum)
        rate = raw_rates(ctrl[take], candidate[take])
        values[stratum] = rate
        values[stratum]["weight"] = weight
    return {
        "method": (
            "same T4a pooled-bank standardization weights as raw c; "
            "strict CTRL Q(best)-Q(other)>0.3"
        ),
        "rate": sum(item["weight"] * item["consequential_change_rate"] for item in values.values()),
        "strata": values,
    }


def aggregate(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    """Write a pooled r62 report; r65 may contain only the B8 drift baseline."""
    result: dict[str, Any] = {"schema": "t4c-drift-report/v2", "rounds": {}}
    evidence: dict[str, str] = {}
    bank_rows = []
    for seed in SEEDS:
        path = POSTHOC.parent / "bank" / f"TWO_SIDED_s{seed}.npz"
        item = receipt(path.with_suffix(".receipt.json"), "t4a-posthoc-bank-capture/v1", evidence)
        if item["out"] != str(path):
            raise ValueError("historical bank capture identity mismatch")
        bound(path, item["out_sha256"], evidence)
        bank_rows.append(_load(path))
    banks = _concat(bank_rows)
    for round_index, arms in ((62, ARMS), (65, ("B8",))):
        per_seed = {seed: validated_capture(out, seed, round_index, evidence) for seed in SEEDS}
        captures = _concat(list(per_seed.values()))
        arms_out = {}
        for arm in arms:
            summary = spillover.spillover_summary(
                banks,
                {
                    "length": captures["length"],
                    "fingerprint": captures["fingerprint"],
                    "ctrl_q": captures["ctrl_q"],
                    "candidate_q": captures[f"{arm}_q"],
                },
            )
            # internal-only helper makes exact bank identity available to the
            # consequential calculation but is removed before serialization.
            summary["_bank_fingerprint"] = banks["fingerprint"]
            standard = report.standardize(summary)
            if any(
                key.split(":")[0] not in {"700_999", "ge1000"}
                for key in standard["overall"]["stratum_weights"]
            ):
                raise ValueError("matched diagnostic includes lengths below 700")
            matched = standard["overall"]["sets"]["c_ctrl_trajectory_outside_bank_rows"]
            arms_out[arm] = {
                "raw": matched,
                "consequential": _consequential_standardized(summary, captures, arm),
                "unstandardized": raw_rates(captures["ctrl_q"], captures[f"{arm}_q"]),
                "per_seed": {
                    str(seed): seed_rates(
                        capture,
                        arm,
                        np.asarray(summary["global_decile_cutpoints"]),
                        standard["overall"]["stratum_weights"],
                        np.unique(banks["fingerprint"]),
                    )
                    for seed, capture in per_seed.items()
                },
                "global_decile_cutpoints": summary["global_decile_cutpoints"],
            }
            seed_rows = list(arms_out[arm]["per_seed"].values())
            arms_out[arm]["five_seed_mean"] = {
                key: (
                    None
                    if any(row[key] is None for row in seed_rows)
                    else sum(row[key] for row in seed_rows) / len(seed_rows)
                )
                for key in ("raw_change_rate", "consequential_change_rate")
            }
        result["rounds"][str(round_index)] = {"mix": "scripted", "arms": arms_out}
    old_path = POSTHOC.parent / "posthoc-summary.json"
    old = json.loads(old_path.read_text())
    for arm in ARMS[1:]:
        historical = old["arms"][arm]["standardized"]["overall"]["sets"]
        if (
            result["rounds"]["62"]["arms"][arm]["raw"]
            != historical["c_ctrl_trajectory_outside_bank_rows"]
        ):
            raise ValueError(f"historical raw readout changed: {arm}")
    evidence[str(old_path)] = sha256(old_path)
    result["evidence_sha256"] = evidence
    result["validation"] = {
        "historical_raw_exact": True,
        "capture_and_input_receipts_verified": True,
        "historical_q_joins_exact": True,
        "timing": "retrospective validation; original capture receipts remain unchanged",
        "source_limit": "Original runtime receipts did not bind capture source. Archived source "
        "is contextual evidence from after capture, not a runtime source attestation.",
        "aggregation": "Headline uses original pooled T4a c weights, not an arithmetic seed mean; "
        "per-seed rows and their separate arithmetic mean use the same pooled cuts/weights.",
    }
    result["report_source_sha256"] = {
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve())
    }
    atomic_json(out / "drift-summary-verified.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(aggregate(), sort_keys=True))
