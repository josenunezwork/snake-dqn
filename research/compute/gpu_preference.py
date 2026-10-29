"""Pure, pre-data H100/A100 packet derivation; never allocate or resize a study."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

from research.colab_cli.accounting import budget_warning

PREFERENCE = ["H100", "A100"]
RATES = {"H100": 12.0, "A100": 8.0}  # H100 provisional; A100 covers 40/80 GB.
FALLBACK_SUFFIX = "-a100-fallback"


def sibling_id(job: str) -> str:
    """Derive a distinct <=64-character allocation identity deterministically."""
    suffix = FALLBACK_SUFFIX
    if len(job + suffix) <= 64:
        return job + suffix
    digest = hashlib.sha256(job.encode()).hexdigest()[:8]
    return job[: 64 - len(suffix) - 9] + "-" + digest + suffix


def variants(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive explicit hardware siblings with informational price forecasts.

    Unit thresholds cannot refuse a variant. Recipe, identity, canary bindings
    and the whole-session timeout remain fixed before either submission.
    """
    req = manifest.get("requirements", {})
    if "prefer" not in req:
        return [copy.deepcopy(manifest)]
    if req["prefer"] != PREFERENCE or req.get("gpu") != "H100":
        raise ValueError('preference requires gpu=H100 and prefer=["H100", "A100"]')
    if req.get("allowed_gpus") != ["H100"]:
        raise ValueError("preference requires H100-only primary admission")
    budget = manifest.get("budget", {})
    numbers = {
        "estimate": req.get("est_minutes"),
        "timeout": manifest.get("timeout_minutes"),
    }
    for name, value in numbers.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"preference {name} must be finite and positive")
    if numbers["timeout"] < max(3, 2 * numbers["estimate"]):
        raise ValueError("preference timeout must cover 2x estimate")
    if budget.get("rate_units_per_hour", RATES["H100"]) != RATES["H100"]:
        budget_warning(
            "PREFERENCE_RATE_DIFFERS_FROM_FORECAST", rate=budget.get("rate_units_per_hour")
        )
    cap = budget.get("unit_cap")
    if type(cap) in (int, float) and cap < numbers["timeout"] * RATES["H100"] / 60:
        budget_warning("PREFERENCE_UNIT_THRESHOLD", unit_cap=cap)
    primary, fallback = copy.deepcopy(manifest), copy.deepcopy(manifest)
    proofs = req.get("hardware_canaries")
    if proofs is not None:
        if (
            not isinstance(proofs, dict)
            or set(proofs) != set(PREFERENCE)
            or manifest.get("canary", {}).get("mode") == "probe"
            or any(
                not isinstance(proof, dict) or set(proof) != {"path", "sha256"}
                for proof in proofs.values()
            )
        ):
            raise ValueError("hardware_canaries requires both bound main-family receipts")
        primary["canary"] = {"receipt": copy.deepcopy(proofs["H100"])}
        fallback["canary"] = {"receipt": copy.deepcopy(proofs["A100"])}
    fallback["requirements"].update(gpu="A100", allowed_gpus=["A100"])
    fallback.setdefault("budget", {})["rate_units_per_hour"] = RATES["A100"]
    fallback_cap = numbers["timeout"] * RATES["A100"] / 60
    fallback["budget"]["unit_cap"] = fallback_cap
    if "technical_envelope" in fallback:
        fallback["technical_envelope"].update(
            base_unit_cap=fallback_cap / 2, effective_unit_cap=fallback_cap
        )
    fallback["job_id"] = sibling_id(manifest["job_id"])
    fallback["output_dir"] = manifest["output_dir"].rstrip("/") + FALLBACK_SUFFIX
    return [primary, fallback]


def prepare_preference(manifest: dict[str, Any]) -> None:
    """Price a newly built generic development packet before it is sealed."""
    req, budget = manifest["requirements"], manifest.setdefault("budget", {})
    req.update(gpu="H100", allowed_gpus=["H100"], prefer=PREFERENCE.copy())
    budget["rate_units_per_hour"] = RATES["H100"]
    # New pre-data envelopes come from the timeout/rate forecast, not an old cap.
    budget["unit_cap"] = manifest["timeout_minutes"] * RATES["H100"] / 60
    # This is a new pre-data envelope, including its undoubled forecast.
    envelope = manifest.setdefault("technical_envelope", {})
    envelope.update(
        gpu_preference=PREFERENCE.copy(),
        base_unit_cap=budget["unit_cap"] / 2,
        effective_unit_cap=budget["unit_cap"],
        base_reserve_units=budget.get("reserve_units", 0) / 2,
        effective_reserve_units=budget.get("reserve_units", 0),
        base_timeout_minutes=manifest["timeout_minutes"] / 2,
        effective_timeout_minutes=manifest["timeout_minutes"],
        h100_rate_status="UNMEASURED_PROVISIONAL_CEILING",
    )
    variants(manifest)
