"""T4c freshness-audit policy around the immutable shared inventory."""

from __future__ import annotations

from typing import Any, Mapping

from research.t4c_20261016.seed_audit import seed_bank_audit


def initial_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Require both unconsumed registered T4c banks before round 66 begins."""
    audit = seed_bank_audit(*args, **kwargs)
    if not audit["pass"]:
        raise ValueError("T4c round-66/67 pre-probe seed audit did not pass")
    return audit


def extension_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Refresh round-67 freshness after the owned round-66 rows now exist.

    The shared audit intentionally reports those real round-66 rows as used.
    That is evidence, not a failure of round 67's separately reserved bank.
    """
    audit = seed_bank_audit(*args, **kwargs)
    check: Mapping[str, Any] = audit["checks"]["67"]
    safe = (
        audit["registered_t2_audit"]["pass"]
        and audit["t3_rollout_training_namespace"]["manifest_binding"]["ok"]
        and audit["malformed_byte_coverage"]["pass"]
        and check["unique"]
        and not check["registered_collisions"]
        and not check["observed_json_or_label_collisions"]
        and not check["other_reserved_round_collisions"]
    )
    audit["extension_round"] = 67
    audit["extension_pass"] = safe
    if not safe:
        raise ValueError("T4c round-67 extension seed audit did not pass")
    return audit
