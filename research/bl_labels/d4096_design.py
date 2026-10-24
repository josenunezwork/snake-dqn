"""Prospective D4096 design disclosure, separate from observed placement."""

from __future__ import annotations

from typing import Any

COLLECTION_DESIGN = (
    "nested mixed-platform (Mac + Colab L4 CPU) collection, tie-tolerant L4 admission"
)
OWNER_DECISION_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/fable-20261024/"
    "OWNER_DECISION_TIE_TOLERANT_20261024.md"
)
# Frozen from the owner's .md.sha256 sidecar, verified against the decision bytes.
OWNER_DECISION_SHA256 = "003399e7ee6835d622ecf48fc04d2e862926c61991f8426243b1bf62db300b20"


def collection_disclosure(*, actual_execution_placement: Any) -> dict[str, Any]:
    """Bind prospective policy without claiming that either platform executed."""
    return {
        "collection_design": COLLECTION_DESIGN,
        "owner_decision": {"path": OWNER_DECISION_PATH, "sha256": OWNER_DECISION_SHA256},
        "actual_execution_placement": actual_execution_placement,
    }
