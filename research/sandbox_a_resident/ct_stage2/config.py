"""Frozen downstream science and explicitly separate CPU preflight recipe."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from research.sandbox_a_resident.ct_h5000.task import CTControl


def expected_science(pin: str, *, fixture: bool = False) -> dict[str, Any]:
    """Retain H5000 collection amendment + original branch/fit/statistical recipe."""
    ctrl = replace(
        CTControl(source_commit=pin),
        max_stage_export_bytes=256 * 1024**2,
        max_total_export_bytes=512 * 1024**2,
    )
    if fixture:
        # CPU wiring proof only, never admitted for production or a registered look.
        ctrl = replace(
            ctrl,
            branch_horizon=2,
            minibatches=1,
            searched_batch_size=2,
            anchor_batch_size=4,
            bootstrap_resamples=32,
        )
    return ctrl.to_dict()
