"""Focused contracts for the T3 fixed bank separation diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from research.t3_diagnostics_20261016.bank import (
    proportional_sample_indices,
    require_finished_status,
    summarize_bank_q,
)


def test_proportional_sample_is_deterministic_and_preserves_bank_ratio() -> None:
    surviving = np.zeros((10, 6), dtype=bool)
    action = np.zeros(10, dtype=np.int64)
    surviving[:6, 1] = True  # six decisive rows: stored B8 action 0 did not survive
    surviving[6:, 0] = True  # four counter rows: stored B8 action survived

    first, counts = proportional_sample_indices(surviving, action, 5, 20261016)
    second, _ = proportional_sample_indices(surviving, action, 5, 20261016)

    assert np.array_equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    assert counts == {
        "total": 10,
        "decisive_total": 6,
        "counter_total": 4,
        "sample_total": 5,
        "decisive_sample": 3,
        "counter_sample": 2,
    }
    assert (~surviving[first, action[first]]).sum() == 3


def test_summary_reports_raw_training_semantics_and_masked_supplement() -> None:
    q = np.array([[2.0, 1.0, 3.0, 0.0, 0.0, 0.0], [1.0, 3.0, 4.0, 0.0, 0.0, 0.0]])
    surviving = np.array(
        [[True, False, True, False, False, False], [False, True, False, False, False, False]]
    )
    mask = np.array(
        [[True, False, True, False, False, False], [False, True, False, False, False, False]]
    )
    summary = summarize_bank_q(q, surviving, mask, np.array([1, 1]))

    assert summary["overall"]["n"] == 2
    assert summary["decisive"]["n"] == 1
    assert summary["counter"]["n"] == 1
    assert summary["overall"]["raw_agreement"] == 0.5
    assert summary["overall"]["masked_agreement"] == 1.0
    assert summary["decisive"]["separation_quartiles"]["q50"] == pytest.approx(2.0)
    assert summary["counter"]["separation_quartiles"]["q50"] == pytest.approx(0.0)


def test_completion_receipt_requires_every_t3_final(tmp_path) -> None:
    status = tmp_path / "status.txt"
    complete = "\n".join(
        f"{arm}_s{seed} exit=0"
        for arm in ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
        for seed in range(10, 15)
    )
    status.write_text(complete + "\n")
    require_finished_status(status)
    status.write_text(complete.replace("LABELS_s14 exit=0", "LABELS_s14 exit=1") + "\n")
    with pytest.raises(RuntimeError, match="not all T3 finals"):
        require_finished_status(status)


def test_completion_receipt_uses_the_later_status_event(tmp_path) -> None:
    status = tmp_path / "status.txt"
    complete = "\n".join(
        f"{arm}_s{seed} exit=0"
        for arm in ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
        for seed in range(10, 15)
    )
    status.write_text(complete + "\nLABELS_s14 exit=1\n")
    with pytest.raises(RuntimeError, match="not all T3 finals"):
        require_finished_status(status)
