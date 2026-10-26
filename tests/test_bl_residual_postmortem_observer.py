"""Synthetic checks for the optional F4 aggregate-only diagnostics observer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import research.bl_residual.heldout as heldout
from research.bl_residual.fit import heldout_metrics
from research.bl_residual.model import FrozenB8Residual
from src.model.ego2s_network import Ego2sNet


def _model() -> FrozenB8Residual:
    torch.manual_seed(91)
    return FrozenB8Residual(Ego2sNet(7, 60))


def _cache() -> dict[str, torch.Tensor]:
    n = 9
    return {
        "features": torch.zeros(n, 256),
        "q": torch.zeros(n, 6),
        "legal": torch.ones(n, 6, dtype=torch.bool),
        "action_mask": torch.ones(n, 6, dtype=torch.bool),
        "bare": torch.zeros(n, dtype=torch.long),
        "died": torch.tensor(
            [
                [False, False, False, False, False, False],  # unchanged
                [False, True, False, False, False, False],  # confirmed
                [False, True, False, False, False, False],  # selected unknown
                [False, True, False, False, False, False],  # B8 unknown
                [False, True, False, False, False, False],  # both unknown
                [False, False, False, False, False, False],  # exact survivor
                [False, True, False, False, False, False],  # confirmed
                [False, True, False, False, False, False],  # non-natural confirmed
                [False, False, False, False, False, False],  # argmax tie
            ],
            dtype=torch.bool,
        ),
        "status": torch.tensor(
            [
                [True] * 6,
                [True] * 6,
                [True, False, True, True, True, True],
                [False, True, True, True, True, True],
                [False, False, True, True, True, True],
                [True] * 6,
                [True] * 6,
                [True] * 6,
                [True] * 6,
            ],
            dtype=torch.bool,
        ),
        "mass": torch.zeros(n, 6),
        "ordinary": torch.ones(n, dtype=torch.bool),
        "natural_stream": torch.tensor([True] * 7 + [False, True]),
        "sampling_weight": torch.ones(n),
        "life_id": torch.arange(n, dtype=torch.long),
    }


def _candidate_q() -> torch.Tensor:
    q = torch.zeros(9, 6)
    q[0, 0] = 1.0  # unchanged; the bare action stays selected
    q[1, 1] = 0.01  # lower edge of [.01, .1)
    q[2, 1] = 0.1  # lower edge of [.1, .5)
    q[3, 1] = 0.5  # lower edge of [.5, inf)
    q[4, 1] = 0.009  # [0, .01)
    q[5, 1] = 0.25
    q[5, 5] = 10.0  # advisory mask must reject the otherwise highest action
    q[6, 1] = 0.6
    q[7, 1] = 0.1
    q[8, 0] = q[8, 1] = 0.3  # torch argmax tie resolves to bare action zero
    return q


def test_observer_exposes_existing_masks_with_exact_partition_and_bin_boundaries() -> None:
    cache = _cache()
    cache["action_mask"][5, 5] = False
    observed: list[dict[str, torch.Tensor | tuple[torch.Tensor, ...]]] = []
    metrics = heldout_metrics(
        _model(),
        cache,
        natural_archive_complete=True,
        candidate_q=_candidate_q(),
        diagnostics_observer=observed.append,
    )

    assert len(observed) == 1
    masks = observed[0]
    assert set(masks) == {
        "natural",
        "changed",
        "changed_both_exact",
        "changed_b8_exact_survives_selected_exact",
        "harmful_exact",
        "unknown_selected_only",
        "unknown_b8_only",
        "unknown_both",
        "harmful_unknown",
        "harmful",
        "margin_bins",
    }
    natural = masks["natural"]
    assert isinstance(natural, torch.Tensor)
    assert int(natural.sum()) == 8
    assert int((masks["changed"] & natural).sum()) == 6
    assert int((masks["changed_both_exact"] & natural).sum()) == 3
    # This denominator uses outcome exactness (not legality), as frozen by the ruling.
    assert int((masks["changed_b8_exact_survives_selected_exact"] & natural).sum()) == 3
    assert int((masks["harmful_exact"] & natural).sum()) == 2
    assert int((masks["unknown_selected_only"] & natural).sum()) == 1
    assert int((masks["unknown_b8_only"] & natural).sum()) == 1
    assert int((masks["unknown_both"] & natural).sum()) == 1
    assert int((masks["harmful_unknown"] & natural).sum()) == 3
    assert int((masks["harmful"] & natural).sum()) == 5
    assert metrics["harmful_override_count_natural"] == 5

    margin_bins = masks["margin_bins"]
    assert isinstance(margin_bins, tuple)
    assert [int(mask.sum()) for mask in margin_bins] == [3, 1, 2, 2]
    assert int(sum(mask.sum() for mask in margin_bins)) == int(natural.sum())


def test_observer_is_optional_and_does_not_change_default_metrics_or_empty_cache_behavior() -> None:
    cache = _cache()
    candidate_q = _candidate_q()
    without_observer = heldout_metrics(
        _model(), cache, natural_archive_complete=True, candidate_q=candidate_q
    )
    observed: list[dict[str, torch.Tensor | tuple[torch.Tensor, ...]]] = []
    with_observer = heldout_metrics(
        _model(),
        cache,
        natural_archive_complete=True,
        candidate_q=candidate_q,
        diagnostics_observer=observed.append,
    )
    assert with_observer == without_observer
    assert len(observed) == 1

    empty = {key: value[:0] for key, value in cache.items()}
    empty_observed: list[dict[str, torch.Tensor | tuple[torch.Tensor, ...]]] = []
    assert (
        heldout_metrics(_model(), empty, diagnostics_observer=empty_observed.append)[
            "controller_source"
        ]
        == "training_wrapper"
    )
    assert empty_observed == []


def test_heldout_plumbing_preserves_table_order_for_parent_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrapper may add only table-order parent labels to a mask-only callback."""

    life_ids = [f"life-{index:02d}" for index in range(64)]
    parents = ["parent-z" if index % 2 == 0 else "parent-a" for index in range(64)]
    table = {
        "q": np.zeros((64, 6), dtype=np.float32),
        "local": np.zeros((64, 7, 31, 31), dtype=np.uint8),
        "global": np.zeros((64, 4, 37, 37), dtype=np.uint8),
        "scalars": np.zeros((64, 60), dtype=np.float32),
        "life_id": np.array(life_ids),
        "ordinary": np.ones(64, dtype=np.bool_),
        "parent_world": np.array(parents),
        "natural_stream": np.ones(64, dtype=np.bool_),
        "row_id": np.array([f"row-{index:02d}" for index in range(64)]),
        "sampling_prob": np.ones(64, dtype=np.float32),
    }
    allocation = {
        "catalog": [
            {"life_id": life, "parent_world": parent, "allocation_index": index}
            for index, (life, parent) in enumerate(zip(life_ids, parents))
        ],
        "doses": {"256": {"heldout_lives": life_ids}},
        "parents": dict(zip(life_ids, parents)),
    }
    receipt = {
        "schema": "bl-residual-natural-archive/v2",
        "table_sha256": "digest",
        "complete": True,
        "natural_row_ids": sorted(table["row_id"].tolist()),
        "parent_worlds": ["parent-a", "parent-z"],
        "eligible_life_ids": sorted(life_ids),
        "reserve_max_allocation_index": 63,
        "includes_non_rescue_lives": True,
    }
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(allocation))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))

    class FakeNpz:
        files = tuple(table)

        def __getitem__(self, key: str) -> np.ndarray:
            return table[key]

        def __enter__(self) -> "FakeNpz":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Candidate:
        def __call__(self, *_args: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 6)

    masks = {"natural": torch.ones(64, dtype=torch.bool)}
    observed: list[tuple[dict[str, torch.Tensor], tuple[str, ...]]] = []
    monkeypatch.setattr(
        heldout, "preflight_look", lambda *_args: {"dose": 0, "checkpoint_sha256": "c"}
    )
    monkeypatch.setattr(
        heldout.torch,
        "load",
        lambda *_args, **_kwargs: {
            "meta": {"dose": 0, "namespace": "synthetic", "recipe_sha256": "r"}
        },
    )
    monkeypatch.setattr(heldout, "digest", lambda *_args: "digest")
    monkeypatch.setattr(heldout, "B8_SHA", "digest")
    monkeypatch.setattr(heldout, "verify_export", lambda *_args: "digest")
    monkeypatch.setattr(heldout, "allocate_lives", lambda *_args: allocation)
    monkeypatch.setattr(heldout, "world_fold", lambda *_args: 0)
    monkeypatch.setattr(heldout.np, "load", lambda *_args, **_kwargs: FakeNpz())
    monkeypatch.setattr(heldout, "validate_rows", lambda *_args: None)
    monkeypatch.setattr(heldout, "load_ego2s_checkpoint", lambda *_args: Candidate())
    monkeypatch.setattr(
        heldout, "tensor_cache", lambda *_args: {"life_id": torch.arange(64, dtype=torch.long)}
    )
    monkeypatch.setattr(heldout, "FrozenB8Residual", lambda value: value)

    def fake_metrics(*_args: object, diagnostics_observer=None, **_kwargs: object) -> dict:
        assert diagnostics_observer is not None
        diagnostics_observer(masks)
        return {"life_retention": []}

    monkeypatch.setattr(heldout, "heldout_metrics", fake_metrics)
    result = heldout.evaluate(
        tmp_path / "candidate.pth",
        tmp_path / "b8.pth",
        tmp_path / "table.npz",
        allocation_path=allocation_path,
        control_path=tmp_path / "control.json",
        approval_path=tmp_path / "approval.json",
        archive_receipt=receipt_path,
        diagnostics_observer=lambda received_masks, parents: observed.append(
            (dict(received_masks), parents)
        ),
    )

    assert result["metrics"] == {"life_retention": []}
    assert observed == [(masks, tuple(parents))]
