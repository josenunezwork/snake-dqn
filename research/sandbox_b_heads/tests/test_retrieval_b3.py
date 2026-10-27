"""Focused contracts for the B-3 frozen-feature retrieval diagnostic."""

import pytest
import torch

from research.sandbox_b_heads.core_b3 import TAU_VETO, veto_actions
from research.sandbox_b_heads.retrieval_b3 import retrieval


def _table(*, bare, died, status=None, legal=None, mask=None, ordinary=None, lives=None):
    n = len(bare)
    return {
        "bare": torch.tensor(bare),
        "died": torch.tensor(died, dtype=torch.bool),
        "status": torch.ones((n, 6), dtype=torch.bool) if status is None else torch.tensor(status),
        "legal": torch.ones((n, 6), dtype=torch.bool) if legal is None else torch.tensor(legal),
        "action_mask": torch.ones((n, 6), dtype=torch.bool) if mask is None else torch.tensor(mask),
        "ordinary": (
            torch.zeros(n, dtype=torch.bool) if ordinary is None else torch.tensor(ordinary)
        ),
        "life_id": torch.arange(n) if lives is None else torch.tensor(lives),
    }


def test_train_only_same_base_exact_legal_retrieval_and_masking():
    train_features = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    val_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    train = _table(
        bare=[0, 0, 1],
        died=[
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],  # action 1 dies
            [False, False, False, False, False, False],
        ],
        # action 2's apparently surviving label in row 1 is unknown and cannot vote.
        status=[
            [True] * 6,
            [True, True, False, True, True, True],
            [True] * 6,
        ],
    )
    val = _table(
        bare=[0, 1],
        died=[[True, False, False, False, False, False], [False] * 6],
        ordinary=[False, True],
        mask=[[True, True, False, False, False, False], [True, True, True, False, False, False]],
    )
    result = retrieval(train_features, val_features, train, val, (5,), chunk_size=1)
    probabilities = result["probabilities"][5]
    # First query matches only base-0 train rows: action 1 = one survival / two known votes.
    assert probabilities[0, 1].item() == pytest.approx(0.5)
    # Unknown-only action 2 has no known neighbors, and unavailable output actions are zeroed.
    assert probabilities[0, 2].item() == 0.0
    # Second query cannot borrow the base-0 rows despite being a cosine neighbor.
    assert probabilities[1, 0].item() == pytest.approx(1.0)


def test_validation_labels_do_not_affect_probabilities_and_diagnostics_are_strict():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    train = _table(
        bare=[0, 0, 0],
        died=[
            [True, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ],
    )
    val = _table(
        bare=[0, 0],
        died=[[True, False, False, False, False, False], [False] * 6],
        ordinary=[False, True],
        lives=[4, 5],
    )
    first = retrieval(features, features[1:], train, val, (1, 2))
    changed = dict(val)
    changed["died"] = ~val["died"]
    changed["status"] = torch.zeros_like(val["status"])
    second = retrieval(features, features[1:], train, changed, (1, 2))
    for k in (1, 2):
        assert torch.equal(first["probabilities"][k], second["probabilities"][k])
    report = first["diagnostics"][1]
    assert report["decisive_count"] == 1
    assert report["ordinary_count"] == 1
    assert report["strict_fpr0_threshold"] == pytest.approx(0.0)
    # Equality with the maximum ordinary score does not count at FPR=0.
    assert report["decisive_tpr_strict_above_max_ordinary"] == 0.0
    assert report["decisive_vs_ordinary_auroc"] == pytest.approx(0.5)
    assert "not a proven Bayes bound" in first["caveat"]


def test_rejects_bad_shapes_and_k_values():
    table = _table(bare=[0], died=[[True] * 6])
    with pytest.raises(ValueError, match="k_values"):
        retrieval(torch.zeros(1, 2), torch.zeros(1, 2), table, table, (0,))
    with pytest.raises(ValueError, match="rank-2"):
        retrieval(torch.zeros(1, 2), torch.zeros(1, 3), table, table)


@pytest.mark.parametrize("missing", ["unknown", "illegal", "unselectable", "absent-base"])
def test_zero_known_base_votes_are_unknown_and_never_veto(missing):
    train = _table(bare=[0], died=[[False] * 6])
    if missing == "absent-base":
        train["bare"][0] = 2
    else:
        key = {"unknown": "status", "illegal": "legal", "unselectable": "action_mask"}[missing]
        train[key][0, 0] = False
    val = _table(bare=[0], died=[[True, False, False, False, False, False]])
    # Also exercise a query mask that omits base: it must not erase unknownness.
    val["action_mask"][0, 0] = False
    features = torch.tensor([[1.0, 0.0]])
    result = retrieval(features, features, train, val, (1, 5))
    for p in result["probabilities"].values():
        assert p[0, 0].item() == 1.0
        assert p[0, 1].item() == (0.0 if missing == "absent-base" else 1.0)
        for tau in TAU_VETO:
            assert veto_actions(p, val, tau).tolist() == [0]


def test_known_base_deaths_and_unknown_alternatives_do_not_cause_arbitrary_flip():
    train = _table(
        bare=[0],
        died=[[True] * 6],
        status=[[True, True, False, False, False, False]],
    )
    val = _table(bare=[0], died=[[True, False, False, False, False, False]])
    features = torch.tensor([[1.0, 0.0]])
    p = retrieval(features, features, train, val, (1,))["probabilities"][1]
    assert p.tolist() == [[0.0] * 6]
    for tau in TAU_VETO:
        assert veto_actions(p, val, tau).tolist() == [0]
