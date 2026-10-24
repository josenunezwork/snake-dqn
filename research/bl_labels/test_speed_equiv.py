"""Synthetic acceptance tests for the T-EQ canonical comparison."""

from __future__ import annotations

from research.bl_labels.speed_equiv import compare_payloads, f4_pairs


def _payload() -> dict:
    row = {
        "schema": "bl-labels-t5-row/v1",
        "world": "w",
        "life": 1,
        "decision": 4,
        "legal": [True] * 6,
        "status": ["EXACT"] * 6,
        "died": [False, True, False, True, True, True],
        "mass": [5.0, 1.0, 3.0, 0.0, 0.0, 0.0],
        "sampling_kind": "ordinary",
        "a_star": 0,
        "code_sha": "old",
        "_speed_equiv_advisory": [True] * 6,
    }
    return {
        "schema": "bl-labels-f0-life/v1",
        "allocation": {"allocation_id": "p"},
        "rows": [row],
        "timing_seconds": {"play": 9},
        "provenance": {"elapsed_seconds": 2, "platform": "darwin"},
    }


def test_allowlisted_timing_row_code_and_provenance_elapsed_pass() -> None:
    old, new = _payload(), _payload()
    new["timing_seconds"]["play"] = 99
    new["provenance"]["elapsed_seconds"] = 88
    new["rows"][0]["code_sha"] = "new"
    assert compare_payloads([old], [new])["status"] == "PASS"


def test_unlisted_difference_fails() -> None:
    old, new = _payload(), _payload()
    new["rows"][0]["q"] = [0.0] * 6
    assert compare_payloads([old], [new])["status"] == "FAIL"


def test_a_star_equal_pair_change_fails_pair_identity() -> None:
    old, new = _payload(), _payload()
    new["rows"][0]["died"][2] = True
    assert old["rows"][0]["a_star"] == new["rows"][0]["a_star"]
    assert f4_pairs([old]) != f4_pairs([new])
    assert compare_payloads([old], [new])["status"] == "FAIL"


def test_arbitrary_nested_code_sha_is_not_allowlisted() -> None:
    old, new = _payload(), _payload()
    old["metadata"] = {"code_sha": "a"}
    new["metadata"] = {"code_sha": "b"}
    assert compare_payloads([old], [new])["status"] == "FAIL"


def test_duplicate_payload_and_unlisted_integrity_fields_fail() -> None:
    import copy

    old = _payload()
    old["life"] = 1
    assert compare_payloads([old, old], [old])["status"] == "FAIL"
    for key in ("counters", "native_v8", "audit_tape", "unknown_suffix", "snapshots"):
        new = copy.deepcopy(old)
        new[key] = {"value": 1}
        assert compare_payloads([old], [new])["status"] == "FAIL"
    new = copy.deepcopy(old)
    old["metadata"] = {"timing_seconds": 1}
    new["metadata"] = {"timing_seconds": 2}
    assert compare_payloads([old], [new])["status"] == "FAIL"


def test_pair_extractor_matches_actual_f3_loss_helpers() -> None:
    import torch
    from torch.nn import functional as functional

    from research.bl_residual.fit import FitRecipe, _ordinary_growth_loss, _preference_loss

    for mask in ([True, False, True, True, False, False], [False] * 6):
        for ordinary in (True, False):
            payload = _payload()
            row = payload["rows"][0]
            row.update(
                legal=[True, True, True, False, True, True],
                status=["EXACT", "EXACT", "EXACT", "EXACT", "NOT_EVALUATED", "EXACT"],
                died=[False, True, False, False, False, True],
                mass=[8.25, 0.0, 1.0, 9.0, 7.0, 0.0],
                _speed_equiv_advisory=mask,
                sampling_kind="ordinary" if ordinary else "doom",
            )
            pairs = f4_pairs([payload])
            q = torch.tensor([[-3.0, 1.0, 9.0, -7.0, 2.0, 5.0]])
            data = {
                "legal": torch.tensor([row["legal"]]),
                "status": torch.tensor([[s == "EXACT" for s in row["status"]]]),
                "died": torch.tensor([row["died"]]),
                "mass": torch.tensor([row["mass"]]),
                "action_mask": torch.tensor([mask]),
                "ordinary": torch.tensor([ordinary]),
            }
            survival = [
                functional.softplus(q[0, p["dying"]] - q[0, p["survivor"]])
                for p in pairs["survival"]
            ]
            expected = torch.stack(survival).mean() if survival else q.sum() * 0
            assert torch.equal(expected, _preference_loss(q, data, FitRecipe()))
            growth = [
                p["mass_gap"] / 4.0 * functional.softplus(q[0, p["poor"]] - q[0, p["good"]])
                for p in pairs["growth"]
            ]
            expected_growth = torch.stack(growth).sum() / len(growth) if growth else q.sum() * 0
            assert torch.equal(expected_growth, _ordinary_growth_loss(q, data, FitRecipe()))
