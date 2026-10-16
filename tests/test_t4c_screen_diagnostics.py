import json

import numpy as np

from research.t3_screen_20261015 import diagnostics_t4c as diag


def test_telemetry_retains_final_100k_and_never_turns_missing_into_zero(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "log.jsonl").write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"transitions": 7_800_000, "bank_agree_decisive": 0.2},
                {
                    "transitions": 8_000_128,
                    "bank_agree_decisive": 0.9,
                    "bank_agree_counter": 0.96,
                    "bank_agree_outside": 0.97,
                },
            )
        )
        + "\n"
    )
    name = f"{diag.ARM}_s10"
    records = diag.read_telemetry({name: root})[name]
    assert len(records["final_100k"]) == 1
    summary = diag._telemetry_summary({name: records})
    assert summary["five_seed_means"]["bank_agree_decisive"]["mean"] == 0.9
    assert summary["five_seed_means"]["bank_agree_counter"]["n"] == 1


def test_family_agreement_uses_surviving_set_for_decisive_and_action_for_counters(monkeypatch):
    retained = {
        "row_family": np.array([0, 0, 1]),
        "row_weight": np.array([1.0, 0.3, 1.0]),
        "surviving": np.array(
            [
                [False, True, False, False, False, False],
                [True, False, False, False, False, False],
                [False] * 6,
            ]
        ),
        "action": np.array([0, 0, 2]),
    }
    outside = {
        "row_family": np.array([2]),
        "row_weight": np.array([1.0]),
        "surviving": np.zeros((1, 6), bool),
        "action": np.array([1]),
    }
    monkeypatch.setattr(diag, "_bank", lambda path: outside if "outside" in str(path) else retained)
    calls = iter((np.array([1, 0, 2]), np.array([1])))
    monkeypatch.setattr(diag, "_actions", lambda checkpoint, arrays: next(calls))
    value = diag._family_agreement(__import__("pathlib").Path("final.pth"))
    assert value["retained_decisive_all"]["agreement"] == 1.0
    assert value["retained_decisive_gap_lt100"]["agreement"] == 1.0
    assert value["retained_decisive_gap_100_300"]["agreement"] == 1.0
    assert value["in_window_counter"]["agreement"] == 1.0
    assert value["outside_counter"]["agreement"] == 1.0


def test_drift_context_does_not_invent_a_final_candidate_rate():
    summary = {
        "rounds": {"62": {"arms": {"B8": {"five_seed_mean": {"consequential_change_rate": 0.22}}}}}
    }
    context = diag.drift_context(summary)
    assert context["baseline"] == 0.22
    assert context["candidate"] is None
    assert context["within_ten_percentage_points"] is None
