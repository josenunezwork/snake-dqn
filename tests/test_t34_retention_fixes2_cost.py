import json

import pytest

from research.t34_pathb import cost_smoke_v2 as cost


class _Session:
    def __init__(self):
        self.steps = 0
        self.closed = False
        self.settles = 0
        self.hero_alive = True

    def settle_between_ticks(self, check_cap):
        self.settles += 1
        check_cap()

    def step(self):
        self.steps += 1

    def close(self, wait=True):
        assert wait is True
        self.closed = True


def _registration():
    return {
        "terminal_rows": [{"seed": 1}],
        "product_rows": [{"seed": 2}],
        "timing_row": {"seed": 3},
    }


def test_development_rows_are_one_per_mix_and_disjoint_from_registration():
    rows = cost.development_rows(_registration())
    assert [row["mix"] for row in rows] == list(cost.MIXES)
    assert {row["seed"] for row in rows}.isdisjoint({1, 2, 3})
    assert all(row["kind"] == "cost_only_development_terminal" for row in rows)


def test_run_is_create_only_cost_only_and_uses_unpaced_settle(monkeypatch, tmp_path):
    rows = [
        {"kind": "cost_only_development_terminal", "mix": mix, "seed": index}
        for index, mix in enumerate(cost.MIXES, start=10)
    ]
    monkeypatch.setattr(cost, "materialize_rows", lambda _rows: rows)
    sessions = []

    def open_session(row, fps, arm):
        assert fps == 0 and arm == "unpaced"
        session = _Session()
        sessions.append(session)
        return session

    receipt = cost.run(
        registration=_registration(),
        out=tmp_path / "cost-01",
        opener=open_session,
        preflight=lambda: {"power": "AC", "lid": "open", "thermal": "clear"},
        max_frames_per_world=cost.HORIZON,
        progress=lambda _message: None,
    )
    saved = json.loads((tmp_path / "cost-01" / "cost-smoke.json").read_text())
    assert receipt == saved
    assert receipt["label"] == "CONTENDED cost-only excluded from all efficacy"
    assert receipt["no_outcome_metrics_inspected_or_stored"] is True
    assert all(
        set(record)
        == {
            "operational_identity",
            "frames_completed",
            "world_completed_before_global_bound",
            "elapsed_seconds_including_setup_and_close",
            "process_cpu_seconds_including_setup_and_close",
            "elapsed_bound_reached",
        }
        for record in receipt["world_costs"]
    )
    assert all(
        session.steps == cost.HORIZON and session.settles == cost.HORIZON and session.closed
        for session in sessions
    )
    with pytest.raises(cost.CostSmokeError, match="existing"):
        cost.run(registration=_registration(), out=tmp_path / "cost-01", opener=open_session)


def test_run_preserves_partial_cost_when_bound_reached(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cost,
        "materialize_rows",
        lambda _rows: [{"kind": "cost_only_development_terminal", "mix": "scripted", "seed": 10}],
    )
    ticks = iter([0.0] * 10 + [1.1] * 10)

    def now():
        return next(ticks, 1.1)

    receipt = cost.run(
        registration=_registration(),
        out=tmp_path / "partial",
        max_elapsed_seconds=1,
        opener=lambda *_args: _Session(),
        preflight=lambda: {},
        now=now,
        progress=lambda _m: None,
    )
    row = receipt["world_costs"][0]
    assert row["elapsed_bound_reached"] is True
    assert row["frames_completed"] < cost.HORIZON
    assert (
        receipt["projection"]["forecast_kind"]
        == "terminal_proxy_with_conservative_full_horizon_pacing"
    )


def test_corrected_projection_binds_immutable_source_receipt(tmp_path):
    receipt = {
        "world_costs": [
            {
                "frames_completed": cost.HORIZON,
                "world_completed_before_global_bound": True,
                "elapsed_seconds_including_setup_and_close": 10.0,
            }
        ]
    }
    path = tmp_path / "cost-smoke.json"
    path.write_text(json.dumps(receipt))
    correction = cost.write_corrected_projection(path)
    assert correction["source_receipt"]["sha256"] == cost._sha256(path)
    assert correction["projection"]["projected_total_hours"] is not None
    with pytest.raises(FileExistsError):
        cost.write_corrected_projection(path)
