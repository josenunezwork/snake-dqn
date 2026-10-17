"""Pure shape checks for T15's bounded control producer."""

from __future__ import annotations

from research.t15_mixed_roster_20261017 import controls


def test_controls_case_has_audit_raw_fields(monkeypatch):
    monkeypatch.setattr(controls, "_session", lambda: object())
    # Verify the exporter contract without a checkpoint or real serving session.
    row = controls._case(
        "shape",
        {"target": "value"},
        lambda: object(),
        check=lambda observed: "construction_error" in observed,
    )
    assert set(row) == {"name", "expected", "observed", "steps", "no_errors", "pass"}
    assert row["name"] == "shape"
    assert row["expected"] == {"target": "value"}
    assert row["steps"] == 1
    assert row["no_errors"] is False


def test_temporary_refusal_is_isolated_to_context():
    from web.backend import served_checkpoint as registry

    original = registry.REGISTRY[registry.NAME_FRP3_S12]
    with controls._temporary_opponent_refusal("missing_bytes"):
        assert registry.REGISTRY[registry.NAME_FRP3_S12] is not original
    assert registry.REGISTRY[registry.NAME_FRP3_S12] is original


def test_identity_accepts_human_play_slot_without_policy():
    class Human:
        id = 0
        safety_veto = None

    class AI:
        id = 1
        policy = object()
        safety_veto = None

    class Session:
        mode = "play"
        checkpoint_path = "checkpoint.pth"
        obs_spec = "vector61"
        served_checkpoint = {"name": "champion"}
        watch_roster = "all_b8"
        watch_roster_reason = None
        watch_opponent_checkpoint = None
        hero_id = 0
        speed_ceiling = 120.0
        last_error = None

        class game:
            snakes = [Human(), AI()]

        @staticmethod
        def safety_veto_state():
            return {"flags": {}}

    result = controls._identity(Session())
    assert result["policy_classes"] == ["NoneType", "object"]
