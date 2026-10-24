"""Closed-loop identity falsifier: tapes, pre-registered comparison, local proof."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from research.gpu_sim_spike import identity_compare as compare
from research.gpu_sim_spike import identity_pack as pack
from research.gpu_sim_spike import identity_tape as tape

SLOTS = 6
B8 = Path(
    os.environ.get(
        "SNAKE_IDENTITY_B8",
        "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth",
    )
)
ROW = {
    "world_id": "synthetic/mixed/000000",
    "mix": "mixed",
    "opponents": [
        ["checkpoint", "a"],
        ["scripted", "random_safe"],
        ["checkpoint", "b"],
        ["scripted", "greedy_food"],
        ["checkpoint", "a"],
    ],
}


def _state(t: int, food: list[tuple[int, int, int]] | None = None) -> dict[str, np.ndarray]:
    snake = np.zeros((SLOTS, len(tape.SNAKE_FIELDS)), dtype=np.int64)
    snake[:, tape.SNAKE_FIELDS.index("alive")] = 1
    snake[:, tape.SNAKE_FIELDS.index("length")] = 3
    snake[:, 0] = np.arange(SLOTS) + t
    body = np.zeros((SLOTS, 32), dtype=np.uint8)
    body[:, 0] = t
    cells = food if food is not None else [(1, 1, 0), (2, 2, 0)]
    return {
        "snake": snake,
        "body": body,
        "frame": np.asarray(t + 1, dtype=np.int64),
        "food": np.asarray(cells, dtype=np.int64).reshape(-1, 3),
        "reward": np.zeros(SLOTS, dtype=np.float64),
    }


def _decision(actions: list[int] | None = None, q: list[float] | None = None) -> dict:
    out = tape.empty_decision(SLOTS)
    out["hero_q"] = np.asarray(q or [1.0, 2.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    out["hero_mask"] = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    out["actions"] = np.asarray(actions or [1, 1, 1, 1, 1, 1], dtype=np.int64)
    for slot in (1, 3, 5):
        out["opp_q"][slot] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        out["opp_mask"][slot] = [1, 1, 1, 0, 0, 0]
        out["opp_recorded"][slot] = True
    return out


def _tape(steps: int = 3, edit=None) -> dict[str, np.ndarray]:
    writer = tape.TapeWriter(SLOTS)
    for t in range(steps + 1):
        state = _state(t)
        decision = _decision() if t < steps else None
        if edit is not None:
            edit(t, state, decision)
        writer.state(state)
        if decision is not None:
            writer.decision(decision)
    return writer.arrays()


def _first(dev, cpu, horizon: int = 3) -> dict:
    return compare.compare_world(dev, cpu, ROW, horizon)


def test_identical_tapes_are_identity_through_horizon() -> None:
    result = _first(_tape(), _tape())
    assert result["status"] == "IDENTICAL"
    assert result["tier_d"]["decisions_compared"] == 3


def test_tape_encoding_round_trip_and_immutable_units(tmp_path: Path) -> None:
    arrays = _tape()
    data = tape.encode_tape(arrays, {"unit_id": "world-00-mixed", "namespace": "ns"})
    decoded, meta = tape.decode_tape(data)
    assert meta == {"namespace": "ns", "unit_id": "world-00-mixed"}
    for key, value in arrays.items():
        assert decoded[key].dtype == value.dtype
        assert decoded[key].tobytes() == value.tobytes()  # bytes: NaN padding included
    allocation = ("world-00-mixed",)
    tape.publish_unit(tmp_path, "ns", allocation, "world-00-mixed", data, {"terminal": "x"})
    record, loaded, _ = tape.load_unit(tmp_path, "ns", allocation, "world-00-mixed")
    assert record["tape"]["size"] == len(data)
    with pytest.raises(FileExistsError):
        tape.publish_unit(tmp_path, "ns", allocation, "world-00-mixed", data, {})
    with pytest.raises(ValueError):
        tape.load_unit(tmp_path, "other", allocation, "world-00-mixed")
    ledger = tape.build_ledger(tmp_path, "ns", allocation)
    tape.publish_json(tmp_path / "ledger.json", ledger)
    assert tape.load_ledger(tmp_path / "ledger.json", "ns", allocation, tmp_path) == set(allocation)
    assert tape.load_ledger(tmp_path / "ledger.json", "ns", allocation) == set(allocation)
    payload = tmp_path / tape.UNIT_DIRECTORY / "world-00-mixed.npz"
    payload.chmod(0o644)
    corrupted = bytearray(payload.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0xFF
    payload.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError):
        tape.load_unit(tmp_path, "ns", allocation, "world-00-mixed")


def test_fresh_ambient_food_tail_is_rng() -> None:
    def edit(t, state, _decision):
        if t >= 2:
            state["food"] = np.asarray([(1, 1, 0), (2, 2, 0), (9, 9, 0)], dtype=np.int64)

    def edit_cpu(t, state, _decision):
        if t >= 2:
            state["food"] = np.asarray([(1, 1, 0), (2, 2, 0), (7, 3, 0), (4, 4, 0)], np.int64)

    result = _first(_tape(edit=edit), _tape(edit=edit_cpu))
    assert result["status"] == "DIVERGED"
    divergence = result["first_divergence"]
    assert (divergence["tick"], divergence["class"]) == (2, "RNG")
    assert divergence["device_correctness_defect"] is False


def test_food_permutation_is_event_order_and_corpse_tail_is_mechanics() -> None:
    def permuted(t, state, _decision):
        if t >= 1:
            state["food"] = np.asarray([(2, 2, 0), (1, 1, 0)], dtype=np.int64)

    result = _first(_tape(edit=permuted), _tape())
    assert result["first_divergence"]["class"] == "EVENT_ORDER"

    def corpse(t, state, _decision):
        if t >= 1:
            state["food"] = np.asarray([(1, 1, 0), (2, 2, 0), (5, 5, 1)], dtype=np.int64)

    result = _first(_tape(edit=corpse), _tape())
    assert result["first_divergence"]["class"] == "MECHANICS"
    assert result["first_divergence"]["device_correctness_defect"] is True


def test_respawn_head_only_is_rng_but_length_is_mechanics() -> None:
    alive = tape.SNAKE_FIELDS.index("alive")

    def respawn(head):
        def edit(t, state, _decision):
            if t == 0:
                state["snake"][4, alive] = 0
                state["body"][4] = 0
            if t >= 1:
                state["snake"][4, 0] = head
                state["body"][4, 1] = head

        return edit

    result = _first(_tape(edit=respawn(40)), _tape(edit=respawn(41)))
    assert result["first_divergence"]["class"] == "RNG"

    def longer(t, state, _decision):
        if t >= 1:
            state["snake"][2, tape.SNAKE_FIELDS.index("length")] = 4

    result = _first(_tape(edit=longer), _tape())
    assert result["first_divergence"]["class"] == "MECHANICS"
    assert "snake.length[2]" in result["first_divergence"]["fields"]


def test_start_state_mismatch_is_invalid() -> None:
    def edit(t, state, _decision):
        if t == 0:
            state["frame"] = np.asarray(99, dtype=np.int64)

    result = _first(_tape(edit=edit), _tape())
    assert result["status"] == "INVALID"
    assert result["first_divergence"]["class"] == "START_STATE"


@pytest.mark.parametrize(
    ("dev_q", "cpu_q", "obs_differs", "expected"),
    [
        ([1.0, 2.0, 0.5, 0, 0, 0], [1.0, 2.0, 0.5, 0, 0, 0], False, "SELECTION"),
        ([1.0, 2.0, 2.0, 0, 0, 0], [1.0, 2.0, 1.9, 0, 0, 0], False, "FLOAT_TIE"),
        ([1.0, 2.0, 1.9, 0, 0, 0], [1.0, 1.95, 1.96, 0, 0, 0], False, "FLOAT_NUMERIC"),
        ([1.0, 2.0, 1.9, 0, 0, 0], [1.0, 1.95, 1.96, 0, 0, 0], True, "FEATURE"),
    ],
)
def test_hero_action_classes(dev_q, cpu_q, obs_differs, expected) -> None:
    def make(q, action, obs):
        def edit(t, _state, decision):
            if t == 1 and decision is not None:
                decision["hero_q"] = np.asarray(q, dtype=np.float32)
                decision["actions"][0] = action
                decision["obs_digest"][0, 0] = obs

        return edit

    result = _first(_tape(edit=make(dev_q, 1, 0)), _tape(edit=make(cpu_q, 2, int(obs_differs))))
    divergence = result["first_divergence"]
    assert (divergence["tick"], divergence["check"], divergence["class"]) == (
        1,
        "hero_action",
        expected,
    )
    assert result["evidence"]["device"]["hero_margin"] is not None


def test_hero_mask_then_opponent_classes_and_dead_opponents() -> None:
    def mask(t, _state, decision):
        if t == 0 and decision is not None:
            decision["hero_mask"][2] = False

    assert _first(_tape(edit=mask), _tape())["first_divergence"]["check"] == "hero_mask"

    def scripted(t, _state, decision):
        if t == 0 and decision is not None:
            decision["actions"][2] = 2  # slot 2 is a random_safe anchor

    found = _first(_tape(edit=scripted), _tape())["first_divergence"]
    assert (found["slot"], found["class"]) == (2, "SCRIPTED_ANCHOR")

    def v8(t, _state, decision):
        if t == 0 and decision is not None:
            decision["actions"][5] = 0  # network slot, identical Q/mask/state

    found = _first(_tape(edit=v8), _tape())["first_divergence"]
    assert (found["slot"], found["class"]) == (5, "V8")

    def feature(t, _state, decision):
        if t == 0 and decision is not None:
            decision["actions"][3] = 0
            decision["opp_state_digest"][3, 0] = 7

    found = _first(_tape(edit=feature), _tape())["first_divergence"]
    assert (found["slot"], found["class"]) == (3, "FEATURE")

    def dead(t, state, decision):
        state["snake"][5, tape.SNAKE_FIELDS.index("alive")] = 0
        state["snake"][5, 0:2] = -1
        state["body"][5] = 0
        if decision is not None:
            decision["actions"][5] = t % 3

    def dead_cpu(t, state, decision):
        dead(t, state, None)

    assert _first(_tape(edit=dead), _tape(edit=dead_cpu))["status"] == "IDENTICAL"


def test_replica_disagreement_is_device_nondeterminism() -> None:
    dev = _tape()
    dev["replica_decision_agree"][1] = False
    divergence = _first(dev, _tape())["first_divergence"]
    assert divergence["class"] == "DEVICE_NONDETERMINISM"
    assert divergence["device_correctness_defect"] is True


def test_short_cpu_tape_is_incomplete_and_verdict_precedence() -> None:
    assert _first(_tape(), _tape(steps=2))["status"] == "INCOMPLETE"
    allocation = ("a", "b")
    assert (
        compare.verdict({"a": {"status": "IDENTICAL"}, "b": {"status": "IDENTICAL"}}, allocation)
        == "OUTCOME_A_IDENTICAL"
    )
    assert compare.verdict({"a": {"status": "DIVERGED"}}, allocation) == "OUTCOME_B_IDENTITY_FAILS"
    assert (
        compare.verdict({"a": {"status": "DIVERGED"}, "b": {"status": "INVALID"}}, allocation)
        == "INVALID"
    )
    assert compare.verdict({"a": {"status": "IDENTICAL"}}, allocation) == "INCOMPLETE"


def test_frozen_design_and_envelope() -> None:
    main = tape.MODES["main"]
    assert (main["worlds"], main["horizon"], main["replicas"]) == (12, 1000, 12)
    allocation = tape.allocation("main")
    assert len(allocation) == 12
    assert sorted(unit.rsplit("-", 1)[1] for unit in allocation).count("frozen") == 4
    canary = set(tape.MODES[m]["namespace"] for m in tape.MODES)
    assert len(canary) == 3
    total = 0.0
    for name in ("compile", "canary", "main"):
        spec = pack.ENVELOPE[name]
        assert spec["timeout_minutes"] >= 2 * spec["est_minutes"]
        assert spec["timeout_minutes"] * spec["rate"] / 60 <= spec["unit_cap"] + 1e-9
        total += spec["unit_cap"]
    assert total <= 10.0 and abs(total - pack.ENVELOPE["total_effective_cap_units"]) < 1e-9
    assert pack.ENVELOPE["canary"]["timeout_minutes"] <= 30


def test_harness_commit_check_refuses_non_child() -> None:
    with pytest.raises(pack.PackRefused):
        pack.check_harness_commit(pack._git("rev-parse", pack.RUNTIME_PIN + "^"))


def test_manifest_shape() -> None:
    value = pack._manifest(
        commit="0" * 40,
        mode="canary",
        output=Path("/tmp/x"),
        job_id="gpu-identity-canary-x",
        metadata={"path": "/tmp/m.tar", "sha256": "1" * 64},
        ledger_path="canary-inputs/verified-ledger.json",
    )
    assert value["canary"]["family"] == value["script"]
    assert value["env"]["IDENTITY_DEVICE"] == "cuda" and value["env"]["SNAKE_COLAB_CANARY"] == "1"
    assert value["resume"]["unit_kind"] == "world-tape"
    json.dumps(value)


@pytest.mark.skipif(not B8.is_file(), reason="Mac B8 checkpoint absent")
def test_local_device_cpu_regeneration_end_to_end() -> None:
    """Device engine on CPU torch (arm R) vs the CPU authority on one dev world."""
    import time

    from research.gpu_sim_spike import identity_cpu, identity_device

    os.environ.setdefault("SNAKE_CKPT_DIR", str(B8.parent))
    row = tape.world_rows("local")[1]
    played = identity_device.play_world(
        row, B8, device="cpu", arm="R", replicas=2, horizon=4, deadline=time.time() + 600
    )
    assert played is not None
    dev, meta = played
    assert meta["start_matches_cpu_constructor"] and bool(dev["replica_state_agree"].all())
    natural, natural_meta = identity_cpu.regenerate_world(row, dev, B8, horizon=4, kind="natural")
    result = compare.compare_world(dev, natural, row, 4)
    if result["status"] == "DIVERGED":
        assert natural_meta["stopped"] == "first_divergence"
        assert result["first_divergence"]["class"] in compare.EXPLAINED
    else:
        assert result["status"] == "IDENTICAL"
    assert result["tier_d"]["observation_digest_differences"] == 0
    bridged, bridged_meta = identity_cpu.regenerate_world(row, dev, B8, horizon=4, kind="bridged")
    assert bridged_meta["first_non_rng_divergence"] is None
    assert compare.compare_world(dev, bridged, row, 4)["status"] == "IDENTICAL"
