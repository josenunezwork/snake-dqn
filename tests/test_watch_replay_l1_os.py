"""exact-stop-l1-os: registration, horizon rule, silent-open-stop identity, Bank-v2 seeds."""

from __future__ import annotations

import numpy as np
import pytest

from research.watch_replay import engine, engine_l1_os
from research.watch_replay.engine import controller_arm_spec
from research.watch_replay.engine_l1_os import (
    L1_OS_CONTROLLER,
    L1_OS_MAX_HORIZON,
    L1_OS_MAX_SESSION_FRAMES,
    L1_OS_SCALE_MIN_LENGTH,
    dense_batch_h,
    exact_horizon_for,
)


def _saved_snakes():
    from research.watch6_alllen_screen.protocol import ROOT

    for candidate in (ROOT.parent / "snake-dqn/saved_snakes", ROOT / "saved_snakes"):
        if (candidate / "b8_s3_t7_20261016.pth").is_file() and (
            candidate / "frp3_m3_s12_u60000_20261005.pth"
        ).is_file():
            return candidate
    pytest.skip("product checkpoints unavailable in test environment")


def test_l1_os_is_registered_everywhere_with_the_l1_clock():
    from research.watch6_alllen_screen.protocol import ARM_SPECS
    from research.watch_replay import preflight, prepare, qualify
    from research.watch_replay.run import allocation
    from research.watch_replay.source import source_paths
    from web.backend.watch_replay import (
        ASSISTED_CONTROLLERS,
        CONTROLLERS,
        VALID_REPLAY_CONTROLLERS,
    )

    # l1-os drives engine.run_arm with the exact-stop-l1 clock (L>=1, interval 5).
    assert engine_l1_os.L1_CONTROLLER == "exact-stop-l1"
    assert L1_OS_CONTROLLER == "exact-stop-l1-os"
    controller, spec = controller_arm_spec(engine_l1_os.L1_CONTROLLER)
    assert spec is ARM_SPECS["exact_stop_alllen"]
    assert spec["min_length"] == 1 and spec["interval"] == 5
    assert L1_OS_CONTROLLER in ASSISTED_CONTROLLERS
    assert L1_OS_CONTROLLER in CONTROLLERS and L1_OS_CONTROLLER in VALID_REPLAY_CONTROLLERS
    assert L1_OS_CONTROLLER in qualify.SUPPORTED_CONTROLLERS
    assert qualify.EXPECTED_DEFAULT_COUNTS[L1_OS_CONTROLLER] == 0
    assert preflight.CANARY_HORIZON_BY_CONTROLLER[L1_OS_CONTROLLER] == 300
    assert L1_OS_CONTROLLER in prepare.SCREEN_MEAN_ROOTS_BY_CONTROLLER
    assert L1_OS_CONTROLLER in prepare.SCREEN_MEAN_BRANCHES_BY_CONTROLLER
    assert allocation("t34-l1-os-unit", 2, L1_OS_CONTROLLER)["controller"] == L1_OS_CONTROLLER
    assert "research/watch_replay/open_stop.py" in source_paths()
    assert "research/watch_replay/engine_l1_os.py" in source_paths()
    # engine.py itself is untouched: it still knows only the three original controllers.
    assert set(engine.CONTROLLER_TO_ALLLEN_ARM) == {
        "bare-b8",
        "exact-stop-l2-prime",
        "exact-stop-l1",
    }
    with pytest.raises(ValueError):
        controller_arm_spec(L1_OS_CONTROLLER)


def test_frozen_watch6_l1_confirm_science_identity_is_untouched():
    """engine.py/run.py are in watch6_l1_confirm's pinned science identity."""
    from research.watch6_l1_confirm import protocol

    sci = set(protocol.science_paths())
    assert "research/watch_replay/engine.py" in sci
    assert "research/watch_replay/engine_l1_os.py" not in sci
    assert "research/watch_replay/open_stop.py" not in sci
    # The pinned identity itself is asserted by tests/test_watch6_l1_confirm.py.


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1, 300),
        (300, 300),
        (799, 300),
        (800, 800),
        (801, 801),
        (1167, 1167),
        (1200, 1200),
        (1201, 1200),
        (5000, 1200),
    ],
)
def test_l1_os_exact_horizon_scaling_rule(length, expected):
    assert exact_horizon_for(L1_OS_CONTROLLER, length) == expected
    # Every other controller always keeps the served H300.
    for controller in ("exact-stop-l1", "exact-stop-l2-prime", "bare-b8"):
        assert exact_horizon_for(controller, length) == 300


def test_l1_os_horizon_bounds_and_storage_budget():
    assert L1_OS_SCALE_MIN_LENGTH == 800
    assert L1_OS_MAX_HORIZON == 1200
    # product.open_watch reserves 14,704 body cells = 14,400 frames + H300 (+4 slack).
    assert L1_OS_MAX_SESSION_FRAMES + L1_OS_MAX_HORIZON == 14400 + 300
    assert L1_OS_MAX_SESSION_FRAMES >= 6000
    legal = np.ones(6, dtype=bool)
    q = np.zeros(6, dtype=np.float32)
    for bad in (300, L1_OS_MAX_HORIZON + 1):
        with pytest.raises(ValueError, match="scaled exact horizon"):
            dense_batch_h(None, 0, q, legal, bad)
    with pytest.raises(ValueError, match="body storage"):
        engine_l1_os.run_arm(1, lambda: None, horizon=L1_OS_MAX_SESSION_FRAMES + 1)


def test_l1_os_routes_check_ticks_to_scaled_exact_and_others_to_open_stop(monkeypatch):
    """Drive the wrapper with a fake engine loop that uses engine's patched globals."""
    q = np.array([0.0, 1.0, 0.5, 0.2, 0.1, 0.3], dtype=np.float32)
    legal = np.ones(6, dtype=bool)
    lengths = [500, 500, 500, 500, 500, 799, 800, 900, 1000, 1100, 1300, 1300, 1300]
    calls: list[tuple] = []

    def h300(world, decision, q, legal):
        calls.append(("h300", decision))
        return {"chosen": 1, "branches_evaluated": 1}

    def scaled(world, decision, q, legal, horizon):
        calls.append(("scaled", decision, horizon))
        return {"chosen": 2, "branches_evaluated": 1}

    def ostop(world, decision, q, legal):
        calls.append(("open", decision))
        return {"chosen": 1 if decision != 3 else 0, "bare": 1, "branches_evaluated": 1}

    def fake_run_arm(seed, controller, check, horizon=6000, record=None):
        assert controller == "exact-stop-l1"
        trigger = engine.DenseTrigger(min_length=1, interval=5)
        batches = []
        for decision, length in enumerate(lengths):
            if trigger.fires(decision, length, True):
                batch = engine.dense_batch(None, decision, q, legal)
                batches.append({"frame": decision + 1, "bare": 1, **batch})
        return {"batches": batches, "frames": [], "decisions": []}

    monkeypatch.setattr(engine, "dense_batch", h300)
    monkeypatch.setattr(engine, "run_arm", fake_run_arm)
    monkeypatch.setattr(engine_l1_os, "dense_batch_h", scaled)
    monkeypatch.setattr(engine_l1_os, "open_stop_batch", ostop)
    out = engine_l1_os.run_arm(1, lambda: None, horizon=len(lengths))
    assert calls == [
        ("h300", 0),
        ("open", 1),
        ("open", 2),
        ("open", 3),
        ("open", 4),
        ("h300", 5),  # L=799 keeps H300
        ("open", 6),
        ("open", 7),
        ("open", 8),
        ("open", 9),
        ("scaled", 10, 1200),  # L=1300 -> capped at 1200
        ("open", 11),
        ("open", 12),
    ]
    assert [b["frame"] for b in out["batches"]] == [1, 6, 11]
    assert [b["exact_horizon"] for b in out["batches"]] == [300, 300, 1200]
    assert len(out["open_stop_batches"]) == 10
    assert sum(b["chosen"] != b["bare"] for b in out["open_stop_batches"]) == 1
    assert out["controller"] == L1_OS_CONTROLLER
    # Engine globals are restored afterwards.
    assert engine.DenseTrigger.__name__ == "DenseTrigger"
    assert engine.dense_batch is h300


def test_open_stop_port_evaluation_order_and_crn_seed():
    from research.watch_replay import open_stop

    q = np.array([0.1, 0.9, 0.5, 2.0, 1.5, -1.0], dtype=np.float32)
    # Bare first, then non-boost by Q, then boost by Q.
    assert open_stop.evaluation_order_nb([0, 1, 2, 3, 4, 5], 3, q) == [3, 1, 2, 0, 4, 5]
    assert open_stop.evaluation_order_nb([0, 2, 4], 0, q) == [0, 2, 4]
    with pytest.raises(ValueError):
        open_stop.evaluation_order_nb([0, 1], 3, q)
    a = open_stop.branch_suffix_seed("ns", 7, 11, 0)
    assert a == open_stop.branch_suffix_seed("ns", 7, 11, 0)
    assert a != open_stop.branch_suffix_seed("ns", 7, 12, 0)
    assert a != open_stop.branch_suffix_seed("other", 7, 11, 0)
    assert 0 <= a < 2**64
    assert open_stop.OPEN_STOP_BRANCH_HORIZON == 32
    assert open_stop.OPEN_STOP_SELECTION_SUFFIXES == (0,)


def test_bank_v2_namespaces_are_registered_and_disjoint():
    from research.watch_replay import seed_audit
    from research.watch_replay.run import allocation

    report = seed_audit.bank_v2_disjointness()
    assert report["status"] == "CLEAR"
    dev = report["namespaces"][seed_audit.BANK_V2_DEV_NAMESPACE]
    screen = report["namespaces"][seed_audit.BANK_V2_SCREEN_NAMESPACE]
    assert len(dev) == seed_audit.BANK_V2_DEV_COUNT and len(screen) == 48
    assert not set(dev) & set(screen)
    # Same derivation as the replay generator's frozen allocation.
    rows = allocation(seed_audit.BANK_V2_SCREEN_NAMESPACE, 48, L1_OS_CONTROLLER)["rows"]
    assert sorted(r["seed"] for r in rows) == screen
    prior = {
        ns for ns, _ in (*seed_audit.KNOWN_NAMESPACE_PREFIXES, *seed_audit.LATER_STUDY_NAMESPACES)
    }
    assert "t34-openstop-gate-20261104-v1" in prior
    assert "t34-hero-probe-20261103-v1" in prior
    assert seed_audit.BANK_V2_DEV_NAMESPACE not in prior
    assert seed_audit.BANK_V2_SCREEN_NAMESPACE not in prior
    gate = {seed_audit.seed_for("t34-openstop-gate-20261104-v1", i) for i in range(96)}
    assert not gate & (set(dev) | set(screen))


def test_bank_v2_disjointness_detects_a_reused_namespace(monkeypatch):
    from research.watch_replay import seed_audit

    monkeypatch.setattr(
        seed_audit,
        "BANK_V2_NAMESPACES",
        (("t34-openstop-gate-20261104-v1", 8), (seed_audit.BANK_V2_SCREEN_NAMESPACE, 48)),
    )
    with pytest.raises(ValueError, match="reuses a prior namespace"):
        seed_audit.bank_v2_disjointness()


def test_l1_os_with_silent_open_stop_is_identical_to_l1(monkeypatch):
    from research.t1_commitment_20261015.replay import configure_cpu
    from web.backend.watch_replay import canonical

    monkeypatch.setenv("SNAKE_CKPT_DIR", str(_saved_snakes()))
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    configure_cpu()
    seed, horizon = 42424242, 12

    def run(controller: str) -> tuple[dict, list[bytes]]:
        lines: list[bytes] = []

        def record(snapshot: dict) -> None:
            veto = snapshot["session"].get("safety_veto")
            if isinstance(veto, dict):
                veto.pop("diagnostics", None)
            lines.append(canonical(snapshot))

        out = engine_l1_os.run_controller(
            seed, controller, lambda: None, horizon=horizon, record=record
        )
        return out, lines

    calls: list[int] = []

    def silent(world, decision, q, legal):
        calls.append(decision)
        bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
        return {"chosen": bare, "bare": bare}

    l1, l1_lines = run("exact-stop-l1")
    monkeypatch.setattr(engine_l1_os, "open_stop_batch", silent)
    os_out, os_lines = run(L1_OS_CONTROLLER)

    assert "open_stop_batches" not in l1
    assert len(os_lines) == len(l1_lines) == horizon
    assert os_lines == l1_lines
    assert canonical(os_out["frames"]) == canonical(l1["frames"])
    assert canonical(os_out["decisions"]) == canonical(l1["decisions"])
    assert all(b["exact_horizon"] == 300 for b in os_out["batches"])

    # root_sha256 is excluded exactly as in test_watch_replay's l1 parity test: it is not
    # reproducible even between two in-process exact-stop-l1 runs after the first root.
    def strip(rows: list[dict], *keys: str) -> list[dict]:
        return [{k: v for k, v in row.items() if k not in keys} for row in rows]

    assert canonical(strip(os_out["batches"], "exact_horizon", "kind", "root_sha256")) == canonical(
        strip(l1["batches"], "root_sha256")
    )
    assert os_out["batches"][0]["root_sha256"] == l1["batches"][0]["root_sha256"]
    # Open-stop runs on exactly the unchecked ticks (4 of every 5 decisions).
    checked = {b["frame"] for b in l1["batches"]}
    unchecked = [d["frame"] for d in l1["decisions"] if d["frame"] not in checked]
    assert [d + 1 for d in calls] == unchecked
    assert len(os_out["open_stop_batches"]) == len(unchecked) == horizon - len(checked)


def test_l1_os_real_open_stop_runs_on_unchecked_ticks(monkeypatch):
    from research.t1_commitment_20261015.replay import configure_cpu

    monkeypatch.setenv("SNAKE_CKPT_DIR", str(_saved_snakes()))
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    configure_cpu()
    out = engine_l1_os.run_arm(42424242, lambda: None, horizon=6)
    checked = {b["frame"] for b in out["batches"]}
    assert checked == {1, 6}
    roots = out["open_stop_batches"]
    assert [b["frame"] for b in roots] == [2, 3, 4, 5]
    for b in roots:
        assert b["native_v8"]["kernel_rows"] > 0
        assert b["branches_evaluated"] >= 1
        assert b["chosen"] in range(6) and b["bare"] in range(6)
    for d in out["decisions"]:
        root = next((b for b in roots if b["frame"] == d["frame"]), None)
        if root is not None:
            assert d["chosen"] == root["chosen"] and d["bare"] == root["bare"]
