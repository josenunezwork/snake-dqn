import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t34_pathb.retention_v2 import (
    PRECHECK_MIN_SURVIVORS,
    RegistrationError,
    _registered_seed_union,
    _validate_checkpoint_coverage,
    precheck_receipt,
    require_registration,
)


def _rows(survivors=50):
    return [
        {
            "seed": 10_000 + env,
            "bare_survives": False,
            "paced_survives": index < survivors,
            "horizon": 5000,
            "t3_namespace": "t3-dense-ceiling/v1",
            "t3_source_arm": arm,
            "t3_env": env,
            "t3_world": f"{arm}_{env}",
            "t3_receipt": {"path": "/tmp/t3", "sha256": "b" * 64},
            "t3_checkpoint": {"path": "/tmp/checkpoint", "sha256": "c" * 64},
            "t3_materialized_sha256": "a" * 64,
        }
        for index, (arm, env) in enumerate(
            (arm, env) for arm in ("b8_s0", "b8_s3") for env in range(32)
        )
    ]


def test_precheck_requires_exact_bank_and_threshold():
    receipt = precheck_receipt(_rows())
    assert receipt["survivors"] == PRECHECK_MIN_SURVIVORS
    assert receipt["proceed"] is True
    assert precheck_receipt(_rows(49))["proceed"] is False
    with pytest.raises(RegistrationError, match="exactly 64"):
        precheck_receipt(_rows()[:-1])


def test_non_smoke_registration_rechecks_raw_evidence_and_source_closure(monkeypatch, tmp_path):
    from research.t34_pathb import retention_v2, source_binding_v2

    source = "research/t34_pathb/retention_v2.py"
    digest = hashlib.sha256(open(source, "rb").read()).hexdigest()
    monkeypatch.setattr(
        "research.t34_pathb.retention_v2.source_inventory", lambda _repo: {source: digest}
    )
    source_commit = "a" * 40

    def git_output(args, **_kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return source_commit + "\n"
        if args[:2] == ["git", "show"]:
            return open(source, "rb").read()
        raise AssertionError(args)

    monkeypatch.setattr(retention_v2.subprocess, "check_output", git_output)
    monkeypatch.setattr(source_binding_v2, "verify_frozen_worktree", lambda *_args: None)
    base = {
        "schema": "t34-pathb-v2-registration/v1",
        "ratified": True,
        "source_commit": source_commit,
        "arms": ["bare", "paced", "unpaced"],
        "mac_arms": list(retention_v2.MAC_ARMS),
        "unpaced_colab": retention_v2.unpaced_colab_policy(),
        "paced_fps": 24,
        "terminal_per_mix": 48,
        "product_sessions": 32,
        "horizon": 5000,
        "workers": 1,
        "load_profile": {
            **json.loads(Path("research/t34_pathb/RETENTION_PROFILE_V2.json").read_text())[
                "load_policy"
            ],
            "max_contended_fraction": 0.02,
        },
        "max_elapsed_hours": 40.0,
        "max_cpu_hours": 60.0,
        "precheck_max_elapsed_hours": 8.0,
        "fresh_disjoint_banks": True,
        "source_hashes": {source: digest},
    }
    # Source/receipt identity gets past the committed-closure guard; the
    # independent external receipt mapping remains mandatory.
    precheck = precheck_receipt(_rows())
    certificate_path = tmp_path / "source_validated.json"
    precheck["source_validation_path"] = str(certificate_path.resolve())
    certificate_path.write_text(
        json.dumps(
            {
                "schema": "t34-pathb-v2-source-validation/v1",
                "status": "PASS",
                "source_commit": source_commit,
                "receipts": {"precheck.json": retention_v2._canonical_sha(precheck)},
            }
        )
    )
    with pytest.raises(RegistrationError, match="checkpoint bytes"):
        require_registration(base, precheck)
    bad_precheck = precheck_receipt(_rows(49))
    bad_certificate_path = tmp_path / "bad-source_validated.json"
    bad_precheck["source_validation_path"] = str(bad_certificate_path.resolve())
    bad_certificate_path.write_text(
        json.dumps(
            {
                "schema": "t34-pathb-v2-source-validation/v1",
                "status": "PASS",
                "source_commit": source_commit,
                "receipts": {"precheck.json": retention_v2._canonical_sha(bad_precheck)},
            }
        )
    )
    with pytest.raises(RegistrationError, match="50/64"):
        require_registration(base, bad_precheck)
    base["source_hashes"] = {source: "0" * 64}
    with pytest.raises(RegistrationError, match="source hash inventory"):
        require_registration(base, precheck_receipt(_rows()))


def test_registration_checkpoint_coverage_and_nested_roster_fences(monkeypatch, tmp_path):
    """Every adapter-loaded checkpoint and row identity is frozen before data."""
    from web.backend import served_checkpoint

    def pinned(name, contents):
        path = tmp_path / name
        path.write_bytes(contents)
        return path, hashlib.sha256(contents).hexdigest()

    b8, b8_sha = pinned("b8.pth", b"b8")
    frp, frp_sha = pinned("frp.pth", b"frp")
    opponent, opponent_sha = pinned("opponent.pth", b"opponent")
    monkeypatch.setattr(served_checkpoint, "B8_S3_SHA256", b8_sha)
    monkeypatch.setattr(served_checkpoint, "FRP3_S12_SHA256", frp_sha)
    registration = {
        "checkpoints": [
            {"path": str(b8), "sha256": b8_sha},
            {"path": str(frp), "sha256": frp_sha},
            {"path": str(opponent), "sha256": opponent_sha},
        ],
        "terminal_rows": [
            {
                "seed": 11,
                "mix": "scripted",
                "roster": {
                    "seed": 11,
                    "mix": "scripted",
                    "specs": [["checkpoint", str(opponent)]],
                },
            }
        ],
        "product_rows": [{"seed": 12, "mix": "mixed"}],
        "timing_row": {"seed": 13, "mix": "mixed"},
    }
    _validate_checkpoint_coverage(registration)
    assert _registered_seed_union(registration) == {11, 12, 13}

    missing = {**registration, "checkpoints": registration["checkpoints"][:-1]}
    with pytest.raises(RegistrationError, match="absent from inventory"):
        _validate_checkpoint_coverage(missing)
    inconsistent = {
        **registration,
        "terminal_rows": [
            {
                **registration["terminal_rows"][0],
                "roster": {**registration["terminal_rows"][0]["roster"], "seed": 999},
            }
        ],
    }
    with pytest.raises(RegistrationError, match="roster seed differs"):
        _validate_checkpoint_coverage(inconsistent)
    reused_timing = {**registration, "timing_row": {"seed": 12, "mix": "mixed"}}
    with pytest.raises(RegistrationError, match="timing seed must be disjoint"):
        _registered_seed_union(reused_timing)


def test_unpaced_collection_never_sleeps(monkeypatch):
    from research.t34_pathb import retention_v2

    class Session:
        hero_length, hero_alive, hero_death = 10, True, False
        decisions_delta, boost_decisions_delta = 1, 0

        def step(self):
            pass

        def controller_state(self):
            return {}

    slept = []
    monkeypatch.setattr(retention_v2.time, "sleep", slept.append)
    retention_v2._collect(
        Session(), horizon=2, fps=0, started=time.monotonic(), cpu_started=time.process_time()
    )
    assert slept == []


def test_unpaced_collection_settles_virtual_clock_only(monkeypatch):
    from research.t34_pathb import retention_v2

    class Session:
        hero_length, hero_alive, hero_death = 10, True, False
        decisions_delta, boost_decisions_delta = 1, 0
        settled = 0

        def settle_between_ticks(self, check_cap):
            check_cap()
            self.settled += 1

        def step(self):
            pass

        def controller_state(self):
            return {}

    session = Session()
    monkeypatch.setattr(
        retention_v2.time, "sleep", lambda _seconds: pytest.fail("unexpected sleep")
    )
    retention_v2._collect(
        session, horizon=2, fps=0, started=time.monotonic(), cpu_started=time.process_time()
    )
    assert session.settled == 2


def test_nonfinite_q_is_unconditional_but_worker_error_is_not():
    from research.t34_pathb import retention_v2

    class NonFiniteQ(ValueError):
        pass

    class BadSession:
        def step(self):
            raise NonFiniteQ("q")

    with pytest.raises(retention_v2.UnconditionalFault, match="nonfinite_q"):
        retention_v2._collect(
            BadSession(),
            horizon=1,
            fps=0,
            started=time.monotonic(),
            cpu_started=time.process_time(),
        )

    class OrdinarySession:
        hero_length, hero_alive, hero_death = 10, True, False
        decisions_delta, boost_decisions_delta = 1, 0

        def step(self):
            pass

        def controller_state(self):
            return {"counts": {"worker_errors": 1}}

    result = retention_v2._collect(
        OrdinarySession(),
        horizon=1,
        fps=0,
        started=time.monotonic(),
        cpu_started=time.process_time(),
    )
    assert result["unconditional_faults"] == []
    assert result["identity_status"] == "NOT_VERIFIED"


@pytest.mark.parametrize("worker_fault", [False, True])
def test_registered_timing_fault_retains_unconditional_marker(monkeypatch, tmp_path, worker_fault):
    from research.t34_pathb import retention_v2
    from research.t34_pathb.engine_v2 import NonFiniteQ

    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "TERMINAL_PER_MIX", 1)
    monkeypatch.setattr(retention_v2, "PRODUCT_SESSIONS", 1)
    monkeypatch.setattr(
        retention_v2,
        "_collect",
        lambda *_args, **_kwargs: {"identity_status": "PASS", "parity_status": "PASS"},
    )
    fault = (retention_v2.timing_v2.ProbeFault if worker_fault else NonFiniteQ)("nonfinite_q")

    def fail_probe(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(retention_v2.timing_v2, "run_probes", fail_probe)
    session = SimpleNamespace(close=lambda **_kwargs: None)
    adapter = SimpleNamespace(
        open_terminal=lambda *_args: session, open_watch=lambda *_args: session
    )
    registration = {
        "terminal_rows": [
            {"mix": mix, "seed": seed} for seed, mix in enumerate(retention_v2.MIXES)
        ],
        "product_rows": [{"mix": "mixed", "seed": 4}],
        "timing_ticks": 1,
        "timing_warmup": 0,
    }
    registration["load_profile"] = {
        **json.loads(Path("research/t34_pathb/RETENTION_PROFILE_V2.json").read_text())[
            "load_policy"
        ],
        "max_contended_fraction": 0.02,
    }
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)
    monkeypatch.setattr(
        retention_v2,
        "HostGuard",
        lambda _policy, **_kwargs: SimpleNamespace(
            records=[],
            check=lambda **_kwargs: {"contended": False},
            start=lambda: None,
            between_rows=lambda **_kwargs: None,
            summary=lambda **_kwargs: {},
            close=lambda: None,
        ),
    )
    out = tmp_path / "fresh"
    with pytest.raises(type(fault)):
        retention_v2.run_timing_gate(
            out, registration, adapter=adapter, registration_path=Path(__file__)
        )
    assert json.loads((out / "incomplete.json").read_text())["unconditional_faults"] == [
        "nonfinite_q"
    ]
    assert json.loads((out / "decision.json").read_text())["verdict"] == "KILL"
