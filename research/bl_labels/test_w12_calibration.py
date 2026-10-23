"""Light, single-process operational tests. No calibration workload is launched."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels import d256_w12, mac_driver
from research.bl_labels import w12_calibration as calibration
from research.bl_labels.exports import _verify, publish_json


def _receipt(tmp_path: Path, monkeypatch, **changes: object) -> Path:
    monkeypatch.setenv("F0_COMMIT", "a" * 40)
    monkeypatch.setattr(calibration, "owner_approval", lambda _: "b" * 64)
    samples = tmp_path / "host-samples.jsonl"
    samples.write_text('{"thermal_not_ok":0}\n')
    value = {
        "schema": calibration.SCHEMA,
        "status": "PASS",
        "platform": "darwin-arm64",
        "code_sha": "a" * 40,
        "bank_sha256": calibration.F1_BANK_SHA256,
        "duration_seconds": 1800,
        "w12_w6_wall_ratio": 1.30,
        "thermal_not_ok": 0,
        "slowdown_trips": 0,
        "owner_approval_sha256": "b" * 64,
        "runner_sha256": calibration.sha(Path(calibration.__file__)),
        "host_samples_sha256": calibration.sha(samples),
        "host_samples_bytes": samples.stat().st_size,
        **changes,
    }
    receipt = tmp_path / "receipt.json"
    publish_json(receipt, value)
    return receipt


def test_worker_wall_ratio_and_equal_complete_work() -> None:
    cells = [
        {"workers": 6, "units": 192, "seconds": 100},
        {"workers": 12, "units": 192, "seconds": 65},
    ]
    assert calibration.wall_ratio(cells) == 1.30
    cells[1]["units"] = 191
    with pytest.raises(calibration.CalibrationHold, match="equal complete"):
        calibration.wall_ratio(cells)


def test_exact_schema_triplet_accepted_by_frozen_verifier(tmp_path: Path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    mac_driver._verify_w12_calibration(receipt)
    width, flags = d256_w12.select_workers(receipt, Path("owner"))
    assert width == 12
    assert flags == ["--owner-w12-ok", "--w12-calibration-receipt", str(receipt)]


@pytest.mark.parametrize(
    "change",
    [
        {"duration_seconds": 1799},
        {"w12_w6_wall_ratio": 0},
        {"w12_w6_wall_ratio": 1.30001},
        {"thermal_not_ok": 1},
        {"slowdown_trips": 1},
        {"code_sha": "c" * 40},
        {"platform": "linux-x86_64"},
        {"schema": "other"},
    ],
)
def test_fail_closed_pass_receipts(tmp_path: Path, monkeypatch, change: dict) -> None:
    receipt = _receipt(tmp_path, monkeypatch, **change)
    with pytest.raises(ValueError):
        d256_w12.select_workers(receipt, Path("owner"))


@pytest.mark.parametrize("changes", [{"w12_w6_wall_ratio": 1.31}, {"slowdown_trips": 1}])
def test_measured_failure_selects_w6(tmp_path: Path, monkeypatch, changes: dict) -> None:
    receipt = _receipt(tmp_path, monkeypatch, status="FAIL", **changes)
    assert d256_w12.select_workers(receipt, Path("owner")) == (6, [])


def test_hold_and_tampered_telemetry_never_fallback(tmp_path: Path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, monkeypatch, status="HOLD")
    with pytest.raises(ValueError, match="HOLD"):
        d256_w12.select_workers(receipt, Path("owner"))
    (tmp_path / "host-samples.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="samples differ"):
        d256_w12.select_workers(receipt, Path("owner"))


def test_owner_approval_is_content_bound(tmp_path: Path) -> None:
    wrong = tmp_path / "owner.txt"
    wrong.write_text("approved")
    with pytest.raises(ValueError, match="standing approval"):
        calibration.owner_approval(wrong)
    wrong.write_text(
        '2026-10-23 07:12:37 PDT OWNER (verbatim): "always gonna- approved"\n'
        "Interpretation: standing approval for routine operational asks (worker counts, "
        "budget clarifications, reruns); immediately: 12 workers (W12) approved for the Mac "
        "F0 factory, subject to the registered 30-min W12 calibration receipt. Not covered: "
        "serving releases (separate explicit OK) and credential entry (gh auth login remains "
        "owner-only).\n"
    )
    assert calibration.owner_approval(wrong) == calibration.OWNER_SHA256


def test_integrity_rejects_reordered_or_divergent_units() -> None:
    row = {
        "root_id": "one",
        "digest": "a" * 64,
        "state_digest": "b" * 64,
        "label_sha256": "c" * 64,
        "digest_checks": 1,
        "bare_audit_mismatches": 0,
    }
    assert calibration.integrity([row], ["one"]) == [("a" * 64, "b" * 64, "c" * 64)]
    with pytest.raises(calibration.CalibrationHold):
        calibration.integrity([row], ["two"])
    with pytest.raises(calibration.CalibrationHold):
        calibration.integrity([{**row, "bare_audit_mismatches": 1}], ["one"])


def test_monitor_preemption_and_missing_rss_hold(tmp_path: Path, monkeypatch) -> None:
    armed = tmp_path / "armed"
    armed.touch()
    monitor = calibration.Monitor(tmp_path, armed, float("inf"))
    with pytest.raises(calibration.CalibrationHold, match="preempted"):
        monitor.poll([])
    armed.unlink()
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _: {"thermal": "nominal"})
    monkeypatch.setattr(mac_driver, "_rss_many", lambda _: {})
    with pytest.raises(calibration.CalibrationHold, match="RSS observation incomplete"):
        monitor.poll([SimpleNamespace(pid=123, is_alive=lambda: True)])


def test_sibling_copies_only_triplets_without_modifying_original(tmp_path: Path) -> None:
    pack = tmp_path / "mac-pack-02"
    pack.mkdir()
    for name in d256_w12.COPY_NAMES:
        publish_json(pack / name, {"units": []} if "ledger" in name else {"name": name})
    before = {p.name: p.read_bytes() for p in pack.iterdir()}
    sibling = d256_w12.prepare_sibling(pack)
    assert sibling.name == d256_w12.OPERATIONAL_NAME
    assert {p.name: p.read_bytes() for p in pack.iterdir()} == before
    for name in d256_w12.COPY_NAMES:
        assert _verify(sibling / name) == _verify(pack / name)
    assert d256_w12.prepare_sibling(pack) == sibling
    (sibling / "d256-state-mac-01").mkdir()
    with pytest.raises(ValueError, match="no reset"):
        d256_w12.prepare_sibling(pack)


@pytest.mark.parametrize(
    "evidence", ["d256-state-mac-01", "d256-output-mac-02", "d256-admitted.json.complete"]
)
def test_any_original_d256_evidence_forbids_relocation(tmp_path: Path, evidence: str) -> None:
    (tmp_path / evidence).touch()
    with pytest.raises(ValueError, match="no relocation"):
        d256_w12.require_no_prior_d256(tmp_path)


def test_native_check_has_no_host_lease_or_output_side_effects(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(calibration.mac_gates, "_source_pin", lambda _: "a" * 40)
    monkeypatch.setattr(calibration.f1_ruler, "bank_digest", lambda _: calibration.F1_BANK_SHA256)
    monkeypatch.setattr(calibration.f1_ruler, "load_bank", lambda _: {"roots": []})
    monkeypatch.setattr(calibration, "owner_approval", lambda _: "b" * 64)
    monkeypatch.setattr(calibration, "held", lambda *a, **k: pytest.fail("check acquired a lease"))
    b8 = tmp_path / "b8"
    b8.touch()
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "--check", "--output", str(output), "--b8", str(b8), "--frp3", str(b8)],
    )
    calibration.main()
    assert json.loads(capsys.readouterr().out)["status"] == "CHECKED"
    assert not output.exists()


def test_runner_publishes_exact_admissible_triplet_without_real_work(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("F0_COMMIT", "a" * 40)
    monkeypatch.setattr(calibration.mac_gates, "_source_pin", lambda _: "a" * 40)
    monkeypatch.setattr(calibration.f1_ruler, "bank_digest", lambda _: calibration.F1_BANK_SHA256)
    monkeypatch.setattr(calibration.f1_ruler, "load_bank", lambda _: {})
    monkeypatch.setattr(calibration, "owner_approval", lambda _: "b" * 64)
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _: {"thermal": "nominal"})

    @contextmanager
    def lease(*_args: object, **_kwargs: object):
        yield SimpleNamespace(receipt={"id": "test-only-lease"})

    monkeypatch.setattr(calibration, "held", lease)
    monkeypatch.setattr(
        calibration,
        "measure",
        lambda *_: (
            [
                {"workers": 6, "units": 192, "seconds": 1200},
                {"workers": 12, "units": 192, "seconds": 600},
            ],
            0,
        ),
    )
    b8 = tmp_path / "b8"
    b8.touch()
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv", ["calibrate", "--output", str(output), "--b8", str(b8), "--frp3", str(b8)]
    )
    calibration.main()
    receipt = output / "receipt.json"
    assert _verify(receipt)
    assert json.loads(receipt.read_text())["duration_seconds"] == 1800
    mac_driver._verify_w12_calibration(receipt)
    assert d256_w12.select_workers(receipt, Path("owner"))[0] == 12


@pytest.mark.parametrize("divergence", [False, True])
def test_measured_schedule_pairs_identical_work_and_cleans_owned_pools(
    tmp_path: Path, monkeypatch, divergence: bool
) -> None:
    clock = [0.0]
    monkeypatch.setattr(calibration.time, "monotonic", lambda: clock[0])
    pools, measured = [], []
    ids = [str(i) for i in range(64)]
    bank = {
        "source_sessions": [{"name": "one"}],
        "roots": [{"root_id": root, "source_session": "one"} for root in ids],
    }

    class Ready:
        def __init__(self):
            self.count = 0

        def get(self, **_kwargs):
            self.count += 1
            return self.count

        def close(self):
            pass

    class Pool:
        def __init__(self, width, **_kwargs):
            self.width, self._pool, self.closed = width, [], False
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

    monkeypatch.setattr(
        calibration.mp, "get_context", lambda _: SimpleNamespace(Pool=Pool, Queue=Ready)
    )

    def collect(pool, function, jobs, _monitor):
        if function is calibration.f1_ruler._prepare_source_job:
            return [{root: tmp_path / root for root in ids}]
        if len(jobs) == 192:
            measured.append((pool.width, [Path(path).name for path, _ in jobs]))
            clock[0] += 300
        return [
            {
                "root_id": Path(path).name,
                "digest": "a" * 64,
                "state_digest": "b" * 64,
                "label_sha256": ("d" if divergence and pool.width == 12 else "c") * 64,
                "digest_checks": 1,
                "bare_audit_mismatches": 0,
            }
            for path, _ in jobs
        ]

    monkeypatch.setattr(calibration, "_collect", collect)
    monitor = SimpleNamespace(deadline=7200, poll=lambda *_a, **_k: None)
    if divergence:
        with pytest.raises(calibration.CalibrationHold, match="integrity mismatch"):
            calibration.measure(bank, Path("b8"), Path("frp3"), tmp_path, monitor)
    else:
        cells, slowdown = calibration.measure(bank, Path("b8"), Path("frp3"), tmp_path, monitor)
        assert sum(c["seconds"] for c in cells) == 1800
        assert slowdown == 0
        assert [width for width, _ in measured] == [6, 12, 12, 6, 6, 12]
        assert all(roots == ids * 3 for _, roots in measured)
    assert all(pool.closed for pool in pools)


@pytest.mark.parametrize("gate_failure", [False, True])
def test_d256_exec_passes_w12_flags_and_never_bypasses_failed_gate(
    tmp_path: Path, monkeypatch, gate_failure: bool
) -> None:
    pack = tmp_path / "mac-pack-02"
    pack.mkdir()
    monkeypatch.setattr(calibration, "PACK", pack)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.touch()
    receipt = _receipt(
        tmp_path,
        monkeypatch,
        b8_sha256=calibration.sha(checkpoint),
        frp3_sha256=calibration.sha(checkpoint),
    )
    monkeypatch.setattr(mac_gates := d256_w12.mac_gates, "_source_pin", lambda _: "a" * 40)
    monkeypatch.setattr(d256_w12, "require_no_prior_d256", lambda _: None)
    calls = []

    def f1(*_args):
        calls.append("f1")
        if gate_failure:
            raise ValueError("F1 HOLD")

    monkeypatch.setattr(mac_gates, "_f1", f1)
    monkeypatch.setattr(mac_gates, "_require_f0", lambda *_: calls.append("f0"))
    monkeypatch.setattr(mac_gates, "_approval", lambda *_: calls.append("f4"))
    monkeypatch.setattr(mac_gates, "_seed_audit", lambda *_: calls.append("seeds"))
    monkeypatch.setattr(mac_gates, "admit_d256", lambda *_: calls.append("admit"))
    monkeypatch.setattr(
        d256_w12,
        "prepare_sibling",
        lambda _: calls.append("clone") or pack.parent / d256_w12.OPERATIONAL_NAME,
    )
    monkeypatch.setattr(mac_driver, "_timing_armed", lambda _: False)
    monkeypatch.setattr(mac_driver, "_guard_snapshot", lambda _: {})
    commands = []
    monkeypatch.setattr(d256_w12.subprocess, "run", lambda command, **_: commands.append(command))
    monkeypatch.setattr(d256_w12.os, "execv", lambda _, command: commands.append(command))
    monkeypatch.setattr(
        "sys.argv",
        [
            "d256",
            "--calibration-receipt",
            str(receipt),
            "--f4-approval",
            "actual-f4.json",
            "--b8",
            str(checkpoint),
            "--frp3",
            str(checkpoint),
        ],
    )
    if gate_failure:
        with pytest.raises(ValueError, match="F1 HOLD"):
            d256_w12.main()
        assert calls == ["f1"]
        assert not commands
    else:
        d256_w12.main()
        assert calls == ["f1", "f0", "f4", "seeds", "clone", "admit"]
        assert commands[0] == commands[1] + ["--check"]
        assert commands[1][commands[1].index("--workers") + 1] == "12"
        assert "--owner-w12-ok" in commands[1]
        assert commands[1][commands[1].index("--w12-calibration-receipt") + 1] == str(receipt)
