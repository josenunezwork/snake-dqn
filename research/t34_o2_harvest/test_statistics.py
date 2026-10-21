"""Synthetic rule/integrity tests only; no gameplay or study outcomes."""

import copy

import pytest

from research.t34_o2_harvest.analyze import decision, holm
from research.t34_o2_harvest.bank import materialize, validate_bank
from research.t34_round0.common import B8_SHA


def _arm(mass: float, dead: bool = False) -> dict:
    n = 4999 if dead else 5000
    actions = [a % 3 for a in range(n - 50)] + [3] * 50
    for i in (1000, 2000, 3000, 4000):
        if i < len(actions):
            actions[i] = 0
    masses = [int(mass)] * n
    if dead:
        masses[-1] = 0
    return dict(
        mass_integral=sum(masses) / 5000,
        mass_trace=masses,
        survival_fraction=(n - int(dead)) / 5000,
        no_death=int(not dead),
        deaths=int(dead),
        decisions=n,
        boost_decisions=50,
        action_counts=[actions.count(a) for a in range(6)],
        action_trace=actions,
        unconditional_faults=[],
        final_digest="a" * 64,
        initial_digest="b" * 64,
    )


def _controller(name: str) -> dict:
    if name == "bare":
        return {"roots": []}
    sources = {s: 0 for s in ("ambient", "death_drop", "boost_pellet")}
    env = {
        "initial_stored_length": 100,
        "final_stored_length": 100,
        "sources": sources,
        "cumulative_intake": sources,
        "weighted_cumulative_intake": sources,
        "boost_cost": 0,
        "weighted_cumulative_boost_cost": 0,
    }
    event = {
        "env": 0,
        "intake": sources,
        "cumulative_intake": sources,
        "boost_cost": 0,
        "cumulative_boost_cost": 0,
    }
    ledger = {"per_env": [env], "events": [event] * 500}
    branch = {
        "action": 0,
        "row_steps": 500,
        "alive_steps": 500,
        "survives": True,
        "mass_sum": 50000,
        "delta_mass_sum": 0,
        "delta_A500": 0.0,
        "ledger": ledger,
    }
    root = {
        "horizon": 500,
        "source_preserved": True,
        "branches": {"0": branch},
        "q": [1, 0, 0, 0, 0, 0],
        "bare_action": 0,
        "advisory_legal": [True, False, False, False, False, False],
        "illegal_actions": [1, 2, 3, 4, 5],
        "ranked_actions": [0],
    }
    choice = {
        "action": 0,
        "kind": "bare",
        "rescue_action": 0,
        "bare_action": 0,
        "bare_survives": True,
        "eligible_harvest_actions": [],
        "rejected_harvest_actions": [] if name == "d" else [0],
        "evaluated_actions": [0],
    }
    return {
        "rule_id": "rescue_only" if name == "d" else "rescue_plus_harvest",
        "branch_budget": {
            "max_roots": 8,
            "max_actions": 6,
            "horizon": 500,
            "harvest_added_branches": 0,
        },
        "nearby_death_decisions": [],
        "search_rows_total": 2000,
        "roots": [
            {
                "decision": i,
                "time_root": True,
                "event_root": False,
                "root": root,
                "choice": choice,
                "scores": [
                    {"action": 0, "delta_mass_sum": 0, "death_delta_mass_sum": 0, "survives": True}
                ],
            }
            for i in (1000, 2000, 3000, 4000)
        ],
    }


@pytest.fixture(scope="module")
def fixture() -> tuple[list, list]:
    bank = materialize()["rows"]
    records = []
    for i, row in enumerate(bank):
        dead = row["mix"] == "scripted" and i < 2
        arms = {"bare": _arm(100, dead), "d": _arm(145), "do": _arm(200)}
        for name, arm in arms.items():
            arm["controller"] = _controller(name)
        records.append(
            dict(
                schema="t34-o2-harvest-world/v1",
                world=row,
                horizon=5000,
                checkpoint_sha256=B8_SHA,
                cfg={"mechanics_version": 2},
                operational={"device": "cpu", "threads": 1},
                arms=arms,
            )
        )
    return records, bank


def test_valid_go_and_two_contrasts(fixture):
    records, bank = fixture
    result = decision(records, bank)
    assert result["status"] == "GO"
    assert result["contrasts"]["do_minus_d"]["mean"] == 55
    assert result["contrasts"]["do_minus_bare"]["mean"] >= 100
    assert result["evidence_authority"] == "UNVERIFIED_NUMERICAL_SUMMARY"
    assert result["release_authorized"] is False


def test_incomplete_precedes_scientific_failure(fixture):
    records, bank = fixture
    assert decision(records[:-1], bank)["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "mutation", ["identity", "counter", "mechanics", "nan", "trace", "checkpoint"]
)
def test_integrity_failures(fixture, mutation):
    records, bank = copy.deepcopy(fixture)
    if mutation == "identity":
        records[0]["world"]["world_seed"] += 1
    elif mutation == "counter":
        records[0]["arms"]["do"]["boost_decisions"] += 1
    elif mutation == "mechanics":
        records[0]["cfg"]["mechanics_version"] = 1
    elif mutation == "nan":
        records[0]["arms"]["do"]["mass_integral"] = float("nan")
    elif mutation == "trace":
        records[0]["arms"]["do"]["action_trace"][0] = 6
    elif mutation == "checkpoint":
        records[0]["checkpoint_sha256"] = "b" * 64
    assert decision(records, fixture[1])["status"] == "INVALID"


def test_no_incremental_gain_cannot_be_go(fixture):
    records, bank = copy.deepcopy(fixture)
    for row in records:
        row["arms"]["do"]["mass_integral"] = row["arms"]["d"]["mass_integral"]
        row["arms"]["do"]["mass_trace"] = list(row["arms"]["d"]["mass_trace"])
    assert decision(records, bank)["status"] == "KILL"


def test_fault_precedes_positive_statistics(fixture):
    records, bank = copy.deepcopy(fixture)
    records[0]["arms"]["do"]["unconditional_faults"] = ["illegal action"]
    assert decision(records, bank)["status"] == "KILL"


def test_holm_requires_first_pass_before_second():
    summaries = {
        k: dict(mean=1.0, se=1.0, df=90, lower=-0.3, upper=2.3)
        for k in ("do_minus_bare", "do_minus_d")
    }
    assert not any(v["holm_pass"] for v in holm(summaries).values())


def test_frozen_bank_rejects_roster_edit():
    bank = materialize()
    bank["rows"][0]["opponents"][0][1] = "other"
    with pytest.raises(ValueError):
        validate_bank(bank)


@pytest.mark.parametrize(
    "mutation", ["ledger", "score", "budget", "missing_root", "mass_trace", "event"]
)
def test_controller_and_provenance_tamper_is_invalid(fixture, mutation):
    records, bank = copy.deepcopy(fixture)
    arm = records[0]["arms"]["do"]
    if mutation == "ledger":
        arm["controller"]["roots"][0]["root"]["branches"]["0"]["ledger"]["events"][0][
            "boost_cost"
        ] = 1
    elif mutation == "score":
        arm["controller"]["roots"][0]["scores"][0]["death_delta_mass_sum"] = 1
    elif mutation == "budget":
        arm["controller"]["branch_budget"]["harvest_added_branches"] = 1
    elif mutation == "missing_root":
        arm["controller"]["roots"].pop()
    elif mutation == "mass_trace":
        arm["mass_trace"][0] += 1
    elif mutation == "event":
        arm["controller"]["nearby_death_decisions"] = [100]
    assert decision(records, bank)["status"] == "INVALID"


def test_copied_registration_cannot_claim_second_look(tmp_path):
    from research.t34_o2_harvest.analyze import InvalidEvidence, analyze

    with pytest.raises(InvalidEvidence, match="canonical"):
        analyze(tmp_path / "copied-registration.json", [], tmp_path / "output")


def test_sliding_window_action_collapse():
    from research.t34_o2_harvest.analyze import _collapse

    records = [{"arms": {"do": {"action_trace": [0] * 5000}}}] * 20
    assert _collapse(records, "do")
    assert not _collapse(records[:19], "do")


@pytest.mark.parametrize("run_status", ["COMPLETE", "SCIENTIFIC_FAULT"])
def test_supervisor_bundle_identity_and_tamper(tmp_path, monkeypatch, run_status):
    import hashlib
    import io
    import json
    import tarfile

    from research.t34_o2_harvest.analyze import InvalidEvidence, _admit_job

    # Admission fixture semantics are covered independently below; this test
    # isolates the supervisor publication and archive hash boundary.
    monkeypatch.setattr("research.t34_o2_harvest.analyze._validate_admissions", lambda *args: None)

    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    caps = {
        "jobs": 3,
        "units_per_job": 8,
        "reserve_units_per_job": 1.6,
        "l4_rate_units_per_hour": 2,
    }
    registration = {
        "source_commit": "a" * 40,
        "bank_sha256": "b" * 64,
        "caps": caps,
        "bank": materialize(),
    }
    registration_sha = "c" * 64
    manifest = {
        "commit": registration["source_commit"],
        "inputs": [{"dest": "registration.json", "sha256": registration_sha}],
        "budget": {"unit_cap": 8, "reserve_units": 1.6, "rate_units_per_hour": 2},
        "timeout_minutes": 240,
        "requirements": {"gpu": "L4"},
        "env": {"CUDA_VISIBLE_DEVICES": ""},
    }
    manifest_sha = write(tmp_path / "manifest.json", manifest)
    world = json.dumps({"world": registration["bank"]["rows"][0]}).encode()
    run = {
        "status": run_status,
        "mode": "campaign",
        "registration_sha256": registration_sha,
        "checkpoint_sha256": B8_SHA,
        "source_commit": registration["source_commit"],
        "bank_sha256": registration["bank_sha256"],
        "parity_pass": True,
        "worlds": [{"path": "world-000.json", "sha256": hashlib.sha256(world).hexdigest()}],
    }
    fault_blob = json.dumps(_fault(registration, registration_sha)).encode()
    fault_sha = hashlib.sha256(fault_blob).hexdigest()
    if run_status == "SCIENTIFIC_FAULT":
        run["worlds"] = []
        run["scientific_fault"] = {"path": "scientific-fault.json", "sha256": fault_sha}
    exports = tmp_path / "exports"
    exports.mkdir()
    archive_path = exports / "o2.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        world_sha = hashlib.sha256(world).hexdigest()
        for name, blob in (
            ("receipt.json", json.dumps(run).encode()),
            ("world-000.json", world),
            ("world-000.json.sha256", (world_sha + "  world-000.json\n").encode()),
            ("world-000.json.complete", b"complete\n"),
            ("scientific-fault.json", fault_blob),
            ("scientific-fault.json.sha256", (fault_sha + "  scientific-fault.json\n").encode()),
            ("scientific-fault.json.complete", b"complete\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (exports / "o2.tar.gz.sha256").write_text(digest + "  o2.tar.gz\n")
    (exports / "o2.tar.gz.complete").write_text("complete\n")
    entry = {"path": "o2.tar.gz", "sha256": digest, "size": archive_path.stat().st_size}
    terminal = {"returncode": 0, "status": "completed", "generation": "test", "exports": [entry]}
    write(tmp_path / "terminal.json", terminal)
    write(
        tmp_path / "receipt.json",
        {
            "status": "PASS",
            "stop_verified": True,
            "manifest_sha256": manifest_sha,
            "commit": registration["source_commit"],
            "terminal": terminal,
            "generation": "test",
            "exports": [entry],
        },
    )
    if run_status == "SCIENTIFIC_FAULT":
        from research.t34_o2_harvest.analyze import ScientificFaultEvidence

        with pytest.raises(ScientificFaultEvidence):
            _admit_job(tmp_path, registration, registration_sha)
    else:
        assert (
            _admit_job(tmp_path, registration, registration_sha)[0]["world"]
            == registration["bank"]["rows"][0]
        )
    archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
    with pytest.raises(InvalidEvidence, match="SHA"):
        _admit_job(tmp_path, registration, registration_sha)


def test_registered_bundle_volume_limit_covers_large_legal_shard():
    from research.t34_o2_harvest.analyze import (
        MAX_BUNDLE_MEMBERS,
        MAX_DECOMPRESSED_BYTES,
        InvalidEvidence,
        _check_bundle_limits,
    )

    # O1's already published transport had 562,776,587 raw bytes; O2 has
    # additional triple-controller provenance. Metadata tests avoid allocating GiB.
    _check_bundle_limits(562_776_587, 59)
    _check_bundle_limits(MAX_DECOMPRESSED_BYTES, MAX_BUNDLE_MEMBERS)
    with pytest.raises(InvalidEvidence, match="unbounded"):
        _check_bundle_limits(MAX_DECOMPRESSED_BYTES + 1, 1)
    with pytest.raises(InvalidEvidence, match="unbounded"):
        _check_bundle_limits(1, MAX_BUNDLE_MEMBERS + 1)


def _fault(registration, registration_sha):
    return {
        "schema": "t34-o2-scientific-fault/v1",
        "phase": "study",
        "world": registration["bank"]["rows"][0],
        "arm": "do",
        "class": "nonfinite_q",
        "decision": 1000,
        "detail": "non-finite Q before branch selection",
        "registration_sha256": registration_sha,
        "source_commit": registration["source_commit"],
        "bank_sha256": registration["bank_sha256"],
        "checkpoint_sha256": B8_SHA,
    }


@pytest.mark.parametrize(
    "change",
    ["source", "registration", "checkpoint", "bank", "world", "arm", "class", "phase", "decision"],
)
def test_scientific_fault_requires_registered_identity(change):
    from research.t34_o2_harvest.analyze import InvalidEvidence, _validate_scientific_fault

    registration = {"source_commit": "a" * 40, "bank_sha256": "b" * 64, "bank": materialize()}
    sha = "c" * 64
    fault = _fault(registration, sha)
    _validate_scientific_fault(fault, registration, sha)
    key = {
        "source": "source_commit",
        "registration": "registration_sha256",
        "checkpoint": "checkpoint_sha256",
        "bank": "bank_sha256",
    }.get(change, change)
    fault[key] = {
        "world": registration["bank"]["excluded_timing_rows"][0],
        "arm": "parity",
        "class": "ValueError",
        "phase": "timing",
        "decision": True,
    }.get(change, "wrong")
    with pytest.raises(InvalidEvidence):
        _validate_scientific_fault(fault, registration, sha)


def test_early_scientific_fault_kill_and_invalid_precedence(tmp_path, monkeypatch):
    from research.t34_o2_harvest import analyze as module
    from research.t34_o2_harvest import bank
    from research.t34_o2_harvest import registration as registration_module

    registration_path = tmp_path / "REGISTRATION_O2.json"
    registration_path.write_text("{}")
    registration = {
        "source_commit": "a" * 40,
        "bank_sha256": "b" * 64,
        "bank": materialize(),
        "caps": {"jobs": 3},
    }
    monkeypatch.setattr(bank, "PACKAGE", tmp_path)
    monkeypatch.setattr(registration_module, "require_registration", lambda path: registration)
    fault = _fault(registration, module.sha256(registration_path))

    def admit(job, *args):
        if job.name == "bad":
            raise module.InvalidEvidence("foreign source")
        raise module.ScientificFaultEvidence(fault)

    monkeypatch.setattr(module, "_admit_job", admit)
    result = module.analyze(registration_path, [tmp_path / "fault"], tmp_path / "out")
    assert result["status"] == "KILL" and result["early_stop"] == "unconditional_fault"
    assert result["completed_worlds"] == 0 and "contrasts" not in result
    with pytest.raises(FileExistsError):
        module.analyze(registration_path, [tmp_path / "fault"], tmp_path / "second-output")
    (tmp_path / "LOOK_O2.json").unlink()
    result = module.analyze(
        registration_path, [tmp_path / "fault", tmp_path / "bad"], tmp_path / "invalid-output"
    )
    assert result["status"] == "INVALID"


def _admission_fixture(tmp_path, fixture):
    import hashlib
    import json

    records, _ = fixture
    bank = materialize()
    payloads = {}

    def publish(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, sort_keys=True))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payloads[name] = path
        for suffix, text in ((".sha256", digest + "  " + name + "\n"), (".complete", "complete\n")):
            side = tmp_path / (name + suffix)
            side.write_text(text)
            payloads[name + suffix] = side
        return {"path": name, "sha256": digest}

    parity = []
    timing = []
    for i in range(3):
        ledger = copy.deepcopy(
            records[0]["arms"]["d"]["controller"]["roots"][0]["root"]["branches"]["0"]["ledger"]
        )
        parity.append(
            publish(
                f"parity-{i}.json",
                {
                    "schema": "t34-o1-world/v1",
                    "row": bank["excluded_parity_rows"][i],
                    "frames": 500,
                    "horizon": 500,
                    "checkpoint_sha256": B8_SHA,
                    "cfg": {"mechanics_version": 2},
                    "complete": True,
                    "accounting": True,
                    "base_steps": 500,
                    "base_survives": True,
                    "base_death_cause": 0,
                    "base_ledger": ledger,
                    "roots": [],
                    "parity": {"pass": True, "frames": 500, "chain": "a" * 64},
                },
            )
        )
        world = copy.deepcopy(records[2])
        world["world"] = bank["excluded_timing_rows"][i]
        world["elapsed_seconds"] = 2.0
        for name, arm in world["arms"].items():
            arm["executed_cfg"] = {"mechanics_version": 2}
            arm["base_row_steps"] = arm["decisions"]
            arm["branch_row_steps"] = 0 if name == "bare" else 1000
            arm["search_elapsed_seconds"] = 0.0 if name == "bare" else 0.5
            arm["elapsed_seconds"] = 0.5 if name == "bare" else 1.0
            if name != "bare":
                controller = arm["controller"]
                controller["roots"] = controller["roots"][:2]
                controller["branch_budget"]["max_roots"] = 2
                controller["search_rows_total"] = 1000
                for event, root_decision in zip(controller["roots"], (10, 20)):
                    event["decision"] = root_decision
                    arm["action_trace"][root_decision] = 0
                arm["action_counts"] = [arm["action_trace"].count(a) for a in range(6)]
        timing.append(publish(f"timing-{i}.json", world))
    registration = {"bank": bank, "caps": {"workers_per_job": 11, "ordinary_job_seconds": 11520}}
    run = {
        "parity_worlds": parity,
        "timing_worlds": timing,
        "timing_admission": {
            "passed": True,
            "formula": "max-rates-2x-plus600/v1",
            "workers": 11,
            "study_worlds": 32,
            "forecast_seconds": 755.0,
            "cap_seconds": 11520.0,
            "pre_admission_elapsed_seconds": 10.0,
            "base_seconds_per_step": 0.0001,
            "branch_seconds_per_step": 0.0005,
            "ordinary_deadline_seconds": 11520.0,
        },
    }
    return run, payloads, registration


def test_fixture_admission_checks_payloads_and_forecast(tmp_path, fixture):
    from research.t34_o2_harvest.analyze import InvalidEvidence, _validate_admissions

    run, payloads, registration = _admission_fixture(tmp_path, fixture)
    _validate_admissions(run, payloads, registration)
    run["timing_admission"]["forecast_seconds"] += 1
    with pytest.raises(InvalidEvidence, match="forecast"):
        _validate_admissions(run, payloads, registration)


@pytest.mark.parametrize(
    "change", ["missingparity", "missingmarker", "parityhash", "missingtiming"]
)
def test_fixture_admission_is_fail_closed(tmp_path, fixture, change):
    from research.t34_o2_harvest.analyze import InvalidEvidence, _validate_admissions

    run, payloads, registration = _admission_fixture(tmp_path, fixture)
    if change == "missingparity":
        run["parity_worlds"].pop()
    elif change == "missingtiming":
        run["timing_worlds"].pop()
    elif change == "missingmarker":
        payloads.pop("parity-0.json.complete")
    elif change == "parityhash":
        run["parity_worlds"][0]["sha256"] = "b" * 64
    with pytest.raises((InvalidEvidence, KeyError)):
        _validate_admissions(run, payloads, registration)
