"""Decision rules are checked on immutable synthetic worlds, never gameplay data."""

import math
import statistics

import pytest

from research.t34_pathb import analyze_v2 as analysis
from research.t34_pathb import source_binding_v2


@pytest.fixture(autouse=True)
def fake_frozen_source_fence(monkeypatch):
    """Keep analyzer unit tests independent of the authoring worktree state."""

    class Fence:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(source_binding_v2, "FrozenWorktreeGuard", lambda *_args, **_kwargs: Fence())
    monkeypatch.setattr(source_binding_v2, "verify_frozen_worktree", lambda *_args: None)


def _arm(mass=160, *, survival=1.0, deaths=0):
    counts = [1650, 1650, 1650, 25, 15, 10]
    return {
        "mass_integral": mass,
        "survival_fraction": survival,
        "no_death": int(deaths == 0),
        "deaths": deaths,
        "decisions": 5000,
        "deaths_per_1k": deaths / 5,
        "boost_decisions": 50,
        "boost_share": 0.01,
        "action_counts": counts,
        "one_action_share": 0.33,
        "unconditional_faults": [],
        "overruns": 0,
        "median_length": 100.0,
        "p95_length": 200.0,
        "deaths_length_ge_800": 0,
        "late_deaths": 0,
        "late_large_overlap": 0,
    }


def _records():
    rows = []
    for mix_index, mix in enumerate(analysis.MIXES):
        for index in range(48):
            bare = _arm(100)
            if mix == "scripted" and index < 2:
                bare = _arm(100, survival=0.5, deaths=1)
            rows.append(
                {
                    "kind": "terminal",
                    "mix": mix,
                    "seed": mix_index * 48 + index,
                    "horizon": 5000,
                    "identity_status": "PASS",
                    "parity_status": "PASS",
                    "arms": {"bare": bare, "paced": _arm()},
                }
            )
    for index in range(32):
        rows.append(
            {
                "kind": "product",
                "mix": "mixed",
                "seed": 144 + index,
                "horizon": 5000,
                "identity_status": "PASS",
                "parity_status": "PASS",
                "arms": {
                    "bare": _arm(100, survival=0.9, deaths=5),
                    "paced": _arm(110, survival=0.95, deaths=3),
                },
            }
        )
    return rows


def _timing():
    def probe(fps):
        return {
            "update_ms": {"n": 10, "p95": 1.0, "p99": 1.0, "max": 1.0},
            "full_tick_ms": {"n": 10, "p95": 2.0, "p99": 2.0, "max": 2.0},
            "ticks": 10,
            "fps": fps,
            "achieved_fps": float(fps),
            "contended": False,
            "overruns": 0,
        }

    return {"probes": {f"{arm}-{fps}": probe(fps) for arm in ("bare", "paced") for fps in (12, 24)}}


def _host_summary(*, contended_seconds: float = 0.0, elapsed_seconds: float = 1.0) -> dict:
    phase_cap_seconds = analysis.PHASE_CAP_SECONDS
    return {
        "phase_cap_seconds": phase_cap_seconds,
        "contention_budget_seconds": phase_cap_seconds * analysis.MAX_CONTENDED_FRACTION,
        "contended_seconds": contended_seconds,
        "contended_fraction": contended_seconds / phase_cap_seconds,
        "elapsed_seconds": elapsed_seconds,
    }


def test_go_requires_fixed_n_holm_and_all_bars():
    report = analysis.decide(_records(), _timing())
    assert report["verdict"] == "GO", report
    assert report["release_authorized"] is False
    assert report["paced"]["mass"]["n"] == 144
    assert report["product"]["death_rate_delta95"]["n"] == 32
    assert report["multiplicity"]["paced_p_one_sided_threshold"] == 0.05


def test_independent_world_stratification_has_no_arbitrary_cross_mix_pairing():
    records = _records()
    effects = {}
    for mix_index, mix in enumerate(analysis.MIXES):
        values = [((index - 23.5) * (mix_index + 1)) for index in range(48)]
        effects[mix] = values
        rows = [row for row in records if row["kind"] == "terminal" and row["mix"] == mix]
        for row, value in zip(rows, values):
            row["arms"]["paced"]["mass_integral"] = 160 + value
    summary = analysis.skill_summary(records, "paced")["mass"]
    expected_variances = [statistics.variance(values) / 48 / 9 for values in effects.values()]
    expected_se = math.sqrt(sum(expected_variances))
    expected_df = sum(expected_variances) ** 2 / sum(v * v / 47 for v in expected_variances)
    assert summary["se"] == pytest.approx(expected_se)
    assert summary["df"] == pytest.approx(expected_df)
    # Reverse just one independent mix; the estimate/SE must stay unchanged.
    frozen = [row for row in records if row["kind"] == "terminal" and row["mix"] == "frozen"]
    masses = [row["arms"]["paced"]["mass_integral"] for row in frozen][::-1]
    for row, mass in zip(frozen, masses):
        row["arms"]["paced"]["mass_integral"] = mass
    assert analysis.skill_summary(records, "paced")["mass"] == summary


@pytest.mark.parametrize("pvalue,rejected", [(0.05, True), (0.0500001, False)])
def test_holm_preserves_fixed_worst_case_threshold_without_unpaced_data(
    monkeypatch, pvalue, rejected
):
    monkeypatch.setattr(analysis, "_one_sided_p", lambda _: pvalue)
    report = analysis.decide(_records(), _timing())
    assert report["paced"]["mass_holm_rejected"] == rejected
    assert "unpaced_descriptive" not in report


def test_survival_band_v2_is_strict_and_has_absolute_alive_floor():
    records = _records()
    for row in records[:144]:
        for name in analysis.MAC_ARMS:
            row["arms"][name] = _arm(100 if name == "bare" else 160, survival=0.2, deaths=1)
    summary = analysis.skill_summary(records, "paced")
    assert summary["clauses"]["survival_guard_a_absolute_floor"] is False
    # A bound exactly on the NI margin fails, unlike >=.
    for row in records[:144]:
        row["arms"]["bare"]["survival_fraction"] = 0.05
        row["arms"]["paced"]["survival_fraction"] = 0.0
    summary = analysis.skill_summary(records, "paced")
    assert summary["clauses"]["survival_guard_a_pooled"] is False
    for row in records[:144]:
        row["arms"]["bare"]["survival_fraction"] = 0.075
        row["arms"]["paced"]["survival_fraction"] = 0.0
    summary = analysis.skill_summary(records, "paced")
    assert summary["clauses"]["survival_guard_a_each_mix"] is False


def test_timing_failure_kills_even_when_skill_positive():
    timing = _timing()
    timing["probes"]["paced-24"]["update_ms"] = {"n": 10, "p95": 12.0, "p99": 12.0, "max": 12.0}
    assert analysis.decide(_records(), timing)["verdict"] == "KILL"


def test_overrun_and_unconditional_fault_override_partial_and_uncertainty():
    rows = _records()
    rows[0]["arms"]["paced"]["unconditional_faults"] = ["illegal_action"]
    assert analysis.decide(rows[:1], {}, complete=False)["verdict"] == "KILL"
    rows = _records()
    rows[0]["arms"]["bare"]["unconditional_faults"] = ["nonfinite_q"]
    assert analysis.decide(rows[:1], {}, complete=False)["verdict"] == "KILL"
    timing = _timing()
    timing["probes"]["paced-12"]["overruns"] = 1
    assert analysis.decide([], timing, complete=False)["verdict"] == "KILL"


def test_paced_skill_failure_kills_without_unpaced_efficacy_comparator():
    rows = _records()
    for row in rows[:144]:
        row["arms"]["paced"]["mass_integral"] = 110
    report = analysis.decide(rows, _timing())
    assert (report["verdict"], report["reason"]) == ("KILL", "paced skill bars failed")


def test_near_zero_lb_with_material_point_precedes_scientific_kill():
    rows = _records()
    for index, row in enumerate(rows[:144]):
        # Nonnegative masses, mean gain+60, large uncertainty near the zero LB.
        row["arms"]["bare"]["mass_integral"] = 1000
        row["arms"]["paced"]["mass_integral"] = 1060 + (500 if index % 2 else -500)
    report = analysis.decide(rows, _timing())
    assert report["paced"]["mass"]["mean"] >= 40
    assert abs(report["paced"]["mass"]["lower"]) <= 10
    assert report["verdict"] == "INCONCLUSIVE"


def test_product_pass_single_skill_miss_with_overlapping_interval_is_inconclusive():
    rows = _records()
    # No scripted no-death improvement, but uncertainty extends past +.03.
    rows[0]["arms"]["paced"] = _arm(160, survival=0.5, deaths=1)
    rows[1]["arms"]["paced"] = _arm(160, survival=0.99, deaths=1)
    rows[2]["arms"]["bare"] = _arm(100, survival=0.99, deaths=1)
    report = analysis.decide(rows, _timing())
    assert report["product"]["pass"]
    assert not report["paced"]["clauses"]["survival_guard_b_scripted"]
    assert report["verdict"] == "INCONCLUSIVE"


def test_product_mass_bar_is_gating_and_sessions_are_the_unit():
    rows = _records()
    for row in rows[144:]:
        row["arms"]["paced"]["mass_integral"] = 90
    report = analysis.decide(rows, _timing())
    assert not report["product"]["clauses"]["relative_mass_lower95_ge_minus005"]
    assert report["verdict"] != "GO"


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing",
        "horizon",
        "identity",
        "parity",
        "nonfinite",
        "survival",
        "no_death",
        "boost",
        "death_rate",
        "action_counts",
        "fault_evidence",
        "product_mix",
    ],
)
def test_bad_or_missing_evidence_is_inconclusive(fault):
    rows = _records()
    if fault == "duplicate":
        rows[-1]["seed"] = rows[0]["seed"]
    elif fault == "missing":
        rows.pop()
    elif fault == "horizon":
        rows[0]["horizon"] = 500
    elif fault in ("identity", "parity"):
        rows[0][f"{fault}_status"] = "FAIL"
    elif fault == "nonfinite":
        rows[0]["arms"]["paced"]["mass_integral"] = float("nan")
    elif fault == "survival":
        rows[0]["arms"]["paced"]["survival_fraction"] = 1.1
    elif fault == "no_death":
        rows[0]["arms"]["paced"]["no_death"] = 0.9
    elif fault == "boost":
        rows[0]["arms"]["paced"]["boost_decisions"] = 5001
    elif fault == "death_rate":
        rows[-1]["arms"]["paced"]["deaths_per_1k"] = 2
    elif fault == "action_counts":
        rows[0]["arms"]["paced"]["action_counts"][0] += 1
    elif fault == "fault_evidence":
        del rows[0]["arms"]["paced"]["unconditional_faults"]
    elif fault == "product_mix":
        rows[-1]["mix"] = "scripted"
    assert analysis.decide(rows, _timing())["verdict"] == "INCONCLUSIVE"


def test_partial_and_contended_are_inconclusive():
    assert analysis.decide([], {}, complete=False)["verdict"] == "INCONCLUSIVE"
    timing = _timing()
    timing["probes"]["paced-24"]["contended"] = True
    assert analysis.decide(_records(), timing)["verdict"] == "INCONCLUSIVE"


def test_timing_contention_within_two_percent_requires_and_accepts_accounting():
    timing = _timing()
    timing["probes"]["paced-24"]["contended"] = True
    timing["host_summary"] = _host_summary(
        contended_seconds=analysis.PHASE_CAP_SECONDS * analysis.MAX_CONTENDED_FRACTION
    )
    assert analysis.decide(_records(), timing)["verdict"] == "GO"

    del timing["host_summary"]
    report = analysis.decide(_records(), timing)
    assert report["verdict"] == "INCONCLUSIVE"
    assert "contention" in report["reason"]


def test_duplicate_json_keys_and_nan_are_rejected(tmp_path):
    source = tmp_path / "source.json"
    for content in ('{"seed": 1, "seed": 2}', '{"value": NaN}'):
        source.write_text(content)
        with pytest.raises(analysis.InvalidRetention):
            analysis._load(source)


def test_null_fault_evidence_cannot_crash_or_establish_a_decision():
    rows = _records()
    rows[0]["arms"]["paced"]["unconditional_faults"] = None
    assert analysis.decide(rows, _timing())["verdict"] == "INCONCLUSIVE"


def test_embedded_unpaced_arm_is_not_a_valid_mac_record():
    rows = _records()
    rows[0]["arms"]["unpaced"] = _arm()
    assert analysis.decide(rows, _timing())["verdict"] == "INCONCLUSIVE"


def _write_artifacts(root, monkeypatch):
    import hashlib
    import json

    from research.t34_pathb import retention_v2

    # Identity verification itself belongs to registration tests. This exercises
    # the consumer's row/completion binding after that verifier accepts a plan.
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_: None)
    rows = _records()
    registration = {"terminal_rows": [], "product_rows": []}
    registration["source_commit"] = "a" * 40
    for row in rows:
        world = {"seed": row["seed"], "mix": row["mix"], "roster": ["frozen", "scripted"]}
        registration[f"{row['kind']}_rows"].append(world)
        row["world"] = world
    (root / "registration.json").write_text(json.dumps(registration))
    digest = hashlib.sha256((root / "registration.json").read_bytes()).hexdigest()
    (root / "precheck.json").write_text("{}")
    (root / "records").mkdir()
    for index, row in enumerate(rows):
        row["registration_sha256"] = digest
        (root / "records" / f"{index:03}.json").write_text(json.dumps(row))
    (root / "timing.json").write_text(json.dumps(_timing()))
    (root / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": 1.0,
                "cpu_seconds": 1.0,
                "registration_sha256": digest,
                "identity_status": "PASS",
                "parity_status": "PASS",
                "host_summary": _host_summary(),
            }
        )
    )


def test_artifact_loader_binds_registered_world_and_completion(tmp_path, monkeypatch):
    import json

    _write_artifacts(tmp_path, monkeypatch)
    assert analysis.analyze_path(tmp_path)["verdict"] == "GO"
    path = tmp_path / "records" / "000.json"
    row = json.loads(path.read_text())
    row["world"]["roster"] = ["different"]
    path.write_text(json.dumps(row))
    report = analysis.analyze_path(tmp_path)
    assert report["verdict"] == "INCONCLUSIVE"
    assert "registered world" in report["reason"]


def test_artifact_loader_partial_run_still_honors_recorded_unconditional_fault(
    tmp_path, monkeypatch
):
    import json

    _write_artifacts(tmp_path, monkeypatch)
    (tmp_path / "complete.json").unlink()
    path = tmp_path / "records" / "000.json"
    row = json.loads(path.read_text())
    row["arms"]["paced"]["unconditional_faults"] = ["illegal_action"]
    path.write_text(json.dumps(row))
    assert analysis.analyze_path(tmp_path)["verdict"] == "KILL"


def test_artifact_loader_missing_completion_is_inconclusive(tmp_path, monkeypatch):
    _write_artifacts(tmp_path, monkeypatch)
    (tmp_path / "complete.json").unlink()
    assert analysis.analyze_path(tmp_path)["verdict"] == "INCONCLUSIVE"


def test_artifact_loader_requires_cumulative_host_accounting(tmp_path, monkeypatch):
    _write_artifacts(tmp_path, monkeypatch)
    path = tmp_path / "complete.json"
    import json

    receipt = json.loads(path.read_text())
    receipt.pop("host_summary")
    path.write_text(json.dumps(receipt))
    report = analysis.analyze_path(tmp_path)
    assert report["verdict"] == "INCONCLUSIVE"
    assert "host contention" in report["reason"]


def test_analyze_path_wraps_identity_analysis_in_source_fence(tmp_path, monkeypatch):
    calls = []

    class Fence:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *_args):
            calls.append("exit")
            return False

    monkeypatch.setattr(
        source_binding_v2,
        "FrozenWorktreeGuard",
        lambda *args, **kwargs: (calls.append((args, kwargs)) or Fence()),
    )
    _write_artifacts(tmp_path, monkeypatch)
    report = analysis.analyze_path(tmp_path)
    assert report["verdict"] == "GO"
    assert "enter" in calls and "exit" in calls


def test_analyze_cli_source_failure_invalidates_saved_decision(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from research.t34_pathb import retention_v2

    _write_artifacts(tmp_path, monkeypatch)
    official = {
        "schema": "t34-pathb-v2-decision/v1",
        "verdict": "GO",
        "reason": "saved official result",
        "release_authorized": False,
    }
    (tmp_path / "decision.json").write_text(json.dumps(official))
    (tmp_path / "source_failure.json").write_text(
        json.dumps(
            {
                "status": "INCONCLUSIVE",
                "reason": "HEAD moved during analysis",
                "invalidates_decision": True,
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["retention_v2", "analyze", "--out", str(tmp_path)])

    assert retention_v2.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["official_decision"]["verdict"] == "INCONCLUSIVE"
    assert report["preserved_decision"] == official
    assert report["source_failure"]["invalidates_decision"] is True


def test_product_late_large_diagnostics_must_reconcile():
    rows = _records()
    rows[-1]["arms"]["paced"]["deaths_length_ge_800"] = 3
    rows[-1]["arms"]["paced"]["late_deaths"] = 3
    assert analysis.decide(rows, _timing())["verdict"] == "INCONCLUSIVE"
    rows[-1]["arms"]["paced"]["late_large_overlap"] = 3
    report = analysis.decide(rows, _timing())
    assert report["verdict"] == "GO"
    assert report["product"]["length_diagnostics"][-1]["late_large_overlap"] == 3


def test_partial_receipt_preserves_fault_even_before_first_world(tmp_path, monkeypatch):
    import json

    _write_artifacts(tmp_path, monkeypatch)
    (tmp_path / "complete.json").unlink()
    for path in (tmp_path / "records").glob("*.json"):
        path.unlink()
    (tmp_path / "incomplete.json").write_text(
        json.dumps({"unconditional_faults": ["illegal_action"]})
    )
    assert analysis.analyze_path(tmp_path)["verdict"] == "KILL"


@pytest.mark.parametrize("cap,value", [("elapsed_seconds", 40 * 3600), ("cpu_seconds", 60 * 3600)])
def test_recorded_cap_reached_is_inconclusive(tmp_path, monkeypatch, cap, value):
    import json

    _write_artifacts(tmp_path, monkeypatch)
    path = tmp_path / "complete.json"
    receipt = json.loads(path.read_text())
    receipt[cap] = value
    path.write_text(json.dumps(receipt))
    report = analysis.analyze_path(tmp_path)
    assert report["verdict"] == "INCONCLUSIVE"
    assert "cap reached" in report["reason"]


def test_registration_verifier_failure_is_an_inconclusive_report(tmp_path, monkeypatch):
    from research.t34_pathb import retention_v2

    _write_artifacts(tmp_path, monkeypatch)

    def reject(*_):
        raise retention_v2.RegistrationError("source hash mismatch")

    monkeypatch.setattr(retention_v2, "require_registration", reject)
    report = analysis.analyze_path(tmp_path)
    assert report["verdict"] == "INCONCLUSIVE"
    assert "source hash mismatch" in report["reason"]
