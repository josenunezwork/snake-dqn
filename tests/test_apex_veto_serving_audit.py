"""Audit rules of the Apex veto web-serving run, tested on REAL record shapes.

Positive tests use records captured from the real producer paths (governance
"Audit scope" rule 4): the dry-run smoke of ``serving_run.py`` (fixtures/smoke-v1,
unmodified) and one run-v3 STRICT_PASS serving record whose ``probes.safety_veto``
was written by ``FreeSpaceVeto.record()``. Every negative test mutates a copy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.apex_veto_serving_20261001 import serving_audit as audit_mod

HERE = Path(__file__).resolve().parents[1] / "research" / "apex_veto_serving_20261001"
SMOKE = HERE / "fixtures" / "smoke-v1"
RUN_V3_RECORD = HERE / "fixtures" / "run_v3_serving_record.json"
WATCH = "records/watch_hero-000.json"
PLAY = "records/play_ai-000.json"
PARITY = "parity/parity-000.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run_copy(tmp_path):
    root = tmp_path / "run"
    shutil.copytree(SMOKE, root)
    return root


def mutate(root: Path, rel: str, change, rehash: bool = True) -> None:
    """Apply ``change`` to one JSON file; by default re-hash it in the receipt so the
    tested rule (not the S1 hash check) is what fails."""
    path = root / rel
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    if rehash:
        receipt = json.loads((root / "receipt.json").read_text())
        receipt["files"][rel] = _sha(path)
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


def failing(result) -> set:
    return {name for name, c in result["criteria"].items() if not c["pass"]}


class TestPositiveOnRealRecords:
    def test_fixture_provenance(self):
        provenance = json.loads((HERE / "fixtures" / "provenance.json").read_text())
        assert _sha(RUN_V3_RECORD) == provenance["run_v3_serving_record.json"]["sha256"]

    def test_smoke_output_passes_every_criterion(self):
        result = audit_mod.audit(SMOKE)
        assert result["verdict"] == "SERVING_PASS", result["criteria"]
        assert failing(result) == set()
        # A smoke never qualifies the serving path.
        assert result["serving_path_qualified"] is False
        assert result["smoke"] is True

    def test_run_v3_probe_shape_passes(self):
        record = json.loads(RUN_V3_RECORD.read_text())
        probe = record["record"]["probes"]["safety_veto"]
        assert audit_mod.probe_record_failures(probe, "run-v3") == []
        assert audit_mod.counter_failures(probe["counters"], "run-v3") == []
        # The serving receipt's counter keys are exactly the strict record's.
        assert set(probe["counters"]) == set(audit_mod.schema.COUNTER_KEYS)

    def test_serving_wrapper_matches_strict_record_identity(self):
        record = json.loads(RUN_V3_RECORD.read_text())
        watch = json.loads((SMOKE / WATCH).read_text())
        assert watch["wrapper"]["source_sha256"] == record["wrapper_source_sha256"]
        strict_descriptor = {
            k: v for k, v in record["record"]["probes"]["safety_veto"].items() if k != "counters"
        }
        assert watch["wrapper"]["descriptor"] == strict_descriptor
        assert watch["checkpoint"]["sha256"] == record["hero_sha256"]

    def test_isolated_cli_is_stdlib_only(self, tmp_path):
        out = tmp_path / "audit"
        proc = subprocess.run(
            [sys.executable, "-I", str(HERE / "serving_audit.py"), "--root", str(SMOKE)]
            + ["--out", str(out)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads((out / "audit.json").read_text())["verdict"] == "SERVING_PASS"


class TestNegativeOnRealRecordCopies:
    def test_run_v3_counter_partition_break(self):
        probe = json.loads(RUN_V3_RECORD.read_text())["record"]["probes"]["safety_veto"]
        broken = copy.deepcopy(probe)
        broken["counters"]["decisions"] += 1
        assert audit_mod.probe_record_failures(broken, "x")
        broken = copy.deepcopy(probe)
        broken["counters"]["vetoes_to_boost"] = broken["counters"]["vetoes_applied"] + 1
        assert audit_mod.counter_failures(broken["counters"], "x")
        broken = copy.deepcopy(probe)
        broken["replacement_rule"] = "highest-q-eligible"
        assert audit_mod.probe_record_failures(broken, "x")

    def test_missing_listed_file_is_invalid(self, run_copy):
        (run_copy / PLAY).unlink()
        result = audit_mod.audit(run_copy)
        assert result["verdict"] == "INVALID"
        assert result["serving_path_qualified"] is False

    def test_hash_mismatch_fails_s1(self, run_copy):
        mutate(run_copy, PLAY, lambda d: d["episode"].update(inputs_sent=0), rehash=False)
        assert failing(audit_mod.audit(run_copy)) == {"S1"}

    @pytest.mark.parametrize(
        "rel, change",
        [
            (PLAY, lambda d: d.update(status="error", error="RuntimeError: x")),
            (PLAY, lambda d: d.update(world_seed=d["world_seed"] + 1)),
            (WATCH, lambda d: d.update(frames_stepped=499)),
            (WATCH, lambda d: d.update(intent_sha256="0" * 64)),
            (PLAY, lambda d: d.update(extra_field=1)),
            (PLAY, lambda d: d.update(ended_by="human_death")),
        ],
    )
    def test_s1_rules(self, run_copy, rel, change):
        mutate(run_copy, rel, change)
        assert "S1" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "rel, change",
        [
            (WATCH, lambda d: d["checkpoint"].update(session_sha256="f" * 64)),
            (PLAY, lambda d: d["wrapper"].update(source_sha256="e" * 64)),
            (PLAY, lambda d: d["wrapper"]["descriptor"].update(free_space_bfs_cap=200)),
            (WATCH, lambda d: d["served"].update(epsilon=0.1)),
            (WATCH, lambda d: d["served"].update(config_sha256="d" * 64)),
            (PLAY, lambda d: d["served"].update(obs_spec="raster31v2")),
            (PLAY, lambda d: d["served"].update(mode="watch")),
            (WATCH, lambda d: d["served"].update(policy_training=True)),
        ],
    )
    def test_s2_rules(self, run_copy, rel, change):
        mutate(run_copy, rel, change)
        assert "S2" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "rel, change",
        [
            (WATCH, lambda d: d["veto"].update(active=False)),
            (WATCH, lambda d: d["veto"].update(wrapped_snake_ids=[0, 1])),
            (PLAY, lambda d: d["veto"].update(wrapped_snake_ids=[0, 1, 2, 3, 4, 5])),
            (PLAY, lambda d: d["veto"].update(flags={"watch_hero": True, "play_ai": True})),
            (PLAY, lambda d: d["veto"].update(scope="watch_hero")),
            (WATCH, lambda d: d["served"].update(hero_id=3)),
        ],
    )
    def test_s3_rules(self, run_copy, rel, change):
        mutate(run_copy, rel, change)
        assert "S3" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "rel, change",
        [
            (WATCH, lambda d: d["veto"]["per_snake"]["0"].update(decisions=501)),
            (WATCH, lambda d: d["decision_frames"]["per_snake"].update({"0": 499})),
            (PLAY, lambda d: d["veto"]["per_snake"]["1"].update(vetoes_to_boost=99)),
            (PLAY, lambda d: d["veto"]["total"].update(kept_base=0)),
            (PLAY, lambda d: d["decision_frames"].update(total=1)),
            (PLAY, lambda d: d["veto"]["per_snake"].pop("5")),
        ],
    )
    def test_s4_rules(self, run_copy, rel, change):
        mutate(run_copy, rel, change)
        assert "S4" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d.update(first_divergence_frame=5),
            lambda d: d.update(frames_compared=199),
            lambda d: d["rollout"]["counters"].update(kept_base=199, vetoes_applied=1),
            lambda d: d["rollout"].update(trace_sha256="a" * 64),
            lambda d: d.update(status="error"),
            lambda d: d["session"].update(config_sha256="b" * 64),
            lambda d: d["session"].update(decision_frames=150),
        ],
    )
    def test_s5_rules(self, run_copy, change):
        mutate(run_copy, PARITY, change)
        assert "S5" in failing(audit_mod.audit(run_copy))

    def test_s5_requires_a_complete_rollout_outside_smoke(self):
        ev = audit_mod.load_evidence(SMOKE)
        design = audit_mod.DESIGN[True]
        assert audit_mod.s5_parity(ev, design, smoke=True) == []
        assert any("record_complete" in f for f in audit_mod.s5_parity(ev, design, smoke=False))

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d["modes"]["watch"].update(active=True),
            lambda d: d["modes"]["play"].update(snakes_with_veto=[1]),
            lambda d: d["flags_from_empty_env"].update(watch_hero=True),
            lambda d: d.update(served_config_sha256="c" * 64),
            lambda d: d.update(default_checkpoint_basename="best_apex.pth"),
            lambda d: d.update(**{"pass": False}),
        ],
    )
    def test_s6_rules(self, run_copy, change):
        mutate(run_copy, "default_check.json", change)
        assert failing(audit_mod.audit(run_copy)) == {"S6"}

    def test_flipping_smoke_off_cannot_qualify(self, run_copy):
        mutate(run_copy, "intent.json", lambda d: d.update(smoke=False), rehash=False)
        result = audit_mod.audit(run_copy)
        assert result["verdict"] == "SERVING_FAIL"
        assert result["serving_path_qualified"] is False
        assert "S1" in failing(result)


def _bound_intent():
    """The real smoke intent shape, mutated into a correctly bound non-smoke intent."""
    intent = json.loads((SMOKE / "intent.json").read_text())
    intent.update(smoke=False, protocol_sha256=audit_mod.PROTOCOL_SHA256)
    intent["git"]["dirty"] = False
    assert intent["seed_report"]["disjoint"] is True
    return intent


class TestPreregistrationBinding:
    def test_pins_equal_the_current_protocol_bytes(self):
        from research.apex_veto_serving_20261001 import serving_run

        sha = _sha(HERE / "protocol.md")
        assert audit_mod.PROTOCOL_SHA256 == sha == serving_run.PROTOCOL_SHA256

    def test_bound_intent_has_no_failures(self):
        assert audit_mod.preregistration_failures(_bound_intent()) == []

    @pytest.mark.parametrize(
        "change, needle",
        [
            (lambda d: d.update(protocol_sha256="a" * 64), "protocol_sha256"),
            (lambda d: d.pop("protocol_sha256"), "protocol_sha256"),
            (lambda d: d["git"].update(dirty=True), "git.dirty"),
            (lambda d: d.pop("git"), "git.dirty"),
            (lambda d: d["seed_report"].update(disjoint=False), "disjoint"),
            (lambda d: d["seed_report"].pop("disjoint"), "disjoint"),
        ],
    )
    def test_each_binding_check(self, change, needle):
        intent = _bound_intent()
        change(intent)
        failures = audit_mod.preregistration_failures(intent)
        assert len(failures) == 1 and needle in failures[0]

    def test_smoke_is_exempt_and_its_real_intent_is_unbound(self):
        # The smoke ran on a dirty tree under the pre-revision protocol ...
        smoke_intent = json.loads((SMOKE / "intent.json").read_text())
        assert len(audit_mod.preregistration_failures(smoke_intent)) == 2
        # ... which S1 ignores for a smoke (a smoke never qualifies).
        assert audit_mod.audit(SMOKE)["criteria"]["S1"]["pass"] is True

    def test_s1_enforces_it_outside_smoke(self, run_copy):
        mutate(run_copy, "intent.json", lambda d: d.update(smoke=False), rehash=False)
        ev = audit_mod.load_evidence(run_copy)
        failures = audit_mod.s1_completeness(ev, audit_mod.DESIGN[False])
        assert any("protocol_sha256" in f for f in failures)
        assert any("git.dirty" in f for f in failures)

    def test_real_design_counts_weight_watch(self):
        assert audit_mod.DESIGN[False]["counts"] == {"watch": 25, "play": 25, "parity": 2}


class TestReplacementBranchExercised:
    def test_watch_gate_fails_the_real_smoke_shape_outside_smoke(self):
        ev = audit_mod.load_evidence(SMOKE)  # the smoke Watch record has 0 vetoes_applied
        assert audit_mod.s3_scope(ev, smoke=True) == []
        failures = audit_mod.s3_scope(ev, smoke=False)
        assert failures and "never replaced an action" in failures[0]

    def test_watch_gate_passes_once_an_action_was_replaced(self, run_copy):
        def one_veto(d):
            d["veto"]["total"].update(vetoes_applied=1)

        mutate(run_copy, WATCH, one_veto)
        assert audit_mod.s3_scope(audit_mod.load_evidence(run_copy), smoke=False) == []

    def test_parity_label(self, run_copy):
        reported = audit_mod.audit(SMOKE)["reported"]["parity"]
        assert reported == {
            "vetoes_applied_per_probe": {"parity-000": 0},
            "vetoes_applied_total": 0,
            "label": "non-exercising",
        }

        def one_veto(d):
            d["session"]["counters"].update(kept_base=199, vetoes_applied=1)
            d["rollout"]["counters"].update(kept_base=199, vetoes_applied=1)

        mutate(run_copy, PARITY, one_veto)
        result = audit_mod.audit(run_copy)
        assert result["reported"]["parity"]["label"] == "exercised"
        assert result["verdict"] == "SERVING_PASS"  # the label is reported, not gated
