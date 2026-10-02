"""Audit rules of the v5 web-serving run, tested on REAL record shapes.

Positive tests use the producer's dry-run smoke (fixtures/smoke-v1, unmodified) and one
v5 strict final record whose ``probes.safety_veto`` was written by
``BoostAwareFreeSpaceVeto.record()``. Every negative test mutates a copy. Nothing here
plays an episode.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.apex_veto_v5_serving_20261001 import serving_audit as audit_mod

HERE = Path(__file__).resolve().parents[1] / "research" / "apex_veto_v5_serving_20261001"
SMOKE = HERE / "fixtures" / "smoke-v1"
STRICT_RECORD = HERE / "fixtures" / "v5_strict_final_record.json"
WATCH = "records/watch_hero-000.json"
PLAY = "records/play-000.json"
PARITY = "parity/parity-000.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run_copy(tmp_path):
    root = tmp_path / "run"
    shutil.copytree(SMOKE, root)
    return root


def mutate(root: Path, rel: str, change, rehash: bool = True) -> None:
    """Apply ``change`` to one JSON file and (by default) re-hash it in the receipt."""
    path = root / rel
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    if rehash and rel != "receipt.json":
        receipt = json.loads((root / "receipt.json").read_text())
        receipt["files"][rel] = _sha(path)
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


def failing(result) -> set:
    return {name for name, c in result["criteria"].items() if not c["pass"]}


def as_real_run(root: Path) -> None:
    """Relabel the smoke as non-smoke (only S1/S3/S5 design checks should then trip)."""
    mutate(root, "intent.json", lambda d: d.update(smoke=False), rehash=False)
    intent_sha = _sha(root / "intent.json")
    mutate(root, "receipt.json", lambda d: d.update(intent_sha256=intent_sha))


class TestPositive:
    def test_fixture_provenance(self):
        provenance = json.loads((HERE / "fixtures" / "provenance.json").read_text())
        assert provenance["v5_strict_final_record.json"]["sha256"] == _sha(STRICT_RECORD)

    def test_smoke_passes_every_criterion_and_never_qualifies(self):
        result = audit_mod.audit(SMOKE)
        assert result["verdict"] == "SERVING_PASS", result["criteria"]
        assert result["serving_path_qualified"] is False
        assert result["reported"]["parity"]["label"] in (
            "exercised",
            "landing-exercised",
            "non-exercising",
        )

    def test_strict_probe_shape_matches_the_pinned_descriptor(self):
        record = json.loads(STRICT_RECORD.read_text())
        probe = record["record"]["probes"]["safety_veto"]
        descriptor = {k: v for k, v in probe.items() if k != "counters"}
        assert descriptor == audit_mod.DESCRIPTOR
        assert audit_mod.counter_failures(probe["counters"], "strict") == []
        assert record["wrapper"] == audit_mod.V5_METHOD
        assert record["wrapper_source_sha256s"] == audit_mod.V5_SOURCE_SHA256S

    def test_isolated_cli_is_stdlib_only(self, tmp_path):
        out = tmp_path / "audit"
        script = HERE / "serving_audit.py"
        proc = subprocess.run(
            [sys.executable, "-I", str(script), "--root", str(SMOKE), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert json.loads((out / "audit.json").read_text())["verdict"] == "SERVING_PASS"


class TestNegative:
    def test_missing_listed_file_is_invalid(self, run_copy):
        (run_copy / PLAY).unlink()
        assert audit_mod.audit(run_copy)["verdict"] == "INVALID"

    def test_hash_mismatch_fails_s1(self, run_copy):
        mutate(run_copy, WATCH, lambda d: d.update(wall_seconds=1.0), rehash=False)
        assert "S1" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "rel,change,criterion",
        [
            (WATCH, lambda d: d.update(status="error"), "S1"),
            (WATCH, lambda d: d.update(world_seed=1), "S1"),
            (PLAY, lambda d: d.update(frames_stepped=499, ended_by="horizon"), "S1"),
            (WATCH, lambda d: d["wrapper"].update(source_sha256="0" * 64), "S2"),
            (
                WATCH,
                lambda d: d["wrapper"]["source_sha256s"].pop("src/evaluation/safety_veto_v3.py"),
                "S2",
            ),  # noqa: E501
            (WATCH, lambda d: d["checkpoint"].update(session_sha256="0" * 64), "S2"),
            (
                WATCH,
                lambda d: d.update(env={**d["env"], "SNAKE_SERVE_VETO_WATCH_HERO": "1"}),
                "S2",
            ),  # noqa: E501
            (PLAY, lambda d: d.update(wrapper={"method": "x"}), "S2"),
            (WATCH, lambda d: d["served"].update(epsilon=0.05), "S2"),
            (WATCH, lambda d: d["veto"].update(variant="v2"), "S3"),
            (WATCH, lambda d: d["veto"].update(strict_receipt_wrapper_match=False), "S3"),
            (WATCH, lambda d: d.update(snakes_with_veto=[0, 1]), "S3"),
            (WATCH, lambda d: d["veto"].update(wrapped_snake_ids=[1]), "S3"),
            (PLAY, lambda d: d["veto"].update(active=True, scope="play_ai"), "S3"),
            (PLAY, lambda d: d.update(snakes_with_veto=[1]), "S3"),
            (PLAY, lambda d: d["veto"].update(variant_requested="v2"), "S3"),
        ],
    )
    def test_rules(self, run_copy, rel, change, criterion):
        mutate(run_copy, rel, change)
        assert criterion in failing(audit_mod.audit(run_copy))

    def _watch_counters(self, run_copy, change):
        def apply(d):
            sid = str(d["served"]["hero_id"])
            change(d["veto"]["per_snake"][sid], d["veto"]["diagnostics_per_snake"][sid], d)

        mutate(run_copy, WATCH, apply)
        return failing(audit_mod.audit(run_copy))

    def test_s4_decisions_must_equal_decision_frames(self, run_copy):
        def change(c, diag, d):
            c["decisions"] += 1
            c["kept_base"] += 1
            diag["decisions"] += 1
            diag["kept"] += 1

        assert "S4" in self._watch_counters(run_copy, change)

    def test_s4_diagnostics_must_mirror_counters(self, run_copy):
        assert "S4" in self._watch_counters(run_copy, lambda c, diag, d: diag.update(kept=0))
        failures = audit_mod.s4_counters(audit_mod.load_evidence(run_copy))
        assert any("mirror_kept" in f for f in failures)

    def test_s4_landing_split_identity(self, run_copy):
        def change(c, diag, d):
            diag["base_landing_failed"] += 1  # with no matching landing veto or no_eligible

        assert "S4" in self._watch_counters(run_copy, change)

    def test_s4_landing_vetoes_are_boost_vetoes(self, run_copy):
        def change(c, diag, d):
            c["kept_base"] -= 1
            c["vetoes_applied"] += 1  # a veto whose base did not boost ...
            diag["kept"] -= 1
            diag["vetoes_applied"] += 1
            diag["base_landing_failed"] += 1  # ... cannot be a landing veto
            diag["boost_landing_vetoes"] += 1
            diag["boost_to_normal_same_direction"] += 1

        assert "S4" in self._watch_counters(run_copy, change)
        failures = audit_mod.s4_counters(audit_mod.load_evidence(run_copy))
        assert any("landing_vetoes_are_boost_vetoes" in f for f in failures)

    def test_s4_play_must_have_no_wrapped_decisions(self, run_copy):
        mutate(run_copy, PLAY, lambda d: d["decision_frames"].update(total=3))
        assert "S4" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d.update(first_divergence_frame=5),
            lambda d: d.update(diagnostics_equal=False),
            lambda d: d["rollout"]["diagnostics"].update(landing_checks=999),
            lambda d: d["rollout"].update(wrapper_method="free-space-veto/v2-speed-preserving"),
            lambda d: d["session"].update(variant="v2"),
            lambda d: d.update(env={}),
        ],
    )
    def test_s5_rules(self, run_copy, change):
        mutate(run_copy, PARITY, change)
        assert "S5" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d.update(variant_from_empty_env="v5"),
            lambda d: d["builds"]["empty_env"]["modes"]["watch"].update(method="x"),
            lambda d: d["builds"]["variant_v2"]["modes"]["play"].update(snakes_with_veto=[1]),
            lambda d: d["builds"]["watch_off_variant_v5"]["modes"]["watch"].update(active=True),
            lambda d: d.update(flags_from_empty_env={"watch_hero": False, "play_ai": False}),
        ],
    )
    def test_s6_rules(self, run_copy, change):
        mutate(run_copy, "default_check.json", change)
        assert "S6" in failing(audit_mod.audit(run_copy))


class TestRealRunGates:
    def test_relabelled_smoke_cannot_qualify(self, run_copy):
        as_real_run(run_copy)
        result = audit_mod.audit(run_copy)
        assert result["serving_path_qualified"] is False
        assert {"S1", "S3"} <= failing(result)  # design/preregistration and zero Watch vetoes

    def test_watch_activity_counts_either_branch(self, run_copy):
        ev = audit_mod.load_evidence(run_copy)
        assert audit_mod.watch_activity(ev)["vetoes_applied"] == 0
        assert any("never replaced" in f for f in audit_mod.s3_scope(ev, smoke=False))

        def landing_veto(d):
            sid = str(d["served"]["hero_id"])
            c, diag = d["veto"]["per_snake"][sid], d["veto"]["diagnostics_per_snake"][sid]
            c["kept_base"] -= 1
            c["vetoes_applied"] += 1
            c["vetoed_base_boost"] += 1
            diag["kept"] -= 1
            diag["vetoes_applied"] += 1
            diag["base_landing_failed"] += 1
            diag["boost_landing_vetoes"] += 1
            diag["boost_to_normal_same_direction"] += 1
            for key in ("vetoes_applied", "vetoed_base_boost", "kept_base"):
                d["veto"]["total"][key] = c[key]
            for key in ("kept", "vetoes_applied", "base_landing_failed"):
                d["veto"]["diagnostics_total"][key] = diag[key]
            for key in ("boost_landing_vetoes", "boost_to_normal_same_direction"):
                d["veto"]["diagnostics_total"][key] = diag[key]

        mutate(run_copy, WATCH, landing_veto)
        ev = audit_mod.load_evidence(run_copy)
        activity = audit_mod.watch_activity(ev)
        assert activity["vetoes_applied"] == 1 and activity["boost_landing_vetoes"] == 1
        assert activity["v2_path_vetoes"] == 0
        assert audit_mod.s3_scope(ev, smoke=False) == []
        assert audit_mod.s4_counters(ev) == []

    def test_bound_intent_has_no_preregistration_failures(self):
        intent = json.loads((SMOKE / "intent.json").read_text())
        intent.update(protocol_sha256=audit_mod.PROTOCOL_SHA256, git={"dirty": False})
        assert audit_mod.preregistration_failures(intent) == []
        intent["git"]["dirty"] = True
        assert audit_mod.preregistration_failures(intent) == ["intent.git.dirty is not false"]
