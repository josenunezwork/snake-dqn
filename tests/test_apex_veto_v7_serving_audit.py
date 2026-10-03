"""Audit rules of the v7 web-serving run, tested on REAL record shapes.

Positive tests use the producer's dry-run smoke (fixtures/smoke-v1, unmodified) and one
v7 strict final record whose ``probes.safety_veto`` was written by
``SpacePreferenceVeto.record()`` and ``veto_diagnostics`` by ``diagnostics_record()``
(a world with a re-rank change and a v5 landing veto). Every negative test mutates a copy.
Nothing here plays an episode.
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

from research.apex_veto_v7_serving_20261002 import serving_audit as audit_mod

HERE = Path(__file__).resolve().parents[1] / "research" / "apex_veto_v7_serving_20261002"
SMOKE = HERE / "fixtures" / "smoke-v1"
STRICT_RECORD = HERE / "fixtures" / "v7_strict_final_record.json"
WATCH = "records/watch_hero-000.json"
PLAY = "records/play-000.json"
PARITY = "parity/parity-000.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def int_diag(record: dict) -> dict:
    """The serving_run.int_diagnostics shape, re-derived from a raw diagnostics_record."""
    schema = audit_mod.schema
    out = {k: record[k] for k in schema.DIAGNOSTIC_KEYS}
    out["v5"] = {k: record["v5"][k] for k in schema.V5_DIAGNOSTIC_KEYS}
    return out


@pytest.fixture
def run_copy(tmp_path):
    root = tmp_path / "run"
    shutil.copytree(SMOKE, root)
    return root


@pytest.fixture
def strict():
    record = json.loads(STRICT_RECORD.read_text())
    return record, int_diag(record["veto_diagnostics"])


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
        assert provenance["v7_strict_final_record.json"]["sha256"] == _sha(STRICT_RECORD)

    def test_smoke_passes_every_criterion_and_never_qualifies(self):
        result = audit_mod.audit(SMOKE)
        assert result["verdict"] == "SERVING_PASS", result["criteria"]
        assert result["serving_path_qualified"] is False
        assert result["reported"]["parity"]["label"] in (
            "rerank-exercised",
            "exercised",
            "non-exercising",
        )

    def test_strict_probe_shape_matches_the_pinned_descriptor(self, strict):
        record, _ = strict
        probe = record["record"]["probes"]["safety_veto"]
        descriptor = {k: v for k, v in probe.items() if k != "counters"}
        assert descriptor == audit_mod.DESCRIPTOR
        assert audit_mod.counter_failures(probe["counters"], "strict") == []
        assert record["wrapper"] == audit_mod.V7_METHOD
        assert record["wrapper_source_sha256s"] == audit_mod.V7_SOURCE_SHA256S

    def test_strict_diagnostics_satisfy_every_v7_identity(self, strict):
        record, diag = strict
        counters = record["record"]["probes"]["safety_veto"]["counters"]
        assert diag["rerank_changes"] > 0 and diag["v5"]["boost_landing_vetoes"] > 0
        assert audit_mod.diagnostics_failures(diag, counters, "strict") == []
        raw = set(record["veto_diagnostics"]) - {"v5"}
        schema = audit_mod.schema
        assert raw == set(schema.DIAGNOSTIC_KEYS) | set(schema.DIAGNOSTIC_TIMING_KEYS)

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


class TestDiagnosticRules:
    @pytest.mark.parametrize(
        "change,rule",
        [
            (lambda d: d.update(kept=d["kept"] + 1), "partition"),
            (lambda d: d["v5"].update(decisions=d["v5"]["decisions"] + 1), "v5"),
            (
                lambda d: d.update(rerank_changes_of_v5_kept=d["rerank_changes_of_v5_kept"] + 1),
                "kept_vs_v5",
            ),
            (lambda d: d.update(rerank_pruned=d["rerank_pruned"] + 1), "rerank_split"),
            (
                lambda d: d.update(static_area_evaluations=d["static_area_evaluations"] + 2),
                "static_area",
            ),
            (lambda d: d.update(area_cap_hits=d["area_evaluations"] + 1), "cap_hits"),
            (
                lambda d: d.update(rerank_changes_tail_release_driven=d["rerank_changes"] + 1),
                "tail_release_driven",
            ),
            (lambda d: d["v5"].update(landing_failures=d["v5"]["landing_checks"] + 1), "v5"),
            (lambda d: d.pop("area_cap_max"), "keys"),
            (lambda d: d.update(rerank_changes=-1), "negative"),
        ],
    )
    def test_each_identity_is_enforced(self, strict, change, rule):
        record, diag = strict
        bad = copy.deepcopy(diag)
        change(bad)
        failures = audit_mod.diagnostics_failures(bad, None, "x")
        assert failures and any(rule in f for f in failures), failures

    def test_mirrors_against_counters(self, strict):
        record, diag = strict
        counters = dict(record["record"]["probes"]["safety_veto"]["counters"])
        counters["kept_base"] -= 1
        counters["fallback_no_spacious"] += 1
        failures = audit_mod.diagnostics_failures(diag, counters, "x")
        assert any("mirror_kept" in f for f in failures)

    def test_total_uses_max_for_area_cap_max(self, strict):
        _, diag = strict
        other = copy.deepcopy(diag)
        other["area_cap_max"] = diag["area_cap_max"] + 7
        total = audit_mod.diagnostics_total({"0": diag, "1": other})
        assert total["area_cap_max"] == diag["area_cap_max"] + 7
        assert total["decisions"] == 2 * diag["decisions"]
        assert total["v5"]["decisions"] == 2 * diag["v5"]["decisions"]


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
                lambda d: d["wrapper"]["source_sha256s"].pop("src/evaluation/safety_veto_v5.py"),
                "S2",
            ),
            (
                WATCH,
                lambda d: d["wrapper"]["descriptor"].update(space_preference_lambda=2.0),
                "S2",
            ),
            (WATCH, lambda d: d["checkpoint"].update(session_sha256="0" * 64), "S2"),
            (
                WATCH,
                lambda d: d.update(env={**d["env"], "SNAKE_SERVE_VETO_WATCH_HERO": "1"}),
                "S2",
            ),
            (PLAY, lambda d: d.update(wrapper={"method": "x"}), "S2"),
            (WATCH, lambda d: d["served"].update(epsilon=0.05), "S2"),
            (WATCH, lambda d: d["veto"].update(variant="v5"), "S3"),
            (WATCH, lambda d: d["veto"].update(strict_receipt_wrapper_match=False), "S3"),
            (WATCH, lambda d: d.update(snakes_with_veto=[0, 1]), "S3"),
            (WATCH, lambda d: d["veto"].update(wrapped_snake_ids=[1]), "S3"),
            (PLAY, lambda d: d["veto"].update(active=True, scope="play_ai"), "S3"),
            (PLAY, lambda d: d.update(snakes_with_veto=[1]), "S3"),
            (PLAY, lambda d: d["veto"].update(variant_requested="v5"), "S3"),
        ],
    )
    def test_rules(self, run_copy, rel, change, criterion):
        mutate(run_copy, rel, change)
        assert criterion in failing(audit_mod.audit(run_copy))

    def test_intent_expected_receipt_is_bound(self, run_copy):
        mutate(
            run_copy,
            "intent.json",
            lambda d: d["expected"].update(v7_strict_receipt_sha256="0" * 64),
            rehash=False,
        )
        intent_sha = _sha(run_copy / "intent.json")
        mutate(run_copy, "receipt.json", lambda d: d.update(intent_sha256=intent_sha))
        failures = audit_mod.s2_identity(audit_mod.load_evidence(run_copy))
        assert any("v7_strict_receipt_sha256" in f for f in failures)

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
            diag["v5"]["decisions"] += 1
            diag["v5"]["kept"] += 1

        assert "S4" in self._watch_counters(run_copy, change)

    def test_s4_diagnostics_must_mirror_counters(self, run_copy):
        def change(c, diag, d):
            diag["kept"] -= 1
            diag["no_spacious"] += 1  # partition holds, mirror and v5 tie-out do not

        assert "S4" in self._watch_counters(run_copy, change)
        failures = audit_mod.s4_counters(audit_mod.load_evidence(run_copy))
        assert any("mirror_kept" in f for f in failures)

    def test_s4_diagnostics_total_must_be_the_aggregate(self, run_copy):
        mutate(run_copy, WATCH, lambda d: d["veto"]["diagnostics_total"].update(area_cap_max=1))
        assert "S4" in failing(audit_mod.audit(run_copy))

    def test_s4_play_must_have_no_wrapped_decisions(self, run_copy):
        mutate(run_copy, PLAY, lambda d: d["decision_frames"].update(total=3))
        assert "S4" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d.update(first_divergence_frame=5),
            lambda d: d.update(diagnostics_equal=False),
            lambda d: d["rollout"]["diagnostics"].update(area_evaluations=999),
            lambda d: d["rollout"]["diagnostics"]["v5"].update(landing_checks=999),
            lambda d: d["rollout"].update(wrapper_method="free-space-veto/v5-boost-aware"),
            lambda d: d["session"].update(variant="v5"),
            lambda d: d.update(env={}),
        ],
    )
    def test_s5_rules(self, run_copy, change):
        mutate(run_copy, PARITY, change)
        assert "S5" in failing(audit_mod.audit(run_copy))

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d.update(variant_from_empty_env="v7"),
            lambda d: d.update(variant_from_empty_env="v2"),
            lambda d: d["builds"]["empty_env"]["modes"]["watch"].update(method="x"),
            lambda d: d["builds"]["variant_v5"]["modes"]["watch"].update(variant="v7"),
            lambda d: d["builds"]["variant_v2"]["modes"]["watch"].update(
                method="free-space-veto/v5-boost-aware"
            ),
            lambda d: d["builds"]["variant_v2"]["modes"]["play"].update(snakes_with_veto=[1]),
            lambda d: d["builds"]["watch_off_variant_v7"]["modes"]["watch"].update(active=True),
            lambda d: d["builds"].pop("variant_v2"),
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
        assert "S1" in failing(result)  # design/preregistration

    def test_watch_activity_counts_a_rerank_change(self, run_copy):
        ev = audit_mod.load_evidence(run_copy)
        if audit_mod.watch_activity(ev)["vetoes_applied"] == 0:
            assert any("never replaced" in f for f in audit_mod.s3_scope(ev, smoke=False))

        def rerank_of_kept(d):
            """A v5-kept base re-ranked to another direction (counters move kept->veto)."""
            sid = str(d["served"]["hero_id"])
            c, diag = d["veto"]["per_snake"][sid], d["veto"]["diagnostics_per_snake"][sid]
            c["kept_base"] -= 1
            c["vetoes_applied"] += 1
            diag["kept"] -= 1
            diag["vetoes_applied"] += 1
            diag["rerank_changes"] += 1
            diag["rerank_changes_of_v5_kept"] += 1
            diag["static_area_evaluations"] += 2
            if diag["rerank_scored"] < diag["rerank_changes"]:
                diag["rerank_scored"] += 1
                diag["rerank_pruned"] -= 1
            for key in ("vetoes_applied", "kept_base"):
                d["veto"]["total"][key] = c[key]
            d["veto"]["diagnostics_total"] = audit_mod.diagnostics_total({sid: diag})

        mutate(run_copy, WATCH, rerank_of_kept)
        ev = audit_mod.load_evidence(run_copy)
        activity = audit_mod.watch_activity(ev)
        assert activity["vetoes_applied"] >= 1 and activity["rerank_changes"] >= 1
        assert audit_mod.s3_scope(ev, smoke=False) == []
        assert audit_mod.s4_counters(ev) == []

    def test_bound_intent_has_no_preregistration_failures(self):
        intent = json.loads((SMOKE / "intent.json").read_text())
        intent.update(protocol_sha256=audit_mod.PROTOCOL_SHA256, git={"dirty": False})
        assert audit_mod.preregistration_failures(intent) == []
        intent["git"]["dirty"] = True
        assert audit_mod.preregistration_failures(intent) == ["intent.git.dirty is not false"]
