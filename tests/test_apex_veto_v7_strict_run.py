"""Tests for the v7 Tier-2 strict runner (research/apex_veto_v7_strict_20261002/strict_run.py).

Record-shape and self-check rules run on real record shapes (one complete frozen A/B pair of
the v7 Tier-1 screen, ``tests/fixtures/apex_veto_v7_strict``) and on mutated copies. No test
plays an episode: ``tournament_eval.rollout``, ``dev_screen.run_episode`` and
``strict_run.run_unit_episode`` raise in every test (one test drives the real
``run_unit_episode`` against a fake rollout to check the arm installs).
"""

from __future__ import annotations

import copy
import fcntl
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v7_strict_20261002 import strict_run as sr
from src.evaluation.strict_promotion import (
    StrictPromotionArtifactError,
    validate_strict_world_record,
)
from tests import test_apex_veto_v7_strict_audit as helpers

REAL_RUN_UNIT_EPISODE = sr.run_unit_episode
HAVE_PILOT = dev_screen.DEFAULT_PILOT_OUTPUT.is_dir()
HAVE_CHECKPOINTS = all(
    (dev_screen.DEFAULT_CHECKPOINT_DIR / n).is_file() for n, _ in dev_screen.POOL
)
REAL_RECEIPT = sr.SCREEN_RUN / "receipt.json"
block_episode_runners = helpers.block_episode_runners  # autouse fixture (re-exported)


@pytest.fixture(autouse=True)
def tier2_root(tmp_path, monkeypatch):
    """The pre-registered Tier-2 root is replaced by a tmp root in every test."""
    monkeypatch.setattr(sr, "TIER2_OUT_ROOT", tmp_path / "run-v1")
    return tmp_path / "run-v1"


@pytest.fixture
def profile(setup_config):
    """The pinned deployment profile, resolved from the pinned config (not from a record)."""
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    load_and_initialize_config(str(dev_screen.DEFAULT_CONFIG))
    resolved = evaluation_profile_for_name(sr.PROFILE_NAME, sr.HORIZON)
    assert resolved.digest == dev_screen.PROFILE_DIGEST
    return {"descriptor": resolved.descriptor(), "digest": resolved.digest}


@pytest.fixture
def wrappers() -> dict:
    return {arm: identity for arm, identity in sr.arm_identities().items()}


# ---------------------------------------------------------------- arms and identities


def veto_imports(rel: str) -> set:
    """``src.evaluation.safety_veto*`` modules a veto module imports (AST, any depth)."""
    import ast

    tree = ast.parse((sr.REPO / rel).read_text())
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("src.evaluation.safety_veto")
    }


def module_rel(name: str) -> str:
    return name.replace(".", "/") + ".py"


class TestArmIdentities:
    def test_incumbent_is_the_released_v5_and_candidate_binds_v7_closure(self, wrappers):
        inc, cand = wrappers["incumbent"], wrappers["candidate"]
        assert inc["method"] == "free-space-veto/v5-boost-aware"
        assert inc["source_sha256s"] == sr.INCUMBENT_SOURCE_SHA256S
        assert inc["source_sha256"] == sr.INCUMBENT_SOURCE_SHA256
        assert inc["source_sha256"].startswith("d86d084e")
        assert list(inc["source_sha256s"]) == [sr.V5_SOURCE, sr.V2_SOURCE, sr.V3_SOURCE]
        assert cand["method"] == "free-space-veto/v7-space-preference(lambda=4.0)"
        assert list(cand["source_sha256s"]) == [
            sr.V7_SOURCE,
            sr.V2_SOURCE,
            sr.V3_SOURCE,
            sr.V5_SOURCE,
        ]
        assert cand["source_sha256"] == cand["source_sha256s"][sr.V7_SOURCE]
        for rel in (sr.V2_SOURCE, sr.V3_SOURCE, sr.V5_SOURCE):
            assert cand["source_sha256s"][rel] == inc["source_sha256s"][rel]
        assert cand["descriptor"]["space_preference_lambda"] == 4.0
        assert "space_preference_lambda" not in inc["descriptor"]
        from src.evaluation.safety_veto_v7 import method_for

        assert method_for(sr.CANDIDATE_LAMBDA) == sr.CANDIDATE_METHOD

    def test_each_arm_binds_its_transitive_veto_imports(self):
        """Every safety_veto* module an arm's primary module reaches is in its identity."""
        for arm in sr.ARMS:
            seen, todo = set(), [sr.ARM_SOURCES[arm][0]]
            while todo:
                rel = todo.pop()
                if rel in seen:
                    continue
                seen.add(rel)
                todo.extend(module_rel(name) for name in veto_imports(rel))
            assert seen == set(sr.ARM_SOURCES[arm]), arm
        assert veto_imports(sr.V7_SOURCE) == {
            "src.evaluation.safety_veto",
            "src.evaluation.safety_veto_v3",
            "src.evaluation.safety_veto_v5",
        }

    def test_run_unit_episode_installs_each_arms_veto(self, monkeypatch):
        """The real episode runner against a fake rollout (no episode is played)."""
        from src.evaluation.safety_veto import FreeSpaceVeto
        from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto
        from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
        from src.scripts import tournament_eval

        calls = []

        def builtin(hero, spec):  # stands in for rollout's v2 install (vector61 guard)
            hero.safety_veto = FreeSpaceVeto()
            return hero.safety_veto

        def fake_rollout(hero, opponents, frames, seed, **kwargs):
            veto = tournament_eval._install_hero_safety_veto(hero, ("checkpoint", "x"))
            calls.append((kwargs.get("hero_safety_veto"), type(veto).__name__, frames))
            return {"probes": {"safety_veto": veto.record()}}

        monkeypatch.setattr(tournament_eval, "_install_hero_safety_veto", builtin)
        monkeypatch.setattr(tournament_eval, "rollout", fake_rollout)
        row = {"mix": "scripted", "world_seed": 5, "slots": [{"member_sha256": "o"}]}
        lookup = {sr.CHAMPION[1]: SimpleNamespace(safety_veto=None), "o": "opponent"}
        kinds = {"incumbent": BoostAwareFreeSpaceVeto, "candidate": SpacePreferenceVeto}
        for arm in sr.ARMS:
            unit = {"arm": arm}
            lookup[sr.CHAMPION[1]] = SimpleNamespace(safety_veto=None)
            record, diag = REAL_RUN_UNIT_EPISODE(unit, row, lookup, None, 40)
            assert record["probes"]["safety_veto"]["method"] == sr.ARM_METHODS[arm]
            assert isinstance(lookup[sr.CHAMPION[1]].safety_veto, kinds[arm])
            assert diag is not None and diag["decisions"] == 0
            assert sr.DIAGNOSTICS_MARKER[arm] in diag
        veto = lookup[sr.CHAMPION[1]].safety_veto
        assert veto.lam == 4.0
        assert calls == [(True, "BoostAwareFreeSpaceVeto", 40), (True, "SpacePreferenceVeto", 40)]
        assert tournament_eval._install_hero_safety_veto is builtin  # installer restored


# ---------------------------------------------------------------- namespaces and rosters


class TestNamespaces:
    def test_counts_unique_and_mutually_disjoint(self):
        seeds = sr.namespace_seeds()
        assert {k: len(v) for k, v in seeds.items()} == {"dev": 16, "final": 300, "serving": 50}
        everything = [s for values in seeds.values() for s in values]
        assert len(set(everything)) == len(everything)
        assert {d for d, _ in sr.NAMESPACES.values()} == {
            "apex-veto-v7-strict-dev-v1",
            "apex-veto-v7-strict-final-v1",
            "apex-veto-v7-strict-serving-v1",
        }

    @pytest.mark.skipif(not HAVE_PILOT, reason="strict pilot records not available")
    def test_fresh_against_every_checked_namespace(self):
        report = sr.namespace_report(sr.namespace_seeds(), dev_screen.DEFAULT_PILOT_OUTPUT)
        assert report["disjoint"] is True
        for row in report["namespaces"].values():
            assert row["earlier_namespaces"]["recipe_reproduces_pilot"] is True
            assert row["earlier_domain_overlap"] == {}
        assert report["earlier_domain_prefix_checked"] == 1000

    def test_fails_closed_without_pilot_records(self):
        assert sr.namespace_report(sr.namespace_seeds(), None)["disjoint"] is False

    @pytest.mark.skipif(not HAVE_PILOT, reason="strict pilot records not available")
    @pytest.mark.parametrize(
        "domain, purpose",
        [
            ("apex-veto-v7-screen-v1", "worlds"),
            ("apex-veto-v7-dev-v1", "worlds"),
            ("apex-veto-v6-screen-v1", "worlds"),
            ("apex-veto-v5-strict-final-v1", "worlds"),
            ("apex-veto-v5-web-serving-v1", "watch"),
            ("apex-veto-strict-final-v3", "worlds"),
            ("apex-veto-web-serving-v1", "parity"),
            ("trap-horizon-v5-dev-v1", "worlds"),
            ("apex-veto-v7-strict-smoke-v1", "worlds"),
        ],
    )
    def test_detects_earlier_domain_overlap(self, domain, purpose):
        seeds = copy.deepcopy(sr.namespace_seeds())
        seeds["final"][5] = dev_screen.uint32_seed(domain, purpose, 999)
        report = sr.namespace_report(seeds, dev_screen.DEFAULT_PILOT_OUTPUT)
        assert report["disjoint"] is False
        assert (
            f"{domain}/{purpose}[0:1000]" in report["namespaces"]["final"]["earlier_domain_overlap"]
        )

    def test_cross_namespace_overlap(self):
        seeds = copy.deepcopy(sr.namespace_seeds())
        seeds["serving"][0] = seeds["dev"][0]
        report = sr.namespace_report(seeds, None)
        assert report["cross_namespace_overlap"]["dev|serving"] == [seeds["dev"][0]]


class TestRostersAndShards:
    def test_prefix_rows_equal_full_bank_prefix(self):
        bank = sr.namespace_seeds()["final"]
        full = sr.build_rosters([], bank, [])["final"]
        prefix = sr.build_rosters([], bank[:97], [])["final"]
        for mix in sr.MIXES:
            assert [r for r in prefix if r["mix"] == mix] == [r for r in full if r["mix"] == mix][
                :97
            ]

    def test_stage_plans_and_deterministic_balanced_shards(self):
        seeds = sr.namespace_seeds()
        rosters = sr.build_rosters(seeds["dev"], seeds["final"][:7], seeds["serving"])
        calibration = sr.stage_plan("calibration", rosters["calibration"])
        final = sr.stage_plan("final", rosters["final"])
        serving = sr.stage_plan("serving", rosters["serving"])
        assert sum(len(u) for u in calibration) == 48
        assert {e["arm"] for u in calibration for e in u} == {"incumbent"}
        assert all([e["arm"] for e in u] == ["incumbent", "candidate"] for u in final)
        assert sum(len(u) for u in final) == 42 and len(serving) == 50
        assert {e["arm"] for u in serving for e in u} == {"candidate"}
        shards = [sr.shard_units(final, k) for k in range(sr.WORKERS)]
        ids = [e["episode_id"] for shard in shards for unit in shard for e in unit]
        assert sorted(ids) == sorted(e["episode_id"] for u in final for e in u)
        assert abs(len(shards[0]) - len(shards[1])) <= 1

    def test_serving_rows_are_selfplay_unwrapped_champions(self):
        rows = sr.serving_rows([7, 8])
        assert [r["mix"] for r in rows] == [sr.SERVING_MIX] * 2
        assert {s["member_sha256"] for r in rows for s in r["slots"]} == {sr.CHAMPION[1]}


# ---------------------------------------------------------------- pilot gate and sizing


@pytest.fixture(scope="module")
def pilot_root(tmp_path_factory) -> Path:
    return helpers.build_pilot(tmp_path_factory.mktemp("pilot") / "screen", delta_sd=10.0)


def copy_pilot(source: Path, target: Path) -> Path:
    import shutil

    shutil.copytree(source, target)
    return target


class TestPilot:
    def test_deltas_gate_and_floor_from_a_recommending_screen(self, pilot_root):
        pilot = sr.screen_pilot_deltas(pilot_root)
        assert pilot["summary_deltas_match"] is True
        assert {m: len(v) for m, v in pilot["deltas_by_mix"].items()} == {m: 40 for m in sr.MIXES}
        assert pilot["deltas_by_mix"] == helpers.pilot_deltas(pilot_root)
        assert not any(name.startswith("C-") for name in pilot["record_files_sha256"])
        gate = sr.screen_receipt_gate(pilot_root, pilot)
        assert gate["passes"] is True and gate["problems"] == []
        assert sr.frozen_final_n(pilot["deltas_by_mix"])["final_worlds_per_mix"] == sr.N_FLOOR
        binding = sr.lambda_binding_gate(pilot_root, helpers.sweep_of(pilot_root))
        assert binding["passes"] is True, binding["problems"]
        assert binding["screen_receipt_lambda"] == binding["sweep_selected_lambda"] == 4.0

    @pytest.mark.parametrize("tamper", sorted(helpers.LAMBDA_TAMPERS))
    def test_lambda_gate_refuses_any_other_lambda_binding(self, pilot_root, tmp_path, tamper):
        pilot_dir = copy_pilot(pilot_root, tmp_path / "screen")
        helpers.LAMBDA_TAMPERS[tamper](pilot_dir)
        binding = sr.lambda_binding_gate(pilot_dir, helpers.sweep_of(pilot_dir))
        assert binding["passes"] is False and binding["problems"]

    def test_lambda_gate_refuses_another_sweep_summary(self, pilot_root, tmp_path):
        other = helpers.build_sweep(tmp_path / "other-sweep")
        binding = sr.lambda_binding_gate(pilot_root, other)
        assert binding["passes"] is False
        assert any("pre-declared sweep summary" in p for p in binding["problems"])
        missing = sr.lambda_binding_gate(pilot_root, tmp_path / "none" / "summary.json")
        assert missing["passes"] is False

    @pytest.mark.parametrize("tamper", sorted(helpers.PILOT_GATE_TAMPERS))
    def test_gate_refuses_anything_but_a_passing_recommendation(self, pilot_root, tmp_path, tamper):
        pilot_dir = copy_pilot(pilot_root, tmp_path / "screen")
        helpers.PILOT_GATE_TAMPERS[tamper](pilot_dir)
        gate = sr.screen_receipt_gate(pilot_dir, sr.screen_pilot_deltas(pilot_dir))
        assert gate["passes"] is False and gate["problems"]

    def test_gate_refuses_a_missing_pair_and_summary_mismatch(self, pilot_root, tmp_path):
        pilot_dir = copy_pilot(pilot_root, tmp_path / "a")
        next((pilot_dir / "records").glob("B-scripted-*.json")).unlink()
        gate = sr.screen_receipt_gate(pilot_dir, sr.screen_pilot_deltas(pilot_dir))
        assert any("pairs per mix" in p for p in gate["problems"])
        pilot_dir = copy_pilot(pilot_root, tmp_path / "b")
        helpers.edit_json(
            pilot_dir / "summary.json",
            lambda d: d["per_mix"]["mixed"]["primary_mass_integral"]["deltas_B_minus_A"].pop(),
        )
        pilot = sr.screen_pilot_deltas(pilot_dir)
        assert pilot["summary_deltas_match"] is False
        assert sr.screen_receipt_gate(pilot_dir, pilot)["passes"] is False

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda e: e.update(safety_veto_method="free-space-veto/v5-boost-aware"), "veto"),
            (
                lambda e: e["record"]["probes"]["safety_veto"].update(space_preference_lambda=2.0),
                "lambda",
            ),
            (lambda e: e["record"]["probes"]["safety_veto"].update(method="x"), "veto"),
            (lambda e: e.update(hero_sha256="0" * 64), "hero"),
            (lambda e: e.update(world_index=e["world_index"] + 1), "recipe"),
        ],
    )
    def test_screen_record_identity_is_checked(self, pilot_root, tmp_path, mutate, message):
        pilot_dir = copy_pilot(pilot_root, tmp_path / "screen")
        helpers.edit_json(next((pilot_dir / "records").glob("B-frozen-*.json")), mutate)
        with pytest.raises(sr.StrictRunError, match=message):
            sr.screen_pilot_deltas(pilot_dir)

    def test_floor_and_kill_criterion(self):
        tight = {m: [20.0 + (i % 2) for i in range(10)] for m in sr.MIXES}
        assert sr.frozen_final_n(tight)["final_worlds_per_mix"] == sr.N_FLOOR
        wide = {m: [(-1) ** i * 150.0 for i in range(10)] for m in sr.MIXES}
        sizing = sr.frozen_final_n(wide)
        assert sizing["final_worlds_per_mix"] <= sr.N_MAX  # SD 150 fits at MDE 40
        wider = {m: [(-1) ** i * 300.0 for i in range(10)] for m in sr.MIXES}
        sizing = sr.frozen_final_n(wider)
        assert sizing["final_worlds_per_mix"] > sr.N_MAX and sizing["feasible"] is False
        assert sizing["mde_absolute_per_mix"] == 40.0

    def test_runtime_projection_uses_screen_wall_times(self, pilot_root):
        walls = sr.screen_pilot_deltas(pilot_root)["episode_wall_seconds"]
        assert {k: len(v) for k, v in walls.items()} == {
            f"{a}-{m}": 40 for a in ("A", "B") for m in sr.MIXES
        }
        projection = sr.runtime_projection(walls, 300)
        assert projection["basis"].startswith("v7 screen")
        assert projection["within_limit"] is True  # ~25 s fixture episodes
        walls = {k: [400.0] * len(v) for k, v in walls.items()}
        assert sr.runtime_projection(walls, 300)["within_limit"] is False

    @pytest.mark.skipif(not REAL_RECEIPT.is_file(), reason="v7 screen receipt not available")
    def test_real_screen_gates_size_n_137_within_the_runtime_limit(self):
        """Read-only: the closed v7 screen passes every gate and sizes N = 137 at MDE 40."""
        receipt = json.loads(REAL_RECEIPT.read_text())
        pilot = sr.screen_pilot_deltas(sr.SCREEN_RUN)
        gate = sr.screen_receipt_gate(sr.SCREEN_RUN, pilot)
        recommends = receipt["decision"] == sr.PILOT_REQUIRED_DECISION
        assert gate["passes"] is (recommends and receipt["self_check"]["passes"] is True)
        assert gate["passes"] is True
        assert sr.lambda_binding_gate(sr.SCREEN_RUN)["passes"] is True
        parity = sr.screen_source_parity(sr.SCREEN_RUN, sr.arm_identities())
        assert parity["passes"] is True and parity["screen_commit"] == sr.PILOT_COMMIT
        sizing = sr.frozen_final_n(pilot["deltas_by_mix"])
        assert sizing["final_worlds_per_mix"] == 137 and sizing["feasible"] is True
        projection = sr.runtime_projection(pilot["episode_wall_seconds"], 137)
        assert projection["within_limit"] is True
        assert projection["fraction_of_worker_budget"] < 0.30


# ---------------------------------------------------------------- record shape (real records)


def screen_row(letter: str = "A") -> dict:
    entry = helpers.SCREEN[letter]
    return {
        "mix": entry["mix"],
        "world_seed": entry["world_seed"],
        "slots": [
            {"slot": i + 1, "member_sha256": h}
            for i, h in enumerate(entry["record"]["world_identity"]["ordered_slot_content_hashes"])
        ],
    }


class TestRecordShape:
    def test_real_records_pass_with_their_own_arm_descriptor(self, profile, wrappers):
        for letter, arm in (("A", "incumbent"), ("B", "candidate")):
            record = helpers.SCREEN[letter]["record"]
            validate_strict_world_record(
                record, profile, screen_row(), candidate_wrapper=wrappers[arm]["descriptor"]
            )
            other = wrappers["candidate" if arm == "incumbent" else "incumbent"]
            with pytest.raises(StrictPromotionArtifactError):
                validate_strict_world_record(
                    record, profile, screen_row(), candidate_wrapper=other["descriptor"]
                )

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda r: r["probes"]["safety_veto"].update(method="free-space-veto/v1"), "identity"),
            (lambda r: r["probes"]["safety_veto"]["counters"].update(kept_base=0), "inconsistent"),
            (lambda r: r["probes"]["safety_veto"]["counters"].update(extra=1), "extra"),
            (lambda r: r["probes"]["safety_veto"].pop("landing_rule"), "missing"),
            (lambda r: r["probes"]["safety_veto"].pop("area_rule"), "missing"),
            (lambda r: r["probes"].pop("safety_veto"), "lacks its wrapper"),
            (lambda r: r.update(survival_fraction=0.5), "survival_fraction"),
        ],
    )
    def test_mutated_candidate_rejected(self, profile, wrappers, mutate, message):
        record = copy.deepcopy(helpers.SCREEN["B"]["record"])
        mutate(record)
        with pytest.raises(StrictPromotionArtifactError, match=message):
            validate_strict_world_record(
                record,
                profile,
                screen_row(),
                candidate_wrapper=wrappers["candidate"]["descriptor"],
            )


# ---------------------------------------------------------------- self-check audit rules


def real_entries(stage: str, wrappers: dict) -> tuple:
    """Producer envelopes (``episode_entry``) around the real records, on two strict worlds."""
    bank_name = {"calibration": "dev", "final": "final", "serving": "serving"}[stage]
    bank = sr.namespace_seeds()[bank_name]
    seeds = bank[:2]
    if stage == "serving":
        rows = sr.serving_rows(seeds)
    else:
        rows = [r for r in sr.build_rosters(bank, bank, [])[stage] if r["world_seed"] in seeds]
        rows = [r for r in rows if r["mix"] in ("frozen", "mixed")]
    index = {seed: i for i, seed in enumerate(bank)}
    entries = {}
    for unit in sr.stage_plan(stage, rows):
        for episode in unit:
            seed, mix = episode["world_seed"], episode["mix"]
            letter = helpers.LETTER[episode["arm"]]
            record = helpers.move_record(letter, mix, index[seed], seed)
            if mix == sr.SERVING_MIX:
                from src.evaluation.strict_promotion import _expected_world_identity

                row = next(r for r in rows if r["world_seed"] == seed)
                record["world_identity"] = _expected_world_identity(row)
            row = next(r for r in rows if r["world_seed"] == seed and r["mix"] == mix)
            diag = helpers.DIAG[episode["arm"]]
            entries[episode["episode_id"]] = json.loads(
                json.dumps(
                    sr.episode_entry(episode, row, record, wrappers, 0, 1.0, index[seed], diag)
                )
            )
    return entries, rows, index


class TestSelfCheckRules:
    @pytest.mark.parametrize("stage", ["calibration", "final", "serving"])
    def test_real_entries_pass_and_report_veto_consistency(self, profile, wrappers, stage):
        entries, rows, index = real_entries(stage, wrappers)
        result = sr.audit_stage_entries(stage, entries, rows, profile, wrappers, index)
        assert result["failures"] == [], result["failures"]
        counters = result["reported_not_gated"]["veto_counters"]
        arms = {e["arm"] for e in entries.values()}
        for arm in arms:
            n = sum(1 for e in entries.values() if e["arm"] == arm)
            assert counters[arm] == {"records_with_veto": n, "decisions_equal_decision_frames": n}

    @pytest.mark.parametrize(
        "arm, mutate, fragment",
        [
            ("candidate", lambda e: e.update(wrapper=sr.INCUMBENT_METHOD), "envelope"),
            ("candidate", lambda e: e.update(safety_veto=False), "envelope"),
            ("candidate", lambda e: e.pop("veto_diagnostics"), "envelope"),
            (
                "candidate",
                lambda e: e.update(wrapper_source_sha256=sr.INCUMBENT_SOURCE_SHA256),
                "envelope",
            ),
            ("candidate", lambda e: e["wrapper_source_sha256s"].popitem(), "envelope"),
            ("incumbent", lambda e: e.update(veto_diagnostics={}), "envelope"),
            ("incumbent", lambda e: e.pop("veto_diagnostics"), "envelope"),
            (
                "incumbent",
                lambda e: e.update(veto_diagnostics=helpers.DIAG["candidate"]),
                "envelope",
            ),
            (
                "candidate",
                lambda e: e.update(veto_diagnostics=helpers.DIAG["incumbent"]),
                "envelope",
            ),
            (
                "candidate",
                lambda e: e["record"]["probes"]["safety_veto"].update(space_preference_lambda=2.0),
                "record shape",
            ),
            ("incumbent", lambda e: e.update(world_index=3), "envelope"),
            ("incumbent", lambda e: e["record"]["probes"].pop("safety_veto"), "record shape"),
            ("candidate", lambda e: e["record"].update(mass_integral=-1.0), "record shape"),
        ],
    )
    def test_mutated_envelopes_fail(self, profile, wrappers, arm, mutate, fragment):
        entries, rows, index = real_entries("final", wrappers)
        key = next(k for k in sorted(entries) if f"-{arm}-" in k)
        mutate(entries[key])
        failures = sr.audit_stage_entries("final", entries, rows, profile, wrappers, index)[
            "failures"
        ]
        assert failures and fragment in failures[0] and key in failures[0]

    def test_swapped_arm_probes_fail(self, profile, wrappers):
        entries, rows, index = real_entries("final", wrappers)
        inc = next(k for k in sorted(entries) if "-incumbent-" in k)
        cand = inc.replace("-incumbent-", "-candidate-")
        a, b = entries[inc]["record"]["probes"], entries[cand]["record"]["probes"]
        a["safety_veto"], b["safety_veto"] = b["safety_veto"], a["safety_veto"]
        failures = sr.audit_stage_entries("final", entries, rows, profile, wrappers, index)[
            "failures"
        ]
        assert any(inc in f and "record shape" in f for f in failures)
        assert any(cand in f and "record shape" in f for f in failures)

    def test_smoke_mode_checks_the_probe_method(self, wrappers):
        entries, rows, index = real_entries("final", wrappers)
        assert (
            sr.audit_stage_entries("final", entries, rows, None, wrappers, index)["failures"] == []
        )
        key = next(k for k in sorted(entries) if "-incumbent-" in k)
        entries[key]["record"]["probes"]["safety_veto"]["method"] = sr.CANDIDATE_METHOD
        failures = sr.audit_stage_entries("final", entries, rows, None, wrappers, index)["failures"]
        assert failures and "incumbent's veto" in failures[0]

    def test_episode_entry_shape(self, wrappers):
        unit = {
            "episode_id": "e",
            "stage": "final",
            "arm": "incumbent",
            "mix": "x",
            "world_seed": 1,
        }
        row = {"slots": [{"member_sha256": "a"}]}
        inc = sr.episode_entry(unit, row, {}, wrappers, 1, 2.0, 7, {"boost_landing_vetoes": 0})
        assert inc["safety_veto"] is True and inc["wrapper"] == sr.INCUMBENT_METHOD
        assert inc["veto_diagnostics"] == {"boost_landing_vetoes": 0}
        assert inc["wrapper_source_sha256s"] == sr.INCUMBENT_SOURCE_SHA256S
        assert sr.episode_entry(unit, row, {}, wrappers, 1, 2.0, 7)["veto_diagnostics"] is None
        cand = sr.episode_entry(dict(unit, arm="candidate"), row, {}, wrappers, 1, 2.0, 7, {})
        assert cand["wrapper_source_sha256s"] == wrappers["candidate"]["source_sha256s"]
        assert cand["veto_diagnostics"] == {}


# ---------------------------------------------------------------- stage evidence integrity


def write_stage(stage_dir: Path, stage: str, entries: dict, rows: list, drop=()) -> None:
    """Lay out real envelopes the way two workers write them."""
    (stage_dir / "records").mkdir(parents=True)
    units = sr.stage_plan(stage, rows)
    for shard in range(sr.WORKERS):
        planned = [e["episode_id"] for u in sr.shard_units(units, shard) for e in u]
        done = {}
        for episode_id in planned:
            if episode_id in drop:
                continue
            done[episode_id] = sr.write_durable(
                stage_dir / "records" / f"{episode_id}.json", entries[episode_id]
            )
        (stage_dir / f"shard-{shard}").mkdir()
        sr.write_durable(
            stage_dir / f"shard-{shard}" / "report.json",
            {
                "planned_episode_ids": planned,
                "records_sha256": done,
                "complete": len(done) == len(planned),
                "stopped_reason": "deadline: 10s left < 45s budget" if drop else None,
            },
        )


class TestCollectStage:
    def test_complete_tampered_extra_and_missing(self, tmp_path, wrappers):
        entries, rows, _ = real_entries("final", wrappers)
        write_stage(tmp_path / "ok", "final", entries, rows)
        collected = sr.collect_stage(tmp_path / "ok", "final", rows)
        assert collected["complete"] and collected["entries"] == entries
        write_stage(tmp_path / "a", "final", entries, rows)
        path = next((tmp_path / "a" / "records").glob("final-candidate-*.json"))
        path.write_text(path.read_text().replace('"mass_integral": ', '"mass_integral": 1'))
        with pytest.raises(sr.StrictRunError, match="bytes changed"):
            sr.collect_stage(tmp_path / "a", "final", rows)
        write_stage(tmp_path / "b", "final", entries, rows)
        (tmp_path / "b" / "records" / "final-candidate-mixed-1.json").write_text("{}")
        with pytest.raises(sr.StrictRunError, match="differ from shard reports"):
            sr.collect_stage(tmp_path / "b", "final", rows)
        dropped = sorted(entries)[0]
        write_stage(tmp_path / "c", "final", entries, rows, drop={dropped})
        collected = sr.collect_stage(tmp_path / "c", "final", rows)
        assert collected["complete"] is False and collected["missing"] == [dropped]


# ---------------------------------------------------------------- calibration and decision


def synthetic_final(wrappers: dict, shift: float, n: int = 12):
    """Final-stage envelopes over n worlds per mix from the real record shapes."""
    seeds = list(range(1000, 1000 + n))
    rows = [
        {"mix": mix, "world_seed": seed, "slots": [{"slot": 1, "member_sha256": sr.CHAMPION[1]}]}
        for mix in sr.MIXES
        for seed in seeds
    ]
    entries = {}
    for row in rows:
        for arm, extra in (("incumbent", 0.0), ("candidate", shift)):
            record = copy.deepcopy(helpers.SCREEN[helpers.LETTER[arm]]["record"])
            wobble = ((row["world_seed"] * 7919) % 13) - 6.0
            record["mass_integral"] = 40.0 + wobble * (1.0 if arm == "candidate" else 0.5) + extra
            record["survival_fraction"] = 0.32
            unit = {
                "episode_id": f"final-{arm}-{row['mix']}-{row['world_seed']}",
                "stage": "final",
                "arm": arm,
                "mix": row["mix"],
                "world_seed": row["world_seed"],
            }
            entries[unit["episode_id"]] = sr.episode_entry(unit, row, record, wrappers, 0, 1.0, 0)
    return entries, rows


def calibration_entries() -> dict:
    """Two incumbent (v5) worlds per mix from the real incumbent record shape."""
    entries = {}
    for mix in sr.MIXES:
        for k, mass in enumerate((30.0, 50.0)):
            record = dict(helpers.SCREEN["A"]["record"], mass_integral=mass)
            entries[f"calibration-incumbent-{mix}-{k}"] = {
                "mix": mix,
                "arm": "incumbent",
                "record": record,
            }
    return entries


class TestCalibrationAndDecision:
    def test_calibration_margin_and_bands(self):
        entries = calibration_entries()
        calibration = sr.calibration_reference(entries)
        assert calibration["absolute_delta_ni"] == pytest.approx(0.03 * 40.0, abs=1e-12)
        band = calibration["survival_bands"]["mixed"]
        assert band["lower"] == pytest.approx(0.3116 - 0.02)
        assert band["upper"] == pytest.approx(0.3116 + 1)
        assert sr.audit_calibration(entries, calibration) == []
        tampered = copy.deepcopy(calibration)
        tampered["absolute_delta_ni"] *= 1.01
        assert "absolute_delta_ni differs" in sr.audit_calibration(entries, tampered)

    @pytest.mark.parametrize("shift, passes", [(25.0, True), (0.0, False)])
    def test_producer_and_audit_agree(self, wrappers, shift, passes):
        entries, rows = synthetic_final(wrappers, shift)
        calibration = sr.calibration_reference(calibration_entries())
        decision = sr.final_decision(sr.paired_final(entries, rows), calibration)
        assert decision["passes"] is passes and decision["valid"] is True
        audited = sr.audit_final(entries, rows, calibration, decision)
        assert audited["problems"] == [] and audited["passes"] is passes
        lying = dict(decision, passes=not passes)
        assert (
            "audit pass/fail differs from the producer decision"
            in sr.audit_final(entries, rows, calibration, lying)["problems"]
        )

    def test_band_out_of_range_fails_strict(self, wrappers):
        entries, rows = synthetic_final(wrappers, 25.0)
        calibration = sr.calibration_reference(calibration_entries())
        calibration["survival_bands"]["frozen"]["lower"] = 0.99
        decision = sr.final_decision(sr.paired_final(entries, rows), calibration)
        assert decision["strict_promotion_decision"]["passes"] is True
        assert decision["bands_pass"] is False and decision["passes"] is False

    def test_outcome_table(self):
        ok = {"valid": True, "passes": True}
        kwargs = dict(
            feasible=True,
            deadline_stop=False,
            failure=None,
            audit_passed=True,
            decision=ok,
            decisions_agree=True,
        )
        assert sr.classify_outcome(**kwargs) == "STRICT_PASS"
        assert sr.classify_outcome(**{**kwargs, "decision": {"valid": True}}) == "STRICT_FAIL"
        assert sr.classify_outcome(**{**kwargs, "feasible": False}) == "STOP_INFEASIBLE"
        assert sr.classify_outcome(**{**kwargs, "deadline_stop": True}) == "INCOMPLETE"
        for bad in ({"failure": "x"}, {"audit_passed": False}, {"decisions_agree": False}):
            assert sr.classify_outcome(**{**kwargs, **bad}) == "INVALID_STOP"


# ---------------------------------------------------------------- supervision and slots


def py(code: str) -> list:
    return [sys.executable, "-c", code]


class TestSupervision:
    def run(self, tmp_path, commands, **kwargs):
        kwargs.setdefault("poll_seconds", 0.05)
        kwargs.setdefault("grace_seconds", 2.0)
        return sr.supervise_children(
            commands,
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            logs=[tmp_path / f"c{i}.log" for i in range(len(commands))],
            heartbeats=kwargs.pop("heartbeats", [None] * len(commands)),
            **kwargs,
        )

    def test_clean_failed_and_watchdogs(self, tmp_path):
        for name in ("a", "b", "c", "d"):
            (tmp_path / name).mkdir()
        result = self.run(
            tmp_path / "a", [py("pass"), py("import time; time.sleep(0.3)")], wall_seconds=30
        )
        assert result["cause"] is None and sr.clean_supervision(result)
        failed = self.run(
            tmp_path / "b",
            [py("raise SystemExit(3)"), py("import time; time.sleep(60)")],
            wall_seconds=60,
        )
        assert failed["cause"] == "child_failed" and not sr.clean_supervision(failed)
        sleeper = py("import time; time.sleep(60)")
        assert self.run(tmp_path / "c", [sleeper], wall_seconds=0.5)["cause"] == "wall_timeout"
        stale = self.run(
            tmp_path / "d",
            [sleeper],
            wall_seconds=30,
            heartbeats=[tmp_path / "d" / "never.json"],
            heartbeat_stale_seconds=0.5,
        )
        assert stale["cause"] == "heartbeat_stale"

    def test_heartbeat_uses_monotonic_clock_not_wall_mtime(self, tmp_path):
        beat = tmp_path / "beat.json"
        code = (
            "import json, os, time\n"
            f"p = {str(beat)!r}\n"
            "for _ in range(15):\n"
            "    open(p + '.tmp', 'w').write(json.dumps({'mono': time.monotonic()}))\n"
            "    os.replace(p + '.tmp', p)\n"
            "    old = time.time() - 86400\n"
            "    os.utime(p, (old, old))\n"
            "    time.sleep(0.1)\n"
        )
        result = self.run(
            tmp_path, [py(code)], wall_seconds=30, heartbeats=[beat], heartbeat_stale_seconds=0.5
        )
        assert result["cause"] is None
        frozen = tmp_path / "frozen.json"
        frozen.write_text(json.dumps({"mono": time.monotonic() - 3600}))
        (tmp_path / "s").mkdir()
        stale = self.run(
            tmp_path / "s",
            [py("import time; time.sleep(60)")],
            wall_seconds=30,
            heartbeats=[frozen],
            heartbeat_stale_seconds=0.5,
        )
        assert stale["cause"] == "heartbeat_stale"
        assert sr._heartbeat_mono(tmp_path / "missing.json", 7.0) == 7.0

    def test_run_slots_all_or_nothing_and_inherited(self, tmp_path):
        busy = (tmp_path / "cpu-slot-2.lock").open("a+")
        fcntl.flock(busy.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(sr.StrictRunError, match="CPU slots unavailable"):
                sr.acquire_run_slots(tmp_path, timeout=0.3)
        finally:
            busy.close()
        slots = sr.acquire_run_slots(tmp_path, timeout=0.3)
        try:
            fds = [handle.fileno() for handle in slots]
            for k, fd in enumerate(fds):
                sr.assert_inherited_slot(fd, tmp_path, k + 1)
            with pytest.raises(sr.StrictRunError, match="is not cpu-slot-2.lock"):
                sr.assert_inherited_slot(fds[0], tmp_path, 2)
            flock = "import fcntl; fcntl.flock({fd}, fcntl.LOCK_EX | fcntl.LOCK_NB)"
            result = self.run(
                tmp_path,
                [py(flock.format(fd=fd)) for fd in fds],
                wall_seconds=30,
                pass_fds=[(fd,) for fd in fds],
            )
            assert sr.clean_supervision(result), result
            with pytest.raises(sr.StrictRunError, match="CPU slots unavailable"):
                sr.acquire_run_slots(tmp_path, timeout=0.3)
        finally:
            sr.release_run_slots(slots)


# ---------------------------------------------------------------- intent


NEEDS_REAL = pytest.mark.skipif(
    not (HAVE_PILOT and HAVE_CHECKPOINTS and sr.SUPERVISOR_SOURCE.is_file()),
    reason="strict pilot, checkpoints or supervisor helpers not available",
)


def real_intent(tmp_path: Path, screen_run: Path, **overrides) -> dict:
    kwargs = dict(
        out_root=tmp_path / "run-v1",
        deadline=datetime.now(timezone.utc) + timedelta(hours=15),
        authorization_quote="i agree, continue",
        screen_run=screen_run,
        sweep_summary=helpers.sweep_of(screen_run),
        slot_lock_root=tmp_path / "locks",
        allow_dirty_source=True,
    )
    kwargs.update(overrides)
    return sr.build_intent(**kwargs)


class TestIntent:
    @NEEDS_REAL
    def test_intent_binds_arms_gate_and_frozen_n(self, tmp_path, pilot_root):
        intent = real_intent(tmp_path, pilot_root)
        assert intent["study_id"] == "apex-veto-v7-strict-20261002"
        assert intent["sizing"]["final_worlds_per_mix"] == 40
        assert intent["final_seeds"] == sr.namespace_seeds()["final"][:40]
        assert intent["namespace_report"]["disjoint"] is True
        assert intent["roster_parity"]["compared"] == 48
        assert intent["pilot"]["screen_receipt_gate"]["passes"] is True
        assert intent["pilot"]["lambda_binding"]["passes"] is True
        assert intent["pilot"]["screen_source_parity"]["screen_commit"] == sr.PILOT_COMMIT
        assert intent["candidate_lambda"] == 4.0
        assert intent["incumbent_released_source_sha256s"] == sr.INCUMBENT_SOURCE_SHA256S
        assert "episode_wall_seconds" not in intent["pilot"]
        inc, cand = intent["incumbent"], intent["candidate"]
        assert inc["wrapper"] == sr.INCUMBENT_METHOD and cand["wrapper"] == sr.CANDIDATE_METHOD
        assert inc["wrapper_source_sha256"] == sr.INCUMBENT_SOURCE_SHA256
        assert inc["wrapper_identity"]["source_sha256s"] == sr.INCUMBENT_SOURCE_SHA256S
        assert cand["wrapper_identity"]["descriptor"]["space_preference_lambda"] == 4.0
        assert inc["checkpoint_sha256"] == cand["checkpoint_sha256"] == sr.CHAMPION[1]
        files = intent["source_closure"]["files"]
        for arm in (inc, cand):
            for rel, digest in arm["wrapper_identity"]["source_sha256s"].items():
                assert files[str((sr.REPO / rel).resolve())] == digest
        design = intent["design"]
        assert design["mde_absolute_per_mix"] == 40.0
        assert design["v7_screen_complete_before_mde"] is True
        assert design["mde_chosen_after_v7_screen"] is True
        assert "N = 547" in design["mde_basis"]
        assert intent["caps"]["stage_seconds"] == {
            "calibration": 1800,
            "final": 45000,
            "serving": 3600,
            "audit": 900,
        }
        assert intent["caps"]["final_runtime_projection"]["within_limit"] is True
        assert intent["serving_path_qualified"] is False
        assert intent["serving_stage_exercises_web_path"] is False
        command = intent["audit"]["independent_command"]
        assert str(sr.INDEPENDENT_AUDIT) in command
        assert command[command.index("--pilot-root") + 1] == str(pilot_root.resolve())
        sweep = str(helpers.sweep_of(pilot_root).resolve())
        assert command[command.index("--sweep-summary") + 1] == sweep
        sr.validate_intent(intent)
        path = sr.prepare(intent)
        assert path == tmp_path / "run-v1" / "intent.json"
        with pytest.raises(sr.StrictRunError, match="already exists"):
            sr.prepare(intent)

    @NEEDS_REAL
    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda i: i["caps"]["stage_seconds"].update(final=40000), "stage caps"),
            (lambda i: i["caps"].update(retry_authorized=True), "retry"),
            (lambda i: i["candidate"]["wrapper_identity"].update(method="x"), "wrapper"),
            (lambda i: i["incumbent"]["wrapper_identity"]["source_sha256s"].clear(), "wrapper"),
            (lambda i: i["incumbent"].update(wrapper_source_sha256="0" * 64), "released v5"),
            (
                lambda i: i["candidate"]["wrapper_identity"]["descriptor"].update(
                    space_preference_lambda=2.0
                ),
                "wrapper identity drift",
            ),
            (lambda i: i["candidate"].update(wrapper="x"), "lambda=4.0"),
            (lambda i: i["pilot"]["lambda_binding"].update(passes=False), "lambda_binding"),
            (lambda i: i["pilot"]["lambda_binding"].update(sweep_selected_lambda=2.0), "lambda"),
            (lambda i: i["protocol"].update(sha256="0" * 64), "protocol"),
            (lambda i: i["source_closure"]["files"].update({"/nonexistent.py": "0" * 64}), "drift"),
            (lambda i: i.update(output_root="/tmp/elsewhere/run-v1"), "pre-registered root"),
            (lambda i: i.update(smoke_frames=40), "smoke dry-run may not use"),
        ],
    )
    def test_validate_intent_rejects(self, tmp_path, pilot_root, mutate, message):
        intent = real_intent(tmp_path, pilot_root)
        mutate(intent)
        with pytest.raises(sr.StrictRunError, match=message):
            sr.validate_intent(intent)

    @NEEDS_REAL
    @pytest.mark.parametrize("tamper", ["not_advanced", "self_check_failed", "receipt_missing"])
    def test_prepare_refuses_without_a_recommending_screen(self, tmp_path, pilot_root, tamper):
        pilot_dir = copy_pilot(pilot_root, tmp_path / "screen")
        helpers.PILOT_GATE_TAMPERS[tamper](pilot_dir)
        with pytest.raises(sr.StrictRunError, match="may not size this study"):
            real_intent(tmp_path, pilot_dir)
        assert not (tmp_path / "run-v1").exists()

    @NEEDS_REAL
    @pytest.mark.parametrize("tamper", ["receipt_lambda", "sweep_selected", "sweep_commit"])
    def test_prepare_refuses_a_screen_not_bound_to_lambda_4(self, tmp_path, pilot_root, tamper):
        pilot_dir = copy_pilot(pilot_root, tmp_path / "screen")
        helpers.LAMBDA_TAMPERS[tamper](pilot_dir)
        with pytest.raises(sr.StrictRunError, match="lambda is not bound to 4.0"):
            real_intent(tmp_path, pilot_dir)
        assert not (tmp_path / "run-v1").exists()

    @NEEDS_REAL
    @pytest.mark.parametrize(
        "rel", ["src/evaluation/safety_veto_v5.py", "src/evaluation/safety_veto_v3.py"]
    )
    def test_prepare_refuses_an_unreleased_incumbent_source(
        self, tmp_path, pilot_root, monkeypatch, rel
    ):
        monkeypatch.setitem(sr.INCUMBENT_SOURCE_SHA256S, rel, "0" * 64)
        with pytest.raises(sr.StrictRunError, match="released v5 sources"):
            real_intent(tmp_path, pilot_root)

    @NEEDS_REAL
    def test_smoke_intent_needs_no_screen_and_uses_its_own_world(self, tmp_path):
        intent = real_intent(
            tmp_path, tmp_path / "no-screen", out_root=tmp_path / "smoke", smoke_frames=500
        )
        seed = dev_screen.uint32_seed(sr.SMOKE_DOMAIN, "worlds", 0)
        assert intent["mixes"] == ["scripted"] and intent["final_seeds"] == [seed]
        assert intent["namespaces"]["final"]["domain"] == sr.SMOKE_DOMAIN
        assert intent["audit"]["independent_command"][-1] == "--smoke"
        sr.validate_intent(intent)

    def test_construction_failures(self, tmp_path, pilot_root, monkeypatch):
        with pytest.raises(sr.StrictRunError, match="deadline"):
            real_intent(
                tmp_path, pilot_root, deadline=datetime.now(timezone.utc) + timedelta(hours=2)
            )
        with pytest.raises(sr.StrictRunError, match="forbidden"):
            real_intent(tmp_path, pilot_root, out_root=tmp_path / "ongoing-research-20260913")
        with pytest.raises(sr.StrictRunError, match="authorization"):
            real_intent(tmp_path, pilot_root, authorization_quote="  ")
        with pytest.raises(sr.StrictRunError, match="pre-registered root"):
            real_intent(tmp_path, pilot_root, out_root=tmp_path / "elsewhere")
        for smoke_root in (tmp_path / "run-v1", tmp_path / "other" / "run-v2"):
            with pytest.raises(sr.StrictRunError, match="smoke dry-run may not use"):
                real_intent(tmp_path, pilot_root, out_root=smoke_root, smoke_frames=40)
        with pytest.raises(sr.StrictRunError, match="500 frames"):
            real_intent(tmp_path, pilot_root, out_root=tmp_path / "smoke", smoke_frames=501)
        if HAVE_CHECKPOINTS:
            monkeypatch.setattr(
                sr, "source_closure", lambda extra=(): {"dirty": [" M src/x.py"], "files": {}}
            )
            with pytest.raises(sr.StrictRunError, match="dirty source"):
                real_intent(tmp_path, pilot_root, allow_dirty_source=False)


def test_output_root_policy_and_cli_defaults(monkeypatch):
    """Smoke defaults to smoke-v1 and never a run root; a real intent only uses run-v1."""
    monkeypatch.setattr(sr, "TIER2_OUT_ROOT", sr.DEFAULT_OUT_ROOT)
    assert sr.DEFAULT_OUT_ROOT == Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-strict-20261002/run-v1"
    )
    assert sr.DEFAULT_SMOKE_ROOT.parent == sr.STUDY_ARTIFACT_ROOT
    sr.check_output_root(sr.DEFAULT_OUT_ROOT, smoke=False)
    sr.check_output_root(sr.DEFAULT_SMOKE_ROOT, smoke=True)
    with pytest.raises(sr.StrictRunError, match="smoke dry-run may not use"):
        sr.check_output_root(sr.DEFAULT_OUT_ROOT, smoke=True)
    for other in (sr.DEFAULT_SMOKE_ROOT, sr.STUDY_ARTIFACT_ROOT / "run-v2"):
        with pytest.raises(sr.StrictRunError, match="pre-registered root"):
            sr.check_output_root(other, smoke=False)
    seen = []

    def capture(**kwargs):
        seen.append(kwargs)
        raise sr.StrictRunError("captured")

    monkeypatch.setattr(sr, "build_intent", capture)
    base = ["prepare", "--deadline-utc", "2026-10-02T00:00:00+00:00"]
    base += ["--authorization-quote", "q"]
    for extra, expected in (
        (["--smoke-frames", "40"], sr.DEFAULT_SMOKE_ROOT),
        ([], sr.DEFAULT_OUT_ROOT),
    ):
        with pytest.raises(sr.StrictRunError, match="captured"):
            sr.main(base + extra)
        assert seen[-1]["out_root"] == expected
        assert seen[-1]["screen_run"] == sr.SCREEN_RUN
        assert seen[-1]["sweep_summary"] == sr.SWEEP_SUMMARY


def test_on_ac_power_parses_pmset(monkeypatch):
    class Done:
        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(sr.sys, "platform", "darwin")
    monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: Done("Now drawing from 'AC Power'"))
    assert sr.on_ac_power() is True
    monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: Done("'Battery Power'"))
    assert sr.on_ac_power() is False

    def boom(*a, **k):
        raise OSError("no pmset")

    monkeypatch.setattr(sr.subprocess, "run", boom)
    assert sr.on_ac_power() is False  # fail closed on darwin


def test_episode_runners_are_blocked_in_this_module():
    from src.scripts import tournament_eval

    for runner in (tournament_eval.rollout, dev_screen.run_episode, sr.run_unit_episode):
        with pytest.raises(AssertionError, match="blocked"):
            runner()
