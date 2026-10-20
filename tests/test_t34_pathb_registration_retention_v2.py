"""Prospective-only checks for the T34 retention registration packager."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_pathb import registration_retention_v2 as registration
from research.t34_pathb.retention_v2 import RegistrationError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def test_allocate_rejects_the_whole_cross_namespace_bank_on_audit_hit(monkeypatch, tmp_path):
    seen: list[int] = []

    def candidates(namespace: str, count: int) -> list[int]:
        base = int(hashlib.sha256(namespace.encode()).hexdigest()[:7], 16)
        return list(range(base, base + count))

    monkeypatch.setattr(registration.bank, "_candidate_seeds", candidates)

    def reject(_repo: Path, values: list[int]):
        seen.extend(values)
        raise RuntimeError("whole allocation rejected")

    monkeypatch.setattr(registration.bank, "_audit_candidates", reject)
    with pytest.raises(RuntimeError, match="whole allocation"):
        registration.allocate_bank(repo=tmp_path)
    assert len(seen) == 113
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("contents", ['{"x": NaN}', '{"x": 1, "x": 2}'])
def test_json_binding_parser_rejects_nonfinite_and_duplicate_keys(tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_text(contents)
    with pytest.raises(RegistrationError, match="expected readable JSON object"):
        registration._read_object(path)


def test_allocate_materializes_exact_fixed_bank_with_separate_namespaces(monkeypatch, tmp_path):
    counter = iter(range(10_000, 20_000))
    monkeypatch.setattr(
        registration.bank,
        "_candidate_seeds",
        lambda _namespace, count: [next(counter) for _ in range(count)],
    )
    monkeypatch.setattr(
        registration.bank, "_audit_candidates", lambda _repo, _values: {"concrete_hits": {}}
    )

    def materialize(rows):
        for row in rows:
            row["roster"] = {"seed": row["seed"], "mix": row["mix"], "specs": []}
        return rows

    monkeypatch.setattr(registration, "_materialize_terminal", materialize)
    payload, audit = registration.allocate_bank(repo=tmp_path)
    assert len(payload["terminal_rows"]) == 96
    assert len(payload["product_rows"]) == 16
    assert len(audit["seeds"]) == 113
    assert audit["pass"] is True and audit["collisions"] == []
    assert set(payload["namespaces"]) == {
        "terminal_scripted",
        "terminal_frozen",
        "terminal_mixed",
        "product_mixed_watch6",
        "timing_mixed_watch6",
    }
    assert all(row["roster"]["seed"] == row["seed"] for row in payload["terminal_rows"])


def _draft_inputs(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    source = tmp_path / "source.py"
    source.write_text("source")
    source_hashes = {"source.py": _sha(source)}
    monkeypatch.setattr(registration, "source_inventory", lambda _root: {"source.py": _sha(source)})

    def git_output(args, **_kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return "a" * 40 + "\n"
        if args[:2] == ["git", "show"]:
            return source.read_bytes()
        raise AssertionError(args)

    monkeypatch.setattr(registration.subprocess, "check_output", git_output)
    monkeypatch.setattr(
        registration.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(registration, "_validate_checkpoint_coverage", lambda _registration: None)
    terminal = []
    seed = 1
    for mix in registration.MIXES:
        for _ in range(32):
            terminal.append(
                {"seed": seed, "mix": mix, "roster": {"seed": seed, "mix": mix, "specs": []}}
            )
            seed += 1
    product = [{"seed": seed + index, "mix": "mixed"} for index in range(16)]
    timing = {"seed": seed + 16, "mix": "mixed"}
    bank = {"terminal_rows": terminal, "product_rows": product, "timing_row": timing}
    audit = {
        "pass": True,
        "collisions": [],
        "seeds": sorted({row["seed"] for row in terminal + product} | {timing["seed"]}),
    }
    amendment = tmp_path / "amendment.md"
    amendment.write_text(
        "- Decision: ratified\n- Option: 1\n- Date: 2026-10-06\n- Changes, if any: none\n"
    )
    profile = {
        "load_policy": {"quiet_ratified": False, "phase0_finished_required": True},
        "survival_band": {"amendment": {"path": str(amendment), "sha256": _sha(amendment)}},
    }
    pool = {"source_files": {"source.py": source_hashes["source.py"]}, "rows": [1, 2]}
    simulator = tmp_path / "research/t34_pathb/retention_oc_v2.py"
    simulator.parent.mkdir(parents=True, exist_ok=True)
    simulator.write_text("simulator")
    oc = {
        "schema": "t34-retention-fixed-n-oc/v1",
        "status": "PASS",
        "source_hashes": source_hashes,
        "simulator_hash": _sha(simulator),
        "world_pool_sha256": hashlib.sha256(
            json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "development_pool": pool,
        "completed": 20_000,
        "reps": 20_000,
        "n_per_mix": 32,
        "looks": [32],
        "threads": 1,
        "development_only": True,
        "efficacy_rows_produced": False,
        "active_arms": ["paced", "unpaced"],
        "models": [
            "t3_dense_b8_s0_scripted_shape_all_mixes",
            "t3_dense_b8_s3_scripted_shape_all_mixes",
            "normal_stress_sd_0.25_watch_mass_scale",
            "normal_stress_sd_0.35_watch_mass_scale",
            "normal_stress_sd_0.5_watch_mass_scale",
        ],
        "scenarios": {
            "no_change": [0.0, 0.0, 0.0],
            "pooled_minus_0.05": [-0.05, -0.05, -0.05],
            "frozen_minus_0.075": [-0.075, 0.0, 0.0],
            "scripted_minus_0.075": [0.0, -0.075, 0.0],
            "mixed_minus_0.075": [0.0, 0.0, -0.075],
        },
        "theta_grid": [20.0, 26.8, 40.0, 60.0, 80.0],
        "alpha_grid": [0.05, 0.10],
        "rates": {
            f"{model}|{scenario}|theta={theta:g}|alpha={alpha:g}|joint_mass_and_band": {
                "k": 0,
                "n": 20_000,
                "rate": 0.0,
            }
            for model in (
                "t3_dense_b8_s0_scripted_shape_all_mixes",
                "t3_dense_b8_s3_scripted_shape_all_mixes",
                "normal_stress_sd_0.25_watch_mass_scale",
                "normal_stress_sd_0.35_watch_mass_scale",
                "normal_stress_sd_0.5_watch_mass_scale",
            )
            for scenario in (
                "no_change",
                "pooled_minus_0.05",
                "frozen_minus_0.075",
                "scripted_minus_0.075",
                "mixed_minus_0.075",
            )
            for theta in (20.0, 26.8, 40.0, 60.0, 80.0)
            for alpha in (0.05, 0.10)
        },
        "acceptance": {"max_joint_regression_pass_rate": 0.0, "failed_cells": {}},
    }
    for name, value in (
        ("bank.json", bank),
        ("audit.json", audit),
        ("profile.json", profile),
        ("oc.json", oc),
    ):
        _json(tmp_path / name, value)
    (tmp_path / "prereg.md").write_text("draft prereg")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    draft = registration.build_draft(
        profile=Path("profile.json"),
        bank_path=Path("bank.json"),
        audit_path=Path("audit.json"),
        preregistration=Path("prereg.md"),
        operating_characteristic=Path("oc.json"),
        checkpoints=[{"path": str(checkpoint), "sha256": _sha(checkpoint)}],
        root=tmp_path,
    )
    return draft, source


def test_draft_validates_bindings_but_stays_unauthorized(monkeypatch, tmp_path):
    draft, _source = _draft_inputs(tmp_path, monkeypatch)
    result = registration.validate_registration(draft, root=tmp_path)
    assert result == {"valid": True, "authorized": False, "mode": "draft"}
    assert draft["trace_receipt"] is None
    assert draft["source_commit"] == "a" * 40


def test_draft_refuses_stale_source_or_weak_operating_characteristic(monkeypatch, tmp_path):
    draft, source = _draft_inputs(tmp_path, monkeypatch)
    source.write_text("changed")
    with pytest.raises(
        RegistrationError, match="source-file bytes differ|source inventory is stale"
    ):
        registration.validate_registration(draft, root=tmp_path)

    draft, _source = _draft_inputs(tmp_path, monkeypatch)
    oc = json.loads((tmp_path / "oc.json").read_text())
    protected = next(name for name in oc["rates"] if "|no_change|" not in name)
    oc["rates"][protected].update({"k": 1220, "rate": 0.061})
    _json(tmp_path / "oc.json", oc)
    draft["operating_characteristic"]["sha256"] = _sha(tmp_path / "oc.json")
    with pytest.raises(RegistrationError, match="0.06"):
        registration.validate_registration(draft, root=tmp_path)


def test_draft_refuses_mutated_survival_band_amendment(monkeypatch, tmp_path):
    draft, _source = _draft_inputs(tmp_path, monkeypatch)
    (tmp_path / "amendment.md").write_text("changed")
    with pytest.raises(RegistrationError, match="survival-band amendment bytes differ"):
        registration.validate_registration(draft, root=tmp_path)


def test_ratification_never_mints_evidence_and_requires_real_precheck(monkeypatch, tmp_path):
    draft, _source = _draft_inputs(tmp_path, monkeypatch)
    trace = tmp_path / "trace.json"
    _json(trace, {"status": "PASS", "source_hashes": draft["source_hashes"]})
    with pytest.raises(RegistrationError, match="observed passing"):
        registration.ratify_registration(
            draft,
            source_commit="a" * 40,
            precheck={},
            trace_receipt=trace,
            load_record=tmp_path / "load.json",
            operator="Claude",
            acknowledge_near_boundary_precedence=True,
            root=tmp_path,
        )
