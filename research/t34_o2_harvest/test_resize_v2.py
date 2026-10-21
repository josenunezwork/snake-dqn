"""Prospective size, bank, and manifest safety regression tests; no gameplay."""

import copy
import math

import pytest

from research.t34_o2_harvest import bank_v2, make_manifest_v2, plan_v2


def test_largest_balanced_capacity_is_exact_and_chunks_cover_bank():
    plan = plan_v2.optimal_capacity()
    assert plan["study_worlds"] == 1260
    assert set(plan["mix_counts"].values()) == {420}
    assert plan["aggregate_unit_cap"] <= 150
    assert plan["next_wave_minimum_timeout_minutes"] > plan["available_timeout_minutes"]
    assert sum(c["count"] for c in plan["chunks"]) == 1260
    for i, chunk in enumerate(plan["chunks"]):
        assert chunk["start"] == sum(c["count"] for c in plan["chunks"][:i])
        caps = chunk["caps"]
        assert caps["timeout_minutes"] == 2 * caps["estimated_minutes"] <= 480
        assert caps["reserve_units_per_job"] == pytest.approx(0.2 * caps["units_per_job"])
        assert caps["forecast_seconds"] < caps["ordinary_job_seconds"]


def test_bank_retains_exact_v1_and_balances_all_three_mixes():
    bank = bank_v2.materialize()
    prior = bank_v2.read_json(bank_v2.PACKAGE / "BANK_O2.json")
    for row in prior["rows"]:
        assert row in bank["rows"]
    assert len(bank["rows"]) == 1260
    assert len({r["world_seed"] for r in bank["rows"]}) == 1260
    assert all(sum(r["mix"] == m for r in bank["rows"]) == 420 for m in plan_v2.MIXES)
    bank_v2.validate_bank(bank)
    bad = copy.deepcopy(bank)
    bad["rows"][0]["world_seed"] += 1
    with pytest.raises(ValueError, match="deterministic"):
        bank_v2.validate_bank(bad)


def test_manifest_uses_each_chunk_budget_and_only_three_tiny_inputs(tmp_path, monkeypatch):
    plan = plan_v2.optimal_capacity()
    frozen = {
        **plan,
        "source_commit": "a" * 40,
        "caps": plan_v2.caps(),
        "source_hashes": {"research/t34_o2_harvest/launch_v2.sh": "b" * 64},
    }
    registration, catalog, preflight = [
        tmp_path / n for n in ("registration.json", "catalog.json", "preflight.json")
    ]
    for p in (registration, catalog, preflight):
        p.write_text("{}")
    monkeypatch.setattr(make_manifest_v2, "require_registration", lambda _: frozen)
    monkeypatch.setattr(make_manifest_v2, "load_public_catalog", lambda _: {})
    monkeypatch.setattr(make_manifest_v2, "require_qualified_transport", lambda *a: None)
    monkeypatch.setattr(
        make_manifest_v2,
        "_preflight",
        lambda *a: {
            "path": str(preflight),
            "sha256": make_manifest_v2._sha(preflight),
            "forecast": {"estimated_minutes": 231},
        },
    )
    manifests = [
        make_manifest_v2.make_manifest(
            registration=registration,
            catalog=catalog,
            preflight_receipt=preflight,
            output_dir=tmp_path / f"out-{i}",
            commit="a" * 40,
            chunk=i,
        )
        for i in range(len(plan["chunks"]))
    ]
    assert sum(m["budget"]["unit_cap"] for m in manifests) <= 150
    assert all(m["timeout_minutes"] <= 480 for m in manifests)
    assert all(len(m["inputs"]) == 3 for m in manifests)
    assert all(m["script"].endswith("launch_v2.sh") for m in manifests)
    assert all(m["env"]["T34_O2_WORKERS"] == "10" for m in manifests)
    assert math.isclose(
        sum(m["budget"]["reserve_units"] for m in manifests), plan_v2.caps()["reserve_units"]
    )


def test_v2_preflight_binds_new_schema_and_measured_rate_floor(tmp_path, monkeypatch):
    import json

    from research.t34_o2_harvest import preflight_v2
    from research.t34_o2_harvest.test_preflight import _fixture

    receipt, registration = _fixture(tmp_path, monkeypatch)
    authority = json.loads(registration.read_text())
    authority.update(study_worlds=1260, caps=plan_v2.caps())
    monkeypatch.setattr(preflight_v2, "require_registration", lambda _: authority)
    value = json.loads(receipt.read_text())
    value["schema"] = "t34-o2-harvest-run/v2"
    for entry in value["worlds"]:
        path = receipt.parent / entry["path"]
        world = json.loads(path.read_text())
        for name, arm in world["arms"].items():
            arm["elapsed_seconds"] = 0.1 if name == "bare" else 12.1
        path.write_text(json.dumps(world))
        entry["sha256"] = preflight_v2._sha(path)
    receipt.write_text(json.dumps(value))
    result = preflight_v2.validate_preflight(receipt, registration)
    assert result["forecast"]["all_study_row_steps_bound"] == 1260 * 63000
    assert result["forecast"]["aggregate_unit_cap"] <= 150
    value["schema"] = "t34-o2-harvest-run/v1"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen registration"):
        preflight_v2.validate_preflight(receipt, registration)
