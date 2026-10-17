"""Pure T7 runner contract tests; gameplay is intentionally never invoked here."""

import json

import pytest

from research.redesign_t7_phase_r_20261016 import phase_r_t7 as r


def test_prefix_identical_ignores_only_veto_probe():
    direct = {"mass_integral": 2.0, "probes": {"safety_veto": {"n": 1}, "x": 3}}
    prefix = {"mass_integral": 2.0, "probes": {"safety_veto": {"n": 9}, "x": 3}}
    assert r.prefix_identical(direct, prefix)
    assert not r.prefix_identical(dict(direct, mass_integral=1.0), prefix)


def test_mix_budget_and_control_count(monkeypatch):
    plan = {"banks": {str(s): list(range(s * 1000, s * 1000 + 192)) for s in range(5)}}

    def rows(bank, mix):
        count = r.MIX_COUNTS[mix]
        return {x: {"world_seed": x} for x in bank[:count]}

    monkeypatch.setattr(r, "bank_rows", rows)
    units = r.units_for(plan)
    controls = [u for u in units if u["control"]]
    assert len(controls) == 45
    assert all(len(u["worlds"]) <= 8 for u in units if not u["control"])
    assert sum(len(u["worlds"]) for u in units if not u["control"]) == 3 * 5 * sum(
        r.MIX_COUNTS.values()
    )


def test_record_names_separate_control():
    unit = {"hero": "B8-noveto", "seed": 0, "mix": "scripted", "control": False}
    assert r.record_name(unit, 7) == "B8-noveto-s0-scripted-7.json"
    assert r.record_name(dict(unit, control=True), 7).startswith("control-")


def test_envelope_detects_record_mutation():
    row = {"hero": "B8-noveto", "record": {"mass_integral": 1.0}}
    row["envelope_sha256"] = r._envelope(row)
    assert row["envelope_sha256"] == r._envelope(row)
    row["record"]["mass_integral"] = 2.0
    assert row["envelope_sha256"] != r._envelope(row)


@pytest.fixture
def bound_record(tmp_path):
    plan = {
        "banks": {str(s): list(range(s * 1000, s * 1000 + 192)) for s in range(5)},
        "source_manifest": {},
        "commit": "synthetic",
        "checkpoints": {"B8-noveto|s0": {"path": "unused", "sha256": "synthetic"}},
        "rosters": {"0": {"scripted": {"0": ["opponent"]}}},
        "world_identities": {"0": {"scripted": {"0": {"seed": 0, "mix_id": "scripted"}}}},
        "engine": {"profile_h10000": "profile10", "profile_h5000": "profile5"},
    }
    r.receipts.write_once(tmp_path / "plan.json", plan)
    r.receipts.write_once(
        tmp_path / "start.json",
        {
            "schema": "t7-phase-r-start/v1",
            "plan_sha256": r.sha256_file(tmp_path / "plan.json"),
            "commit": plan["commit"],
            "units": [u["unit_id"] for u in r.units_for(plan)],
            "started_epoch": 0,
            "started_utc": "synthetic",
        },
    )
    unit = r.units_for(plan)[0]
    row = {
        "schema": r.SCHEMA,
        "unit_id": unit["unit_id"],
        "plan_sha256": r.sha256_file(tmp_path / "plan.json"),
        "start_sha256": r.validate_start(tmp_path, plan),
        "source_manifest_sha256": r.canonical_sha({}),
        "commit": plan["commit"],
        "hero": unit["hero"],
        "seed": 0,
        "mix": "scripted",
        "control": False,
        "horizon": 10000,
        "world_index_range": [0, 8],
        "world_seed": 0,
        "world_index": 0,
        "roster_member_sha256s": ["opponent"],
        "hero_checkpoint": plan["checkpoints"]["B8-noveto|s0"],
        "engine": {
            "sim_engine": "grid",
            "vector61_forward": "rowwise",
            "ego2s_forward": "rowwise",
            "ego2s_mask": "advisory",
            "veto": None,
        },
        "record": {
            "seed": 0,
            "evaluation_profile_digest": "profile10",
            "world_identity": {"seed": 0, "mix_id": "scripted"},
            "denominators": {"scored_frames": 10000},
        },
        "prefix_h5000": {"denominators": {"scored_frames": 5000}},
    }
    path = tmp_path / "records" / r.record_name(unit, 0)
    row["envelope_sha256"] = r._envelope(row)
    r.receipts.write_once(path, row)
    r.receipts.write_once(
        path.with_suffix(".sha256.json"), {"record": path.name, "sha256": r.sha256_file(path)}
    )
    return path, unit, plan, row


def test_saved_record_identity_and_crash_recovery(bound_record):
    path, unit, plan, _ = bound_record
    r._validate_existing(path, unit, plan)
    path.with_suffix(".sha256.json").unlink()
    with pytest.raises(SystemExit, match="digest"):
        r._validate_existing(path, unit, plan)
    r._validate_existing(path, unit, plan, allow_missing_sidecar=True)


@pytest.mark.parametrize("change", ["profile", "horizon", "veto", "world_index", "start"])
def test_rehashed_record_still_cannot_change_identity(bound_record, change):
    path, unit, plan, row = bound_record
    if change == "profile":
        row["record"]["evaluation_profile_digest"] = "other"
    elif change == "horizon":
        row["prefix_h5000"]["denominators"]["scored_frames"] = 4999
    elif change == "veto":
        row["engine"]["veto"] = "v8"
    elif change == "world_index":
        row["world_index"] = 1
    else:
        row["start_sha256"] = "other"
    row["envelope_sha256"] = r._envelope(row)
    path.write_text(json.dumps(row))
    path.with_suffix(".sha256.json").write_text(
        json.dumps({"record": path.name, "sha256": r.sha256_file(path)})
    )
    with pytest.raises(SystemExit):
        r._validate_existing(path, unit, plan)
