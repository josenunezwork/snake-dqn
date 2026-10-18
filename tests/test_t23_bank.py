from __future__ import annotations

from research.t23_r0_screen_20261017 import bank


def test_t23_bank_has_independent_fixed_initial_extension_and_parity_worlds() -> None:
    value = bank.build_world_bank()
    assert value["profile"]["profile_id"] == bank.PROFILE_ID
    assert value["profile"]["base_profile_id"] == "promotion-v2-watch-rect"
    seen = set()
    for training_seed in bank.TRAINING_SEEDS:
        initial = bank.worlds_for_seed(value, training_seed)
        extension = bank.worlds_for_seed(value, training_seed, "extension")
        assert len(initial) == len(extension) == 128
        assert {row["mix"] for row in initial} == {"scripted", "frozen", "mixed"}
        assert sum(row["mix"] == "scripted" for row in initial) == 64
        assert sum(row["mix"] == "frozen" for row in initial) == 32
        assert sum(row["mix"] == "mixed" for row in initial) == 32
        for row in initial + extension:
            assert row["horizon"] == 5000 and row["terminal_hero_slot"] == 0
            assert row["world_seed"] not in seen
            assert len(row["opponents"]) == 5
            seen.add(row["world_seed"])
    assert len(value["parity_worlds"]) == 4
    assert not seen & {row["world_seed"] for row in value["parity_worlds"]}


def test_t23_bank_semantic_hash_rejects_tampering(tmp_path) -> None:
    value = bank.build_world_bank()
    path = tmp_path / "bank.json"
    import json

    path.write_text(json.dumps(value))
    assert bank.load_world_bank(path)["schema"] == bank.SCHEMA
    value["worlds"]["7101"]["initial"][0]["world_seed"] += 1
    path.write_text(json.dumps(value))
    try:
        bank.load_world_bank(path)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("tampered bank was accepted")


def test_t23_audit_scan_detects_candidate_in_jsonl_fixture(tmp_path) -> None:
    candidate = 3901234567
    (tmp_path / "strict.jsonl").write_text('{"world_seed": 3901234567}\n')
    assert bank._scan_candidate_tokens((tmp_path,), tmp_path / "own", {candidate}) == {candidate}
