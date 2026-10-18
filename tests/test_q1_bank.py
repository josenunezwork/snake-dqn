from research.q0q1_20261017 import q1_bank


def test_q1_bank_is_fresh_terminal_shared_and_complete() -> None:
    bank = q1_bank.build_world_bank()
    seen = set()
    assert bank["namespace"] != "t23-r0-h5000-terminal-hero/v2"
    for seed in q1_bank.TRAINING_SEEDS:
        rows = q1_bank.worlds_for_seed(bank, seed)
        assert len(rows) == 128
        assert {
            m: sum(x["mix"] == m for x in rows) for m in q1_bank.MIX_COUNTS
        } == q1_bank.MIX_COUNTS
        for row in rows:
            assert row["terminal_hero_slot"] == 0 and row["horizon"] == 5000
            assert row["world_seed"] not in seen
            seen.add(row["world_seed"])
    assert not seen & {x["world_seed"] for x in bank["parity_worlds"]}
