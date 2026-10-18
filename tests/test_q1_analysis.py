import hashlib
from research.q0q1_20261017 import q1_analyze, q1_bank


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest(delta: float = 25.0) -> dict:
    bank = q1_bank.build_world_bank()
    rows = []
    for seed in q1_bank.TRAINING_SEEDS:
        for world in q1_bank.worlds_for_seed(bank, seed):
            common = {
                "training_seed": seed,
                "mix": world["mix"],
                "world_semantic_sha256": world["world_semantic_sha256"],
                "horizon": 5000,
                "frames_stepped": 5000,
                "clean_exit": True,
                "world_bank_sha256": _h("bank"),
                "initial_world_sha256": world["initial_world_sha256"],
                "alive_frame_fraction": 1.0,
                "no_death_episode": True,
            }
            for arm in (*q1_analyze.ARMS, "B8"):
                rows.append(
                    {
                        **common,
                        "schema": "q1-eval-row/v1",
                        "arm": arm,
                        "mass_integral": 100.0 + (delta if arm != "B8" else 0.0),
                    }
                )
    return {"bank": bank, "rows": rows, "tripwires": {}}


def test_q1_go_selects_fixed_id_d_and_reports_all_package_contrasts() -> None:
    result = q1_analyze.analyze(_manifest(), 10)
    assert result["verdict"] == "GO_TO_MAC_PHASE_R" and result["selected_recipe"] == "D"
    assert set(result["contrasts"]) == {"B-A", "D-C", "C-A", "D-B", "interaction"}


def test_q1_identity_coverage_fails_closed() -> None:
    data = _manifest()
    data["rows"].pop()
    assert q1_analyze.analyze(data, 10)["verdict"] == "INVALID"
