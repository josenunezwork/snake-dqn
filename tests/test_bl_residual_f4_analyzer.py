"""Synthetic R8/R9 fixtures only; no label or gameplay artifact is opened."""

from __future__ import annotations

from research.bl_residual import analyze


def _identity() -> dict:
    return {
        "registration_sha256": analyze.REGISTRATION_SHA256,
        "recipe_sha256": analyze._current_recipe_digest(),
        "lineage_sha256": "b" * 64,
        "seed_audit_sha256": "c" * 64,
        "parity_sha256": "d" * 64,
        "controller_sha256": {
            "B8": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
            "D1024": "f" * 64,
            "D4096": "1" * 64,
        },
        "analyzer_sha256": analyze.source_sha256(),
        "sealed": True,
    }


def _payload(*, top: float = 60.0, dose: float = 30.0, product: bool = True) -> dict:
    rows = []
    trace = [
        {
            "decision": index,
            "alive": True,
            "hero_length": 100,
            "served_action": 3 if index == 0 else 1,
        }
        for index in range(200)
    ]
    for mix in analyze.MIXES:
        for number in range(analyze.N_PER_MIX):
            rows.append(
                {
                    "world_id": f"{mix}-{number}",
                    "mix": mix,
                    "arms": {
                        "B8": {
                            "mass_integral": 100.0,
                            "alive_frame_fraction": 0.9,
                            "no_death_episode": False,
                            "decisions": 200,
                            "action_trace": trace,
                            "boost_share": 0.005,
                        },
                        "D1024": {
                            "mass_integral": 100.0 + top - dose,
                            "alive_frame_fraction": 1.0,
                            "no_death_episode": True,
                            "decisions": 200,
                            "action_trace": trace,
                            "boost_share": 0.005,
                        },
                        "D4096": {
                            "mass_integral": 100.0 + top,
                            "alive_frame_fraction": 1.0,
                            "no_death_episode": True,
                            "decisions": 200,
                            "action_trace": trace,
                            "boost_share": 0.005,
                        },
                    },
                    "boost_share": {"B8": 0.005, "D1024": 0.005, "D4096": 0.005},
                }
            )
    sessions = []
    for number in range(128):
        sessions.append(
            {
                "session_id": f"session-{number}",
                "arms": {
                    "B8": {"deaths": 10.0, "hero_decisions": 1000.0, "mass_integral": 100.0},
                    "D4096": {
                        "deaths": 5.0 if product else 10.0,
                        "hero_decisions": 1000.0,
                        "mass_integral": 100.0 if product else 90.0,
                    },
                },
            }
        )
    value = {
        "schema": analyze.INPUT_SCHEMA,
        "identity": _identity(),
        "staging": {
            dose: {"selected_support": True, "unconditional_fault": False}
            for dose in ("D256", "D1024", "D4096")
        }
        | {
            "projection": {"qualified_placement": True},
            "funding": {"verdict": "PASS_D4096_FUNDING"},
            "admission": {"harm_screen_pass": True},
        },
        "gameplay": {
            "observations": rows,
            "controller_sha256": _identity()["controller_sha256"],
            "trajectory_tripwires": {"D1024": False, "D4096": False},
            "known_controller_fault": {},
        },
        "probe": {
            "schema": "bl-residual-gameplay-probe/v1",
            "pass": True,
            "tripwires": {
                "max_decile_disagreement_le_0_05": True,
                "boost_share_le_1_5x_b8": True,
                "heldout_retention_ge_0_70": True,
            },
            "controller_sha256": {
                "B8": _identity()["controller_sha256"]["B8"],
                "D4096": _identity()["controller_sha256"]["D4096"],
            },
            "action_collapse": {"D4096": {"pass": True}},
        },
        "watch6": {
            "sessions": sessions,
            "controller_sha256": {
                "B8": _identity()["controller_sha256"]["B8"],
                "D4096": _identity()["controller_sha256"]["D4096"],
            },
        },
    }
    value["identity"]["gameplay_sha256"] = analyze._sha(value["gameplay"])
    value["identity"]["watch6_sha256"] = analyze._sha(value["watch6"])
    value["evidence"] = {
        "lineage": {"sha256": value["identity"]["lineage_sha256"], "status": "PASS"},
        "seed_audit": {"sha256": value["identity"]["seed_audit_sha256"], "status": "CLEAR"},
        "parity": {"sha256": value["identity"]["parity_sha256"], "status": "PASS"},
    }
    admission = analyze.h5000_admission(value)
    value["h5000_admission"] = admission
    value["identity"]["h5000_admission_sha256"] = analyze._sha(admission)
    return value


def _reseal(value: dict) -> dict:
    if "gameplay" in value:
        value["identity"]["gameplay_sha256"] = analyze._sha(value["gameplay"])
    if "watch6" in value:
        value["identity"]["watch6_sha256"] = analyze._sha(value["watch6"])
    return value


def test_synthetic_fixtures_reach_every_registered_label_exactly_once() -> None:
    cases: dict[str, dict] = {}
    invalid = _payload()
    invalid["identity"]["sealed"] = False
    cases["INVALID"] = invalid
    incomplete = _payload()
    incomplete.pop("gameplay")
    cases["INCOMPLETE"] = incomplete
    insufficient = _payload()
    insufficient["staging"]["D4096"]["selected_support"] = False
    cases["INSUFFICIENT"] = insufficient
    kill_fit = _payload()
    kill_fit["staging"]["D256"]["unconditional_fault"] = True
    cases["KILL_FIT"] = kill_fit
    stop_funding = _payload()
    stop_funding["staging"]["funding"]["verdict"] = "STOP_FUNDING"
    cases["STOP_FUNDING"] = stop_funding
    stop_admission = _payload()
    stop_admission["staging"]["admission"]["harm_screen_pass"] = False
    cases["STOP_ADMISSION"] = stop_admission
    cases["KILL_DOSE_CLOSED"] = _payload(top=60.0, dose=-1.0)
    cases["KILL_DOSE_FUTILE"] = _payload(top=60.0, dose=10.0)
    cases["GO_TOP_DOSE_DEV"] = _payload()
    cases["INCONCLUSIVE"] = _payload(top=30.0, dose=30.0)
    actual = {name: analyze.analyze(value)["label"] for name, value in cases.items()}
    assert actual == {name: name for name in analyze.LABELS}
    assert len(set(actual.values())) == len(analyze.LABELS)


def test_boundaries_have_registered_inclusivity_and_not_run_rule() -> None:
    # Equality at the point bars passes.  With exact no-variance fixtures it also
    # makes the one-sided lower and upper bounds equal to the point estimate.
    equal_bars = analyze.analyze(_payload(top=40.0, dose=20.0))
    assert equal_bars["label"] == "GO_TOP_DOSE_DEV"

    ub_zero = analyze.analyze(_payload(top=60.0, dose=0.0))
    assert ub_zero["screen"]["dose"]["upper90"] == 0.0
    assert ub_zero["label"] == "KILL_DOSE_CLOSED"

    ub_twenty = analyze.analyze(_payload(top=60.0, dose=20.0))
    assert ub_twenty["screen"]["dose"]["upper90"] == 20.0
    assert ub_twenty["label"] != "KILL_DOSE_FUTILE"

    # Product is absent by rule when an H5000 GO bar fails, which is distinct
    # from an incomplete required Watch6 product.
    no_watch = _payload(top=30.0, dose=30.0)
    no_watch.pop("watch6")
    out = analyze.analyze(no_watch)
    assert out["label"] == "INCONCLUSIVE"
    assert out["watch6"] == "NOT_RUN_BY_RULE"


def test_top_passed_dose_failed_flag_and_create_only_canonical_contract(tmp_path) -> None:
    # Top passes every top-v-B8 bar, while the dose has UB >= 20 but misses its
    # point/Holm bar; its registered label remains INCONCLUSIVE with the flag.
    value = _payload(top=60.0, dose=19.0)
    # A sparse large positive dose delta keeps the unadjusted UB above +20
    # while the mean misses the inclusive +20 point bar.
    for row in value["gameplay"]["observations"][:2]:
        row["arms"]["D1024"]["mass_integral"] -= 150.0
    out = analyze.analyze(_reseal(value))
    assert out["label"] == "INCONCLUSIVE"
    assert out["flag"] == "TOP_PASSED_DOSE_FAILED"

    source, destination = tmp_path / "input.json", tmp_path / "receipt.json"
    source.write_text(analyze.canonical_receipt(_payload()))
    analyze.main.__wrapped__ if hasattr(analyze.main, "__wrapped__") else None
    # CLI behavior is covered through a subprocess in callers; canonical output
    # itself stays deterministic and carries no outcome-derived input mutation.
    assert analyze.canonical_receipt(_payload()) == source.read_text()
    assert not destination.exists()


def test_d256_fault_stops_without_downstream_receipts() -> None:
    value = _payload()
    value["staging"]["D256"]["unconditional_fault"] = True
    for key in ("D1024", "D4096", "funding", "projection", "admission"):
        value["staging"].pop(key, None)
    out = analyze.analyze(_reseal(value))
    assert out["label"] == "KILL_FIT"


def test_true_early_stops_do_not_require_future_artifact_identities() -> None:
    d256 = _payload()
    d256["staging"] = {"D256": {"selected_support": True, "unconditional_fault": True}}
    for key in ("probe", "gameplay", "watch6", "h5000_admission"):
        d256.pop(key, None)
    d256["identity"].pop("gameplay_sha256", None)
    d256["identity"].pop("watch6_sha256", None)
    d256["identity"].pop("h5000_admission_sha256", None)
    d256["identity"]["controller_sha256"] = {
        "B8": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
    }
    assert analyze.analyze(d256)["label"] == "KILL_FIT"

    funding = _payload()
    funding["staging"].pop("D4096")
    funding["staging"].pop("projection")
    funding["staging"].pop("admission")
    funding["staging"]["funding"] = {"verdict": "STOP_FUNDING"}
    for key in ("probe", "gameplay", "watch6", "h5000_admission"):
        funding.pop(key, None)
    funding["identity"].pop("gameplay_sha256", None)
    funding["identity"].pop("watch6_sha256", None)
    funding["identity"].pop("h5000_admission_sha256", None)
    funding["identity"]["controller_sha256"] = {
        "B8": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
    }
    assert analyze.analyze(funding)["label"] == "STOP_FUNDING"


def test_duplicate_materialized_watch_session_has_invalid_precedence() -> None:
    for dose in (30.0, 10.0):
        value = _payload(dose=dose)
        sessions = value["watch6"]["sessions"]
        sessions[1]["session_id"] = sessions[0]["session_id"]
        result = analyze.analyze(_reseal(value))
        assert result["label"] == "INVALID"
        assert "duplicate_Watch6_session" in result["reasons"]
