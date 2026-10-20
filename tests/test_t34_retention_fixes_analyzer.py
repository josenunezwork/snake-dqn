"""Regression rules from the pre-data T34 retention FIX-FIRST rulings."""

from research.t34_pathb import analyze_v2 as analysis


def _arm(mass: float = 160.0, *, survival: float = 1.0, deaths: int = 0) -> dict:
    counts = [1650, 1650, 1650, 25, 15, 10]
    return {
        "mass_integral": mass,
        "survival_fraction": survival,
        "no_death": int(deaths == 0),
        "deaths": deaths,
        "decisions": 5000,
        "deaths_per_1k": deaths / 5,
        "boost_decisions": 50,
        "boost_share": 0.01,
        "action_counts": counts,
        "one_action_share": 0.33,
        "unconditional_faults": [],
        "overruns": 0,
        "median_length": 100.0,
        "p95_length": 200.0,
        "deaths_length_ge_800": 0,
        "late_deaths": 0,
        "late_large_overlap": 0,
    }


def _records() -> list[dict]:
    rows = []
    for mix_index, mix in enumerate(analysis.MIXES):
        for index in range(48):
            bare = _arm(100)
            if mix == "scripted" and index < 2:
                bare = _arm(100, survival=0.5, deaths=1)
            rows.append(
                {
                    "kind": "terminal",
                    "mix": mix,
                    "seed": mix_index * 48 + index,
                    "horizon": 5000,
                    "identity_status": "PASS",
                    "parity_status": "PASS",
                    "arms": {"bare": bare, "paced": _arm(), "unpaced": _arm()},
                }
            )
    for index in range(32):
        rows.append(
            {
                "kind": "product",
                "mix": "mixed",
                "seed": 144 + index,
                "horizon": 5000,
                "identity_status": "PASS",
                "parity_status": "PASS",
                "arms": {
                    "bare": _arm(100, survival=0.9, deaths=5),
                    "paced": _arm(110, survival=0.95, deaths=3),
                    "unpaced": _arm(110, survival=0.95, deaths=3),
                },
            }
        )
    return rows


def _timing() -> dict:
    def probe(fps: int) -> dict:
        return {
            "update_ms": {"n": 10, "p95": 1.0, "p99": 1.0, "max": 1.0},
            "full_tick_ms": {"n": 10, "p95": 2.0, "p99": 2.0, "max": 2.0},
            "ticks": 10,
            "fps": fps,
            "achieved_fps": float(fps),
            "contended": False,
            "overruns": 0,
        }

    return {"probes": {f"{arm}-{fps}": probe(fps) for arm in ("bare", "paced") for fps in (12, 24)}}


def _make_mass_near_zero(records: list[dict]) -> None:
    for index, row in enumerate(records[:144]):
        row["arms"]["bare"]["mass_integral"] = 1000
        row["arms"]["paced"]["mass_integral"] = 1060 + (500 if index % 2 else -500)
        row["arms"]["unpaced"]["mass_integral"] = 1060


def test_paced_efficacy_overruns_are_reported_but_do_not_kill() -> None:
    records = _records()
    records[0]["arms"]["paced"]["overruns"] = 3

    report = analysis.decide(records, _timing())

    assert report["verdict"] == "GO"
    assert report["paced_efficacy_overruns_diagnostic"] == 3


def test_only_the_24fps_p95_limit_is_a_timing_gate() -> None:
    timing = _timing()
    timing["probes"]["paced-12"]["update_ms"] = {
        "n": 10,
        "p95": 12.0,
        "p99": 12.0,
        "max": 12.0,
    }

    report = analysis.decide(_records(), timing)

    assert report["verdict"] == "GO"
    assert report["timing"]["probes"]["paced-12"]["update_ms"]["p95"] == 12.0
    assert report["timing"]["clauses"]["update_p95_le_11688ms_24"] is True

    timing["probes"]["paced-24"]["update_ms"] = {
        "n": 10,
        "p95": 12.0,
        "p99": 12.0,
        "max": 12.0,
    }
    assert analysis.decide(_records(), timing)["verdict"] == "KILL"


def test_near_zero_exception_does_not_mask_a_non_mass_skill_failure() -> None:
    records = _records()
    _make_mass_near_zero(records)
    for row in records:
        arm = row["arms"]["paced"]
        arm["boost_decisions"] = 0
        arm["boost_share"] = 0.0
        arm["action_counts"] = [1667, 1667, 1666, 0, 0, 0]
        arm["one_action_share"] = 1667 / 5000

    report = analysis.decide(records, _timing())

    assert report["paced"]["mass"]["mean"] >= 40
    assert abs(report["paced"]["mass"]["lower"]) <= 10
    assert report["paced"]["clauses"]["boost_share"] is False
    assert (report["verdict"], report["reason"]) == ("KILL", "deadline loss")
