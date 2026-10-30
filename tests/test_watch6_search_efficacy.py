"""Efficacy decision boundaries, product config, and outcome-blind resume fences."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.watch6_search_efficacy import analyze
from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy import resume, run
from tools.ops.package_spec import empty_ledger, publish_json


def rows(death_delta: int = -2, relative: float = 0) -> list[dict]:
    result = []
    for index, allocation in enumerate(p.allocation()):
        bare = {
            "frames": p.HORIZON,
            "mirror_identity_frames": p.HORIZON,
            "mass_sum": (500 + index * 20) * p.HORIZON,
            "deaths": 4 + index % 2,
        }
        dense = {
            **bare,
            "mass_sum": bare["mass_sum"] * (1 + relative),
            "deaths": bare["deaths"] + death_delta,
        }
        result.append(
            {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "allocation": allocation,
                "fixture": False,
                "unit_id": run.unit_id(p.NAMESPACE, "watch6", index),
                "arms": {"bare": bare, "exact_stop": dense},
            }
        )
    return result


def test_co_primary_success_and_mass_failure() -> None:
    assert analyze.decide(rows())["verdict"] == "GO_EFFICACY_ONLY"
    assert analyze.decide(rows(relative=-0.1))["verdict"] == "FAIL_EFFICACY"
    assert analyze.decide(rows(death_delta=0))["verdict"] == "INCONCLUSIVE"
    assert analyze.decide(rows(death_delta=1))["verdict"] == "FAIL_EFFICACY"


def test_mass_boundary_and_ratio_of_means() -> None:
    decision = analyze.decide(rows(relative=-0.04))
    assert decision["mass_pass"]
    assert decision["mass_relative_difference"] == pytest.approx(-0.04)
    assert decision["mass_relative_ci95_fieller"][0] == pytest.approx(-0.04)


def test_unbounded_fieller_cannot_pass() -> None:
    values = rows()
    for row in values:
        row["arms"]["bare"]["mass_sum"] = 0
    assert not analyze.decide(values)["mass_pass"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "fixture", "foreign", "short"])
def test_incomplete_or_foreign_never_gets_a_look(mutation: str) -> None:
    values = rows()
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[-1] = copy.deepcopy(values[0])
    elif mutation == "fixture":
        values[0]["fixture"] = True
    elif mutation == "foreign":
        values[0]["allocation"]["seed"] += 1
    else:
        values[0]["arms"]["exact_stop"]["frames"] -= 1
    with pytest.raises(ValueError):
        analyze.decide(values)


def test_ledger_exact_skip_and_foreign_refusal(tmp_path: Path) -> None:
    ledger = empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity())
    publish_json(tmp_path / "empty.json", ledger)
    assert run.completed(tmp_path / "empty.json") == []
    ledger["units"] = [
        {
            "index": 3,
            "unit_id": run.unit_id(p.NAMESPACE, "watch6", 3),
            "sha256": "a" * 64,
            "size": 12,
        }
    ]
    publish_json(tmp_path / "one.json", ledger)
    assert [r["index"] for r in run.completed(tmp_path / "one.json")] == [3]
    ledger["units"] *= 2
    publish_json(tmp_path / "duplicate.json", ledger)
    with pytest.raises(ValueError):
        run.completed(tmp_path / "duplicate.json")
    ledger["namespace"] = "foreign"
    publish_json(tmp_path / "foreign.json", ledger)
    with pytest.raises(ValueError):
        run.completed(tmp_path / "foreign.json")


def test_outcome_blind_metadata_builder_refuses_foreign(tmp_path: Path) -> None:
    payload = tmp_path / "pair-00.json"
    descriptor = publish_json(payload, {"sealed_science": "must not be parsed"})
    publish_json(
        payload.with_suffix(".meta.json"),
        {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "allocation": p.allocation()[0],
            "sha256": descriptor["sha256"],
            "size": descriptor["size"],
        },
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"stop_verified": True, "cleanup_error": None, "output_dir": str(tmp_path)})
    )
    ledger = resume.build([payload], [receipt])
    assert ledger["units"][0]["index"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        resume.build([payload, payload], [receipt])
    receipt.write_text(json.dumps({"stop_verified": False}))
    with pytest.raises(ValueError, match="stop"):
        resume.build([payload], [receipt])


def test_allocation_and_fixture_isolation() -> None:
    assert len({r["seed"] for r in p.allocation()}) == p.COUNT
    assert not {r["seed"] for r in p.allocation()} & set(p.FIXTURE_SEEDS)
    assert not {r["seed"] for r in p.allocation()} & {
        r["seed"]
        for r in __import__(
            "research.bl_served_search.c1_bank", fromlist=["allocation"]
        ).allocation()
    }
