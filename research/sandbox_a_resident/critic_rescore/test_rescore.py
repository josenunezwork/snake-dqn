"""Independent numeric, artifact/resume and support checks for the dev packet."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research.sandbox_a_resident.critic_rescore.run import (
    MIXES,
    aggregate,
    analyse,
    build_ledger,
    empty_ledger,
    fingerprint,
    identity,
    load_ledger,
    scarcity,
    selected_late,
    unit_filename,
    write_json_unit,
)


def fixture_control() -> dict:
    return dict(
        worlds=3,
        baseline_offset=100,
        holdout_offset=200,
        max_waves=2,
        late_length=800,
        late_lives_per_mix=1,
        horizon=4,
        test_only=True,
    )


def life(i: int, mix: str, bias: float = 0) -> dict:
    return dict(
        id=i,
        mix=mix,
        length=[1, 800, 801, 802],
        clock=[1, 0.9998, 0.9996, 0.9994],
        rewards=[10.0, 5.0, 5.0, 0.0],
        end_reason="fixture-cut",
        target=[20.0, 10.0, 5.0, 0.0],
        predictions=[[20.0, 10.0 + bias, 5.0 + bias, bias], [20.0, 8.0, 4.0, 0.0]],
    )


def test_ev_bias_and_mse_baseline_are_distinct() -> None:
    rows = [life(i, m, 20) for i, m in enumerate(MIXES)]
    result = analyse(rows, 0, 800, 80)
    assert result["explained_variance"] == pytest.approx(1)
    assert result["mse"] == pytest.approx(400)
    assert result["late_row_mean_baseline_mse"] == pytest.approx(np.var([10, 5, 0]))
    assert result["mse_ratio_to_late_mean"] > 1
    assert result["life_clustered_ev_ci95"] == pytest.approx([1, 1])
    assert result["mean_per_life_ev_ci95"] == pytest.approx([1, 1])


def test_balanced_first_life_selection() -> None:
    c = fixture_control()
    rows = [life(0, "scripted"), life(1, "scripted"), life(2, "frozen"), life(3, "mixed")]
    selected, counts = selected_late(rows, c)
    assert [r["id"] for r in selected] == [0, 2, 3]
    assert list(counts.values()) == [1, 1, 1]


def test_sparse_160_rows_reports_life_shortfall() -> None:
    row = life(1, "scripted")
    row.update(length=[800] * 200, target=list(range(200)), predictions=[list(range(200))] * 2)
    result = scarcity([row], 800, 50)
    assert result["160_late_rows"]["frequency"] == 1
    assert result["160_all_rows"]["frequency"] == 1


def test_immutable_unit_ledger_skip_and_foreign_refusal(tmp_path: Path) -> None:
    c = fixture_control()
    p = tmp_path / unit_filename(c, "holdout", 0)
    write_json_unit(p, dict(values=[1, 2]))
    ledger = build_ledger(c, [p])
    q = tmp_path / "ledger.json"
    q.write_text(json.dumps(ledger))
    assert identity("holdout", 0, 200) in load_ledger(q, c)
    with pytest.raises(FileExistsError):
        write_json_unit(p, dict(values=[1, 2]))
    ledger["allocation"] = "foreign"
    q.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="foreign"):
        load_ledger(q, c)
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        build_ledger(c, [p])


def test_aggregate_disjoint_bank_and_old_support(tmp_path: Path) -> None:
    c = fixture_control()
    paths = []
    for phase, offset in [("baseline", 100), ("holdout", 200)]:
        p = tmp_path / unit_filename(c, phase, 0)
        write_json_unit(
            p,
            dict(
                schema="critic-rescore-wave/v1",
                allocation=fingerprint(c),
                id=identity(phase, 0, offset),
                phase=phase,
                wave=0,
                lives=[life(offset + i, m) for i, m in enumerate(MIXES)],
            ),
        )
        paths.append(p)
    result = aggregate(c, paths, 40)
    assert result["late_lives"] == dict.fromkeys(MIXES, 1)
    assert result["critics"]["seed1"]["explained_variance"] == pytest.approx(1)
    assert result["critics"]["seed1"]["old_gate"]["status"] == "INSUFFICIENT_SUPPORT"
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(c, [*paths, paths[1]], 40)


def test_duplicate_or_foreign_ledger_units(tmp_path: Path) -> None:
    c = fixture_control()
    ledger = empty_ledger(c)
    row = dict(id=identity("holdout", 0, 200), size=1, sha256="a" * 64)
    ledger["units"] = [row, copy.deepcopy(row)]
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="duplicate"):
        load_ledger(p, c)


def test_outcome_blind_ledger_refuses_foreign_allocation_filename(tmp_path: Path) -> None:
    c = fixture_control()
    foreign = dict(c, late_lives_per_mix=2)
    p = tmp_path / unit_filename(foreign, "holdout", 0)
    write_json_unit(p, dict(values=[1]))
    with pytest.raises(ValueError, match="foreign unit path"):
        build_ledger(c, [p])


def test_registry_admits_only_complete_bounded_critic_units(tmp_path: Path) -> None:
    from research.sandbox_a_resident.units import digest
    from tools.sandbox.registry import validate_input

    c = fixture_control()
    p = tmp_path / unit_filename(c, "holdout", 0)
    write_json_unit(p, dict(values=[1]))
    record = dict(kind="verified-unit", path=str(p), sha256=digest(p))
    validate_input(record)
    Path(str(p) + ".complete").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        validate_input(record)
    q = tmp_path / unit_filename(c, "holdout", 32)
    write_json_unit(q, dict(values=[1]))
    with pytest.raises(ValueError, match="foreign critic"):
        validate_input(dict(kind="verified-unit", path=str(q), sha256=digest(q)))
