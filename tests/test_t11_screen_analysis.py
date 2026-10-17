"""Focused contracts for the fail-closed T11 LOCAL30 result."""

from __future__ import annotations

import json
from pathlib import Path

from research.t11_screen_20261017.analyze import (
    ARM,
    analyze,
    read_training_telemetry,
    telemetry_svg,
)


def _row(name: str, mix: str) -> dict:
    worlds = {str(index): 100.0 for index in range(96 if mix == "scripted" else 32)}
    return {
        "name": name,
        "round_index": 74,
        "mix": mix,
        "frames": 5000,
        "mass": worlds,
        "deaths": {key: 0 for key in worlds},
        "alive_frames": {key: 5000 for key in worlds},
        "length_at_frames": {
            str(frame): {key: 1000 + frame for key in worlds} for frame in range(1000, 5001, 1000)
        },
    }


def test_halted_lane_kills_without_relabeling_four_seed_gameplay() -> None:
    rows = []
    for seed in (10, 11, 12, 14):
        for mix in ("scripted", "frozen", "mixed"):
            rows += [_row(f"{ARM}_s{seed}", mix), _row(f"CTRL_s{seed}", mix)]
    result = analyze(rows, {f"{ARM}_s{seed}": (1 if seed == 13 else 0) for seed in range(10, 15)})
    data = result["arms"][ARM]
    assert data["decision"] == "KILL"
    assert data["gameplay"] is None
    assert data["four_seed_descriptive_gameplay"]["scripted"]["mass"]["seed_n"] == 4
    assert json.dumps(result, allow_nan=False)


def test_shared_survivors_support_int_and_string_frame_keys() -> None:
    rows = []
    for name in (f"{ARM}_s10", "CTRL_s10"):
        row = _row(name, "scripted")
        row["length_at_frames"] = {
            1000: row["length_at_frames"]["1000"],
            **{frame: row["length_at_frames"][str(frame)] for frame in range(2000, 5001, 1000)},
        }
        rows.append(row)
    rows[0]["length_at_frames"][1000]["0"] = 1500
    rows[1]["length_at_frames"][1000]["0"] = 1000
    result = analyze(rows, {f"{ARM}_s10": 1})["arms"][ARM]
    summary = result["shared_survivor_lengths"]["frames"]["1000"]["scripted"]
    assert summary["shared_survivors"] == 96
    assert summary["pooled_mean_delta"] == 500 / 96


def _write_lane(root: Path, *, halted: bool) -> None:
    root.mkdir()
    summary = {
        "transitions": 300160 if halted else 8_000_000,
        "halt": "boost" if halted else None,
        "actor_exitcodes": [0, 0],
    }
    (root / "summary.json").write_text(json.dumps(summary))
    records = []
    for transition in (100000, 8_000_000 if not halted else 300160):
        records.append(
            {
                "transitions": transition,
                "bank_agree_decisive": 1.0,
                "bank_agree_counter": 1.0,
                "greedy_boost_rate": 0.002,
                "greedy_boost_rate_lt_700": 0.003,
                "td": 0.1,
                "anchor": 0.01,
                "bank_margin_loss": 0.02,
                "qmax": 10.0,
            }
        )
    (root / "log.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    if not halted:
        (root / "final.pth").write_bytes(b"final")


def test_telemetry_retains_halted_seed_and_null_final_window(tmp_path: Path) -> None:
    roots = {}
    for seed in range(10, 15):
        root = tmp_path / f"{ARM}_s{seed}"
        _write_lane(root, halted=seed == 13)
        roots[root.name] = root
    report = read_training_telemetry(roots)
    assert report["per_seed"]["13"]["status"] == "halted"
    assert report["per_seed"]["13"]["final_window"] is None
    assert report["final_8m_four_seed_descriptive_n"] == 4
    svg = tmp_path / "curve.svg"
    telemetry_svg(report, svg)
    assert svg.read_text().startswith("<svg")
    assert json.dumps(report, allow_nan=False)


def _registered_rows() -> list[dict]:
    from research.redesign_m2_20261008 import dev_probe

    rows = []
    for name in ("baseline", *(f"{arm}_s{s}" for arm in (ARM, "CTRL") for s in range(10, 15))):
        for mix in ("scripted", "frozen", "mixed"):
            row = _row(name, mix)
            seeds = list(map(str, dev_probe.probe_seeds(len(row["mass"]), 74)))
            for field in ("mass", "deaths", "alive_frames"):
                row[field] = dict(zip(seeds, row[field].values()))
            row["length_at_frames"] = {
                frame: dict(zip(seeds, values.values()))
                for frame, values in row["length_at_frames"].items()
            }
            rows.append(row)
    return rows


def test_nonhalt_requires_complete_receipts_and_exact_world_identity() -> None:
    import pytest

    rows = _registered_rows()
    statuses = {f"{ARM}_s{s}": 0 for s in range(10, 15)}
    with pytest.raises(ValueError, match="terminal receipts"):
        analyze(rows, {})
    rows[0]["mass"]["wrong-world"] = rows[0]["mass"].pop(next(iter(rows[0]["mass"])))
    with pytest.raises(ValueError, match="world count|world roster"):
        analyze(rows, statuses)


def test_exact_zero_survival_delta_is_kill() -> None:
    rows = _registered_rows()
    for seed, delta in zip(range(10, 15), (3, -1, -1, -1, 0)):
        name = f"CTRL_s{seed}" if delta > 0 else f"{ARM}_s{seed}"
        row = next(r for r in rows if r["name"] == name and r["mix"] == "scripted")
        for key in list(row["deaths"])[: abs(delta)]:
            row["deaths"][key] = 1
    result = analyze(rows, {f"{ARM}_s{s}": 0 for s in range(10, 15)})
    assert result["arms"][ARM]["decision"] == "KILL"
