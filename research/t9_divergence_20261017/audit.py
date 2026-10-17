"""Independently validate cohort, trace coverage, paired origins and artifact bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.b8_death_census_20261014.census import sha256_file
from research.t1_commitment_20261015.replay import ARTIFACTS


def audit(out: Path) -> dict:
    """Fail closed on missing cases, changed evidence or malformed live-state metrics."""
    manifest = json.loads((out / "manifest.json").read_text())
    completion = json.loads((out / "completion.json").read_text())
    assert completion["status"] == "PASS" and completion["full_cohort"]
    assert completion["rows_sha256"] == sha256_file(out / "rows.jsonl")
    assert completion["manifest_sha256"] == sha256_file(out / "manifest.json")
    for path, expected in manifest["input_source_sha256"].items():
        assert sha256_file(Path(path)) == expected, path
    original = [json.loads(line) for line in (ARTIFACTS / "rows.jsonl").read_text().splitlines()]
    expected_rows = {(r["arm"], r["env"]): r for r in original if r["length"] >= 800 and r["lad"]}
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == len(expected_rows) == 46
    assert {(r["arm"], r["env"]) for r in rows} == set(expected_rows)
    traces = 0
    for row in rows:
        source = expected_rows[(row["arm"], row["env"])]
        lad = source["lad"]
        assert row["world_seed"] == source["world_seed"]
        assert row["fatal_t"] == source["fatal_t"]
        assert row["length"] == source["length"]
        assert row["lad_decision"] == lad["decision"]
        assert row["H_LAD"] == source["fatal_t"] - lad["decision"]
        assert row["fatal"]["death_decision"] == source["fatal_t"]
        assert row["fatal"]["first_action"] == lad["chosen_action"]
        assert row["surviving"]["first_action"] == lad["best_surviving_action"]
        assert (
            row["surviving"]["validated_transitions"]
            == lad["outcomes"][str(lad["best_surviving_action"])]["horizon"]
        )
        assert row["surviving"]["endpoint_alive"]
        assert row["lad_q_and_action_geometry_exact"] and row["source_clone_unchanged"]
        f, s = row["fatal"]["trace"], row["surviving"]["trace"]
        assert [t["offset"] for t in f] == list(range(min(101, row["H_LAD"] + 1)))
        assert [t["offset"] for t in s] == list(range(101))
        for key in f[0]:
            if key not in ("action", "toward_food"):
                assert f[0][key] == s[0][key], (row["arm"], row["env"], key)
        for t in f + s:
            assert 0 < t["head_region"] <= t["largest_region"] <= 145 * 83
            assert t["head_in_largest"] == (t["head_region"] == t["largest_region"])
            assert t["tail_reachable"] == (t["tail_bfs"] is not None)
            if t["tail_bfs"] is not None:
                assert t["tail_bfs"] >= t["tail_manhattan"]
            boundary = [t["boundary_" + k] for k in ("own", "opponent", "wall")]
            assert all(0 <= v <= 1 for v in boundary)
            assert abs(sum(boundary) - 1) < 1e-12
            assert 0 <= t["boundary_outside15_fraction"] <= 1
            assert t["own_cells_r15"] <= t["own_cells_r30"]
            assert t["opponent_cells_r15"] <= t["opponent_cells_r30"]
            race = [t[f"race_{h}"] for h in (16, 32, 64, 128, 256)]
            assert race == sorted(race) and 0 <= race[0] <= race[-1] <= 145 * 83
            traces += 1
    return {
        "status": "PASS",
        "cohort": len(rows),
        "trace_states": traces,
        "source_and_input_hashes_checked": len(manifest["input_source_sha256"]),
        "rows_sha256": sha256_file(out / "rows.jsonl"),
        "manifest_sha256": sha256_file(out / "manifest.json"),
        "audit_source_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.out)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
