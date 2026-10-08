"""phase_r_verify on a synthetic two-look root (the smoke stopped at look 0)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/redesign_m3_20261008"))
import phase_r_verify as v  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _record(look, binding, unit, seed, world, index, control=False):
    return {
        "look": look,
        "binding": binding,
        "commit": "c0",
        "unit_id": unit,
        "seed": seed,
        "world_seed": world,
        "world_index": index,
        "control": control,
        "horizon": 5000 if control else 10000,
        "prefix_h5000": None if control else {"mass_integral": 1.0},
        "roster_member_sha256s": ["r"],
        "hero_checkpoint": {"path": "x", "sha256": "h"},
    }


def _root(tmp: Path, started1: str) -> Path:
    plan = {"plan": {"look_sizes": [1, 2]}, "plan_sha256": "p", "banks": {"0": [10, 11]},
            "study": {"commit": "c0"}}  # fmt: skip
    (tmp / "plan.json").write_text(json.dumps(plan))
    (tmp / "looks").mkdir()
    r0 = tmp / "looks/look-0.json"
    r0.write_text(
        json.dumps({"action": "CONTINUE", "extra": {"analysed_utc": "2026-10-08T12:00:00+00:00"}})
    )
    (tmp / "looks/look-1.json").write_text(json.dumps({"action": "STOP"}))
    for k, (lo, hi), started in ((0, (0, 1), "2026-10-08T11:00:00+00:00"), (1, (1, 2), started1)):
        b = {
            "look": k,
            "plan_file_sha256": _sha(tmp / "plan.json"),
            "plan_sha256": "p",
            "previous_receipt_sha256": _sha(r0) if k else None,
            "world_index_range": [lo, hi],
        }
        sd = tmp / f"shards/look-{k}"
        (sd / "records").mkdir(parents=True)
        (sd / "start.json").write_text(
            json.dumps({"binding": b, "commit": "c0", "units": [f"u{k}"], "started_utc": started})
        )
        w = [10, 11][k]
        (sd / "records/a.json").write_text(json.dumps(_record(k, b, f"u{k}", 0, w, k)))
        if k == 0:
            (sd / "records/c.json").write_text(json.dumps(_record(0, b, "u0", 0, 10, 0, True)))
    return tmp


def test_two_looks_pass(tmp_path):
    out = v.verify(_root(tmp_path, "2026-10-08T12:30:00+00:00"))
    assert out["verdict"] == "PASS", out["problems"]


def test_look_started_before_previous_analysis_fails(tmp_path):
    out = v.verify(_root(tmp_path, "2026-10-08T11:30:00+00:00"))
    assert out["verdict"] == "FAIL" and any("before look 0" in p for p in out["problems"])


def test_record_outside_its_look_fails(tmp_path):
    root = _root(tmp_path, "2026-10-08T12:30:00+00:00")
    p = root / "shards/look-1/records/a.json"
    r = json.loads(p.read_text())
    r["world_seed"], r["world_index"] = 10, 0  # world index 0 is look 0's
    p.write_text(json.dumps(r))
    assert v.verify(root)["verdict"] == "FAIL"
