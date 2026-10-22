"""X5 accounting and immutable receipt boundaries, with no gameplay."""

import json

import pytest

from research.bl_x5 import timing


def test_seals_refuse_overwrite_and_detect_tamper(tmp_path):
    path = tmp_path / "receipt.json"
    timing.seal(path, {"value": 1})
    assert timing.read_sealed(path) == {"value": 1}
    with pytest.raises(FileExistsError):
        timing.seal(path, {"value": 2})
    path.write_text(json.dumps({"value": 3}))
    with pytest.raises(ValueError, match="hash mismatch"):
        timing.read_sealed(path)


def test_incomplete_marker_is_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    timing.seal(path, {"value": 1})
    path.with_name("receipt.json.complete").write_text("wrong\n")
    with pytest.raises(ValueError, match="incomplete"):
        timing.read_sealed(path)


def test_mix_weighted_forecasts_distinguish_cpu_and_wall():
    workers = [
        {
            "mix": mix,
            "units": [
                {"worlds": list(range(32)), "cpu_seconds": 32 * cost, "wall_seconds": 64 * cost}
            ],
            "process_cpu_seconds": 32 * cost + 32,
        }
        for mix, cost in zip(timing.MIXES, (1, 2, 3))
    ]
    result = timing.forecasts(workers)
    banks = result["banks"]
    bare = banks["bare-192-96-96"]
    assert bare["episodes"] == 384
    assert bare["projected_cpu_hours_from_measured_rate"] == pytest.approx(672 / 3600)
    assert bare["projected_cpu_hours_including_amortized_startup"] == pytest.approx(1056 / 3600)
    assert bare["ideal_wall_hours_at_measured_3_workers"] == pytest.approx(224 / 3600)
    assert banks["paired-192-96-96"]["episodes"] == 768
    assert banks["paired-384-128-128"]["episodes"] == 1280
    assert banks["e13-768-episodes"]["episodes"] == 768
    assert banks["e13-768-episodes"]["projected_cpu_hours_from_measured_rate"] == pytest.approx(
        1536 / 3600
    )


def test_partial_launch_drains_before_releasing_slots(tmp_path, monkeypatch):
    from research.apex_safety_20260926 import dev_screen as ds

    timing.seal(tmp_path / "x5-manifest.json", {"checkpoints": {}})
    events = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self):
            assert (tmp_path / "STOP").exists()
            events.append("wait")

    def launch(*args, **kwargs):
        if events:
            raise OSError("injected second launch failure")
        events.append("launch")
        return Process()

    monkeypatch.setattr(ds, "acquire_cpu_slots", lambda *a, **k: [])
    monkeypatch.setattr(ds, "release_cpu_slots", lambda slots: events.append("release"))
    monkeypatch.setattr(timing, "host", lambda: {"reasons": []})
    monkeypatch.setattr(timing, "command", lambda argv: "tmux inventory")
    monkeypatch.setattr(timing.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="second launch"):
        timing.run(tmp_path, lease_path=tmp_path / "mac-lease.json")
    assert events == ["launch", "wait", "release"]
    assert timing.read_sealed(tmp_path / "x5-coordinator-error.json")["status"] == "INCOMPLETE"


def test_foreign_cpu_lane_blocks_host_admission(monkeypatch):
    def command(argv):
        if argv == ["pmset", "-g", "batt"]:
            return "AC Power"
        if argv[0] == "ioreg":
            return '"AppleClamshellState" = No'
        if argv == ["pmset", "-g", "therm"]:
            return (
                "Note: No thermal warning level has been recorded\n"
                "Note: No performance warning level has been recorded\n"
                "Note: No CPU power status has been recorded"
            )
        if argv[0] == "ps":
            return "PID PPID %CPU COMM\n999999 999998 100.0 /bin/Python"
        return "nominal"

    monkeypatch.setattr(timing, "command", command)
    assert timing.host()["reasons"] == ["foreign_python_cpu_lane"]


def test_unit_reducer_rejects_changed_world_identity(tmp_path):
    timing.seal(tmp_path / "x5-manifest.json", {"frozen": True})
    plan = {"arms": ["B8-repeat-1", "B8-repeat-2"], "banks": {}, "rows": {}}
    workers = []
    for mix in timing.MIXES:
        plan["banks"][mix] = list(range(16))
        plan["rows"][mix] = {str(s): {"identity": {"seed": s}} for s in range(16)}
        refs = []
        for arm in plan["arms"]:
            for offset in (0, 8):
                worlds = list(range(offset, offset + 8))
                unit = {
                    "mix": mix,
                    "arm": arm,
                    "offset": offset,
                    "worlds": worlds,
                    "cpu_seconds": 1,
                    "wall_seconds": 2,
                    "records": [{"seed": s, "world_identity": {"seed": s}} for s in worlds],
                }
                path = tmp_path / f"{mix}-{arm}-{offset}.json"
                timing.seal(path, unit)
                refs.append(
                    {
                        "path": str(path),
                        "sha256": timing.sha(path),
                        **{k: v for k, v in unit.items() if k != "records"},
                    }
                )
        workers.append(
            {
                "mix": mix,
                "units": refs,
                "manifest_sha256": timing.sha(tmp_path / "x5-manifest.json"),
            }
        )
    timing.validate_units(tmp_path, plan, workers)
    plan["rows"]["scripted"]["0"]["identity"] = {"seed": 999}
    with pytest.raises(ValueError, match="world identity"):
        timing.validate_units(tmp_path, plan, workers)
