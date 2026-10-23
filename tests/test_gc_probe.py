"""Untimed A1 reducer/clock checks; no Mac timing is admitted by this suite."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from research.bl_gc import probe


@pytest.mark.parametrize(
    "name, argv, forbidden",
    [
        (
            "Codex (Service)",
            [
                "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/"
                "Versions/155.0.8059.80/Helpers/Codex (Service).app/Contents/MacOS/"
                "Codex (Service)",
                "--type=gpu-process",
            ],
            False,
        ),
        (
            "codex",
            [
                "/Applications/ChatGPT.app/Contents/Resources/codex-cli/"
                "CodexCLI.app/Contents/MacOS/codex",
                "-c",
                "features.code_mode_host=true",
                "app-server",
                "--analytics-default-enabled",
                "-c",
                "plugins.codex-app-tools@openai-bundled.mcp_servers.codex_app.enabled=true",
                "-c",
                "plugins.code-review@openai-bundled.mcp_servers.code-review.enabled=true",
            ],
            False,
        ),
        (
            "codex-code-mode-host",
            ["/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex-code-mode-host"],
            False,
        ),
        (
            "codex",
            [
                "/Applications/ChatGPT.app/Contents/Resources/codex-cli/"
                "CodexCLI.app/Contents/MacOS/codex",
                "exec",
                "job",
            ],
            True,
        ),
        ("codex", ["/opt/homebrew/bin/codex", "app-server"], False),
        ("codex-code-mode-host", ["/opt/homebrew/bin/codex-code-mode-host"], False),
        ("codex", [], False),
        ("codex", ["/opt/homebrew/bin/codex", "exec", "--cd", "/tmp/job"], True),
        ("node", ["node", "/opt/homebrew/bin/codex", "exec", "job"], True),
        (
            "bash",
            ["bash", "/Users/josenunez/.claude/bin/codex-handoff.sh", "-C", "/tmp/job"],
            True,
        ),
        ("python", ["python", "-m", "research.colab_cli.supervisor", "run"], True),
        ("python", ["python", "research/colab_cli/supervisor.py", "run"], True),
        ("colab", ["/opt/homebrew/bin/colab", "run", "job"], True),
        ("colab", ["/opt/homebrew/bin/colab", "sessions"], False),
        ("python", ["python", "-c", "print('codex exec codex-handoff.sh')"], False),
        ("bash", ["bash", "/tmp/codex-handoff.sh.backup"], False),
    ],
)
def test_forbidden_processes_matches_jobs_not_desktop_helpers(
    monkeypatch: pytest.MonkeyPatch, name: str, argv: list[str], forbidden: bool
) -> None:
    import psutil

    process = SimpleNamespace(pid=123, info={"name": name, "cmdline": argv})
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: iter((process,)))
    expected = [{"pid": 123, "name": name.lower(), "argv": argv}] if forbidden else []
    assert probe.forbidden_processes() == expected


def _gc(policy: str) -> dict:
    return {
        "gc_policy": policy,
        "gc_freeze_count": 100,
        "gc_pause_ms": 0.0,
        **{f"gc_gen{g}_count": 0 for g in range(3)},
        **{f"gc_gen{g}_pause_ms": 0.0 for g in range(3)},
        "gc_histograms": {
            f"gen{g}": dict.fromkeys(("0-1", "1-5", "5-10", "10-20", "20+"), 0) for g in range(3)
        },
    }


def _cell(policy: str, seed: int, ticks: int) -> dict:
    epoch = 10_000_000_000
    rows = []
    for index in range(ticks):
        target = epoch + index * 1_000_000_000 // 24
        rows.append(
            {
                "index": index,
                "target_ns": target,
                "tick_started_ns": target,
                "finished_ns": target + 2_000_000,
                "update_ms": 1.0,
                "snapshot_ms": 1.0,
                "overrun": False,
                "hero_length": 1,
                "food_count": 1,
                "total_segments": 6,
                "late_ms_deadline": 0.0,
                "deaths": 0,
                "respawns": 0,
                **_gc(policy),
                **(
                    {
                        "rss_mb": 10.0,
                        "rss_sample_ns": target + 3_000_000,
                        "rss_sample_outside_scored_interval": True,
                    }
                    if index % 24 == 0
                    else {}
                ),
            }
        )
    state = {
        "rss_mb": 10.0,
        "tracked_objects": 1000,
        "gc_freeze_count": 100,
        "enabled": True,
        "thresholds": [700, 10, 10],
        "monotonic_ns": epoch,
    }
    return {
        "schema": probe.SCHEMA,
        "status": "COMPLETE",
        "mechanical_reason": None,
        "policy": policy,
        "seed": seed,
        "ticks_requested": ticks,
        "warmup_ticks": 200,
        "fps": 24,
        "epoch_ns": epoch,
        "ticks": rows,
        "gc_policy_receipt": {
            "policy": policy,
            "stage": "probe",
            "frozen_count": 100,
            "enabled": True,
            "thresholds": [700, 10, 10],
        },
        "attestation": {
            "torch_num_threads": 1,
            "pathb": "0",
            "judgment_sha256": probe.common.JUDGMENT_SHA256,
            "interpreter": "synthetic",
            "version": "3.12.3",
            "git_commit": "f" * 40,
            "source_hashes": {"probe.py": "f" * 64},
        },
        "roster": {"watch_roster": "mixed", "served": {"name": "b8-s3"}, "snakes": [0] * 6},
        "timing_admission": {"quiet": True},
        "host_observations": [{"quiet": True}],
        "mac_lease": {"id": "synthetic-test"},
        "start_state": state,
        "end_state": dict(state),
        "final_boundary_gc": _gc(policy),
    }


@pytest.fixture(scope="module")
def six() -> list[dict]:
    return [
        _cell(policy, seed, ticks)
        for ticks, seeds in ((2400, (2026102300, 2026102303)), (14400, (2026102304,)))
        for seed in seeds
        for policy in ("default", "freeze_startup")
    ]


def _changed(six: list[dict], index: int = 1) -> tuple[list[dict], dict]:
    cells = list(six)
    cells[index] = copy.deepcopy(cells[index])
    return cells, cells[index]


def test_complete_synthetic_six_cells_pass(six: list[dict]) -> None:
    assert probe.check(six)["verdict"] == "PASS"


def test_default_gc_failure_does_not_fail_candidate(six: list[dict]) -> None:
    cells, cell = _changed(six, 0)
    row = cell["ticks"][1]
    row.update(
        overrun=True,
        gc_gen2_count=1,
        gc_gen2_pause_ms=21.0,
        gc_pause_ms=21.0,
        finished_ns=row["target_ns"] + 50_000_000,
    )
    row["gc_histograms"]["gen2"]["20+"] = 1
    assert probe.check(cells)["verdict"] == "PASS"


def test_non_gc_overrun_has_distinct_product_finding(six: list[dict]) -> None:
    cells, cell = _changed(six)
    row = cell["ticks"][1]
    row.update(overrun=True, finished_ns=row["target_ns"] + 50_000_000, gc_attributed=True)
    assert probe.check(cells)["verdict"] == "GC_FIXED+BAR_FAIL(non-GC)"


def test_p95_boundary_and_failure_label(six: list[dict]) -> None:
    cells, cell = _changed(six)
    for row in cell["ticks"]:
        row["update_ms"] = 11.688
    assert probe.check(cells)["verdict"] == "PASS"
    for row in cell["ticks"]:
        row["update_ms"] = 11.689
    assert probe.check(cells)["verdict"] == "BAR_FAIL(p95, non-GC)"


def test_final_boundary_gc_cannot_escape_pause_bar(six: list[dict]) -> None:
    cells, cell = _changed(six)
    cell["final_boundary_gc"].update(gc_gen2_count=1, gc_gen2_pause_ms=20.0, gc_pause_ms=20.0)
    cell["final_boundary_gc"]["gc_histograms"]["gen2"]["20+"] = 1
    assert probe.check(cells)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "defect", ["partial", "nan", "counter", "clock", "freeze", "enabled", "missing"]
)
def test_missing_or_invalid_evidence_cannot_pass(six: list[dict], defect: str) -> None:
    cells, cell = _changed(six)
    if defect == "partial":
        cell["status"] = "INCOMPLETE"
    elif defect == "nan":
        cell["ticks"][0]["update_ms"] = float("nan")
    elif defect == "counter":
        cell["final_boundary_gc"]["gc_gen2_count"] = 1
    elif defect == "clock":
        cell["ticks"][3]["target_ns"] -= 1
    elif defect == "freeze":
        cell["start_state"] = {**cell["start_state"], "gc_freeze_count": 99}
    elif defect == "enabled":
        cell["start_state"] = {**cell["start_state"], "enabled": False}
    else:
        del cell["timing_admission"]
    assert probe.check(cells)["verdict"] in ("FAIL", "INCOMPLETE")


def test_foreign_seed_refused(six: list[dict]) -> None:
    cells, cell = _changed(six)
    cell["seed"] = 7
    with pytest.raises(ValueError):
        probe.check(cells)


def test_rss_residual_bound_is_inclusive(six: list[dict]) -> None:
    cells, cell = _changed(six, 5)
    cell["end_state"]["rss_mb"] += 32
    assert probe.check(cells)["verdict"] == "PASS"
    cell["end_state"]["rss_mb"] += 0.01
    assert probe.check(cells)["verdict"] == "FAIL"


def test_tick_clock_uses_multiply_then_integer_divide(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter((0, 125_000_000, 130_000_000))
    sleeps = []
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(times))
    monkeypatch.setattr(probe.time, "sleep", sleeps.append)
    monkeypatch.setattr(probe, "_timed_tick", lambda _: (1.0, 2.0, {}))
    snake = SimpleNamespace(id=0, is_alive=True, length=1, segments=[(0, 0)])
    session = SimpleNamespace(
        game=SimpleNamespace(
            snakes=[snake], frame_death_causes={}, food_manager=SimpleNamespace(food=[])
        )
    )
    meter = SimpleNamespace(drain=lambda: _gc("default"))
    row = probe._tick_row(session, meter, 0, 3)
    assert row["target_ns"] == 125_000_000
    assert sleeps == [0.125]


def test_mechanical_incomplete_cli_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from contextlib import contextmanager

    from research import mac_lease

    @contextmanager
    def lease(*args, **kwargs):
        yield SimpleNamespace(receipt={"id": "test"})

    monkeypatch.setattr(mac_lease, "held", lease)
    monkeypatch.setattr(probe, "_timing_admission", lambda: {})
    monkeypatch.setattr(probe, "run", lambda *args: {"status": "INCOMPLETE"})
    monkeypatch.setattr(probe.common, "write_result", lambda *args: None)
    status = probe.main(
        [
            "--policy",
            "default",
            "--seed",
            "2026102300",
            "--ticks",
            "2400",
            "--out",
            str(tmp_path / "test.json"),
        ]
    )
    assert status == 2


@pytest.mark.parametrize("policy", ["default", "freeze_startup"])
def test_bar5_decreasing_ordered_cell_passes(six: list[dict], policy: str) -> None:
    cells, cell = _changed(six, int(policy == "freeze_startup"))
    cell["start_state"]["gc_freeze_count"] = 99
    for row in cell["ticks"]:
        row["gc_freeze_count"] = 98
    cell["final_boundary_gc"]["gc_freeze_count"] = 97
    cell["end_state"]["gc_freeze_count"] = 96
    result = probe.check_cell(cell)
    assert result["verdict"] == "PASS"
    assert result["gc_freeze_count_total_decrease"] == 3
    assert result["gc_freeze_count_decrease_since_startup"] == 4
    assert probe.check(cells)["verdict"] == "PASS"


@pytest.mark.parametrize("where", ["start", "tick", "final_boundary", "end"])
def test_bar5_rejects_increase_even_below_startup(six: list[dict], where: str) -> None:
    cells, cell = _changed(six)
    cell["start_state"]["gc_freeze_count"] = 90
    for row in cell["ticks"]:
        row["gc_freeze_count"] = 89
    cell["final_boundary_gc"]["gc_freeze_count"] = 88
    cell["end_state"]["gc_freeze_count"] = 87
    if where == "start":
        cell["start_state"]["gc_freeze_count"] = 101
    elif where == "tick":
        cell["ticks"][1]["gc_freeze_count"] = 90
    elif where == "final_boundary":
        cell["final_boundary_gc"]["gc_freeze_count"] = 90
    else:
        cell["end_state"]["gc_freeze_count"] = 89
    assert probe.check(cells)["verdict"] == "FAIL"
    assert (
        "gc_freeze_count is not non-increasing since startup"
        in probe.check_cell(cell)["violations"]
    )


def test_bar5_negative_counter_is_incomplete(six: list[dict]) -> None:
    cells, cell = _changed(six)
    cell["ticks"][-1]["gc_freeze_count"] = -1
    cell["final_boundary_gc"]["gc_freeze_count"] = -1
    assert probe.check(cells)["verdict"] == "INCOMPLETE"


def test_run_discloses_background_load_without_counting_teardown(monkeypatch) -> None:
    from research.bl_k1 import m1
    from research.bl_k1.m1_background import allowed_background

    command = "/usr/libexec/runningboardd"
    samples = [
        {
            "monotonic_ns": index * 2_000_000_000,
            "allowed_background_processes": allowed_background(
                {"processes": f"PID PPID %CPU COMM\n20 1 {cpu} {command}\n"}
            ),
        }
        for index, cpu in enumerate((60.5, 4.0, 0.0))
    ]
    clock = {"ns": 0}
    monitor = SimpleNamespace(
        rows=samples,
        error=None,
        thread=SimpleNamespace(start=lambda: None, ident=123),
        close=lambda: clock.update(ns=6_000_000_000),
    )
    session = SimpleNamespace(close=lambda **kw: clock.update(ns=100_000_000_000))
    meter = SimpleNamespace(drain=lambda: {}, close=lambda: None)
    monkeypatch.setattr(m1, "HostMonitor", lambda _: monitor)
    monkeypatch.setattr(m1, "new_watch", lambda _: (session, {}))
    monkeypatch.setattr(probe, "_timing_admission", lambda: {})
    monkeypatch.setattr(probe.common, "initialize", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(probe, "_meter", lambda: meter)
    monkeypatch.setattr(probe, "_timed_tick", lambda _: None)
    monkeypatch.setattr(probe, "_gc_state", lambda _: {})
    monkeypatch.setattr(probe, "_rss_mb", lambda _: 1.0)
    monkeypatch.setattr(probe, "_tick_row", lambda *a: {})
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: clock["ns"])
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    monkeypatch.setattr(probe.gc, "get_objects", lambda: [])
    result = probe.run("default", 2026102300, 2400)
    assert result["status"] == "COMPLETE"
    assert result["host_observations"] == samples
    summary = result["background_load"]
    assert summary["observed_seconds"] == 6.0
    assert summary["by_command"][command] == {
        "max_cpu_percent": 60.5,
        "mean_cpu_percent": 21.5,
        "time_at_or_above_50_seconds": 2.0,
    }


def test_background_disclosure_is_report_only(six: list[dict]) -> None:
    cells, cell = _changed(six)
    cell["background_load"] = {"total": {"max_cpu_percent": 250.0}}
    result = probe.check_cell(cell)
    assert result["verdict"] == "PASS"
    assert result["background_load"] == cell["background_load"]
    cell["status"] = "INCOMPLETE"
    result = probe.check_cell(cell)
    assert result["verdict"] == "INCOMPLETE"
    assert result["background_load"] == cell["background_load"]
