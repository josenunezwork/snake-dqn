"""Focused contracts for the T34 CPU process-fanout transport."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t34_exit_b8 import benchmark, process_fanout
from research.t34_exit_b8.process_fanout import aggregate_results, shard_rows


def test_fanout_materializes_the_same_default_mix_bank_as_serial(monkeypatch, tmp_path) -> None:
    from research.t34_exit_b8 import process_fanout, trajectory

    def bank(config, count):
        assert config["mixes"] == ("scripted", "frozen", "mixed")
        return [{"world_id": str(i), "mix": config["mixes"][i % 3]} for i in range(count)]

    def stop_after_bank(config, *, runner, rows, nproc):
        assert [row["mix"] for row in rows] == ["scripted", "frozen", "mixed"] * 2
        raise RuntimeError("bank inspected before spawning")

    monkeypatch.setattr(trajectory, "make_world_rows", bank)
    monkeypatch.setattr(process_fanout, "run_fanout", stop_after_bank)
    with pytest.raises(RuntimeError, match="bank inspected"):
        benchmark.run_benchmark(
            dict(
                kernel="v2",
                checkpoint=tmp_path / "b8",
                output_dir=tmp_path / "worlds",
                worlds=1,
                max_worlds=6,
                nproc=2,
                threads=1,
                device="cpu",
                seconds=1,
            )
        )


def test_round_robin_shards_are_disjoint_and_reassemble_global_rows() -> None:
    rows = [{"world_id": f"w{index}"} for index in range(7)]
    shards = shard_rows(rows, 3)

    assert [[row["world_id"] for row in shard] for shard in shards] == [
        ["w0", "w3", "w6"],
        ["w1", "w4"],
        ["w2", "w5"],
    ]
    assert sorted(row["world_id"] for shard in shards for row in shard) == [
        f"w{i}" for i in range(7)
    ]


def test_aggregate_keeps_partial_work_separate_from_completed_worlds() -> None:
    complete = {
        "worlds": 1,
        "requested_worlds": 1,
        "complete": True,
        "decisions": 4,
        "export_bytes": 10,
        "active_row_steps": 9,
        "resident_row_steps": 9,
        "completed_world_row_steps": 9,
        "unfinished_row_steps": 0,
        "configured_root_capacity": 1,
        "peak_root_worlds": 1,
        "mixes": {"scripted": {"searched": 1, "expanded": 1, "worlds": 1}},
        "teacher_profile": {"active_row_steps": 9, "resident_row_steps": 9},
    }
    partial = complete | {
        "worlds": 0,
        "complete": False,
        "decisions": 2,
        "export_bytes": 0,
        "active_row_steps": 5,
        "resident_row_steps": 5,
        "completed_world_row_steps": 0,
        "unfinished_row_steps": 5,
        "mixes": {"frozen": {"searched": 0, "expanded": 0, "worlds": 0}},
    }

    result = aggregate_results([complete, partial], nproc=2, elapsed=2.0)

    assert result["worlds"] == 1
    assert result["completed_world_row_steps"] == 9
    assert result["unfinished_row_steps"] == 5
    assert result["executed_row_steps_per_world"] == 9
    assert result["fanout"]["nproc"] == 2
    assert result["mixes"]["scripted"]["worlds"] == 1
    assert result["mixes"]["frozen"]["worlds"] == 0


def test_normal_cpu_w1_configuration_is_not_rejected_as_legacy_cuda_a5(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    raw = {
        "complete": True,
        "mixes": {"scripted": {"searched": 0, "expanded": 0, "worlds": 1}},
        "teacher_profile": {"active_row_steps": 1, "resident_row_steps": 1},
        "worlds": 1,
        "worlds_per_hour": 1.0,
        "decisions": 1,
        "decisions_per_second": 1.0,
        "export_bytes": 1,
        "export_bytes_per_world": 1.0,
        "configured_root_capacity": 1,
        "peak_root_worlds": 1,
        "search_batch_peak": 1,
    }
    monkeypatch.setattr(benchmark, "_cpu_ceiling", lambda _threads: None)
    monkeypatch.setattr(benchmark, "_load_policy", lambda _config: None)
    monkeypatch.setattr(benchmark, "_kernel_runner", lambda _kernel: lambda _config: raw)
    report = benchmark.run_benchmark(
        {
            "kernel": "v2",
            "checkpoint": checkpoint,
            "output_dir": tmp_path / "worlds",
            "worlds": 1,
            "max_worlds": 1,
            "frames": 1,
            "seconds": 1,
            "threads": 1,
            "nproc": 1,
            "device": "cpu",
        }
    )
    assert report["recipe"]["device"] == "cpu"
    assert report["recipe"]["root_worlds_per_worker"] == 1


def test_colab_launchers_default_to_cpu_single_thread_w1() -> None:
    benchmark_shell = Path("research/t34_exit_b8/bench.sh").read_text()
    label_shell = Path("research/t34_exit_b8/label.sh").read_text()

    assert 'device="${T34_DEVICE:-cpu}"' in benchmark_shell
    assert 'threads="${T34_THREADS:-1}"; worlds="${T34_WORLDS:-1}"' in benchmark_shell
    assert 'device="${T34_DEVICE:-cpu}"' in label_shell
    assert 'threads="${T34_THREADS:-1}"' in label_shell


def test_cuda_lane_aggregate_accepts_new_serial_nproc_recipe(tmp_path) -> None:
    recipe = {
        "workers": 1,
        "nproc": 1,
        "threads_per_worker": 5,
        "root_worlds_per_worker": 256,
        "horizon": 300,
        "device": "cuda",
    }
    report = {
        "schema": "t34-a5-benchmark/v2",
        "kernel": "v2",
        "complete": True,
        "preflight": False,
        "mac_full_round": False,
        "cpu_runtime_probe": False,
        "recipe": recipe,
        "elapsed_seconds": 1.0,
        "active_row_steps": 4000,
        "resident_row_steps": 4000,
    }
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text(json.dumps(report | {"namespace": "one"}))
    second.write_text(json.dumps(report | {"namespace": "two"}))
    assert benchmark.aggregate_lane_reports([first, second])["go_rule"]["met"]


def test_fanout_failure_terminates_started_siblings(monkeypatch, tmp_path) -> None:
    """A child-reported failure promptly cleans up every already-started process."""

    class Child:
        exitcode = None

        def __init__(self, **_kwargs) -> None:
            self.alive = True
            self.terminated = False

        def start(self) -> None:
            return None

        def join(self, timeout=None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

    children: list[Child] = []

    class Context:
        def Queue(self):
            class Queue:
                used = False

                def get(self, timeout):
                    del timeout
                    if not self.used:
                        self.used = True
                        return 0, None, "worker failed"
                    raise process_fanout.queue.Empty

            return Queue()

        def Process(self, **kwargs):
            child = Child(**kwargs)
            children.append(child)
            return child

    monkeypatch.setattr(process_fanout.mp, "get_context", lambda _name: Context())
    with pytest.raises(RuntimeError, match="worker failed"):
        process_fanout.run_fanout(
            {"output_dir": tmp_path, "seconds": 1},
            runner="unused:runner",
            rows=[{"world_id": "one"}, {"world_id": "two"}],
            nproc=2,
        )
    assert len(children) == 2
    assert all(child.terminated for child in children)
