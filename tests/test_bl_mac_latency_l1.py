"""Focused evidence-boundary tests for the L1-prime diagnostic reducer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.bl_mac_latency import l1


def test_nearest_rank_and_distribution_are_inclusive_and_keep_raw_count():
    assert l1.nearest_rank([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert l1.nearest_rank([4.0, 1.0, 3.0, 2.0], 0.99) == 4.0
    assert l1.distribution([0.3, 0.1, 0.2]) == {
        "quantile": "nearest-rank/inclusive/v1",
        "count": 3,
        "p50_seconds": 0.2,
        "p95_seconds": 0.3,
        "p99_seconds": 0.3,
        "max_seconds": 0.3,
    }


def test_nearest_rank_rejects_invalid_percentile_and_empty_distribution():
    with pytest.raises(ValueError, match="percentile"):
        l1.nearest_rank([1.0], 0.0)
    assert l1.distribution([])["p99_seconds"] is None


def test_fixed_cells_start_with_conservative_cpu_and_cover_authorized_envelope():
    cells = l1.fixed_cells()
    assert [(row["device"], row["workers"], row["max_pending"]) for row in cells] == [
        ("cpu", 1, 1),
        ("cpu", 1, 8),
        ("cpu", 2, 8),
        ("cpu", 3, 8),
        ("cpu", 4, 8),
        ("mps", 1, 1),
        ("mps", 1, 8),
        ("mps", 2, 8),
        ("mps", 3, 8),
        ("mps", 4, 8),
    ]


def test_cell_result_never_turns_micro_timing_into_a_paced_pass():
    micro = {
        "sha256": "a" * 64,
        "kind": "UNCHANGED_EXACT_T3_DENSE_BATCH_MICRO_TIMING",
        "distribution": l1.distribution([0.2, 0.3]),
    }
    cpu = l1._cell_result(l1.fixed_cells()[0], micro)
    mps = l1._cell_result(l1.fixed_cells()[-1], micro)
    assert cpu["status"] == "NOT_READY"
    assert cpu["paced_execution"]["valid_consumptions"] == 0
    assert cpu["bar_verdict"] == "NOT_EVALUABLE"
    assert mps["status"] == "NOT_READY"
    assert mps["reason"] == l1.ROOT_PROVIDER_REASON


def test_load_fixtures_keeps_file_and_canonical_ruler_identities_distinct(monkeypatch, tmp_path):
    fixture_doc = {"schema": "fixture", "fixtures": [{"fixture_id": "frozen-800"}]}
    monkeypatch.setattr(l1, "_read_completed_json", lambda _path: (fixture_doc, "f" * 64))
    monkeypatch.setattr(l1, "validate_fixtures", lambda value: value["fixtures"])
    rows, file_sha, ruler_sha = l1.load_fixtures(tmp_path / "fixtures.json")
    assert rows == fixture_doc["fixtures"]
    assert file_sha == "f" * 64
    assert ruler_sha == l1.sha256_bytes(l1.canonical(fixture_doc).encode())
    assert ruler_sha != file_sha


def test_device_operator_installs_geometry_before_binding_root_digest(monkeypatch):
    order = []
    operator = object.__new__(l1.DeviceOperator)
    operator._exact = SimpleNamespace(_roots={})
    operator._world = lambda _fixture: "world"
    fixture = {"fixture_id": "frozen-800", "target_length": 800, "root_sha256": "a" * 64}

    import research.bl_served_search.fixture_geometry as geometry
    import research.bl_served_search.ruler_operator as ruler_operator

    monkeypatch.setattr(
        geometry, "install", lambda world, length: order.append(("install", world, length))
    )
    monkeypatch.setattr(
        ruler_operator, "_root_bytes", lambda world: order.append(("digest", world)) or b"x"
    )
    monkeypatch.setattr(ruler_operator, "_clone_root", lambda world: ("clone", world))
    monkeypatch.setattr(l1, "_sha256_bytes", lambda value: "a" * 64)

    operator._ensure_root(fixture)
    assert order == [("install", "world", 800), ("digest", "world")]
    assert operator._exact._roots == {"frozen-800": ("clone", "world")}


def test_cooperative_stop_and_deadline_fail_before_next_step(monkeypatch):
    class Stop:
        def is_set(self):
            return True

    monkeypatch.setattr(l1, "_WORKER_STOP", Stop())
    monkeypatch.setattr(l1, "_WORKER_DEADLINE", float("inf"))
    with pytest.raises(RuntimeError, match="stopped before next"):
        l1._check_worker_stop()
    monkeypatch.setattr(l1, "_WORKER_STOP", None)
    monkeypatch.setattr(l1, "_WORKER_DEADLINE", -1)
    with pytest.raises(RuntimeError, match="stopped before next"):
        l1._check_worker_stop()


def test_pool_submission_failure_signals_stop_and_joins(monkeypatch):
    class Stop:
        value = False

        def set(self):
            self.value = True

        def is_set(self):
            return self.value

    stop = Stop()

    class Context:
        def Event(self):
            return stop

    class Future:
        def done(self):
            return True

        def result(self):
            raise RuntimeError("worker stopped")

    class Pool:
        submissions = 0
        joined = False

        def submit(self, *args):
            self.submissions += 1
            if self.submissions == 2:
                raise RuntimeError("submit failed after first admission")
            return Future()

        def shutdown(self, **kwargs):
            self.joined = True

    pool = Pool()
    monkeypatch.setattr(l1.multiprocessing, "get_context", lambda _: Context())
    monkeypatch.setattr(l1.concurrent.futures, "ProcessPoolExecutor", lambda **kw: pool)
    monkeypatch.setattr(l1.concurrent.futures, "wait", lambda *a, **kw: None)
    monkeypatch.setattr(l1, "_semantic_reference", lambda _: {})
    result = l1.run_stress_cell(
        {"device": "cpu", "workers": 2, "max_pending": 8},
        {"mix": "frozen", "fixture_id": "frozen-800", "target_length": 800},
        {},
        burst_seconds=12,
        guard=lambda: {},
    )
    assert result["status"] == "STOPPED"
    assert "submit failed" in result["stop_reason"]
    assert stop.value and pool.joined
    assert result["admitted_roots"] == result["actual_max_pending"] == 1
    assert result["valid_consumptions"] == 0


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_bad_burst_rejected_before_allocating_workers(duration):
    with pytest.raises(ValueError, match="finite and positive"):
        l1.run_stress_cell({}, {}, {}, burst_seconds=duration, guard=lambda: {})
