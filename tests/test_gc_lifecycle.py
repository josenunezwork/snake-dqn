"""Permanent subprocess d.3 retention check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest

if TYPE_CHECKING:
    from src.game.snake_geometry import SnakeGeometry

Census = Callable[..., dict[str, Any]]


@pytest.fixture
def census(monkeypatch: pytest.MonkeyPatch) -> Census:
    """Inspect only fixture objects; never run a session or alter the real cache."""
    from research.bl_gc import lifecycle
    from src.game import snake_geometry

    monkeypatch.setattr(snake_geometry, "_cache", {})
    types = lifecycle._gameplay_types()

    def inspect(objects: list[Any], *, birth: bool = False) -> dict[str, Any]:
        monkeypatch.setattr(lifecycle.gc, "get_objects", lambda: objects)
        return lifecycle.gameplay_census(types, birth=birth)

    return inspect


def _cached_geometry() -> SnakeGeometry:
    from src.game import snake_geometry

    geometry = snake_geometry.SnakeGeometry(((10, 10), (20, 10)), 2, 10)
    snake_geometry._cache[0] = (((10, 10), (20, 10)), 2, 10, geometry)
    return geometry


def test_census_accepts_only_module_shapes_identity(census: Census) -> None:
    from src.model.obs_spec import RASTER31V2_SHAPES, RasterObsShapes

    allowed = census([RASTER31V2_SHAPES], birth=True)
    assert allowed["clean"] and allowed["total"] == 0
    assert allowed["A1_count"] == 1 and allowed["A2_count"] == 0
    assert allowed["P"]["total"] == allowed["A"]["total"] == 1
    rejected = census([RasterObsShapes()])
    assert not rejected["clean"] and rejected["total"] == 1
    assert rejected["A1_count"] == 0


def test_census_rejects_geometry_outside_owner_cache(census: Census) -> None:
    from src.game.snake_geometry import SnakeGeometry

    result = census([SnakeGeometry(((10, 10),), 1, 10)])
    assert not result["clean"] and result["total"] == 1
    assert result["A2_count"] == 0


def test_census_rejects_cached_geometry_at_birth_only(census: Census) -> None:
    geometry = _cached_geometry()
    birth = census([geometry], birth=True)
    assert not birth["clean"] and birth["total"] == 1
    assert birth["A2_count"] == 1
    teardown = census([geometry])
    assert teardown["clean"] and teardown["total"] == 0
    assert teardown["A2_count"] == 1 and not teardown["isolation_failures"]


@pytest.mark.parametrize("module", ["src.game.new", "src.model.new", "web.backend.session.new"])
def test_census_rejects_unlisted_prefix_object(census: Census, module: str) -> None:
    obj = type("Unlisted", (), {"__module__": module})()
    result = census([obj])
    assert not result["clean"] and result["total"] == 1
    assert result["P"]["types"] == {f"{module}.Unlisted": 1}
    assert result["A"]["total"] == 0


@pytest.mark.parametrize("root", ["A1", "A2"])
@pytest.mark.parametrize("target", ["strict", "prefix", "unowned_shapes", "unowned_geometry"])
def test_census_allowlisted_object_must_be_isolated(
    census: Census, monkeypatch: pytest.MonkeyPatch, root: str, target: str
) -> None:
    from src.game.snake import Snake
    from src.game.snake_geometry import SnakeGeometry
    from src.model.obs_spec import RASTER31V2_SHAPES, RasterObsShapes

    if target == "strict":
        retained = object.__new__(Snake)
    elif target == "prefix":
        retained = type("Unlisted", (), {"__module__": "src.game.new"})()
    elif target == "unowned_shapes":
        retained = RasterObsShapes()
    else:
        retained = SnakeGeometry((), 0, 10)
    # A container cycle proves termination; the target is absent from the census
    # list so it can only be detected through the transitive referent walk.
    containers = [{"nested": (frozenset({1}), [retained])}]
    containers.append(containers)
    if root == "A1":
        obj = RASTER31V2_SHAPES
        monkeypatch.setitem(vars(obj), "test_retained", containers)
    else:
        obj = _cached_geometry()
        obj.obstacle_cells[(0, 0)] = containers
    result = census([obj])
    assert not result["clean"] and result["total"] == 1
    assert result["isolation_failures"][0]["owner"] == root
    assert sum(result["isolation_failures"][0]["reached"].values()) == 1


def test_census_cache_bound_and_exact_geometry_type(census: Census) -> None:
    from src.game import snake_geometry

    geometry = _cached_geometry()
    for key in range(64):
        snake_geometry._cache[key] = ((), 0, 10, geometry)
    assert census([geometry])["clean"]
    snake_geometry._cache[64] = ((), 0, 10, geometry)
    result = census([geometry])
    assert not result["clean"] and not result["cache_bound_ok"]
    assert result["cache_size"] == 65 and result["A2_count"] == 0


def test_census_rejects_geometry_subclass_even_in_cache(census: Census) -> None:
    from src.game import snake_geometry

    child = type("ChildGeometry", (snake_geometry.SnakeGeometry,), {"__module__": "src.game.test"})
    geometry = child((), 0, 10)
    snake_geometry._cache[0] = ((), 0, 10, geometry)
    result = census([geometry])
    assert not result["clean"] and result["A2_count"] == 0


@pytest.mark.parametrize("index", range(8))
def test_census_rejects_each_live_strict_type_at_teardown(census: Census, index: int) -> None:
    from research.bl_gc.lifecycle import _gameplay_types

    cls = _gameplay_types()[index]
    obj = object.__new__(cls)
    result = census([obj])
    assert not result["clean"] and result["total"] == result["S"]["total"] == 1
    assert result["A"]["total"] == 0


def test_census_strict_type_has_no_allowlist_exception(
    census: Census, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.game.snake import Snake
    from src.model import obs_spec

    obj = object.__new__(Snake)
    monkeypatch.setattr(obs_spec, "RASTER31V2_SHAPES", obj)
    result = census([obj])
    assert not result["clean"] and result["S"]["total"] == 1
    assert result["A1_count"] == 0


def _run(tmp_path: Path, policy: str) -> dict:
    path = tmp_path / f"lifecycle-{policy}.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.bl_gc.lifecycle",
            "--policy",
            policy,
            "--out",
            str(path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, SNAKE_SERVE_GC_POLICY=policy, SNAKE_SERVE_PATHB="0"),
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]
    assert path.with_suffix(".json.sha256").is_file()
    assert path.with_suffix(".json.complete").read_text() == "complete\n"
    return json.loads(path.read_text())


@pytest.mark.parametrize("policy", ["default", "freeze_startup"])
def test_watch_session_replacements_release_under_both_gc_policies(
    tmp_path: Path, policy: str
) -> None:
    result = _run(tmp_path, policy)
    assert result["cycles"] == 20
    assert result["tracked_objects"]["cycle_20"] <= result["tracked_objects"]["cycle_5"] + 2000

    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["weakrefs_cleared"] and result["teardown_weakrefs_cleared"]
    assert result["birth_census"]["total"] == 0
    assert result["teardown_census_pre_collect"]["total"] == 0
    assert result["freeze_calls_before_negative_control"] == int(policy == "freeze_startup")
    assert result["negative_control_retained_while_frozen"]
    assert result["negative_control_released_after_unfreeze"]
    assert result["gameplay_objects_released_by_teardown"] == 0
    assert len(result["cycle_checks"]) == 20
    for cycle, row in enumerate(result["cycle_checks"], 1):
        assert row["cycle"] == cycle
        assert all(
            row[key] for key in ("weakrefs_cleared", "freeze_non_increasing", "single_freeze")
        )
        assert all(value is not None for value in row.values())
    trajectory = result["freeze_count_trajectory"]
    assert len(trajectory) == 22  # receipt, after initialize, 20 boundaries
    assert all(b <= a <= result["startup_freeze_count"] for a, b in zip(trajectory, trajectory[1:]))
    assert result["freeze_count_total_decrease"] == trajectory[0] - trajectory[-1]
    assert (
        result["attestation"]["git_commit"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    assert type(result["attestation"]["git_dirty"]) is bool


def test_i1_real_frozen_decrease_passes_and_second_freeze_fails() -> None:
    code = """
import gc
from research.bl_gc.lifecycle import freeze_non_increasing
container = [object()]
gc.freeze()
baseline = gc.get_freeze_count()
del container
decreased = gc.get_freeze_count()
assert decreased < baseline
assert freeze_non_increasing([decreased], baseline)
young = [[] for _ in range(1000)]
gc.freeze()  # injected prohibited second freeze
increased = gc.get_freeze_count()
assert increased > decreased
assert not freeze_non_increasing([decreased, increased], baseline)
gc.unfreeze()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_census_detects_explicit_subclasses_and_module_prefixes() -> None:
    from research.bl_gc.lifecycle import gameplay_census

    class Gameplay:
        pass

    class Child(Gameplay):
        pass

    prefix_type = type("PrefixObject", (), {"__module__": "src.model.test_census"})
    objects = [Child(), prefix_type()]
    census = gameplay_census((Gameplay,))
    assert census["total"] >= len(objects)
    assert census["types"]["src.model.test_census.PrefixObject"] == 1


def test_failed_invariant_records_all_twenty_cycles_in_subprocess() -> None:
    code = """
import gc
from research.bl_gc import lifecycle as l
calls = 0
def cycle(seed):
    global calls
    calls += 1
    if calls == 1:
        gc.freeze()  # injected second freeze: I1/I2 FAIL; continue all cycles
    return (), {}
l._exercise_cycle = cycle
result = l.run("freeze_startup")
assert result["status"] == "FAIL", result
assert result["cycles"] == 20
assert result["checks"]["I1"] is False
assert result["checks"]["I2"] is False
assert result["freeze_calls_before_negative_control"] == 2
assert all(v is not None for row in result["cycle_checks"] for v in row.values())
assert result["negative_control_retained_while_frozen"]
assert result["negative_control_released_after_unfreeze"]
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_crash_preserves_reached_checks_and_nulls_only_unreached() -> None:
    code = """
import gc
from research.bl_gc import lifecycle as l
calls = 0
def cycle(seed):
    global calls
    calls += 1
    if calls == 2:
        raise RuntimeError("injected crash")
    return (), {}
l._exercise_cycle = cycle
r = l.run("default")
assert r["status"] == "FAIL" and r["cycles"] == 1
assert r["failure"]["message"] == "injected crash"
assert r["checks"]["I3"] is not None
assert r["checks"]["I1"] is not None
assert r["checks"]["weakrefs"] is True
assert all(v is not None for v in r["cycle_checks"][0].values())
assert all(v is None for k,v in r["cycle_checks"][1].items() if k != "cycle")
assert r["checks"]["I4"] is None
assert r["checks"]["negative_retained"] is None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
