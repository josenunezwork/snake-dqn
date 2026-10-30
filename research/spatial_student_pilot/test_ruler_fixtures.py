"""Historical fixture isolation and real process-thread admission regressions."""

from pathlib import Path

import pytest
import torch

from research.spatial_student_pilot.ruler_fixtures import (
    THREAD_ENV,
    load_fixtures,
    thread_evidence,
)


def test_science_refuses_to_decode_historical_fixture(tmp_path: Path) -> None:
    # Refuse before even opening or unpickling a supplied scientific input.
    with pytest.raises(ValueError, match="ruler-only"):
        load_fixtures(tmp_path / "absent.pt", "pilot")
    with pytest.raises(ValueError, match="ruler-only"):
        load_fixtures(tmp_path / "absent.pt", "preflight")


def test_ruler_refuses_foreign_fixture(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.pt"
    torch.save(dict(schema="foreign", rows=[]), path)
    with pytest.raises(ValueError, match="foreign ruler fixture"):
        load_fixtures(path, "ruler")


def test_thread_environment_and_effective_pools_are_both_required(monkeypatch) -> None:
    for key in THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    assert thread_evidence(enforce=True)["environment"] == {key: "1" for key in THREAD_ENV}
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    with pytest.raises(ValueError, match="one-thread"):
        thread_evidence(enforce=True)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 8)
    with pytest.raises(ValueError, match="one-thread"):
        thread_evidence(enforce=True)


def test_science_rejects_partial_collection_or_fixture_injection() -> None:
    from research.spatial_student_pilot.collect import collect_batch
    from research.spatial_student_pilot.config import PilotConfig

    for probe in (dict(ruler_ticks=2), dict(ruler_seconds=1), dict(ruler_fixtures=[])):
        with pytest.raises(ValueError, match="ruler-only"):
            collect_batch(
                PilotConfig(),
                range(3),
                b8=None,
                networks={},
                device="cpu",
                stop=lambda: False,
                progress=lambda _: None,
                **probe,
            )


def test_native_warmup_reuses_verified_preloaded_event_binary(monkeypatch) -> None:
    """A loaded cache must avoid the inline event loader's cold compilation too."""
    from types import SimpleNamespace

    from research.gpu_sim_spike import event_cuda, flood_cuda, move_cuda, vector_geometry
    from research.spatial_student_pilot.ruler_fixtures import warm_native

    binary = SimpleNamespace(__file__="/fixture/verified-cache.so")
    monkeypatch.setattr(event_cuda, "_EXTENSION", binary)
    monkeypatch.setattr(flood_cuda, "_load", lambda: binary)
    monkeypatch.setattr(move_cuda, "_extension", lambda: binary)
    monkeypatch.setattr(vector_geometry, "_load_exact_extension", lambda: binary)
    import torch.utils.cpp_extension

    def refuse_compile(**kwargs):
        raise AssertionError("preloaded binary must not compile")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", refuse_compile)
    report = warm_native(lambda _: None)
    assert len(report["loaders"]) == 4
    assert all(row["binary"] == "verified-cache.so" for row in report["loaders"])
