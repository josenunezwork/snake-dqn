"""Unit tests for the explicitly loaded factory pytest plugin."""

from __future__ import annotations

import pytest

from research.compute import factory_companion_pytest as plugin


def test_plugin_refuses_missing_cpu_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNAKE_DQN_DEVICE", raising=False)
    with pytest.raises(pytest.UsageError, match="SNAKE_DQN_DEVICE"):
        plugin.pytest_configure(None)  # type: ignore[arg-type]


class _Item:
    nodeid = "tests/test_gpu.py::test_x"
    fixturenames = ("cuda",)

    @staticmethod
    def get_closest_marker(_name: str) -> None:
        return None


def test_plugin_refuses_gpu_fixture() -> None:
    with pytest.raises(pytest.UsageError, match="GPU/MPS"):
        plugin.pytest_collection_modifyitems([_Item()])  # type: ignore[list-item]
