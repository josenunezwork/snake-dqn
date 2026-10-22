"""CPU-only contracts for the CUDA-decision transport canary observer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.colab_cli import canary_observer as observer


class _Cuda:
    def __init__(self) -> None:
        self.synced: list[object] = []

    def synchronize(self, device: object) -> None:
        self.synced.append(device)


class _Device:
    def __init__(self, kind: str, label: str) -> None:
        self.type, self.label = kind, label

    def __str__(self) -> str:
        return self.label


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(cuda=_Cuda())


def test_observe_records_only_attributable_cuda_action_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    package = root / "repo" / "research" / "bl_ppo"
    package.mkdir(parents=True)
    source = package / "driver.py"
    source.write_text("# caller marker\n", encoding="utf-8")
    tensor = SimpleNamespace(shape=(4, 6), device=_Device("cuda", "cuda:0"))
    torch = _fake_torch()
    namespace = {
        "observer": observer,
        "tensor": tensor,
        "torch": torch,
        "root": root,
        "package": "research/bl_ppo",
        "package_root": package,
    }
    exec(
        compile(
            "result = observer._observe('torch.argmax', tensor, lambda: 'action', "
            "root=root, package_root=package_root, package=package, torch=torch)",
            str(source),
            "exec",
        ),
        namespace,
    )

    assert namespace["result"] == "action"
    assert torch.cuda.synced == [tensor.device]
    rows = observer.collect(root, "research/bl_ppo")
    assert rows[0]["cuda_decisions"] == 1
    assert rows[0]["operation"] == "torch.argmax"
    assert rows[0]["caller"] == "research/bl_ppo/driver.py:1:<module>"
    assert rows[0]["shape"] == [4, 6]


def test_observe_does_not_count_cpu_or_non_action_shape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    package = root / "repo" / "research" / "bl_ppo"
    package.mkdir(parents=True)
    torch = _fake_torch()
    cpu = SimpleNamespace(shape=(4, 6), device=_Device("cpu", "cpu"))
    wrong_shape = SimpleNamespace(shape=(4, 5), device=_Device("cuda", "cuda:0"))

    assert (
        observer._observe(
            "torch.argmax",
            cpu,
            lambda: "cpu",
            root=root,
            package_root=package,
            package="research/bl_ppo",
            torch=torch,
        )
        == "cpu"
    )
    assert (
        observer._observe(
            "torch.argmax",
            wrong_shape,
            lambda: "wrong",
            root=root,
            package_root=package,
            package="research/bl_ppo",
            torch=torch,
        )
        == "wrong"
    )
    assert torch.cuda.synced == []
    assert not (root / "canary-observations").exists()


def test_install_wraps_torch_action_apis_and_preserves_cpu_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "repo" / "research" / "bl_ppo").mkdir(parents=True)
    calls: list[str] = []

    class Tensor:
        shape = (1, 6)
        device = _Device("cpu", "cpu")

        def argmax(self) -> str:
            calls.append("method")
            return "method-result"

    fake_torch = SimpleNamespace(
        Tensor=Tensor,
        argmax=lambda tensor: calls.append("function") or "function-result",
        multinomial=lambda tensor, count: calls.append("multinomial") or "sample-result",
        cuda=_Cuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("SNAKE_COLAB_CANARY", "1")
    monkeypatch.setenv("COLAB_CLI_ROOT", str(root))
    monkeypatch.setenv("SNAKE_COLAB_CANARY_PACKAGE", "research/bl_ppo")
    monkeypatch.setattr(observer, "_INSTALLED", False)

    observer.install()

    tensor = Tensor()
    assert tensor.argmax() == "method-result"
    assert fake_torch.argmax(tensor) == "function-result"
    assert fake_torch.multinomial(tensor, 1) == "sample-result"
    assert calls == ["method", "function", "multinomial"]
    assert fake_torch.cuda.synced == []


def test_collect_rejects_foreign_or_malformed_rows(tmp_path: Path) -> None:
    root = tmp_path / "root"
    observations = root / "canary-observations"
    observations.mkdir(parents=True)
    (observations / "1.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "package": "research/other",
                "cuda_decisions": 1,
                "operation": "torch.argmax",
                "caller": "research/other/run.py:1:main",
                "device": "cuda:0",
                "shape": [1, 6],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(observer.CanaryObservationError, match="foreign"):
        observer.collect(root, "research/bl_ppo")


def test_bootstrap_installs_pinned_observer_via_pth() -> None:
    script = (Path(__file__).parents[1] / "research" / "colab_cli" / "bootstrap.sh").read_text()
    assert "snake_colab_canary.pth" in script
    assert "importlib.import_module('research.colab_cli.canary_observer').install()" in script
    assert "export PYTHONPATH" not in script
