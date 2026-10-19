"""Exercise the macOS runtime fence with real threaded torch/sklearn work."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests import openmp_runtime


def test_duplicate_runtime_guard_reports_both_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ["/torch/lib/libomp.dylib", "/sklearn/.dylibs/libomp.dylib"]
    monkeypatch.setattr(openmp_runtime, "loaded_libomp_images", lambda: paths)
    with pytest.raises(AssertionError, match="Multiple libomp") as error:
        openmp_runtime.assert_single_libomp()
    assert all(path in str(error.value) for path in paths)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS wheel runtime regression")
def test_threaded_torch_then_sklearn_uses_one_runtime() -> None:
    spec = find_spec("torch")
    if spec is None or spec.origin is None:
        pytest.skip("torch is not installed")
    if not (Path(spec.origin).parent / "lib" / "libomp.dylib").is_file():
        pytest.skip("torch does not bundle libomp")
    # Exec, never fork: the loader must see the routing before loading either wheel.
    code = """
import importlib.util
import numpy as np
import torch
from tests.openmp_runtime import assert_single_libomp, loaded_libomp_images
if importlib.util.find_spec("sklearn") is None:
    raise SystemExit(77)
from sklearn.ensemble import HistGradientBoostingClassifier
torch.set_num_threads(2)
for seed in range(3):
    x = np.random.default_rng(seed).normal(size=(256, 16))
    y = np.arange(len(x)) % 2
    HistGradientBoostingClassifier(max_iter=3).fit(x, y)
    tensor = torch.from_numpy(x)
    assert torch.isfinite(tensor.T @ tensor).all()
    assert repr(np.array([1., 2., 3.])) == 'array([1., 2., 3.])'
    assert_single_libomp()
images = loaded_libomp_images()
assert len(images) == 1, images
print(images[0])
"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["DYLD_PRINT_LIBRARIES"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("sklearn is not installed")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "/torch/lib/libomp.dylib" in result.stdout
    assert "/sklearn/.dylibs/libomp.dylib" not in result.stderr


def test_failure_reporting_does_not_repr_closed_shared_memory(tmp_path: Path) -> None:
    """An ordinary actor assertion must remain a failure, never a native crash."""
    test_file = tmp_path / "test_closed_mapping.py"
    test_file.write_text(
        "import mmap\n"
        "import numpy as np\n"
        "def test_closed_mapping():\n"
        "    backing = mmap.mmap(-1, 4096)\n"
        "    view = np.ndarray((8,), dtype=np.int64, buffer=backing)\n"
        "    backing.close()\n"
        "    raise AssertionError('closed-mapping-failure-sentinel')\n"
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(root / "pytest.ini"), str(test_file)],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "closed-mapping-failure-sentinel" in result.stdout
    assert "1 failed" in result.stdout
    assert "Fatal Python error" not in result.stderr
