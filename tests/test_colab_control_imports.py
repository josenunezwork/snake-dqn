"""Reject parent code before it can execute in control infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.ops.control_imports import PACKAGE_FILES, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "injected", [None, "research/__init__.py", "research/compute/__init__.py", "research.py"]
)
def test_pinned_namespaces_refuse_untracked_parent_code_without_execution(tmp_path, injected):
    (tmp_path / "research/compute").mkdir(parents=True)
    for relative in PACKAGE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    marker = tmp_path / "must-not-execute"
    if injected:
        (tmp_path / injected).write_text(f"open({str(marker)!r}, 'w').close()\n")
    names = ("research", "research.compute")
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        if injected:
            with pytest.raises(RuntimeError, match="namespace"):
                verify(tmp_path)
        else:
            verify(tmp_path)
        assert not marker.exists()
    finally:
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
