"""Ensure pyproject.toml mirrors requirements.txt and requirements-dev.txt."""

from __future__ import annotations

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _parse_requirements(path: Path) -> list[str]:
    specs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        specs.append(line)
    return specs


def test_pyproject_mirrors_requirements_files_and_omits_placeholders() -> None:
    """requirements.txt and requirements-dev.txt are the sources of truth."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    expected_runtime = _parse_requirements(ROOT / "requirements.txt")
    expected_dev = _parse_requirements(ROOT / "requirements-dev.txt")

    assert project["dependencies"] == expected_runtime
    assert project["optional-dependencies"]["dev"] == expected_dev

    assert "authors" not in project
    assert "urls" not in project
    assert "pytest" not in pyproject.get("tool", {})
