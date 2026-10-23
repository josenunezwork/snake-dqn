"""Real isolated-process regression for Mac dependency leakage and lazy imports."""

import shutil
import subprocess

import pytest

from research.a100_a2 import make_pack
from tools.ops.import_preflight import check, import_statements


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def fixture_repo(path):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Preflight test")
    git(path, "config", "user.email", "preflight@example.invalid")
    (path / "job.py").write_text("def run():\n    import pydantic\n")
    (path / "requirements.txt").write_text("")
    git(path, "add", ".")
    git(path, "commit", "-qm", "Fixture")
    return git(path, "rev-parse", "HEAD")


def test_clean_environment_refuses_ambient_and_lazy_dependency(tmp_path, monkeypatch):
    if not shutil.which("uv"):
        pytest.fail("uv is required for the real isolated preflight regression")
    repo = tmp_path / "repo"
    commit = fixture_repo(repo)
    # Ambient PYTHONPATH could satisfy the lazy import if isolation regresses.
    poison = tmp_path / "ambient"
    poison.mkdir()
    (poison / "pydantic.py").write_text("raise AssertionError('ambient leak')\n")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    with pytest.raises(subprocess.CalledProcessError) as failure:
        check(repo, commit, ["job.py", "requirements.txt"], "requirements.txt", ["job"])
    assert "ModuleNotFoundError: No module named 'pydantic'" in failure.value.output
    assert "ambient leak" not in failure.value.output
    # Working-tree changes cannot repair the pinned source's missing dependency.
    (repo / "job.py").write_text("import json\n")
    with pytest.raises(subprocess.CalledProcessError, match="non-zero"):
        check(repo, commit, ["job.py", "requirements.txt"], "requirements.txt", ["job"])
    git(repo, "add", "job.py")
    git(repo, "commit", "-qm", "Remove fixture dependency")
    report = check(
        repo,
        git(repo, "rev-parse", "HEAD"),
        ["job.py", "requirements.txt"],
        "requirements.txt",
        ["job"],
    )
    assert report["status"] == "PASS"
    assert report["installed"] == []
    assert report["python"].startswith("3.12.")
    assert report["user_site"] is False
    assert report["system_site_packages"] is False
    assert report["local_modules"] == {"job": "job.py"}


def test_relative_and_selected_lazy_imports(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "entry.py").write_text(
        "from . import helper\ndef work():\n    from .helper import run\n"
        "def unused():\n    import absent\n"
    )
    assert import_statements(tmp_path, "pkg.entry:work") == ["from pkg.helper import run"]
    assert "from pkg import helper" in import_statements(tmp_path, "pkg.entry")
    assert "import absent" in import_statements(tmp_path, "pkg.entry")


def test_a2_preflight_failure_prevents_pack_publication(tmp_path, monkeypatch):
    from tools.ops import import_preflight, qualification

    repo = tmp_path / "repo"
    commit = fixture_repo(repo)
    git(repo, "update-ref", "refs/remotes/origin/main", commit)
    monkeypatch.setattr(make_pack, "ROOT", repo)
    monkeypatch.setattr(qualification, "check", lambda *a: make_pack.QUALIFIED_TREE)
    calls = []

    def fail(*args):
        calls.append(args)
        raise ValueError("isolated import failed")

    monkeypatch.setattr(import_preflight, "check", fail)
    output = tmp_path / "pack"
    with pytest.raises(ValueError, match="isolated import failed"):
        make_pack.make_pack(tmp_path / "inputs", output, commit)
    assert not output.exists()
    assert calls[0][1:] == (
        commit,
        make_pack.SOURCE_PATHS,
        make_pack.REQUIREMENTS,
        make_pack.IMPORT_ROOTS,
    )


def test_replacement_charges_failed_canary_to_original_series(tmp_path):
    import json

    receipt = tmp_path / "receipt.json"
    prior = dict(
        status="INCOMPLETE",
        stop_verified=True,
        failure_class="CHILD_NONZERO_EXIT",
        exports=[],
        job_id="a100-a2-canary-old",
        error="ModuleNotFoundError: No module named 'pydantic'",
        elapsed_seconds=792.4886501249857,
    )
    receipt.write_text(json.dumps(prior))
    envelope = make_pack.replacement_envelope(receipt)
    assert envelope["main_minutes"] == 256
    assert envelope["main_units"] == 42.79
    assert envelope["main_minutes"] + 18 + prior["elapsed_seconds"] / 60 <= 288
    assert envelope["main_units"] + 3 + envelope["prior_charged_units_at_ceiling"] <= 48
    prior["stop_verified"] = False
    receipt.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="mechanical"):
        make_pack.replacement_envelope(receipt)


def test_recursive_transitive_lazy_dependency_is_not_hidden(tmp_path):
    repo = tmp_path / "repo"
    fixture_repo(repo)
    (repo / "job.py").write_text("from helper import run\n")
    (repo / "helper.py").write_text("def run():\n    import pydantic\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Transitive lazy dependency")
    with pytest.raises(subprocess.CalledProcessError) as failure:
        check(
            repo,
            git(repo, "rev-parse", "HEAD"),
            ["job.py", "helper.py", "requirements.txt"],
            "requirements.txt",
            ["job"],
        )
    assert "ModuleNotFoundError: No module named 'pydantic'" in failure.value.output
