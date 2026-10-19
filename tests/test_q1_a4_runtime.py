"""A4 durable partial import and fresh-attempt safety contracts."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from research.q0q1_20261017 import run_queue_job as subject


def _archive(path: Path, files: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:gz") as stream:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            stream.addfile(member, io.BytesIO(raw))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resume_import_requires_bound_archive_and_registered_partial_names(tmp_path):
    archive = tmp_path / "old.tar.gz"
    prefix = "attempt/evaluation/seed-261701/B8/shard-0-of-4.partials/"
    sha = _archive(archive, {prefix + "manifest.json": b"{}", prefix + "world-0000.json": b"{}"})
    env = {
        "Q1_RESUME_ARCHIVE": str(archive),
        "Q1_RESUME_ARCHIVE_SHA256": sha,
        "Q1_SHARD": "0/4",
        "Q1_SEED": "261701",
        "Q1_ARM": "B8",
    }
    subject.stage_evaluation_partials(env, tmp_path / "new")
    assert (tmp_path / "new" / prefix.removeprefix("attempt/") / "manifest.json").is_file()
    env["Q1_RESUME_ARCHIVE_SHA256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256"):
        subject.stage_evaluation_partials(env, tmp_path / "wrong")
    env["Q1_RESUME_ARCHIVE_SHA256"] = _archive(
        archive, {prefix + "manifest.json": b"{}", prefix + "../escape": b"bad"}
    )
    with pytest.raises(RuntimeError, match="unsafe"):
        subject.stage_evaluation_partials(env, tmp_path / "unsafe")


def test_fresh_attempt_permits_only_empty_queue_precreated_export_dir(tmp_path):
    partials = tmp_path / "evaluation-partials"
    partials.mkdir()
    subject.validate_output_directory(tmp_path, {"Q0Q1_JOB_KIND": "q1_eval"})
    (partials / "world-0000.json").write_text("{}")
    with pytest.raises(RuntimeError, match="prior or foreign"):
        subject.validate_output_directory(tmp_path, {"Q0Q1_JOB_KIND": "q1_eval"})
