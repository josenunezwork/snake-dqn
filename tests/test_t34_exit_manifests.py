"""Transport-only checks for the unallocated T34 A5 and label drafts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from research.t34_exit_b8 import make_manifest as manifests
from research.t34_exit_b8 import public_inputs


@pytest.fixture(autouse=True)
def qualified_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate job shape tests from separately tested transport admission."""
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})


@pytest.mark.parametrize("entrypoint", ("benchmark", "label"))
def test_legacy_imports_cannot_reset_initialized_numba_ceiling(entrypoint: str) -> None:
    """Use two threads to expose the production-only five-to-one reset bug."""
    program = "\n".join(
        [
            f"from research.t34_exit_b8 import {entrypoint} as entrypoint",
            "entrypoint._cpu_ceiling(2)",
            "import os, numba",
            "import research.phase0_search_20261019.teacher",
            "assert os.environ['NUMBA_NUM_THREADS'] == '2'",
            "assert numba.get_num_threads() == 2",
            "f = numba.njit(lambda value: value + 1)",
            "assert f(1) == 2",  # compilation reloads Numba config after the import boundary
            "assert numba.get_num_threads() == 2",
        ]
    )
    env = dict(
        os.environ,
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        SNAKE_DQN_DEVICE="cpu",
    )
    subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )


DESTS = (
    "B8_s3_final.pth",
    "checkpoints/champion_a5_freespace_20260621.pth",
    "checkpoints/best_apex_fs.pth",
    "checkpoints/best_apex_stage1_fs.pth",
    "checkpoints/best_apex_pre_fs.pth",
    "checkpoints/frp3_m3_s12_u60000_20261005.pth",
)


def _inventory(tmp_path: Path) -> Path:
    rows = []
    for index, dest in enumerate(DESTS):
        source = tmp_path / f"input-{index}.pth"
        source.write_bytes(bytes([index]))
        rows.append(
            {
                "name": source.name,
                "local_path": str(source),
                "dest": dest,
                "size_bytes": 1,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"inputs": rows}))
    return path


def _catalog(tmp_path: Path, inventory: Path, *, urls: bool = False) -> Path:
    rows = json.loads(inventory.read_text())["inputs"]
    path = tmp_path / "public.json"
    path.write_text(
        json.dumps(
            {
                "inputs": [
                    {key: row[key] for key in ("name", "dest", "size_bytes", "sha256")}
                    | {"public_url": f"https://example.invalid/{row['name']}" if urls else None}
                    for row in rows
                ]
            }
        )
    )
    return path


def test_a5_public_draft_is_l4_only_and_publication_blocked(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    manifest = manifests.make_manifest(
        kind="benchmark",
        commit="a" * 40,
        output_dir=tmp_path / "fresh",
        inventory_path=inventory,
        public_catalog_path=_catalog(tmp_path, inventory),
    )
    assert manifest["transport"]["status"] == "PUBLICATION_BLOCKED"
    assert manifest["transport"]["dispatch_ready"] is False
    assert manifest["requirements"] == {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 12,
        "est_minutes": 60,
    }
    assert manifest["env"]["T34_DEVICE"] == "cpu"
    assert manifest["env"]["T34_NPROC"] == "12"
    assert manifest["env"]["T34_THREADS"] == "1"
    assert manifest["env"]["T34_WORLDS"] == "1"
    assert manifest["env"]["T34_MAX_WORLDS"] == "256"
    assert manifest["env"]["T34_HORIZON"] == "300"
    assert manifest["requirements"]["est_minutes"] == 60
    assert manifest["timeout_minutes"] == 120
    assert manifest["budget"]["unit_cap"] == 4.0
    assert manifest["budget"]["reserve_units"] == pytest.approx(0.8)
    with pytest.raises(manifests.ManifestRefused, match="publication is pending"):
        manifests.validate_manifest(manifest)


def test_label_manifest_uses_unique_round_shard_attempt_and_bounded_caps(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    manifest = manifests.make_manifest(
        kind="label",
        commit="a" * 40,
        output_dir=tmp_path / "fresh",
        inventory_path=inventory,
        public_catalog_path=_catalog(tmp_path, inventory),
        round_id="r1",
        shard=3,
        attempt=1,
        worlds=100,
    )
    assert manifest["job_id"] == "t34-exit-b8-r1-s3-a1"
    assert manifest["timeout_minutes"] == 360
    assert manifest["requirements"]["est_minutes"] == 180
    assert manifest["max_export_bytes"] == 64 * 1024**2
    assert manifest["env"]["T34_DEVICE"] == "cpu"
    assert manifest["env"]["T34_NPROC"] == "12"
    assert manifest["env"]["T34_THREADS"] == "1"
    assert manifest["env"]["T34_TELEMETRY_SECONDS"] == "120"


def test_local_upload_cannot_be_described_as_dispatch_ready(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    manifest = manifests.make_manifest(
        kind="benchmark",
        commit="a" * 40,
        output_dir=tmp_path / "fresh",
        inventory_path=inventory,
        local_upload_preflight=True,
    )
    assert manifest["claim"] == "LOCAL_UPLOAD_PREFLIGHT_ONLY"
    assert len(manifest["inputs"]) == 6
    assert manifest["transport"]["dispatch_ready"] is False


def test_catalog_identity_drift_is_refused(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    catalog = _catalog(tmp_path, inventory)
    data = json.loads(catalog.read_text())
    data["inputs"][0]["sha256"] = "0" * 64
    catalog.write_text(json.dumps(data))
    with pytest.raises(manifests.ManifestRefused, match="does not exactly bind"):
        manifests.make_manifest(
            kind="benchmark",
            commit="a" * 40,
            output_dir=tmp_path / "fresh",
            inventory_path=inventory,
            public_catalog_path=catalog,
        )


def test_public_input_materializer_verifies_archive_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b8 = b"b8"
    member = b"league-member"
    source = tmp_path / "member.pth"
    source.write_bytes(member)
    archive = tmp_path / "league.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="nested/member.pth")
    catalog = tmp_path / "catalog.json"
    rows = [
        {
            "name": "B8_s3_final.pth",
            "dest": "B8_s3_final.pth",
            "size_bytes": len(b8),
            "sha256": hashlib.sha256(b8).hexdigest(),
        }
    ]
    rows.extend(
        {
            "name": f"member{i}.pth",
            "dest": f"checkpoints/member{i}.pth",
            "size_bytes": len(member),
            "sha256": hashlib.sha256(member).hexdigest(),
        }
        for i in range(5)
    )
    # Supply five names from one safe archive only for this transport unit test.
    with tarfile.open(archive, "w:gz") as tar:
        for index in range(5):
            tar.add(source, arcname=f"nested/member{index}.pth")
    catalog.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": "B8_s3_final.pth",
                        "url": "https://example.invalid/b8",
                        "size_bytes": len(b8),
                        "sha256": hashlib.sha256(b8).hexdigest(),
                    },
                    {
                        "name": "league.tar.gz",
                        "url": "https://example.invalid/league",
                        "size_bytes": archive.stat().st_size,
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    },
                ],
                "inputs": rows,
            }
        )
    )
    data = {
        "https://example.invalid/b8": b8,
        "https://example.invalid/league": archive.read_bytes(),
    }
    monkeypatch.setattr(
        public_inputs,
        "EXPECTED_ASSETS",
        {
            "B8_s3_final.pth": (len(b8), hashlib.sha256(b8).hexdigest()),
            "league.tar.gz": (
                archive.stat().st_size,
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            ),
        },
    )
    monkeypatch.setattr(
        public_inputs,
        "EXPECTED_MEMBERS",
        {
            "B8_s3_final.pth": ("B8_s3_final.pth", len(b8), hashlib.sha256(b8).hexdigest()),
            **{
                f"member{i}.pth": (
                    f"checkpoints/member{i}.pth",
                    len(member),
                    hashlib.sha256(member).hexdigest(),
                )
                for i in range(5)
            },
        },
    )

    class Response:
        def __init__(self, content: bytes):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size: int):
            value, self.content = self.content[:size], self.content[size:]
            return value

    paths = public_inputs.materialize(
        catalog, tmp_path / "out", opener=lambda url, **_kwargs: Response(data[url])
    )
    assert len(paths) == 6
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize(
    "round_id,beta,interval",
    (("r0", "1", "4"), ("r1", "1", "8"), ("r2", "0.5", "8"), ("r3", "0", "8")),
)
def test_manifest_carries_round_schedule_and_trigger_overrides(
    tmp_path: Path, round_id: str, beta: str, interval: str
) -> None:
    inventory = _inventory(tmp_path)
    manifest = manifests.make_manifest(
        kind="label",
        commit="a" * 40,
        output_dir=tmp_path / "fresh",
        inventory_path=inventory,
        public_catalog_path=_catalog(tmp_path, inventory),
        round_id=round_id,
        worlds=2,
        triggers={"death_window": 75},
    )
    assert manifest["env"]["T34_BETA"] == beta
    assert manifest["env"]["T34_CONFIRM_INTERVAL"] == interval
    assert manifest["env"]["T34_DEATH_WINDOW"] == "75"


def test_manifest_refuses_unsupported_cpu_fanout_policy_factory(tmp_path: Path) -> None:
    with pytest.raises(manifests.ManifestRefused, match="unsupported for CPU process fan-out"):
        manifests.make_manifest(
            kind="label",
            commit="a" * 40,
            output_dir=tmp_path / "fresh",
            worlds=2,
            policy_factory="future.module:factory",
        )


@pytest.mark.parametrize(
    "options",
    (
        {"student_checkpoint": "student.pth"},
        {"checkpoint": "other.pth"},
        {"triggers": {"max_rewinds": 3}},
    ),
)
def test_unbound_students_and_excess_rewinds_are_refused(tmp_path: Path, options: dict) -> None:
    with pytest.raises(manifests.ManifestRefused):
        manifests.make_manifest(
            kind="label", commit="a" * 40, output_dir=tmp_path / "fresh", worlds=2, **options
        )


@pytest.mark.parametrize("entrypoint", ("benchmark", "label"))
def test_student_checkpoint_cannot_be_silently_ignored(entrypoint: str) -> None:
    import importlib

    module = importlib.import_module(f"research.t34_exit_b8.{entrypoint}")
    with pytest.raises(ValueError, match="requires --policy-factory"):
        module._load_policy({"student_checkpoint": "student.pth"})
