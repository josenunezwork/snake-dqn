"""Content-addressed, immutable inputs for a single evaluation invocation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

CHECKPOINT_DIGEST = "sha256"
EVALUATOR_ARTIFACT_VERSION = "evaluation-input-snapshot-v1"
_COPY_CHUNK_BYTES = 1024 * 1024


class SnapshotError(RuntimeError):
    """Raised when an evaluation input cannot be captured consistently."""


@dataclass(frozen=True)
class CheckpointSnapshot:
    """The immutable checkpoint bytes used by an evaluation command."""

    artifact_kind: str
    source_path: str
    source_identity: Dict[str, int]
    sha256: str
    size_bytes: int
    snapshot_path: str

    def receipt(self) -> Dict[str, Any]:
        """Return a JSON-safe provenance record."""
        return asdict(self)


AgentSpec = Tuple[str, str]


def _identity(file_stat: os.stat_result) -> Dict[str, int]:
    """Return the file attributes that establish one opened file identity."""
    return {
        "device": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
        "size_bytes": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
    }


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    """Tell whether writes changed a source while it was being copied."""
    return _identity(before) == _identity(after)


def _sha256_path(path: Path) -> str:
    """Hash a stable local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


class EvaluationArtifacts:
    """Freeze checkpoint bytes and write a durable evaluation-input receipt.

    The source is copied from one file descriptor. Its identity is checked both
    before and after the copy, so an in-place write aborts the command. An
    atomic replacement changes the pathname but not the already-open descriptor,
    leaving a complete snapshot of the initial file rather than mixed bytes.
    """

    def __init__(self, root: str | Path) -> None:
        self.parent = Path(root).expanduser().resolve()
        self.root = self.parent / f"run-{uuid.uuid4().hex}"
        self._by_source: Dict[str, CheckpointSnapshot] = {}
        self._by_digest: Dict[str, CheckpointSnapshot] = {}

    def snapshot_checkpoint(self, source_path: str | Path) -> CheckpointSnapshot:
        """Copy one checkpoint into this run's content-addressed input store.

        Args:
            source_path: Trusted local checkpoint path supplied to the evaluator.

        Returns:
            The immutable snapshot and its source provenance.

        Raises:
            SnapshotError: If the path is invalid, mutates while copied, or an
                existing content-addressed snapshot is corrupted.
        """
        return self._snapshot_file(source_path, category="checkpoints", suffix=".pth")

    def snapshot_config(self, source_path: str | Path) -> CheckpointSnapshot:
        """Freeze a config before loading it into the evaluator."""
        return self._snapshot_file(source_path, category="configs", suffix=".yaml")

    def _snapshot_file(
        self, source_path: str | Path, *, category: str, suffix: str
    ) -> CheckpointSnapshot:
        """Copy one stable file descriptor into this invocation's input store."""
        try:
            source = Path(source_path).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise SnapshotError(f"checkpoint not found: {source_path}") from exc
        source_key = f"{category}:{source}"
        existing = self._by_source.get(source_key)
        if existing is not None:
            return existing

        self.root.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with source.open("rb") as input_file:
                before = os.fstat(input_file.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise SnapshotError(f"checkpoint is not a regular file: {source}")
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self.root, prefix=".checkpoint-", delete=False
                ) as output_file:
                    temp_path = Path(output_file.name)
                    for block in iter(lambda: input_file.read(_COPY_CHUNK_BYTES), b""):
                        digest.update(block)
                        output_file.write(block)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                after = os.fstat(input_file.fileno())

            if not _same_identity(before, after):
                raise SnapshotError(f"checkpoint changed while being snapshotted: {source}")

            sha256 = digest.hexdigest()
            destination = self.root / CHECKPOINT_DIGEST / category / f"{sha256}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _sha256_path(destination) != sha256:
                    raise SnapshotError(f"corrupted content-addressed snapshot: {destination}")
                temp_path.unlink()
            else:
                os.replace(temp_path, destination)
                destination.chmod(0o444)
            temp_path = None

            snapshot = CheckpointSnapshot(
                artifact_kind=category,
                source_path=str(source),
                source_identity=_identity(before),
                sha256=sha256,
                size_bytes=int(before.st_size),
                snapshot_path=str(destination),
            )
            self._by_source[source_key] = snapshot
            self._by_digest.setdefault(sha256, snapshot)
            return snapshot
        except FileNotFoundError as exc:
            raise SnapshotError(f"checkpoint not found: {source_path}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def snapshot_agent_specs(self, specs: Sequence[AgentSpec]) -> List[AgentSpec]:
        """Replace checkpoint references with immutable paths, preserving scripted specs."""
        frozen: List[AgentSpec] = []
        for kind, reference in specs:
            if kind == "checkpoint":
                frozen.append((kind, self.snapshot_checkpoint(reference).snapshot_path))
            else:
                frozen.append((kind, reference))
        return frozen

    def checkpoint_receipts(self) -> List[Dict[str, Any]]:
        """Return one receipt per source path in deterministic input order."""
        return [
            snapshot.receipt()
            for snapshot in self._by_source.values()
            if snapshot.artifact_kind == "checkpoints"
        ]

    def verify_integrity(self) -> None:
        """Fail if any captured input bytes changed after snapshot publication."""
        for snapshot in self._by_source.values():
            path = Path(snapshot.snapshot_path)
            try:
                file_stat = path.stat()
            except FileNotFoundError as exc:
                raise SnapshotError(f"evaluation snapshot disappeared: {path}") from exc
            if file_stat.st_size != snapshot.size_bytes or _sha256_path(path) != snapshot.sha256:
                raise SnapshotError(f"evaluation snapshot was modified after capture: {path}")

    def write_receipt(
        self,
        *,
        config_snapshot: CheckpointSnapshot,
        effective_config: Any,
        evaluator_path: str | Path,
        source_specs: Iterable[AgentSpec],
        evaluator_sources: Iterable[str | Path] | None = None,
    ) -> Path:
        """Persist the frozen inputs and evaluator/config identities as JSON."""
        self.root.mkdir(parents=True, exist_ok=True)
        evaluator = Path(evaluator_path).expanduser().resolve(strict=True)
        if hasattr(effective_config, "__dataclass_fields__"):
            effective = asdict(effective_config)
            # AppConfig retains supplied YAML paths as a frozen set. Keep that
            # provenance in the receipt using a deterministic JSON sequence.
            if "provided_fields" in effective:
                effective["provided_fields"] = sorted(effective["provided_fields"])
        else:
            effective = effective_config
        payload = {
            "artifact_version": EVALUATOR_ARTIFACT_VERSION,
            "checkpoint_digest_algorithm": CHECKPOINT_DIGEST,
            "checkpoint_snapshots": self.checkpoint_receipts(),
            "config": {
                **config_snapshot.receipt(),
                "effective": effective,
            },
            "evaluator": evaluator_provenance(evaluator, evaluator_sources),
            "source_agents": [{"kind": kind, "reference": ref} for kind, ref in source_specs],
        }
        receipt_path = self.root / "receipt.json"
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.root, prefix=".receipt-", suffix=".json", delete=False
        ) as output_file:
            temp_path = Path(output_file.name)
            output_file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, receipt_path)
        return receipt_path


def evaluator_provenance(
    evaluator_path: str | Path, source_paths: Iterable[str | Path] | None = None
) -> Dict[str, Any]:
    """Return source-closure hashes plus the repository revision/dirty state."""
    evaluator = Path(evaluator_path).expanduser().resolve(strict=True)
    repository = evaluator.parents[2]
    source_paths = source_paths or _default_evaluator_sources(repository)
    manifest: Dict[str, str] = {}
    for source in source_paths:
        path = Path(source).expanduser().resolve(strict=True)
        try:
            label = str(path.relative_to(repository))
        except ValueError:
            label = str(path)
        manifest[label] = _sha256_path(path)
    return {
        "source_path": str(evaluator),
        "sha256": _sha256_path(evaluator),
        "source_manifest": dict(sorted(manifest.items())),
        "git": _git_identity(repository),
    }


def _default_evaluator_sources(repository: Path) -> List[Path]:
    """List the local modules that define the current tournament evaluator."""
    relative_paths = (
        "src/scripts/tournament_eval.py",
        "src/scripts/eval_cli.py",
        "src/scripts/eval_stats.py",
        "src/core/config_loader.py",
        "src/core/game_config.py",
        "src/game/game_state_factory.py",
        "src/game/scripted_snake.py",
        "src/game/snake_factory.py",
        "src/training/behavior_probes.py",
        "src/model/inference_agent.py",
        "src/simd_env/eval_engine.py",
    )
    return [
        repository / relative for relative in relative_paths if (repository / relative).exists()
    ]


def _git_identity(repository: Path) -> Dict[str, Any]:
    """Report a dirty checkout honestly without requiring Git for test fixtures."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "diff_sha256": None}
