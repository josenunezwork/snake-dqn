"""Immutable input lineage for fresh-RNG Apex warm starts and continuations."""

import copy
import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def resume_parent(content_sha256: str, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the exact bytes deserialized, without consulting the path again."""
    runtime = checkpoint.get("apex_recipe_runtime")
    return {
        "source_content_sha256": content_sha256,
        "source_run_seed_manifest": copy.deepcopy(checkpoint.get("run_seed_manifest")),
        "source_recipe_runtime_seed_identity": (
            copy.deepcopy(runtime.get("seed_identity")) if isinstance(runtime, Mapping) else None
        ),
        "rng_state_restored": False,
        "replay_state_restored": False,
    }


def load_checkpoint_snapshot(
    path: str | Path, map_location: Any = "cpu"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and deserialize one private snapshot with bounded copying memory.

    Path replacement after opening cannot change the opened source. Detect
    in-place writes while copying; deserialize only the private temporary file.
    Embedded replay may be large, so never retain a second full byte buffer.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as source, tempfile.TemporaryFile(mode="w+b") as snapshot:
        before = os.fstat(source.fileno())
        while block := source.read(1024 * 1024):
            snapshot.write(block)
            digest.update(block)
        after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("checkpoint changed while its immutable snapshot was copied")
        snapshot.flush()
        os.fsync(snapshot.fileno())
        snapshot.seek(0)
        checkpoint = torch.load(snapshot, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return checkpoint, resume_parent(digest.hexdigest(), checkpoint)
