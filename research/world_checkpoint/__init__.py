"""Exact mid-world checkpoint/resume for Watch6 paired-world simulations."""

from typing import Any

from research.world_checkpoint.checkpoint import (
    WORLD_CHECKPOINT_SCHEMA,
    CheckpointIdentityError,
    capture_runtime_identity,
    configure_cpu,
    find_latest_checkpoint,
    highest_checkpoint_seq_on_disk,
    load_checkpoint,
    quarantine_checkpoint_file,
    quarantine_incomplete_checkpoints,
    restore_live_session_state,
    restore_opponents_state,
    restore_rng_state,
    restore_run_session_state,
    restore_runtime_state,
    restore_sim_state,
    restore_trigger_state,
    save_checkpoint,
    snapshot_live_session_state,
    snapshot_opponents_state,
    snapshot_rng_state,
    snapshot_run_session_state,
    snapshot_runtime_state,
    snapshot_sim_state,
    snapshot_trigger_state,
    state_digest,
    verify_checkpoint,
    verify_checkpoint_identity,
    verify_restored_types,
    verify_session_parity,
)
from research.world_checkpoint.runner import (
    DEFAULT_CHECKPOINT_INTERVAL,
    GRACE_SECONDS,
    WORLD_UNIT_SCHEMA,
    CheckpointInterrupted,
    inject_live_and_mirror_serpentines,
    measure_checkpoint_scaling,
    run_paired_world_resumable,
    set_external_signal_getter,
)


def __getattr__(name: str) -> Any:
    if name in ("resumable_world_unit", "run", "stage_exported_checkpoints"):
        from research.world_checkpoint import job

        return getattr(job, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CheckpointIdentityError",
    "CheckpointInterrupted",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "GRACE_SECONDS",
    "WORLD_CHECKPOINT_SCHEMA",
    "WORLD_UNIT_SCHEMA",
    "capture_runtime_identity",
    "configure_cpu",
    "find_latest_checkpoint",
    "highest_checkpoint_seq_on_disk",
    "inject_live_and_mirror_serpentines",
    "load_checkpoint",
    "measure_checkpoint_scaling",
    "quarantine_checkpoint_file",
    "quarantine_incomplete_checkpoints",
    "restore_live_session_state",
    "restore_opponents_state",
    "restore_rng_state",
    "restore_run_session_state",
    "restore_runtime_state",
    "restore_sim_state",
    "restore_trigger_state",
    "resumable_world_unit",
    "run",
    "run_paired_world_resumable",
    "save_checkpoint",
    "set_external_signal_getter",
    "snapshot_live_session_state",
    "snapshot_opponents_state",
    "snapshot_rng_state",
    "snapshot_run_session_state",
    "snapshot_runtime_state",
    "snapshot_sim_state",
    "snapshot_trigger_state",
    "stage_exported_checkpoints",
    "state_digest",
    "verify_checkpoint",
    "verify_checkpoint_identity",
    "verify_restored_types",
    "verify_session_parity",
]
