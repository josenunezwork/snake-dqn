"""Compatibility facade and CLI entrypoint (`python -m research.v3_ppo.ppo`) for Game V3.1 PPO."""

from __future__ import annotations

from research.v3_ppo.checkpoint_io import (
    _INIT_ANCHOR_LIGHT_NAME,
    INIT_ANCHOR_LIGHT_NAME,
    V3_PPO_FULL_INDEX_SCHEMA,
    V3_PPO_LIGHT_CHECKPOINT_SCHEMA,
    V3_PPO_SIM_SHARD_SCHEMA,
    _ckpt_tag,
    _prune_work_checkpoints_after_rewind,
    ckpt_tag,
    concat_sim_state_shards,
    pack_sim_state_dict,
    prune_old_checkpoints,
    prune_work_checkpoints_after_rewind,
    publish_checkpoint_to_output,
    slice_sim_state_dict,
    unpack_sim_state_dict,
    verify_full_checkpoint_set,
    verify_light_checkpoint,
)
from research.v3_ppo.config import RESUME_MUTABLE_CONFIG_KEYS
from research.v3_ppo.exports import (
    RotatingCreateOnlyJSONLWriter,
    _artifact_or_sidecar_exists,
    _exported_payload_bytes,
    artifact_or_sidecar_exists,
    exported_payload_bytes,
    publish_hardlinked_artifacts,
    write_create_only_artifact,
    write_create_only_json,
)
from research.v3_ppo.job import (
    V3_PPO_TRAIN_RESULT_SCHEMA,
    CheckpointScheduleDecision,
    EffectiveTrainJobConfig,
    EvalScheduleDecision,
    _format_progress_line,
    build_arg_parser,
    check_eval_wall_budget,
    check_periodic_export_budget,
    decide_checkpoint_actions,
    decide_eval_action,
    resolve_effective_train_config,
    run_training_job,
)
from research.v3_ppo.resume import (
    _choose_between_full_and_light,
    _collect_valid_checkpoints_in_dir,
    _select_best_resume_candidate,
    _validate_staged_ledger_resume,
    choose_between_full_and_light,
    collect_valid_checkpoints_in_dir,
    resolve_resume_checkpoint,
    select_best_resume_candidate,
    validate_staged_ledger_resume,
    verify_staged_checkpoint,
    write_latest_checkpoint_manifest,
)
from research.v3_ppo.trainer import (
    StopRequestedError,
    V3PPOTrainer,
    _build_adamw_optimizer,
    _move_optimizer_state_to_device,
    _to_cpu_tree,
    seed_everything,
)

_RESUME_MUTABLE_CONFIG_KEYS = RESUME_MUTABLE_CONFIG_KEYS


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; exits non-zero whenever terminal ``status != 'PASS'``."""
    args = build_arg_parser().parse_args(argv)
    eff = resolve_effective_train_config(args)
    res = run_training_job(
        cfg=eff.cfg,
        output_dir=eff.output_dir,
        work_dir=eff.work_dir,
        device=eff.device,
        input_dir=eff.input_dir,
        ledger_path=eff.ledger_path,
        max_wall_seconds=eff.max_wall_seconds,
        max_session_export_bytes=eff.max_session_export_bytes,
    )
    if res.get("status") != "PASS":
        return 1
    return 0


__all__ = [
    "CheckpointScheduleDecision",
    "EffectiveTrainJobConfig",
    "EvalScheduleDecision",
    "INIT_ANCHOR_LIGHT_NAME",
    "RESUME_MUTABLE_CONFIG_KEYS",
    "RotatingCreateOnlyJSONLWriter",
    "StopRequestedError",
    "V3PPOTrainer",
    "V3_PPO_FULL_INDEX_SCHEMA",
    "V3_PPO_LIGHT_CHECKPOINT_SCHEMA",
    "V3_PPO_SIM_SHARD_SCHEMA",
    "V3_PPO_TRAIN_RESULT_SCHEMA",
    "_INIT_ANCHOR_LIGHT_NAME",
    "_RESUME_MUTABLE_CONFIG_KEYS",
    "_artifact_or_sidecar_exists",
    "_build_adamw_optimizer",
    "_choose_between_full_and_light",
    "_ckpt_tag",
    "_collect_valid_checkpoints_in_dir",
    "_exported_payload_bytes",
    "_format_progress_line",
    "_move_optimizer_state_to_device",
    "_prune_work_checkpoints_after_rewind",
    "_select_best_resume_candidate",
    "_to_cpu_tree",
    "_validate_staged_ledger_resume",
    "artifact_or_sidecar_exists",
    "build_arg_parser",
    "check_eval_wall_budget",
    "check_periodic_export_budget",
    "choose_between_full_and_light",
    "ckpt_tag",
    "collect_valid_checkpoints_in_dir",
    "concat_sim_state_shards",
    "decide_checkpoint_actions",
    "decide_eval_action",
    "exported_payload_bytes",
    "main",
    "pack_sim_state_dict",
    "prune_old_checkpoints",
    "prune_work_checkpoints_after_rewind",
    "publish_checkpoint_to_output",
    "publish_hardlinked_artifacts",
    "resolve_effective_train_config",
    "resolve_resume_checkpoint",
    "run_training_job",
    "seed_everything",
    "select_best_resume_candidate",
    "slice_sim_state_dict",
    "unpack_sim_state_dict",
    "validate_staged_ledger_resume",
    "verify_full_checkpoint_set",
    "verify_light_checkpoint",
    "verify_staged_checkpoint",
    "write_create_only_artifact",
    "write_create_only_json",
    "write_latest_checkpoint_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
