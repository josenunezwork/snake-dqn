"""Shared checkpoint compatibility checks for training entry points."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

from src.model.obs_spec import DEFAULT_OBS_SPEC, OBS_SPEC_KEY

logger = logging.getLogger(__name__)


def _is_reward_contract_key(key: str) -> bool:
    """Return whether a contract key carries reward semantics (overridable)."""
    return key == "reward_contract" or key.startswith("reward_")


def checkpoint_contract_values(checkpoint: Mapping[str, object], key: str) -> list[object]:
    """Return all recorded checkpoint contract values for a key."""
    values = []
    if key in checkpoint:
        values.append(checkpoint[key])

    for config_key in ("apex_config", "config"):
        config = checkpoint.get(config_key)
        if isinstance(config, Mapping) and key in config:
            values.append(config[key])
    return values


def validate_checkpoint_contract(
    checkpoint: Mapping[str, object],
    expected_config: Mapping[str, object],
    checkpoint_path: str = "checkpoint",
    *,
    integer_keys: Sequence[str] = ("input_size", "hidden_size", "output_size", "n_step"),
    float_keys: Sequence[str] = ("gamma",),
    bool_keys: Sequence[str] = (),
    str_keys: Sequence[str] = (),
    mapping_keys: Sequence[str] = (),
    required_keys: Sequence[str] = (),
    error_type: type[Exception] = RuntimeError,
    override_reward_contract: bool = False,
) -> None:
    """Reject checkpoints that declare target semantics incompatible with current config.

    Args:
        checkpoint: Loaded checkpoint mapping (may carry contract metadata at the
            top level and/or under ``apex_config``/``config``).
        expected_config: Current-config contract values to enforce.
        checkpoint_path: Human-readable checkpoint identifier for messages.
        integer_keys: Keys compared as ints.
        float_keys: Keys compared as finite floats (isclose).
        bool_keys: Keys compared as bools.
        str_keys: Keys compared as exact strings. ``obs_spec`` is special-cased:
            a checkpoint that omits it is backfilled to
            :data:`~src.model.obs_spec.DEFAULT_OBS_SPEC` (``'vector61'``) so every
            pre-contract-v2 champion keeps validating as the vector spec (and a
            raster checkpoint therefore correctly mismatches a vector config).
        mapping_keys: Keys compared child-by-child as finite floats.
        required_keys: Keys whose contract metadata must be present.
        error_type: Exception type raised on violation.
        override_reward_contract: Escape hatch for deliberate reward-economics
            changes (e.g. fine-tuning champion_a5 under reward v2): when True,
            mismatched or missing REWARD fields (``reward_contract`` and any
            ``reward_*`` key) log a loud warning instead of aborting. All
            non-reward contract violations (gamma, n_step, shapes, ...) still
            abort. Default False keeps full enforcement.
    """
    overridden_violations: list[str] = []

    def _report(key: str, message: str, cause: Exception | None = None) -> None:
        """Raise for the violation, or collect it when the reward override applies."""
        if override_reward_contract and _is_reward_contract_key(key):
            overridden_violations.append(message)
            return
        if cause is not None:
            raise error_type(message) from cause
        raise error_type(message)

    for key in required_keys:
        if key not in expected_config:
            continue
        if not checkpoint_contract_values(checkpoint, key):
            _report(
                key,
                f"Checkpoint missing required {key} contract metadata for {checkpoint_path}",
            )

    for key in integer_keys:
        if key not in expected_config:
            continue
        expected_value = int(expected_config[key])
        for raw_value in checkpoint_contract_values(checkpoint, key):
            checkpoint_value = int(raw_value)
            if checkpoint_value != expected_value:
                _report(
                    key,
                    f"Checkpoint {key}={checkpoint_value} does not match current "
                    f"{key}={expected_value} for {checkpoint_path}",
                )

    for key in float_keys:
        if key not in expected_config:
            continue
        expected_value = float(expected_config[key])
        for raw_value in checkpoint_contract_values(checkpoint, key):
            checkpoint_value = float(raw_value)
            if not math.isfinite(checkpoint_value) or not math.isclose(
                checkpoint_value,
                expected_value,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                _report(
                    key,
                    f"Checkpoint {key}={checkpoint_value:g} does not match current "
                    f"{key}={expected_value:g} for {checkpoint_path}",
                )

    for key in mapping_keys:
        if key not in expected_config:
            continue
        expected_mapping = expected_config[key]
        if not isinstance(expected_mapping, Mapping):
            raise error_type(f"Current checkpoint contract {key} must be a mapping")
        for raw_mapping in checkpoint_contract_values(checkpoint, key):
            if not isinstance(raw_mapping, Mapping):
                _report(key, f"Checkpoint {key} must be a mapping for {checkpoint_path}")
                continue
            for child_key, expected_raw_value in expected_mapping.items():
                child_contract_key = f"{key}.{child_key}"
                if child_key not in raw_mapping or raw_mapping[child_key] is None:
                    if child_key == "version":
                        # Checkpoints saved before the reward-contract version
                        # field existed are implicitly v1. Backfill so they keep
                        # validating strictly on the real economics fields (and
                        # correctly mismatch a v2 config, which then requires the
                        # explicit reward override to resume).
                        raw_value: object = 1
                    else:
                        _report(
                            key,
                            f"Checkpoint missing required {child_contract_key} "
                            f"contract metadata for {checkpoint_path}",
                        )
                        continue
                else:
                    raw_value = raw_mapping[child_key]
                if isinstance(raw_value, (bool, str, bytes, bytearray, memoryview)):
                    _report(
                        key,
                        f"Checkpoint {child_contract_key} must be finite for {checkpoint_path}",
                    )
                    continue
                try:
                    checkpoint_value = float(raw_value)
                    expected_value = float(expected_raw_value)
                except (TypeError, ValueError) as exc:
                    _report(
                        key,
                        f"Checkpoint {child_contract_key} must be finite for {checkpoint_path}",
                        cause=exc,
                    )
                    continue
                if not math.isfinite(checkpoint_value) or not math.isfinite(expected_value):
                    _report(
                        key,
                        f"Checkpoint {child_contract_key} must be finite for {checkpoint_path}",
                    )
                    continue
                if not math.isclose(
                    checkpoint_value,
                    expected_value,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                ):
                    _report(
                        key,
                        f"Checkpoint {child_contract_key}={checkpoint_value:g} does not "
                        f"match current {child_contract_key}={expected_value:g} "
                        f"for {checkpoint_path}",
                    )

    for key in bool_keys:
        if key not in expected_config:
            continue
        expected_value = bool(expected_config[key])
        for raw_value in checkpoint_contract_values(checkpoint, key):
            checkpoint_value = bool(raw_value)
            if checkpoint_value != expected_value:
                _report(
                    key,
                    f"Checkpoint {key}={checkpoint_value} does not match current "
                    f"{key}={expected_value} for {checkpoint_path}",
                )

    for key in str_keys:
        if key not in expected_config:
            continue
        expected_value = str(expected_config[key])
        recorded = checkpoint_contract_values(checkpoint, key)
        if not recorded and key == OBS_SPEC_KEY:
            # Pre-contract-v2 checkpoints predate the obs_spec field; they were
            # all trained on the 61-D vector, so backfill the default spec rather
            # than aborting. This keeps every historical champion loadable while
            # still catching a genuine raster/vector mismatch (a raster checkpoint
            # *does* record obs_spec='raster31v2', which then mismatches here).
            recorded = [DEFAULT_OBS_SPEC]
        for raw_value in recorded:
            checkpoint_value = str(raw_value)
            if checkpoint_value != expected_value:
                _report(
                    key,
                    f"Checkpoint {key}={checkpoint_value!r} does not match current "
                    f"{key}={expected_value!r} for {checkpoint_path}",
                )

    if overridden_violations:
        banner = "!" * 78
        details = "\n".join(f"  - {message}" for message in overridden_violations)
        logger.warning(
            "\n%s\n"
            "REWARD CONTRACT OVERRIDE ACTIVE for %s.\n"
            "Proceeding despite %d reward-contract violation(s) that would normally abort:\n"
            "%s\n"
            "The checkpoint's replay/TD targets were produced under DIFFERENT reward "
            "economics. Only do this for a deliberate reward-migration fine-tune "
            "(e.g. the P1 3-arm reward-v2 experiment on champion_a5).\n%s",
            banner,
            checkpoint_path,
            len(overridden_violations),
            details,
            banner,
        )
