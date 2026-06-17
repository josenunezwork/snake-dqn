"""Shared checkpoint compatibility checks for training entry points."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


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
    bool_keys: Sequence[str] = ("use_gru",),
    mapping_keys: Sequence[str] = (),
    required_keys: Sequence[str] = (),
    error_type: type[Exception] = RuntimeError,
) -> None:
    """Reject checkpoints that declare target semantics incompatible with current config."""
    for key in required_keys:
        if key not in expected_config:
            continue
        if not checkpoint_contract_values(checkpoint, key):
            raise error_type(
                f"Checkpoint missing required {key} contract metadata for {checkpoint_path}"
            )

    for key in integer_keys:
        if key not in expected_config:
            continue
        expected_value = int(expected_config[key])
        for raw_value in checkpoint_contract_values(checkpoint, key):
            checkpoint_value = int(raw_value)
            if checkpoint_value != expected_value:
                raise error_type(
                    f"Checkpoint {key}={checkpoint_value} does not match current "
                    f"{key}={expected_value} for {checkpoint_path}"
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
                raise error_type(
                    f"Checkpoint {key}={checkpoint_value:g} does not match current "
                    f"{key}={expected_value:g} for {checkpoint_path}"
                )

    for key in mapping_keys:
        if key not in expected_config:
            continue
        expected_mapping = expected_config[key]
        if not isinstance(expected_mapping, Mapping):
            raise error_type(f"Current checkpoint contract {key} must be a mapping")
        for raw_mapping in checkpoint_contract_values(checkpoint, key):
            if not isinstance(raw_mapping, Mapping):
                raise error_type(f"Checkpoint {key} must be a mapping for {checkpoint_path}")
            for child_key, expected_raw_value in expected_mapping.items():
                child_contract_key = f"{key}.{child_key}"
                if child_key not in raw_mapping or raw_mapping[child_key] is None:
                    raise error_type(
                        f"Checkpoint missing required {child_contract_key} "
                        f"contract metadata for {checkpoint_path}"
                    )
                raw_value = raw_mapping[child_key]
                if isinstance(raw_value, (bool, str, bytes, bytearray, memoryview)):
                    raise error_type(
                        f"Checkpoint {child_contract_key} must be finite for {checkpoint_path}"
                    )
                try:
                    checkpoint_value = float(raw_value)
                    expected_value = float(expected_raw_value)
                except (TypeError, ValueError) as exc:
                    raise error_type(
                        f"Checkpoint {child_contract_key} must be finite for {checkpoint_path}"
                    ) from exc
                if not math.isfinite(checkpoint_value) or not math.isfinite(expected_value):
                    raise error_type(
                        f"Checkpoint {child_contract_key} must be finite for {checkpoint_path}"
                    )
                if not math.isclose(
                    checkpoint_value,
                    expected_value,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                ):
                    raise error_type(
                        f"Checkpoint {child_contract_key}={checkpoint_value:g} does not "
                        f"match current {child_contract_key}={expected_value:g} "
                        f"for {checkpoint_path}"
                    )

    for key in bool_keys:
        if key not in expected_config:
            continue
        expected_value = bool(expected_config[key])
        for raw_value in checkpoint_contract_values(checkpoint, key):
            checkpoint_value = bool(raw_value)
            if checkpoint_value != expected_value:
                raise error_type(
                    f"Checkpoint {key}={checkpoint_value} does not match current "
                    f"{key}={expected_value} for {checkpoint_path}"
                )
