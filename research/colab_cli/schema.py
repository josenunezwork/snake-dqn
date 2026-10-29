"""Shared manifest and immutable-export schema checks for Colab CLI jobs.

The supervisor and source-preparation path deliberately use this small module
instead of maintaining subtly different interpretations of a job manifest.
Runtime remains the authority for discovering completed exports: it accepts an
export only after observing the producer's payload, ``.sha256`` sidecar and
``.complete`` marker together.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from research.colab_cli import accounting


class SchemaError(ValueError):
    """A manifest cannot safely reach preparation or allocation."""


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise SchemaError(f"{label} must be a nonempty relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SchemaError(f"{label} must be a normalized relative path")
    return value


def required_export_paths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Normalize optional required exports without weakening old manifests.

    New family manifests should declare a nonempty list of paths.  The field is
    optional here solely for previously qualified manifests; the canary gate
    applies the stricter new-family admission rule.
    """
    value = manifest.get("required_exports")
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise SchemaError("required_exports must be a nonempty list of relative paths")
    paths = tuple(_relative_path(path, "required_exports entry") for path in value)
    if len(set(paths)) != len(paths):
        raise SchemaError("required_exports must not contain duplicate paths")
    return paths


def validate_export_inventory(
    manifest: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Check scanner-shaped inventory and require every declared export.

    Sidecar validation is performed by ``runtime._exports`` before entries get
    here.  Rechecking the scanner's stable record shape keeps terminal
    admission from trusting an arbitrary producer-provided list.
    """
    actual: set[str] = set()
    for entry in inventory:
        if not isinstance(entry, Mapping):
            raise SchemaError("export inventory entry must be an object")
        path = _relative_path(entry.get("path"), "export inventory path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SchemaError("export inventory entry must contain a SHA-256 digest")
        if type(size) is not int or size < 0:
            raise SchemaError("export inventory entry must contain a nonnegative byte size")
        if path in actual:
            raise SchemaError("export inventory contains duplicate paths")
        actual.add(path)
    missing = sorted(set(required_export_paths(manifest)) - actual)
    if missing:
        raise SchemaError(
            "required exports absent from runtime scanner inventory: " + ", ".join(missing)
        )
    return tuple(sorted(actual))


def validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate shared nested manifest contracts before preparation/allocation."""
    if not isinstance(manifest, Mapping):
        raise SchemaError("manifest must be an object")
    budget = manifest.get("budget", {})
    if not isinstance(budget, Mapping):
        raise SchemaError("budget must be an object")
    if "other_assignments" in budget and not isinstance(budget["other_assignments"], Mapping):
        raise SchemaError("budget.other_assignments must map endpoint to rate_units_per_hour")
    try:
        for field in ("rate_units_per_hour", "unit_cap"):
            if field in budget:
                accounting.warning_threshold(budget[field], f"budget.{field}")
        # This is the exact accounting admission parser, including rejection of
        # the historical list-vs-map ``other_assignments`` shape.
        accounting._manifest_rates(budget.get("other_assignments"))
    except accounting.AccountingError as exc:
        raise SchemaError(str(exc)) from exc
    accounting.warning_threshold(budget.get("reserve_units"), "reserve_units", 0)
    return required_export_paths(manifest)
