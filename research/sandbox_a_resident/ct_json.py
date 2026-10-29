"""Strict RFC-8259 JSON for every C/T producer and consumer surface."""

from __future__ import annotations

import json as _json
import math
from typing import Any, Mapping

JSONDecodeError = _json.JSONDecodeError


def refuse_constant(value: str) -> None:
    """Refuse Python's non-standard NaN/Infinity decoder extension."""
    raise ValueError("non-finite JSON constant")


def require_finite(value: Any) -> None:
    """Check decoded objects too, including exponent overflow and mapping keys."""
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 128:
            raise ValueError("excessively nested diagnostic JSON")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON float")
        if isinstance(item, Mapping):
            pending.extend((part, depth + 1) for pair in item.items() for part in pair)
        elif isinstance(item, (list, tuple)):
            pending.extend((part, depth + 1) for part in item)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def loads(text: str | bytes, **kwargs: Any) -> Any:
    """Decode with duplicate-key, constant, overflow and nesting refusal."""
    kwargs["parse_constant"] = refuse_constant
    kwargs.setdefault("object_pairs_hook", _pairs)
    try:
        value = _json.loads(text, **kwargs)
        require_finite(value)
    except RecursionError as error:
        raise ValueError("excessively nested diagnostic JSON") from error
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """Preserve healthy serialization bytes while refusing non-finite state."""
    require_finite(value)
    kwargs["allow_nan"] = False
    return _json.dumps(value, **kwargs)
