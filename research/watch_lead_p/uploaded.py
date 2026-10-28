"""P-only uploaded checkpoint selection, independent of Mac serving evidence.

The sealed manifest uploads these exact bytes. This adapter changes only P's
construction dependencies; it preserves GameSession and its mixed v8 roster.
Never install it in the web server or mutate the shared serving registry.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from research.bl_served_search.exports import digest

MECHANICS_V2 = "1"


@contextmanager
def uploaded_checkpoints(inputs: Path | None = None) -> Iterator[dict]:
    """Require exact uploaded B8/FRP3 files; never search or fall back."""
    from web.backend.served_checkpoint import (
        B8_S3_SHA256,
        FRP3_S12_SHA256,
        NAME_B8_S3,
        NAME_FRP3_S12,
        ServedCheckpointChoice,
    )

    if inputs is None:
        raw = os.environ.get("COLAB_INPUT_DIR")
        if not raw:
            raise ValueError("P requires COLAB_INPUT_DIR with sealed uploaded checkpoints")
        inputs = Path(raw)
    if not inputs.is_absolute() or not inputs.is_dir() or inputs.is_symlink():
        raise ValueError("P uploaded inputs must be an existing absolute directory")
    choices = {}
    for name, sha in (("b8-s3", B8_S3_SHA256), ("frp3-s12", FRP3_S12_SHA256)):
        path = inputs / f"{name}.pth"
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(inputs.resolve())
            or not path.is_file()
            or digest(path) != sha
        ):
            raise ValueError(f"P sealed uploaded checkpoint mismatch: {name}")
        choices[name] = ServedCheckpointChoice(
            requested=None if name == NAME_B8_S3 else NAME_FRP3_S12,
            name=name,
            path=str(path),
            sha256=sha,
        )

    # These are the exact GameSession construction calls. Reject new roles or
    # arguments instead of silently supplying a different checkpoint to them.
    def hero(*args: object, **kwargs: object) -> ServedCheckpointChoice:
        from web.backend.session import DEFAULT_CHECKPOINT

        if args or kwargs != {"champion_path": DEFAULT_CHECKPOINT}:
            raise ValueError("unexpected P hero resolver arguments")
        return choices[NAME_B8_S3]

    def opponent(*args: object, **kwargs: object) -> ServedCheckpointChoice:
        if args or kwargs:
            raise ValueError("unexpected P opponent resolver arguments")
        return choices[NAME_FRP3_S12]

    def gated(*args: object, **kwargs: object) -> list[str]:
        if args or kwargs:
            raise ValueError("unexpected P v8 resolver arguments")
        # Mac-side regression binds the complete original set, before patching.
        return [FRP3_S12_SHA256]

    with (
        patch("web.backend.session.resolve_served_checkpoint", side_effect=hero) as hero_binding,
        patch(
            "web.backend.session.resolve_mixed_watch_opponent", side_effect=opponent
        ) as opp_binding,
        patch(
            "web.backend.safety_veto_serving.v8_gated_checkpoint_sha256s", side_effect=gated
        ) as v8_binding,
        patch.dict(
            os.environ, {"SNAKE_MECHANICS_V2": MECHANICS_V2, "COLAB_INPUT_DIR": str(inputs)}
        ),
    ):
        yield {"hero": hero_binding, "opponent": opp_binding, "v8": v8_binding}
