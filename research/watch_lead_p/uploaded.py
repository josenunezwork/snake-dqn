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


@contextmanager
def uploaded_checkpoints(inputs: Path | None = None) -> Iterator[None]:
    """Require exact uploaded B8/FRP3 files; never search or fall back."""
    from web.backend.served_checkpoint import (
        B8_S3_SHA256,
        FRP3_S12_SHA256,
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
        choices[name] = ServedCheckpointChoice(None, name, str(path), sha)

    # P is an offline experiment, not a serving qualification. The registered
    # uploaded FRP3 hash replaces the Mac strict-receipt lookup; wrapper source
    # checks, installation and mixed-roster assertions remain authoritative.
    with (
        patch("web.backend.session.resolve_served_checkpoint", return_value=choices["b8-s3"]),
        patch("web.backend.session.resolve_mixed_watch_opponent", return_value=choices["frp3-s12"]),
        patch(
            "web.backend.safety_veto_serving.v8_gated_checkpoint_sha256s",
            return_value=[FRP3_S12_SHA256],
        ),
        patch.dict(os.environ, {"SNAKE_MECHANICS_V2": "1", "COLAB_INPUT_DIR": str(inputs)}),
    ):
        yield
