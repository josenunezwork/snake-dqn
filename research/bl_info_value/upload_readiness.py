"""E3's package-local Z-9 upload admission fence.

The shared calculator owns the measured transport constants.  Keeping this
small wrapper in the experiment package makes manifest publication depend on a
receipt for E3's actual compact catalog and sparse source closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research.bl_upload import qualify


def require_upload_readiness(
    manifest: dict[str, Any], *, repo: Path, receipt_path: Path
) -> dict[str, Any]:
    """Write E3's create-only upload preflight or refuse publication."""
    return qualify(manifest, repo=repo, receipt_path=receipt_path)
