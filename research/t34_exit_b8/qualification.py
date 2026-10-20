"""Refuse T34 manifests whose local or pinned transport lacks qualification."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


class QualificationRefused(ValueError):
    """The selected transport must pass final-only acceptance before dispatch."""


def require_qualified_transport(repo: Path, commit: str) -> dict[str, Any]:
    """Compare the complete executable inventory to the reviewed acceptance record.

    Both the Mac broker and the manifest's Git commit must match. Documentation
    is excluded because acceptance binds only maintained Python and shell files.
    This check performs no provider calls and never rewrites qualification.
    """
    try:
        document = (repo / "research/colab_cli/QUALIFICATION.md").read_text()
        blocks = re.findall(r"```json\n(.*?)\n```", document, re.DOTALL)
        if len(blocks) != 1:
            raise ValueError("expected one qualification record")
        record = json.loads(blocks[0])
        expected = record["source_manifest"]
        if (
            record.get("schema") != "snake-dqn-t34-transport-qualification/v1"
            or record.get("profile") != "final-only"
            or record.get("status") != "PASS"
            or not isinstance(expected, dict)
            or not expected
            or any(
                not re.fullmatch(r"research/colab_cli/[\w./-]+\.(py|sh)", name)
                or ".." in Path(name).parts
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for name, digest in expected.items()
            )
        ):
            raise ValueError("invalid final-only qualification record")
        current = {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (repo / "research/colab_cli").rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"}
        }
        if current != expected:
            changed = sorted(
                name
                for name in current.keys() | expected.keys()
                if current.get(name) != expected.get(name)
            )
            raise QualificationRefused(
                "colab_cli hashes differ from QUALIFICATION.md; operator must re-run "
                "final-only acceptance before T34 dispatch: " + ", ".join(changed)
            )
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("manifest must pin a full commit SHA")
        listed = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", "research/colab_cli"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.splitlines()
        inventory = {name for name in listed if Path(name).suffix in {".py", ".sh"}}
        if inventory != set(expected):
            raise QualificationRefused("pinned colab_cli inventory differs from QUALIFICATION.md")
        for name, digest in expected.items():
            pinned = subprocess.run(
                ["git", "show", f"{commit}:{name}"],
                cwd=repo,
                check=True,
                capture_output=True,
                timeout=15,
            ).stdout
            if hashlib.sha256(pinned).hexdigest() != digest:
                raise QualificationRefused(
                    f"pinned colab_cli hash differs from QUALIFICATION.md: {name}"
                )
        return record
    except QualificationRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise QualificationRefused(f"transport qualification cannot be verified: {exc}") from exc
