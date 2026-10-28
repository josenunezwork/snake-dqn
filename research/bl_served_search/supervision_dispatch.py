"""Read-only registry generation from a prospectively opted-in dispatch journal.

This never finds a root by name, scans process environments, contacts Colab, or
changes a job. A dispatch declaration is not a qualified live observer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.bl_paced_identity.record import export_json, read_export
from research.bl_served_search import supervision_io as io


def capture_dispatch_registry(manifest_path: Path) -> dict[str, Any]:
    """Bind the immutable startup identity to the exact current job assignment."""
    from tools.ops.dispatch import sandbox_collector_directory
    from tools.sandbox.collector_v2 import Collector
    from tools.sandbox.supervision_accounting import COLLECTOR_VERSION

    manifest_path = manifest_path.resolve(strict=True)
    packet = json.loads(manifest_path.read_text())
    directory = sandbox_collector_directory(packet)
    if directory is None:
        raise ValueError("dispatch did not opt into collector-v2")
    context = read_export(directory / "collector.json")
    startup = context["startup"]
    if (
        context.get("version") != COLLECTOR_VERSION
        or context.get("policy") != io.POLICY
        or context.get("sources") != Collector.sources()
        or startup.get("manifest") != str(manifest_path)
        or startup.get("manifest_sha256") != io.file_digest(manifest_path)
        or startup.get("argv", [])[1:3] != ["-m", "tools.sandbox.supervisor_v2"]
        or (directory / "COMPLETE.json").exists()
        or (directory / "status.json").exists()
    ):
        raise ValueError("dispatch collector source/startup/manifest changed or job is terminal")
    registry = io.capture_registry(startup["pid"])
    root = registry["roots"][0]
    cwd = Path(startup["cwd"]).resolve(strict=True)
    expected_sources = {
        str((cwd / name).resolve(strict=True)): value for name, value in context["sources"].items()
    }
    expected_sources[startup["colab"]] = startup["colab_sha256"]
    # The qualified CLI generates this hook after collector startup. Bind its
    # deterministic bytes, rather than trusting a post-startup file hash.
    hook = Path(startup["argv"][4]) / packet["job_id"] / "cli-hook/sitecustomize.py"
    guard = str(cwd / "research/colab_cli/cli_guard.py")
    hook_bytes = (
        "import os,runpy\ntry:\n    runpy.run_path("
        + repr(guard)
        + ", run_name='snake_cli_guard')\nexcept BaseException:\n    os._exit(70)\n"
    ).encode()
    expected_sources[str(hook.resolve(strict=True))] = hashlib.sha256(hook_bytes).hexdigest()
    if (
        any(
            root[key] != startup[key]
            for key in ("pid", "uid", "birth_ns", "argv", "cwd", "interpreter")
        )
        or root.get("source_hashes") != expected_sources
        or root.get("colab") != startup["colab"]
        or context.get("root_process_id") != f'{root["pid"]}@{root["birth_ns"]}'
    ):
        raise ValueError("dispatch root restarted or differs from prospective startup")
    assignment_path = Path(registry["assignment_path"])
    assignment_bytes = assignment_path.read_bytes()
    assignment = json.loads(assignment_bytes)
    if any(
        root[key] != assignment[field]
        for key, field in (
            ("assignment", "endpoint"),
            ("session", "session"),
            ("generation", "generation"),
        )
    ):
        raise ValueError("assignment changed during registry capture")
    io.verify_root(root)
    # Recheck the actual manifest and assignment after identity/source capture.
    # A mixed old/new declaration must never survive an allocation transition.
    if (
        io.file_digest(manifest_path) != startup["manifest_sha256"]
        or assignment_path.read_bytes() != assignment_bytes
        or (directory / "COMPLETE.json").exists()
        or (directory / "status.json").exists()
    ):
        raise ValueError("dispatch changed or became terminal during registry capture")
    registry["dispatch_binding"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": startup["manifest_sha256"],
        "collector_directory": str(directory),
        "collector_sha256": io.digest(context),
        "job_id": packet["job_id"],
        "assignment_path": str(assignment_path),
        "assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
    }
    return registry


def main(argv: list[str] | None = None) -> int:
    """Seal a source-bound declaration from live dispatch state, without admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    registry = capture_dispatch_registry(args.dispatch_manifest)
    from research.bl_served_search.remote_supervision import validate_registry

    result = validate_registry(registry)
    if not result["valid"]:
        raise ValueError(f"invalid dispatch registry: {result['errors']}")
    export_json(args.output, registry)
    io.read_registry(args.output)
    print(json.dumps({"registry": str(args.output), "admission": "UNAVAILABLE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
