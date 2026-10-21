"""Exercise the real pinned supervisor file transport locally, without Colab."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import digest, export_json
from research.bl_upload import qualify
from research.bl_upload.qualification import _pinned_source_module
from research.colab_cli import transfers

ROOT = Path(__file__).resolve().parents[2]


class LocalContents:
    """Local ContentsClient boundary; execute the unchanged transfer RPC code."""

    session = "LOCAL_ONLY_NO_COLAB"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.transfer_records: list[dict] = []
        self.contents_calls = 0
        self.rpc_calls = 0

    def call(self, *args: str, timeout: float) -> str:
        if args[:3] != ("upload", "-s", self.session) or len(args) != 5 or timeout <= 0:
            raise ValueError("only a bounded local ContentsClient upload is allowed")
        source, destination = Path(args[3]), Path(args[4]).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("local transport path escape")
        # The official ContentsClient serializes whole uploads as JSON/base64.
        encoded = json.dumps({"content": base64.b64encode(source.read_bytes()).decode()})
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(json.loads(encoded)["content"], validate=True))
        self.contents_calls += 1
        return "{}"

    def rpc(self, code: str, timeout: float) -> Any:
        if not code.startswith("# SNAKE_TRANSFER_") or timeout <= 0:
            raise ValueError("only pinned supervisor transfer RPC code is permitted")
        namespace: dict[str, Any] = {}
        exec(compile(code, "<pinned-local-transfer-rpc>", "exec"), namespace)
        self.rpc_calls += 1
        return namespace["result"]


def preflight(manifest: dict, receipt: Path | None = None, *, repo: Path = ROOT) -> dict:
    """Stage exact bootstrap/inputs/fallback with a shared 300-second deadline."""
    result = qualify(manifest, repo=repo)
    commit = manifest["commit"]
    # This comparison also binds policy imports used by the transfer primitive.
    for name in ("transfers.py", "policy.py"):
        pinned = subprocess.check_output(
            ["git", "show", f"{commit}:research/colab_cli/{name}"], cwd=repo, timeout=30
        )
        if pinned != (repo / "research/colab_cli" / name).read_bytes():
            raise ValueError("local transfer implementation differs from manifest pin")
    with tempfile.TemporaryDirectory(prefix="wp1-real-transport-") as temporary:
        root = Path(temporary)
        source = _pinned_source_module(repo, commit)
        stage = root / "source"
        source.prepare(repo, manifest, stage)
        actual_plan = json.loads((stage / source.PLAN_NAME).read_text())
        if (
            actual_plan["fallback_size"] != result["source_plan"]["fallback_bytes"]
            or actual_plan["tree"] != result["source_plan"]["tree"]
        ):
            raise ValueError("exercised source fallback differs from qualified bytes/tree")
        # gzip headers can change between preparations. Bind the receipt to the
        # exact source-plan/archive pair exercised below, not the earlier sizing pass.
        result["source_plan_sha256"] = digest(stage / source.PLAN_NAME)
        result["source_plan"]["fallback_sha256"] = actual_plan["fallback_sha256"]
        local = []
        for name in ("bootstrap.sh", "source.py"):
            path = root / name
            path.write_bytes(
                subprocess.check_output(
                    ["git", "show", f"{commit}:research/colab_cli/{name}"], cwd=repo, timeout=30
                )
            )
            local.append((path, name))
        local += [
            (Path(item["path"]), "inputs/" + item["dest"])
            for item in manifest["inputs"]
            if "path" in item
        ]
        local.append((stage / source.ARCHIVE_NAME, source.ARCHIVE_NAME))
        cli = LocalContents(root / "remote")
        started = time.monotonic()
        deadline = started + transfers.UPLOAD_TOTAL_TIMEOUT_SECONDS
        verified = []
        for path, destination in local:
            target = cli.root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            actual = transfers.upload_file(
                cli, path, str(target), renew=lambda: None, deadline=deadline
            )
            if digest(target) != actual or actual != digest(path):
                raise ValueError("real transport byte mismatch")
            verified.append({"dest": destination, "bytes": path.stat().st_size, "sha256": actual})
        result.update(
            schema="bl-served-search-transport-preflight/v2",
            boundary=(
                "real supervisor upload_file + local JSON/base64 ContentsClient; " "no provider RPC"
            ),
            colab_calls=0,
            live_provider_latency_measured=False,
            actual_transport_elapsed_seconds=time.monotonic() - started,
            contents_calls=cli.contents_calls,
            transfer_rpc_calls=cli.rpc_calls,
            shared_deadline_seconds=300,
            files_transferred=verified,
            all_bytes_verified=True,
        )
    if receipt is not None:
        export_json(receipt, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(json.loads(args.manifest.read_text()), args.receipt)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "normal_forecast_seconds",
                    "fallback_upload_forecast_seconds",
                    "fallback_total_bound_seconds",
                    "contents_calls",
                    "colab_calls",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
