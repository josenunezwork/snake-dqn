"""Fail-closed upload-size qualification for checkpoint-carrying manifests.

The supervisor's initial ContentsClient staging shares one 300-second deadline.
This module makes a package prove its exact pinned source closure and local
input payload before a manifest can be declared dispatchable.  It intentionally
does not allocate a Colab runtime or modify the supervisor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_TOTAL_CAP_SECONDS = 300
OBSERVED_UPLOAD_BYTES_PER_SECOND = 44484.78697659882
NORMAL_UPLOAD_FORECAST_SECONDS = 150
FIXED_UPLOAD_SETUP_SECONDS = 65
ADDITIONAL_LOCAL_INPUT_SECONDS = 2

_CATALOG_PATH = "research/t34_exit_b8/PUBLIC_INPUTS.json"
_CATALOG_DEST = "public-inputs.json"
_PUBLIC_SOURCE_PATHS = (
    "research/bl_upload",
    "research/t34_exit_b8/public_inputs.py",
    _CATALOG_PATH,
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo, timeout=30)


def _pinned_blob(repo: Path, commit: str, path: str) -> bytes:
    try:
        return _git(repo, "show", f"{commit}:{path}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"pinned source is unavailable: {path}") from exc


def _verify_transport_constants(repo: Path, commit: str) -> dict[str, Any]:
    """Refuse if the selected commit does not retain the qualified cap."""
    body = _pinned_blob(repo, commit, "research/colab_cli/transfers.py").decode("utf-8")
    namespace: dict[str, Any] = {}
    # The two assignments are literal arithmetic in the qualified source.  Do
    # not import a working-tree module: it could drift from the manifest pin.
    for line in body.splitlines():
        if line.startswith("UPLOAD_TOTAL_TIMEOUT_SECONDS =") or line.startswith(
            "UPLOAD_CHUNK_BYTES ="
        ):
            exec(line, {"__builtins__": {}}, namespace)
    cap = namespace.get("UPLOAD_TOTAL_TIMEOUT_SECONDS")
    if cap != UPLOAD_TOTAL_CAP_SECONDS:
        raise ValueError(
            f"pinned transfer cap drifted: expected {UPLOAD_TOTAL_CAP_SECONDS}, got {cap!r}"
        )
    return {
        "cap_seconds": cap,
        "transfers_sha256": _sha256_bytes(body.encode()),
        "chunk_bytes": namespace.get("UPLOAD_CHUNK_BYTES"),
    }


def _pinned_source_module(repo: Path, commit: str) -> types.ModuleType:
    """Load the supervisor source builder from the manifest pin, never HEAD."""
    body = _pinned_blob(repo, commit, "research/colab_cli/source.py")
    module = types.ModuleType("bl_upload_pinned_source")
    module.__file__ = f"{commit}:research/colab_cli/source.py"
    exec(compile(body, module.__file__, "exec"), module.__dict__)
    if getattr(module, "FETCH_TIMEOUT_SECONDS", None) != 120:
        raise ValueError("pinned source fetch bound drifted from qualified 120 seconds")
    return module


def _manifest_commit(manifest: dict[str, Any]) -> str:
    commit = manifest.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise ValueError("manifest must pin a full lowercase commit SHA")
    return commit


def _input_rows(manifest: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    rows = manifest.get("inputs", [])
    if not isinstance(rows, list):
        raise ValueError("manifest inputs must be a list")
    result: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("manifest input must be an object")
        if "path" not in item:
            if "url" in item:
                if not isinstance(item.get("url"), str) or not isinstance(
                    item.get("size_bytes"), int
                ):
                    raise ValueError("public input requires URL and byte size")
                digest = item.get("sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("public input requires SHA-256")
            continue
        raw, digest, dest = item.get("path"), item.get("sha256"), item.get("dest")
        if not isinstance(raw, str) or not isinstance(digest, str) or not isinstance(dest, str):
            raise ValueError("local input requires path, sha256, and dest")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"local input is not a file: {path}")
        actual = _sha256_file(path)
        if actual != digest:
            raise ValueError(f"local input hash mismatch: {dest}")
        result.append(
            {"dest": dest, "path": str(path), "bytes": path.stat().st_size, "sha256": actual}
        )
    return result


def _pinned_bootstrap_bytes(repo: Path, commit: str) -> list[dict[str, Any]]:
    return [
        {
            "dest": name,
            "bytes": len(_pinned_blob(repo, commit, f"research/colab_cli/{name}")),
            "sha256": _sha256_bytes(_pinned_blob(repo, commit, f"research/colab_cli/{name}")),
        }
        for name in ("bootstrap.sh", "source.py")
    ]


def _forecast(bytes_total: int, local_count: int) -> float:
    return (
        FIXED_UPLOAD_SETUP_SECONDS
        + ADDITIONAL_LOCAL_INPUT_SECONDS * max(0, local_count - 1)
        + bytes_total / OBSERVED_UPLOAD_BYTES_PER_SECOND
    )


def assert_upload_budget(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Fast local-input forecast used before a package writes a manifest.

    It checks the two supervisor bootstrap blobs from the manifest's pin and
    every local input identity.  Full source fallback sizing belongs to
    :func:`qualify`, because it calls the real sparse-source builder.
    """
    repo = repo.resolve()
    commit = _manifest_commit(manifest)
    transport = _verify_transport_constants(repo, commit)
    inputs = _input_rows(manifest, repo)
    bootstrap = _pinned_bootstrap_bytes(repo, commit)
    total = sum(row["bytes"] for row in [*inputs, *bootstrap])
    forecast = _forecast(total, len(inputs))
    if forecast > NORMAL_UPLOAD_FORECAST_SECONDS:
        raise ValueError(
            f"upload forecast {forecast:.3f}s exceeds qualified "
            f"{NORMAL_UPLOAD_FORECAST_SECONDS}s budget"
        )
    return {
        "schema": "bl-upload-budget/v1",
        "manifest_sha256": _sha256_bytes(_canonical(manifest)),
        "commit": commit,
        "transport": transport,
        "rate_bytes_per_second": OBSERVED_UPLOAD_BYTES_PER_SECOND,
        "fixed_setup_seconds": FIXED_UPLOAD_SETUP_SECONDS,
        "additional_local_input_seconds": ADDITIONAL_LOCAL_INPUT_SECONDS,
        "normal_budget_seconds": NORMAL_UPLOAD_FORECAST_SECONDS,
        "normal_upload_bytes": total,
        "normal_forecast_seconds": forecast,
        "local_input_count": len(inputs),
        "files": [*bootstrap, *inputs],
    }


def _source_plan(
    repo: Path, manifest: dict[str, Any], pinned_source: types.ModuleType
) -> tuple[dict[str, Any], int, str]:
    with tempfile.TemporaryDirectory(prefix="bl-upload-source-") as directory:
        stage = Path(directory)
        pinned_source.prepare(repo, manifest, stage)
        plan_bytes = (stage / pinned_source.PLAN_NAME).read_bytes()
        fallback = stage / pinned_source.ARCHIVE_NAME
        return json.loads(plan_bytes), fallback.stat().st_size, _sha256_bytes(plan_bytes)


def qualify(
    manifest: dict[str, Any], *, repo: Path = ROOT, receipt_path: Path | None = None
) -> dict[str, Any]:
    """Exercise the pinned supervisor source preparation and bind a receipt."""
    repo = repo.resolve()
    result = assert_upload_budget(manifest, repo=repo)
    pinned_source = _pinned_source_module(repo, result["commit"])
    plan, fallback_bytes, plan_sha = _source_plan(repo, manifest, pinned_source)
    if plan.get("commit") != result["commit"] or not isinstance(plan.get("tree"), str):
        raise ValueError("source plan identity differs from manifest pin")
    if plan.get("fallback_size") != fallback_bytes:
        raise ValueError("source fallback archive size differs from source plan")
    fallback_total = result["normal_upload_bytes"] + fallback_bytes
    fallback_forecast = _forecast(fallback_total, result["local_input_count"] + 1)
    if fallback_forecast > NORMAL_UPLOAD_FORECAST_SECONDS:
        raise ValueError(
            f"fallback upload forecast {fallback_forecast:.3f}s exceeds qualified "
            f"{NORMAL_UPLOAD_FORECAST_SECONDS}s budget"
        )
    fallback_total_seconds = fallback_forecast + pinned_source.FETCH_TIMEOUT_SECONDS
    if fallback_total_seconds > UPLOAD_TOTAL_CAP_SECONDS:
        raise ValueError(
            f"fallback forecast plus fetch bound {fallback_total_seconds:.3f}s exceeds cap "
            f"{UPLOAD_TOTAL_CAP_SECONDS}s"
        )
    result.update(
        {
            "status": "PASS_NOT_DISPATCHED",
            "source_plan_sha256": plan_sha,
            "source_plan": {
                "commit": plan["commit"],
                "tree": plan["tree"],
                "paths": plan["paths"],
                "fallback_sha256": plan["fallback_sha256"],
                "fallback_bytes": fallback_bytes,
                "fallback_unpacked_bytes": plan["fallback_unpacked_size"],
                "file_count": len(plan["files"]),
            },
            "fallback_upload_bytes": fallback_total,
            "fallback_upload_forecast_seconds": fallback_forecast,
            "fallback_fetch_bound_seconds": pinned_source.FETCH_TIMEOUT_SECONDS,
            "fallback_total_bound_seconds": fallback_total_seconds,
        }
    )
    if receipt_path is not None:
        receipt = receipt_path.expanduser().resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
        try:
            with receipt.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError as exc:
            raise ValueError(f"upload qualification receipt already exists: {receipt}") from exc
    return result


def verify_receipt(
    manifest: dict[str, Any], receipt_path: Path, *, repo: Path = ROOT
) -> dict[str, Any]:
    """Recompute qualification and require a receipt to bind those exact bytes."""
    receipt = receipt_path.expanduser().resolve()
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid upload qualification receipt: {receipt}") from exc
    current = qualify(manifest, repo=repo)
    keys = (
        "status",
        "manifest_sha256",
        "normal_upload_bytes",
        "normal_forecast_seconds",
        "fallback_upload_bytes",
        "fallback_upload_forecast_seconds",
        "fallback_total_bound_seconds",
    )
    plan_keys = (
        "commit",
        "tree",
        "paths",
        "fallback_bytes",
        "fallback_unpacked_bytes",
        "file_count",
    )
    same_plan = isinstance(saved, dict) and all(
        saved.get("source_plan", {}).get(key) == current.get("source_plan", {}).get(key)
        for key in plan_keys
    )
    if (
        not isinstance(saved, dict)
        or not same_plan
        or any(saved.get(key) != current.get(key) for key in keys)
    ):
        raise ValueError("upload qualification receipt does not bind current manifest bytes")
    return saved


def _catalog_rows(repo: Path) -> list[dict[str, Any]]:
    value = json.loads((repo / _CATALOG_PATH).read_text(encoding="utf-8"))
    rows = value.get("inputs") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("public checkpoint catalog must bind exactly six inputs")
    return rows


def convert_public_checkpoints(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Replace exactly the registered six checkpoint uploads with their catalog.

    A partial match is deliberately left alone: unregistered bytes must remain
    visible to the upload budget rather than becoming an implicit publication.
    """
    repo = repo.resolve()
    converted = copy.deepcopy(manifest)
    rows = _catalog_rows(repo)
    expected = {(str(row["dest"]), int(row["size_bytes"]), str(row["sha256"])) for row in rows}
    aliases = {"B8_s3_final.pth": "b8.pth"}
    inputs = converted.get("inputs", [])
    if not isinstance(inputs, list):
        raise ValueError("manifest inputs must be a list")
    local: list[tuple[int, tuple[str, int, str]]] = []
    for index, item in enumerate(inputs):
        if isinstance(item, dict) and "path" in item:
            path = Path(str(item["path"])).expanduser()
            if path.is_file():
                actual = _sha256_file(path)
                if actual != str(item.get("sha256")):
                    return converted
                dest = str(item.get("dest"))
                identity = (dest, path.stat().st_size, str(item.get("sha256")))
                if dest == aliases["B8_s3_final.pth"]:
                    identity = ("B8_s3_final.pth", identity[1], identity[2])
                local.append((index, identity))
    catalog_dests = {identity[0] for identity in expected} | set(aliases.values())
    # A package may also ship a small registration, bank, or sidecar.  Require
    # every catalog identity, but preserve those unrelated local inputs.
    if any(identity[0] in catalog_dests and identity not in expected for _index, identity in local):
        return converted
    matched = [(index, identity) for index, identity in local if identity in expected]
    if {identity for _index, identity in matched} != expected or len(matched) != len(expected):
        return converted
    catalog = repo / _CATALOG_PATH
    replacement = {
        "path": str(catalog.resolve()),
        "dest": _CATALOG_DEST,
        "sha256": _sha256_file(catalog),
    }
    matched_indexes = {index for index, _identity in matched}
    original_dests = {str(inputs[index].get("dest")): identity[0] for index, identity in matched}
    converted["inputs"] = [
        item for index, item in enumerate(inputs) if index not in matched_indexes
    ]
    converted["inputs"].append(replacement)
    env = converted.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError("manifest env must be an object")
    env["BL_PUBLIC_INPUTS"] = "1"
    env["BL_PUBLIC_INPUT_ALIASES"] = json.dumps(
        {public: original for original, public in original_dests.items() if public != original},
        sort_keys=True,
    )
    paths = converted.get("source_paths")
    if paths is None:
        commit = _manifest_commit(converted)
        pinned_source = _pinned_source_module(repo, commit)
        # Mirror the supervisor's default closure before adding the runtime
        # materializer.  An absent source_paths field must never turn this
        # conversion into a sparse fallback with only helper files.
        paths = pinned_source._defaults(converted, pinned_source._tracked(repo, commit))
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("manifest source_paths must be a list of strings")
    converted["source_paths"] = list(dict.fromkeys([*paths, *_PUBLIC_SOURCE_PATHS]))
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps(qualify(manifest, receipt_path=args.receipt), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
