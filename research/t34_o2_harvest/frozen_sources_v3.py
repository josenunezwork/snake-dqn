"""Validate V2 against its pinned source snapshot, independent of later main.

Main has evolved since registration. Neither re-freeze science nor require a
current checkout to impersonate the old runtime. Run the original validator and
manifest builder in a temporary exact Git snapshot of the registered closure.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_SHA256 = "82c1986ff42c7a98161b1f6ab657db0a61f82ab527cdd8bba4620410b698ce88"
PIN = "f19d3f39050adfedc6eaef001a3364318fbb28bc"


def analysis_adapter(original: str) -> str:
    """Construct the transport adapter from pinned science, never current code."""
    old = """    if (
        manifest.get("requirements", {}).get("gpu") != "L4"
        or manifest.get("env", {}).get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise InvalidEvidence("job execution placement differs")"""
    new = """    env = manifest.get("env", {})
    if (
        manifest.get("requirements", {}).get("gpu") != "L4"
        or "CUDA_VISIBLE_DEVICES" in env
        or env.get("SNAKE_DQN_DEVICE") != "cpu"
        or manifest.get("script") != "research/t34_o2_harvest/launch_v2.sh"
    ):
        raise InvalidEvidence(
            "O2 V3 requires visible CUDA during shared L4 bootstrap and the frozen "
            "launch_v2.sh to hide CUDA before CPU-only scientific workers"
        )"""
    signature = "def analyze(registration_path: Path, run_dirs: list[Path], output_dir: Path)"
    if original.count(old) != 1 or original.count(signature) != 1:
        raise ValueError("frozen V2 analyzer transport block/signature changed")
    return original.replace(old, new).replace(
        signature, signature.replace("analyze(", "_analyze_frozen(")
    )


def invoke(
    registration: Path,
    kwargs: dict[str, Any] | None = None,
    *,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the unchanged validator/builder, using only exact registered bytes."""
    canonical = ROOT / "research/t34_o2_harvest/REGISTRATION_O2_V2.json"
    if analysis is not None and registration.resolve() != canonical.resolve():
        raise ValueError("analysis requires the canonical frozen package registration")
    raw = registration.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRATION_SHA256:
        raise ValueError("O2 recovery requires the exact original V2 registration SHA-256")
    frozen = json.loads(raw)
    if frozen["source_commit"] != PIN:
        raise ValueError("O2 recovery runtime commit differs from frozen V2")
    qualification = subprocess.check_output(
        ["git", "show", f"{PIN}:research/colab_cli/QUALIFICATION.md"],
        cwd=ROOT,
        text=True,
        timeout=20,
    )
    records = re.findall(r"```json\n(.*?)\n```", qualification, re.DOTALL)
    if len(records) != 1:
        raise ValueError("frozen transport qualification record missing")
    transport = json.loads(records[0])["source_manifest"]
    expected = {**frozen["source_hashes"], **transport}
    if any(expected[name] != digest for name, digest in frozen["source_hashes"].items()):
        raise ValueError("frozen transport qualification conflicts with registration")
    names = sorted(expected)
    archive = subprocess.check_output(["git", "archive", PIN, *names], cwd=ROOT, timeout=30)
    git_dir = subprocess.check_output(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, text=True, timeout=10
    ).strip()
    with tempfile.TemporaryDirectory(prefix="o2-frozen-v2-") as directory:
        repo = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source:
                if member.isdir():
                    continue
                if member.name not in expected or not member.isfile():
                    raise ValueError(f"unexpected registered source member: {member.name}")
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"missing registered source payload: {member.name}")
                payload = stream.read()
                if hashlib.sha256(payload).hexdigest() != expected[member.name]:
                    raise ValueError(f"pinned V2 source digest differs: {member.name}")
                path = repo / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
        (repo / ".git").write_text(f"gitdir: {git_dir}\n")
        # Resolve the canonical registration and its sole-look marker to the
        # original publication, even while scientific imports use the snapshot.
        (repo / "research/t34_o2_harvest/REGISTRATION_O2_V2.json").symlink_to(
            registration.resolve()
        )
        if analysis is not None:
            (repo / "research/t34_o2_harvest/analyze_v3.py").write_text(
                analysis_adapter((repo / "research/t34_o2_harvest/analyze_v2.py").read_text())
            )
        code = (
            "import sys,json\nfrom pathlib import Path\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "from research.t34_o2_harvest.registration_v2 import require_registration\n"
            "args=json.loads(sys.argv[3])\nanalysis=json.loads(sys.argv[4])\n"
            "if analysis is not None:\n"
            "    from research.t34_o2_harvest.analyze_v3 import _analyze_frozen\n"
            "    result=_analyze_frozen(Path(sys.argv[2]),\n"
            "        [Path(p) for p in analysis['run_dirs']],Path(analysis['output_dir']))\n"
            "elif args is None:\n"
            "    result=require_registration(Path(sys.argv[2]))\n"
            "else:\n"
            "    from research.t34_o2_harvest.make_manifest_v2 import make_manifest\n"
            "    for k in ['registration','catalog','preflight_receipt','output_dir']:\n"
            "        args[k]=Path(args[k])\n"
            "    result=make_manifest(**args)\n"
            "print(json.dumps(result))\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                code,
                str(repo),
                str(registration.resolve()),
                json.dumps(kwargs, default=str),
                json.dumps(analysis, default=str),
            ],
            cwd=repo,
            env={k: v for k, v in os.environ.items() if k not in {"GIT_DIR", "GIT_WORK_TREE"}},
            text=True,
            capture_output=True,
            timeout=3600 if analysis is not None else 60,
        )
        if result.returncode:
            raise ValueError(f"frozen O2 V2 validation failed: {result.stderr.strip()}")
        return json.loads(result.stdout)


def require_registration(path: Path) -> dict[str, Any]:
    """Verify the unchanged authority against the source actually run remotely."""
    return invoke(path)
