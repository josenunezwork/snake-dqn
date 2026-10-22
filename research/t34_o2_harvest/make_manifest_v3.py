"""Mechanical recovery transport for the unchanged, frozen O2 V2 study.

Keep the original runtime commit and launcher. Fetch committed control inputs
through the qualified public-input path instead of uploading their bytes twice.
CUDA must remain visible until the shared bootstrap has checked its CUDA wheel;
launch_v2.sh then hides it before any scientific worker starts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from research.t34_o2_harvest import make_manifest_v2 as v2
from research.t34_o2_harvest.frozen_sources_v3 import invoke, require_registration

ROOT = v2.ROOT
REGISTRATION_PUBLICATION = "040a6fcd853d04a57bc36da47dcbebca65c62023"
RAW = "https://raw.githubusercontent.com/josenunezwork/snake-dqn"


def check_bootstrap_environment(manifest: dict[str, Any]) -> None:
    """Explain the exact bootstrap/worker boundary before a job can be prepared."""
    env = manifest.get("env", {})
    if "CUDA_VISIBLE_DEVICES" in env:
        raise v2.ManifestRefused(
            "O2 L4 bootstrap requires CUDA_VISIBLE_DEVICES to be unset: the shared "
            "bootstrap checks torch.cuda.is_available() before launch_v2.sh hides "
            "CUDA for CPU-only scientific workers"
        )
    if manifest.get("script") != v2.SCRIPT or env.get("SNAKE_DQN_DEVICE") != "cpu":
        raise v2.ManifestRefused("O2 recovery requires the frozen CPU-only launch_v2.sh")


def make_manifest(**kwargs: Any) -> dict[str, Any]:
    """Reuse every frozen V2 cap, bank slice, source pin, and timing admission."""
    manifest = invoke(kwargs["registration"], kwargs)
    frozen = require_registration(kwargs["registration"])
    v2._verify_sources(manifest, frozen)
    manifest["job_id"] = f"t34-o2-harvest-v3-{kwargs['chunk']:02d}"
    del manifest["env"]["CUDA_VISIBLE_DEVICES"]
    for item, publication, name in zip(
        manifest["inputs"][:2],
        (REGISTRATION_PUBLICATION, manifest["commit"]),
        (
            "research/t34_o2_harvest/REGISTRATION_O2_V2.json",
            "research/t34_exit_b8/PUBLIC_INPUTS.json",
        ),
        strict=True,
    ):
        payload = subprocess.check_output(
            ["git", "show", f"{publication}:{name}"], cwd=ROOT, timeout=20
        )
        local = Path(item.pop("path"))
        if payload != local.read_bytes():
            raise v2.ManifestRefused(f"published control input differs from local bytes: {name}")
        item.update(url=f"{RAW}/{publication}/{name}", size_bytes=len(payload))
    check_bootstrap_environment(manifest)
    from research.t34_o2_harvest.upload_readiness import check

    check(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "catalog", "preflight-receipt", "output-dir", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        manifest = make_manifest(
            registration=args.registration,
            catalog=args.catalog,
            preflight_receipt=args.preflight_receipt,
            output_dir=args.output_dir,
            commit=args.commit,
            chunk=args.chunk,
        )
        from research.colab_cli.supervisor import validate

        validate(manifest, repo=ROOT)
        from research.t34_o2_harvest.bootstrap_preflight_v3 import run_bootstrap

        run_bootstrap(manifest)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
