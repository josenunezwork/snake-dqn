"""Run the pinned remote bootstrap and launcher locally with hardware probes.

This is a control-flow preflight, not GPU/dependency qualification or gameplay.
Only the Mac-inaccessible /content root guard is relocated. Bash, every pinned
bootstrap statement, CUDA environment ordering, and launch_v2.sh are executed.
Installers, nvidia-smi and torch hardware are deterministic offline stand-ins;
the scientific runner is a probe that refuses a GPU-visible worker boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from research.t34_o2_harvest.make_manifest_v3 import ROOT, check_bootstrap_environment


def _blob(commit: str, name: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{name}"], cwd=ROOT, text=True, timeout=20
    )


def run_bootstrap(
    manifest: dict[str, Any], *, enforce_admission: bool = True, torch_cuda: str = "12.6"
) -> dict[str, Any]:
    """Exercise the exact pinned bootstrap; convert its opaque failure to context."""
    if enforce_admission:
        check_bootstrap_environment(manifest)
    with tempfile.TemporaryDirectory(prefix="o2-bootstrap-") as directory:
        base = Path(directory).resolve()
        root = base / "snake-cli" / "local-generation"
        repo = root / "repo"
        bindir = base / "bin"
        bindir.mkdir()
        (root / "inputs").mkdir(parents=True)
        (repo / "research/t34_o2_harvest").mkdir(parents=True)
        (repo / "research/t34_exit_b8").mkdir()
        (repo / "pyproject.toml").write_text("")
        (repo / "requirements.txt").write_text("")
        script = repo / manifest["script"]
        script.write_text(_blob(manifest["commit"], manifest["script"]))
        # The production export code runs unchanged after the probe runner.
        (repo / "research/t34_o2_harvest/export.py").write_text(
            _blob(manifest["commit"], "research/t34_o2_harvest/export.py")
        )
        (repo / "research/t34_exit_b8/public_inputs.py").write_text(
            "# Offline public-checkpoint materialization stand-in.\n"
        )
        (repo / "research/t34_o2_harvest/runner_v2.py").write_text(
            "import json, os\nfrom pathlib import Path\n"
            "keys=['CUDA_VISIBLE_DEVICES','SNAKE_DQN_DEVICE','OMP_NUM_THREADS',"
            "'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS',"
            "'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']\n"
            "expected=['','cpu','1','1','1','1','1','1']\n"
            "seen={k:os.environ.get(k) for k in keys}\n"
            "if list(seen.values()) != expected:\n"
            "    raise RuntimeError('O2 CPU worker boundary violated: '+repr(seen))\n"
            "root=Path(os.environ['COLAB_CLI_ROOT'])\n"
            "(root/'worker-probe.json').write_text(json.dumps(seen))\n"
            "out=root/'t34-o2-run'\nout.mkdir()\n"
            "(out/'receipt.json').write_text('{}')\n"
        )
        (repo / "torch.py").write_text(
            "import os\nfrom types import SimpleNamespace\n"
            f"__version__='2.9.1'\nversion=SimpleNamespace(cuda={torch_cuda!r})\n"
            "cuda=SimpleNamespace(\n"
            " is_available=lambda:os.environ.get('CUDA_VISIBLE_DEVICES') != '',\n"
            " get_device_capability=lambda _: (8,9))\n"
        )
        # The shim skips network/package writes, creates a venv executable,
        # and otherwise executes the real Python including bootstrap's -c.
        shim = (
            f"#!{sys.executable}\n"
            "import os,sys\nfrom pathlib import Path\n"
            "args=sys.argv[1:]\n"
            "if args[:2]==['-m','pip']: sys.exit(0)\n"
            "if args[:3]==['-m','uv','python']: sys.exit(0)\n"
            "if args[:3]==['-m','uv','venv']:\n"
            "    p=Path(args[-1])/'bin/python'\n    p.parent.mkdir(parents=True)\n"
            "    p.write_text(Path(__file__).read_text())\n    p.chmod(0o755)\n"
            "    sys.exit(0)\n"
            f"os.execv({sys.executable!r},[{sys.executable!r},*args])\n"
        )
        (bindir / "python3").write_text(shim)
        (bindir / "python3").chmod(0o755)
        (bindir / "nvidia-smi").write_text("#!/bin/sh\necho 8.9\n")
        (bindir / "nvidia-smi").chmod(0o755)
        (bindir / "getconf").write_text("#!/bin/sh\necho 12\n")
        (bindir / "getconf").chmod(0o755)
        bootstrap = _blob(manifest["commit"], "research/colab_cli/bootstrap.sh")
        guard = 'case "$ROOT" in /content/snake-cli/*)'
        if bootstrap.count(guard) != 1:
            raise RuntimeError("pinned bootstrap root guard changed; local simulation refused")
        bootstrap = bootstrap.replace(
            guard, f'case "$ROOT" in {shlex.quote(str(base))}/snake-cli/*)'
        )
        path = root / "bootstrap.sh"
        path.write_text(bootstrap)
        env = {key: value for key, value in os.environ.items() if not key.startswith("COLAB_")}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("BASH_ENV", None)
        env.update(manifest["env"])
        env.update(
            PATH=f"{bindir}:/usr/bin:/bin",
            PYTHONPATH=str(repo),
            COLAB_CLI_ROOT=str(root),
            COLAB_INPUT_DIR=str(root / "inputs"),
            T34_O2_PYTHON=str(root / ".venv/bin/python"),
        )
        result = subprocess.run(
            ["bash", str(path), str(root), manifest["script"]],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(
                "O2 pinned remote bootstrap failed before the CPU worker probe: "
                f"exit={result.returncode}; CUDA_VISIBLE_DEVICES="
                f"{env.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}; "
                f"torch CUDA={torch_cuda!r}, L4 wheel expects 12.6. "
                "CUDA must be visible during bootstrap and hidden only by launch_v2.sh. "
                f"stderr={result.stderr.strip()}"
            )
        probe = json.loads((root / "worker-probe.json").read_text())
        if not (root / "output/o2.tar.gz.complete").is_file():
            raise RuntimeError("pinned O2 launcher did not complete its export path")
        return {
            "status": "PASS_LOCAL_BOOTSTRAP_SIMULATION",
            "commit": manifest["commit"],
            "script": manifest["script"],
            "worker_environment": probe,
            "root_guard_relocated_only": True,
            "stand_ins": ["installers", "L4/torch hardware", "public checkpoints", "runner"],
            "stdout": result.stdout.strip(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_bootstrap(json.loads(args.manifest.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
