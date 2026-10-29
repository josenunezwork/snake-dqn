"""Provider-free hardware admission and actual launcher architecture twins."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.sandbox_a_resident import run


@pytest.mark.parametrize(
    "available,name,accepted",
    [
        (True, "NVIDIA H100 80GB HBM3", True),
        (True, "NVIDIA A100-SXM4-40GB", True),
        (True, "NVIDIA L4", False),
        (False, "NVIDIA H100 80GB HBM3", False),
    ],
)
def test_assay_cli_hardware_guard(tmp_path, monkeypatch, available, name, accepted):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"tiny": False}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--checkpoint",
            str(tmp_path / "unused.pth"),
            "--control",
            str(control),
            "--ledger",
            str(tmp_path / "unused-ledger.json"),
            "--output",
            str(tmp_path / "output"),
            "--device",
            "cuda",
        ],
    )
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(run.torch.cuda, "get_device_name", lambda: name)
    monkeypatch.setattr(run.torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(run.torch, "set_num_interop_threads", lambda *_: None)
    monkeypatch.setattr(run.torch, "manual_seed", lambda *_: None)

    class PassedHardwareGuard(Exception):
        pass

    def ledger(*_):
        raise PassedHardwareGuard

    monkeypatch.setattr(run, "load_ledger", ledger)
    if accepted:
        with pytest.raises(PassedHardwareGuard):
            run.main()
    else:
        with pytest.raises(ValueError, match="H100 or A100 required"):
            run.main()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("script", ["job.sh", "train_job.sh"])
@pytest.mark.parametrize("gpu,arch", [("H100", "90"), ("A100", "80")])
@pytest.mark.parametrize("missing_hopper", [False, True])
def test_launchers_compile_for_requested_hardware(tmp_path, script, gpu, arch, missing_hopper):
    root = Path(__file__).resolve().parents[2]
    source = root / "research/sandbox_a_resident" / script
    launched = tmp_path / script
    code = source.read_text()
    if missing_hopper:
        # Fault twin restores the reviewed sm_80-only defect in an isolated copy.
        assert "TORCH_CUDA_ARCH_LIST='8.0;9.0'" in code
        code = code.replace("TORCH_CUDA_ARCH_LIST='8.0;9.0'", "TORCH_CUDA_ARCH_LIST=8.0")
    launched.write_text(code)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_python = binaries / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] != '-m':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "from torch.utils.cpp_extension import _get_cuda_arch_flags\n"
        "flags = _get_cuda_arch_flags()\n"
        "Path(os.environ['FLAG_RECORD']).write_text(json.dumps(flags))\n"
        "required = 'code=sm_' + os.environ['REQUIRED_ARCH']\n"
        "raise SystemExit(0 if any(required in flag for flag in flags) else 73)\n"
    )
    fake_python.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    record = tmp_path / "flags.json"
    result = subprocess.run(
        ["bash", str(launched)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "COLAB_CLI_ROOT": str(runtime),
            "COLAB_INPUT_DIR": str(tmp_path / "inputs"),
            "SANDBOX_A_SOURCE_COMMIT": "a" * 40,
            "SANDBOX_A_CONTROL_JSON": "{}",
            "FLAG_RECORD": str(record),
            "REQUIRED_ARCH": arch,
            "TORCH_CUDA_ARCH_LIST": "bogus-inherited-value",
        },
    )
    expected = 73 if missing_hopper and gpu == "H100" else 0
    assert result.returncode == expected, result.stderr
    flags = json.loads(record.read_text())
    assert "-gencode=arch=compute_80,code=sm_80" in flags
    assert ("-gencode=arch=compute_90,code=sm_90" in flags) is (not missing_hopper)
