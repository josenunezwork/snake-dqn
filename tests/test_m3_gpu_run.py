"""Pure parts of the M3-B GPU runner (no RunPod call is ever made under pytest)."""

from __future__ import annotations

from research.redesign_m3_20261008 import gpu_run


def test_pod_body_is_money_safe():
    body = gpu_run.pod_body("m3b-gpu-x", "a" * 64, 1000.0, 2000.0)
    assert body["cloudType"] == "SECURE" and body["gpuCount"] == 1
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert body["env"]["M3_DEADLINE_EPOCH"] == "1000"
    assert body["env"]["M3_SELF_DELETE_EPOCH"] == "2000"
    assert "RUNPOD_API_KEY" not in body["env"] and body["volumeInGb"] == 0
    assert body["minRAMPerGPU"] >= 64 and body["minVCPUPerGPU"] >= 8
    assert gpu_run.MAX_HOURLY <= 1.0 and gpu_run.G2_RATE >= 4 * gpu_run.MAC_RATE


def test_g2_decision():
    ok = gpu_run.g2_decision({s: 1500.0 for s in gpu_run.SEEDS}, 1800, 6.5 * 3600)
    assert ok["pass"] and ok["combined_rate"] == 7500.0
    slow = gpu_run.g2_decision({s: 1000.0 for s in gpu_run.SEEDS}, 1800, 6.5 * 3600)
    assert not slow["pass"] and slow["reason"] == "rate"
    late = gpu_run.g2_decision({s: 1200.0 for s in gpu_run.SEEDS}, 1800, 3 * 3600)
    assert not late["pass"] and late["reason"] == "lifetime"
    missing = gpu_run.g2_decision({0: 9000.0}, 1800, 6 * 3600)
    assert not missing["pass"]


def test_spend_cap():
    assert not gpu_run.spend_exceeded(66.0, 60.0, 8.0)
    assert gpu_run.spend_exceeded(66.0, 58.4, 8.0)


def test_commands():
    argv = gpu_run.train_argv(3)
    assert "--device" in argv and argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--transitions") + 1] == str(20_000_000)
    setup = gpu_run.setup_argv("f" * 64)[2]
    assert "sha256sum -c" in setup and "numba==0.68.0" in setup and "SETUP_OK" in setup
