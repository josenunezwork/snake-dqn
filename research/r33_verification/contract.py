"""Frozen R33 package boundary; no allocation or scientific authority."""

NAMESPACE = "r33-linux-contract-pilot"
ALLOCATION = "allocation-0001"
RULING_SHA256 = "7db57b262f1bfa4ae70597772d79e2904f8d6a885ea530c54178d869911cf07a"
SOURCE_PATHS = [
    "research/r33_verification",
    "research/colab_cli",
    "tools/ops/package_spec.py",
    "tools/ops/task_journal.py",
    "tools/ops/proof_dependencies.py",
    "tools/ops/import_preflight.py",
    "tests/test_r33_package.py",
    "tests/test_r33_journal.py",
    "tests/test_r33_proof_dependencies.py",
    "tests/test_r33_independent.py",
    "tests/test_r33_pilot_integration.py",
    "tests/test_colab_cli_schema_failures.py",
    "tests/test_colab_cli_fault_properties.py",
    "tests/fixtures/fake_colab.py",
    "tests/fixtures/fake_colab_helper.py",
    "research/bl_labels/slowdown_guard.py",
    "research/bl_labels/test_slowdown_guard.py",
    "research/bl_labels/fixtures/d256_slowdown_timestamps.json",
    "research/bl_labels/mac_driver.py",
    "research/bl_labels/slowdown_resume.py",
    "research/bl_labels/exports.py",
    "research/bl_labels/w12_runtime.py",
    "research/compute/factory_interim_launcher.py",
    "research/compute/factory_companion_commands.py",
    "research/mac_lease.py",
    "pytest.ini",
    "pyproject.toml",
]
ATTEMPTS = [
    {"effective_minutes": 240, "unit_cap": 0.320, "reserve_units": 0.053333333, "estimate": 120},
    {"effective_minutes": 24, "unit_cap": 0.032, "reserve_units": 0.005333333, "estimate": 12},
    {"effective_minutes": 24, "unit_cap": 0.032, "reserve_units": 0.005333334, "estimate": 12},
]


def environment_identity(root):
    """Hash the declared Python/CPU/bootstrap dependency and thread contract."""
    import hashlib
    import json

    descriptor = {
        "python": "3.12",
        "pytest": "9.0.1",
        "device": "cpu",
        "threads": 1,
        "workers": 1,
        "requirements_sha256": hashlib.sha256(
            (root / "research/r33_verification/requirements.txt").read_bytes()
        ).hexdigest(),
        "bootstrap_sha256": hashlib.sha256(
            (root / "research/colab_cli/bootstrap.sh").read_bytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
