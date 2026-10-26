"""Package-level exhaustive diagnostic coverage."""

from __future__ import annotations

import copy

from research.bl_residual import f5_environment as environment
from tests.test_bl_residual_f5_environment import _package


def test_package_collector_reports_content_metadata_and_aggregate_failures() -> None:
    package = _package("/venv")
    package["installed_contents"]["second.py"] = {
        "raw_sha256": "0" * 64,
        "canonical_sha256": "f" * 64,
    }
    package["installed_contents_sha256"] = "0" * 64
    package["installed_metadata"]["METADATA"]["raw_sha256"] = "0" * 64
    package["installed_metadata"]["WHEEL"]["raw_sha256"] = "0" * 64

    failures = environment.collect_package_failures("torch", copy.deepcopy(package))

    assert any("installed library content was normalized" in failure for failure in failures)
    assert any("installed contents digest differs" in failure for failure in failures)
    assert any("raw metadata hash differs" in failure for failure in failures)

    assert any("torch: METADATA" in failure for failure in failures)
    assert any("torch: WHEEL" in failure for failure in failures)
