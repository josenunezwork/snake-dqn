"""Portable, deterministic seed-history artifacts for campaign harness tests.

The governed runs keep their complete pilot and strict artifacts outside the repository.
These helpers replace only those read-only inputs with small test-local equivalents; callers
still exercise the production parsers, collision checks, and fail-closed missing-input paths.
"""

from __future__ import annotations

import json
from pathlib import Path


def install_v8_seed_history(monkeypatch, root: Path) -> None:
    """Install the minimum complete prior-history inputs used by v8 serving tests."""
    from research.apex_safety_20260926 import dev_screen
    from research.apex_veto_v8_serving_20261003 import serving_run as v8
    from research.apex_veto_v8_strict_20261003 import spec as strict

    root.mkdir(parents=True, exist_ok=True)
    intent = root / "intent.json"
    rosters = root / "rosters.json"
    intent.write_text(json.dumps({"banks": {"final": [1_000_001, 1_000_002]}}))
    rosters.write_text(json.dumps([{"world_seed": 1_000_003}]))
    monkeypatch.setattr(v8, "V8_STRICT_INTENT", intent)
    monkeypatch.setattr(v8, "V8_STRICT_ROSTERS", rosters)

    # Preserve the production exclusion/report implementations.  Only their unavailable
    # read-only artifact inputs are replaced with complete, parser-readable local files.
    pilot = root / "pilot"
    namespaces = dev_screen.challenger_namespaces()
    for stage in ("development", "pilot"):
        records = pilot / stage / "producer" / "records"
        records.mkdir(parents=True)
        for index, seed in enumerate(namespaces[stage]):
            (records / f"{index}.json").write_text(json.dumps({"world_identity": {"seed": seed}}))
    monkeypatch.setattr(dev_screen, "DEFAULT_PILOT_OUTPUT", pilot)
    observed = []
    for index, source in enumerate(strict.OBSERVED_SEED_SOURCES):
        path = root / "observed" / f"{index}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"world_seed": 1_100_000 + index}))
        observed.append(path)
    original_observed_seeds = strict.observed_seeds
    monkeypatch.setattr(strict, "observed_seeds", lambda: original_observed_seeds(tuple(observed)))


def install_frp3_training_exclusions(monkeypatch, root: Path, module) -> None:
    """Install a hash-bound test-local FRP training-exclusion artifact."""
    import hashlib

    path = root / "frp3-training-exclusions.json"
    path.write_text(json.dumps([2_000_001, 2_000_002]))
    monkeypatch.setattr(module, "FRP3_EXCLUSIONS", path)
    monkeypatch.setattr(
        module, "FRP3_EXCLUSIONS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
