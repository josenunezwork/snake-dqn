"""Registry of Tier-1 episode wrappers the fan-out runner may execute (pure data).

A wrapper names an existing screen/sweep harness whose ``ScreenSpec`` decides the arm's
veto and the entry schema; :mod:`research.runpod_fanout.episode` builds it and calls the
harness's own ``dev_screen.run_episode`` (live) so the records equal a local screen's.
Strict gates and serving qualification are deliberately absent and are refused by name
(:data:`research.runpod_fanout.jobspec` checks ``forbidden_name_fragments``).

``spec`` is ``(module, factory)``: ``factory`` is ``"make_spec"`` (called as
``make_spec(module.DOMAIN, {})``), ``"build_spec:<lambda>"`` (``build_spec(DOMAIN, lam, {})``)
or ``"SPEC"`` (the module attribute). ``simd`` maps arm -> ``(variant, lambda,
reference_lambda)`` for ``engine: simd``; a wrapper without it is live-only.
"""

from __future__ import annotations

from typing import Any, Dict

# Scripted anchors (src.evaluation.strict_promotion.scripted_agent; pinned by a test).
SCRIPTED_SHA256 = {
    "e1202f168b09b1b5ca94dbdcfafb511206595f834b9af8110afaee06de78dd41": "greedy_food",
    "ae30b423bef16f0cf99eae0322b6f19cf1eb1f24d2cf4c8c207a7551ffc045e4": "random_safe",
}
HERO_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"

_V8_SIMD = {"A": ("v7", 4.0, None), "C": ("v7", 4.0, None)}
_V8_SIMD.update({"B": ("v8", 8.0, 4.0), "D": ("v8", 8.0, 4.0)})

WRAPPERS: Dict[str, Dict[str, Any]] = {
    "apex-veto-v8-screen-v1": {
        "module": "research.apex_veto_v8_screen_20261002.screen",
        "spec": "make_spec",
        "arms": ("A", "B", "C", "D"),
        "simd": _V8_SIMD,
        "tier": "tier1",
    },
    "apex-veto-v7-screen-v1": {
        "module": "research.apex_veto_v7_screen_20261002.screen",
        "spec": "build_spec:4.0",
        "arms": ("A", "B", "C", "D"),
        "simd": None,
        "tier": "tier1",
    },
    "apex-veto-v8-dev-v1": {
        "module": "research.apex_veto_v8_lambda_sweep_20261002.sweep",
        "spec": "SPEC",
        "arms": ("A", "H400", "H800", "H1600", "R"),
        "simd": None,
        "tier": "tier1",
    },
    "apex-veto-v7-dev-v1": {
        "module": "research.apex_veto_v7_lambda_sweep_20261002.sweep",
        "spec": "SPEC",
        "arms": ("A", "L100", "L200", "L400", "R"),
        "simd": None,
        "tier": "tier1",
    },
}
