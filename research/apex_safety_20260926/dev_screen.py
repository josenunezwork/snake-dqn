#!/usr/bin/env python3
"""NON-AUTHORITATIVE development screen: Apex champion vs Apex + free-space veto.

Pre-registration: ``research/apex_safety_20260926/protocol.md``. This screen has
no bridge governance and cannot promote anything; see the protocol for the
decision rule and what its data may be used for.

Arms (paired live H5000 rollouts at the promotion-v2-watch-rect profile, with
the strict balanced roster construction for the frozen/scripted/mixed mixes):
  A  Apex champion (incumbent), unchanged.
  B  Apex champion + serving-time free-space veto (src/evaluation/safety_veto.py).
  C  Determinism control: arm A repeated on the first K worlds per mix, run
     after all A/B episodes; each C record must equal its A record exactly.

Usage (about 80 min at the defaults, see protocol.md):
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_safety_20260926/dev_screen.py \
    --out /path/to/new/dir --deadline-utc 2026-09-27T06:00:00+00:00

Opt-in additions (all default off; the default screen is unchanged): ``--use-slot-locks``
holds a shared CPU slot lock around the run, ``--require-ac-power`` refuses to start on
battery, and :class:`ScreenSpec` lets another screen reuse this harness with its own
namespace and per-arm vetoes (``research/apex_veto_v3_screen_20261001/screen.py``).
"""

from __future__ import annotations

import os

# Thread caps must be set before torch/numpy import anywhere in the process.
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "apex-safety-dev-screen/v1"
AUTHORITY = "non-authoritative-development-screen"
HORIZON = 5000
PROFILE_NAME = "promotion-v2-watch-rect"
MIXES = ("frozen", "scripted", "mixed")
ROSTER_WIDTH = 5
ALPHA = 0.05
REQUIRED_MIX_SUCCESSES = 2
MDE_FRACTION = 0.10  # informational sizing only, as in the strict calibration
# Informational pooled summaries (screen_stats); governance clear-loser filter is at 90%.
POOLED_CONFIDENCE = 0.90
# Pre-registered design sizes (protocol.md "Arms"). A non-smoke run with any other
# sizes gets decision NON_PREREGISTERED_DESIGN, so a smaller look followed by a
# larger one (the seeds are a prefix of one fixed recipe) cannot reach a decision.
PREREGISTERED_WORLDS_PER_MIX = 40
PREREGISTERED_DETERMINISM_WORLDS = 8
PREREGISTERED_DESIGN = (PREREGISTERED_WORLDS_PER_MIX, PREREGISTERED_DETERMINISM_WORLDS)
# Roster preflight: the strict pilot's 16 development worlds x 3 mixes.
EXPECTED_ROSTER_PARITY = 16 * len(MIXES)

# Seed namespace (see protocol.md "Worlds"): uint32 big-endian SHA-256 prefix.
SCREEN_DOMAIN = "apex-safety-screen-v1"
SCREEN_NAMESPACE = "worlds"

# Frozen recipe of research/task_aligned_challenger_20260924/namespaces.py; only
# its SHA-derived namespaces are recomputable without reading admission files.
CHALLENGER_DOMAIN = "task-aligned-challenger-20260924/original6102/v1"
CHALLENGER_COUNTS = {"development": 16, "shakedown": 4, "pilot": 16, "final": 120, "serving": 50}

DEFAULT_CONFIG = Path(__file__).resolve().parent / "deployment.yaml"
CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
PROFILE_DIGEST = "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b"
DEFAULT_CHECKPOINT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
# Governed research artifacts are read-only for this screen.
FORBIDDEN_OUTPUT_ROOT = "ongoing-research-20260913"
DEFAULT_PILOT_OUTPUT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/"
    "task-aligned-strict-challenge-20260926/pilot-v1/output"
)

CHAMPION = (
    "champion_a5_freespace_20260621.pth",
    "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
)
# Checkpoint pool in the strict pilot's order (the default tournament_eval pool).
POOL = (
    CHAMPION,
    ("best_apex_fs.pth", "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195"),
    ("best_apex_stage1_fs.pth", "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d"),
    ("best_apex_pre_fs.pth", "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e"),
)

# Opt-in shared CPU slot locks (the strict runner's lock files; never created here).
DEFAULT_SLOT_LOCK_ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909"
)
SLOT_LOCK_FILES = ("cpu-slot-1.lock", "cpu-slot-2.lock")
# ``ScreenSpec.arm_vetoes`` value: rollout's own ``hero_safety_veto=True`` (v2) install.
BUILTIN_VETO = "tournament_eval-builtin"


@dataclass(frozen=True)
class ScreenSpec:
    """What a screen built on this harness varies (opt-in; default = this screen).

    :data:`DEFAULT_SPEC` is this file's original Apex vs Apex + v2 veto screen and
    leaves its records, intent and summary unchanged. Another screen (for example
    ``research/apex_veto_v3_screen_20261001``) passes its own spec to :func:`main`.
    ``arm_vetoes`` maps arm -> ``None`` (no veto), :data:`BUILTIN_VETO`, or a callable
    ``install(hero) -> veto`` that replaces rollout's built-in install for that arm
    (the built-in vector61 guard still runs first). Arm C must mirror arm A.
    """

    name: str = "apex-safety-dev-screen"
    schema: str = SCHEMA
    authority: str = AUTHORITY
    domain: str = SCREEN_DOMAIN
    namespace: str = SCREEN_NAMESPACE
    protocol: Path = Path(__file__).resolve().parent / "protocol.md"
    arm_vetoes: Mapping[str, Any] = field(
        default_factory=lambda: {"A": None, "B": BUILTIN_VETO, "C": None}
    )
    arm_descriptions: Mapping[str, str] | None = None
    extra_namespaces: Mapping[str, Sequence[int]] | None = None
    require_slot_locks: bool = False
    require_ac_power: bool = False

    def veto_arms(self) -> Tuple[str, ...]:
        """Arms (of A, B) whose records carry a ``safety_veto`` probe."""
        return tuple(arm for arm in ("A", "B") if self.arm_vetoes.get(arm) is not None)


DEFAULT_SPEC = ScreenSpec()


@contextmanager
def hero_veto_installer(install: Callable[[Any], Any] | None) -> Iterator[List[Any]]:
    """Route rollout's hero veto install through ``install`` for one episode.

    Yields the list of installed veto objects. ``None`` patches nothing. The
    original installer still runs first, so its vector61-hero guard applies.
    """
    installed: List[Any] = []
    if install is None:
        yield installed
        return
    from src.scripts import tournament_eval

    original = tournament_eval._install_hero_safety_veto

    def patched(hero: Any, hero_spec: Any) -> Any:
        original(hero, hero_spec)
        veto = install(hero)
        installed.append(veto)
        return veto

    tournament_eval._install_hero_safety_veto = patched
    try:
        yield installed
    finally:
        tournament_eval._install_hero_safety_veto = original


def acquire_cpu_slots(root: Path, count: int, timeout: float) -> List[Any]:
    """Hold ``count`` free shared CPU slot locks (first free in file order) or raise.

    The lock files must already exist: they are opened read-only and never created.
    """
    if not 1 <= int(count) <= len(SLOT_LOCK_FILES):
        raise ValueError(f"slot count must be 1..{len(SLOT_LOCK_FILES)}")
    paths = [Path(root) / name for name in SLOT_LOCK_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"slot lock files missing (not created here): {missing}")
    started = time.monotonic()
    while True:
        held: List[Any] = []
        for path in paths:
            handle = path.open("r")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            held.append(handle)
            if len(held) == count:
                return held
        release_cpu_slots(held)
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"{count} CPU slot(s) unavailable for {timeout:.0f} s")
        time.sleep(0.5)


def release_cpu_slots(handles: Sequence[Any]) -> None:
    """Unlock and close slot handles from :func:`acquire_cpu_slots`."""
    for handle in handles:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def on_ac_power() -> bool:
    """The strict runner's AC-power check (macOS ``pmset``; other platforms pass)."""
    from research.apex_veto_strict_20260927.strict_run import on_ac_power as strict_on_ac

    return strict_on_ac()


def uint32_seed(domain: str, namespace: str, index: int) -> int:
    """SHA-256 prefix seed, the recipe of the task-aligned namespaces module."""
    digest = hashlib.sha256(f"{domain}|{namespace}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def screen_seeds(
    count: int, domain: str = SCREEN_DOMAIN, namespace: str = SCREEN_NAMESPACE
) -> List[int]:
    """The screen's world seeds (shared by all three mixes, as in the strict design)."""
    seeds = [uint32_seed(domain, namespace, index) for index in range(count)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("screen seed collision inside the namespace; stop, no repair")
    return seeds


def challenger_namespaces() -> Dict[str, List[int]]:
    """Recompute the task-aligned challenger's SHA-derived seed namespaces."""
    return {
        name: [uint32_seed(CHALLENGER_DOMAIN, name, index) for index in range(count)]
        for name, count in CHALLENGER_COUNTS.items()
    }


def observed_pilot_seeds(pilot_output: Path) -> Dict[str, List[int]]:
    """World seeds actually played by the strict pilot's saved records (read-only)."""
    observed: Dict[str, List[int]] = {}
    for stage in ("development", "pilot"):
        seeds = set()
        for path in sorted((pilot_output / stage / "producer" / "records").glob("*.json")):
            seeds.add(int(json.loads(path.read_text())["world_identity"]["seed"]))
        observed[stage] = sorted(seeds)
    return observed


def disjointness_report(
    seeds: Sequence[int],
    pilot_output: Path | None,
    extra_exclusions: Sequence[Path] = (),
    *,
    screen_domain: str = SCREEN_DOMAIN,
    extra_namespaces: Mapping[str, Sequence[int]] | None = None,
) -> Dict[str, Any]:
    """Check the screen worlds against every earlier namespace we can reach.

    Covers: the five SHA-derived task-aligned challenger namespaces (recomputed
    from the frozen recipe); the seeds actually played by the strict pilot's
    development and pilot records (which must equal the recomputed namespaces,
    proving the recipe); the small-integer seeds used by tournament_eval CLI
    defaults and tests (0..999); any JSON int lists passed explicitly; and, opt-in,
    ``extra_namespaces`` (name -> seeds, reported under the same key).
    """
    screen = set(int(seed) for seed in seeds)
    namespaces = challenger_namespaces()
    report: Dict[str, Any] = {
        "screen_domain": screen_domain,
        "screen_count": len(screen),
        "challenger_domain": CHALLENGER_DOMAIN,
        "challenger_namespaces": {
            name: {"count": len(values), "overlap": sorted(screen & set(values))}
            for name, values in namespaces.items()
        },
        "small_integer_seeds_0_999_overlap": sorted(screen & set(range(1000))),
        "extra_exclusions": {},
        "not_checked": [
            "task-aligned 'training' and selection world ids (bound in four admission "
            "metadata records outside the allowed read scope); they are historical ids, "
            "not SHA prefixes of this domain"
        ],
    }
    recipe_ok = True
    if pilot_output is not None and pilot_output.is_dir():
        observed = observed_pilot_seeds(pilot_output)
        recipe_ok = all(observed[s] == sorted(namespaces[s]) for s in ("development", "pilot"))
        report["strict_pilot_observed"] = {
            stage: {
                "count": len(values),
                "matches_recomputed_namespace": values == sorted(namespaces[stage]),
                "overlap": sorted(screen & set(values)),
            }
            for stage, values in observed.items()
        }
    else:
        report["strict_pilot_observed"] = None
    for path in extra_exclusions:
        values = {int(value) for value in json.loads(Path(path).read_text())}
        report["extra_exclusions"][str(path)] = sorted(screen & values)
    if extra_namespaces is not None:
        report["extra_namespaces"] = {
            name: {"count": len(values), "overlap": sorted(screen & {int(v) for v in values})}
            for name, values in extra_namespaces.items()
        }
    overlaps = [row["overlap"] for row in report["challenger_namespaces"].values()]
    overlaps.append(report["small_integer_seeds_0_999_overlap"])
    overlaps.extend(report["extra_exclusions"].values())
    overlaps.extend(row["overlap"] for row in report.get("extra_namespaces", {}).values())
    if report["strict_pilot_observed"]:
        overlaps.extend(row["overlap"] for row in report["strict_pilot_observed"].values())
    report["recipe_reproduces_pilot"] = recipe_ok
    report["disjoint"] = bool(recipe_ok and not any(overlaps))
    return report


def _design_rows(seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Strict balanced rosters (src.evaluation.strict_promotion.materialize_rosters)."""
    from src.evaluation.strict_promotion import materialize_rosters, scripted_agent

    anchors = [scripted_agent("greedy_food"), scripted_agent("random_safe")]
    pool = [{"kind": "checkpoint", "sha256": sha} for _, sha in POOL]
    rules = [
        {
            "slot": slot,
            "eligible_member_sha256s": (
                [row["sha256"] for row in pool] if slot % 2 else [anchors[1]["sha256"]]
            ),
        }
        for slot in range(1, ROSTER_WIDTH + 1)
    ]
    return materialize_rosters(
        final_world_seeds=list(seeds),
        roster_width=ROSTER_WIDTH,
        checkpoint_pool=pool,
        scripted_anchors=anchors,
        mixed_slot_rules=rules,
    )


def roster_parity_report(pilot_output: Path | None) -> Dict[str, Any]:
    """Rebuild the strict pilot's development rosters and compare identities.

    Proves this screen's roster construction (pool order, mixed-slot rules)
    reproduces the world identities recorded by the strict pilot.
    """
    from src.evaluation.strict_promotion import _expected_world_identity

    if pilot_output is None or not pilot_output.is_dir():
        return {"checked": False, "reason": "strict pilot output not available"}
    seeds = challenger_namespaces()["development"]
    mismatches = []
    compared = 0
    for row in _design_rows(seeds):
        path = (
            pilot_output
            / "development"
            / "producer"
            / "records"
            / f"incumbent-{row['mix']}-{row['world_seed']}.json"
        )
        if not path.is_file():
            mismatches.append({"missing": str(path.name)})
            continue
        recorded = json.loads(path.read_text())["world_identity"]
        compared += 1
        if recorded != _expected_world_identity(row):
            mismatches.append({"mix": row["mix"], "seed": row["world_seed"]})
    return {"checked": True, "compared": compared, "mismatches": mismatches}


def preflight_failures(
    disjoint: Mapping[str, Any], parity: Mapping[str, Any], smoke: bool
) -> List[str]:
    """Reasons the pre-run checks fail; empty means the screen may start.

    Outside smoke mode the checks must actually have run against the strict
    pilot's saved records: a missing or wrong ``--pilot-output`` is a failure,
    not a silent skip. Smoke mode only rejects detected overlaps or mismatches.
    """
    failures: List[str] = []
    if not disjoint.get("disjoint"):
        failures.append("screen worlds are not disjoint from earlier namespaces")
    if parity.get("checked") and parity.get("mismatches"):
        failures.append("roster parity mismatches against the strict pilot records")
    if smoke:
        return failures
    if disjoint.get("strict_pilot_observed") is None:
        failures.append("strict pilot output unavailable: recipe reproduction not checked")
    elif not disjoint.get("recipe_reproduces_pilot"):
        failures.append("recomputed namespaces do not reproduce the strict pilot's seeds")
    if not parity.get("checked"):
        failures.append("roster parity not checked (strict pilot output unavailable)")
    elif parity.get("compared") != EXPECTED_ROSTER_PARITY:
        failures.append(
            f"roster parity compared {parity.get('compared')} worlds, "
            f"expected {EXPECTED_ROSTER_PARITY}"
        )
    return failures


def preregistered_design_report(
    worlds_per_mix: int,
    determinism_worlds: int,
    preregistered: Tuple[int, int] = PREREGISTERED_DESIGN,
) -> Dict[str, Any]:
    """Pre-registered sizes vs this run's sizes (recorded in intent and summary)."""
    run = (int(worlds_per_mix), int(determinism_worlds))
    return {
        "preregistered_worlds_per_mix": int(preregistered[0]),
        "preregistered_determinism_worlds": int(preregistered[1]),
        "run_worlds_per_mix": run[0],
        "run_determinism_worlds": run[1],
        "matches": run == (int(preregistered[0]), int(preregistered[1])),
    }


def json_safe(value: Any, path: str = "", found: List[str] | None = None) -> Any:
    """Replace non-finite floats with ``None`` plus a sibling ``<key>_nonfinite`` tag.

    ``screen_stats`` returns ``df = inf`` (and can return other non-finite
    statistics) when every delta is identical, e.g. an inert veto; the
    create-only writer uses ``allow_nan=False``, so the summary must be
    sanitized first. ``found`` collects the dotted paths that were replaced.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if found is not None:
            found.append(path)
        return None
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(item, float) and not math.isfinite(item):
                tag = (
                    "nan"
                    if math.isnan(item)
                    else ("positive_infinity" if item > 0 else "negative_infinity")
                )
                out[f"{key}_nonfinite"] = tag
            out[key] = json_safe(item, child, found)
        return out
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[{i}]", found) for i, item in enumerate(value)]
    return value


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_checkpoints(checkpoint_dir: Path, out: Path) -> Dict[str, str]:
    """Verify each source checkpoint's SHA, copy it, re-verify the copy.

    Returns ``{sha256: snapshot_path}``; rollouts load only the snapshots.
    """
    target = out / "checkpoints"
    target.mkdir()
    lookup: Dict[str, str] = {}
    for name, expected in POOL:
        source = checkpoint_dir / name
        if sha256_file(source) != expected:
            raise ValueError(f"checkpoint {source} does not match pinned sha256 {expected}")
        copy = target / f"{expected}.pth"
        shutil.copyfile(source, copy)
        if sha256_file(copy) != expected:
            raise ValueError(f"snapshot copy of {source} changed during copy")
        lookup[expected] = str(copy)
    return lookup


def canonical_json(value: Any) -> str:
    """Canonical JSON used for record equality (determinism control)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_new_json(path: Path, value: Any) -> None:
    """Create-only JSON write (refuses to overwrite)."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def agent_lookup(snapshots: Mapping[str, str]) -> Dict[str, tuple]:
    """Map roster member sha256 -> tournament_eval AgentSpec."""
    from src.evaluation.strict_promotion import scripted_agent

    lookup = {sha: ("checkpoint", path) for sha, path in snapshots.items()}
    for name in ("greedy_food", "random_safe"):
        lookup[scripted_agent(name)["sha256"]] = ("scripted", name)
    return lookup


def run_episode(
    arm: str,
    row: Mapping[str, Any],
    world_index: int,
    lookup: Mapping[str, tuple],
    profile: Any,
    records_dir: Path,
    smoke_frames: int | None = None,
    spec: ScreenSpec = DEFAULT_SPEC,
) -> Dict[str, Any]:
    """Play one hero episode for ``arm`` on one roster row and persist it.

    ``spec`` (opt-in) picks the arm's veto; :data:`DEFAULT_SPEC` is the original
    "B = built-in v2 veto, A and C none" and writes the original entry shape.
    """
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import rollout

    hero = lookup[CHAMPION[1]]
    opponents = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    install = spec.arm_vetoes.get(arm)
    veto = install is not None
    started = time.monotonic()
    with hero_veto_installer(install if callable(install) else None) as installed:
        if smoke_frames is not None:
            # Plumbing smoke only: legacy (profile-free) path, truncated horizon.
            record = rollout(
                hero, opponents, smoke_frames, row["world_seed"], hero_safety_veto=veto
            )
        else:
            record = rollout(
                hero,
                opponents,
                HORIZON,
                row["world_seed"],
                profile=profile,
                world_identity=_expected_world_identity(row),
                mix_id=row["mix"],
                hero_safety_veto=veto,
            )
    entry = {
        "schema_version": spec.schema,
        "authority": spec.authority,
        "arm": arm,
        "mix": row["mix"],
        "world_index": world_index,
        "world_seed": row["world_seed"],
        "roster_member_sha256s": [slot["member_sha256"] for slot in row["slots"]],
        "hero_sha256": CHAMPION[1],
        "safety_veto": veto,
        "wall_seconds": time.monotonic() - started,
        "record": record,
    }
    if spec is not DEFAULT_SPEC:
        entry["screen"] = spec.name
        entry["safety_veto_method"] = record["probes"]["safety_veto"]["method"] if veto else None
        diagnostics = getattr(installed[-1], "diagnostics_record", None) if installed else None
        entry["veto_diagnostics"] = diagnostics() if callable(diagnostics) else None
    write_new_json(records_dir / f"{arm}-{row['mix']}-{row['world_seed']}.json", entry)
    return entry


def _death_table(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    causes = Counter(str(r["probes"].get("death_cause") or "survived") for r in records)
    return dict(sorted(causes.items()))


def _veto_totals(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals: Counter = Counter()
    consistent = 0
    for record in records:
        counters = record["probes"]["safety_veto"]["counters"]
        totals.update(counters)
        frames = record.get("denominators", {}).get("decision_frames")
        consistent += int(frames is None or frames == counters["decisions"])
    decisions = totals.get("decisions", 0)
    return {
        "episodes": len(records),
        "totals": dict(sorted(totals.items())),
        "veto_rate_per_decision": (totals["vetoes_applied"] / decisions) if decisions else None,
        "episodes_with_any_veto": sum(
            1 for r in records if r["probes"]["safety_veto"]["counters"]["vetoes_applied"] > 0
        ),
        "episodes_decisions_match_decision_frames": consistent,
    }


def pooled_summaries(
    deltas_by_mix: Mapping[str, Sequence[float]], paired_seeds: Mapping[str, Sequence[int]]
) -> Dict[str, Any]:
    """Informational pooled B-A mass-integral summaries across mixes (not the decision).

    Uses ``src.evaluation.screen_stats``: the governance serving-time estimate
    (equal-weight mean of per-mix means, 90% Welch CI and clear-loser flag), a
    DerSimonian-Laird pool with mixes as strata, and the crossed mixes x worlds
    mean over worlds paired in every mix (the mixes share world seeds, so the
    first two ignore a covariance that the crossed estimate models).
    """
    from src.evaluation import screen_stats

    out: Dict[str, Any] = {
        "role": "informational; the pre-registered decision uses holm_primary only",
        "metric": "mass_integral B - A",
    }
    usable = {m: list(d) for m, d in deltas_by_mix.items() if len(d) >= 2}

    def attempt(name: str, fn: Any) -> None:
        try:
            out[name] = fn()
        except ValueError as exc:  # too few worlds, zero variance, ...
            out[name] = {"available": False, "reason": str(exc)}

    attempt(
        "stratified_mean_of_means",
        lambda: screen_stats.stratified_mean_of_means(usable, confidence=POOLED_CONFIDENCE),
    )

    def random_effects() -> Dict[str, Any]:
        effects = screen_stats.per_seed_effects([usable[m] for m in usable])
        pooled = screen_stats.random_effects_pool(
            effects["effects"], effects["variances"], confidence=POOLED_CONFIDENCE
        )
        pooled["strata"] = list(usable)
        return pooled

    attempt("random_effects_across_mixes", random_effects)

    def crossed() -> Dict[str, Any]:
        mixes = list(deltas_by_mix)
        index = {m: {int(s): d for s, d in zip(paired_seeds[m], deltas_by_mix[m])} for m in mixes}
        common = sorted(set.intersection(*(set(index[m]) for m in mixes))) if mixes else []
        matrix = [[index[m][s] for s in common] for m in mixes]
        result = screen_stats.crossed_seed_world_mean(matrix, confidence=POOLED_CONFIDENCE)
        result.update({"rows": mixes, "note": "rows are mixes (reported as n_seeds)"})
        return result

    attempt("crossed_mixes_by_worlds", crossed)
    # Identical deltas (e.g. an inert veto) give df = inf; report null + a tag.
    found: List[str] = []
    safe = json_safe(out, "", found)
    safe["non_finite_fields"] = found
    return safe


def summarize(
    entries: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    determinism_worlds: int,
    mixes: Sequence[str] = MIXES,
    smoke: bool = False,
    preregistered: Tuple[int, int] = PREREGISTERED_DESIGN,
    veto_arms: Sequence[str] = ("B",),
) -> Dict[str, Any]:
    """Pre-registered analysis (protocol.md): paired B-A deltas, Holm, controls.

    ``preregistered`` is the ``(worlds_per_mix, determinism_worlds)`` design the
    decision is valid for; any other run size yields ``NON_PREREGISTERED_DESIGN``
    (tests pass their own small design). ``veto_arms`` (opt-in) lists the arms
    whose ``safety_veto`` counters are totalled as ``safety_veto_<arm>``. The result is JSON-safe
    (non-finite floats become ``None`` with a ``<key>_nonfinite`` tag).
    """
    from src.scripts.eval_stats import (
        holm_three_mix_superiority,
        mean,
        paired_delta_pilot_size,
        paired_delta_test,
    )

    by_key = {(e["arm"], e["mix"], int(e["world_seed"])): e["record"] for e in entries}
    per_mix: Dict[str, Any] = {}
    deltas_by_mix: Dict[str, List[float]] = {}
    seeds_by_mix: Dict[str, List[int]] = {}
    for mix in mixes:
        paired = [s for s in seeds if ("A", mix, s) in by_key and ("B", mix, s) in by_key]
        a = [by_key[("A", mix, s)] for s in paired]
        b = [by_key[("B", mix, s)] for s in paired]
        mass = [rb["mass_integral"] - ra["mass_integral"] for ra, rb in zip(a, b)]
        survival = [rb["survival_fraction"] - ra["survival_fraction"] for ra, rb in zip(a, b)]
        deltas_by_mix[mix] = mass
        seeds_by_mix[mix] = paired
        per_mix[mix] = {
            "planned_worlds": len(seeds),
            "paired_worlds": len(paired),
            "paired_seeds": paired,
            "primary_mass_integral": {
                "mean_A": mean([r["mass_integral"] for r in a]),
                "mean_B": mean([r["mass_integral"] for r in b]),
                "deltas_B_minus_A": mass,
                "wins_B": sum(1 for d in mass if d > 0),
                "losses_B": sum(1 for d in mass if d < 0),
                "ties": sum(1 for d in mass if d == 0),
                "one_sided_test": paired_delta_test(mass, ALPHA),
            },
            "secondary_survival_fraction": {
                "mean_A": mean([r["survival_fraction"] for r in a]),
                "mean_B": mean([r["survival_fraction"] for r in b]),
                "deltas_B_minus_A": survival,
                "one_sided_test_unadjusted": paired_delta_test(survival, ALPHA),
            },
            "death_causes": {"A": _death_table(a), "B": _death_table(b)},
        }
        arm_records = {"A": a, "B": b}
        for arm in veto_arms:
            per_mix[mix][f"safety_veto_{arm}"] = _veto_totals(arm_records[arm])
    holm = None
    sizing = None
    if tuple(mixes) == MIXES and all(len(deltas_by_mix[m]) >= 2 for m in MIXES):
        holm = holm_three_mix_superiority(
            deltas_by_mix, alpha=ALPHA, required_successes=REQUIRED_MIX_SUCCESSES
        )
        means_a = {m: per_mix[m]["primary_mass_integral"]["mean_A"] for m in MIXES}
        if all(value > 0 for value in means_a.values()):
            sizing = paired_delta_pilot_size(
                deltas_by_mix, {m: MDE_FRACTION * means_a[m] for m in MIXES}
            )
            sizing["mde_fraction_of_arm_A_mean"] = MDE_FRACTION
            sizing["note"] = "informational: sizes a later strict run on FRESH worlds"

    control = [e for e in entries if e["arm"] == "C"]
    mismatches = []
    for entry in control:
        key = ("A", entry["mix"], int(entry["world_seed"]))
        if key not in by_key or canonical_json(by_key[key]) != canonical_json(entry["record"]):
            mismatches.append({"mix": entry["mix"], "world_seed": entry["world_seed"]})
    planned_control = min(determinism_worlds, len(seeds)) * len(mixes)
    determinism = {
        "planned": planned_control,
        "compared": len(control),
        "identical": len(control) - len(mismatches),
        "mismatches": mismatches,
        "passes": len(control) == planned_control and not mismatches,
    }
    complete = all(per_mix[m]["paired_worlds"] == len(seeds) for m in mixes) and (
        len(control) == planned_control
    )
    design = preregistered_design_report(len(seeds), determinism_worlds, preregistered)
    if smoke:
        decision = "SMOKE_NO_DECISION"
    elif not design["matches"]:
        decision = "NON_PREREGISTERED_DESIGN"
    elif not complete:
        decision = "INCOMPLETE"
    elif not determinism["passes"]:
        decision = "INVALID_NONDETERMINISTIC"
    elif holm is not None and holm["passes"]:
        decision = "RECOMMEND_STRICT_GATE"
    else:
        decision = "NOT_ADVANCED"
    arm_seconds = {
        arm: [e["wall_seconds"] for e in entries if e["arm"] == arm] for arm in ("A", "B", "C")
    }
    summary = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "smoke": smoke,
        "complete": complete,
        "decision": decision,
        "decision_rule": (
            "NON_PREREGISTERED_DESIGN unless worlds_per_mix and determinism_worlds equal the "
            "pre-registered sizes; RECOMMEND_STRICT_GATE iff complete, A/C deterministic, and "
            f"one-sided Holm (alpha {ALPHA}) rejects mass-integral H0 in >= "
            f"{REQUIRED_MIX_SUCCESSES} of 3 mixes"
        ),
        "preregistered_design": design,
        "per_mix": per_mix,
        "holm_primary": holm,
        "pooled_informational": pooled_summaries(deltas_by_mix, seeds_by_mix),
        "determinism_control": determinism,
        "independent_pilot_sizing": sizing,
        "wall_seconds": {
            arm: {"episodes": len(v), "mean": mean(v), "total": sum(v)}
            for arm, v in arm_seconds.items()
        },
    }
    return json_safe(summary)


def _git_state() -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": git("status", "--porcelain")}


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--deadline-utc needs an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path, help="new output dir (must not exist)")
    parser.add_argument("--worlds-per-mix", type=int, default=40)
    parser.add_argument("--determinism-worlds", type=int, default=8)
    parser.add_argument("--deadline-utc", required=True, type=_utc)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    parser.add_argument(
        "--exclude-seeds-json",
        type=Path,
        action="append",
        default=[],
        help="extra JSON int list whose seeds must not overlap the screen worlds",
    )
    parser.add_argument(
        "--min-episode-budget-seconds",
        type=float,
        default=45.0,
        help="do not start an episode unless this much (or 2x the mean so far) remains",
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        help="plumbing smoke only (<= 500 frames, <= 2 episodes, legacy path, no decision)",
    )
    parser.add_argument("--smoke-mixes", default="scripted", help="smoke only: mixes to run")
    parser.add_argument(
        "--use-slot-locks",
        action="store_true",
        help="opt-in: hold shared CPU slot lock(s) from before --out exists to the summary",
    )
    parser.add_argument("--slot-lock-root", type=Path, default=DEFAULT_SLOT_LOCK_ROOT)
    parser.add_argument("--slots", type=int, default=1, help="slot locks to hold (1 process)")
    parser.add_argument("--slot-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--require-ac-power", action="store_true", help="opt-in: refuse to start on battery"
    )
    args = parser.parse_args(argv)
    minimum_worlds = 1 if args.smoke_frames is not None else 2
    if args.worlds_per_mix < minimum_worlds or not (
        0 <= args.determinism_worlds <= args.worlds_per_mix
    ):
        parser.error("need --worlds-per-mix >= 2 and 0 <= --determinism-worlds <= worlds")
    if args.smoke_frames is not None:
        args.mixes = tuple(m for m in args.smoke_mixes.split(",") if m)
        episodes = len(args.mixes) * (2 * args.worlds_per_mix + args.determinism_worlds)
        if not 0 < args.smoke_frames <= 500 or episodes > 2 or not set(args.mixes) <= set(MIXES):
            parser.error("smoke is limited to <= 500 frames and <= 2 episodes on known mixes")
    else:
        args.mixes = MIXES
    return args


def _configure_torch() -> None:
    import torch

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set in this process
        pass


def main(argv: Sequence[str] | None = None, spec: ScreenSpec = DEFAULT_SPEC) -> int:
    args = parse_args(argv)
    smoke = args.smoke_frames is not None
    if spec.arm_vetoes.get("C") is not spec.arm_vetoes.get("A"):
        print("screen spec: arm C must repeat arm A's veto configuration", file=sys.stderr)
        return 2
    use_locks = bool(args.use_slot_locks or spec.require_slot_locks)
    need_ac = bool(args.require_ac_power or spec.require_ac_power)
    if args.deadline_utc <= datetime.now(timezone.utc):
        print("deadline already passed", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    if out.exists():
        print(f"--out {out} already exists", file=sys.stderr)
        return 2
    if FORBIDDEN_OUTPUT_ROOT in out.parts:
        print(f"--out must not be inside {FORBIDDEN_OUTPUT_ROOT}", file=sys.stderr)
        return 2

    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    _configure_torch()
    if sha256_file(args.config) != CONFIG_SHA256:
        print("config bytes differ from the pinned deployment config", file=sys.stderr)
        return 2
    load_and_initialize_config(str(args.config))
    profile = evaluation_profile_for_name(PROFILE_NAME, HORIZON)
    if profile.digest != PROFILE_DIGEST:
        print("resolved profile differs from the strict pilot's profile", file=sys.stderr)
        return 2

    seeds = screen_seeds(args.worlds_per_mix, spec.domain, spec.namespace)
    disjoint = disjointness_report(
        seeds,
        args.pilot_output,
        args.exclude_seeds_json,
        screen_domain=spec.domain,
        extra_namespaces=spec.extra_namespaces,
    )
    parity = roster_parity_report(args.pilot_output)
    failures = preflight_failures(disjoint, parity, smoke)
    if failures:
        print(
            json.dumps(
                {"preflight_failures": failures, "disjointness": disjoint, "roster_parity": parity}
            ),
            file=sys.stderr,
        )
        return 2

    if need_ac and not on_ac_power():
        print("refusing to start on battery power (--require-ac-power)", file=sys.stderr)
        return 2
    slots: List[Any] = []
    if use_locks:
        try:
            slots = acquire_cpu_slots(args.slot_lock_root, args.slots, args.slot_timeout_seconds)
        except (OSError, TimeoutError, ValueError) as exc:
            print(f"CPU slot locks not acquired: {exc}", file=sys.stderr)
            return 2
    try:
        return _run_screen(
            args, argv, spec, out, seeds, disjoint, parity, profile, use_locks, smoke
        )
    finally:
        release_cpu_slots(slots)


def _run_screen(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    spec: ScreenSpec,
    out: Path,
    seeds: Sequence[int],
    disjoint: Mapping[str, Any],
    parity: Mapping[str, Any],
    profile: Any,
    use_locks: bool,
    smoke: bool,
) -> int:
    """The screen body (``main`` holds any slot locks around it)."""
    out.mkdir(parents=True)
    records_dir = out / "records"
    records_dir.mkdir()
    snapshots = snapshot_checkpoints(args.checkpoint_dir, out)
    lookup = agent_lookup(snapshots)
    rows = [row for row in _design_rows(seeds) if row["mix"] in args.mixes]
    # A then B per world (mix-major); every C control runs after all A/B pairs.
    plan = [(arm, row) for row in rows for arm in ("A", "B")]
    for mix in args.mixes:
        control_rows = [row for row in rows if row["mix"] == mix][: args.determinism_worlds]
        plan.extend(("C", row) for row in control_rows)
    intent = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "protocol": str(Path(spec.protocol).resolve()),
        "argv": list(sys.argv if argv is None else argv),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_utc": args.deadline_utc.isoformat(),
        "git": _git_state(),
        "config": {"path": str(Path(args.config).resolve()), "sha256": CONFIG_SHA256},
        "profile": {"name": PROFILE_NAME, "digest": PROFILE_DIGEST, "horizon": HORIZON},
        "smoke_frames": args.smoke_frames,
        "mixes": list(args.mixes),
        "hero": {"name": CHAMPION[0], "sha256": CHAMPION[1]},
        "checkpoint_pool": [{"name": name, "sha256": sha} for name, sha in POOL],
        "checkpoint_snapshots": snapshots,
        "arms": (
            dict(spec.arm_descriptions)
            if spec.arm_descriptions is not None
            else {
                "A": "champion",
                "B": "champion + free-space veto (src/evaluation/safety_veto.py)",
                "C": f"champion repeated on the first {args.determinism_worlds} worlds per mix",
            }
        ),
        "worlds": {"domain": spec.domain, "namespace": spec.namespace, "seeds": list(seeds)},
        "disjointness": disjoint,
        "roster_parity": parity,
        "preregistered_design": preregistered_design_report(
            args.worlds_per_mix, args.determinism_worlds
        ),
        "planned_episodes": len(plan),
        "threads": {"torch_intraop": 2, "torch_interop": 1},
    }
    if spec is not DEFAULT_SPEC:
        intent.update({"schema_version": spec.schema, "authority": spec.authority})
        intent["screen"] = spec.name
    if use_locks:
        intent["slot_locks"] = {
            "root": str(Path(args.slot_lock_root)),
            "held": args.slots,
            "held_from": "before --out existed until summary.json was written",
        }
    if args.require_ac_power or spec.require_ac_power:
        intent["ac_power_checked_at_start"] = True
    write_new_json(out / "intent.json", intent)

    entries: List[Dict[str, Any]] = []
    stopped_reason = None
    world_index = {seed: index for index, seed in enumerate(seeds)}
    with (out / "events.jsonl").open("x", encoding="utf-8") as events:
        for arm, row in plan:
            spent = [e["wall_seconds"] for e in entries]
            budget = max(
                args.min_episode_budget_seconds, 2 * sum(spent) / len(spent) if spent else 0
            )
            remaining = (args.deadline_utc - datetime.now(timezone.utc)).total_seconds()
            if remaining < budget:
                stopped_reason = f"deadline: {remaining:.0f}s left < {budget:.0f}s budget"
                break
            entry = run_episode(
                arm,
                row,
                world_index[row["world_seed"]],
                lookup,
                profile,
                records_dir,
                smoke_frames=args.smoke_frames,
                spec=spec,
            )
            entries.append(entry)
            events.write(
                canonical_json(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "done": len(entries),
                        "planned": len(plan),
                        "arm": arm,
                        "mix": row["mix"],
                        "world_seed": row["world_seed"],
                        "wall_seconds": round(entry["wall_seconds"], 3),
                        "mass_integral": entry["record"]["mass_integral"],
                    }
                )
                + "\n"
            )
            events.flush()

    for sha, path in snapshots.items():
        if sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the screen")
    summary = summarize(
        entries,
        seeds,
        args.determinism_worlds,
        args.mixes,
        smoke=smoke,
        veto_arms=spec.veto_arms(),
    )
    if spec is not DEFAULT_SPEC:
        summary.update({"schema_version": spec.schema, "authority": spec.authority})
    summary.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_run": len(entries),
            "planned_episodes": len(plan),
            "stopped_reason": stopped_reason,
        }
    )
    write_new_json(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "episodes_run": len(entries),
                "summary": str(out / "summary.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
