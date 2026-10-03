#!/usr/bin/env python3
"""Group-sequential Tier-2 strict runner template (method strict-sequential-obf-bonferroni-v1).

A parameterized harness for the group-sequential strict gate of
``docs/research/governance_amendment_sequential_gates_2026-10-02.md``.  The study-specific
parts (arms, episode function, rosters, world namespaces, bands) are supplied by a
:class:`StudySpec`; everything else is fixed here and mirrors the hardening of the fixed-N
strict packages (``research/apex_veto_v{5,7}_strict_*``): monotonic-clock heartbeats, an
AC-power guard (also polled while children run), two-slot sharding under inherited CPU slot
locks, stage caps plus a handoff reserve, create-only (write-once) outputs, intent hashing,
a source-closure re-hash before every segment, an independent stdlib-only audit
(``sequential_audit.py``) and a closeout that never relabels a failure.

Template versions: ``sequential-strict-template/v1`` (spec ``band_policy="block_at_stop"``,
calibration-reference bands) and ``sequential-strict-template/v2`` (``"paired_ni_at_stop"``,
the paired survival bands of ``docs/research/governance_amendment_paired_bands_2026-10-03.md``:
judged once at the qualifying look on candidate-minus-incumbent per-world deltas, with the
per-study paired-band check as a hash-bound pre-registration artifact).  v1 intents, plan
dicts, spec descriptors and look receipts keep exactly their v1 content.

Flow of ``run`` (one intent, one attempt, one closeout):

1. **Calibration** (incumbent only, dev worlds, both shards): reference means, the absolute
   non-inferiority margin ``delta_NI`` and the behavioral bands (v2: descriptive band-metric
   means only) -> ``calibration.json``.
2. **Skew check** (required by the amendment, before the first final world): the frozen
   resampling probe (``skew_probe.py --part resample``) on the study's saved screen/pilot
   paired deltas, per mix, with the frozen ``N_max``, look fractions and ``delta_NI``
   -> ``skew_check.json``.  A failure ends the run ``SKEW_CHECK_FAILED``.
3. **Final looks**: world-major round-robin units (``sequential_gate.round_robin_plan``);
   for look ``k`` every worker runs exactly its next ``worker_look_counts`` units and exits
   (the barrier).  The parent then checks that the records on disk are exactly the look's
   unit prefix, computes ``sequential_decision`` and writes ``looks/look-<k>.json``
   create-only.  A worker refuses to start look ``k + 1`` unless that receipt exists and
   says ``continue``, and it binds the receipt's sha256 into its own ``started.json``.
4. **Audit**: the independent audit child, then ``closeout.json`` (and ``receipt.json`` on
   ``STRICT_PASS``).

There is no resume (as in the v7 package): a parent that dies without ``closeout.json``
leaves the run permanently ``ABANDONED`` (see ``status``); ``prepare`` and ``run`` refuse
it.  Any failure the parent sees (worker crash, watchdog, battery, SIGTERM/SIGHUP/SIGINT,
drift) is closed out ``INVALID_STOP``.  A ``dry_run`` intent (in-process executor or
injected runners, dirty sources, any lock root) can only end ``DRY_RUN_*`` or a failure.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # the worker child runs with ``python -I``
    sys.path.insert(0, str(REPO))

from src.evaluation.sequential_gate import (  # noqa: E402
    DEFAULT_BAND_MARGIN_Z,
    DEFAULT_FRACTIONS,
    DEFAULT_FUTILITY_CP,
    SEQUENTIAL_GATE_METHOD,
    SequentialGatePlan,
    band_check,
    plan_paired_band_check,
    round_robin_plan,
    sequential_decision,
    sequential_gate_plan,
    worker_look_counts,
)

TEMPLATE_VERSION = "sequential-strict-template/v1"
# Template v2 adds band_policy "paired_ni_at_stop" (governance amendment paired bands,
# 2026-10-03).  An intent declares its version: v1 iff the spec's band policy is the legacy
# "block_at_stop" (v1 intents, plan dicts and spec descriptors are unchanged), v2 iff it is
# "paired_ni_at_stop".
TEMPLATE_VERSION_PAIRED = "sequential-strict-template/v2"
TEMPLATE_VERSIONS = {
    "block_at_stop": TEMPLATE_VERSION,
    "paired_ni_at_stop": TEMPLATE_VERSION_PAIRED,
}
HERE = Path(__file__).resolve().parent
TEMPLATE_README = HERE / "README.md"
INDEPENDENT_AUDIT = HERE / "sequential_audit.py"
SKEW_PROBE = REPO / "research" / "sequential_gate_validation_20261002" / "skew_probe.py"
PAIRED_BAND_SIMULATOR = REPO / "research" / "paired_band_validation_20261003" / "simulate.py"
PYTHON = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
# The global CPU slot lock root the v7 strict package uses (strict_run.SLOT_LOCK_ROOT).
SLOT_LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")
# Append-only ledger of started sequential strict runs, beside the global slot locks and
# outside every output root.  A final namespace may start at most one production run.
LEDGER_PATH = SLOT_LOCK_ROOT / "sequential-strict-ledger.jsonl"
DEFAULT_CLOSURE_ROOTS = (
    "src",
    "research/sequential_strict_template",
    "research/sequential_gate_validation_20261002",
)
CLOSURE_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".md")

ARMS = ("incumbent", "candidate")
PHASES = ("calibration", "final")
TERMINAL_DECISIONS = ("STOP_PASS", "STOP_FAIL_BANDS", "FINAL_PASS", "FINAL_FAIL")
PASSING_DECISIONS = ("STOP_PASS", "FINAL_PASS")
OUTCOMES = (
    "STRICT_PASS",
    "STRICT_FAIL",
    "SKEW_CHECK_FAILED",
    "DRY_RUN_PASS",
    "DRY_RUN_FAIL",
    "DRY_RUN_SKEW_CHECK_FAILED",
    "STOP_INFEASIBLE",
    "INCOMPLETE",
    "INVALID_STOP",
)
FUTILITY_ACTIONS = ("stop", "continue")
# Amendment thresholds: per-mix efficacy any-look rate <= 1.2 x (family alpha / n_mixes)
# (0.020 at the default 0.05 / 3) and scripted NI any-look rate <= 0.06.
SKEW_EFFICACY_FACTOR = 1.2
SKEW_NI_LIMIT = 0.06
SKEW_MIN_DELTAS = 20
SKEW_REMEDY_ON_FAIL = "stop_and_escalate"
SKEW_REMEDY_LADDER = [
    "stop: the study does not proceed under this method (no final world is played)",
    "path 1: a new method version with a skew-robust interim and final statistic "
    "(bootstrap-t or skew-corrected t), validated by its own Monte Carlo before use",
    "path 2: a governance decision on the fixed-N gate, which shares the sensitivity "
    "and needs the same resampling check",
    "not a remedy: moving the first look later (amendment item 9)",
]
DEFAULT_SKEW_REPS = 300_000
# Paired survival bands (template v2).  Ratified 2026-10-03: M 0.05, alpha 0.05, pointwise at
# the qualifying look, absolute floor 0.30.  An intent always states its values explicitly.
RATIFIED_PAIRED_BAND = {
    "band_ni_margin": 0.05,
    "band_alpha": 0.05,
    "band_bound": "pointwise",
    "band_floor": 0.30,
}
PAIRED_BAND_KEYS = tuple(RATIFIED_PAIRED_BAND)
PAIRED_BAND_METRIC = "survival_fraction"
# Per-study pre-registration check (amendment "Resampling check on the study's own data"):
# simulate.py --part gate on the study's screen pool with the frozen plan; acceptance is a
# joint error of an exactly-M one-mix regression <= 1.2 x band_alpha in every row.
PAIRED_CHECK_FACTOR = 1.2
PAIRED_CHECK_MIN_REPS = 20_000
# Mass effects the check must cover, as multiples of the MDE (0.67 is matched to 0.6-0.7).
PAIRED_CHECK_THETA_MULTIPLES = (0.5, 1.0, 1.5, 2.0)
PAIRED_CHECK_TWO_THIRDS = (0.6, 0.7)
CALIBRATION_BAND_RULES = {
    "block_at_stop": "reference = calibration incumbent mean of the band metric in the band mix",
    "paired_ni_at_stop": "none: paired bands have no reference; calibration band-metric "
    "means are descriptive only",
}

DEFAULT_CAPS = {"calibration": 1800, "skew_check": 900, "final": 45000, "audit": 900}
HANDOFF_SECONDS = 120
WORKERS = 2
THREADS_PER_WORKER = 2
N_MAX_CAP = 300
RSS_BYTES = 8 * 1024**3
AVAILABLE_BYTES = math.ceil(9.6 * 1024**3)
POLL_SECONDS = 1.0
POWER_POLL_SECONDS = 30.0
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_STALE_SECONDS = 600.0
SLOT_TIMEOUT_SECONDS = 180.0
WORKER_STOP_MARGIN_SECONDS = 30.0
MIN_EPISODE_BUDGET_SECONDS = 45.0
PARENT_GONE_EXIT = 75
PRODUCTION_EXECUTOR = "subprocess"
PRODUCTION_SKEW_RUNNER = "subprocess_skew_runner"
PRODUCTION_AUDIT_RUNNER = "subprocess_audit_runner"


class StrictRunError(RuntimeError):
    """A fail-closed check in this harness."""


class DeadlineStop(Exception):
    """A cap or deadline ended the run (outcome INCOMPLETE, never relabelled)."""


class Interrupted(BaseException):
    """SIGTERM/SIGHUP/SIGINT received by the run parent."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`StrictRunError` unless ``condition`` (not disabled by -O)."""
    if not condition:
        raise StrictRunError(message)


# ---------------------------------------------------------------- io and hashing


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "timestamps need an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def read_json(path: Path) -> Any:
    """Read JSON, rejecting NaN/Infinity constants."""

    def reject(token: str) -> None:
        raise StrictRunError(f"non-finite JSON constant {token} in {path}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with strings (JSON is written allow_nan=False)."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_once(path: Path, value: Any) -> str:
    """Atomic create-only, fsynced JSON write; returns the file's sha256.

    The bytes go to a private temporary file which is hard-linked to ``path``: ``os.link``
    fails if ``path`` exists (create-only) and a crash never leaves a truncated ``path``.
    """
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- study spec (pluggable)


def default_row(phase: str, mix: str, world_index: int, world_seed: int) -> Dict[str, Any]:
    """Roster row of one world; a spec adds its opponent roster / world identity here."""
    return {"mix": mix, "world_index": int(world_index), "world_seed": int(world_seed)}


def _no_setup(intent: Mapping[str, Any]) -> Any:
    return None


def _no_problems(entry: Mapping[str, Any], row: Mapping[str, Any]) -> List[str]:
    return []


def survival_bands(
    mixes: Sequence[str], below: float = 0.02, above: float = 1.0
) -> Tuple[Dict[str, Any], ...]:
    """The fixed-N packages' band: candidate mean survival_fraction in [ref - 0.02, ref + 1]."""
    return tuple(
        {"metric": "survival_fraction", "mix": mix, "lower_offset": -below, "upper_offset": above}
        for mix in mixes
    )


def paired_survival_bands(mixes: Sequence[str]) -> Tuple[Dict[str, Any], ...]:
    """Template v2 bands: paired survival_fraction noninferiority, one band per mix."""
    return tuple({"metric": PAIRED_BAND_METRIC, "mix": mix} for mix in mixes)


@dataclass(frozen=True)
class StudySpec:
    """The study-specific (pluggable) part of a sequential strict run.

    Declarative fields enter the intent through :meth:`descriptor` and are re-checked by the
    workers and the audit; callables are code bound by the source closure:

    * ``arm_identities()`` returns ``{"incumbent": {...}, "candidate": {...}}`` (checkpoint
      sha256, wrapper method, module sha256s, ...).  It is frozen in the intent and
      recomputed before every segment; any difference stops the run.
    * ``episode_runner(episode, row, context)`` plays one hero episode and returns the
      rollout record (a mapping with finite ``primary_metric`` and every band metric).
    * ``worker_setup(intent)`` builds the per-worker ``context`` once.
    * ``build_row(phase, mix, world_index, world_seed)`` returns the roster row of a world
      (must keep those four keys); rows are frozen in ``rosters.json``.
    * ``validate_record(entry, row)`` returns record-shape problems (empty = valid).
    * ``excluded_seeds()`` returns every earlier namespace whose seeds must be disjoint.
    * ``protocol_path``, ``oc_report_path`` (operating-characteristics simulation of the
      frozen plan) and ``band_cost_report_path`` (look-1 early-stop band cost) are required
      pre-registration documents; their sha256s are frozen in the intent.
    * ``band_policy`` is ``"block_at_stop"`` (template v1, calibration-reference bands with
      offsets, see :func:`survival_bands`) or ``"paired_ni_at_stop"`` (template v2, paired
      survival bands, see :func:`paired_survival_bands`).  v2 also needs
      ``paired_band_check_path``: the ``simulate.py --part gate`` output of the per-study
      paired-band check on the study's screen pool with the frozen plan.
    """

    study_id: str
    schema: str
    namespaces: Mapping[str, str]
    arm_identities: Callable[[], Mapping[str, Any]]
    episode_runner: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Mapping[str, Any]]
    excluded_seeds: Callable[[], Mapping[str, Sequence[int]]]
    protocol_path: str
    oc_report_path: str
    band_cost_report_path: str
    mixes: Tuple[str, ...] = ("frozen", "scripted", "mixed")
    scripted_mix: str = "scripted"
    primary_metric: str = "mass_integral"
    ni_fraction: float = 0.03
    bands: Tuple[Mapping[str, Any], ...] = field(
        default_factory=lambda: survival_bands(("frozen", "scripted", "mixed"))
    )
    worker_setup: Callable[[Mapping[str, Any]], Any] = _no_setup
    build_row: Callable[[str, str, int, int], Dict[str, Any]] = default_row
    validate_record: Callable[[Mapping[str, Any], Mapping[str, Any]], List[str]] = _no_problems
    closure_roots: Tuple[str, ...] = DEFAULT_CLOSURE_ROOTS
    band_policy: str = "block_at_stop"
    paired_band_check_path: str = ""

    @property
    def template_version(self) -> str:
        return TEMPLATE_VERSIONS[self.band_policy]

    @property
    def paired(self) -> bool:
        return self.band_policy == "paired_ni_at_stop"

    def preregistration_docs(self) -> Tuple[str, ...]:
        if self.paired:
            return PREREGISTRATION_DOCS + ("paired_band_check_path",)
        return PREREGISTRATION_DOCS

    def descriptor(self) -> Dict[str, Any]:
        out = self._descriptor_v1()
        if self.paired:  # v1 descriptors stay byte-identical
            out["template_version"] = TEMPLATE_VERSION_PAIRED
            out["band_policy"] = self.band_policy
            out["paired_band_check_path"] = self.paired_band_check_path
        return out

    def _descriptor_v1(self) -> Dict[str, Any]:
        return {
            "template_version": TEMPLATE_VERSION,
            "study_id": self.study_id,
            "schema": self.schema,
            "namespaces": {k: str(v) for k, v in sorted(self.namespaces.items())},
            "mixes": list(self.mixes),
            "scripted_mix": self.scripted_mix,
            "primary_metric": self.primary_metric,
            "ni_fraction": float(self.ni_fraction),
            "bands": [dict(band) for band in self.bands],
            "closure_roots": list(self.closure_roots),
            "protocol_path": self.protocol_path,
            "oc_report_path": self.oc_report_path,
            "band_cost_report_path": self.band_cost_report_path,
        }


PREREGISTRATION_DOCS = ("protocol_path", "oc_report_path", "band_cost_report_path")


def validate_spec(spec: StudySpec) -> None:
    require(isinstance(spec, StudySpec), "spec must be a StudySpec")
    require(set(spec.namespaces) == set(PHASES), f"namespaces must cover exactly {PHASES}")
    require(len(set(spec.namespaces.values())) == len(PHASES), "namespace domains must differ")
    require(len(spec.mixes) >= 1 and len(set(spec.mixes)) == len(spec.mixes), "mixes")
    require(spec.scripted_mix in spec.mixes, "scripted mix is not a mix")
    require(0.0 < float(spec.ni_fraction) < 1.0, "ni_fraction must be in (0, 1)")
    require(
        spec.band_policy in TEMPLATE_VERSIONS,
        f"band_policy must be one of {sorted(TEMPLATE_VERSIONS)}",
    )
    for name in spec.preregistration_docs():
        value = getattr(spec, name)
        require(isinstance(value, str) and bool(value.strip()), f"spec.{name} is required")
    if spec.paired:
        validate_paired_bands(spec)
        return
    require(
        spec.paired_band_check_path == "",
        "paired_band_check_path is only used under band_policy paired_ni_at_stop",
    )
    for band in spec.bands:
        require(
            set(band) == {"metric", "mix", "lower_offset", "upper_offset"}
            and band["mix"] in spec.mixes
            and float(band["lower_offset"]) <= float(band["upper_offset"]),
            f"malformed band {band}",
        )


def validate_paired_bands(spec: StudySpec) -> None:
    """Template v2 bands: ``{"metric": "survival_fraction", "mix": m}``, one per mix.

    The per-study check (``simulate.py``) validates survival bands over the mixes frozen,
    scripted and mixed only, so v2 requires exactly that metric and mix set.
    """
    require(len(spec.bands) >= 1, "paired_ni_at_stop needs at least one band")
    seen = set()
    for band in spec.bands:
        require(
            set(band) == {"metric", "mix"}
            and band["metric"] == PAIRED_BAND_METRIC
            and band["mix"] in spec.mixes,
            f"malformed paired band {band} (needs exactly metric {PAIRED_BAND_METRIC} and a mix)",
        )
        require(band["mix"] not in seen, f"two paired bands for mix {band['mix']}")
        seen.add(band["mix"])
    require(
        tuple(spec.mixes) == ("frozen", "scripted", "mixed"),
        "the paired-band check (simulate.py) only simulates the mixes frozen, scripted, mixed",
    )


def resolve_spec(reference: str) -> StudySpec:
    """Import ``package.module:ATTRIBUTE`` (the intent's ``spec_ref``)."""
    module_name, _, attribute = reference.partition(":")
    require(bool(module_name) and bool(attribute), "spec_ref must be 'module:attribute'")
    spec = getattr(importlib.import_module(module_name), attribute)
    validate_spec(spec)
    return spec


# ---------------------------------------------------------------- worlds, units and segments


def uint32_seed(domain: str, index: int) -> int:
    """``uint32_be(sha256("<domain>|worlds|<i>")[:4])`` (the dev_screen recipe)."""
    digest = hashlib.sha256(f"{domain}|worlds|{int(index)}".encode("utf-8")).digest()
    return int(struct.unpack(">I", digest[:4])[0])


def seed_bank(domain: str, count: int) -> List[int]:
    return [uint32_seed(domain, i) for i in range(int(count))]


def bank_report(
    banks: Mapping[str, Sequence[int]], excluded: Mapping[str, Sequence[int]]
) -> Dict[str, Any]:
    """Seeds unique within and across this study's banks and disjoint from ``excluded``."""
    problems = []
    seen: Dict[int, str] = {}
    for name, seeds in banks.items():
        if len(set(seeds)) != len(seeds):
            problems.append(f"duplicate seeds within {name}")
        for seed in seeds:
            if seed in seen and seen[seed] != name:
                problems.append(f"seed {seed} in both {seen[seed]} and {name}")
            seen[seed] = name
    for name, seeds in excluded.items():
        overlap = sorted(set(seen) & {int(s) for s in seeds})
        if overlap:
            problems.append(f"{len(overlap)} seeds overlap excluded namespace {name}")
    return {"passes": not problems, "problems": problems, "excluded": sorted(excluded)}


def episode_id(phase: str, arm: str, mix: str, world_index: int) -> str:
    return f"{phase}-{arm}-{mix}-w{int(world_index):05d}"


def phase_units(intent: Mapping[str, Any], phase: str) -> List[Dict[str, Any]]:
    """World-major units ``u = world_index * n_mixes + mix_position``, worker ``u % W``.

    Final: :func:`sequential_gate.round_robin_plan` over ``N_max`` worlds; a unit is both
    arms of one (mix, world).  Calibration: the same order over the dev worlds, incumbent
    only.
    """
    mixes = intent["spec"]["mixes"]
    workers = intent["caps"]["workers"]
    if phase == "final":
        units = round_robin_plan(intent["plan"]["n_max"], mixes, workers)
        seeds = intent["banks"]["final"]
        arms = list(ARMS)
    else:
        count = len(intent["banks"]["calibration"])
        units = round_robin_plan(count, mixes, workers)
        seeds = intent["banks"]["calibration"]
        arms = ["incumbent"]
    return [{**unit, "world_seed": int(seeds[unit["world_index"]]), "arms": arms} for unit in units]


def unit_episodes(phase: str, unit: Mapping[str, Any], look: int) -> List[Dict[str, Any]]:
    return [
        {
            "episode_id": episode_id(phase, arm, unit["mix"], unit["world_index"]),
            "phase": phase,
            "look": look,
            "unit": unit["unit"],
            "worker": unit["worker"],
            "arm": arm,
            "mix": unit["mix"],
            "world_index": unit["world_index"],
            "world_seed": unit["world_seed"],
        }
        for arm in unit["arms"]
    ]


def phase_looks(intent: Mapping[str, Any], phase: str) -> List[Dict[str, Any]]:
    """Per-look cumulative per-worker unit counts (calibration: one look, all units)."""
    workers = intent["caps"]["workers"]
    n_mixes = len(intent["spec"]["mixes"])
    if phase == "final":
        return worker_look_counts(intent["plan"]["look_sizes"], n_mixes, workers)
    count = len(intent["banks"]["calibration"])
    return worker_look_counts([count], n_mixes, workers)


def segment_units(
    intent: Mapping[str, Any], phase: str, look: int, shard: int
) -> List[Dict[str, Any]]:
    """Units worker ``shard`` runs for look ``look``: its shard between consecutive counts."""
    counts = phase_looks(intent, phase)
    shard_units = [u for u in phase_units(intent, phase) if u["worker"] == shard]
    start = counts[look - 1]["per_worker"][shard] if look > 0 else 0
    end = counts[look]["per_worker"][shard]
    return shard_units[start:end]


def prefix_episode_ids(intent: Mapping[str, Any], phase: str, look: int) -> List[str]:
    """Every episode of the look's unit prefix (exactly ``n_mixes x look_size`` units)."""
    total = phase_looks(intent, phase)[look]["units"]
    units = phase_units(intent, phase)[:total]
    ids = []
    for unit in units:
        ids.extend(e["episode_id"] for e in unit_episodes(phase, unit, 0))
    return ids


def segment_dir(output: Path, phase: str, look: int) -> Path:
    return Path(output) / phase / "segments" / f"look-{look}"


# ---------------------------------------------------------------- plan and source closure


def plan_parameters(
    *,
    n_max: int,
    mde: float,
    mixes: Sequence[str],
    scripted_mix: str,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    family_alpha: float = 0.05,
    ni_alpha: float = 0.05,
    futility_cp: float = DEFAULT_FUTILITY_CP,
    required_successes: int = 2,
    futility_policy: str = "followed",
    band_policy: str = "block_at_stop",
    band_margin_z: float = DEFAULT_BAND_MARGIN_Z,
    paired_band: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Keyword arguments of ``sequential_gate_plan``.  The paired band fields are present
    only under ``band_policy="paired_ni_at_stop"`` (legacy dicts are byte-identical)."""
    out = {
        "n_max": int(n_max),
        "mde": float(mde),
        "mixes": list(mixes),
        "scripted_mix": scripted_mix,
        "fractions": [float(f) for f in fractions],
        "family_alpha": float(family_alpha),
        "ni_alpha": float(ni_alpha),
        "futility_cp": float(futility_cp),
        "required_successes": int(required_successes),
        "futility_policy": futility_policy,
        "band_policy": band_policy,
        "band_margin_z": float(band_margin_z),
    }
    if band_policy == "paired_ni_at_stop":
        out.update(normalize_paired_band(paired_band))
    else:
        require(paired_band is None, f"paired band settings under band_policy {band_policy}")
    return out


def _real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_paired_band(paired_band: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The four pre-registered paired band settings, stated explicitly (no defaults)."""
    require(
        isinstance(paired_band, Mapping) and set(paired_band) == set(PAIRED_BAND_KEYS),
        f"paired_ni_at_stop needs explicit {list(PAIRED_BAND_KEYS)} "
        f"(ratified 2026-10-03: {RATIFIED_PAIRED_BAND})",
    )
    floor = paired_band["band_floor"]
    require(
        _real(paired_band["band_ni_margin"])
        and _real(paired_band["band_alpha"])
        and (floor is None or _real(floor)),
        "paired band margin, alpha and floor must be real numbers (floor may be None)",
    )
    return {
        "band_ni_margin": float(paired_band["band_ni_margin"]),
        "band_alpha": float(paired_band["band_alpha"]),
        "band_bound": str(paired_band["band_bound"]),
        "band_floor": None if floor is None else float(floor),
    }


def plan_from_parameters(parameters: Mapping[str, Any]) -> SequentialGatePlan:
    params = dict(parameters)
    n_max = params.pop("n_max")
    return sequential_gate_plan(n_max, **params)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def source_closure(roots: Sequence[str], repo: Path = REPO) -> Dict[str, Any]:
    """sha256 of every tracked or untracked-unignored runtime file under ``roots``."""
    listed = _git(
        repo, "ls-files", "--cached", "--others", "--exclude-standard", "--", *roots
    ).splitlines()
    files = {
        str((repo / item).resolve()): sha256_file(repo / item)
        for item in sorted(set(listed))
        if item.endswith(CLOSURE_SUFFIXES) and (repo / item).is_file()
    }
    dirty = _git(repo, "status", "--porcelain", "--", *roots).splitlines()
    return {
        "git_commit": _git(repo, "rev-parse", "HEAD").strip(),
        "roots": list(roots),
        "files": files,
        "digest": canonical_sha(files),
        "dirty": dirty,
    }


def closure_drift(closure: Mapping[str, Any]) -> List[str]:
    """Closure files whose bytes changed (or vanished) since ``prepare``."""
    return [
        path
        for path, expected in closure["files"].items()
        if not Path(path).is_file() or sha256_file(Path(path)) != expected
    ]


# ---------------------------------------------------------------- skew-check input


def load_skew_input(path: Path, mixes: Sequence[str]) -> Dict[str, List[float]]:
    """Saved screen/pilot paired deltas ``{mix: [delta, ...]}`` (Tier-0 data)."""
    data = read_json(path)
    require(isinstance(data, dict) and set(data) == set(mixes), f"skew input must cover {mixes}")
    out = {}
    for mix in mixes:
        values = data[mix]
        require(
            isinstance(values, list)
            and len(values) >= SKEW_MIN_DELTAS
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in values
            ),
            f"skew input mix {mix} needs >= {SKEW_MIN_DELTAS} finite deltas",
        )
        out[mix] = [float(v) for v in values]
    return out


def probe_compatible(plan: Mapping[str, Any]) -> bool:
    """``skew_probe.py`` builds its own plan with the default alphas and three mixes; the
    study's nominal levels must be the ones the probe uses or its rates mean nothing."""
    fractions = [size / plan["n_max"] for size in plan["look_sizes"]]
    probe = sequential_gate_plan(plan["n_max"], mde=30.0, fractions=fractions).as_dict()
    return all(
        probe[key] == plan[key]
        for key in ("look_sizes", "efficacy_nominal_p", "ni_nominal_p", "efficacy_alpha_per_mix")
    )


def skew_limits(plan: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "efficacy_any_look": SKEW_EFFICACY_FACTOR * plan["efficacy_alpha_per_mix"],
        "ni_any_look_scripted": SKEW_NI_LIMIT,
    }


# ---------------------------------------------------------------- paired-band check input


def paired_rule_name(margin: float, alpha: float, bound: str) -> str:
    """``simulate.py``'s rule label (``rule_name``), e.g. ``paired_M0.05_a0.05_pointwise``."""
    return f"paired_M{margin:g}_a{alpha:g}_{bound}"


def frozen_plan_document(params: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"plan_parameters", "plan"}`` exactly as ``intent.json`` will freeze them: the
    ``--plan-params`` input of ``simulate.py`` for the per-study paired-band check."""
    return {"plan_parameters": dict(params), "plan": plan_from_parameters(params).as_dict()}


def judge_paired_check(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    bands: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Re-judge a ``simulate.py --part gate`` output against the frozen plan (fail closed).

    It must have been run with ``--plan-params`` equal to this intent's plan parameters, at
    least :data:`PAIRED_CHECK_MIN_REPS` replicates, mass effects covering 0.5, ~0.67, 1, 1.5
    and 2 x MDE, and its joint rates (P(qualify and the band regressed by exactly M passes))
    must cover every pool x theta x band mix and all be <= 1.2 x band_alpha.  The runner
    applies the threshold itself; the report's own ``check.passes`` must agree.
    """
    problems: List[str] = []
    config = report.get("config") if isinstance(report, Mapping) else None
    check = report.get("check") if isinstance(report, Mapping) else None
    results = report.get("results") if isinstance(report, Mapping) else None
    if not (isinstance(config, Mapping) and isinstance(check, Mapping)):
        return {"passes": False, "problems": ["not a simulate.py --part gate check output"]}
    if not isinstance(results, Mapping):
        results = {}
    margin, alpha, bound = params["band_ni_margin"], params["band_alpha"], params["band_bound"]
    rule = paired_rule_name(margin, alpha, bound)
    threshold = PAIRED_CHECK_FACTOR * alpha
    if config.get("plan_params") != dict(params):
        problems.append("check was not run with this intent's frozen plan parameters")
    if list(config.get("check_rule") or []) != [margin, alpha, bound]:
        problems.append(f"check rule {config.get('check_rule')} is not the plan's band rule")
    if check.get("rule") != rule:
        problems.append(f"check judged rule {check.get('rule')!r}, not {rule!r}")
    reps = report.get("reps")
    if not (isinstance(reps, int) and reps >= PAIRED_CHECK_MIN_REPS):
        problems.append(f"check reps {reps!r} < {PAIRED_CHECK_MIN_REPS}")
    thetas = [float(t) for t in config.get("thetas") or []]
    mde = float(params["mde"])
    for multiple in PAIRED_CHECK_THETA_MULTIPLES:
        if not any(math.isclose(t, multiple * mde, rel_tol=1e-9) for t in thetas):
            problems.append(f"check lacks theta {multiple:g} x MDE")
    low, high = PAIRED_CHECK_TWO_THIRDS
    if not any(low * mde <= t <= high * mde for t in thetas):
        problems.append("check lacks a theta near 0.67 x MDE")
    if not _real(config.get("delta_ni")) or not config.get("delta_ni") > 0:
        problems.append("check delta_ni must be a positive number (development estimate)")
    pools = sorted({key.split("|", 1)[0] for key in results})
    joint = check.get("joint_rates") if isinstance(check.get("joint_rates"), Mapping) else {}
    expected = {
        f"{pool}|theta={theta:g}|one_mix_at_margin@{band['mix']}"
        for pool in pools
        for theta in thetas
        for band in bands
    }
    if not pools:
        problems.append("check has no pools")
    missing = sorted(expected - set(joint))
    if missing:
        problems.append(f"check lacks joint rates {missing[:5]}")
    rates = {k: joint[k] for k in sorted(joint) if k in expected}
    bad = sorted(k for k, v in rates.items() if not (_real(v) and 0.0 <= v <= threshold))
    if bad:
        problems.append(f"joint rate above {threshold:g} (or not a rate): {bad[:5]}")
    for key in rates:
        row = results.get(key, {}).get(rule, {}) if isinstance(results.get(key), Mapping) else {}
        if row.get("p_regressed_band_and_qualify") != rates[key]:
            problems.append(f"joint rate {key} differs from its results row")
    worst = max((v for v in rates.values() if _real(v)), default=None)
    passes = not problems
    if check.get("passes") is not passes:
        problems.append(f"check.passes {check.get('passes')!r} != runner verdict {passes}")
        passes = False
    return {
        "rule": rule,
        "threshold": threshold,
        "reps": reps,
        "thetas": thetas,
        "pools": pools,
        "rows_judged": len(rates),
        "max_joint_rate": worst,
        "delta_ni_used": config.get("delta_ni"),
        "problems": problems,
        "passes": passes,
    }


def paired_check_record(spec: StudySpec, params: Mapping[str, Any], repo: Path) -> Dict[str, Any]:
    """Hash-bind the per-study check output and its pool data; require that it passes."""
    path = (Path(repo) / spec.paired_band_check_path).resolve()
    require(path.is_file(), f"paired band check {path} does not exist")
    report = read_json(path)
    data_path = Path(str((report.get("config") or {}).get("data", "")))
    require(data_path.is_file(), f"paired band check pool data {data_path} does not exist")
    data_sha = sha256_file(data_path)
    require(
        report.get("data_sha256") == data_sha,
        "paired band check pool data changed since the check was run",
    )
    judged = judge_paired_check(report, params, spec.bands)
    require(judged["passes"], f"paired band check fails: {judged['problems']}")
    data = read_json(data_path)
    pools = sorted(k for k, v in data.items() if isinstance(v, dict) and "world_seeds" in v)
    require(pools == judged["pools"], f"check pools {judged['pools']} != data pools {pools}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "data_path": str(data_path.resolve()),
        "data_sha256": data_sha,
        "simulator_path": str(PAIRED_BAND_SIMULATOR),
        "acceptance": f"every joint rate <= {PAIRED_CHECK_FACTOR} x band_alpha",
        "remedy_on_fail": "band_bound rci_obf, or do not adopt paired_ni_at_stop",
        **judged,
    }


# ---------------------------------------------------------------- intent


def validate_caps(caps: Mapping[str, Any]) -> None:
    require(set(caps) == set(DEFAULT_CAPS), f"caps must cover exactly {sorted(DEFAULT_CAPS)}")
    for name, value in caps.items():
        require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 < value <= DEFAULT_CAPS[name],
            f"cap {name}={value!r} must be a positive integer <= {DEFAULT_CAPS[name]}",
        )


def preregistration_files(spec: StudySpec, repo: Path) -> Dict[str, Dict[str, str]]:
    """Protocol, OC simulation report and look-1 band-cost report (template v2 also the
    paired-band check output): path and sha256."""
    out = {}
    for name in spec.preregistration_docs():
        path = (Path(repo) / getattr(spec, name)).resolve()
        require(path.is_file(), f"{name} {path} does not exist")
        out[name.replace("_path", "")] = {"path": str(path), "sha256": sha256_file(path)}
    return out


def build_intent(
    spec: StudySpec,
    *,
    spec_ref: str,
    out_root: Path,
    n_max: int,
    mde: float,
    n_calibration: int,
    skew_input: Path,
    deadline: datetime,
    authorization_quote: str,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    family_alpha: float = 0.05,
    ni_alpha: float = 0.05,
    futility_cp: float = DEFAULT_FUTILITY_CP,
    required_successes: int = 2,
    futility_policy: str = "followed",
    futility_action: str = "stop",
    band_margin_z: float = DEFAULT_BAND_MARGIN_Z,
    paired_band: Optional[Mapping[str, Any]] = None,
    skew_reps: int = DEFAULT_SKEW_REPS,
    n_max_cap: int = N_MAX_CAP,
    caps: Optional[Mapping[str, int]] = None,
    workers: int = WORKERS,
    slot_lock_root: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
    python: Path = PYTHON,
    dry_run: bool = False,
    allow_dirty: bool = False,
    repo: Path = REPO,
) -> Dict[str, Any]:
    """Freeze everything the amendment requires before the first final world.

    A production intent (``dry_run=False``) must use the global slot lock root and a clean
    source closure; ``dry_run=True`` relaxes both and can never yield a pass.  The band policy
    comes from the spec; under ``paired_ni_at_stop`` (template v2) ``paired_band`` states the
    four band settings and the spec's paired-band check must pass for this exact plan.
    """
    validate_spec(spec)
    require(bool(authorization_quote.strip()), "authorization quote required")
    require(deadline.tzinfo is not None, "deadline needs a UTC offset")
    require(futility_action in FUTILITY_ACTIONS, f"futility_action in {FUTILITY_ACTIONS}")
    require(
        futility_action == "stop" or futility_policy == "overridable",
        "futility_action='continue' needs futility_policy='overridable'",
    )
    require(isinstance(n_calibration, int) and n_calibration >= 2, "n_calibration >= 2")
    require(isinstance(workers, int) and workers >= 1, "workers >= 1")
    require(isinstance(skew_reps, int) and skew_reps >= 1000, "skew_reps >= 1000")
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    validate_caps(caps)
    lock_root = Path(slot_lock_root or SLOT_LOCK_ROOT)
    require(
        dry_run or lock_root == SLOT_LOCK_ROOT,
        f"a production intent must use the global slot lock root {SLOT_LOCK_ROOT}",
    )
    require(dry_run or not allow_dirty, "allow_dirty is only permitted on a dry_run intent")
    require(
        not dry_run or lock_root != SLOT_LOCK_ROOT,
        "a dry run may not take the global CPU slots; give it its own slot_lock_root",
    )
    ledger = Path(ledger_path or LEDGER_PATH)
    require(
        (ledger == LEDGER_PATH) != bool(dry_run),
        "a production intent uses the global ledger; a dry run must use its own ledger path",
    )
    require_namespace_unused(ledger, spec.namespaces["final"])
    out_root = Path(out_root).resolve()
    status = run_status(out_root)
    require(status["state"] == "NOT_STARTED", f"{out_root} is {status['state']}: never reused")
    params = study_plan_parameters(
        spec,
        n_max=n_max,
        mde=mde,
        fractions=fractions,
        family_alpha=family_alpha,
        ni_alpha=ni_alpha,
        futility_cp=futility_cp,
        required_successes=required_successes,
        futility_policy=futility_policy,
        band_margin_z=band_margin_z,
        paired_band=paired_band,
    )
    plan = plan_from_parameters(params).as_dict()
    require(
        probe_compatible(plan),
        "skew_probe.py assumes family alpha 0.05 over 3 mixes and NI alpha 0.05; this plan "
        "differs, so the pre-registered resampling check cannot be run for it",
    )
    banks = {
        "calibration": seed_bank(spec.namespaces["calibration"], n_calibration),
        "final": seed_bank(spec.namespaces["final"], n_max),
    }
    excluded = {k: [int(s) for s in v] for k, v in spec.excluded_seeds().items()}
    banks_check = bank_report(banks, excluded)
    require(banks_check["passes"], f"world banks: {banks_check['problems']}")
    skew_input = Path(skew_input).resolve()
    skew_deltas = load_skew_input(skew_input, spec.mixes)
    closure = source_closure(spec.closure_roots, repo)
    require(allow_dirty or not closure["dirty"], f"source closure is dirty: {closure['dirty']}")
    counts = worker_look_counts(plan["look_sizes"], len(spec.mixes), workers)
    intent = {
        "schema_version": spec.schema,
        "template_version": spec.template_version,
        "study_id": spec.study_id,
        "authority": "tier-2-strict-promotion-sequential",
        "dry_run": bool(dry_run),
        "created_utc": now_utc().isoformat(),
        "authorization_quote": authorization_quote,
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat(),
        "spec_ref": spec_ref,
        "spec": spec.descriptor(),
        "arms": json_safe(dict(spec.arm_identities())),
        "method": SEQUENTIAL_GATE_METHOD,
        "plan_parameters": params,
        "plan": plan,
        "plan_sha256": canonical_sha(plan),
        "futility_action": futility_action,
        "interleaving": {
            "order": "world-major round robin: u = world_index * n_mixes + mix_position",
            "worker_rule": "worker u % workers runs both arms of unit u, in unit order",
            "arm_order": list(ARMS),
            "workers": workers,
            "worker_look_counts": counts,
        },
        "banks": banks,
        "banks_check": banks_check,
        "calibration_rule": {
            "phase": "incumbent only, calibration bank, every mix",
            "ni": f"{spec.ni_fraction} x calibration incumbent mean {spec.primary_metric} "
            f"of mix {spec.scripted_mix}",
            "bands": CALIBRATION_BAND_RULES[spec.band_policy],
        },
        "skew_check": {
            "input_path": str(skew_input),
            "input_sha256": sha256_file(skew_input),
            "input_counts": {mix: len(v) for mix, v in skew_deltas.items()},
            "probe_path": str(SKEW_PROBE),
            "probe_sha256": sha256_file(SKEW_PROBE),
            "probe_part": "resample",
            "reps": skew_reps,
            "limits": skew_limits(plan),
            "ni_judged_on": spec.scripted_mix,
            "remedy_on_fail": SKEW_REMEDY_ON_FAIL,
            "remedy_ladder": SKEW_REMEDY_LADDER,
            "timing": "after calibration (delta_NI frozen), before the first final world",
            "deviations_from_amendment": [
                "runs after calibration (delta_NI is a calibration output), result in "
                "skew_check.json, not in intent.json",
                "NI any-look rate judged on the scripted mix only; efficacy on every mix",
            ],
        },
        "preregistration": preregistration_files(spec, repo),
        "sizing": {"n_max": n_max, "n_max_cap": n_max_cap, "feasible": n_max <= n_max_cap},
        "caps": {
            "stage_seconds": caps,
            "handoff_seconds": HANDOFF_SECONDS,
            "workers": workers,
            "threads_per_worker": THREADS_PER_WORKER,
            "retry_authorized": False,
            "resume_authorized": False,
        },
        "source_closure": closure,
        "allow_dirty": bool(allow_dirty),
        "template_readme_sha256": sha256_file(TEMPLATE_README),
        "output_root": str(out_root),
        "repo": str(repo),
        "python": str(python),
        "slot_lock_root": str(lock_root),
        "ledger_path": str(ledger),
        "audit": {
            "path": str(INDEPENDENT_AUDIT),
            "sha256": sha256_file(INDEPENDENT_AUDIT),
            "command": [
                str(python),
                "-I",
                "-B",
                str(INDEPENDENT_AUDIT),
                "--root",
                str(out_root),
            ],
        },
    }
    if spec.paired:  # template v2 only: v1 intents keep exactly their v1 keys
        intent["paired_band_check"] = paired_check_record(spec, params, repo)
        intent["band_rule"] = paired_band_rule(params)
    return intent


def paired_band_rule(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Human-readable statement of the frozen paired band rule (also in the plan dict)."""
    return {
        "policy": "paired_ni_at_stop",
        "judged": "once, at the qualifying look, on the look-k prefix of both arms; never "
        "delays a stop (STOP_FAIL_BANDS at an interim look, FINAL_FAIL at the last)",
        "per_band": "d_i = candidate_i - incumbent_i (same final world); pass iff "
        "mean(d) - t_{n-1}(band_nominal_p[k]) * sd(d) / sqrt(n) > -band_ni_margin and "
        "(if band_floor) mean(candidate) >= band_floor",
        "all_bands_must_pass": True,
        "band_ni_margin": params["band_ni_margin"],
        "band_alpha": params["band_alpha"],
        "band_bound": params["band_bound"],
        "band_floor": params["band_floor"],
        "source": "docs/research/governance_amendment_paired_bands_2026-10-03.md",
    }


def study_plan_parameters(spec: StudySpec, **kwargs: Any) -> Dict[str, Any]:
    """:func:`plan_parameters` with the spec's mixes and band policy."""
    paired_band = kwargs.pop("paired_band", None)
    return plan_parameters(
        mixes=spec.mixes,
        scripted_mix=spec.scripted_mix,
        band_policy=spec.band_policy,
        paired_band=paired_band if spec.paired else _none_or_refuse(paired_band),
        **kwargs,
    )


def _none_or_refuse(paired_band: Any) -> None:
    require(paired_band is None, "paired_band settings need a spec with band_policy paired")
    return None


def prepare(intent: Mapping[str, Any]) -> Path:
    """Write ``intent.json`` once (create-only); a started or abandoned root is refused."""
    out_root = Path(intent["output_root"])
    status = run_status(out_root)
    require(status["state"] == "NOT_STARTED", f"{out_root} is {status['state']}: never reused")
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "intent.json"
    write_once(path, intent)
    return path


def validate_intent(intent: Mapping[str, Any], spec: StudySpec) -> None:
    """Reject a malformed, drifted or out-of-envelope intent before any child starts."""
    require(intent["template_version"] == spec.template_version, "template version")
    require(intent["spec"] == spec.descriptor(), "spec descriptor drift")
    require(
        intent["schema_version"] == spec.schema and intent["study_id"] == spec.study_id, "schema"
    )
    require(intent["method"] == SEQUENTIAL_GATE_METHOD, "method version")
    require(isinstance(intent["dry_run"], bool), "dry_run flag")
    if not intent["dry_run"]:
        require(
            Path(intent["slot_lock_root"]) == SLOT_LOCK_ROOT,
            f"a production intent must use the global slot lock root {SLOT_LOCK_ROOT}",
        )
        require(intent["allow_dirty"] is False, "allow_dirty on a production intent")
        require(Path(intent["ledger_path"]) == LEDGER_PATH, "production ledger path")
    else:
        require(Path(intent["ledger_path"]) != LEDGER_PATH, "a dry run may not use the ledger")
        require(
            Path(intent["slot_lock_root"]) != SLOT_LOCK_ROOT,
            "a dry run may not take the global CPU slots",
        )
    require(
        plan_from_parameters(intent["plan_parameters"]).as_dict() == intent["plan"],
        "frozen plan differs from its parameters",
    )
    require(canonical_sha(intent["plan"]) == intent["plan_sha256"], "plan sha256")
    workers = intent["caps"]["workers"]
    require(
        intent["interleaving"]["worker_look_counts"]
        == worker_look_counts(intent["plan"]["look_sizes"], len(spec.mixes), workers),
        "interleaving counts",
    )
    require(
        intent["banks"]["final"] == seed_bank(spec.namespaces["final"], intent["plan"]["n_max"])
        and intent["banks"]["calibration"]
        == seed_bank(spec.namespaces["calibration"], len(intent["banks"]["calibration"])),
        "world banks differ from their namespaces",
    )
    require(intent["caps"]["retry_authorized"] is False, "retry must not be authorized")
    require(intent["caps"]["resume_authorized"] is False, "resume must not be authorized")
    validate_caps(intent["caps"]["stage_seconds"])
    require(intent["caps"]["handoff_seconds"] == HANDOFF_SECONDS, "handoff reserve")
    require(intent["futility_action"] in FUTILITY_ACTIONS, "futility action")
    require(
        intent["futility_action"] == "stop" or intent["plan"]["futility_policy"] == "overridable",
        "futility override not pre-registered",
    )
    require(intent["skew_check"]["remedy_on_fail"] == SKEW_REMEDY_ON_FAIL, "skew remedy")
    require(intent["skew_check"]["limits"] == skew_limits(intent["plan"]), "skew limits")
    require(probe_compatible(intent["plan"]), "plan is not the skew probe's plan")
    require(
        sha256_file(Path(intent["skew_check"]["input_path"]))
        == intent["skew_check"]["input_sha256"],
        "skew input drift",
    )
    require(sha256_file(SKEW_PROBE) == intent["skew_check"]["probe_sha256"], "skew probe drift")
    require(
        set(intent["preregistration"])
        == {n.replace("_path", "") for n in spec.preregistration_docs()},
        "pre-registration documents",
    )
    require(
        intent["plan"]["band_policy"] == spec.band_policy
        and intent["calibration_rule"]["bands"] == CALIBRATION_BAND_RULES[spec.band_policy],
        "band policy differs between the spec and the frozen plan",
    )
    if spec.paired:
        frozen = intent["paired_band_check"]
        require(intent["band_rule"] == paired_band_rule(intent["plan_parameters"]), "band rule")
        require(
            frozen["path"] == intent["preregistration"]["paired_band_check"]["path"]
            and Path(frozen["data_path"]).is_file()
            and sha256_file(Path(frozen["data_path"])) == frozen["data_sha256"],
            "paired band check pool data drift",
        )
        again = judge_paired_check(
            read_json(Path(frozen["path"])), intent["plan_parameters"], spec.bands
        )
        require(again["passes"], f"paired band check no longer passes: {again['problems']}")
    else:
        require(
            "paired_band_check" not in intent and "band_rule" not in intent,
            "paired band fields on a block_at_stop intent",
        )
    for name, row in intent["preregistration"].items():
        require(
            Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == row["sha256"],
            f"{name} drift",
        )
    drift = closure_drift(intent["source_closure"])
    require(not drift, f"source closure drift: {drift}")
    require(json_safe(dict(spec.arm_identities())) == intent["arms"], "arm identity drift")
    parse_utc(intent["deadline_utc"])


# ---------------------------------------------------------------- guards


def on_ac_power() -> bool:
    """True when macOS reports the host drawing from AC; other platforms are not gated."""
    if sys.platform != "darwin":
        return True
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "'AC Power'" in out


def capacity_environment(threads: int = THREADS_PER_WORKER) -> Dict[str, str]:
    return {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "TORCH_NUM_THREADS": str(threads),
        "TORCH_NUM_INTEROP_THREADS": "1",
        "SNAKE_DQN_DEVICE": "cpu",
    }


def acquire_run_slots(
    lock_root: Path, workers: int, timeout: float = SLOT_TIMEOUT_SECONDS
) -> List[Any]:
    """All-or-nothing: hold every worker's CPU slot (``cpu-slot-1..W.lock``) or raise."""
    handles = [(Path(lock_root) / f"cpu-slot-{k + 1}.lock").open("a+") for k in range(workers)]
    started = time.monotonic()
    while True:
        acquired: List[Any] = []
        try:
            for handle in handles:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired.append(handle)
            return handles
        except BlockingIOError:
            for handle in acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if time.monotonic() - started > timeout:
                for handle in handles:
                    handle.close()
                raise StrictRunError(f"CPU slots unavailable for {timeout:.0f} s")
            time.sleep(0.1)


def release_run_slots(handles: Sequence[Any]) -> None:
    for handle in handles:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def assert_inherited_slot(fd: int, lock_root: Path, slot: int) -> None:
    """Worker side: ``fd`` is the parent's open ``cpu-slot-<slot>.lock`` and holds its lock."""
    target = Path(lock_root) / f"cpu-slot-{slot}.lock"
    info, expected = os.fstat(fd), target.stat()
    require(
        (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino),
        f"inherited fd {fd} is not {target.name}",
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise StrictRunError(f"CPU slot {slot} is held by another process") from exc


def _slots_held_elsewhere(lock_root: Path, workers: int) -> bool:
    """True if any existing ``cpu-slot-k.lock`` is locked by some process (never creates)."""
    for k in range(workers):
        path = Path(lock_root) / f"cpu-slot-{k + 1}.lock"
        if not path.is_file():
            continue
        with path.open("r") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_ledger(ledger: Path) -> List[Dict[str, Any]]:
    """Every started-run line of the append-only ledger (missing file = no runs)."""
    if not Path(ledger).is_file():
        return []
    return [json.loads(line) for line in Path(ledger).read_text().splitlines() if line.strip()]


def require_namespace_unused(ledger: Path, final_namespace: str) -> None:
    used = [e for e in read_ledger(ledger) if e["final_namespace"] == final_namespace]
    require(
        not used,
        f"final namespace {final_namespace} already started a run "
        f"({used[0]['output_root'] if used else ''}); a namespace is never reused",
    )


def append_ledger(ledger: Path, entry: Mapping[str, Any]) -> None:
    """Under an exclusive lock: re-check the namespace, then append one fsynced line."""
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        require_namespace_unused(ledger, entry["final_namespace"])
        os.write(descriptor, (canonical_json(dict(entry)) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def ledger_status(ledger: Path) -> List[Dict[str, Any]]:
    """Each ledger entry with its run state; started without closeout = ``ABANDONED``."""
    rows = []
    for entry in read_ledger(ledger):
        state = run_status(Path(entry["output_root"]))
        if state["state"] == "NOT_STARTED":  # in the ledger but no output: never closed
            state = {"state": "ABANDONED", "reason": "ledger entry without output"}
        rows.append({**entry, **state})
    return rows


def run_status(out_root: Path) -> Dict[str, Any]:
    """``NOT_STARTED``, ``IN_PROGRESS``, ``ABANDONED`` (started, no closeout, no live parent
    holding the slots; permanent) or ``CLOSED`` (with its outcome)."""
    output = Path(out_root) / "output"
    closeout = output / "closeout.json"
    if closeout.is_file():
        return {"state": "CLOSED", "outcome": read_json(closeout)["outcome"]}
    if not output.exists():
        return {"state": "NOT_STARTED"}
    started = output / "started.json"
    if not started.is_file():
        return {"state": "ABANDONED", "reason": "output exists without started.json"}
    marker = read_json(started)
    intent = read_json(Path(out_root) / "intent.json")
    live = _pid_alive(marker["pid"]) and _slots_held_elsewhere(
        Path(intent["slot_lock_root"]), intent["caps"]["workers"]
    )
    if live:
        return {"state": "IN_PROGRESS", "pid": marker["pid"]}
    return {"state": "ABANDONED", "reason": "parent gone without closeout.json; never resumed"}


def _write_heartbeat(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _heartbeat_mono(path: Path, started: float) -> float:
    """Last heartbeat on the monotonic clock (pauses while the host sleeps)."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8")).get("mono")
    except (OSError, ValueError, AttributeError):
        return started
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else started


# ---------------------------------------------------------------- worker


def prior_gate(intent: Mapping[str, Any], output: Path, phase: str, look: int) -> Dict[str, Any]:
    """What must exist before a segment may start; returns the sha256s it binds.

    Final look 0 needs ``calibration.json`` and a passing ``skew_check.json``; final look
    ``k > 0`` needs ``looks/look-<k-1>.json`` whose action is ``continue``.
    """
    output = Path(output)
    if phase == "calibration":
        return {}
    gate: Dict[str, Any] = {}
    calibration = output / "calibration.json"
    require(calibration.is_file(), "final segment before calibration.json")
    gate["calibration_sha256"] = sha256_file(calibration)
    if look == 0:
        skew = output / "skew_check.json"
        require(skew.is_file(), "final look 0 before skew_check.json")
        require(read_json(skew)["passes"] is True, "skew check failed: no final world may run")
        gate["skew_check_sha256"] = sha256_file(skew)
    else:
        receipt_path = output / "looks" / f"look-{look - 1}.json"
        require(receipt_path.is_file(), f"look {look} segment before look {look - 1} receipt")
        receipt = read_json(receipt_path)
        require(
            receipt["action"] == "continue",
            f"look {look - 1} receipt says {receipt['decision']}: no unit beyond it may run",
        )
        gate["prior_look_receipt_sha256"] = sha256_file(receipt_path)
    return gate


def envelope(
    intent: Mapping[str, Any],
    intent_sha: str,
    episode: Mapping[str, Any],
    record: Mapping[str, Any],
    wall_seconds: float,
) -> Dict[str, Any]:
    return {
        "schema_version": intent["schema_version"],
        "study_id": intent["study_id"],
        "intent_sha256": intent_sha,
        **{k: episode[k] for k in sorted(episode)},
        "arm_identity_sha256": canonical_sha(intent["arms"][episode["arm"]]),
        "wall_seconds": wall_seconds,
        "record": json_safe(dict(record)),
    }


def envelope_problems(
    intent: Mapping[str, Any], intent_sha: str, entry: Mapping[str, Any], episode: Mapping
) -> List[str]:
    """Shape and identity of one record envelope against its planned episode."""
    problems = []
    for key, value in episode.items():
        if entry.get(key) != value:
            problems.append(f"{episode['episode_id']}: {key} {entry.get(key)!r} != {value!r}")
    if entry.get("intent_sha256") != intent_sha:
        problems.append(f"{episode['episode_id']}: intent sha256")
    if entry.get("arm_identity_sha256") != canonical_sha(intent["arms"][episode["arm"]]):
        problems.append(f"{episode['episode_id']}: arm identity")
    record = entry.get("record")
    metrics = [intent["spec"]["primary_metric"]] + [b["metric"] for b in intent["spec"]["bands"]]
    for metric in metrics:
        value = record.get(metric) if isinstance(record, Mapping) else None
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            problems.append(f"{episode['episode_id']}: metric {metric} missing or non-finite")
    return problems


def parent_alive(parent_pid: Optional[int]) -> bool:
    """A child is re-parented when its parent dies, so ``getppid`` changes."""
    return parent_pid is None or os.getppid() == parent_pid


def worker_body(
    intent: Mapping[str, Any],
    intent_sha: str,
    spec: StudySpec,
    phase: str,
    look: int,
    shard: int,
    stage_deadline: datetime,
    parent_pid: Optional[int] = None,
) -> int:
    """Run one shard of one segment; every record is new (create-only, no adoption).

    ``parent_pid`` (subprocess workers) is checked before and after every episode; if the
    parent died, the worker exits ``PARENT_GONE_EXIT`` without writing anything more.
    """
    output = Path(intent["output_root"]) / "output"
    shard_dir = segment_dir(output, phase, look) / f"shard-{shard}"
    shard_dir.mkdir()
    gate = prior_gate(intent, output, phase, look)
    admitted = read_json(output / "admitted.json")
    rosters_path = output / "rosters.json"
    require(sha256_file(rosters_path) == admitted["rosters_sha256"], "roster drift")
    rows = {(r["mix"], int(r["world_index"])): r for r in read_json(rosters_path)[phase]}
    planned = [
        episode
        for unit in segment_units(intent, phase, look, shard)
        for episode in unit_episodes(phase, unit, look)
    ]
    if not parent_alive(parent_pid):
        return PARENT_GONE_EXIT
    write_once(
        shard_dir / "started.json",
        {
            "phase": phase,
            "look": look,
            "shard": shard,
            "gate": gate,
            "intent_sha256": intent_sha,
            "planned_episode_ids": [e["episode_id"] for e in planned],
            "utc": now_utc().isoformat(),
        },
    )
    heartbeat = shard_dir / "heartbeat.json"
    _write_heartbeat(heartbeat, {"utc": now_utc().isoformat(), "mono": time.monotonic(), "done": 0})
    context = spec.worker_setup(intent)
    records_dir = output / phase / "records"
    records_dir.mkdir(exist_ok=True)
    done: Dict[str, str] = {}
    spent: List[float] = []
    stopped = None
    for episode in planned:
        if not parent_alive(parent_pid):
            return PARENT_GONE_EXIT
        path = records_dir / f"{episode['episode_id']}.json"
        require(not path.exists(), f"record {path.name} already exists (create-only)")
        budget = max(MIN_EPISODE_BUDGET_SECONDS, 2 * sum(spent) / len(spent) if spent else 0)
        remaining = (stage_deadline - now_utc()).total_seconds()
        if remaining < budget:
            stopped = f"deadline: {remaining:.0f}s left < {budget:.0f}s budget"
            break
        row = rows[(episode["mix"], episode["world_index"])]
        require(int(row["world_seed"]) == episode["world_seed"], "roster seed mismatch")
        began = time.monotonic()
        record = spec.episode_runner(episode, row, context)
        elapsed = time.monotonic() - began
        spent.append(elapsed)
        if not parent_alive(parent_pid):
            return PARENT_GONE_EXIT
        entry = envelope(intent, intent_sha, episode, record, elapsed)
        problems = envelope_problems(intent, intent_sha, entry, episode)
        problems += list(spec.validate_record(entry, row))
        require(not problems, f"record shape: {problems}")
        done[episode["episode_id"]] = write_once(path, entry)
        _write_heartbeat(
            heartbeat, {"utc": now_utc().isoformat(), "mono": time.monotonic(), "done": len(done)}
        )
    write_once(
        shard_dir / "report.json",
        {
            "phase": phase,
            "look": look,
            "shard": shard,
            "slot": shard + 1,
            "planned_episode_ids": [e["episode_id"] for e in planned],
            "records_sha256": done,
            "complete": stopped is None and len(done) == len(planned),
            "stopped_reason": stopped,
            "wall_seconds_total": sum(spent),
            "utc": now_utc().isoformat(),
        },
    )
    return 0


def worker_main(
    intent_path: Path,
    phase: str,
    look: int,
    shard: int,
    stage_deadline: datetime,
    slot_fd: int,
) -> int:
    """Internal child entry: verify parentage, slot, intent and closure, then run the shard."""
    intent = read_json(intent_path)
    intent_sha = sha256_file(intent_path)
    output = Path(intent["output_root"]) / "output"
    marker = read_json(output / "started.json")
    parent_pid = os.getppid()
    require(marker["pid"] == parent_pid, "worker is not supervised by the run parent")
    require(marker["intent_sha256"] == intent_sha, "intent drift at worker start")
    assert_inherited_slot(slot_fd, Path(intent["slot_lock_root"]), shard + 1)
    require(not closure_drift(intent["source_closure"]), "source drift at worker start")
    spec = resolve_spec(intent["spec_ref"])
    require(intent["spec"] == spec.descriptor(), "spec drift at worker start")
    return worker_body(
        intent, intent_sha, spec, phase, look, shard, stage_deadline, parent_pid=parent_pid
    )


# ---------------------------------------------------------------- supervision (parent side)


def _group_rss(psutil: Any, pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(child: subprocess.Popen, grace: float) -> str:
    """SIGTERM the child's process group, SIGKILL after ``grace``; always reap."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
        termination = "terminated"
    except subprocess.TimeoutExpired:
        termination = "killed"
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        termination = "killed"
    child.wait()
    return termination


def supervise_children(
    commands: Sequence[Sequence[str]],
    *,
    cwd: Path,
    env: Mapping[str, str],
    logs: Sequence[Path],
    heartbeats: Sequence[Optional[Path]],
    wall_seconds: float,
    rss_limit_bytes: int = RSS_BYTES,
    available_floor_bytes: int = AVAILABLE_BYTES,
    poll_seconds: float = POLL_SECONDS,
    grace_seconds: float = TERM_GRACE_SECONDS,
    heartbeat_stale_seconds: float = HEARTBEAT_STALE_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    pass_fds: Optional[Sequence[Sequence[int]]] = None,
    power_check: Optional[Callable[[], bool]] = None,
    power_poll_seconds: float = POWER_POLL_SECONDS,
) -> Dict[str, Any]:
    """Run children concurrently, each in its own process group, under shared watchdogs.

    Causes (first wins, all children are then stopped): ``child_failed``, ``wall_timeout``,
    ``rss_limit``, ``available_memory_floor``, ``heartbeat_stale`` (monotonic clock),
    ``on_battery`` (``power_check`` polled every ``power_poll_seconds``; default
    :func:`on_ac_power`), ``watchdog_error``/``interrupted``.  ``None`` means every child
    exited on its own.  KeyboardInterrupt (SIGINT) is ``interrupted`` and is re-raised.
    """
    import psutil

    power_check = power_check or on_ac_power
    started = clock()
    last_power = started
    children: List[subprocess.Popen] = []
    streams = []
    cause = None
    peaks = [0 for _ in commands]
    min_available = None
    interrupt: Optional[BaseException] = None
    terminations: List[str] = []
    try:
        inherited = list(pass_fds) if pass_fds is not None else [() for _ in commands]
        require(len(inherited) == len(commands), "one pass_fds entry per child")
        for command, log, fds in zip(commands, logs, inherited):
            stream = Path(log).open("x", encoding="utf-8")
            streams.append(stream)
            children.append(
                subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=dict(env),
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(fds),
                )
            )
        while any(child.poll() is None for child in children):
            elapsed = clock() - started
            available = int(psutil.virtual_memory().available)
            min_available = available if min_available is None else min(min_available, available)
            for index, child in enumerate(children):
                peaks[index] = max(peaks[index], _group_rss(psutil, child.pid))
            stale = [
                beat
                for beat, child in zip(heartbeats, children)
                if beat is not None
                and child.poll() is None
                and clock() - _heartbeat_mono(Path(beat), started) > heartbeat_stale_seconds
                and elapsed > heartbeat_stale_seconds
            ]
            on_battery = False
            if clock() - last_power >= power_poll_seconds:
                last_power = clock()
                on_battery = not power_check()
            failed = any(child.poll() not in (None, 0) for child in children)
            if failed and any(child.poll() is None for child in children):
                cause = "child_failed"
            elif on_battery:
                cause = "on_battery"
            elif elapsed > wall_seconds:
                cause = "wall_timeout"
            elif max(peaks) > rss_limit_bytes:
                cause = "rss_limit"
            elif available < available_floor_bytes:
                cause = "available_memory_floor"
            elif stale:
                cause = "heartbeat_stale"
            if cause is not None:
                break
            time.sleep(poll_seconds)
    except (Interrupted, KeyboardInterrupt) as exc:
        cause, interrupt = f"interrupted: {type(exc).__name__} {exc}", exc
    except BaseException as exc:  # noqa: BLE001 - children are still stopped below
        cause = f"watchdog_error: {type(exc).__name__}: {exc}"
    finally:
        for child in children:
            if child.poll() is None or _group_alive(child.pid):
                cause = cause or "process_group_stragglers"
                terminations.append(_stop_group(child, grace_seconds))
            else:
                terminations.append("natural_exit")
        for stream in streams:
            stream.close()
    result = {
        "cause": cause,
        "elapsed_seconds": clock() - started,
        "wall_seconds": wall_seconds,
        "children": [
            {
                "command": list(command),
                "returncode": child.returncode,
                "termination": termination,
                "confirmed_exit": child.returncode is not None and not _group_alive(child.pid),
                "peak_group_rss_bytes": peak,
            }
            for command, child, termination, peak in zip(commands, children, terminations, peaks)
        ],
        "min_available_bytes": min_available,
    }
    if interrupt is not None:
        interrupt.supervision = result  # type: ignore[attr-defined]
        raise interrupt
    return result


def clean_supervision(result: Mapping[str, Any]) -> bool:
    """Every child exited naturally with code 0 and no watchdog fired."""
    return result["cause"] is None and all(
        child["termination"] == "natural_exit"
        and child["returncode"] == 0
        and child["confirmed_exit"]
        for child in result["children"]
    )


class SubprocessExecutor:
    """Production executor: one supervised ``worker`` child per shard, slot fd inherited."""

    identity = PRODUCTION_EXECUTOR

    def __init__(self, slots: Sequence[Any]):
        self.slots = list(slots)

    def preflight(self) -> None:
        import psutil

        available = int(psutil.virtual_memory().available)
        require(available >= AVAILABLE_BYTES, f"available memory {available} below floor")

    def run_segment(
        self,
        ctx: "RunContext",
        phase: str,
        look: int,
        wall_seconds: float,
        worker_deadline: datetime,
    ) -> Dict[str, Any]:
        workers = ctx.intent["caps"]["workers"]
        seg = segment_dir(ctx.output, phase, look)
        commands = [
            [
                ctx.intent["python"],
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "worker",
                "--intent",
                str(ctx.intent_path),
                "--phase",
                phase,
                "--look",
                str(look),
                "--shard",
                str(k),
                "--stage-deadline",
                worker_deadline.isoformat(),
                "--slot-fd",
                str(self.slots[k].fileno()),
            ]
            for k in range(workers)
        ]
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update(capacity_environment())
        return supervise_children(
            commands,
            cwd=Path(ctx.intent["repo"]),
            env=env,
            logs=[seg / f"worker-{k}.log" for k in range(workers)],
            heartbeats=[seg / f"shard-{k}" / "heartbeat.json" for k in range(workers)],
            wall_seconds=wall_seconds,
            pass_fds=[(self.slots[k].fileno(),) for k in range(workers)],
        )


class InProcessExecutor:
    """Dry runs and tests only: run each shard's :func:`worker_body` in this process."""

    identity = "in-process"

    def preflight(self) -> None:
        return None

    def run_segment(self, ctx, phase, look, wall_seconds, worker_deadline):
        began = time.monotonic()
        children = []
        for shard in range(ctx.intent["caps"]["workers"]):
            code = worker_body(
                ctx.intent, ctx.intent_sha, ctx.spec, phase, look, shard, worker_deadline
            )
            children.append(
                {
                    "command": ["in-process", phase, str(look), str(shard)],
                    "returncode": code,
                    "termination": "natural_exit",
                    "confirmed_exit": True,
                    "peak_group_rss_bytes": 0,
                }
            )
        return {
            "cause": None,
            "elapsed_seconds": time.monotonic() - began,
            "wall_seconds": wall_seconds,
            "children": children,
            "min_available_bytes": None,
            "executor": self.identity,
        }


def subprocess_skew_runner(
    ctx: "RunContext", mix: str, deltas_path: Path, out_path: Path, wall_seconds: float
) -> Dict[str, Any]:
    """Production skew check: ``skew_probe.py --part resample`` as a supervised child."""
    skew = ctx.intent["skew_check"]
    command = [
        ctx.intent["python"],
        "-I",
        "-B",
        skew["probe_path"],
        "--part",
        "resample",
        "--deltas",
        str(deltas_path),
        "--n-max",
        str(ctx.intent["plan"]["n_max"]),
        "--delta-ni",
        repr(float(ctx.state["delta_ni"])),
        "--fractions",
        ",".join(repr(float(f)) for f in ctx.intent["plan_parameters"]["fractions"]),
        "--reps",
        str(skew["reps"]),
        "--out",
        str(out_path),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(capacity_environment(1))
    return supervise_children(
        [command],
        cwd=Path(ctx.intent["repo"]),
        env=env,
        logs=[out_path.with_suffix(".log")],
        heartbeats=[None],
        wall_seconds=wall_seconds,
    )


def subprocess_audit_runner(ctx: "RunContext", out_dir: Path, wall_seconds: float) -> Dict:
    """Production audit: the stdlib-only ``sequential_audit.py`` as a supervised child."""
    command = [*ctx.intent["audit"]["command"], "--pre-closeout", "--out", str(out_dir)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    result = supervise_children(
        [command],
        cwd=Path(ctx.intent["repo"]),
        env=env,
        logs=[out_dir / "child.log"],
        heartbeats=[None],
        wall_seconds=wall_seconds,
    )
    child = result["children"][0] if result["children"] else {}
    return {
        "cause": result["cause"],
        "returncode": child.get("returncode"),
        "clean": result["cause"] is None and child.get("termination") == "natural_exit",
        "supervision": result,
    }


def runner_identity(function: Callable[..., Any]) -> str:
    """``subprocess_skew_runner`` / ``subprocess_audit_runner`` or ``injected:<module.name>``."""
    if function is subprocess_skew_runner:
        return PRODUCTION_SKEW_RUNNER
    if function is subprocess_audit_runner:
        return PRODUCTION_AUDIT_RUNNER
    module = getattr(function, "__module__", "?")
    return f"injected:{module}.{getattr(function, '__qualname__', repr(function))}"


# ---------------------------------------------------------------- run context and analysis


@dataclass
class RunContext:
    intent: Dict[str, Any]
    intent_path: Path
    intent_sha: str
    spec: StudySpec
    output: Path
    executor: Any
    skew_runner: Callable[..., Dict[str, Any]]
    audit_runner: Callable[..., Dict[str, Any]]
    provenance: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    prior_tree: Dict[str, str] = field(default_factory=dict)


def _tree_hashes(directory: Path) -> Dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and path.name != "heartbeat.json" and not path.name.endswith(".tmp")
    }


def stage_elapsed(ctx: RunContext, phase: str) -> float:
    """Seconds charged to ``phase`` since its first segment start marker (monotonic).

    The clock starts when the parent writes the first ``segment.json`` of the phase and
    keeps running through barriers and look analyses; the phase cap cannot be reset.
    """
    started = ctx.state.setdefault("stage_started_mono", {}).get(phase)
    return 0.0 if started is None else time.monotonic() - started


def required_seconds(intent: Mapping[str, Any], before: str, final_left: float) -> float:
    caps = intent["caps"]["stage_seconds"]
    order = ["calibration", "skew_check", "final", "audit"]
    total = 0.0
    for stage in order[order.index(before) :]:
        total += final_left if stage == "final" else caps[stage]
    return total + intent["caps"]["handoff_seconds"]


def check_continuation(ctx: RunContext, before: str, final_left: float) -> None:
    """Pre-segment guards: deadline + caps + handoff, AC power, closure, identities, tree."""
    remaining = (parse_utc(ctx.intent["deadline_utc"]) - now_utc()).total_seconds()
    if remaining < required_seconds(ctx.intent, before, final_left):
        raise DeadlineStop(f"{before}: {remaining:.0f}s left < remaining caps + handoff")
    require(on_ac_power(), f"host on battery before {before}: no further segment is admitted")
    require(not closure_drift(ctx.intent["source_closure"]), f"source drift before {before}")
    require(
        json_safe(dict(ctx.spec.arm_identities())) == ctx.intent["arms"],
        f"arm identity drift before {before}",
    )
    require(_tree_hashes(ctx.output) == ctx.prior_tree, f"output drift before {before}")


def run_segment(ctx: RunContext, phase: str, look: int) -> None:
    """Launch one segment (all shards), wait at the barrier, require a clean finish."""
    cap = ctx.intent["caps"]["stage_seconds"][phase]
    wall = cap - stage_elapsed(ctx, phase)
    if wall <= WORKER_STOP_MARGIN_SECONDS:
        raise DeadlineStop(f"{phase} stage cap exhausted")
    final_left = wall if phase == "final" else ctx.intent["caps"]["stage_seconds"]["final"]
    check_continuation(ctx, phase, final_left)
    seg = segment_dir(ctx.output, phase, look)
    seg.mkdir(parents=True)  # once per look: no attempt is ever repeated
    ctx.state["stage_started_mono"].setdefault(phase, time.monotonic())
    worker_deadline = now_utc() + timedelta(seconds=wall - WORKER_STOP_MARGIN_SECONDS)
    write_once(
        seg / "segment.json",
        {
            "phase": phase,
            "look": look,
            "wall_seconds": wall,
            "stage_cap_seconds": cap,
            "stage_elapsed_seconds_at_start": stage_elapsed(ctx, phase),
            "worker_deadline_utc": worker_deadline.isoformat(),
            "gate": prior_gate(ctx.intent, ctx.output, phase, look),
            "executor": ctx.provenance["executor"],
            "utc": now_utc().isoformat(),
        },
    )
    result = ctx.executor.run_segment(ctx, phase, look, wall, worker_deadline)
    write_once(seg / "supervision.json", result)
    if result["cause"] == "wall_timeout":
        raise DeadlineStop(f"{phase} look {look}: wall cap reached")
    require(clean_supervision(result), f"{phase} look {look} supervision: {result['cause']}")
    reports = [
        read_json(seg / f"shard-{k}" / "report.json") for k in range(ctx.intent["caps"]["workers"])
    ]
    if not all(r["complete"] for r in reports):
        stopped = [r["stopped_reason"] for r in reports if r["stopped_reason"]]
        require(
            bool(stopped) and all(s.startswith("deadline") for s in stopped),
            f"{phase} look {look} incomplete without a deadline stop",
        )
        raise DeadlineStop(f"{phase} look {look}: {stopped}")
    ctx.prior_tree = _tree_hashes(ctx.output)


def collect_prefix(ctx: RunContext, phase: str, look: int) -> Dict[str, Dict[str, Any]]:
    """Records of the look's unit prefix; fail closed unless disk == prefix exactly.

    Every segment ``0..look`` must have a complete report from every shard whose planned
    episodes are that segment's; their union must be exactly the prefix; the record files
    on disk must be exactly those (no record beyond the look, no tolerance) and have the
    reported bytes.
    """
    workers = ctx.intent["caps"]["workers"]
    listed: Dict[str, str] = {}
    for j in range(look + 1):
        seg = segment_dir(ctx.output, phase, j)
        for shard in range(workers):
            report = read_json(seg / f"shard-{shard}" / "report.json")
            planned = [
                e["episode_id"]
                for unit in segment_units(ctx.intent, phase, j, shard)
                for e in unit_episodes(phase, unit, j)
            ]
            require(report["complete"] is True, f"{phase} look {j} shard {shard} incomplete")
            require(report["planned_episode_ids"] == planned, f"{phase} look {j} shard plan")
            require(set(report["records_sha256"]) == set(planned), f"{phase} look {j} records")
            listed.update(report["records_sha256"])
    expected = prefix_episode_ids(ctx.intent, phase, look)
    require(set(listed) == set(expected), f"{phase} look {look}: reports differ from prefix")
    records_dir = ctx.output / phase / "records"
    on_disk = {p.stem for p in records_dir.glob("*.json")}
    require(on_disk == set(expected), f"{phase} look {look}: records on disk differ from prefix")
    entries = {}
    for eid in expected:
        path = records_dir / f"{eid}.json"
        require(sha256_file(path) == listed[eid], f"record bytes changed: {eid}")
        entries[eid] = read_json(path)
    return entries


def calibration_reference(
    intent: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Calibration incumbent means -> absolute delta_NI and band bounds (declarative)."""
    spec = intent["spec"]
    metrics = sorted({spec["primary_metric"], *(b["metric"] for b in spec["bands"])})
    per_mix: Dict[str, Dict[str, Any]] = {}
    for mix in spec["mixes"]:
        records = [
            e["record"]
            for e in sorted(entries.values(), key=lambda e: e["world_index"])
            if e["mix"] == mix and e["arm"] == "incumbent"
        ]
        require(len(records) >= 2, f"calibration needs >= 2 incumbent worlds for {mix}")
        per_mix[mix] = {"n": len(records)}
        for metric in metrics:
            per_mix[mix][f"mean_{metric}"] = math.fsum(r[metric] for r in records) / len(records)
    scripted_mean = per_mix[spec["scripted_mix"]][f"mean_{spec['primary_metric']}"]
    delta_ni = spec["ni_fraction"] * scripted_mean
    bands = []
    paired = intent["plan"]["band_policy"] == "paired_ni_at_stop"
    for band in spec["bands"]:
        if paired:  # no calibration reference: the means are descriptive only
            bands.append(
                {
                    **band,
                    "incumbent_calibration_mean": per_mix[band["mix"]][f"mean_{band['metric']}"],
                    "role": "descriptive only (paired_ni_at_stop has no reference bound)",
                }
            )
            continue
        reference = per_mix[band["mix"]][f"mean_{band['metric']}"]
        bands.append(
            {
                **band,
                "reference_mean": reference,
                "lower": reference + band["lower_offset"],
                "upper": reference + band["upper_offset"],
            }
        )
    return {
        "per_mix": per_mix,
        "absolute_delta_ni": delta_ni,
        "ni_rule": intent["calibration_rule"]["ni"],
        "bands": bands,
    }


def look_inputs(
    intent: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]], look: int
) -> Dict[str, Any]:
    """Per-mix paired deltas (world order) and candidate band values for the look prefix.

    Under ``paired_ni_at_stop`` the incumbent's band values of the same worlds are returned
    too (``incumbent_band_values``), after checking that each pair is one world's record pair.
    """
    spec = intent["spec"]
    size = intent["plan"]["look_sizes"][look]
    paired = intent["plan"]["band_policy"] == "paired_ni_at_stop"
    deltas: Dict[str, List[float]] = {}
    band_values: Dict[str, Dict[str, List[float]]] = {}
    incumbent_values: Dict[str, Dict[str, List[float]]] = {}
    for mix in spec["mixes"]:
        deltas[mix] = []
        band_values[mix] = {}
        incumbent_values[mix] = {}
        for world in range(size):
            inc_entry = entries[episode_id("final", "incumbent", mix, world)]
            cand_entry = entries[episode_id("final", "candidate", mix, world)]
            inc, cand = inc_entry["record"], cand_entry["record"]
            deltas[mix].append(cand[spec["primary_metric"]] - inc[spec["primary_metric"]])
            if paired:
                require(
                    all(
                        inc_entry[k] == cand_entry[k]
                        for k in ("mix", "world_index", "world_seed", "unit")
                    )
                    and (inc_entry["arm"], cand_entry["arm"]) == ARMS,
                    f"paired band: {mix} world {world} is not one world's record pair",
                )
            for band in spec["bands"]:
                if band["mix"] == mix:
                    band_values[mix].setdefault(band["metric"], []).append(cand[band["metric"]])
                    if paired:
                        incumbent_values[mix].setdefault(band["metric"], []).append(
                            inc[band["metric"]]
                        )
    out: Dict[str, Any] = {"deltas_by_mix": deltas, "band_values": band_values}
    if paired:
        out["incumbent_band_values"] = incumbent_values
    return out


def look_analysis(
    intent: Mapping[str, Any],
    calibration: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    look: int,
) -> Dict[str, Any]:
    """The pre-registered decision at ``look`` (bands checked per look 0..look)."""
    plan = plan_from_parameters(intent["plan_parameters"])
    inputs = look_inputs(intent, entries, look)
    bands_by_look = []
    for j in range(look + 1):
        size = plan.look_sizes[j]
        checks = []
        if plan.band_policy == "paired_ni_at_stop":
            bands_by_look.append(paired_band_checks(plan, j, intent["spec"]["bands"], inputs))
            continue
        for band in calibration["bands"]:
            values = inputs["band_values"][band["mix"]][band["metric"]][:size]
            check = band_check(
                values,
                band["lower"],
                band["upper"],
                n_final=plan.n_max,
                margin_z=plan.band_margin_z,
            )
            checks.append({"metric": band["metric"], "mix": band["mix"], **check})
        bands_by_look.append(checks)
    bands_pass = [all(c["passes"] for c in checks) for checks in bands_by_look]
    decision = sequential_decision(
        plan,
        look,
        inputs["deltas_by_mix"],
        calibration["absolute_delta_ni"],
        bands_pass,
    )
    decision.pop("plan", None)
    return {
        "deltas_digest": canonical_sha(inputs["deltas_by_mix"]),
        "bands_by_look": bands_by_look,
        "bands_pass_by_look": bands_pass,
        "sequential_decision": json_safe(decision),
    }


def paired_band_checks(
    plan: SequentialGatePlan,
    look: int,
    bands: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Every paired band at ``look`` on the look's prefix of both arms (same world order)."""
    size = plan.look_sizes[look]
    checks = []
    for band in bands:
        candidate = inputs["band_values"][band["mix"]][band["metric"]][:size]
        incumbent = inputs["incumbent_band_values"][band["mix"]][band["metric"]][:size]
        check = plan_paired_band_check(plan, look, candidate, incumbent)
        checks.append(
            {
                "metric": band["metric"],
                "mix": band["mix"],
                "policy": "paired_ni_at_stop",
                "df": check["n"] - 1,
                "pairs_digest": canonical_sha({"candidate": candidate, "incumbent": incumbent}),
                **check,
            }
        )
    return checks


def look_action(intent: Mapping[str, Any], decision: str) -> str:
    if decision in TERMINAL_DECISIONS:
        return "stop"
    if decision == "STOP_FUTILE":
        return "continue" if intent["futility_action"] == "continue" else "stop"
    require(decision == "CONTINUE", f"unknown decision {decision}")
    return "continue"


def build_receipt(ctx: RunContext, look: int, entries: Mapping[str, Any]) -> Dict[str, Any]:
    analysis = look_analysis(ctx.intent, ctx.state["calibration"], entries, look)
    decision = analysis["sequential_decision"]
    workers = ctx.intent["caps"]["workers"]
    reports = {}
    for j in range(look + 1):
        for shard in range(workers):
            path = segment_dir(ctx.output, "final", j) / f"shard-{shard}" / "report.json"
            reports[str(path.relative_to(ctx.output))] = sha256_file(path)
    prior = ctx.output / "looks" / f"look-{look - 1}.json"
    counts = ctx.intent["interleaving"]["worker_look_counts"][look]
    records = {
        eid: sha256_file(ctx.output / "final" / "records" / f"{eid}.json")
        for eid in prefix_episode_ids(ctx.intent, "final", look)
    }
    return {
        "schema_version": ctx.intent["schema_version"],
        "study_id": ctx.intent["study_id"],
        "method": SEQUENTIAL_GATE_METHOD,
        "look": look,
        "final_look": look == len(ctx.intent["plan"]["look_sizes"]) - 1,
        "n_per_mix": ctx.intent["plan"]["look_sizes"][look],
        "units": counts["units"],
        "per_worker": counts["per_worker"],
        "intent_sha256": ctx.intent_sha,
        "plan_sha256": ctx.intent["plan_sha256"],
        "calibration_sha256": sha256_file(ctx.output / "calibration.json"),
        "skew_check_sha256": sha256_file(ctx.output / "skew_check.json"),
        "prior_look_receipt_sha256": sha256_file(prior) if look > 0 else None,
        "segment_reports_sha256": reports,
        "records_sha256": records,
        "deltas_digest": analysis["deltas_digest"],
        **paired_receipt_fields(ctx.intent, analysis, decision),
        "bands_by_look": analysis["bands_by_look"],
        "bands_pass_by_look": analysis["bands_pass_by_look"],
        "decision": decision["decision"],
        "valid": decision["valid"],
        "passes": decision["passes"],
        "action": look_action(ctx.intent, decision["decision"]),
        "futility_action": ctx.intent["futility_action"],
        "sequential_decision": decision,
    }


def paired_receipt_fields(
    intent: Mapping[str, Any], analysis: Mapping[str, Any], decision: Mapping[str, Any]
) -> Dict[str, Any]:
    """Template v2 receipt fields (none under block_at_stop, so v1 receipts are unchanged).

    ``band_judged_look`` is the qualifying look whose band verdict decides the run (``None``
    if the run has not qualified): its per-mix results are copied to ``band_results``.
    """
    if intent["plan"]["band_policy"] != "paired_ni_at_stop":
        return {}
    judged = [h["look"] for h in decision["history"] if h["bands_judged"]]
    look = judged[0] if judged else None
    return {
        "band_policy": "paired_ni_at_stop",
        "band_judged_look": look,
        "band_results": analysis["bands_by_look"][look] if look is not None else None,
    }


# ---------------------------------------------------------------- phases


def run_calibration(ctx: RunContext) -> None:
    run_segment(ctx, "calibration", 0)
    entries = collect_prefix(ctx, "calibration", 0)
    reference = json_safe(calibration_reference(ctx.intent, entries))
    require(reference["absolute_delta_ni"] > 0, "non-positive delta_NI")
    write_once(ctx.output / "calibration.json", reference)
    ctx.prior_tree = _tree_hashes(ctx.output)
    ctx.state["calibration"] = reference
    ctx.state["delta_ni"] = reference["absolute_delta_ni"]


def judge_skew(intent: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the amendment thresholds to the per-mix probe outputs."""
    limits = intent["skew_check"]["limits"]
    scripted = intent["spec"]["scripted_mix"]
    per_mix = {}
    for mix in intent["spec"]["mixes"]:
        result = outputs[mix]["result"]
        efficacy = float(result["efficacy_null"]["any_look"])
        ni = float(result["ni_null"]["any_look"])
        problems = []
        if result["look_sizes"] != intent["plan"]["look_sizes"]:
            problems.append("probe look sizes differ from the plan")
        if result["n_saved"] != intent["skew_check"]["input_counts"][mix]:
            problems.append("probe read a different number of saved deltas")
        if result["efficacy_null"]["reps"] != intent["skew_check"]["reps"]:
            problems.append("probe reps differ from the intent")
        efficacy_ok = efficacy <= limits["efficacy_any_look"]
        ni_ok = mix != scripted or ni <= limits["ni_any_look_scripted"]
        per_mix[mix] = {
            "efficacy_any_look": efficacy,
            "ni_any_look": ni,
            "ni_judged": mix == scripted,
            "sample_skewness": result.get("sample_skewness"),
            "probe_passes_field": result.get("passes"),
            "problems": problems,
            "passes": bool(efficacy_ok and ni_ok and not problems),
        }
    return {"per_mix": per_mix, "passes": all(row["passes"] for row in per_mix.values())}


def run_skew_check(ctx: RunContext) -> bool:
    """The amendment's resampling check, after calibration, before the first final world."""
    cap = ctx.intent["caps"]["stage_seconds"]["skew_check"]
    check_continuation(ctx, "skew_check", ctx.intent["caps"]["stage_seconds"]["final"])
    work = ctx.output / "skew_check"
    work.mkdir()
    saved = load_skew_input(Path(ctx.intent["skew_check"]["input_path"]), ctx.spec.mixes)
    require(
        sha256_file(Path(ctx.intent["skew_check"]["input_path"]))
        == ctx.intent["skew_check"]["input_sha256"],
        "skew input drift",
    )
    began = time.monotonic()
    outputs, files = {}, {}
    for mix in ctx.spec.mixes:
        deltas_path = work / f"deltas-{mix}.json"
        write_once(deltas_path, saved[mix])
        out_path = work / f"resample-{mix}.json"
        left = cap - (time.monotonic() - began)
        if left <= 0:
            raise DeadlineStop("skew check cap reached")
        result = ctx.skew_runner(ctx, mix, deltas_path, out_path, left)
        write_once(work / f"supervision-{mix}.json", result)
        if result["cause"] == "wall_timeout":
            raise DeadlineStop("skew check cap reached")
        require(clean_supervision(result), f"skew probe {mix}: {result['cause']}")
        outputs[mix] = read_json(out_path)
        files[mix] = {
            "deltas_path": str(deltas_path),
            "deltas_sha256": sha256_file(deltas_path),
            "output_path": str(out_path),
            "output_sha256": sha256_file(out_path),
        }
    judged = judge_skew(ctx.intent, outputs)
    receipt = {
        "schema_version": ctx.intent["schema_version"],
        "intent_sha256": ctx.intent_sha,
        "calibration_sha256": sha256_file(ctx.output / "calibration.json"),
        "delta_ni": ctx.state["delta_ni"],
        "n_max": ctx.intent["plan"]["n_max"],
        "fractions": ctx.intent["plan_parameters"]["fractions"],
        "reps": ctx.intent["skew_check"]["reps"],
        "limits": ctx.intent["skew_check"]["limits"],
        "skew_runner": ctx.provenance["skew_runner"],
        "files": files,
        **judged,
        "remedy_on_fail": SKEW_REMEDY_ON_FAIL,
        "remedy_ladder": SKEW_REMEDY_LADDER,
        "utc": now_utc().isoformat(),
    }
    write_once(ctx.output / "skew_check.json", receipt)
    ctx.prior_tree = _tree_hashes(ctx.output)
    return bool(judged["passes"])


def run_final(ctx: RunContext) -> Dict[str, Any]:
    """The look loop; returns the stopping look's receipt."""
    looks_dir = ctx.output / "looks"
    looks_dir.mkdir()
    receipt: Dict[str, Any] = {}
    for look in range(len(ctx.intent["plan"]["look_sizes"])):
        run_segment(ctx, "final", look)
        entries = collect_prefix(ctx, "final", look)
        receipt = json_safe(build_receipt(ctx, look, entries))
        write_once(looks_dir / f"look-{look}.json", receipt)
        ctx.prior_tree = _tree_hashes(ctx.output)
        if receipt["action"] == "stop":
            break
    return receipt


def producer_claim(intent: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = "DRY_RUN_" if intent["dry_run"] else ""
    if state.get("skew_failed"):
        return {"outcome": f"{prefix}SKEW_CHECK_FAILED"}
    receipt = state.get("stop_receipt")
    if not receipt or receipt["action"] != "stop" or not receipt["valid"]:
        return {"outcome": "INVALID_STOP", "stop_reason": "no valid stopping decision"}
    verdict = "PASS" if receipt["passes"] else "FAIL"
    return {
        "outcome": f"DRY_RUN_{verdict}" if intent["dry_run"] else f"STRICT_{verdict}",
        "stop_look": receipt["look"],
        "decision": receipt["decision"],
    }


def run_audit(ctx: RunContext) -> None:
    """Producer claim, then the independent audit child (its report gates the outcome)."""
    caps = ctx.intent["caps"]["stage_seconds"]
    check_continuation(ctx, "audit", 0.0)
    claim = producer_claim(ctx.intent, ctx.state)
    write_once(
        ctx.output / "producer-outcome.json",
        {**claim, "role": "producer claim before the independent audit"},
    )
    ctx.state["producer_claim"] = claim
    out_dir = ctx.output / "audit"
    out_dir.mkdir()
    result = ctx.audit_runner(ctx, out_dir, caps["audit"])
    write_once(
        out_dir / "runner.json",
        json_safe({**result, "identity": runner_identity(ctx.audit_runner)}),
    )
    if result["cause"] == "wall_timeout":
        raise DeadlineStop("independent audit: wall cap reached")
    report_path = out_dir / "audit.json"
    require(
        result["clean"] and result["returncode"] in (0, 1) and report_path.is_file(),
        f"independent audit unusable: {result['cause']}",
    )
    report = read_json(report_path)
    ctx.state["audit_report_sha256"] = sha256_file(report_path)
    ctx.state["audit_passed"] = bool(result["returncode"] == 0 and report["status"] == "PASS")
    ctx.state["audit_failures"] = [row["rule"] for row in report.get("failures", [])]
    recomputed = report.get("recomputed", {})
    ctx.state["decisions_agree"] = bool(
        recomputed.get("expected_outcome") == claim["outcome"]
        and recomputed.get("stop_look") == claim.get("stop_look")
        and recomputed.get("decision") == claim.get("decision")
    )


def classify_outcome(intent: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    """Map run facts to one pre-registered outcome; a failure is never relabelled."""
    if not intent["sizing"]["feasible"]:
        return "STOP_INFEASIBLE"
    if state.get("failure") is not None:
        return "INVALID_STOP"
    if state.get("deadline_stop"):
        return "INCOMPLETE"
    if state.get("audit_passed") is not True or state.get("decisions_agree") is not True:
        return "INVALID_STOP"
    outcome = state["producer_claim"]["outcome"]
    if intent["dry_run"]:
        require(outcome.startswith("DRY_RUN_") or outcome == "INVALID_STOP", "dry run outcome")
    return outcome


def closeout(ctx: RunContext) -> Dict[str, Any]:
    """Write ``closeout.json`` (and ``receipt.json`` on STRICT_PASS only) exactly once."""
    state = ctx.state
    outcome = classify_outcome(ctx.intent, state)
    remaining = (parse_utc(ctx.intent["deadline_utc"]) - now_utc()).total_seconds()
    receipt = state.get("stop_receipt")
    record = {
        "schema_version": ctx.intent["schema_version"],
        "study_id": ctx.intent["study_id"],
        "method": SEQUENTIAL_GATE_METHOD,
        "intent_sha256": ctx.intent_sha,
        "plan_sha256": ctx.intent["plan_sha256"],
        "outcome": outcome,
        "dry_run": ctx.intent["dry_run"],
        "provenance": ctx.provenance,
        "precedence": "closeout.json takes precedence over every stage record",
        "failure": state.get("failure"),
        "deadline_stop": state.get("deadline_stop"),
        "skew_check_passed": state.get("skew_passed"),
        "stop_look": receipt["look"] if receipt else None,
        "stop_decision": receipt["decision"] if receipt else None,
        "worlds_per_mix_used": receipt["n_per_mix"] if receipt else 0,
        "n_max": ctx.intent["plan"]["n_max"],
        "producer_claim": state.get("producer_claim"),
        "audit_passed": state.get("audit_passed"),
        "audit_report_sha256": state.get("audit_report_sha256"),
        "audit_failures": state.get("audit_failures"),
        "decisions_agree": state.get("decisions_agree"),
        "remaining_seconds_at_closeout": remaining,
        "handoff_reserve_met": remaining >= ctx.intent["caps"]["handoff_seconds"],
        "naive_means_are_descriptive_only": True,
        "promotion_performed": False,
        "retry_authorized": False,
        "resume_authorized": False,
        "utc": now_utc().isoformat(),
    }
    if outcome == "STRICT_PASS":
        require(ctx.intent["dry_run"] is False, "a dry run never writes receipt.json")
        record["receipt_sha256"] = write_once(
            ctx.output / "receipt.json",
            {
                "schema_version": ctx.intent["schema_version"],
                "receipt": "strict-pass-receipt-only",
                "method": SEQUENTIAL_GATE_METHOD,
                "arms": ctx.intent["arms"],
                "intent_sha256": ctx.intent_sha,
                "plan_sha256": ctx.intent["plan_sha256"],
                "audit_report_sha256": state["audit_report_sha256"],
                "stop_look": receipt["look"],
                "decision": receipt["decision"],
                "look_receipt_sha256": sha256_file(
                    ctx.output / "looks" / f"look-{receipt['look']}.json"
                ),
            },
        )
    write_once(ctx.output / "closeout.json", json_safe(record))
    return record


def run_phases(ctx: RunContext) -> None:
    run_calibration(ctx)
    ctx.state["skew_passed"] = run_skew_check(ctx)
    if not ctx.state["skew_passed"]:
        ctx.state["skew_failed"] = True
    else:
        receipt = run_final(ctx)
        ctx.state["stop_receipt"] = receipt
        write_once(
            ctx.output / "decision.json",
            {
                "stop_look": receipt["look"],
                "decision": receipt["decision"],
                "valid": receipt["valid"],
                "passes": receipt["passes"],
                "action": receipt["action"],
                "look_receipt_sha256": sha256_file(
                    ctx.output / "looks" / f"look-{receipt['look']}.json"
                ),
            },
        )
        ctx.prior_tree = _tree_hashes(ctx.output)
    run_audit(ctx)


_AUDIT_MODULE: Dict[str, Any] = {}


def in_process_audit_runner(ctx: RunContext, out_dir: Path, wall_seconds: float) -> Dict:
    """Dry runs and tests only: call the audit's ``main`` in this process."""
    module = _AUDIT_MODULE.get("module")
    if module is None:
        spec = importlib.util.spec_from_file_location("sequential_audit_inproc", INDEPENDENT_AUDIT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _AUDIT_MODULE["module"] = module
    code = module.main(
        ["--root", str(ctx.intent["output_root"]), "--pre-closeout", "--out", str(out_dir)]
    )
    return {"cause": None, "returncode": code, "clean": True}


def run(
    intent_path: Path,
    *,
    spec: Optional[StudySpec] = None,
    executor: Any = None,
    skew_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    audit_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Admit the intent once and close out with exactly one outcome.  Never resumes.

    Injected executors/runners are refused unless the intent is a ``dry_run``.
    """
    intent_path = Path(intent_path).resolve()
    intent = read_json(intent_path)
    spec = spec if spec is not None else resolve_spec(intent["spec_ref"])
    validate_intent(intent, spec)
    require(intent_path == Path(intent["output_root"]) / "intent.json", "intent location")
    output = Path(intent["output_root"]) / "output"
    status = run_status(Path(intent["output_root"]))
    require(
        status["state"] == "NOT_STARTED",
        f"{output} is {status['state']}: create-only, never resumed or retried",
    )
    skew_runner = skew_runner or subprocess_skew_runner
    audit_runner = audit_runner or subprocess_audit_runner
    if intent["dry_run"]:
        require(
            isinstance(executor, InProcessExecutor),
            "a dry run must use the in-process executor (never real workers or slots)",
        )
    else:
        require(
            executor is None or type(executor) is SubprocessExecutor,
            "a production run uses SubprocessExecutor only",
        )
        require_namespace_unused(Path(intent["ledger_path"]), intent["spec"]["namespaces"]["final"])
    provenance = {
        "executor": PRODUCTION_EXECUTOR if executor is None else executor.identity,
        "skew_runner": runner_identity(skew_runner),
        "audit_runner": runner_identity(audit_runner),
        "allow_dirty": intent["allow_dirty"],
        "dry_run": intent["dry_run"],
        "slot_lock_root": intent["slot_lock_root"],
    }
    production = (
        provenance["executor"] == PRODUCTION_EXECUTOR
        and provenance["skew_runner"] == PRODUCTION_SKEW_RUNNER
        and provenance["audit_runner"] == PRODUCTION_AUDIT_RUNNER
    )
    require(intent["dry_run"] or production, f"non-production run needs dry_run: {provenance}")
    if intent["sizing"]["feasible"]:
        require(on_ac_power(), "host is on battery; plug in before a Tier-2 run")
    slots: List[Any] = []
    try:
        if intent["sizing"]["feasible"]:
            slots = acquire_run_slots(Path(intent["slot_lock_root"]), intent["caps"]["workers"])
        executor = executor if executor is not None else SubprocessExecutor(slots)
        if intent["sizing"]["feasible"]:
            executor.preflight()
        append_ledger(
            Path(intent["ledger_path"]),
            {
                "study_id": intent["study_id"],
                "final_namespace": intent["spec"]["namespaces"]["final"],
                "intent_sha256": sha256_file(intent_path),
                "output_root": intent["output_root"],
                "dry_run": intent["dry_run"],
                "pid": os.getpid(),
                "utc": now_utc().isoformat(),
            },
        )
        ctx = RunContext(
            intent=intent,
            intent_path=intent_path,
            intent_sha=sha256_file(intent_path),
            spec=spec,
            output=output,
            executor=executor,
            skew_runner=skew_runner,
            audit_runner=audit_runner,
            provenance=provenance,
        )
        return _admitted_run(ctx, slots)
    finally:
        release_run_slots(slots)


def _admitted_run(ctx: RunContext, slots: Sequence[Any]) -> Dict[str, Any]:
    """From ``started.json`` on, every stop the parent sees is final (closeout is written)."""
    ctx.output.mkdir(parents=False)
    write_once(
        ctx.output / "started.json",
        {
            "utc": now_utc().isoformat(),
            "pid": os.getpid(),
            "intent_sha256": ctx.intent_sha,
            "cpu_slot_locks_held": [str(handle.name) for handle in slots],
            "provenance": ctx.provenance,
        },
    )
    ctx.state["stage_started_mono"] = {}

    def interrupt(signum: int, frame: Any) -> None:
        raise Interrupted(signal.Signals(signum).name)

    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {sig: signal.signal(sig, interrupt) for sig in signals}
    try:
        if ctx.intent["sizing"]["feasible"]:
            admit_rosters(ctx)
            run_phases(ctx)
    except DeadlineStop as exc:
        ctx.state["deadline_stop"] = str(exc)
    except (Exception, Interrupted, KeyboardInterrupt) as exc:  # noqa: BLE001 - recorded
        ctx.state["failure"] = f"{type(exc).__name__}: {exc}"
    finally:  # default handlers are back before the closeout is written
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return closeout(ctx)


def admit_rosters(ctx: RunContext) -> None:
    """Freeze ``rosters.json`` and ``admitted.json`` once."""
    rosters = {}
    for phase in PHASES:
        rows = []
        for unit in phase_units(ctx.intent, phase):
            row = ctx.spec.build_row(phase, unit["mix"], unit["world_index"], unit["world_seed"])
            require(
                row["mix"] == unit["mix"]
                and int(row["world_index"]) == unit["world_index"]
                and int(row["world_seed"]) == unit["world_seed"],
                "build_row must keep mix, world_index and world_seed",
            )
            rows.append(json_safe(row))
        rosters[phase] = rows
    sha = write_once(ctx.output / "rosters.json", rosters)
    write_once(
        ctx.output / "admitted.json",
        {"utc": now_utc().isoformat(), "rosters_sha256": sha, "intent_sha256": ctx.intent_sha},
    )
    ctx.prior_tree = _tree_hashes(ctx.output)


# ---------------------------------------------------------------- CLI


PAIRED_BAND_HELP = (
    "template v2 only: band_ni_margin,band_alpha,band_bound,band_floor "
    "(floor 'none' for no floor), e.g. 0.05,0.05,pointwise,0.30 (ratified 2026-10-03)"
)


def parse_paired_band(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """``M,alpha,bound,floor`` -> the four paired band settings (``None`` if not given)."""
    if text is None:
        return None
    parts = [part.strip() for part in text.split(",")]
    require(len(parts) == 4, f"--paired-band needs M,alpha,bound,floor; got {text!r}")
    return {
        "band_ni_margin": float(parts[0]),
        "band_alpha": float(parts[1]),
        "band_bound": parts[2],
        "band_floor": None if parts[3].lower() == "none" else float(parts[3]),
    }


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="write the intent once")
    prep.add_argument("--spec", required=True, help="module:ATTRIBUTE of a StudySpec")
    prep.add_argument("--out-root", type=Path, required=True)
    prep.add_argument("--n-max", type=int, required=True)
    prep.add_argument("--mde", type=float, required=True)
    prep.add_argument("--n-calibration", type=int, required=True)
    prep.add_argument("--skew-input", type=Path, required=True)
    prep.add_argument("--deadline-utc", required=True)
    prep.add_argument("--authorization-quote", required=True)
    prep.add_argument("--fractions", default=",".join(str(f) for f in DEFAULT_FRACTIONS))
    prep.add_argument("--futility-policy", default="followed")
    prep.add_argument("--futility-action", default="stop", choices=FUTILITY_ACTIONS)
    prep.add_argument("--band-margin-z", type=float, default=DEFAULT_BAND_MARGIN_Z)
    prep.add_argument("--paired-band", default=None, help=PAIRED_BAND_HELP)
    prep.add_argument("--skew-reps", type=int, default=DEFAULT_SKEW_REPS)
    frozen = sub.add_parser(
        "plan",
        help="write the frozen {plan_parameters, plan} (simulate.py --plan-params input)",
    )
    frozen.add_argument("--spec", required=True, help="module:ATTRIBUTE of a StudySpec")
    frozen.add_argument("--out", type=Path, required=True)
    frozen.add_argument("--n-max", type=int, required=True)
    frozen.add_argument("--mde", type=float, required=True)
    frozen.add_argument("--fractions", default=",".join(str(f) for f in DEFAULT_FRACTIONS))
    frozen.add_argument("--futility-policy", default="followed")
    frozen.add_argument("--band-margin-z", type=float, default=DEFAULT_BAND_MARGIN_Z)
    frozen.add_argument("--paired-band", default=None, help=PAIRED_BAND_HELP)
    child = sub.add_parser("run")
    child.add_argument("--intent", type=Path, required=True)
    status = sub.add_parser("status", help="one run's state and every ledger entry's state")
    status.add_argument("--intent", type=Path, default=None)
    status.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    worker = sub.add_parser("worker")
    worker.add_argument("--intent", type=Path, required=True)
    worker.add_argument("--phase", choices=PHASES, required=True)
    worker.add_argument("--look", type=int, required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--stage-deadline", required=True)
    worker.add_argument("--slot-fd", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        spec = resolve_spec(args.spec)
        intent = build_intent(
            spec,
            spec_ref=args.spec,
            out_root=args.out_root,
            n_max=args.n_max,
            mde=args.mde,
            n_calibration=args.n_calibration,
            skew_input=args.skew_input,
            deadline=parse_utc(args.deadline_utc),
            authorization_quote=args.authorization_quote,
            fractions=[float(f) for f in args.fractions.split(",")],
            futility_policy=args.futility_policy,
            futility_action=args.futility_action,
            band_margin_z=args.band_margin_z,
            paired_band=parse_paired_band(args.paired_band),
            skew_reps=args.skew_reps,
        )
        path = prepare(intent)
        print(json.dumps({"intent": str(path), "sha256": sha256_file(path)}))
        return 0
    if args.command == "plan":
        spec = resolve_spec(args.spec)
        params = study_plan_parameters(
            spec,
            n_max=args.n_max,
            mde=args.mde,
            fractions=[float(f) for f in args.fractions.split(",")],
            futility_policy=args.futility_policy,
            band_margin_z=args.band_margin_z,
            paired_band=parse_paired_band(args.paired_band),
        )
        sha = write_once(args.out, frozen_plan_document(params))
        print(json.dumps({"plan_document": str(args.out), "sha256": sha}))
        return 0
    if args.command == "status":
        report: Dict[str, Any] = {"ledger": ledger_status(args.ledger)}
        if args.intent is not None:
            report["run"] = run_status(args.intent.resolve().parent)
        print(json.dumps(report, indent=1))
        return 0
    if args.command == "run":
        record = run(args.intent)
        print(json.dumps({"outcome": record["outcome"]}))
        return 0
    return worker_main(
        args.intent.resolve(),
        args.phase,
        args.look,
        args.shard,
        parse_utc(args.stage_deadline),
        args.slot_fd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
