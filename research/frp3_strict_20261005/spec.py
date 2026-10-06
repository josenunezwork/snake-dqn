"""StudySpec: FRP-v3 seed-12 M3@60000 + v8 vs the released champion + v8 (champion change).

Instantiation of ``research/sequential_strict_template`` (method
``strict-sequential-obf-bonferroni-v1``), template **v2-pooled: survival band v2**
(``pooled_ni_continue``: pooled paired NI M 0.05 + per-mix catastrophic NI 0.10, alpha 0.05
``rci_obf``, floor 0.30, judged at the qualifying look and every later look; governance
amendment survival band v2, 2026-10-06, option 1, **ratification pending**: a production
``prepare`` refuses until it is ratified with this option). It replaces the template-v2
``paired_ni_at_stop`` band this package first froze (2026-10-05; no intent was ever written),
run through template **v3** (``prepare --remote-config``: RunPod serverless under
``docs/research/governance_amendment_strict_on_runpod_2026-10-05.md``, Mac 2-slot fallback on an
identity-check mismatch). Pre-registration: ``protocol.md`` beside this file; the sizing, the
operating-characteristics report, the look-1 band cost, the skew input and the paired-band pool
come from ``preregistration.py`` (FRP-v3 Phase R, the candidate's own paired worlds); the band
check is survival band v2's ``simulate.py --part study`` on that pool with the frozen plan (a
validity check of the rule's guarantee; the rule and its margins were chosen on independent
pre-FRP-v3 data, not on this pool).

Arms (hero only; opponents never carry a veto). Both arms run
``tournament_eval.rollout(hero_safety_veto=True)`` with the install routed through
``dev_screen.hero_veto_installer(install_space_and_head_veto(hero, 8.0))``; they differ ONLY in
the hero checkpoint:

* incumbent: ``champion_a5_freespace_20260621.pth`` + ``free-space-veto/v8-space-and-head
  (lambda=8.0)``, the released Watch hero, bound fail closed to the v8 STRICT_PASS receipt
  (receipt bytes, method, descriptor, checkpoint and the seven veto source sha256s it records);
* candidate: FRP-v3 arm M3, training seed 12, 60k updates (``apex_mark_u60000.pth``, sha256
  ``eec144bf...``) + the same v8 (same descriptor, same source bytes), bound to the FRP-v3
  Phase R summary that pre-declared it (GO_R) and to the LH-1 screen receipt of this exact
  candidate, which must exist and read CLEAR: :func:`lh1_binding` is part of
  :func:`arm_identities`, so ``prepare`` refuses without it and the template re-checks it
  (bytes included) before every segment.

Nothing here changes a champion, default, config, released veto or deployment. A STRICT_PASS
is a receipt; a release is a separate owner action after Mac serving qualification (protocol).
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from research.apex_safety_20260926 import dev_screen
from research.frp3_strict_20261005 import preflight
from research.sequential_strict_template.sequential_runner import (
    DEFAULT_CLOSURE_ROOTS,
    N_MAX_CAP,
    StrictRunError,
    StudySpec,
    paired_survival_bands,
    require,
    seed_bank,
)

HERE = Path(__file__).resolve().parent
PACKAGE = "research/frp3_strict_20261005"
STUDY_ID = "frp3-m3s12-strict-20261005"
SCHEMA = "frp3-m3s12-strict-sequential/v1"
NAMESPACES = dict(preflight.NAMESPACES)
MIXES = tuple(dev_screen.MIXES)  # frozen, scripted, mixed
SCRIPTED_MIX = "scripted"
PRIMARY_METRIC = "mass_integral"
NI_FRACTION = 0.03  # as the v7/v8 strict: delta_NI = 0.03 x calibration incumbent scripted mean
N_CALIBRATION = 16  # as the v7/v8 strict (16 dev worlds x 3 mixes, incumbent only)
HORIZON = dev_screen.HORIZON  # 5000
PROFILE_NAME = dev_screen.PROFILE_NAME  # promotion-v2-watch-rect
PROFILE_DIGEST = dev_screen.PROFILE_DIGEST
CONFIG_PATH = dev_screen.DEFAULT_CONFIG  # research/apex_safety_20260926/deployment.yaml
CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
CHECKPOINT_DIR = dev_screen.DEFAULT_CHECKPOINT_DIR
CHAMPION_NAME, CHAMPION_SHA256 = dev_screen.CHAMPION
POOL = tuple(dev_screen.POOL)  # champion + 3 opponents (the roster pool, unchanged)
# Survival band v2 (governance amendment 2026-10-06), option 1, stated explicitly at prepare /
# plan: pooled margin 0.05, alpha 0.05, rci_obf, floor 0.30, per-mix catastrophic margin 0.10.
# History: this package first froze template v2 ``paired_ni_at_stop`` (M 0.05, alpha 0.05,
# floor 0.30; ``pointwise`` failed its per-study check, ``rci_obf`` passed); that band failed
# 53-72% of no-regression runs at the qualifying look for a between-checkpoint change, so the
# owner had the band redesigned before the gate (superseded_v2_band/; no intent was written).
PAIRED_BAND = {
    "band_ni_margin": 0.05,
    "band_alpha": 0.05,
    "band_bound": "rci_obf",
    "band_floor": 0.30,
    "band_mix_margin": 0.10,
}
PAIRED_BAND_ARG = "0.05,0.05,rci_obf,0.30,0.10"

ARTIFACTS = preflight.ARTIFACTS
CANDIDATE_NAME = "frp-v3 M3@60000 seed 12 (apex_mark_u60000.pth)"
CANDIDATE_PATH = ARTIFACTS / "frp-v3-20261005/train/arm-M3/seed-12/checkpoints/apex_mark_u60000.pth"
CANDIDATE_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
HERO_SHA256 = {"incumbent": CHAMPION_SHA256, "candidate": CANDIDATE_SHA256}

ARMS = ("incumbent", "candidate")
VETO_LAMBDA = 8.0
VETO_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"
DIAGNOSTICS_MARKER = "head_checks"  # every SpaceAndHeadVeto diagnostics record carries it
V2 = "src/evaluation/safety_veto.py"
V3 = "src/evaluation/safety_veto_v3.py"
V4 = "src/evaluation/safety_veto_v4.py"
V5 = "src/evaluation/safety_veto_v5.py"
V6 = "src/evaluation/safety_veto_v6.py"
V7 = "src/evaluation/safety_veto_v7.py"
V8 = "src/evaluation/safety_veto_v8.py"
VETO_SOURCES = (V8, V7, V6, V5, V4, V3, V2)  # v8 imports v2, v3, v5, v6, v7; v6 imports v4
VETO_SOURCE_SHA256S = {
    V2: "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    V3: "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
    V4: "3f0881afb7ff8ff980c9b425ef40134f127cc90f1f40260aac39f2264ed44578",
    V5: "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c",
    V6: "a6117cb98383f9f28bf8ebcf22d2ef63d7a03995734a36fb96bb1bf8bb1757d5",
    V7: "56ff7009ce2e0c4c93b6570336b35d48341757fc53b386d309b00126f67e2980",
    V8: "faf3695fa9d60550f1f681c04b8c8aba0447fecaa34cd75f16f75efe91e4ac05",
}

# The released incumbent: the v8 STRICT_PASS receipt (web/backend/safety_veto_serving.py pins
# the same receipt sha256 for the served default since a0cd04b).
V8_STRICT_RECEIPT = ARTIFACTS / "apex-veto-v8-strict-20261003/run-v1/output/receipt.json"
V8_STRICT_RECEIPT_SHA256 = "29b1f7f6f095cac1990e8f7eafccc806e498cd503bb964bd083b3436ef2b7507"

# The candidate's development evidence (Tier 1, non-authoritative): FRP-v3 Phase R.
PHASE_R_SUMMARY = ARTIFACTS / "frp-v3-20261005/phaseR/rp-v1/merged/summary.json"
PHASE_R_SUMMARY_SHA256 = "0fcb80f689f47f70877ac4c9f6b87c2a76bafb0e21bf270c1aecc7f27b3ab206"

# LH-1 long-horizon screen of THIS candidate (branch lh1-screen, job lh1-frp3-s12 at c9adbc3;
# research/longh_screen/lh1.py analyze writes receipt.json create-only). Gate precondition.
LH1_RECEIPT = ARTIFACTS / "lh1-screen-frp3-s12/run-v1/receipt.json"
LH1_JOB_ID = "lh1-frp3-s12"
LH1_JOB_COMMIT = "bc48efa66f24fbee835c4803d32be6dbceaff19f"  # the job's repo_commit
LH1_JOB_FILE_COMMIT = "c9adbc3"  # the commit that added the job file (lh1-screen branch)
LH1_JOB_PATH = "research/longh_screen/screens/frp3_s12.json"
LH1_JOB_SHA256 = "6a52cee470c7736b15f09963211f0a92c81913de76816640fb9ec2898c0be9ff"
LH1_REQUIRED_OUTCOME = "CLEAR"
# The receipt bytes this pre-registration binds (written 2026-10-05 16:51 PDT, outcome CLEAR,
# analysis commit c9adbc3); a re-analysis or any other receipt is refused.
LH1_RECEIPT_SHA256 = "f9879836e1c478906f346dbd8dc48b73f0faa22932364cf011d0a9152ae0044a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded_seeds() -> Dict[str, List[int]]:
    """(5) Every earlier namespace and every saved world seed: :mod:`preflight`."""
    return preflight.excluded_seeds()


# ---------------------------------------------------------------- bindings (fail closed)


def v8_receipt_identity(path: Path = V8_STRICT_RECEIPT) -> Dict[str, Any]:
    """The released incumbent's identity, read from the v8 STRICT_PASS receipt."""
    require(Path(path).is_file(), f"v8 strict receipt {path} missing")
    actual = sha256_file(path)
    require(actual == V8_STRICT_RECEIPT_SHA256, f"v8 strict receipt sha256 {actual} not pinned")
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        receipt.get("receipt") == "strict-pass-receipt-only"
        and receipt.get("decision") in ("STOP_PASS", "FINAL_PASS"),
        "the v8 strict receipt is not a STRICT_PASS receipt",
    )
    cand = receipt["arms"]["candidate"]
    return {
        "receipt_path": str(path),
        "receipt_sha256": actual,
        "decision": receipt["decision"],
        "checkpoint_sha256": cand["checkpoint_sha256"],
        "method": cand["method"],
        "descriptor": cand["descriptor"],
        "source_sha256s": dict(cand["source_sha256s"]),
    }


def phase_r_binding(path: Path = PHASE_R_SUMMARY) -> Dict[str, Any]:
    """FRP-v3 Phase R: GO_R, and the pre-declared candidate is exactly this checkpoint."""
    require(Path(path).is_file(), f"FRP-v3 Phase R summary {path} missing")
    actual = sha256_file(path)
    require(actual == PHASE_R_SUMMARY_SHA256, f"Phase R summary sha256 {actual} not pinned")
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    decision = summary["decision"]
    selection = summary["analysis"]["candidate_selection"]
    require(decision["status"] == "GO_R" and not summary["smoke"], "Phase R is not a real GO_R")
    require(
        selection["checkpoint"]["sha256"] == CANDIDATE_SHA256 and selection["seed"] == 12,
        "Phase R's pre-declared candidate is not seed 12 M3@60000 (eec144bf...)",
    )
    return {
        "summary_path": str(path),
        "summary_sha256": actual,
        "status": decision["status"],
        "authority": summary["authority"],
        "cell": selection["cell"],
        "seed": selection["seed"],
        "selection_rule": selection["rule"],
        "pooled_h5000": decision["pooled_h5000"],
    }


def lh1_binding(path: Optional[Path] = None, pinned_sha256: Optional[str] = None) -> Dict[str, Any]:
    """The LH-1 screen receipt of this candidate: it must exist and be CLEAR (fail closed).

    Schema (``research/longh_screen/analyze.py`` on branch lh1-screen): ``kind == "screen"``,
    ``job_id``, ``repo_commit``, ``job_sha256``, ``missing`` / ``problems`` (empty for a valid
    receipt), ``prefix_controls.verified``, ``contrasts`` (one, C1 vs I), ``outcomes``
    (``{"C1": ...}``) and ``outcome`` (BLOCK > INCOMPLETE > REASON_REQUIRED > CLEAR; INVALID
    when problems). A partial analysis is written as ``receipt.partial.json`` and never counts;
    any twin (``receipt.<ts>.json`` from a re-analysis, or a partial) beside it refuses, and the
    receipt bytes must have the pinned sha256 (default :data:`LH1_RECEIPT_SHA256`).
    """
    path = Path(LH1_RECEIPT if path is None else path)
    pinned = LH1_RECEIPT_SHA256 if pinned_sha256 is None else pinned_sha256
    require(
        path.is_file(),
        f"LH-1 screen receipt {path} missing: the strict gate refuses to start until the LH-1 "
        "screen of this candidate (lh1.py analyze --job research/longh_screen/screens/"
        "frp3_s12.json) has written receipt.json with outcome CLEAR",
    )
    twins = sorted(
        p.name for p in path.parent.glob("receipt*.json") if p.name != path.name and p.is_file()
    )
    require(not twins, f"LH-1 receipt has twins {twins} beside it (re-analysis or partial)")
    actual = sha256_file(path)
    require(actual == pinned, f"LH-1 receipt sha256 {actual} is not the pinned {pinned}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    contrasts = receipt.get("contrasts") or []
    checks = {
        "kind is screen": receipt.get("kind") == "screen",
        f"job_id is {LH1_JOB_ID}": receipt.get("job_id") == LH1_JOB_ID,
        "repo_commit is the job's": receipt.get("repo_commit") == LH1_JOB_COMMIT,
        "job_sha256 is the pinned job file's": receipt.get("job_sha256") == LH1_JOB_SHA256,
        "no missing episodes": receipt.get("missing") == [],
        "no problems": receipt.get("problems") == [],
        "decision code did not drift": receipt.get("decision_code_drift") == [],
        "prefix controls verified": (receipt.get("prefix_controls") or {}).get("verified") is True,
        "one contrast, C1 vs I": len(contrasts) == 1
        and contrasts[0].get("name") == "C1"
        and receipt.get("outcomes") == {"C1": contrasts[0].get("decision", {}).get("outcome")},
        f"outcome is {LH1_REQUIRED_OUTCOME}": receipt.get("outcome") == LH1_REQUIRED_OUTCOME
        and receipt.get("outcomes") == {"C1": LH1_REQUIRED_OUTCOME},
    }
    failed = [name for name, ok in checks.items() if not ok]
    require(
        not failed,
        f"LH-1 screen receipt {path} does not clear this candidate for Tier 2 "
        f"(outcome {receipt.get('outcome')!r}): failed {failed}",
    )
    return {
        "receipt_path": str(path),
        "receipt_sha256": actual,
        "job_id": receipt["job_id"],
        "job_sha256": receipt["job_sha256"],
        "repo_commit": receipt["repo_commit"],
        "analysis_commit": receipt.get("analysis_commit"),
        "intent_sha256": receipt.get("intent_sha256"),
        "outcome": receipt["outcome"],
        "label": receipt.get("label"),
        "rule": "the gate starts only on an LH-1 receipt with outcome CLEAR (REASON_REQUIRED, "
        "BLOCK, INCOMPLETE, INVALID or a missing/partial receipt refuse)",
    }


def _veto_identity(root: Path) -> Dict[str, Any]:
    from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

    sources = {rel: sha256_file(root / rel) for rel in VETO_SOURCES}
    require(sources == VETO_SOURCE_SHA256S, "v8 veto sources differ from the pinned bytes")
    descriptor = SpaceAndHeadVeto(VETO_LAMBDA).descriptor()
    require(descriptor["method"] == VETO_METHOD, "v8 veto method differs")
    require(
        descriptor.get("space_preference_lambda") == VETO_LAMBDA
        and descriptor.get("head_avoidance") is True,
        "the veto is not v8 at lambda 8 with head avoidance",
    )
    return {"descriptor": descriptor, "source_sha256s": sources}


def arm_identities(
    repo: Optional[Path] = None,
    *,
    lh1_receipt: Optional[Path] = None,
    lh1_receipt_sha256: Optional[str] = None,
    v8_receipt: Optional[Path] = None,
    phase_r_summary: Optional[Path] = None,
    candidate_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """(1) Each arm's identity, computed from the files (never typed in), fail closed.

    Both arms: v8 at lambda 8, every veto module pinned to the v8 STRICT_PASS receipt's bytes.
    Incumbent: the champion checkpoint, equal to the receipt's candidate. Candidate: the
    FRP-v3 checkpoint (sha256 pinned), Phase R's pre-declared GO_R candidate, LH-1 CLEAR.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    v8_receipt = V8_STRICT_RECEIPT if v8_receipt is None else v8_receipt
    phase_r_summary = PHASE_R_SUMMARY if phase_r_summary is None else phase_r_summary
    candidate_path = CANDIDATE_PATH if candidate_path is None else candidate_path
    checkpoint_dir = CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    released = v8_receipt_identity(v8_receipt)
    veto = _veto_identity(root)
    champion = sha256_file(Path(checkpoint_dir) / CHAMPION_NAME)
    require(champion == CHAMPION_SHA256, f"champion checkpoint sha256 {champion} differs")
    require(Path(candidate_path).is_file(), f"candidate checkpoint {candidate_path} missing")
    candidate = sha256_file(candidate_path)
    require(candidate == CANDIDATE_SHA256, f"candidate checkpoint sha256 {candidate} differs")
    require(
        released["method"] == VETO_METHOD
        and released["descriptor"] == veto["descriptor"]
        and released["source_sha256s"] == veto["source_sha256s"]
        and released["checkpoint_sha256"] == champion,
        "incumbent is not the released champion + v8 bound by the v8 STRICT_PASS receipt",
    )
    install = (
        "dev_screen.hero_veto_installer(install_space_and_head_veto(hero, 8.0)) around "
        "rollout(hero_safety_veto=True); opponents unwrapped"
    )
    common = {
        "method": VETO_METHOD,
        "descriptor": veto["descriptor"],
        "source_sha256": veto["source_sha256s"][V8],
        "source_sha256s": veto["source_sha256s"],
        "install": install,
    }
    return {
        "incumbent": {
            **common,
            "checkpoint_name": CHAMPION_NAME,
            "checkpoint_sha256": champion,
            "release_binding": {
                "v8_strict_receipt_path": released["receipt_path"],
                "v8_strict_receipt_sha256": released["receipt_sha256"],
                "v8_strict_decision": released["decision"],
                "served_default_since": "a0cd04b (web/backend/safety_veto_serving.py pins it)",
            },
        },
        "candidate": {
            **common,
            "checkpoint_name": CANDIDATE_NAME,
            "checkpoint_path": str(candidate_path),
            "checkpoint_sha256": candidate,
            "phase_r_binding": phase_r_binding(phase_r_summary),
            "screen_binding": lh1_binding(lh1_receipt, lh1_receipt_sha256),
        },
    }


# ---------------------------------------------------------------- rosters


@functools.lru_cache(maxsize=4)
def _bank_rows(domain: str) -> Dict[tuple, Dict[str, Any]]:
    """Strict balanced rosters of a domain's first ``N_MAX_CAP`` worlds, keyed by world (the
    v8 strict construction; a prefix bank has the prefix rows)."""
    seeds = seed_bank(domain, N_MAX_CAP)
    index = {seed: i for i, seed in enumerate(seeds)}
    require(len(index) == len(seeds), f"seed collision inside {domain}")
    rows = {}
    for row in dev_screen._design_rows(seeds):
        rows[(row["mix"], index[int(row["world_seed"])])] = row
    return rows


def build_row(phase: str, mix: str, world_index: int, world_seed: int) -> Dict[str, Any]:
    """(3) The strict balanced roster of the world (dev_screen._design_rows), plus its index."""
    require(phase in NAMESPACES, f"unknown phase {phase}")
    row = _bank_rows(NAMESPACES[phase])[(mix, int(world_index))]
    require(int(row["world_seed"]) == int(world_seed), "roster seed differs from the bank")
    return {**row, "world_index": int(world_index)}


# ---------------------------------------------------------------- episodes


def checkpoint_paths() -> Dict[str, str]:
    """sha256 -> path of every checkpoint the gate loads on the Mac (pool + candidate)."""
    out = {sha: str(Path(CHECKPOINT_DIR) / name) for name, sha in POOL}
    out[CANDIDATE_SHA256] = str(CANDIDATE_PATH)
    return out


def remote_checkpoints() -> List[str]:
    """Checkpoint sha256s a RunPod worker needs (each on the upload allow-list)."""
    return sorted({sha for _, sha in POOL} | {CANDIDATE_SHA256})


def verify_checkpoints(paths: Mapping[str, str]) -> Dict[str, str]:
    """Re-hash every mapped checkpoint against its sha256 key (fail closed)."""
    require(set(paths) == set(remote_checkpoints()), "checkpoint map is not pool + candidate")
    for expected, path in paths.items():
        actual = sha256_file(Path(path))
        require(actual == expected, f"checkpoint {path} sha256 {actual} != pinned {expected}")
    return dict(paths)


def _context(intent: Mapping[str, Any], paths: Mapping[str, str]) -> Dict[str, Any]:
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    dev_screen._configure_torch()
    require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "deployment config drift")
    load_and_initialize_config(str(CONFIG_PATH))
    profile = evaluation_profile_for_name(PROFILE_NAME, HORIZON)
    require(profile.digest == PROFILE_DIGEST, "evaluation profile digest drift")
    paths = verify_checkpoints(paths)
    return {
        "profile": profile,
        "profile_ref": {"descriptor": profile.descriptor(), "digest": profile.digest},
        "lookup": dev_screen.agent_lookup(paths),
        "paths": paths,
        "arms": intent["arms"],
    }


def worker_setup(intent: Mapping[str, Any]) -> Dict[str, Any]:
    """Per Mac worker: torch threads, pinned config and profile, verified checkpoints."""
    return _context(intent, checkpoint_paths())


def remote_worker_setup(intent: Mapping[str, Any], ckpt_dir: Path) -> Dict[str, Any]:
    """Per RunPod worker: the same, with the checkpoints at ``ckpt_dir/<sha256>.pth``."""
    paths = {sha: str(Path(ckpt_dir) / f"{sha}.pth") for sha in remote_checkpoints()}
    return _context(intent, paths)


def install_v8(hero: Any) -> Any:
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto

    return install_space_and_head_veto(hero, VETO_LAMBDA)


def episode_runner(
    episode: Mapping[str, Any], row: Mapping[str, Any], context: Mapping[str, Any]
) -> Dict[str, Any]:
    """(2) One H5000 hero episode of the arm on the world; strict record shape enforced.

    The arm's hero checkpoint is the one its frozen identity names; both arms carry v8 at
    lambda 8. Returns the rollout record plus ``veto_diagnostics`` (reported, not gated) and
    ``hero_checkpoint_sha256``. Any shape or identity problem raises (the gate stops).
    """
    from src.evaluation.strict_promotion import (
        _expected_world_identity,
        validate_strict_world_record,
    )
    from src.scripts.tournament_eval import rollout

    arm = episode["arm"]
    require(arm in HERO_SHA256, f"unknown arm {arm}")
    require(int(row["world_seed"]) == int(episode["world_seed"]), "row/episode seed mismatch")
    identity = context["arms"][arm]
    hero_sha = identity["checkpoint_sha256"]
    require(hero_sha == HERO_SHA256[arm], f"{arm} identity names checkpoint {hero_sha[:12]}")
    verify_checkpoints(context["paths"])  # the bytes rollout loads are the pinned ones
    lookup = context["lookup"]
    hero = lookup[hero_sha]
    require(hero == ("checkpoint", context["paths"][hero_sha]), f"{arm} hero is not its file")
    opponents = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    with dev_screen.hero_veto_installer(install_v8) as installed:
        record = rollout(
            hero,
            opponents,
            HORIZON,
            int(row["world_seed"]),
            profile=context["profile"],
            world_identity=_expected_world_identity(row),
            mix_id=row["mix"],
            hero_safety_veto=True,
        )
    require(len(installed) == 1, f"the {arm} veto was not installed exactly once")
    try:
        validate_strict_world_record(
            record, context["profile_ref"], row, candidate_wrapper=identity["descriptor"]
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a run failure
        raise StrictRunError(f"{episode['episode_id']}: strict record shape: {exc}") from exc
    return {
        **dict(record),
        "veto_diagnostics": installed[0].diagnostics_record(),
        "hero_checkpoint_sha256": hero_sha,
    }


def validate_record(entry: Mapping[str, Any], row: Mapping[str, Any]) -> List[str]:
    """(4) Envelope-level checks on top of the strict validator run in ``episode_runner``."""
    problems: List[str] = []
    arm = entry.get("arm")
    record = entry.get("record") or {}
    if arm not in HERO_SHA256:
        return [f"unknown arm {arm!r}"]
    probe = (record.get("probes") or {}).get("safety_veto") or {}
    if probe.get("method") != VETO_METHOD:
        problems.append(f"{arm} probe method {probe.get('method')!r}")
    diagnostics = record.get("veto_diagnostics")
    if not isinstance(diagnostics, Mapping) or DIAGNOSTICS_MARKER not in diagnostics:
        problems.append(f"{arm} veto_diagnostics of the wrong kind")
    if record.get("hero_checkpoint_sha256") != HERO_SHA256[arm]:
        problems.append(f"{arm} hero checkpoint {record.get('hero_checkpoint_sha256')!r}")
    identity = record.get("world_identity") or {}
    if identity.get("seed") != row.get("world_seed") or identity.get("mix_id") != row.get("mix"):
        problems.append("world identity differs from the roster row")
    if record.get("seed") != row.get("world_seed"):
        problems.append("record seed differs from the roster row")
    return problems


SPEC = StudySpec(
    study_id=STUDY_ID,
    schema=SCHEMA,
    namespaces=dict(NAMESPACES),
    arm_identities=arm_identities,
    episode_runner=episode_runner,
    excluded_seeds=excluded_seeds,
    protocol_path=f"{PACKAGE}/protocol.md",
    oc_report_path=f"{PACKAGE}/operating_characteristics.json",
    band_cost_report_path=f"{PACKAGE}/look1_band_cost.json",
    mixes=MIXES,
    scripted_mix=SCRIPTED_MIX,
    primary_metric=PRIMARY_METRIC,
    ni_fraction=NI_FRACTION,
    bands=paired_survival_bands(MIXES),
    worker_setup=worker_setup,
    build_row=build_row,
    validate_record=validate_record,
    closure_roots=(*DEFAULT_CLOSURE_ROOTS, "research/apex_safety_20260926", PACKAGE),
    band_policy="pooled_ni_continue",
    paired_band_check_path=f"{PACKAGE}/paired_band_check.json",
    paired_band_pool_path=f"{PACKAGE}/paired_pool.json",
    remote_worker_setup=remote_worker_setup,
    remote_checkpoints=remote_checkpoints,
)
