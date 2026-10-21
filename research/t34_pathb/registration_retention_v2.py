"""Create and validate the prospective T34 Path B v2 registration package.

This module only freezes study inputs.  It cannot execute a pre-check, construct
an adapter, or start an efficacy world.  ``ratify`` is deliberately create-only:
the operator supplies the independently
produced trace receipt after the executable source closure is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

# ``bank`` reaches the roster materializer; cap native pools before that import.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

from research.t34_pathb import bank  # noqa: E402
from research.t34_pathb.retention_v2 import (  # noqa: E402
    ARMS,
    BARE_UNPACED_AMENDMENT,
    DEFAULT_STUDY_ROOT,
    MAC_ARMS,
    MAX_CPU_HOURS,
    MAX_ELAPSED_HOURS,
    MIXES,
    PRECHECK_MAX_ELAPSED_HOURS,
    PRODUCT_SESSIONS,
    TERMINAL_PER_MIX,
    TIMING_MAX_ATTEMPTS,
    TIMING_MAX_ELAPSED_HOURS,
    RegistrationError,
    _registered_seed_union,
    _validate_checkpoint_coverage,
    ledger_policy,
    require_registration,
    source_inventory,
    timing_attempt_parent,
    unpaced_colab_policy,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "t34-pathb-v2-registration/v1"
NAMESPACE_ROOT = "t34-retention-prereg-20261020/v1"
DEFAULT_BANK = Path("research/t34_pathb/RETENTION_BANK_V2.json")
DEFAULT_AUDIT = Path("research/t34_pathb/SEED_AUDIT_RETENTION_V2.json")
DEFAULT_PROFILE = Path("research/t34_pathb/RETENTION_PROFILE_V2.json")
DEFAULT_OC = Path("research/t34_pathb/RETENTION_OC_V2.json")
DEFAULT_REGISTRATION = Path("research/t34_pathb/REGISTRATION_RETENTION_V2.json")
DEFAULT_PREREG = Path("research/t34_pathb/PREREGISTRATION_RETENTION_V2.md")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _read_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_pairs
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RegistrationError(f"{path}: expected readable JSON object") from exc
    if not isinstance(value, dict):
        raise RegistrationError(f"{path}: expected JSON object")
    return value


def _binding(path: Path, *, root: Path = ROOT) -> dict[str, str]:
    absolute = path if path.is_absolute() else root / path
    if not absolute.is_file():
        raise RegistrationError(f"required binding file is absent: {path}")
    # Keep project files relocatable and artifact files explicit.
    display = str(path) if not path.is_absolute() else str(path)
    return {"path": display, "sha256": _sha256(absolute)}


def _bound_path(item: Any, *, root: Path = ROOT, label: str) -> Path:
    if (
        not isinstance(item, Mapping)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        or len(item["sha256"]) != 64
    ):
        raise RegistrationError(f"{label} requires a path and sha256")
    path = Path(item["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise RegistrationError(f"{label} bytes differ from its binding")
    return path


def _namespace(role: str) -> str:
    return f"{NAMESPACE_ROOT}/{role}"


def _materialize_terminal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the historical roster authority for every fresh terminal world."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine

    context = gid._context(1)
    plans = bank_engine.planned_rosters(
        context, [row["seed"] for row in rows], [row["mix"] for row in rows]
    )
    if len(plans) != len(rows):
        raise RegistrationError("historical roster authority returned an incomplete allocation")
    for row, roster in zip(rows, plans):
        if roster.get("seed") != row["seed"] or roster.get("mix") != row["mix"]:
            raise RegistrationError("historical roster authority changed a fresh row identity")
        row["roster"] = roster
    return rows


def allocate_bank(*, repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allocate all 177 identities atomically and audit them before materializing.

    There is no collision replacement.  A hit in any namespace invalidates the
    complete allocation, including the otherwise independent product and timing
    namespaces.
    """
    repo = repo.resolve()
    terminal_rows: list[dict[str, Any]] = []
    candidates: list[int] = []
    namespaces: dict[str, str] = {}
    for mix in MIXES:
        name = _namespace(f"terminal-{mix}")
        seeds = bank._candidate_seeds(name, TERMINAL_PER_MIX)
        namespaces[f"terminal_{mix}"] = name
        candidates.extend(seeds)
        terminal_rows.extend(
            {"seed": seed, "mix": mix, "kind": "terminal", "namespace": name} for seed in seeds
        )
    product_name = _namespace("product-mixed-watch6")
    product_seeds = bank._candidate_seeds(product_name, PRODUCT_SESSIONS)
    namespaces["product_mixed_watch6"] = product_name
    candidates.extend(product_seeds)
    product_rows = [
        {"seed": seed, "mix": "mixed", "kind": "product", "namespace": product_name}
        for seed in product_seeds
    ]
    timing_name = _namespace("timing-mixed-watch6")
    timing_seed = bank._candidate_seeds(timing_name, 1)[0]
    namespaces["timing_mixed_watch6"] = timing_name
    candidates.append(timing_seed)
    if len(candidates) != len(set(candidates)):
        raise RegistrationError("fresh namespaces contain an internal cross-bank collision")
    audit = bank._audit_candidates(repo, candidates)
    terminal_rows = _materialize_terminal(terminal_rows)
    timing_row = {
        "seed": timing_seed,
        "mix": "mixed",
        "kind": "timing",
        "namespace": timing_name,
    }
    payload = {
        "schema": "t34-pathb-v2-retention-bank/v1",
        "fresh_disjoint_banks": True,
        "namespaces": namespaces,
        "terminal_rows": terminal_rows,
        "product_rows": product_rows,
        "timing_row": timing_row,
    }
    audit.update(
        {
            "schema": "t34-pathb-v2-retention-seed-audit/v1",
            "pass": True,
            "seeds": sorted(candidates),
            "collisions": [],
            "namespaces": namespaces,
            "terminal_per_mix": TERMINAL_PER_MIX,
            "product_sessions": PRODUCT_SESSIONS,
            "timing_seeds": 1,
        }
    )
    return payload, audit


def write_allocation(
    *, bank_path: Path = DEFAULT_BANK, audit_path: Path = DEFAULT_AUDIT, repo: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the audited allocation as two create-only committed-input candidates."""
    payload, audit = allocate_bank(repo=repo)
    _write_new(repo / bank_path, payload)
    _write_new(repo / audit_path, audit)
    return payload, audit


def resize_allocation(
    *,
    bank_path: Path = DEFAULT_BANK,
    audit_path: Path = DEFAULT_AUDIT,
    repo: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain the original 113 identities and audit all 64 additions at once.

    The frozen namespace/index allocator is unchanged. Existing terminal rows
    keep their order and rosters; the new terminal block follows them. Audit
    additions against the still-published original bank and all other history
    before producing any output. A collision rejects the complete extension.
    """
    repo = repo.resolve()
    previous = _read_object(repo / bank_path)
    previous_audit = _read_object(repo / audit_path)
    terminal = previous["terminal_rows"]
    product = previous["product_rows"]
    timing = previous["timing_row"]
    retained = _registered_seed_union(previous)
    if (
        len(terminal) != 96
        or len(product) != 16
        or len(retained) != 113
        or previous_audit.get("pass") is not True
        or previous_audit.get("collisions")
        or any(previous_audit.get("concrete_hits", {}).values())
        or set(previous_audit.get("seeds", ())) != retained
    ):
        raise RegistrationError("resize requires the original collision-free 113-seed bank")
    additions: list[dict[str, Any]] = []
    for mix in MIXES:
        namespace = _namespace(f"terminal-{mix}")
        old = [row for row in terminal if row["mix"] == mix]
        seeds = bank._candidate_seeds(namespace, TERMINAL_PER_MIX)
        if len(old) != 32 or [row["seed"] for row in old] != seeds[:32]:
            raise RegistrationError("retained terminal seeds differ from frozen allocator prefix")
        additions.extend(
            {"seed": seed, "mix": mix, "kind": "terminal", "namespace": namespace}
            for seed in seeds[32:]
        )
    namespace = _namespace("product-mixed-watch6")
    seeds = bank._candidate_seeds(namespace, PRODUCT_SESSIONS)
    if [row["seed"] for row in product] != seeds[:16] or timing["seed"] != bank._candidate_seeds(
        _namespace("timing-mixed-watch6"), 1
    )[0]:
        raise RegistrationError("retained product/timing seeds differ from frozen allocator")
    products = [
        {"seed": seed, "mix": "mixed", "kind": "product", "namespace": namespace}
        for seed in seeds[16:]
    ]
    candidates = [row["seed"] for row in additions + products]
    if len(candidates) != 64 or len(set(candidates) | retained) != 177:
        raise RegistrationError(
            "whole resize allocation rejected: internal/retained seed collision"
        )
    audit = bank._audit_candidates(repo, candidates)
    # Explicit coverage binding for the Round 0 draft, when visible in this checkout.
    round0 = Path("research/t34_round0/BANK_ROUND0.json")
    audit.update(
        {
            "schema": "t34-pathb-v2-retention-seed-audit/v1",
            "pass": True,
            "seeds": sorted(retained | set(candidates)),
            "collisions": [],
            "namespaces": previous["namespaces"],
            "terminal_per_mix": TERMINAL_PER_MIX,
            "product_sessions": PRODUCT_SESSIONS,
            "timing_seeds": 1,
            "resize": {
                "specification": "docs/research/specs/2026-10-20_t34_retention_resize.md",
                "retained_count": len(retained),
                "retained_seeds": sorted(retained),
                "added_seeds": sorted(candidates),
                "previous_bank": _binding(bank_path, root=repo),
                "previous_audit": _binding(audit_path, root=repo),
                "previous_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "previous_audit_evidence": previous_audit,
                "round0_bank": _binding(round0, root=repo) if (repo / round0).is_file() else None,
            },
        }
    )
    result = dict(previous)
    result["terminal_rows"] = terminal + _materialize_terminal(additions)
    result["product_rows"] = product + products
    return result, audit


def _check_bank_and_audit(registration: Mapping[str, Any], *, root: Path) -> None:
    bank_payload = _read_object(_bound_path(registration.get("bank"), root=root, label="bank"))
    audit = _read_object(_bound_path(registration.get("seed_audit"), root=root, label="seed_audit"))
    for key in ("terminal_rows", "product_rows", "timing_row"):
        if bank_payload.get(key) != registration.get(key):
            raise RegistrationError(f"bank does not reproduce registered {key}")
    expected = _registered_seed_union(registration)
    if audit.get("pass") is not True or audit.get("collisions"):
        raise RegistrationError("seed audit is not collision-free")
    if any(audit.get("concrete_hits", {}).values()):
        raise RegistrationError("seed audit records a concrete historical collision")
    if set(audit.get("seeds", ())) != expected:
        raise RegistrationError("seed audit does not bind the registered seed union")


def _check_rows(registration: Mapping[str, Any]) -> None:
    terminal, product, timing = (
        registration.get("terminal_rows"),
        registration.get("product_rows"),
        registration.get("timing_row"),
    )
    if not isinstance(terminal, list) or len(terminal) != TERMINAL_PER_MIX * len(MIXES):
        raise RegistrationError("registration needs exactly 48 terminal rows per mix")
    if not isinstance(product, list) or len(product) != PRODUCT_SESSIONS:
        raise RegistrationError("registration needs exactly 32 product rows")
    if not isinstance(timing, Mapping) or timing.get("mix") != "mixed":
        raise RegistrationError("registration needs one mixed timing row")
    if any(sum(row.get("mix") == mix for row in terminal) != TERMINAL_PER_MIX for mix in MIXES):
        raise RegistrationError("terminal rows do not have the registered mix balance")
    if any(row.get("mix") != "mixed" for row in product):
        raise RegistrationError("product rows must be fixed mixed Watch6 sessions")
    seeds = _registered_seed_union(registration)
    if len(seeds) != TERMINAL_PER_MIX * len(MIXES) + PRODUCT_SESSIONS + 1:
        raise RegistrationError("registered rows repeat a fresh-bank seed")


def _check_common(registration: Mapping[str, Any], *, root: Path) -> None:
    if registration.get("schema") != SCHEMA:
        raise RegistrationError("missing T34 retention registration schema")
    if registration.get("arms") != list(ARMS) or registration.get("paced_fps") != 24:
        raise RegistrationError("arms or 24 fps paced design differs from protocol")
    if registration.get("bare_fps") != 0:
        raise RegistrationError("bare arm must run unpaced under the pre-data amendment")
    amendments = registration.get("amendments")
    if (
        not isinstance(amendments, list)
        or len(amendments) != 1
        or not isinstance(amendments[0], Mapping)
        or amendments[0].get("path") != BARE_UNPACED_AMENDMENT
    ):
        raise RegistrationError("registration must point to the bare-unpaced amendment")
    _bound_path(amendments[0], root=root, label="bare-unpaced amendment")
    expected = {
        "mac_arms": list(MAC_ARMS),
        "unpaced_colab": unpaced_colab_policy(),
        "terminal_per_mix": TERMINAL_PER_MIX,
        "product_sessions": PRODUCT_SESSIONS,
        "horizon": 5000,
        "workers": 1,
        "max_elapsed_hours": MAX_ELAPSED_HOURS,
        "max_cpu_hours": MAX_CPU_HOURS,
        "precheck_max_elapsed_hours": PRECHECK_MAX_ELAPSED_HOURS,
        "timing_max_elapsed_hours": TIMING_MAX_ELAPSED_HOURS,
        "fresh_disjoint_banks": True,
        "timing_ticks": 5000,
        "timing_warmup": 200,
    }
    for key, value in expected.items():
        if registration.get(key) != value:
            raise RegistrationError(f"registration {key} differs from the fixed protocol")
    timing_attempt_parent(registration)
    ledger_policy(registration)
    _check_rows(registration)
    for label in ("profile", "bank", "seed_audit", "preregistration", "operating_characteristic"):
        _bound_path(registration.get(label), root=root, label=label)
    profile_payload = _read_object(
        _bound_path(registration.get("profile"), root=root, label="profile")
    )
    amendment = profile_payload.get("survival_band", {}).get("amendment")
    amendment_path = _bound_path(amendment, root=root, label="survival-band amendment")
    required_ratification = (
        "- Decision: ratified\n- Option: 1\n- Date: 2026-10-06\n- Changes, if any: none"
    )
    if required_ratification not in amendment_path.read_text():
        raise RegistrationError("survival-band amendment lacks the exact final ratification text")
    _check_load_policy(registration, root=root)
    _check_bank_and_audit(registration, root=root)
    checkpoints = registration.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RegistrationError("checkpoint binding inventory is required")
    for index, checkpoint in enumerate(checkpoints):
        _bound_path(checkpoint, root=root, label=f"checkpoints[{index}]")
    _validate_checkpoint_coverage(registration)


def _check_load_policy(registration: Mapping[str, Any], *, root: Path) -> None:
    """Require the same prospective measured-host policy in both bindings."""
    profile = _read_object(_bound_path(registration.get("profile"), root=root, label="profile"))
    for key in (
        "max_elapsed_hours",
        "max_cpu_hours",
        "precheck_max_elapsed_hours",
        "timing_max_elapsed_hours",
    ):
        if profile.get(key) != registration.get(key):
            raise RegistrationError(f"profile {key} differs from registered cap")
    if registration.get("load_profile") != profile.get("load_policy"):
        raise RegistrationError("load_profile differs from profile load_policy")
    from research.t34_pathb.host_guard_v2 import validate_policy

    try:
        validate_policy(registration["load_profile"])
    except (ValueError, TypeError, KeyError) as exc:
        raise RegistrationError(f"invalid measured-host load policy: {exc}") from exc


def _check_operating_characteristic(registration: Mapping[str, Any], *, root: Path) -> None:
    """Accept a completed OC receipt only when it binds a real fixed-N simulation."""
    payload = _read_object(
        _bound_path(
            registration.get("operating_characteristic"),
            root=root,
            label="operating_characteristic",
        )
    )
    if (
        payload.get("schema") != "t34-retention-fixed-n-oc/v1"
        or payload.get("status") != "PASS"
        or payload.get("source_hashes") != registration.get("source_hashes")
    ):
        raise RegistrationError(
            "operating characteristic does not bind passing registered source bytes"
        )
    for key in ("simulator_hash", "world_pool_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RegistrationError(f"operating characteristic lacks a SHA-256 {key} binding")
    pool = payload.get("development_pool")
    if not isinstance(pool, (list, Mapping)):
        raise RegistrationError("operating characteristic lacks its materialized development pool")
    pool_bytes = json.dumps(pool, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(pool_bytes).hexdigest() != payload["world_pool_sha256"]:
        raise RegistrationError("operating characteristic world-pool bytes differ from its hash")
    source_files = pool.get("source_files") if isinstance(pool, Mapping) else None
    if not isinstance(source_files, Mapping) or not source_files:
        raise RegistrationError("operating characteristic pool lacks source-file provenance")
    for name, digest in source_files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RegistrationError(
                "operating characteristic pool source-file binding is malformed"
            )
        path = Path(name)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or _sha256(path) != digest:
            raise RegistrationError("operating characteristic pool source-file bytes differ")
    simulator = root / "research/t34_pathb/retention_oc_v2.py"
    if not simulator.is_file() or _sha256(simulator) != payload["simulator_hash"]:
        raise RegistrationError("operating characteristic simulator bytes differ from its hash")
    if (
        type(payload.get("completed")) is not int
        or payload["completed"] < 20_000
        or payload.get("reps") != payload["completed"]
        or payload.get("n_per_mix") != TERMINAL_PER_MIX
        or payload.get("looks") != [TERMINAL_PER_MIX]
        or payload.get("threads") != 1
        or payload.get("development_only") is not True
        or payload.get("efficacy_rows_produced") is not False
        or payload.get("active_arms") != ["paced", "unpaced"]
    ):
        raise RegistrationError(
            "operating characteristic is not a completed fixed-N development simulation"
        )
    rates = payload.get("rates")
    if not isinstance(rates, Mapping) or not rates:
        raise RegistrationError("operating characteristic lacks fixed-N error rates")
    models, scenarios = payload.get("models"), payload.get("scenarios")
    theta_grid, alpha_grid = payload.get("theta_grid"), payload.get("alpha_grid")
    if (
        not isinstance(models, list)
        or models
        != [
            "t3_dense_b8_s0_scripted_shape_all_mixes",
            "t3_dense_b8_s3_scripted_shape_all_mixes",
            "normal_stress_sd_0.25_watch_mass_scale",
            "normal_stress_sd_0.35_watch_mass_scale",
            "normal_stress_sd_0.5_watch_mass_scale",
        ]
        or not isinstance(scenarios, Mapping)
        or scenarios
        != {
            "no_change": [0.0, 0.0, 0.0],
            "pooled_minus_0.05": [-0.05, -0.05, -0.05],
            "frozen_minus_0.075": [-0.075, 0.0, 0.0],
            "scripted_minus_0.075": [0.0, -0.075, 0.0],
            "mixed_minus_0.075": [0.0, 0.0, -0.075],
        }
        or theta_grid != [20.0, 26.8, 40.0, 60.0, 80.0]
        or alpha_grid != [0.05, 0.10]
    ):
        raise RegistrationError("operating characteristic lacks the registered full fixed-N grid")
    expected_protected = {
        f"{model}|{scenario}|theta={theta:g}|alpha={alpha:g}|joint_mass_and_band"
        for model in models
        for scenario in scenarios
        for theta in theta_grid
        for alpha in alpha_grid
    }
    if not expected_protected.issubset(rates):
        raise RegistrationError("operating characteristic omits registered fixed-N grid cells")
    protected = []
    for name, row in rates.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise RegistrationError("operating-characteristic rate row is malformed")
        count, trials, rate = row.get("k"), row.get("n"), row.get("rate")
        if (
            type(count) is not int
            or type(trials) is not int
            or trials != payload["completed"]
            or not 0 <= count <= trials
            or type(rate) not in (int, float)
            or abs(rate - count / trials) > 1e-12
        ):
            raise RegistrationError("operating-characteristic rate row is invalid")
        if name.endswith("|joint_mass_and_band") and "|no_change|" not in name:
            protected.append(rate)
    if not protected or any(rate > 0.06 for rate in protected):
        raise RegistrationError("operating-characteristic protected regression rate exceeds 0.06")
    acceptance = payload.get("acceptance")
    if (
        not isinstance(acceptance, Mapping)
        or type(acceptance.get("max_joint_regression_pass_rate")) not in (int, float)
        or acceptance["max_joint_regression_pass_rate"] != max(protected)
        or acceptance.get("joint_regression_pass_rate_limit") != 0.06
        or acceptance.get("failed_cells") not in ({}, [])
    ):
        raise RegistrationError(
            "operating characteristic acceptance record is not a passing fixed-N result"
        )

    from research.t34_pathb.retention_oc_v2 import pathb_power_scenarios

    power = payload.get("pathb_joint_go_power", {})
    product = payload.get("watch6_product_power", {})
    if not isinstance(power, Mapping) or not isinstance(product, Mapping):
        raise RegistrationError("operating characteristic lacks Path B / Watch6 power objects")
    if not isinstance(power.get("rates"), Mapping) or not isinstance(product.get("rates"), Mapping):
        raise RegistrationError("operating characteristic lacks Path B / Watch6 power rates")
    effects = pathb_power_scenarios()
    product_keys = {
        f"{baseline}|reduction={reduction:g}|paired_ci95_excludes_zero"
        for baseline in ("t15_mixed_watch6", "t13b_mixed_watch6")
        for reduction in (0.2, 0.4)
    }
    joint_keys = {
        f"{model}|pathb_joint_go|{effect}|{product_key}"
        for model in models
        for effect in effects
        for product_key in product_keys
    }
    if (
        power.get("n_per_mix") != TERMINAL_PER_MIX
        or power.get("effects") != effects
        or set(power.get("rates", {})) != joint_keys
        or product.get("sessions") != PRODUCT_SESSIONS
        or product.get("relative_death_reductions") != [0.2, 0.4]
        or set(product.get("rates", {})) != {f"product_power|{key}" for key in product_keys}
    ):
        raise RegistrationError("operating characteristic lacks full Path B / Watch6 power grid")
    for table in (power["rates"], product["rates"]):
        if any(row != rates.get(key) for key, row in table.items()):
            raise RegistrationError("operating-characteristic power rates differ from trials")


def build_draft(
    *,
    profile: Path = DEFAULT_PROFILE,
    bank_path: Path = DEFAULT_BANK,
    audit_path: Path = DEFAULT_AUDIT,
    preregistration: Path = DEFAULT_PREREG,
    operating_characteristic: Path = DEFAULT_OC,
    checkpoints: Sequence[Mapping[str, Any]],
    study_root: Path = DEFAULT_STUDY_ROOT,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build an unratified package; it remains unauthorized by construction."""
    bank_payload = _read_object(root / bank_path)
    profile_payload = _read_object(root / profile)
    load_policy = profile_payload.get("load_policy")
    if not isinstance(load_policy, Mapping):
        raise RegistrationError("profile requires a prospective load_policy object")
    registration = {
        "schema": SCHEMA,
        "study_root": str(study_root),
        "timing_attempt_parent": str(study_root / "timing"),
        "timing_max_attempts": TIMING_MAX_ATTEMPTS,
        "history_ledger": {
            "repo": str(root.resolve()),
            "path": f"research/t34_pathb/ledger/{study_root.name}.jsonl",
        },
        "ratified": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_hashes": source_inventory(root),
        "arms": list(ARMS),
        "mac_arms": list(MAC_ARMS),
        "unpaced_colab": unpaced_colab_policy(),
        "paced_fps": 24,
        "bare_fps": 0,
        "amendments": [_binding(Path(BARE_UNPACED_AMENDMENT), root=root)],
        "terminal_per_mix": TERMINAL_PER_MIX,
        "product_sessions": PRODUCT_SESSIONS,
        "horizon": 5000,
        "workers": 1,
        "max_elapsed_hours": MAX_ELAPSED_HOURS,
        "max_cpu_hours": MAX_CPU_HOURS,
        "precheck_max_elapsed_hours": PRECHECK_MAX_ELAPSED_HOURS,
        "timing_max_elapsed_hours": TIMING_MAX_ELAPSED_HOURS,
        "fresh_disjoint_banks": True,
        "terminal_rows": bank_payload["terminal_rows"],
        "product_rows": bank_payload["product_rows"],
        "timing_row": bank_payload["timing_row"],
        "timing_ticks": 5000,
        "timing_warmup": 200,
        "load_profile": dict(load_policy),
        "profile": _binding(profile, root=root),
        "bank": _binding(bank_path, root=root),
        "seed_audit": _binding(audit_path, root=root),
        "preregistration": _binding(preregistration, root=root),
        "operating_characteristic": _binding(operating_characteristic, root=root),
        "checkpoints": [dict(item) for item in checkpoints],
        "trace_receipt": None,
    }
    _check_common(registration, root=root)
    _check_operating_characteristic(registration, root=root)
    return registration


def validate_registration(
    registration: Mapping[str, Any],
    *,
    root: Path = ROOT,
    ready: bool = False,
    precheck: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate package bytes; only ready mode can authorize execution."""
    _check_common(registration, root=root)
    _check_operating_characteristic(registration, root=root)
    sources = registration.get("source_hashes")
    if not isinstance(sources, Mapping) or not sources:
        raise RegistrationError("full source inventory is required")
    if ready:
        trace = _bound_path(registration.get("trace_receipt"), root=root, label="trace_receipt")
        trace_payload = _read_object(trace)
        if trace_payload.get("status") != "PASS" or trace_payload.get("source_hashes") != sources:
            raise RegistrationError("trace receipt does not bind passing registered source bytes")
        require_registration(registration, precheck)
        return {"valid": True, "authorized": True, "mode": "ready"}
    if registration.get("ratified") is not False or not isinstance(
        registration.get("source_commit"), str
    ):
        raise RegistrationError(
            "draft validation requires an explicitly unratified, source-pinned package"
        )
    if registration.get("trace_receipt") is not None:
        raise RegistrationError("draft may not claim a trace receipt")
    if dict(sources) != source_inventory(root):
        raise RegistrationError("draft source inventory is stale; rebuild after source changes")
    source_commit = registration["source_commit"]
    if (
        len(source_commit) != 40
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=root
        ).returncode
    ):
        raise RegistrationError("draft source commit is not an ancestor of execution HEAD")
    for name, digest in sources.items():
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{name}"], cwd=root
            )
        except subprocess.CalledProcessError as exc:
            raise RegistrationError(f"draft source is absent from source commit: {name}") from exc
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RegistrationError(f"draft source commit differs: {name}")
    return {"valid": True, "authorized": False, "mode": "draft"}


def ratify_registration(
    draft: Mapping[str, Any],
    *,
    source_commit: str,
    precheck: Mapping[str, Any] | None = None,
    trace_receipt: Path,
    operator: str,
    acknowledge_near_boundary_precedence: bool,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Convert a validated draft to a ready package without manufacturing evidence."""
    validate_registration(draft, root=root, ready=False)
    if operator != "Claude" or acknowledge_near_boundary_precedence is not True:
        raise RegistrationError(
            "ratification requires Claude's near-boundary precedence acknowledgement"
        )
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RegistrationError("ratification requires the exact 40-character source commit")
    current_sources = source_inventory(root)
    if draft.get("source_hashes") != current_sources:
        raise RegistrationError("source closure changed since draft; rebuild before ratification")
    if precheck is not None:
        raise RegistrationError("ratification is pre-data; run timing before the T3 pre-check")
    result = dict(draft)
    result.update(
        {
            "ratified": True,
            "source_commit": source_commit,
            "trace_receipt": _binding(trace_receipt, root=root),
            "ratified_by": operator,
            "near_boundary_precedence_acknowledged": True,
        }
    )
    trace_payload = _read_object(
        _bound_path(result["trace_receipt"], root=root, label="trace_receipt")
    )
    if (
        trace_payload.get("status") != "PASS"
        or trace_payload.get("source_hashes") != current_sources
    ):
        raise RegistrationError("trace receipt does not bind passing frozen source bytes")
    # Verify the supplied revision has exactly the bytes which the pre-check observed.
    for name, digest in current_sources.items():
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{name}"], cwd=root
            )
        except subprocess.CalledProcessError as exc:
            raise RegistrationError(f"source is absent from ratified commit: {name}") from exc
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RegistrationError(f"ratified commit differs for {name}")
    _check_common(result, root=root)
    return result


def _checkpoint_args(path: Path) -> list[dict[str, Any]]:
    value = _read_object(path)
    rows = value.get("checkpoints", value.get("items"))
    if not isinstance(rows, list):
        raise RegistrationError("checkpoint inventory JSON needs a checkpoints list")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    allocate = commands.add_parser("allocate", help="create fresh bank and collision audit")
    allocate.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    allocate.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    resize = commands.add_parser("resize", help="audit additions while retaining the original bank")
    resize.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    resize.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    resize.add_argument("--out-bank", type=Path, required=True)
    resize.add_argument("--out-audit", type=Path, required=True)
    build = commands.add_parser("build", help="create an unauthorized draft registration")
    build.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    build.add_argument("--out", type=Path, default=DEFAULT_REGISTRATION)
    build.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    build.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    build.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    build.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    build.add_argument("--operating-characteristic", type=Path, default=DEFAULT_OC)
    build.add_argument("--checkpoints", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate draft or ready package")
    validate.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    validate.add_argument("--ready", action="store_true")
    validate.add_argument("--precheck", type=Path)
    ratify = commands.add_parser(
        "ratify", help="create the ready registration after observed evidence"
    )
    ratify.add_argument("--draft", type=Path, default=DEFAULT_REGISTRATION)
    ratify.add_argument("--out", type=Path, required=True)
    ratify.add_argument("--source-commit", required=True)
    ratify.add_argument("--trace-receipt", type=Path, required=True)
    ratify.add_argument("--operator", required=True)
    ratify.add_argument("--acknowledge-near-boundary-precedence", action="store_true")
    args = parser.parse_args()
    if args.command == "allocate":
        payload, audit = write_allocation(bank_path=args.bank, audit_path=args.audit)
        print(
            json.dumps(
                {"bank_rows": len(payload["terminal_rows"]), "seed_count": len(audit["seeds"])}
            )
        )
    elif args.command == "resize":
        payload, audit = resize_allocation(bank_path=args.bank, audit_path=args.audit)
        _write_new(args.out_bank, payload)
        _write_new(args.out_audit, audit)
        print(json.dumps({"retained": 113, "added": 64, "total": len(audit["seeds"])}))
    elif args.command == "build":
        value = build_draft(
            study_root=args.study_root,
            profile=args.profile,
            bank_path=args.bank,
            audit_path=args.audit,
            preregistration=args.preregistration,
            operating_characteristic=args.operating_characteristic,
            checkpoints=_checkpoint_args(args.checkpoints),
        )
        _write_new(args.out, value)
        print(json.dumps({"registration": str(args.out), "authorized": False}))
    elif args.command == "validate":
        precheck = _read_object(args.precheck) if args.precheck else None
        print(
            json.dumps(
                validate_registration(
                    _read_object(args.registration), ready=args.ready, precheck=precheck
                )
            )
        )
    else:
        result = ratify_registration(
            _read_object(args.draft),
            source_commit=args.source_commit,
            trace_receipt=args.trace_receipt,
            operator=args.operator,
            acknowledge_near_boundary_precedence=args.acknowledge_near_boundary_precedence,
        )
        _write_new(args.out, result)
        print(json.dumps({"registration": str(args.out), "authorized": True}))


if __name__ == "__main__":
    main()
