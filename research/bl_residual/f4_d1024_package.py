"""Prepare, seal and publish the registered F4 D1024 fit without reporting outcomes.

This mirrors ``f4_d256_package`` exactly where F4_REGISTRATION binds it: the same
wrapper-only life catalog (``_life_catalog``), the same ancestry allocator with
the frozen D256 allocation as its append-only prior (R2: nested prefixes, the
same 64-life reserve), the same strict training-table conversion (``_table``),
the same metadata-archive transport, canary-before-main and doubled caps (R7).

It consumes only *published* F0 union ledgers (D256 and the final D1024 union)
and the Codex-V accepted D1024 staging inputs. It never invokes ``fit_residual``,
a held-out/reserve evaluator, or an aggregate statistic. Fit execution is the
unchanged ``job.run`` on L4; R-A stays the separate joint ``funding.main`` look.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_residual import f4_d256_package as d256
from research.bl_residual.data import DOSES, allocate_lives
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import (
    B8_SHA,
    F4_REGISTRATION_SHA,
    NAMESPACE,
    QUALIFICATION_RECORD,
    QUALIFIED_TREE,
    allocation_hash,
    canonical_sha,
    empty_ledger,
    read_ledger,
    require_judgment,
)

ROOT = d256.ROOT
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
F4 = ART / "big-lever/exp/f4"
DOSE = 1024
INDEX = DOSES.index(DOSE)
REQUIREMENTS = d256.REQUIREMENTS
SOURCE_PATHS = d256.SOURCE_PATHS
# The D256 entrypoint is dose-agnostic: it unpacks the bound archive and runs
# job.py main, or the excluded canary. Reusing it keeps the canary family exact.
SCRIPT = "research/bl_residual/f4_d256_colab.sh"
D256_UNION_PATH = d256.PUBLICATION_PATH
D256_UNION_SHA = d256.UNION_SHA
D1024_UNION_PATH = "research/bl_labels/ledger/D1024_UNION_FINAL_20261024.json"
D256_FIT_PUBLICATION_PATH = d256.FIT_PUBLICATION_PATH
FIT_PUBLICATION_PATH = "research/bl_residual/ledger/F4_D1024_FIT_PHASE_20261024.json"
STAGING = F4 / "d1024-metadata-staging-03/inputs"
D256_PACK = F4 / "d256-metadata-pack-02"
MAIN_OUTPUT = "d1024-output-l4-01"
CANARY_OUTPUT = "canary-output-l4-01"
PREFLIGHT_SCHEMA = "bl-residual-f4-d1024-nofit-preflight/v1"
PREFLIGHT_FIXTURES = {
    "tests/test_bl_residual_factory_masks.py",
    "tests/test_bl_residual_f4_d256_package.py",
    "tests/test_bl_residual_f4_d1024_package.py",
    "research/bl_labels/test_factory_executable.py",
}
# Fit-series accounting (R7: whole series 10 / 20 u, first cap wins). Prior F4
# fit-series dispatches and their frozen liabilities (unit_cap + reserve):
# bl-f4-d256-canary-02 1.2, bl-f4-d256-metadata-canary-02 1.2,
# bl-f4-d256-metadata-fit-02 4.8; bl-f4-d1024-canary-01/-02 1.2 each.
# Both failed D1024 attempts remain charged operational history. Receipt SHA-256
# identities are in colab_cli/STAGING_REPAIR_20261024.md; no scientific look.
PRIOR_SERIES_LIABILITY_UNITS = 9.6
PRIOR_SERIES_WALL_SECONDS = (
    428.9102839999832
    + 335.9559656250058
    + 676.4081622079248
    + 346.1885202090489
    + 464.9269054160686
)
# Transport ruling JUDGMENT_D1024_FIT_TRANSPORT_20261024.md (sha 4beccde2...8618):
# the training data and metadata.tar stay qualified private uploads; only b8.pth
# moves to its existing owner-supplied public release URL (same bytes, same dest).
PUBLIC_INPUTS = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
PUBLIC_INPUTS_STATUS = "PUBLIC_RELEASE_VERIFIED_BY_OWNER_ADDENDUM"
B8_PUBLIC_ASSET = "B8_s3_final.pth"
# Historical upload model retained only as a diagnostic.  It is explicitly not
# an admission clock: the prospective ruling freezes the observed replacement
# acceptance timings below, while canary/acceptance enforce their measured
# transfer and whole-staging gates independently.
CHUNK_BYTES = 2 * 1024**2
CHUNK_SECONDS = 2.8
MULTI_CHUNK_OVERHEAD_SECONDS = 16.0
SINGLE_CHUNK_OVERHEAD_SECONDS = 4.0
TRANSPORT_FILE_SECONDS = {"bootstrap.sh": 4.0, "source.py": 4.0}
HISTORICAL_OBSERVED_TRANSFER_SECONDS = 221.68512525002006
HISTORICAL_OBSERVED_STAGING_SECONDS = 235.52283416606952
TRANSFER_GATE_SECONDS = 300.0
STAGING_GATE_SECONDS = 300.0


def _attempt(value: str) -> str:
    """Accept the create-only, two-digit transport-attempt suffix."""
    if not re.fullmatch(r"[0-9]{2}", value):
        raise ValueError("D1024 attempt must be a two-digit suffix")
    return value


def output_names(attempt: str) -> tuple[str, str]:
    """Return the fresh main/canary output names for one transport attempt."""
    attempt = _attempt(attempt)
    return f"d1024-output-l4-{attempt}", f"canary-output-l4-{attempt}"


def require_repaired_transport(repo: Path, record: Path) -> dict[str, Any]:
    """Require source-bound acceptance under the one D1024 transfer authority."""
    from research.colab_cli.acceptance import (
        D1024_F4_INVENTORY_SHA256,
        verify_qualified_transport_record,
    )
    from research.colab_cli.transfer_policy import RULING_SHA256, authority_binding

    qualified = verify_qualified_transport_record(
        repo,
        record,
        required_inventory_sha256=D1024_F4_INVENTORY_SHA256,
        reject_tree=QUALIFIED_TREE,
        required_transfer_authority_sha256=RULING_SHA256,
    )
    if qualified.get("transfer_exception") != authority_binding("acceptance"):
        raise ValueError("qualified transport lacks the D1024 acceptance transfer authority")
    return qualified


def require_publication(local: Path, relative: str, expected_sha: str | None) -> str:
    """Local ledger bytes must equal checkout and freshly fetched origin/main bytes."""
    value = digest(local)
    if expected_sha is not None and value != expected_sha:
        raise ValueError(f"foreign published ledger: {relative}")
    published = subprocess.check_output(
        ["git", "show", f"origin/main:{relative}"], cwd=ROOT, timeout=30
    )
    if published != local.read_bytes():
        raise ValueError(f"local ledger differs from origin/main: {relative}")
    if (ROOT / relative).read_bytes() != published:
        raise ValueError(f"checkout ledger differs from origin/main: {relative}")
    return value


def staged_inputs(staging: Path) -> dict[str, Any]:
    """Verify the accepted D1024 staging bytes and their fault-free D256 predecessor."""
    from research.bl_residual.job import _validate_staging

    verify_export(staging / "control.json")
    control = json.loads((staging / "control.json").read_text())
    recipe_sha = canonical_sha(dataclasses.asdict(FitRecipe()))
    names = {
        "b8.pth",
        "F4_REGISTRATION.md",
        "judgment.json",
        "life-allocation.json",
        "d256-fit.json",
        "d256-control.json",
    }
    if (
        control.get("schema") != "bl-residual-control/v1"
        or control.get("synthetic") is not False
        or control.get("namespace") != NAMESPACE
        or control.get("series_id") != NAMESPACE
        or control.get("doses") != [DOSE]
        or control.get("recipe_sha256") != recipe_sha
        or control.get("stage_receipts")
        != {"D256": "d256-fit.json", "D256_CONTROL": "d256-control.json"}
        or set(control.get("input_hashes", {})) != names
        or allocation_hash(control) != control.get("allocation")
    ):
        raise ValueError("foreign or unaccepted D1024 staging control")
    payloads: dict[str, bytes] = {}
    for name in sorted(names):
        source = staging / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"D1024 staging input is not a regular file: {name}")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != control["input_hashes"][name]:
            raise ValueError(f"D1024 staging input differs from sealed control: {name}")
        if verify_export(source) != control["input_hashes"][name]:
            raise ValueError(f"D1024 staging sidecar differs: {name}")
        payloads[name] = payload
    if hashlib.sha256(payloads["b8.pth"]).hexdigest() != B8_SHA:
        raise ValueError("B8 identity differs")
    if hashlib.sha256(payloads["F4_REGISTRATION.md"]).hexdigest() != F4_REGISTRATION_SHA:
        raise ValueError("staged registration differs from F4_REGISTRATION")
    if (ROOT / "research/bl_residual/F4_REGISTRATION.md").read_bytes() != payloads[
        "F4_REGISTRATION.md"
    ]:
        raise ValueError("source registration differs from staged registration")
    prior_control = json.loads(payloads["d256-control.json"])
    if (
        prior_control.get("input_hashes", {}).get("life-allocation.json")
        != control["input_hashes"]["life-allocation.json"]
        or prior_control.get("reserve_allocation_sha256") != control["reserve_allocation_sha256"]
    ):
        raise ValueError("staged D256 allocation differs from its sealed fit control")
    # Registered R6 predecessor stop: D1024 needs the passed D256 stage receipt.
    _validate_staging(staging, control, [DOSE], recipe_sha, set())
    return {"control": control, "payloads": payloads, "recipe_sha256": recipe_sha}


def require_d256_fit_publication(stage_fit: dict[str, Any]) -> None:
    """The staged D256 fit identity must be the published fit-phase unit 0."""
    publication = ROOT / D256_FIT_PUBLICATION_PATH
    require_publication(publication, D256_FIT_PUBLICATION_PATH, None)
    units = read_ledger(publication, stage_fit["namespace"], stage_fit["allocation"])
    if set(units) != {0} or units[0]["sha256"] != stage_fit["fit_sha256"]:
        raise ValueError("staged D256 fit differs from the published D256 fit phase")


def nested_allocation(catalog: list[dict[str, Any]], prior: dict[str, Any]) -> dict[str, Any]:
    """Freeze D1024 with the D256 allocation as append-only prior (R2)."""
    allocation = allocate_lives(catalog, prior=prior)
    by_id = {row["life_id"]: row for row in catalog}
    old = {row["life_id"] for row in prior["catalog"]}
    if any(by_id.get(row["life_id"]) != row for row in prior["catalog"]):
        raise ValueError("D256 catalog changed in the D1024 union")
    if allocation["doses"]["256"] != prior["doses"]["256"]:
        raise ValueError("D256 dose allocation is not reproduced exactly")
    reserve = allocation["doses"][str(DOSE)]["heldout_lives"]
    if reserve != prior["doses"]["256"]["heldout_lives"]:
        raise ValueError("D1024 reserve differs from the fixed D256 reserve")
    if not allocation["doses"][str(DOSE)]["ready"]:
        report = allocation["allocation_report"]
        raise ValueError(
            "INSUFFICIENT: D1024 selected support unavailable "
            f"(available fit rescue lives {report['available_fit_rescue_lives']}, "
            f"required {DOSE - 64})"
        )
    train = allocation["doses"][str(DOSE)]["train_lives"]
    if train[:192] != prior["doses"]["256"]["train_lives"]:
        raise ValueError("D256 fit lives are not the D1024 prefix")
    if any(allocation["folds"][life] == 0 for life in train):
        raise ValueError("fold-zero life in the D1024 fit partition")
    new = [row for row in catalog if row["life_id"] not in old]
    last = max(row["allocation_index"] for row in prior["catalog"])
    if any(row["allocation_index"] <= last for row in new):
        raise ValueError("D1024 tranche precedes the frozen D256 prefix")
    return allocation


def _metadata_archive_sha256(inventory: list[dict[str, Any]]) -> str:
    return d256._metadata_archive_sha256(inventory)


def manifests(
    output: Path,
    commit: str,
    inventory: list[dict],
    *,
    attempt: str = "01",
    qualified_transport: dict[str, Any] | None = None,
) -> tuple[dict, dict]:
    """Freeze D1024/canary reservations inside the registered 20-unit fit series.

    Base envelopes equal D256's (the 2,000-update recipe does not grow with the
    dose); effective = 2 x base, applied once.
    """
    attempt = _attempt(attempt)
    from research.colab_cli.transfer_policy import authority_binding

    qualified_transport = qualified_transport or {"tree_hash": QUALIFIED_TREE}
    main_output, canary_output = output_names(attempt)
    reserved = 6
    after = PRIOR_SERIES_LIABILITY_UNITS + reserved
    main_template = {
        "job_id": f"f4-d1024-fit-{commit[:8]}-{attempt}",
        "commit": commit,
        "script": SCRIPT,
        "source_paths": SOURCE_PATHS,
        "device": "cuda:0",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 1, "est_minutes": 30},
        "timeout_minutes": 120,
        "output_dir": str((output / main_output).resolve()),
        "inputs": inventory,
        "required_exports": [
            "fit/allocation.json",
            f"fit/fit-{INDEX:04d}.pth",
            "fit/receipt.json",
        ],
        "env": {
            "F4_MODE": "main",
            "F4_DEVICE": "cuda:0",
            "F4_DOSE": str(DOSE),
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS,
            "METADATA_ARCHIVE_SHA256": _metadata_archive_sha256(inventory),
        },
        "budget": {
            "unit_cap": 4,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0.8,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "complete-dose-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "max_export_bytes": 128 * 1024**2,
        # The ruling is deliberately explicit and narrowly scoped.  Transport
        # consumers validate this binding plus the frozen inventory and recipe.
        "transfer_exception": authority_binding("fit"),
        "f4": {
            "dose": DOSE,
            "fit_series_base_units": 10,
            "fit_series_effective_units": 20,
            "registered_updates": 2000,
            "qualified_tree": qualified_transport["tree_hash"],
            "base_units": 2,
            "effective_units": 4,
            "base_reserve_units": 0.4,
            "effective_reserve_units": 0.8,
            "base_timeout_minutes": 60,
            "effective_timeout_minutes": 120,
            "estimate_status": "CONSERVATIVE_ENVELOPE_D256_MAIN_WALL_676S",
            "prior_series_liability_units": PRIOR_SERIES_LIABILITY_UNITS,
            "prior_series_wall_seconds": PRIOR_SERIES_WALL_SECONDS,
            "d1024_total_reserved_units_including_canary": reserved,
            "series_liability_after_d1024_units": after,
            "remaining_series_uncommitted_units": 20 - after,
            "canary_required": True,
            "automatic_completion_retries": False,
        },
    }
    canary_manifest = {
        **main_template,
        "job_id": f"f4-d1024-canary-{commit[:8]}-{attempt}",
        "output_dir": str((output / canary_output).resolve()),
        "timeout_minutes": 20,
        "budget": {
            "unit_cap": 1,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0.2,
            "other_assignments": {},
        },
        "requirements": {**main_template["requirements"], "est_minutes": 10},
        "env": {
            **main_template["env"],
            "F4_MODE": "canary",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
        },
        "canary": {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT},
        "f4": {
            **main_template["f4"],
            "base_units": 0.5,
            "effective_units": 1,
            "base_reserve_units": 0.1,
            "effective_reserve_units": 0.2,
            "base_timeout_minutes": 10,
            "effective_timeout_minutes": 20,
        },
    }
    return main_template, canary_manifest


def public_b8_row(path: Path) -> dict[str, Any]:
    """Replace the staged b8.pth upload by its registered public URL, or refuse.

    The staged bytes must equal the owner-released asset exactly (size and
    SHA-256) and be B8 s3. Only this one input may become a URL.
    """
    catalog = json.loads(PUBLIC_INPUTS.read_text())
    assets = [row for row in catalog.get("assets", []) if row.get("name") == B8_PUBLIC_ASSET]
    if catalog.get("status") != PUBLIC_INPUTS_STATUS or len(assets) != 1:
        raise ValueError("B8 public asset is not an owner-verified release")
    asset = assets[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError("staged b8.pth is not a regular file")
    checksum, size = digest(path), path.stat().st_size
    if (
        checksum != asset.get("sha256")
        or size != asset.get("size_bytes")
        or checksum != B8_SHA
        or not str(asset.get("url", "")).startswith("https://")
    ):
        raise ValueError("staged b8.pth differs from the public B8 asset; refusing URL transport")
    return {"dest": "b8.pth", "url": asset["url"], "sha256": checksum, "size_bytes": size}


def transport_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exactly one row (b8.pth) becomes HTTPS; every other input stays an upload."""
    rows = []
    replaced = 0
    for row in raw:
        if row["dest"] == "b8.pth":
            public = public_b8_row(Path(row["path"]))
            if public["sha256"] != row["sha256"]:
                raise ValueError("b8.pth changed while preparing transport")
            rows.append(public)
            replaced += 1
        else:
            rows.append(dict(row))
    if replaced != 1:
        raise ValueError("exactly one staged b8.pth input is required")
    return rows


def _row_seconds(size: int) -> tuple[int, float]:
    chunks = max(1, -(-size // CHUNK_BYTES))
    overhead = MULTI_CHUNK_OVERHEAD_SECONDS if chunks > 1 else SINGLE_CHUNK_OVERHEAD_SECONDS
    return chunks, overhead + chunks * CHUNK_SECONDS


def upload_projection(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an obsolete diagnostic forecast; never use it as a transport gate."""
    rows = [
        {"dest": name, "kind": "transport", "seconds": seconds}
        for name, seconds in TRANSPORT_FILE_SECONDS.items()
    ]
    for row in inventory:
        if "path" not in row:
            rows.append({"dest": row["dest"], "kind": "https", "seconds": 0.0})
            continue
        size = Path(row["path"]).stat().st_size
        chunks, seconds = _row_seconds(size)
        rows.append(
            {
                "dest": row["dest"],
                "kind": "upload",
                "bytes": size,
                "chunks": chunks,
                "seconds": seconds,
            }
        )
    total = sum(row["seconds"] for row in rows)
    return {
        "model": (
            "OBSOLETE_DIAGNOSTIC: per-file 16 s multi-chunk / 4 s single-chunk "
            "+ 2.8 s per 2 MiB chunk"
        ),
        "rows": rows,
        "upload_bytes": sum(row.get("bytes", 0) for row in rows),
        "projected_transfer_seconds": total,
        "forecast_status": "DIAGNOSTIC_ONLY_NOT_AN_ADMISSION_GATE",
        "historical_observed_transfer_seconds": HISTORICAL_OBSERVED_TRANSFER_SECONDS,
        "historical_observed_whole_staging_seconds": HISTORICAL_OBSERVED_STAGING_SECONDS,
        "transfer_gate_seconds": TRANSFER_GATE_SECONDS,
        "whole_staging_gate_seconds": STAGING_GATE_SECONDS,
    }


def _require_source(commit: str) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
        raise ValueError("prepare from the exact pinned source checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("prepare from a clean pinned source checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True
    )
    return commit


def _require_preflight(receipt: Path, commit: str) -> dict[str, Any]:
    verify_export(receipt)
    value = json.loads(receipt.read_text())
    if (
        value.get("schema") != PREFLIGHT_SCHEMA
        or value.get("status") != "PASS"
        or value.get("commit") != commit
        or value.get("fits_run") is not False
        or value.get("study_outcomes_opened") is not False
        or set(value.get("fixtures", {})) != PREFLIGHT_FIXTURES
        or any(digest(ROOT / name) != sha for name, sha in value["fixtures"].items())
    ):
        raise ValueError("same-source no-fit D1024 package preflight required before labels")
    return value


def _prepare(
    output: Path,
    *,
    d256_ledger: Path,
    d1024_ledger: Path,
    d1024_union_sha: str,
    staging: Path,
    commit: str,
    preflight_receipt: Path,
    attempt: str = "01",
) -> dict[str, Any]:
    """Write a fresh D1024-only package; no reserve or archive rows are uploaded."""
    d256.require_window_closed()
    require_publication(d256_ledger, D256_UNION_PATH, D256_UNION_SHA)
    require_publication(d1024_ledger, D1024_UNION_PATH, d1024_union_sha)
    if output.exists():
        raise FileExistsError(output)
    commit = _require_source(commit)
    preflight = _require_preflight(preflight_receipt, commit)
    from tools.ops.import_preflight import check as imports
    from tools.ops.qualification import check as qualification

    qualified_transport = require_repaired_transport(ROOT, Path(QUALIFICATION_RECORD))
    if qualification(ROOT, Path(QUALIFICATION_RECORD), commit) != qualified_transport["tree_hash"]:
        raise ValueError("F4 transport qualification differs")
    import_receipt = imports(
        ROOT,
        commit,
        SOURCE_PATHS,
        REQUIREMENTS,
        ["research.bl_residual.job", "research.bl_residual.f4_canary"],
    )
    staged = staged_inputs(staging)
    payloads, recipe_sha = staged["payloads"], staged["recipe_sha256"]
    require_d256_fit_publication(json.loads(payloads["d256-fit.json"]))
    require_judgment(staging / "judgment.json", staging / "F4_REGISTRATION.md", recipe_sha)
    lives = d256._life_catalog(d256_ledger) + d256._life_catalog(d1024_ledger)
    catalog = [
        {
            "life_id": str(life["life_id"]),
            "parent_world": str(life["parent_world"]),
            "stream": str(life["stream"]),
            "rescue_bearing": bool(life["rescue_bearing"]),
            "allocation_index": int(life["allocation_index"]),
        }
        for life in lives
    ]
    prior = json.loads(payloads["life-allocation.json"])
    allocation = nested_allocation(catalog, prior)
    train_lives = allocation["doses"][str(DOSE)]["train_lives"]
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("b8.pth", "F4_REGISTRATION.md", "judgment.json", "d256-fit.json"):
        export_bytes(inputs / name, payloads[name])
    export_bytes(inputs / "d256-control.json", payloads["d256-control.json"])
    d256._copy(d1024_ledger, inputs / "factory-verified-ledger.json")
    d256._copy(d256_ledger, inputs / "d256-factory-verified-ledger.json")
    export_json(inputs / "life-allocation.json", allocation)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **d256._table(
            lives,
            set(train_lives),
            {allocation["parents"][life] for life in train_lives},
        ),
    )
    export_bytes(inputs / f"train-{DOSE}.npz", buffer.getvalue())
    stage_names = {"D256": "d256-fit.json", "D256_CONTROL": "d256-control.json"}
    input_hashes = {
        path.name: digest(path)
        for path in sorted(inputs.iterdir())
        if not path.name.endswith((".sha256", ".complete"))
    }
    control = {
        "schema": "bl-residual-control/v1",
        "synthetic": False,
        "namespace": NAMESPACE,
        "doses": [DOSE],
        "recipe_sha256": recipe_sha,
        "registration_sha256": F4_REGISTRATION_SHA,
        "factory_union_ledger_sha256": digest(d1024_ledger),
        "d256_factory_union_ledger_sha256": digest(d256_ledger),
        "input_hashes": input_hashes,
        "series_id": NAMESPACE,
        "reserve_allocation_sha256": canonical_sha(allocation["doses"][str(DOSE)]["heldout_lives"]),
        "stage_receipts": stage_names,
    }
    if control["reserve_allocation_sha256"] != staged["control"]["reserve_allocation_sha256"]:
        raise ValueError("D1024 reserve identity differs from the staged series")
    control["allocation"] = allocation_hash(control)
    export_json(inputs / "verified-ledger.json", empty_ledger(NAMESPACE, control["allocation"]))
    control["ledger_sha256"] = digest(inputs / "verified-ledger.json")
    export_json(inputs / "control.json", control)
    raw_inventory = [
        {"path": str(path.resolve()), "dest": path.name, "sha256": digest(path)}
        for path in sorted(inputs.iterdir())
    ]
    from research.metadata_archive import bundle_inputs

    inventory = bundle_inputs(transport_rows(raw_inventory), inputs / "metadata.tar")
    if [row["dest"] for row in inventory if "url" in row] != ["b8.pth"]:
        raise ValueError("only b8.pth may use public HTTPS transport")
    attempt = _attempt(attempt)
    main_template, canary_manifest = manifests(
        output, commit, inventory, attempt=attempt, qualified_transport=qualified_transport
    )
    export_json(output / "main-template.json", main_template)
    export_json(output / "canary.json", canary_manifest)
    export_json(output / "clean-import-preflight.json", import_receipt)
    export_json(output / "numerical-preflight.json", preflight)
    from research.bl_residual.job import preflight_main_dispatch
    from research.colab_cli.supervisor import validate

    export_json(output / "main-input-preflight.json", preflight_main_dispatch(inputs))
    validate(canary_manifest, ROOT)
    projection = upload_projection(inventory)
    export_json(
        output / "pack.json",
        {
            "schema": "bl-residual-f4-d1024-pack/v1",
            "commit": commit,
            "attempt": attempt,
            "qualified_transport": qualified_transport,
            "dose": DOSE,
            "factory_union_ledger_sha256": digest(d1024_ledger),
            "d256_factory_union_ledger_sha256": digest(d256_ledger),
            "staging_control_sha256": digest(staging / "control.json"),
            "life_allocation_sha256": digest(inputs / "life-allocation.json"),
            "reserve_allocation_sha256": control["reserve_allocation_sha256"],
            "main_template_sha256": digest(output / "main-template.json"),
            "canary_manifest_sha256": digest(output / "canary.json"),
            "upload_projection": projection,
            "outcome_reporting": "none",
        },
    )
    return {
        "pack": str(output),
        "canary_manifest": str(output / "canary.json"),
        "main_template": str(output / "main-template.json"),
        "factory_union_ledger_sha256": digest(d1024_ledger),
        "upload_projection": projection,
    }


def prepare(output: Path, **kwargs: Any) -> dict[str, Any]:
    """Hold the Mac resource boundary during the mechanical conversion."""
    d256.require_window_closed()
    from research.mac_lease import held

    with held(
        "f4-d1024-input-preparation", "mechanical partition; no fits or reserve look", seconds=2400
    ):
        return _prepare(output, **kwargs)


def seal_main(template: Path, canary_receipt: Path, output: Path) -> dict[str, Any]:
    """Bind a passed same-source D1024 canary receipt into a fresh main manifest."""
    if output.exists():
        raise FileExistsError(output)
    verify_export(template)
    value = json.loads(template.read_text())
    if (
        value.get("env", {}).get("F4_MODE") != "main"
        or "canary" in value
        or value.get("f4", {}).get("dose") != DOSE
    ):
        raise ValueError("only the unbound D1024 main template may be sealed")
    value["canary"] = {
        "receipt": {"path": str(canary_receipt.resolve()), "sha256": digest(canary_receipt)}
    }
    from research.colab_cli.supervisor import validate

    validate(value, ROOT)
    export_json(output, value)
    return {"manifest": str(output), "sha256": digest(output)}


def publish_fit(pack: Path, output: Path) -> dict[str, Any]:
    """Publish path/size/hash only, before any fit metadata or reserve look."""
    from research.bl_residual.protocol import build_ledger

    control = json.loads((pack / "inputs/control.json").read_text())
    if control.get("doses") != [DOSE]:
        raise ValueError("HOLD: not a D1024 fit pack")
    pack_record = json.loads((pack / "pack.json").read_text())
    _main_output, _canary_output = output_names(str(pack_record.get("attempt", "01")))
    run_output = pack / _main_output
    receipt = json.loads((run_output / "receipt.json").read_text())
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("error")
        or receipt.get("cleanup_error")
    ):
        raise ValueError("HOLD: D1024 supervisor completion and release not verified")
    if any((run_output / "exports/fit").glob("fault-*.json")):
        raise ValueError("HOLD: D1024 sealed a KILL_FIT fault; never refit")
    value = build_ledger(
        [run_output / "exports/fit"], output, control["namespace"], control["allocation"]
    )
    if [row["index"] for row in value["units"]] != [INDEX]:
        raise ValueError("HOLD: D1024 complete immutable fit unit required; never refit a fault")
    return {"path": str(output), "sha256": digest(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--d256-ledger", type=Path)
    parser.add_argument("--d1024-ledger", type=Path)
    parser.add_argument("--d1024-union-sha256")
    parser.add_argument("--staging", type=Path, default=STAGING)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--attempt", default="01")
    parser.add_argument("--publish-fit", type=Path, metavar="PACK")
    parser.add_argument("--seal-main-template", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--main-manifest-out", type=Path)
    args = parser.parse_args()
    if args.publish_fit:
        if args.out is None:
            parser.error("--publish-fit requires --out")
        print(json.dumps(publish_fit(args.publish_fit, args.out)))
        return
    if args.seal_main_template:
        if not args.canary_receipt or not args.main_manifest_out:
            parser.error("--seal-main-template requires --canary-receipt and --main-manifest-out")
        print(
            json.dumps(
                seal_main(args.seal_main_template, args.canary_receipt, args.main_manifest_out)
            )
        )
        return
    required = (
        args.out,
        args.d256_ledger,
        args.d1024_ledger,
        args.d1024_union_sha256,
        args.preflight_receipt,
    )
    if not all(required):
        parser.error(
            "--out, --d256-ledger, --d1024-ledger, --d1024-union-sha256 and "
            "--preflight-receipt required"
        )
    print(
        json.dumps(
            prepare(
                args.out,
                d256_ledger=args.d256_ledger,
                d1024_ledger=args.d1024_ledger,
                d1024_union_sha=args.d1024_union_sha256,
                staging=args.staging,
                commit=args.commit,
                preflight_receipt=args.preflight_receipt,
                attempt=args.attempt,
            )
        )
    )


if __name__ == "__main__":
    main()
