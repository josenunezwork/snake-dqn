"""Prospective chunk, envelope, provenance, transport and one-look guards."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest

from research.bl_served_search import chunks, evidence_v2, pricing_v2, ruler, transport_v2
from research.bl_served_search.ceiling import validate_bank
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.finalize_v2 import finalize
from research.bl_served_search.provenance_v2 import (
    CHUNKS_PATH,
    LEGACY_CHUNKS_SHA256,
    compatible_source,
)
from research.bl_served_search.test_ruler import _fixtures


def sizing_receipt(rates: dict[str, float] | None = None) -> tuple[dict, dict]:
    fixtures = {**_fixtures(), "source_identity": ruler.source_identity()}
    rates = rates or {
        "frozen": 0.0550080940374578,
        "mixed": 0.10373858917357195,
        "scripted": 1.5012383309424502,
    }
    identity = [
        {key: row[key] for key in ("fixture_id", "mix", "target_length", "root_sha256")}
        for row in ruler.validate_fixtures(fixtures)
    ]
    mac = {
        "schema": ruler.RECEIPT_SCHEMA,
        "status": "RULER_PASS",
        "platform": {"label": "mac", "system": "Darwin"},
        "source_identity": ruler.source_identity(),
        "min_batches_per_fixture": 30,
        "fixture_sha256": ruler.sha256_bytes(ruler.canonical(fixtures).encode()),
        "fixture_identity": identity,
        "workers": 1,
        "operator_spec": "research.bl_served_search.ruler_operator:create_operator",
        "completed_batches": 270,
        "elapsed_seconds": 270,
        "aggregate_batches_per_second": 1,
        "records": [
            {
                "fixture_id": row["fixture_id"],
                "batches": [
                    {
                        "worker_wall_seconds": 1 / rates[row["mix"]],
                        "semantic": {"fired": True},
                        "stage_seconds": {},
                    }
                    for _ in range(30)
                ],
            }
            for row in fixtures["fixtures"]
        ],
    }
    return mac, fixtures


def test_eight_chunks_cover_exact_original_bank_without_overlap() -> None:
    source = Path(__file__).with_name("BANK_L2.json")
    original_bytes = source.read_bytes()
    rows = validate_bank(json.loads(original_bytes))
    selected = [chunks.chunk_rows(rows, index) for index in range(8)]
    assert all(len(part) == 20 for part in selected)
    assert len({row["world_id"] for part in selected for row in part}) == 160
    assert {row["world_id"] for part in selected for row in part} == {
        row["world_id"] for row in rows
    }
    assert len({chunks.world_ids_sha256(part) for part in selected}) == 8
    assert source.read_bytes() == original_bytes
    for index in (-1, 8):
        with pytest.raises(ValueError):
            chunks.chunk_rows(rows, index)


def test_actual_mac_fallback_cannot_be_clipped_to_fit() -> None:
    mac, fixtures = sizing_receipt()
    value = pricing_v2.forecast(mac, fixtures)
    assert value["status"] == "NOT_DISPATCHABLE"
    assert value["timeout_minutes"] > 720
    assert value["liability_plus_reserve_units"] > 200
    assert value["timeout_minutes"] == 2 * value["conservative_allocation_minutes"]
    assert value["reserve_units"] == 40
    assert not pricing_v2.fallback_open(
        value, pricing_v2.FALLBACK_NOT_BEFORE - timedelta(seconds=1)
    )
    assert pricing_v2.fallback_open(value, pricing_v2.FALLBACK_NOT_BEFORE)


def test_qualified_l4_can_fit_repriced_envelope_without_old_scalar_bar() -> None:
    mac, fixtures = sizing_receipt()
    l4 = copy.deepcopy(mac)
    l4.update(
        platform={"label": "l4", "system": "Linux"},
        workers=12,
        hardware={"sha256": "a" * 64, "reported": {"hardware": {"kind": "L4"}}, "vcpu_count": 12},
    )
    for row in l4["records"]:
        for batch in row["batches"]:
            batch["worker_wall_seconds"] = 0.5
    # Aggregate R=1 is below the old2.18 rule; per-mix price replaces that rule.
    value = pricing_v2.forecast(mac, fixtures, l4)
    assert value["status"] == "FITS"
    assert 2 * value["conservative_allocation_minutes"] <= value["timeout_minutes"] <= 720
    assert value["liability_plus_reserve_units"] <= 200
    l4["records"][0]["batches"][0]["semantic"]["fired"] = False
    with pytest.raises(ValueError):
        pricing_v2.forecast(mac, fixtures, l4)


def test_changed_operator_or_forged_worker_timings_cannot_size() -> None:
    mac, fixtures = sizing_receipt()
    mac["source_identity"]["research/bl_served_search/ruler_operator.py"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        pricing_v2.forecast(mac, fixtures)
    mac, fixtures = sizing_receipt()
    mac["records"][0]["batches"][0]["worker_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="timing"):
        pricing_v2.forecast(mac, fixtures)


def test_provenance_exception_is_only_exact_legacy_partition() -> None:
    current = {CHUNKS_PATH: "new", "operator.py": "fixed"}
    prior = {CHUNKS_PATH: LEGACY_CHUNKS_SHA256, "operator.py": "fixed"}
    assert compatible_source(prior, current)
    assert not compatible_source({**prior, "operator.py": "drift"}, current)
    assert not compatible_source({**prior, CHUNKS_PATH: "other"}, current)


def test_evidence_single_upload_roundtrip_and_hash_refusal(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    export_json(source, {"status": "CLEAR"})
    bundle = evidence_v2.pack([("audit.json", source)], tmp_path / evidence_v2.NAME)
    restored = evidence_v2.unpack(bundle, tmp_path / "restored", digest(bundle))
    assert verify_export(restored / "audit.json") == digest(source)
    with pytest.raises(FileExistsError):
        evidence_v2.unpack(bundle, restored, digest(bundle))
    with pytest.raises(ValueError, match="hash"):
        evidence_v2.unpack(bundle, tmp_path / "bad", "0" * 64)


def test_real_transport_exercises_chunk_assembly_and_verification(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.write_bytes(b"x" * (transport_v2.transfers.UPLOAD_CHUNK_BYTES + 11))
    remote = tmp_path / "remote"
    remote.mkdir()
    cli = transport_v2.LocalContents(remote)
    actual = transport_v2.transfers.upload_file(
        cli, source, str(remote / "out"), renew=lambda: None
    )
    assert actual == digest(source) == digest(remote / "out")
    assert cli.contents_calls == 2
    assert cli.rpc_calls >= 3
    assert cli.transfer_records[0]["status"] == "verified"


def test_incomplete_terminal_set_emits_no_partial_scientific_metrics(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    export_json(plan, {"manifests": [{}] * 9, "commit": "a" * 40})
    output = tmp_path / "decision.json"
    value = finalize(plan, output)
    assert value["status"] == "INCONCLUSIVE"
    assert "metrics" not in value
    assert len(value["errors"]) == 8
    with pytest.raises(FileExistsError):
        finalize(plan, output)


def ready_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    from research.bl_served_search import analyze, ceiling, readiness

    mac, fixtures = sizing_receipt({mix: 2 for mix in chunks.CHUNK_MIX_COUNTS})
    files = {
        "bank.json": json.loads(Path(__file__).with_name("BANK_L2.json").read_text()),
        "ruler-mac.json": mac,
        "parity-mac.json": {},
        "ruler-fixtures/ruler-fixtures.json": fixtures,
    }
    for name, value in files.items():
        export_json(tmp_path / name, value)
    export_json(
        tmp_path / "audit.json", {"status": "CLEAR", "bank_sha256": digest(tmp_path / "bank.json")}
    )
    hashes = {name: digest(tmp_path / name) for name in (*files, "audit.json")}
    rows = chunks.chunk_rows(validate_bank(files["bank.json"]), 0)
    plan = {
        "schema": "bl-served-search-plan/v3",
        "dispatchable": True,
        "chunk_count": 8,
        "chunk_index": 0,
        "world_ids": [row["world_id"] for row in rows],
        "world_ids_sha256": chunks.world_ids_sha256(rows),
        "input_hashes": hashes,
        "forecast": pricing_v2.forecast(mac, fixtures),
        "analyzer_sha256": digest(Path(analyze.__file__)),
    }
    assert plan["forecast"]["status"] == "FITS"
    monkeypatch.setattr(pricing_v2, "fallback_open", lambda value: True)
    monkeypatch.setenv("COLAB_CLI_ROOT", str(tmp_path))
    import platform

    parity = {
        "status": "VERIFIED",
        "bank_sha256": digest(tmp_path / "bank.json"),
        "b8_sha256": ceiling.B8_SHA256,
        "source_manifest": ceiling.source_manifest(),
        "mac_reference_sha256": "2" * 64,
        "runtime_verification_sha256": "3" * 64,
        "runtime_identity": {
            "node": platform.node(),
            "machine": platform.machine(),
            "colab_cli_root": str(tmp_path.resolve()),
        },
    }
    export_json(tmp_path / "runtime-parity.json", parity)
    readiness.check(plan, tmp_path, tmp_path / "runtime-parity.json")
    return plan


def test_ready_path_recomputes_price_and_rejects_forged_plan(tmp_path: Path, monkeypatch) -> None:
    from research.bl_served_search.readiness import check

    plan = ready_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="per-assignment"):
        check(plan, tmp_path)
    forged = copy.deepcopy(plan)
    forged["forecast"]["timeout_minutes"] = 1
    with pytest.raises(ValueError, match="differs"):
        check(forged, tmp_path, tmp_path / "runtime-parity.json")
    forged = copy.deepcopy(plan)
    forged["analyzer_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="analyzer"):
        check(forged, tmp_path, tmp_path / "runtime-parity.json")
    alternate = json.loads((tmp_path / "bank.json").read_text())
    alternate["rows"][0]["world_seed"] += 1
    export_json(tmp_path / "alternate.json", alternate)
    # All supplied hashes can agree and still not authorize another bank.
    alternate_bytes = (tmp_path / "alternate.json").read_bytes()
    (tmp_path / "bank.json").write_bytes(alternate_bytes)
    plan["input_hashes"]["bank.json"] = digest(tmp_path / "bank.json")
    with pytest.raises(ValueError):
        check(plan, tmp_path, tmp_path / "runtime-parity.json")


def test_v2_ruler_legacy_cli_boundary_preserves_fixture_hash(tmp_path: Path, monkeypatch) -> None:
    from research.bl_served_search import ruler_v2

    fixtures = {
        **_fixtures(),
        "source_identity": {**ruler.source_identity(), CHUNKS_PATH: LEGACY_CHUNKS_SHA256},
    }
    path = tmp_path / "ruler-fixtures.json"
    export_json(path, fixtures)
    monkeypatch.setattr(ruler, "_load_factory", lambda spec: lambda: object())

    def fake_run(value, operator, **kwargs):
        assert value == fixtures
        return {
            "status": "PRECHECK_ONLY",
            "completed_batches": 9,
            "fixture_sha256": ruler.sha256_bytes(ruler.canonical(value).encode()),
        }

    monkeypatch.setattr(ruler, "run_ruler", fake_run)
    value = ruler_v2.run(
        path,
        tmp_path / "out.json",
        label="mac",
        workers=1,
        batches=1,
        operator_spec="fake",
        smoke=True,
    )
    assert value["fixture_sha256"] == ruler.sha256_bytes(ruler.canonical(fixtures).encode())


def terminal_family(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from research.bl_served_search import analyze
    from research.bl_upload.qualification import _canonical, _sha256_bytes

    bank = root / "evidence/bank.json"
    export_json(bank, json.loads(Path(__file__).with_name("BANK_L2.json").read_text()))
    manifests = [{}]
    for index in range(8):
        directory = root / f"chunk-{index+1:02d}"
        chunk_plan = directory / "plan.json"
        export_json(
            chunk_plan,
            {
                "analyzer_sha256": digest(Path(analyze.__file__)),
                "input_hashes": {"bank.json": digest(bank)},
            },
        )
        output = directory / "runtime-output"
        manifest = {
            "job_id": f"job-{index}",
            "commit": "a" * 40,
            "output_dir": str(output),
            "inputs": [{"dest": "l2-plan.json", "sha256": digest(chunk_plan)}],
        }
        path = directory / "manifest.json"
        export_json(path, manifest)
        transport = directory / "transport-preflight.json"
        export_json(
            transport,
            {"manifest_sha256": _sha256_bytes(_canonical(manifest)), "all_bytes_verified": True},
        )
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        shard = output / f"exports/l2-shard-{index}.json"
        export_json(shard, {"test_only": index})
        supervisor = output / "receipt.json"
        supervisor.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "stop_verified": True,
                    "manifest_sha256": digest(output / "manifest.json"),
                    "job_id": manifest["job_id"],
                    "commit": manifest["commit"],
                    "endpoint": "test",
                    "generation": "g",
                    "terminal": {"status": "completed", "generation": "g", "returncode": 0},
                    "exports": [
                        {"path": shard.name, "sha256": digest(shard), "size": shard.stat().st_size}
                    ],
                }
            )
        )
        export_json(
            directory / "receipt.json",
            {
                "schema": "bl-served-search-terminal-chunk/v2",
                "chunk_index": index,
                "supervisor_receipt": str(supervisor),
                "supervisor_sha256": digest(supervisor),
                "shard": str(shard),
                "shard_sha256": digest(shard),
            },
        )
        manifests.append(
            {
                "sha256": digest(path),
                "transport_receipt": str(transport),
                "transport_sha256": digest(transport),
            }
        )
    plan = root / "plan.json"
    export_json(
        plan,
        {
            "schema": "bl-served-search-wp1-plan/v2",
            "commit": "a" * 40,
            "bank_sha256": digest(bank),
            "manifests": manifests,
        },
    )

    # Isolate the terminal-chain gate. Real world validation remains in load_rows;
    # here the spy proves the analyzer cannot be reached on missing/broken chains.
    def load(bank_path, shards):
        assert len(shards) == 8
        return ["all-eight-opened"], []

    monkeypatch.setattr(analyze, "load_rows", load)
    monkeypatch.setattr(
        analyze,
        "analyze",
        lambda rows: {"status": "completed", "metrics": rows, "decision": {"label": "TEST_ONLY"}},
    )
    return plan


def test_eight_terminal_chain_opens_pinned_analyzer(tmp_path: Path, monkeypatch) -> None:
    plan = terminal_family(tmp_path, monkeypatch)
    assert finalize(plan, tmp_path / "decision.json")["metrics"] == ["all-eight-opened"]


@pytest.mark.parametrize(
    "fault", ["chunk-plan", "supervisor-manifest", "export-hash", "missing-eighth"]
)
def test_broken_terminal_chain_never_opens_analyzer(
    tmp_path: Path, monkeypatch, fault: str
) -> None:
    plan = terminal_family(tmp_path, monkeypatch)

    def republish(path: Path, value: dict) -> None:
        for suffix in ("", ".sha256", ".complete"):
            Path(str(path) + suffix).unlink()
        export_json(path, value)

    wrapper_path = tmp_path / "chunk-01/receipt.json"
    wrapper = json.loads(wrapper_path.read_text())
    supervisor_path = tmp_path / "chunk-01/runtime-output/receipt.json"
    supervisor = json.loads(supervisor_path.read_text())
    if fault == "chunk-plan":
        path = tmp_path / "chunk-01/plan.json"
        changed = json.loads(path.read_text())
        changed["analyzer_sha256"] = "a" * 64
        republish(path, changed)
    elif fault == "supervisor-manifest":
        path = tmp_path / "chunk-01/runtime-output/manifest.json"
        changed = json.loads(path.read_text())
        changed["job_id"] = "foreign-preparation"
        path.write_text(json.dumps(changed))
        supervisor["manifest_sha256"] = digest(path)
        supervisor_path.write_text(json.dumps(supervisor))
        wrapper["supervisor_sha256"] = digest(supervisor_path)
        republish(wrapper_path, wrapper)
    elif fault == "export-hash":
        shard = tmp_path / "chunk-01/runtime-output/exports/l2-shard-0.json"
        republish(shard, {"test_only": "foreign-shard"})
        wrapper["shard_sha256"] = digest(shard)
        republish(wrapper_path, wrapper)
    else:
        (tmp_path / "chunk-08/receipt.json.complete").unlink()
    value = finalize(plan, tmp_path / "decision.json")
    assert value["status"] == "INCONCLUSIVE"
    assert "metrics" not in value
    expected = {
        "chunk-plan": "chunk plan differs",
        "supervisor-manifest": "supervisor manifest differs",
        "export-hash": "terminal supervisor exports",
        "missing-eighth": "missing",
    }
    assert any(expected[fault] in error for error in value["errors"])


def test_preflight_fallback_hash_is_exactly_the_exercised_archive(tmp_path: Path) -> None:
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    payload = tmp_path / "input.json"
    payload.write_text("{}")
    manifest = {
        "commit": commit,
        "script": "research/bl_served_search/run_l2.sh",
        "source_paths": ["research/bl_served_search/transport_v2.py"],
        "inputs": [{"dest": "input.json", "path": str(payload), "sha256": digest(payload)}],
    }
    result = transport_v2.preflight(manifest, tmp_path / "receipt.json", repo=repo)
    actual = next(row for row in result["files_transferred"] if row["dest"] == "source.tar.gz")
    assert actual["sha256"] == result["source_plan"]["fallback_sha256"]
    assert actual["bytes"] == result["source_plan"]["fallback_bytes"]
    assert verify_export(tmp_path / "receipt.json")
