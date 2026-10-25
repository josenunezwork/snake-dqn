"""No-fit contract tests for the F4 D1024 package preparer, sealer and publisher.

Fixtures are synthetic (tests/f4_d1024_fixtures.py). No residual fit, optimizer,
reserve look, factory payload or real fit output is touched. The end-to-end
test stops ``job.run`` at its tensor-cache boundary, after every registered
partition check has passed and before any feature, loss or update is computed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_residual import f4_d1024_package as package
from research.bl_residual.data import allocate_lives
from tests import f4_d1024_fixtures as fx


class PartitionAccepted(Exception):
    """Raised at job.run's tensor-cache boundary: no feature/loss/update computed."""


def test_real_d1024_sparse_source_import_preflight(tmp_path: Path) -> None:
    """Probe the real pack source list in a clean venv, including package imports."""
    from research.bl_residual import f4_d256_package as d256
    from tools.ops.import_preflight import check

    assert package.SOURCE_PATHS is d256.SOURCE_PATHS
    # Pin current source bytes in an isolated fixture repository so this also
    # catches uncommitted source-list fixes without changing the real Git index.
    repo = tmp_path / "repo"
    repo.mkdir()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *package.SOURCE_PATHS], cwd=package.ROOT
    )
    for name in tracked.decode().split("\0"):
        if name:
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package.ROOT / name, target)
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Import regression",
            "-c",
            "user.email=import@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Pin real F4 sparse source",
        ],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    report = check(
        repo,
        "HEAD",
        package.SOURCE_PATHS,
        package.REQUIREMENTS,
        ["research.bl_residual.job", "research.bl_residual.f4_canary"],
    )
    assert report["status"] == "PASS"
    assert report["system_site_packages"] is False and report["user_site"] is False
    for module in (
        "research.bl_labels",
        "research.bl_labels.exports",
        "research.bl_labels.d4096_design",
    ):
        assert module in report["closure"] and module in report["local_modules"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return fx.build_world(tmp_path_factory.mktemp("f4-d1024-world"))


@pytest.fixture()
def prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict) -> dict:
    calls: dict = {}
    fx.patch_package(monkeypatch, world, calls)
    d256_pack = fx.build_d256_pack(tmp_path, world)
    staging = fx.build_staging(tmp_path, d256_pack)
    ledgers = tmp_path / "ledgers"
    ledgers.mkdir()
    for name in ("d256", "d1024"):
        export_json(ledgers / f"{name}.json", {"synthetic_ledger": name})
    result = package._prepare(
        tmp_path / "pack",
        d256_ledger=ledgers / "d256.json",
        d1024_ledger=ledgers / "d1024.json",
        d1024_union_sha="d" * 64,
        staging=staging,
        commit="HEAD",
        preflight_receipt=tmp_path / "preflight.json",
    )
    return {
        "result": result,
        "pack": tmp_path / "pack",
        "staging": staging,
        "d256_pack": d256_pack,
        "calls": calls,
        "ledgers": ledgers,
    }


def test_prepare_publishes_first_and_writes_nested_bound_d1024_partition(
    prepared: dict, world: dict
) -> None:
    pack, calls = prepared["pack"], prepared["calls"]
    assert calls["published"] == [package.D256_UNION_PATH, package.D1024_UNION_PATH]
    inputs = pack / "inputs"
    control = json.loads((inputs / "control.json").read_text())
    staged = json.loads((prepared["staging"] / "control.json").read_text())
    allocation = json.loads((inputs / "life-allocation.json").read_text())
    prior = world["prior"]
    assert control["doses"] == [1024] and control["synthetic"] is False
    assert control["stage_receipts"] == staged["stage_receipts"]
    assert control["reserve_allocation_sha256"] == staged["reserve_allocation_sha256"]
    for name in ("b8.pth", "F4_REGISTRATION.md", "judgment.json", "d256-fit.json"):
        assert (inputs / name).read_bytes() == (prepared["staging"] / name).read_bytes()
    assert (inputs / "d256-control.json").read_bytes() == (
        prepared["staging"] / "d256-control.json"
    ).read_bytes()
    assert allocation["doses"]["256"] == prior["doses"]["256"]
    train = allocation["doses"]["1024"]["train_lives"]
    assert len(train) == 960 and train[:192] == prior["doses"]["256"]["train_lives"]
    assert allocation["doses"]["1024"]["heldout_lives"] == prior["doses"]["256"]["heldout_lives"]
    with np.load(inputs / "train-1024.npz", allow_pickle=False) as values:
        lives = set(values["life_id"].tolist())
        doom = set(values["life_id"][~values["ordinary"]].tolist())
    assert doom == set(train)
    assert not lives & set(prior["doses"]["256"]["heldout_lives"])
    preflight = json.loads((pack / "main-input-preflight.json").read_text())
    assert preflight["doses"] == [1024] and preflight["allocation"] == control["allocation"]
    projection = json.loads((pack / "pack.json").read_text())["upload_projection"]
    assert projection["forecast_status"] == "DIAGNOSTIC_ONLY_NOT_AN_ADMISSION_GATE"
    assert projection["transfer_gate_seconds"] == projection["whole_staging_gate_seconds"] == 300
    qualified = json.loads((pack / "pack.json").read_text())["qualified_transport"]
    assert qualified["d1024_synthetic_inventory_sha256"] == "b" * 64
    assert json.loads((pack / "main-template.json").read_text())["f4"]["qualified_tree"] == (
        qualified["tree_hash"]
    )
    assert calls["validated"] == [f"f4-d1024-canary-{'c' * 8}-01"]


def test_prepared_partition_passes_every_job_run_check_before_any_fit(
    monkeypatch: pytest.MonkeyPatch, prepared: dict, tmp_path: Path
) -> None:
    from research.bl_residual import job

    monkeypatch.setattr(job, "load_ego2s_checkpoint", lambda *_: object())

    def stop(*_args, **_kwargs):
        raise PartitionAccepted

    monkeypatch.setattr(job, "tensor_cache", stop)
    monkeypatch.setattr(job, "fit_residual", lambda *_: pytest.fail("fit invoked"))
    with pytest.raises(PartitionAccepted):
        job.run(prepared["pack"] / "inputs", tmp_path / "run", mode="main")


def test_prepare_refuses_insufficient_support_without_shrinking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict
) -> None:
    small = {**world, "d1024": world["d1024"][:400]}
    calls: dict = {}
    fx.patch_package(monkeypatch, small, calls)
    d256_pack = fx.build_d256_pack(tmp_path, world)
    staging = fx.build_staging(tmp_path, d256_pack)
    for name in ("d256", "d1024"):
        export_json(tmp_path / f"{name}.json", {})
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        package._prepare(
            tmp_path / "pack",
            d256_ledger=tmp_path / "d256.json",
            d1024_ledger=tmp_path / "d1024.json",
            d1024_union_sha="d" * 64,
            staging=staging,
            commit="HEAD",
            preflight_receipt=tmp_path / "preflight.json",
        )
    assert not (tmp_path / "pack").exists()


def test_publication_failure_precedes_label_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.bl_residual import f4_d256_package as d256

    monkeypatch.setattr(d256, "require_window_closed", lambda: None)
    monkeypatch.setattr(d256, "_life_catalog", lambda *_: pytest.fail("labels opened"))
    monkeypatch.setattr(
        package,
        "require_publication",
        lambda *_: (_ for _ in ()).throw(ValueError("unpublished D1024 union")),
    )
    with pytest.raises(ValueError, match="unpublished"):
        package._prepare(
            tmp_path / "pack",
            d256_ledger=tmp_path / "a",
            d1024_ledger=tmp_path / "b",
            d1024_union_sha="d" * 64,
            staging=tmp_path / "staging",
            commit="HEAD",
            preflight_receipt=tmp_path / "preflight",
        )


def test_repaired_acceptance_is_required_before_d1024_label_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.bl_residual import f4_d256_package as d256

    monkeypatch.setattr(d256, "require_window_closed", lambda: None)
    monkeypatch.setattr(package, "require_publication", lambda *_: None)
    monkeypatch.setattr(package, "_require_source", lambda commit: "c" * 40)
    monkeypatch.setattr(package, "_require_preflight", lambda *_: {"fixture": True})
    monkeypatch.setattr(
        package,
        "require_repaired_transport",
        lambda *_: (_ for _ in ()).throw(ValueError("D1024 transport acceptance required")),
    )
    monkeypatch.setattr(d256, "_life_catalog", lambda *_: pytest.fail("labels opened"))

    with pytest.raises(ValueError, match="acceptance required"):
        package._prepare(
            tmp_path / "pack",
            d256_ledger=tmp_path / "d256.json",
            d1024_ledger=tmp_path / "d1024.json",
            d1024_union_sha="d" * 64,
            staging=tmp_path / "staging",
            commit="HEAD",
            preflight_receipt=tmp_path / "preflight.json",
        )


def test_timing_arm_refuses_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.bl_residual import f4_d256_package as d256

    arm = tmp_path / "mac-timing-armed"
    arm.write_text("armed\n")
    monkeypatch.setattr(d256, "TIMING_ARM", arm)
    monkeypatch.setattr(package, "require_publication", lambda *_: pytest.fail("published"))
    with pytest.raises(ValueError, match="window remains armed"):
        package._prepare(
            tmp_path / "pack",
            d256_ledger=tmp_path / "a",
            d1024_ledger=tmp_path / "b",
            d1024_union_sha="d" * 64,
            staging=tmp_path / "staging",
            commit="HEAD",
            preflight_receipt=tmp_path / "preflight",
        )


def test_require_publication_compares_origin_and_checkout_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(package, "ROOT", tmp_path)
    relative = "research/bl_labels/ledger/X.json"
    local = tmp_path / "local.json"
    local.write_text("ledger\n")
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / relative).write_text("ledger\n")
    monkeypatch.setattr(package.subprocess, "check_output", lambda *_a, **_k: b"ledger\n")
    assert package.require_publication(local, relative, digest(local)) == digest(local)
    with pytest.raises(ValueError, match="foreign"):
        package.require_publication(local, relative, "0" * 64)
    monkeypatch.setattr(package.subprocess, "check_output", lambda *_a, **_k: b"other\n")
    with pytest.raises(ValueError, match="origin/main"):
        package.require_publication(local, relative, None)


@pytest.mark.parametrize("fault", ["payload", "sidecar", "symlink", "control"])
def test_staged_inputs_refuse_changed_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict, fault: str
) -> None:
    from research.bl_residual import job

    monkeypatch.setattr(package, "B8_SHA", fx.FAKE_B8_SHA)
    monkeypatch.setattr(job, "B8_SHA", fx.FAKE_B8_SHA)
    staging = fx.build_staging(tmp_path, fx.build_d256_pack(tmp_path, world))
    assert package.staged_inputs(staging)["control"]["doses"] == [1024]
    target = staging / "judgment.json"
    if fault == "payload":
        target.chmod(0o600)
        target.write_text('{"changed": true}\n')
    elif fault == "sidecar":
        Path(str(target) + ".complete").unlink()
    elif fault == "symlink":
        target.unlink()
        target.symlink_to(staging / "life-allocation.json")
    else:
        control = json.loads((staging / "control.json").read_text())
        control["doses"] = [256]
        for suffix in ("", ".sha256", ".complete"):
            Path(str(staging / "control.json") + suffix).unlink()
        export_json(staging / "control.json", control)
    with pytest.raises(ValueError):
        package.staged_inputs(staging)


def test_nested_allocation_refuses_changed_prefix_or_inserted_tranche(world: dict) -> None:
    catalog = [
        {k: v for k, v in life.items() if k != "path"} for life in world["d256"] + world["d1024"]
    ]
    value = package.nested_allocation(catalog, world["prior"])
    assert value == allocate_lives(catalog)
    changed = [dict(row) for row in catalog]
    changed[0]["stream"] = "S-gate" if changed[0]["stream"] == "S-Watch6" else "S-Watch6"
    with pytest.raises(ValueError):
        package.nested_allocation(changed, world["prior"])
    inserted = catalog + [
        {
            "life_id": "inserted",
            "parent_world": "d1024-p-99999",
            "stream": "S-gate",
            "rescue_bearing": True,
            "allocation_index": 5,
        }
    ]
    with pytest.raises(ValueError):
        package.nested_allocation(inserted, world["prior"])


def test_canary_covers_main_transport_with_dose_1024_exports_and_series_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli.canary import family, job_device

    entrypoint = (package.ROOT / package.SCRIPT).read_text()
    monkeypatch.setattr("research.colab_cli.canary._entrypoint", lambda *_: entrypoint)
    inventory = [
        {"path": "/fixture/b8.pth", "dest": "b8.pth", "sha256": "a" * 64},
        {"path": "/fixture/train-1024.npz", "dest": "train-1024.npz", "sha256": "c" * 64},
        {"path": "/fixture/metadata.tar", "dest": "metadata.tar", "sha256": "b" * 64},
    ]
    main, canary = package.manifests(tmp_path, "b" * 40, inventory)
    from research.colab_cli.transfer_policy import authority_binding

    assert canary["inputs"] == main["inputs"] == inventory
    assert main["transfer_exception"] == canary["transfer_exception"] == authority_binding("fit")
    assert canary["required_exports"] == main["required_exports"]
    assert "fit/fit-0001.pth" in main["required_exports"]
    assert canary["env"]["F4_DOSE"] == main["env"]["F4_DOSE"] == "1024"
    assert canary["canary"]["family"] == family(canary) == family(main) == package.SCRIPT
    assert main["output_dir"] != canary["output_dir"]
    liability = 0.0
    for manifest in (main, canary):
        assert job_device(manifest, package.ROOT) == "cuda"
        limits, budget = manifest["f4"], manifest["budget"]
        assert limits["dose"] == 1024
        assert budget["unit_cap"] == 2 * limits["base_units"]
        assert budget["reserve_units"] == 2 * limits["base_reserve_units"]
        assert manifest["timeout_minutes"] == 2 * limits["base_timeout_minutes"]
        assert (
            manifest["timeout_minutes"] * budget["rate_units_per_hour"] / 60 <= budget["unit_cap"]
        )
        assert manifest["requirements"]["allowed_gpus"] == ["L4"]
        assert manifest["resume"]["skip_verified"] is True
        liability += budget["unit_cap"] + budget["reserve_units"]
    assert liability == pytest.approx(6)
    after = package.PRIOR_SERIES_LIABILITY_UNITS + liability
    assert main["f4"]["series_liability_after_d1024_units"] == pytest.approx(after)
    assert after <= main["f4"]["fit_series_effective_units"] == 20


def test_attempt_three_is_a_fresh_full_inventory_canary_and_main_path(tmp_path: Path) -> None:
    inventory = [
        {
            "dest": "b8.pth",
            "url": "https://example.invalid/B8_s3_final.pth",
            "sha256": "a" * 64,
            "size_bytes": 6_562_005,
        },
        {
            "path": "/fixture/factory-verified-ledger.json",
            "dest": "factory-verified-ledger.json",
            "sha256": "b" * 64,
        },
        {
            "path": "/fixture/train-1024.npz",
            "dest": "train-1024.npz",
            "sha256": "c" * 64,
        },
        {
            "path": "/fixture/metadata.tar",
            "dest": "metadata.tar",
            "sha256": "d" * 64,
        },
    ]
    main, canary = package.manifests(tmp_path, "b" * 40, inventory, attempt="03")

    assert canary["inputs"] == main["inputs"] == inventory
    assert {row["dest"] for row in canary["inputs"]} == {
        "b8.pth",
        "factory-verified-ledger.json",
        "train-1024.npz",
        "metadata.tar",
    }
    assert canary["commit"] == main["commit"]
    assert canary["script"] == main["script"]
    assert canary["source_paths"] == main["source_paths"]
    assert canary["device"] == main["device"] == "cuda:0"
    assert canary["requirements"]["gpu"] == main["requirements"]["gpu"] == "L4"
    assert canary["requirements"]["allowed_gpus"] == main["requirements"]["allowed_gpus"] == ["L4"]
    assert canary["resume"] == main["resume"]
    assert canary["job_id"] == "f4-d1024-canary-bbbbbbbb-03"
    assert main["job_id"] == "f4-d1024-fit-bbbbbbbb-03"
    # The fresh attempt must retain the D256 7.2-unit liabilities and both
    # failed D1024 canary envelopes (1.2 each), with the same 20-unit guard.
    assert main["f4"]["prior_series_liability_units"] == pytest.approx(9.6)
    assert main["f4"]["series_liability_after_d1024_units"] == pytest.approx(15.6)
    assert main["f4"]["fit_series_effective_units"] == 20
    assert Path(canary["output_dir"]).name == "canary-output-l4-03"
    assert Path(main["output_dir"]).name == "d1024-output-l4-03"
    assert package.output_names("01") == ("d1024-output-l4-01", "canary-output-l4-01")
    with pytest.raises(ValueError, match="two-digit"):
        package.manifests(tmp_path, "b" * 40, inventory, attempt="3")


def test_excluded_canary_dose_selects_producer_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.bl_residual import f4_canary

    monkeypatch.delenv("F4_DOSE", raising=False)
    assert f4_canary.canary_dose() == 256
    monkeypatch.setenv("F4_DOSE", "1024")
    assert f4_canary.canary_dose() == 1024
    monkeypatch.setenv("F4_DOSE", "4096")
    with pytest.raises(ValueError):
        f4_canary.canary_dose()


def _sparse(path: Path, size: int) -> Path:
    with path.open("wb") as stream:
        stream.truncate(size)
    return path


def test_per_file_model_reproduces_the_d256_canary_and_flags_the_bar(tmp_path: Path) -> None:
    rows = [
        {"dest": name, "path": str(_sparse(tmp_path / name, size))}
        for name, size in (
            ("b8.pth", 6562005),
            ("train-256.npz", 9654571),
            ("metadata.tar", 1300480),
        )
    ]
    d256 = package.upload_projection(rows)
    # D256 canary receipt b9d53c5c: transfer_seconds 72.6 (ruling: about 71 +/- 5).
    assert d256["projected_transfer_seconds"] == pytest.approx(71, abs=5)
    assert d256["forecast_status"] == "DIAGNOSTIC_ONLY_NOT_AN_ADMISSION_GATE"
    assert d256["transfer_gate_seconds"] == 300
    assert d256["whole_staging_gate_seconds"] == 300
    assert d256["historical_observed_transfer_seconds"] == pytest.approx(221.68512525002006)
    assert d256["historical_observed_whole_staging_seconds"] == pytest.approx(235.52283416606952)
    https = [{"dest": "b8.pth", "url": "https://x", "sha256": "a" * 64, "size_bytes": 6562005}]
    assert package.upload_projection(https + rows[1:])["projected_transfer_seconds"] == (
        pytest.approx(d256["projected_transfer_seconds"] - (16 + 4 * 2.8))
    )
    large = [{"dest": "train-1024.npz", "path": str(_sparse(tmp_path / "big", 140 * 1024**2))}]
    assert (
        package.upload_projection(large)["forecast_status"]
        == "DIAGNOSTIC_ONLY_NOT_AN_ADMISSION_GATE"
    )


REAL_B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")


@pytest.mark.skipif(not REAL_B8.is_file(), reason="released B8 checkpoint not on this host")
def test_public_b8_row_is_the_registered_release_asset() -> None:
    catalog = json.loads(package.PUBLIC_INPUTS.read_text())
    asset = next(row for row in catalog["assets"] if row["name"] == "B8_s3_final.pth")
    row = package.public_b8_row(REAL_B8)
    assert row == {
        "dest": "b8.pth",
        "url": asset["url"],
        "sha256": asset["sha256"],
        "size_bytes": asset["size_bytes"],
    }
    assert row["sha256"] == package.B8_SHA


@pytest.mark.parametrize("fault", ["sha", "size", "status"])
def test_public_b8_row_refuses_any_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict, fault: str
) -> None:
    monkeypatch.setattr(package, "B8_SHA", fx.FAKE_B8_SHA)
    catalog = json.loads(world["public_inputs"].read_text())
    staged = tmp_path / "b8.pth"
    staged.write_bytes(fx.FAKE_B8)
    if fault == "sha":
        staged.write_bytes(fx.FAKE_B8[:-1] + b"X")
    elif fault == "size":
        catalog["assets"][0]["size_bytes"] += 1
    else:
        catalog["status"] = "UNVERIFIED"
    path = tmp_path / "PUBLIC_INPUTS.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setattr(package, "PUBLIC_INPUTS", path)
    with pytest.raises(ValueError):
        package.public_b8_row(staged)


def test_transport_rows_keep_training_data_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict
) -> None:
    monkeypatch.setattr(package, "B8_SHA", fx.FAKE_B8_SHA)
    monkeypatch.setattr(package, "PUBLIC_INPUTS", world["public_inputs"])
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(fx.FAKE_B8)
    npz = _sparse(tmp_path / "train-1024.npz", 40 * 1024**2)
    raw = [
        {"path": str(b8), "dest": "b8.pth", "sha256": digest(b8)},
        {"path": str(npz), "dest": "train-1024.npz", "sha256": digest(npz)},
    ]
    rows = package.transport_rows(raw)
    assert "url" in rows[0] and rows[0]["dest"] == "b8.pth" and "path" not in rows[0]
    assert rows[1] == raw[1]
    with pytest.raises(ValueError, match="exactly one"):
        package.transport_rows(raw[1:])


def test_prepared_canary_and_main_share_inputs_and_transport(prepared: dict) -> None:
    from research.colab_cli.canary import input_inventory

    pack = prepared["pack"]
    main = json.loads((pack / "main-template.json").read_text())
    canary = json.loads((pack / "canary.json").read_text())
    assert canary["inputs"] == main["inputs"]
    assert input_inventory(canary) == input_inventory(main)
    transport = {row["dest"]: row["transport"] for row in input_inventory(main)}
    assert transport["b8.pth"] == "https"
    assert transport["metadata.tar"] == "upload"
    assert all(kind == "upload" for dest, kind in transport.items() if dest != "b8.pth")
    assert any(dest == "train-1024.npz" or dest == "metadata.tar" for dest in transport)
    for row in main["inputs"]:
        if row["dest"].endswith(".npz"):
            assert "url" not in row and "path" in row
    projection = json.loads((pack / "pack.json").read_text())["upload_projection"]
    assert {row["dest"]: row["kind"] for row in projection["rows"]}["b8.pth"] == "https"


def test_seal_main_binds_only_the_d1024_main_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("research.colab_cli.supervisor.validate", lambda *_: None)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    main, canary = package.manifests(
        tmp_path, "b" * 40, [{"path": "/x", "dest": "metadata.tar", "sha256": "b" * 64}]
    )
    export_json(tmp_path / "main-template.json", main)
    export_json(tmp_path / "canary.json", canary)
    sealed = package.seal_main(tmp_path / "main-template.json", receipt, tmp_path / "main.json")
    value = json.loads((tmp_path / "main.json").read_text())
    assert value["canary"]["receipt"]["sha256"] == digest(receipt)
    assert sealed["sha256"] == digest(tmp_path / "main.json")
    with pytest.raises(ValueError, match="D1024 main template"):
        package.seal_main(tmp_path / "canary.json", receipt, tmp_path / "other.json")
    d256_main = {**main, "f4": {**main["f4"], "dose": 256}}
    export_json(tmp_path / "d256-main.json", d256_main)
    with pytest.raises(ValueError, match="D1024 main template"):
        package.seal_main(tmp_path / "d256-main.json", receipt, tmp_path / "third.json")


def _fit_pack(
    tmp_path: Path,
    receipt: dict,
    *,
    index: int = 1,
    fault: bool = False,
    attempt: str = "01",
) -> Path:
    from research.bl_residual.protocol import NAMESPACE

    pack = tmp_path / "pack"
    export_json(
        pack / "inputs/control.json",
        {"namespace": NAMESPACE, "allocation": "a" * 64, "doses": [1024]},
    )
    export_json(pack / "pack.json", {"schema": "bl-residual-f4-d1024-pack/v1", "attempt": attempt})
    main_output, _canary_output = package.output_names(attempt)
    run = pack / main_output
    run.mkdir(parents=True, exist_ok=True)
    (run / "receipt.json").write_text(json.dumps(receipt))
    fit = run / "exports/fit"
    export_json(fit / "allocation.json", {"namespace": NAMESPACE, "allocation": "a" * 64})
    export_bytes(fit / f"fit-{index:04d}.pth", b"opaque checkpoint bytes; never loaded")
    if fault:
        export_json(fit / f"fault-{index:04d}.json", {"kind": "KILL_FIT"})
    return pack


PASS = {"status": "PASS", "stop_verified": True, "error": None, "cleanup_error": None}


def test_publish_fit_binds_only_the_complete_index_one_unit(tmp_path: Path) -> None:
    pack = _fit_pack(tmp_path, PASS, attempt="03")
    result = package.publish_fit(pack, tmp_path / "ledger.json")
    value = json.loads((tmp_path / "ledger.json").read_text())
    assert [row["index"] for row in value["units"]] == [1]
    assert result["sha256"] == digest(tmp_path / "ledger.json")


@pytest.mark.parametrize(
    "receipt,index,fault",
    [
        ({**PASS, "status": "INCOMPLETE"}, 1, False),
        ({**PASS, "stop_verified": False}, 1, False),
        ({**PASS, "cleanup_error": "stop failed"}, 1, False),
        (PASS, 0, False),
        (PASS, 1, True),
    ],
)
def test_publish_fit_holds_without_verified_complete_fault_free_unit(
    tmp_path: Path, receipt: dict, index: int, fault: bool
) -> None:
    pack = _fit_pack(tmp_path, receipt, index=index, fault=fault)
    with pytest.raises(ValueError, match="HOLD"):
        package.publish_fit(pack, tmp_path / "ledger.json")
