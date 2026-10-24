"""Outcome-blind fresh-attempt preparation from a verified consumed attempt."""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pytest

from research.bl_ppo import p1_attempt2_pack as pack
from research.bl_ppo.exports import digest, export_bytes, export_json


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "old"
    failed = root / "repair-output/receipt.json"
    export_json(
        failed,
        {
            "status": "INCOMPLETE",
            "failure_class": "UPLOAD_CAP",
            "stop_verified": True,
            "cleanup_error": None,
            "terminal": None,
            "exports": [],
        },
    )
    report = root / "P1_REPAIR_FAILURE_CODEX_V_20261023.json"
    export_json(report, {"status": "VERIFIED_NO_RETRY_UNDER_CURRENT_RULES"})
    monkeypatch.setattr(pack, "FAILURE_RECEIPT", digest(failed))
    monkeypatch.setattr(pack, "FAILURE_REVIEW", digest(report))
    members = {"origin-evidence.json": b'{"unchanged":true}\n'}
    for c, l, ns, allocation in (
        (
            "p1-control.json",
            "verified-ledger.json",
            "old-repair",
            "source_latebody_06_e8_04_pack04",
        ),
        ("canary-control.json", "canary-verified-ledger.json", "old-canary", "canary-allocation"),
    ):
        members[c] = json.dumps(
            {
                "resume": {"namespace": ns, "allocation": allocation},
                "seed": 20261021,
                "checkpoint_every_seconds": 900,
            }
        ).encode()
        members[l] = json.dumps({"namespace": ns, "allocation": allocation, "units": []}).encode()
        for n in (c, l):
            import hashlib

            members[n + ".sha256"] = f"{hashlib.sha256(members[n]).hexdigest()}  {n}\n".encode()
            members[n + ".complete"] = b"complete\n"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, body in sorted(members.items()):
                header = tarfile.TarInfo(name)
                header.size = len(body)
                archive.addfile(header, io.BytesIO(body))
    export_bytes(root / "p1-inputs.tar.gz", buffer.getvalue())
    urls = {
        n: (
            f"https://drive.usercontent.google.com/download?id=fixture_file_{i:02d}"
            "&export=download&confirm=t"
        )
        for i, n in enumerate(["p1-inputs.tar.gz"] + [f"origin-{i}.pt" for i in range(8)])
    }
    export_json(root / "drive-urls.json", urls)
    for name, ns in zip(pack.MANIFESTS, ("old-canary", "old-repair")):
        export_json(
            root / name,
            {
                "commit": pack.PIN,
                "scientific_admission": False,
                "job_id": "p1-source-oldtag",
                "output_dir": str(root / name.removesuffix("-manifest.json")),
                "p1_finalisation_ruling_sha256": pack.FINALISATION,
                "resume": {"namespace": ns, "max_outcome_blind_retries": 1},
                "inputs": [
                    {
                        "dest": n,
                        "url": u,
                        "sha256": digest(root / n) if n == "p1-inputs.tar.gz" else "a" * 64,
                        "size_bytes": (root / n).stat().st_size if n == "p1-inputs.tar.gz" else 100,
                    }
                    for n, u in urls.items()
                ],
                "unchanged": {"scientific": [1, 2, 3]},
            },
        )
    monkeypatch.setattr(
        pack, "SOURCE_MANIFEST_SHA256", {n: digest(root / n) for n in pack.MANIFESTS}
    )
    return root


def test_consumed_source_produces_deterministic_fresh_empty_ledgers(source, tmp_path):
    output = tmp_path / "new"
    templates, bundle, members, changed = pack.attempt_payloads(source, output)
    assert (templates, bundle, members, changed) == pack.attempt_payloads(source, output)
    assert len(changed) == 8
    for n in ("verified-ledger.json", "canary-verified-ledger.json"):
        ledger = json.loads(members[n])
        assert ledger["units"] == []
        assert ledger["namespace"] not in {"old-repair", "old-canary"}
    assert (
        json.loads(members["verified-ledger.json"])["allocation"]
        == "source_latebody_06_e8_04_pack04"
    )
    assert members["origin-evidence.json"] == b'{"unchanged":true}\n'
    assert templates["repair-manifest.json"]["resume"]["max_outcome_blind_retries"] == 2
    assert templates["repair-manifest.json"]["p1_attempt2"]["further_repair_retries"] == 0


def test_fresh_attempt_preparation_is_create_only_and_preserves_old_pack(source, tmp_path):
    before = {p: digest(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "new"
    receipt = pack.prepare(source=source, output=output)
    assert receipt["status"].startswith("HOLD")
    assert before == {p: digest(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        pack.prepare(source=source, output=output)


def test_finalize_refuses_old_bundle_url_or_changed_origin(source, tmp_path):
    output = tmp_path / "new"
    pack.prepare(source=source, output=output)
    with pytest.raises(ValueError, match="new immutable Drive object"):
        pack.finalize(output=output, urls_path=source / "drive-urls.json")
    urls = json.loads((source / "drive-urls.json").read_text())
    urls["origin-1.pt"] = urls["origin-1.pt"].replace("fixture_file", "changed_file")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(urls))
    with pytest.raises(ValueError, match="scientific origin URLs"):
        pack.finalize(output=output, urls_path=path)


def test_finalize_binds_exact_new_url_and_eight_unchanged_origins(source, tmp_path):
    output = tmp_path / "new"
    pack.prepare(source=source, output=output)
    urls = json.loads((source / "drive-urls.json").read_text())
    urls["p1-inputs.tar.gz"] = urls["p1-inputs.tar.gz"].replace("fixture_file", "newbundle_file")
    path = tmp_path / "urls.json"
    path.write_text(json.dumps(urls))
    receipt = pack.finalize(output=output, urls_path=path)
    assert receipt["provider_calls"] == 0
    value = pack.read_export(output / "repair-manifest.json")
    assert value["scientific_admission"] is False
    assert {r["dest"]: r["url"] for r in value["inputs"]} == urls
    assert all(c["field"] != "" for c in receipt["changes"]["repair-manifest.json"])
    with pytest.raises(FileExistsError):
        pack.finalize(output=output, urls_path=path)


def test_consumed_target_is_refused(source, tmp_path):
    output = tmp_path / "new"
    (output / "repair-manifest.json".removesuffix("-manifest.json")).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="consumed"):
        pack.attempt_payloads(source, output)


def test_unverified_or_unreleased_failure_is_refused(source, tmp_path):
    (source / "repair-output/receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="ruled consumed"):
        pack.attempt_payloads(source, tmp_path / "new")


def test_changed_old_bundle_with_fresh_sidecars_cannot_change_science(source, tmp_path):
    path = source / "p1-inputs.tar.gz"
    for suffix in ("", ".sha256", ".complete"):
        path.with_name(path.name + suffix).unlink()
    export_bytes(path, b"changed source bundle with valid self-provided sidecars")
    with pytest.raises(ValueError, match="pinned -04 manifest input"):
        pack.attempt_payloads(source, tmp_path / "new")
