"""Release transport changes scientific input delivery without changing bytes."""

import argparse
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, manifests, prepare_release, review_jobs, run


def test_release_input_retains_digest_size_and_runtime_destination(tmp_path: Path) -> None:
    path = tmp_path / "SEED_BANK.json"
    path.write_bytes(b"immutable seed bank")
    source = manifests.ExternalInput("seed-bank", path, "seed-bank.json")
    row = source.from_release().as_manifest()
    assert row == {
        "url": manifests.RELEASE_BASE + "/SEED_BANK.json",
        "dest": "seed-bank.json",
        "sha256": admission.sha256(path),
        "size_bytes": path.stat().st_size,
    }
    assert "path" not in row


def test_staging_estimate_rejects_slow_or_oversized_contents_before_reservation(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "acceptance.json"
    evidence.write_bytes(b"small evidence")
    inputs = [
        {"url": manifests.RELEASE_BASE + "/demo.tar.gz", "size_bytes": 62_552_352},
        {"path": str(evidence)},
    ]
    row = {"job_id": "job", "inputs": inputs}
    packaging = {"jobs": {"job": {"fallback_size": 100_000}}}
    result = admission.staging_preflight([row], packaging)
    assert result["jobs"]["job"]["estimated_seconds_upper_bound"] < 300
    packaging["jobs"]["job"]["fallback_size"] = 32 * 1024**2
    with pytest.raises(admission.AdmissionRefused, match="exceeds its transfer cap"):
        admission.staging_preflight([row], packaging)
    evidence.write_bytes(b"x" * 100_001)
    with pytest.raises(admission.AdmissionRefused, match="100 KB"):
        admission.staging_preflight([row], packaging)


@pytest.mark.parametrize("current_acceptance", [False, True])
def test_offline_candidate_is_pinned_but_cannot_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current_acceptance: bool
) -> None:
    seed = tmp_path / "seed"
    seed.write_bytes(b"frozen seeds")
    frozen = [manifests.ExternalInput("seed", seed, "seed-bank.json")]
    acceptance = tmp_path / "acceptance"
    acceptance.write_bytes(b"historical only")
    paths = []
    for i in range(10):
        path = tmp_path / f"old-{i}.json"
        admission.write_once(
            path,
            {
                "job_id": f"job-{i}",
                "inputs": [
                    frozen[0].as_manifest(),
                    manifests.ExternalInput(
                        "acceptance", acceptance, "cli-acceptance-result.json"
                    ).as_manifest(),
                ],
            },
        )
        paths.append({"job_id": f"job-{i}", "path": str(path), "sha256": admission.sha256(path)})
    previous = tmp_path / "previous.json"
    admission.write_once(
        previous,
        {"schema": admission.SCHEMA, "status": "PREPARED_NOT_DISPATCHED", "manifests": paths},
    )
    monkeypatch.setattr(admission, "published_commit", lambda *_: "b" * 40)
    monkeypatch.setattr(admission, "source_closure", lambda *_: {"source": "digest"})
    monkeypatch.setattr(
        admission,
        "packaging_preflight",
        lambda _, rows: {"jobs": {row["job_id"]: {"fallback_size": 100} for row in rows}},
    )
    monkeypatch.setattr(review_jobs, "verified_inputs", lambda: frozen)
    if current_acceptance:
        monkeypatch.setattr(prepare_release.final_artifacts, "acceptance_result", lambda _: {})
    args = argparse.Namespace(
        repo=tmp_path,
        published_ref="origin/branch",
        previous_plan=previous,
        artifact_root=tmp_path / "production",
        manifest_dir=tmp_path / "jobs",
        out=tmp_path / "plan.json",
    )
    plan_path = prepare_release.prepare(args)
    plan = json.loads(plan_path.read_text())
    assert plan["commit"] == "b" * 40
    assert plan["status"] == (
        "PENDING_REPRICE_AND_SETTLEMENT" if current_acceptance else "PENDING_FINAL_ONLY_ACCEPTANCE"
    )
    assert plan["cli_acceptance_current"] is current_acceptance
    assert (
        "fresh final-only acceptance and independent review" in plan["admission_blockers"]
    ) is not current_acceptance
    assert len(plan["manifests"]) == 10
    assert plan["input_staging"]["status"] == "PASS"
    first = json.loads(Path(plan["manifests"][0]["path"]).read_text())
    assert "url" in first["inputs"][0]
    assert first["env"]["COLAB_CLI_REQUIREMENTS"] == manifests.RUNTIME_REQUIREMENTS
    assert first["source_paths"] == list(manifests.RUNTIME_SOURCE_PATHS)
    with pytest.raises(admission.AdmissionRefused, match="not an immutable prepared"):
        run._verify_plan(plan_path, tmp_path)


def test_missing_release_refuses_before_account_or_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.colab_cli import inputs

    rows = []
    root = tmp_path / "production"
    for i in range(10):
        path = tmp_path / f"job-{i}.json"
        admission.write_once(
            path,
            {
                "job_id": f"job-{i}",
                "output_dir": str(root / "training" / str(i)),
                "inputs": [
                    {
                        "url": manifests.RELEASE_BASE + "/demo.tar.gz",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                        "dest": "demo.tar.gz",
                    }
                ],
            },
        )
        rows.append({"job_id": f"job-{i}", "path": str(path), "sha256": admission.sha256(path)})
    monkeypatch.setattr(run, "_verify_plan", lambda *_: {"manifests": rows})
    monkeypatch.setattr(
        admission,
        "packaging_preflight",
        lambda _, values: {"jobs": {row["job_id"]: {"fallback_size": 1} for row in values}},
    )

    def missing(*_):
        raise inputs.InputError("release asset missing")

    def forbidden(*_, **__):
        pytest.fail("account or ledger accessed before missing-input refusal")

    monkeypatch.setattr(inputs, "preflight", missing)
    monkeypatch.setattr(run.ledger, "Ledger", forbidden)
    monkeypatch.setattr(run, "_snapshot", forbidden)
    args = argparse.Namespace(
        plan=tmp_path / "plan",
        repo=tmp_path,
        artifact_root=root,
        supervisor=tmp_path / "research/colab_cli/launch.sh",
    )
    with pytest.raises(admission.AdmissionRefused, match="release asset missing"):
        run.dispatch(args)


def test_live_preflight_uses_measured_slow_rate_to_refuse_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.colab_cli import inputs

    row = {
        "job_id": "job",
        "inputs": [
            {
                "url": manifests.RELEASE_BASE + "/demo.tar.gz",
                "size_bytes": 62_552_352,
                "sha256": "a" * 64,
                "dest": "demo.tar.gz",
            }
        ],
    }
    packaging = {"jobs": {"job": {"fallback_size": 1}}}
    monkeypatch.setattr(
        inputs,
        "preflight",
        lambda _: [{"size_bytes": 62_552_352, "bytes_per_second": 13_000}],
    )
    with pytest.raises(admission.AdmissionRefused, match="exceeds its transfer cap"):
        admission.delivery_preflight([row], packaging)


@pytest.mark.parametrize("large_rate", [2 * 1024**2, 512 * 1024])
def test_delivery_preflight_ignores_tiny_file_latency(
    monkeypatch: pytest.MonkeyPatch, large_rate: float
) -> None:
    from research.colab_cli import inputs

    verified = [
        {"size_bytes": 330, "bytes_per_second": 110},
        {"size_bytes": 52_000, "bytes_per_second": 26_000},
        {"size_bytes": 62_552_352, "bytes_per_second": large_rate},
    ]
    row = {
        "job_id": "job",
        "inputs": [
            {"url": manifests.RELEASE_BASE + f"/{i}", "size_bytes": item["size_bytes"]}
            for i, item in enumerate(verified)
        ],
    }
    packaging = {"jobs": {"job": {"fallback_size": 1}}}
    monkeypatch.setattr(inputs, "preflight", lambda _: verified)
    result = admission.delivery_preflight([row], packaging)
    assert result["status"] == "PASS"
    expected = 2 * sum(item["size_bytes"] for item in verified) / min(1024**2, large_rate) + 15
    assert result["staging"]["jobs"]["job"]["release_seconds_upper_bound"] == expected


@pytest.mark.parametrize("change", [None, "seed", "acceptance", "unsettled", "wrong_original"])
def test_staging_continuation_keeps_science_and_same_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    seed = tmp_path / "seed-bank.json"
    seed.write_bytes(b"immutable seed bank")
    old_acceptance, fresh_acceptance = tmp_path / "old-acceptance", tmp_path / "new-acceptance"
    old_acceptance.write_bytes(b"historical acceptance")
    fresh_acceptance.write_bytes(b"fresh qualified acceptance")
    binding = {"path": str(fresh_acceptance), "sha256": admission.sha256(fresh_acceptance)}
    paths, old_paths = [], []
    for i in range(10):
        row = {
            "job_id": f"job-{i}",
            "commit": "a" * 40,
            "env": {"S1_SEED": str(i)},
            "inputs": [
                manifests.ExternalInput("seed", seed, "seed-bank.json").as_manifest(),
                manifests.ExternalInput(
                    "acceptance", old_acceptance, "cli-acceptance-result.json"
                ).as_manifest(),
            ],
        }
        old = tmp_path / f"old-{i}.json"
        admission.write_once(old, row)
        old_paths.append(
            {"job_id": row["job_id"], "path": str(old), "sha256": admission.sha256(old)}
        )
        current = tmp_path / f"new-{i}.json"
        row.update(commit="b" * 40, output_dir=str(tmp_path / "outputs" / str(i)))
        row["inputs"] = [
            manifests.ExternalInput("seed", seed, "seed-bank.json").from_release().as_manifest(),
            {"dest": "cli-acceptance-result.json", **binding},
        ]
        admission.write_once(current, row)
        paths.append((row["job_id"], current))
    mac = tmp_path / "mac.json"
    admission.write_once(mac, {"kind": "mac"})
    plan = {
        "schema": admission.SCHEMA,
        "status": "PREPARED_NOT_DISPATCHED",
        "liabilities": {"job-0": {}},
        "pricing_receipt": {"path": str(tmp_path / "pricing")},
        "mac_manifest": {"path": str(mac), "sha256": admission.sha256(mac)},
    }
    previous = tmp_path / "previous.json"
    admission.write_once(previous, {**plan, "manifests": old_paths})
    review = tmp_path / "review.json"
    admission.write_once(review, {"acceptance_result": binding})
    state = {
        "planned": plan["liabilities"],
        "outstanding": {},
        "settled": {
            "cli-acceptance": {},
            "staging-failure:job-0": {
                "source_manifest": {k: old_paths[0][k] for k in ("path", "sha256")}
            },
        },
    }
    ledger_path = tmp_path / "ledger.json"
    if change in {"seed", "acceptance"}:
        row = json.loads(paths[0][1].read_text())
        if change == "seed":
            row["env"]["S1_SEED"] = "999"
        else:
            row["inputs"][1]["sha256"] = "a" * 64
        paths[0][1].write_text(json.dumps(row))
    elif change == "unsettled":
        state["outstanding"] = {"job-0": {}}
    elif change == "wrong_original":
        state["settled"]["staging-failure:job-0"]["source_manifest"]["sha256"] = "a" * 64
    admission.write_once(ledger_path, state)
    calls = []

    class FakeLedger:
        def __init__(self, _):
            pass

        def requalify_acceptance(self, value):
            calls.append(value)

        def status(self):
            return {}

    monkeypatch.setattr(run.ledger, "Ledger", FakeLedger)
    monkeypatch.setattr(
        admission, "priced_mac_bound", lambda _: {"cli_acceptance_receipt": binding}
    )
    args = argparse.Namespace(previous_plan=previous, acceptance_review=review, ledger=ledger_path)
    if change:
        with pytest.raises((admission.AdmissionRefused, run.ledger.Refused)):
            run._staging_continuation(args, plan, paths)
        assert not calls
    else:
        run._staging_continuation(args, plan, paths)
        assert calls == [review]


def test_real_input_child_validator_accepts_parent_output_before_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the real validator; replace only the Colab filesystem prefix boundary."""
    import io

    from research.colab_cli import runtime

    root = tmp_path / "runtime"
    (root / "output").mkdir(parents=True)
    payload = b"verified public bytes"
    source = root / "source"
    source.write_bytes(payload)
    row = {
        "url": manifests.RELEASE_BASE + "/demo.tar.gz",
        "dest": "demo.tar.gz",
        "sha256": admission.sha256(source),
        "size_bytes": len(payload),
    }
    spec = {
        "job_id": "test",
        "generation": "g",
        "root": str(root),
        "command": ["trainer", "--fixed-recipe"],
        "env": {},
        "timeout_seconds": 60,
        "lease_seconds": 180,
        "checkpoint_grace_seconds": 1,
        "max_export_bytes": 1024,
        "release_endpoint": "unitendpoint",
        "public_inputs": [row],
    }
    spec_path = root / "spec.json"
    admission.write_once(spec_path, spec)

    class ColabPrefix:
        def resolve(self):
            return self

        def __str__(self):
            return "/content/snake-cli/test"

        def is_dir(self):
            return root.is_dir()

        def __truediv__(self, relative):
            return root / relative

    def runtime_path(value):
        if str(value) == str(root):
            return ColabPrefix()
        if str(value) == "/content/snake-cli/test":
            return root
        return Path(value)

    monkeypatch.setattr(runtime, "Path", runtime_path)
    with pytest.raises(runtime.SpecError, match="fresh output directory"):
        runtime._validate_spec(spec)
    monkeypatch.setattr(runtime, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))

    class Executed(Exception):
        pass

    def execute(executable, command, environment):
        assert executable == "trainer" and command == spec["command"]
        assert (root / "inputs/demo.tar.gz").read_bytes() == payload
        assert (root / "public-inputs-verified.json").exists()
        raise Executed

    monkeypatch.setattr(runtime.os, "execvpe", execute)
    with pytest.raises(Executed):
        runtime._fetch_inputs_then_exec(spec_path)
