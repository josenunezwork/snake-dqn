"""Retained-chain fault injection and immutable cache regression tests."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.colab_cli import cache_environment, extension_cache, retention, supervisor


@pytest.fixture
def chain_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    monkeypatch.setattr(supervisor, "validate", lambda value: value)
    monkeypatch.setattr(retention, "validate_a100_chain", lambda *_: ({}, "key"))
    monkeypatch.setattr(retention, "require_native_qualification", lambda *_: None)
    monkeypatch.setattr(supervisor, "account_locked", lambda *_a, **_k: contextlib.nullcontext())
    events = []
    monkeypatch.setattr(
        supervisor, "stop_verified", lambda _cli, endpoint: events.append(("stop", endpoint))
    )
    broker = supervisor.Broker(tmp_path / "lanes", "never-allocate")
    cli = SimpleNamespace(usage=lambda: {"balance": 100, "rate": 0, "assignments": 0})
    manifests = [
        {
            "job_id": name,
            "requirements": {"gpu": "A100"},
            "budget": {"rate_units_per_hour": 10},
            "timeout_minutes": 3,
            "retention_chain": {"qualification": {}},
        }
        for name in ("first", "second")
    ]

    def complete(manifest, retained):
        out = tmp_path / manifest["job_id"]
        out.mkdir()
        active, intent = tmp_path / "active.json", tmp_path / "pending.json"
        active.touch(exist_ok=True)
        intent.touch(exist_ok=True)
        retained.update(
            cli=cli,
            endpoint="owned-a100",
            active=active,
            intent=intent,
            accounting=SimpleNamespace(receipt=lambda: {"continuous": True}),
            assignment_allocated_started_monotonic=0,
        )
        retained.setdefault("outputs", []).append(str(out))
        events.append(("run", manifest["job_id"]))

    return broker, manifests, complete, events


def test_interrupted_second_launch_releases_owned_assignment(chain_harness, monkeypatch, tmp_path):
    broker, manifests, complete, events = chain_harness

    def run(manifest, retained):
        if manifest["job_id"] == "second":
            raise OSError("uncertain second-launch RPC")
        complete(manifest, retained)

    monkeypatch.setattr(broker, "run", run)
    with pytest.raises(OSError, match="uncertain second"):
        broker.chain(manifests, 10, 2)
    assert events == [("run", "first"), ("stop", "owned-a100")]
    release = json.loads((tmp_path / "first/session-release.json").read_text())
    assert release["stop_verified"] is True and "uncertain second" in release["chain_error"]
    assert not (tmp_path / "active.json").exists()


def test_handoff_wall_cap_prevents_later_job(chain_harness, monkeypatch, tmp_path):
    broker, manifests, complete, events = chain_harness
    clock = [0.0]
    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def run(manifest, retained):
        complete(manifest, retained)
        clock[0] = 601

    monkeypatch.setattr(broker, "run", run)
    with pytest.raises(supervisor.Refused, match="retention lease elapsed"):
        broker.chain(manifests, 10, 2)
    assert events == [("run", "first"), ("stop", "owned-a100")]
    assert json.loads((tmp_path / "first/session-release.json").read_text())["stop_verified"]


def test_uncertain_stop_keeps_durable_recovery_record(chain_harness, monkeypatch, tmp_path):
    broker, manifests, complete, events = chain_harness
    monkeypatch.setattr(broker, "run", complete)

    def stop(*_):
        raise supervisor.Refused("backend endpoint still listed")

    monkeypatch.setattr(supervisor, "stop_verified", stop)
    with pytest.raises(supervisor.Refused, match="still listed"):
        broker.chain(manifests, 10, 2)
    assert (tmp_path / "active.json").exists()
    release = json.loads((tmp_path / "first/session-release.json").read_text())
    assert release["stop_verified"] is False and "still listed" in release["error"]


def test_chain_child_failure_stops_before_later_packet(chain_harness, monkeypatch):
    broker, manifests, complete, events = chain_harness

    def run(manifest, retained):
        complete(manifest, retained)
        raise supervisor.Refused("child exited 23")

    monkeypatch.setattr(broker, "run", run)
    with pytest.raises(supervisor.Refused, match="exited 23"):
        broker.chain(manifests, 10, 2)
    assert events == [("run", "first"), ("stop", "owned-a100")]


@pytest.fixture
def extension_harness(tmp_path, monkeypatch):
    extension_cache._PROCESS_MODULES.clear()
    extension_cache._MODULE_KEYS.clear()
    cache, job = tmp_path / "cache", tmp_path / "job"
    (job / "repo").mkdir(parents=True)
    source = job / "repo/kernel.cu"
    source.write_text("frozen exact kernel")
    monkeypatch.setenv("COLAB_CLI_ROOT", str(job))
    calls = []

    def compile(name, build_directory, **kwargs):
        calls.append(name)
        output = Path(build_directory) / "test.so"
        output.write_bytes(b"compiled-native-binary")
        return SimpleNamespace(__name__=name, __file__=str(output))

    loader = SimpleNamespace(exec_module=lambda _module: None)
    monkeypatch.setattr(
        extension_cache.importlib.util,
        "spec_from_file_location",
        lambda *_: SimpleNamespace(loader=loader),
    )
    monkeypatch.setattr(
        extension_cache.importlib.util,
        "module_from_spec",
        lambda *_: SimpleNamespace(__name__="kernel"),
    )
    return cache, job, source, compile, calls


def test_immutable_extension_warm_import_does_not_compile(extension_harness, monkeypatch):
    cache, job, source, compile, calls = extension_harness
    extension_cache.cached_load(
        compile,
        "kernel",
        {"sources": [str(source)]},
        job=job,
        cache_root=cache,
        parent_key="a" * 64,
    )
    extension_cache._PROCESS_MODULES.clear()
    extension_cache._MODULE_KEYS.clear()
    next_job = job.with_name("next-job")
    (next_job / "repo").mkdir(parents=True)
    other_source = next_job / "repo/kernel.cu"
    other_source.write_bytes(source.read_bytes())
    monkeypatch.setenv("COLAB_CLI_ROOT", str(next_job))
    extension_cache.cached_load(
        compile,
        "kernel",
        {"sources": [str(other_source)]},
        job=next_job,
        cache_root=cache,
        parent_key="a" * 64,
    )
    assert calls == ["kernel"]
    first = json.loads((job / "extension-cache-events.jsonl").read_text())
    second = json.loads((next_job / "extension-cache-events.jsonl").read_text())
    assert first["status"] == "MISS_COMPILED" and second["status"] == "HIT_VERIFIED_IMPORT"
    assert first["key"] == second["key"] and first["sha256"] == second["sha256"]


def test_corrupt_extension_never_loads_or_recompiles(extension_harness):
    cache, job, source, compile, calls = extension_harness
    options = {"sources": [str(source)]}
    extension_cache.cached_load(
        compile, "kernel", options, job=job, cache_root=cache, parent_key="a" * 64
    )
    extension_cache._PROCESS_MODULES.clear()
    extension_cache._MODULE_KEYS.clear()
    binary = next(cache.rglob("module.so"))
    binary.chmod(0o644)
    binary.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        extension_cache.cached_load(
            compile, "kernel", options, job=job, cache_root=cache, parent_key="a" * 64
        )
    assert calls == ["kernel"]


def test_partial_extension_publication_is_not_replaced(extension_harness):
    cache, job, source, compile, calls = extension_harness
    options = {"sources": [str(source)]}
    key = extension_cache._key("kernel", options, "a" * 64)
    directory = cache / "frozen" / key
    directory.mkdir(parents=True)
    (directory / "module.so").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="partial immutable"):
        extension_cache.cached_load(
            compile, "kernel", options, job=job, cache_root=cache, parent_key="a" * 64
        )
    assert calls == []


def test_extension_source_flags_and_parent_pins_invalidate(extension_harness):
    _cache, _job, source, _compile, _calls = extension_harness
    options = {"sources": [str(source)]}
    original = extension_cache._key("kernel", options, "a" * 64)
    assert extension_cache._key("kernel", options, "b" * 64) != original
    assert (
        extension_cache._key("kernel", {**options, "extra_cuda_cflags": ["--fmad=false"]}, "a" * 64)
        != original
    )
    source.write_text("changed source")
    assert extension_cache._key("kernel", options, "a" * 64) != original


def test_environment_cache_detects_mutation_and_relocates_scripts(tmp_path):
    venv = tmp_path / "first"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/tool").write_text("#!" + str(venv / "bin/python") + "\nprint(1)\n")
    cache = tmp_path / "cache"
    cache_environment.publish(cache, venv, "frozen-contract")
    second = tmp_path / "second"
    cache_environment.restore(cache, second, "frozen-contract")
    assert (second / "bin/tool").read_text().startswith("#!" + str(second / "bin/python") + "\n")
    (cache / ".venv/bin/tool").write_text("mutation")
    with pytest.raises(RuntimeError, match="cache bytes changed"):
        cache_environment.restore(cache, tmp_path / "third", "frozen-contract")


def test_idle_handoff_cap_refuses_rpc_even_with_remaining_chain_time(tmp_path, monkeypatch):
    cli = supervisor.CLI("never-call", tmp_path / "sessions.json")
    cli.deadline = supervisor.time.monotonic() + 2400
    cli.priced_deadline = cli.deadline
    cli.handoff_deadline = supervisor.time.monotonic() - 1
    with pytest.raises(supervisor.Refused, match="deadline"):
        cli.remaining_seconds(60)
    with pytest.raises(supervisor.Refused, match="deadline"):
        cli._call_once("exec", timeout=60)
