"""Offline credential custody tests; no real Keychain or RunPod access."""

import hashlib
import io
import json
import subprocess
import traceback
import urllib.error
from types import SimpleNamespace

import pytest

from research.redesign_m3c_20261008 import gpu_run_v2 as gr
from tests import test_m3c_gpu_run_v2 as training
from tests import test_perf_podbench as bench

FAKE_KEY = "fake-owner-restricted-selfdelete-credential-782193"


def _assert_no_secret(root, *values):
    for path in root.rglob("*"):
        if path.is_file():
            assert FAKE_KEY.encode() not in path.read_bytes(), str(path)
    assert FAKE_KEY not in repr(values)


def test_keychain_reader_never_runs_under_pytest(monkeypatch):
    monkeypatch.setattr(gr.sys, "platform", "darwin")
    monkeypatch.setattr(gr.subprocess, "run", lambda *a, **kw: pytest.fail("real key lookup"))
    assert gr.read_self_delete_key() is None


@pytest.mark.parametrize("code,stdout,expected", [(0, FAKE_KEY + "\n", FAKE_KEY), (44, "", None)])
def test_keychain_reader_pipe_only(monkeypatch, capsys, code, stdout, expected):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=code, stdout=stdout.encode(), stderr=b"ignored")

    # Install the fake BEFORE enabling the reader; no real Keychain is ever consulted.
    monkeypatch.setattr(gr.subprocess, "run", run)
    monkeypatch.setattr(gr.sys, "platform", "darwin")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert gr.read_self_delete_key() == expected
    assert calls[0][0] == [
        "security",
        "find-generic-password",
        "-s",
        "runpod-selfdelete",
        "-a",
        "runpod",
        "-w",
    ]
    assert calls[0][1]["capture_output"] is True
    assert "env" not in calls[0][1]
    assert FAKE_KEY not in repr(calls) + repr(capsys.readouterr())


@pytest.mark.parametrize("failure", ["access", "timeout", "decode", "empty"])
def test_keychain_errors_never_echo_secret(monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired([FAKE_KEY], 15, output=FAKE_KEY)
        return SimpleNamespace(
            returncode=1 if failure == "access" else 0,
            stdout=(
                b"\xff"
                if failure == "decode"
                else (b"" if failure == "empty" else FAKE_KEY.encode())
            ),
            stderr=FAKE_KEY.encode(),
        )

    monkeypatch.setattr(gr.subprocess, "run", run)
    monkeypatch.setattr(gr.sys, "platform", "darwin")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    with pytest.raises(RuntimeError) as exc:
        gr.read_self_delete_key()
    assert FAKE_KEY not in "".join(traceback.format_exception(exc.value))


class Provisioned:
    def set_self_delete_key(self, key):
        assert key == FAKE_KEY
        assert not self.uploads and not self.jobs
        self.provisioned = True
        return {"self_delete_armed": True, "self_delete_probe": 200, "ignored": key}


def test_training_handoff_precedes_jobs_and_leaves_no_artifact(tmp_path, monkeypatch, capsys):
    class Agent(Provisioned, training.FakeAgent):
        pass

    monkeypatch.setattr(gr, "read_self_delete_key", lambda: FAKE_KEY)
    agent = Agent(polls=40, gpu_util=95.0)
    outcome, state = training._drive(
        tmp_path, monkeypatch, agent, training._plan(transitions=2400 * 50), waiver=None
    )
    assert outcome == "complete" and agent.provisioned
    assert "self_delete_waived" not in state
    gr.save_state(tmp_path, state)
    body = gr.pod_body("fake-pod", "a" * 64, 1000, 2000)
    _assert_no_secret(tmp_path, body, state, agent.jobs, capsys.readouterr())


def test_bench_handoff_leaves_no_body_argv_state_receipt_ledger_or_result_secret(
    tmp_path, monkeypatch, capsys
):
    class Agent(Provisioned, bench.FakeAgent):
        pass

    monkeypatch.setattr(gr, "read_self_delete_key", lambda: FAKE_KEY)
    rp, agent = bench.FakeRp(), Agent()
    code, state, receipt, out, _, _ = bench._session(
        tmp_path, monkeypatch, rp, agent, accept_no_self_delete=None
    )
    assert code == 0 and agent.provisioned
    assert "self_delete_waived" not in state
    assert list((out / "out").glob("*.json"))
    _assert_no_secret(
        out, rp.created, state, receipt, agent.jobs, bench.FakeLedger.calls, capsys.readouterr()
    )


@pytest.mark.parametrize("waiver", [None, "explicit test waiver"])
def test_missing_key_retains_probe_and_waiver(tmp_path, monkeypatch, waiver):
    monkeypatch.setattr(gr, "read_self_delete_key", lambda: None)
    outcome, state = training._drive(
        tmp_path,
        monkeypatch,
        training.FakeAgent(),
        training._plan(transitions=2400 * 50),
        waiver=waiver,
    )
    assert outcome == ("complete" if waiver else "pod self-delete not proven (probe 403)")
    assert ("self_delete_waived" in state) == bool(waiver)


def test_delivery_failure_aborts_before_upload_even_with_waiver(tmp_path, monkeypatch, capsys):
    class Agent(training.FakeAgent):
        def set_self_delete_key(self, key):
            raise RuntimeError(key)

    monkeypatch.setattr(gr, "read_self_delete_key", lambda: FAKE_KEY)
    agent = Agent()
    outcome, state = training._drive(tmp_path, monkeypatch, agent, training._plan())
    assert outcome == "self-delete credential provisioning failed"
    assert not agent.uploads and not agent.jobs
    _assert_no_secret(tmp_path, state, capsys.readouterr())


def test_handoff_uses_authenticated_https_body_no_redirects(monkeypatch):
    requests, handlers = [], []

    def open_request(req, **kwargs):
        requests.append(req)
        return io.BytesIO(b'{"self_delete_armed":true,"self_delete_probe":200}')

    def opener(*args):
        handlers.extend(args)
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(gr.urllib.request, "build_opener", opener)
    agent = gr.Agent("fake-pod", "channel-token")
    assert agent.set_self_delete_key(FAKE_KEY)["self_delete_probe"] == 200
    req = requests[0]
    assert req.full_url == "https://fake-pod-8000.proxy.runpod.net/self-delete-key"
    assert req.get_header("X-m3-token") == "channel-token"
    assert json.loads(req.data) == {"key": FAKE_KEY}
    assert FAKE_KEY not in repr(req.headers) + req.full_url
    redirect = next(h for h in handlers if isinstance(h, gr._NoCredentialRedirect))
    with pytest.raises(RuntimeError, match="redirect refused"):
        redirect.redirect_request(req, None, 302, "redirect", {}, "https://elsewhere.invalid/")


def test_plain_http_cannot_receive_key(monkeypatch):
    agent = gr.Agent("fake", "token", base="http://127.0.0.1:9999")
    monkeypatch.setattr(agent, "_req", lambda *a, **kw: pytest.fail("plaintext transfer"))
    with pytest.raises(RuntimeError, match="delivery failed"):
        agent.set_self_delete_key(FAKE_KEY)


@pytest.mark.parametrize("path", ["/health", "/log/job", "/out/result.json", "/ls/out"])
def test_reflected_key_cannot_reach_logs_state_or_pulled_results(tmp_path, monkeypatch, path):
    agent = gr.Agent("fake", "token")
    agent._self_delete_key = FAKE_KEY.encode()
    monkeypatch.setattr(
        gr.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(FAKE_KEY.encode())
    )
    with pytest.raises(RuntimeError) as exc:
        agent.get(path)
    gr.save_state(tmp_path, {"error": str(exc.value)})
    gr.append_jsonl(tmp_path / "progress.jsonl", {"error": str(exc.value)})
    _assert_no_secret(tmp_path)


def test_secret_result_is_not_written_by_real_pull_path(tmp_path, monkeypatch):
    agent = gr.Agent("fake", "token")
    agent._self_delete_key = FAKE_KEY.encode()
    payload = json.dumps({"key": FAKE_KEY}).encode()
    listing = [{"path": "result.json", "sha256": hashlib.sha256(payload).hexdigest()}]

    def response(req, **kwargs):
        return io.BytesIO(
            json.dumps(listing).encode() if req.full_url.endswith("/ls/out") else payload
        )

    monkeypatch.setattr(gr.urllib.request, "urlopen", response)
    monkeypatch.setattr(gr.time, "sleep", lambda _: None)
    result = gr.download_rest(agent, tmp_path)
    assert result == {"files": 0, "failed": ["result.json"]}
    assert not (tmp_path / "out/result.json").exists()
    _assert_no_secret(tmp_path, result)


def test_transport_exception_is_sanitized(monkeypatch):
    agent = gr.Agent("fake", "token")
    agent._self_delete_key = FAKE_KEY.encode()

    def fail(*a, **kw):
        raise urllib.error.HTTPError(
            "https://invalid/" + FAKE_KEY, 503, FAKE_KEY, {}, io.BytesIO(FAKE_KEY.encode())
        )

    monkeypatch.setattr(gr.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError) as exc:
        agent.health()
    assert exc.value.code == 503
    assert FAKE_KEY not in "".join(traceback.format_exception(exc.value))


def test_exec_retries_transport_failure_with_owner_key(monkeypatch):
    agent = gr.Agent("fake", "token")
    agent._self_delete_key = FAKE_KEY.encode()
    calls = []

    def response(req, **kwargs):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.URLError(FAKE_KEY)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(gr.urllib.request, "urlopen", response)
    monkeypatch.setattr(gr.time, "sleep", lambda _: None)
    agent.exec("job", ["python", "train.py"], "/r/repo", {})
    assert len(calls) == 2
    assert all(FAKE_KEY.encode() not in req.data for req in calls)


def test_runtime_key_is_absent_from_real_upload_archive_and_boot_payload(tmp_path, monkeypatch):
    import base64
    import gzip
    import tarfile

    # A tiny committed tree exercises the real archive path without real model inputs.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "agent.py").write_bytes(gr.AGENT.read_bytes())
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "agent.py"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Offline Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    agent = gr.Agent("fake", "token")
    agent._self_delete_key = FAKE_KEY.encode()
    monkeypatch.setattr(
        gr, "read_self_delete_key", lambda: pytest.fail("archive must not read key")
    )
    inputs = bench.pb.build_bench_inputs(tmp_path / "uploads", "HEAD", repo=repo, ckpts=[])
    with tarfile.open(inputs["files"][0]["src"]) as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert FAKE_KEY.encode() not in archive.extractfile(member).read()
    body = gr.pod_body("fake-pod", "a" * 64, 1000, 2000)
    encoded_agent = body["dockerStartCmd"][2].split("echo ", 1)[1].split(" |", 1)[0]
    assert FAKE_KEY.encode() not in gzip.decompress(base64.b64decode(encoded_agent))
    _assert_no_secret(tmp_path / "uploads", inputs, body)


@pytest.mark.parametrize("probe", [403, 204, "200", FAKE_KEY])
def test_owner_key_still_requires_integer_200_probe(tmp_path, monkeypatch, capsys, probe):
    class Agent(training.FakeAgent):
        def set_self_delete_key(self, key):
            return {"self_delete_armed": True, "self_delete_probe": probe}

    monkeypatch.setattr(gr, "read_self_delete_key", lambda: FAKE_KEY)
    agent = Agent()
    outcome, state = training._drive(tmp_path, monkeypatch, agent, training._plan(), waiver=None)
    assert outcome.startswith("pod self-delete not proven")
    assert not agent.uploads and not agent.jobs
    _assert_no_secret(tmp_path, outcome, state, capsys.readouterr())


def test_bench_missing_key_still_aborts_without_waiver(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "read_self_delete_key", lambda: None)
    agent, rp = bench.FakeAgent(), bench.FakeRp()
    code, state, receipt, out, _, _ = bench._session(
        tmp_path, monkeypatch, rp, agent, accept_no_self_delete=None
    )
    assert code != 0 and receipt["outcome"] == "pod self-delete not proven (probe 403)"
    assert not agent.uploads and not agent.jobs
    assert state["pods_deleted_verified"]
    _assert_no_secret(out, state, receipt)
