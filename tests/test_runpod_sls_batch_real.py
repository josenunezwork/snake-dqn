"""REAL episodes through the serverless handler: a K=3 batch vs the same units one at a time.

No network and no RunPod: ``sls_handler.handle`` runs in-process against a local fake
volume (the git archive of the e2e job commit 373c301 + the four allow-listed checkpoints),
and each episode is the job commit's own ``research.runpod_fanout.episode`` in a subprocess
(the project venv's python stands in for the volume's ``evenv``; ``clean_env`` + the runner's
numerics env apply exactly as on a worker). The episodes are the 9-episode e2e job
(``e2e-sls``: v8 screen arms A/B/C x frozen/scripted/mixed, one world, H1000), i.e. three
world units of three episodes.

Records must be equal after removing timing fields (any key containing ``seconds``) — the
runner's duplicate rule (``jobspec.deterministic_bytes``) — and that holds for the WHOLE
entry here, platform and fanout stamps included: the only bytes that differ between a
batch and a one-unit job are wall-clock timings.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

import pytest

from research.runpod_fanout import (
    jobspec,
    platform_rule,
    runner,
    serverless,
    sls_handler,
)
from research.runpod_fanout.wrappers import HERO_SHA256
from tests import test_runpod_serverless as tsl
from tests.test_runpod_fanout import Clock

pytestmark = pytest.mark.slow
no_network = tsl.no_network  # autouse: RunPod calls / non-loopback sockets are fatal

E2E_COMMIT = "373c3011d49dedbe73ba4e4c4aa0c18c588549b4"  # e2e-sls / e2e-probe job commit
SEED = 131496057
OPP = (
    "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195",
    "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d",
    "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
)
GREEDY = "e1202f168b09b1b5ca94dbdcfafb511206595f834b9af8110afaee06de78dd41"
RANDOM = "ae30b423bef16f0cf99eae0322b6f19cf1eb1f24d2cf4c8c207a7551ffc045e4"
ROSTERS = {
    "frozen": [HERO_SHA256, OPP[0], OPP[1], OPP[2], HERO_SHA256],
    "scripted": [GREEDY] * 5,
    "mixed": [HERO_SHA256, RANDOM, OPP[0], RANDOM, OPP[1]],
}


def e2e_episodes(horizon=1000):
    return [
        {
            "wrapper": "apex-veto-v8-screen-v1",
            "arm": arm,
            "mix": mix,
            "world_seed": SEED,
            "world_index": 0,
            "horizon": horizon,
            "engine": "live",
            "roster_member_sha256s": list(ROSTERS[mix]),
        }
        for mix in ("frozen", "scripted", "mixed")
        for arm in "ABC"
    ]


@pytest.fixture
def real_volume(tmp_path, monkeypatch):
    repo = runner.REPO
    if jobspec.git(repo, "cat-file", "-e", f"{E2E_COMMIT}^{{commit}}").returncode != 0:
        pytest.skip("e2e commit 373c301 not in this clone")
    fp = dict(jobspec.load_policy(), artifacts_root=str(tmp_path / "artifacts"))
    allow = jobspec.load_allowlist()
    try:
        ckpts = jobspec.resolve_checkpoints(sorted(allow), fp, allow)
    except jobspec.JobError as exc:
        pytest.skip(f"allow-listed checkpoints not available locally: {exc}")
    sp = dict(serverless.load_sls_policy(), poll_seconds=5, cold_start_seconds_est=60)
    root = tmp_path / "vol" / "rpf"
    cdir = root / "commits" / E2E_COMMIT
    cdir.mkdir(parents=True)
    repo_sha = jobspec.git_archive(repo, E2E_COMMIT, cdir / "repo.tar.gz")
    (cdir / "manifest.json").write_text(json.dumps({"commit": E2E_COMMIT, "repo_sha256": repo_sha}))
    (root / "ckpt").mkdir()
    for sha, path in ckpts.items():
        os.symlink(path, root / "ckpt" / f"{sha}.pth")
    rid = serverless.runtime_id(fp, sp)
    rdir = root / "runtime" / rid
    rdir.mkdir(parents=True)
    ready = {"runtime_id": rid, "handler_sha256": serverless.handler_sha256()}
    (rdir / "READY.json").write_text(json.dumps(ready))
    monkeypatch.setattr(sls_handler, "ROOT", root)
    monkeypatch.setattr(sls_handler, "RUNTIME_DIR", rdir)
    monkeypatch.setattr(sls_handler, "LOCAL", tmp_path / "worker-local")
    sls_handler._VERIFIED.clear()
    # the volume's evenv python -> this project's venv python (same pins), same module path
    monkeypatch.setenv(
        "RPF_TEST_EPISODE_EXEC_JSON",
        json.dumps([sys.executable, "-m", "research.runpod_fanout.episode"]),
    )
    return {
        "fp": fp,
        "sp": sp,
        "allow": allow,
        "root": root,
        "rid": rid,
        "repo_sha": repo_sha,
        "ckpts": sorted(ckpts),
        "ready": ready,
        "tmp": tmp_path,
    }


def handler_input(v, episodes, slots):
    return {
        "op": "episodes",
        "runtime_id": v["rid"],
        "commit": E2E_COMMIT,
        "repo_sha256": v["repo_sha"],
        "checkpoints": v["ckpts"],
        "job_id": "e2e-sls",
        "provider": "runpod-serverless:cpu5c/cpu3c",
        "slots": slots,
        "timeout_seconds": 900,
        "deadline_epoch": time.time() + 3600,
        "require_cpu_model": None,
        "numerics_env": runner.numerics_env(v["fp"]),
        "episodes": [{"key": jobspec.episode_key(e), "spec": e} for e in episodes],
    }


def records(out):
    assert out.get("ok") is True, out
    got = {}
    for r in out["results"]:
        assert r["ok"], {k: r.get(k) for k in ("key", "rc", "status", "stderr_tail")}
        data = r["record"].encode()
        assert hashlib.sha256(data).hexdigest() == r["sha256"]
        got[r["key"]] = data
    return got


def timing_paths(a, b, path=""):
    """JSON paths where ``a`` and ``b`` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += timing_paths(a.get(k), b.get(k), f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in timing_paths(x, y, f"{path}/{i}")]
    return [] if a == b else [path]


def test_k3_batch_equals_units_one_at_a_time_handler_level(real_volume):
    v = real_volume
    eps = e2e_episodes()
    units = platform_rule.group_units(jobspec.episode_key(e) for e in eps)
    assert len(units) == 3 and all(len(k) == 3 for k in units.values())
    batch = records(sls_handler.handle(handler_input(v, eps, slots=9)))  # K=3, one wave
    single = {}
    for keys in units.values():
        mine = [e for e in eps if jobspec.episode_key(e) in keys]
        single.update(records(sls_handler.handle(handler_input(v, mine, slots=3))))
    assert sorted(batch) == sorted(single) and len(batch) == 9
    for key in sorted(batch):
        a, b = json.loads(batch[key]), json.loads(single[key])
        assert jobspec.deterministic_bytes(a) == jobspec.deterministic_bytes(b), key
        # the WHOLE entry (platform + fanout stamps too) is equal up to timing fields
        assert jobspec.strip_timing(a) == jobspec.strip_timing(b), key
        assert all("seconds" in p for p in timing_paths(a, b)), timing_paths(a, b)
        assert a["fanout"]["repo_commit"] == E2E_COMMIT and a["fanout"]["horizon"] == 1000
    # the three units are distinct worlds' arms, each with nontrivial play
    masses = {json.loads(batch[k])["record"]["mass_integral"] for k in batch}
    assert len(masses) > 1


def make_runner(v, name, units_per_job, fake):
    j = {
        "schema": jobspec.JOB_SCHEMA,
        "job_id": "e2e-sls",
        "tier": "tier1",
        "purpose": "Tier-1 serverless batch test (e2e-sls episodes, commit 373c301)",
        "repo_commit": E2E_COMMIT,
        "max_wall_minutes": 40.0,
        "checkpoints": v["ckpts"],
        "episodes": e2e_episodes(),
    }
    j = jobspec.validate_job(j, v["fp"], v["allow"])
    reg = serverless.Registry(v["fp"])
    with reg.locked() as st:
        st["volume"] = {"id": "vol1", "dataCenterId": "EU-RO-1", "size": 10}
        st["commits"][E2E_COMMIT] = {"repo_sha256": v["repo_sha"]}
        for sha in v["ckpts"]:
            st["ckpts"][sha] = "t"
        st["runtimes"][v["rid"]] = {"ready": v["ready"], "template_id": "tpl1"}
    clock = Clock()
    fp = dict(v["fp"], artifacts_root=str(v["tmp"] / f"art-{name}"))
    return serverless.ServerlessRunner(
        j,
        fp,
        v["allow"],
        v["tmp"] / "runs" / name,
        sls_policy=v["sp"],
        sizing={"workers": 1, "vcpu_per_worker": 32, "units_per_job": units_per_job},
        flavors=["cpu5c", "cpu3c"],
        registry=reg,
        rp=fake,
        budget=3.0,
        confirm=True,
        repo=runner.REPO,
        clock=clock,
        sleep=clock.sleep,
        spawn_watchdog=lambda r: None,
        preflight_fn=lambda: [],
        probe_workers=1,
    )


def test_k3_batch_run_equals_k1_run_runner_level(real_volume):
    """The whole runner (dispatch, collect, per-unit publish) on real episodes: one job of
    three units (auto, one 32-vCPU worker) vs three one-unit jobs (K=1)."""
    v = real_volume
    out = {}
    for name, k in (("batch", "auto"), ("k1", 1)):
        fake = tsl.FakeSls(models=[tsl.MODEL])
        r = make_runner(v, name, k, fake)
        assert r.run() == 0, r.stop_reason
        eps = [j["input"] for j in fake.jobs.values() if j["input"]["op"] == "episodes"]
        units_per_job = sorted(
            len({platform_rule.unit_of_key(e["key"]) for e in i["episodes"]}) for i in eps
        )
        out[name] = (units_per_job, runner.load_records(r.run_dir), r.run_dir)
        shutil.rmtree(sls_handler.LOCAL / "jobs", ignore_errors=True)
    assert out["batch"][0] == [3] and out["k1"][0] == [1, 1, 1]
    a, b = out["batch"][1], out["k1"][1]
    assert sorted(a) == sorted(b) and len(a) == 9
    for key in a:
        assert jobspec.deterministic_bytes(a[key]) == jobspec.deterministic_bytes(b[key]), key
    assert platform_rule.check_per_world(a) == [] and platform_rule.check_per_world(b) == []
    cmp = runner.compare_runs(out["batch"][2], out["k1"][2])  # runner.py compare
    assert cmp["compared"] == 9 and cmp["identical"] == 9 and cmp["different"] == []
