"""Behavioral dispatch tests; all supervisor/tmux commands are local fixtures."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/ops/dispatch.sh"
EXACT = "AccountingError('assignment endpoints changed during allocation')"


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("fixture did not reach expected state")


@pytest.fixture
def fixture(tmp_path):
    home = tmp_path / "home"
    binaries = home / ".local/bin"
    binaries.mkdir(parents=True)
    # Only the read-only inventory is allowed; prove it runs inside the ops lock.
    colab = binaries / "colab"
    colab.write_text(f"#!{sys.executable}\n" + """import fcntl, json, os, sys
from pathlib import Path
if sys.argv[1:] != ['--auth', 'oauth2', 'sessions']:
    Path(os.environ['HOME'], 'real-colab-called').touch()
    raise SystemExit(99)
with (Path(os.environ['OPS_DIR'])/'alloc.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise SystemExit('inventory outside allocation lock')
if 'INVENTORY_RAW' in os.environ:
    print(os.environ['INVENTORY_RAW'])
    raise SystemExit(0)
sessions=json.loads(Path(os.environ['INVENTORY']).read_text())
for endpoint, hardware in sessions.items():
    print(f'[?] {endpoint} | Hardware: {hardware} | Shape: Standard | Variant: GPU')
if not sessions:
    print('No active sessions found on server.')
""")
    colab.chmod(0o755)
    tmux = binaries / "tmux"
    tmux.write_text(f"#!{sys.executable}\n" + """import os, subprocess, sys
if sys.argv[1] == 'has-session':
    raise SystemExit(0)
if os.environ.get('TMUX_FAIL'):
    raise SystemExit(2)
subprocess.Popen(['bash', '-c', sys.argv[-1]], stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
""")
    tmux.chmod(0o755)
    broker = tmp_path / "broker with spaces"
    transport = broker / "research/colab_cli"
    transport.mkdir(parents=True)
    launch = transport / "launch.sh"
    launch.write_text(f"#!{sys.executable}\n" + """import json, os, sys, time
from pathlib import Path
args=sys.argv[1:]
state=Path(args[args.index('--state-dir')+1])
action=args[-2]; manifest=json.loads(Path(args[-1]).read_text())
job=manifest['job_id']; output=Path(manifest['output_dir'])
with open(os.environ['TRACE'], 'a') as trace:
    trace.write(json.dumps({'action':action, 'job':job, 'output':str(output),
                           'manifest':sys.argv[-1],
                           'other':manifest['budget']['other_assignments']})+'\\n')
if action == 'validate':
    if os.environ.get('MUTATE_SOURCE'):
        Path(os.environ['MUTATE_SOURCE']).write_text('{}')
    if os.environ.get('MUTATE_LIVE') and sys.argv[-1].endswith('.live.json'):
        Path(sys.argv[-1]).write_text('{}')
    raise SystemExit(3 if os.environ.get('VALIDATE_FAIL') else 0)
assert action == 'run', action
if os.environ.get('NEXT_INVENTORY') and not job.endswith('-retry-1'):
    Path(os.environ['INVENTORY']).write_text(os.environ['NEXT_INVENTORY'])
output.mkdir()
(state/job).mkdir(parents=True)
(state/job/'assignment.json').write_text('{}')
time.sleep(float(os.environ.get('ALLOC_DELAY', '.05')))
mode=manifest.get('test_mode', 'success')
retry=job.endswith('-retry-1')
if mode == 'success' or (retry and mode == 'retry_success'):
    (output/'rate-calibration.json').write_text(json.dumps({'status':'PASS'}))
    (output/'receipt.json').write_text(json.dumps({'status':'PASS', 'stop_verified':True}))
    time.sleep(.05)
    raise SystemExit(0)
if mode == 'early_exit':
    raise SystemExit(5)
receipt={'status':'INCOMPLETE', 'failure_class':'ACCOUNTING',
         'error':manifest.get('test_error',
                    "AccountingError('assignment endpoints changed during allocation')"),
         'cleanup_error':manifest.get('test_cleanup'),
         'stop_verified':manifest.get('test_stop', True)}
(output/'receipt.json').write_text(json.dumps(receipt))
print('original failure evidence', flush=True)
time.sleep(float(os.environ.get('EXIT_DELAY', '.03')))
raise SystemExit(1)
""")
    launch.chmod(0o755)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(broker), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Ops Tests")
    git("config", "user.email", "ops@example.invalid")
    git("add", ".")
    git("commit", "-qm", "fixture transport")
    ops = tmp_path / "ops with spaces"
    ops.mkdir()
    (ops / "colab-cli-qualified.json").write_text(
        json.dumps(
            {
                "schema": "snake-dqn-colab-cli-qualified/v1",
                "tree_hash": git("rev-parse", "HEAD:research/colab_cli"),
            }
        )
    )
    manifest = tmp_path / "job.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "job",
                "commit": git("rev-parse", "HEAD"),
                "output_dir": str(tmp_path / "output"),
                "budget": {"other_assignments": {}},
                "resume": {
                    "unit_kind": "episode",
                    "ledger_path": "inputs/verified-ledger.json",
                    "skip_verified": True,
                    "max_outcome_blind_retries": 2,
                },
            }
        )
    )
    env = dict(
        os.environ,
        HOME=str(home),
        OPS_DIR=str(ops),
        TRACE=str(tmp_path / "trace"),
        PATH=f'{binaries}:{os.environ["PATH"]}',
        OPS_ALLOC_POLL_INTERVAL_SECONDS=".01",
        OPS_ALLOC_LOCK_TIMEOUT_SECONDS="3",
        INVENTORY=str(tmp_path / "inventory.json"),
    )
    (tmp_path / "inventory.json").write_text("{}")

    def inventory(sessions, records):
        (tmp_path / "inventory.json").write_text(json.dumps(sessions))
        account = home / ".config/colab-cli/snake"
        account.mkdir(parents=True, exist_ok=True)
        for path in account.glob("active-*.json"):
            path.unlink()
        for record in records:
            path = account / f"active-{record['session']}.json"
            path.write_text(json.dumps(record))

    def invoke(name="lane", *, capture=True, dry_run=False):
        argv = [str(SCRIPT), name, str(manifest), "--broker", str(broker), "--cwd", str(broker)]
        if dry_run:
            argv.append("--dry-run")
        if capture:
            return subprocess.run(argv, env=env, text=True, capture_output=True, timeout=10)
        return subprocess.Popen(
            argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def change(**values):
        value = json.loads(manifest.read_text())
        value.update(values)
        manifest.write_text(json.dumps(value))

    yield {
        "ops": ops,
        "broker": broker,
        "manifest": manifest,
        "env": env,
        "invoke": invoke,
        "change": change,
        "tmp": tmp_path,
        "git": git,
        "inventory": inventory,
    }
    # Drain only fixture jobs; each has a short finite workload and no provider.
    logs = list((ops / "logs").glob("*.log")) if (ops / "logs").exists() else []
    for log in logs:
        wait_for(lambda: "EXIT=" in log.read_text() or env.get("TMUX_FAIL"))
    assert not (home / "real-colab-called").exists()


@pytest.mark.parametrize("invalid", [False, True])
def test_dry_run_validates_without_reservation_or_allocation(fixture, invalid):
    f = fixture
    before = {p: p.read_bytes() for p in f["ops"].rglob("*") if p.is_file()}
    if invalid:
        f["env"]["VALIDATE_FAIL"] = "1"
    result = f["invoke"](dry_run=True)
    assert result.returncode == (1 if invalid else 0), result.stderr
    if not invalid:
        assert "PASS_DRY_RUN" in result.stdout
    trace = [json.loads(row) for row in (f["tmp"] / "trace").read_text().splitlines()]
    assert [row["action"] for row in trace] == ["validate"]
    assert {p: p.read_bytes() for p in f["ops"].rglob("*") if p.is_file()} == before
    assert not (f["ops"] / "dispatch").exists()
    assert not (f["tmp"] / "output").exists()


def test_dry_run_refuses_used_dispatch_name(fixture):
    f = fixture
    (f["ops"] / "dispatch/lane").mkdir(parents=True)
    result = f["invoke"](dry_run=True)
    assert result.returncode == 1 and "identity already used" in result.stderr
    assert not (f["tmp"] / "trace").exists()


def runs(f):
    trace = f["tmp"] / "trace"
    return (
        [
            json.loads(row)
            for row in trace.read_text().splitlines()
            if json.loads(row)["action"] == "run"
        ]
        if trace.exists()
        else []
    )


def active(endpoint, rate, **fields):
    return {
        "endpoint": endpoint,
        "session": f"snake-{endpoint}",
        "rate_units_per_hour": rate,
        **fields,
    }


@pytest.mark.parametrize(
    "declared,sessions,records,expected",
    [
        ({"departed": 1.54}, {}, [], {}),
        (
            {"departed": 1.54},
            {"new": "L4"},
            [active("new", 2.0, priced_rate_units_per_hour=2.0, observed_rate_units_per_hour=1.54)],
            {"new": 1.54},
        ),
        (
            {},
            {"a100": "A100", "cpu": "CPU"},
            [
                active("a100", 10, priced_rate_units_per_hour=10, observed_rate_units_per_hour=5.3),
                active("cpu", 0.08),
            ],
            {"a100": 5.3, "cpu": 0.08},
        ),
    ],
)
def test_refreshes_departed_and_new_owned_endpoints(fixture, declared, sessions, records, expected):
    f = fixture
    f["change"](budget={"other_assignments": declared, "unit_cap": 3, "reserve_units": 1})
    original = f["manifest"].read_bytes()
    f["inventory"](sessions, records)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    run = runs(f)[0]
    assert run["other"] == expected
    live = Path(run["manifest"])
    assert live.name == "lane.live.json"
    source = json.loads(original)
    source["budget"]["other_assignments"] = expected
    assert json.loads(live.read_bytes()) == source
    assert f["manifest"].read_bytes() == original
    assert (f["ops"] / "manifests/lane.json").read_bytes() == original
    proof = json.loads((f["ops"] / "dispatch/lane/lane.accounting.json").read_bytes())
    assert proof["manifest_sha256"] == hashlib.sha256(live.read_bytes()).hexdigest()
    assert proof["source_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    for row in proof["broker_records"]:
        assert row["sha256"] == hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
    traces = [json.loads(row) for row in (f["tmp"] / "trace").read_text().splitlines()]
    assert [row["action"] for row in traces] == ["validate", "validate", "run"]
    assert traces[1]["manifest"] == run["manifest"]


@pytest.mark.parametrize("declared", [{}, {"foreign": 1.54}])
def test_refuses_foreign_endpoint_even_if_manifest_declares_it(fixture, declared):
    f = fixture
    f["change"](budget={"other_assignments": declared})
    f["inventory"]({"foreign": "L4"}, [])
    result = f["invoke"]()
    assert result.returncode == 1 and "not owned by our broker: foreign" in result.stderr
    assert not runs(f)
    assert not (f["ops"] / "watch-logs.txt").exists()


@pytest.mark.parametrize(
    "tail",
    [
        "[?] foreign | Hardware:",
        "[?] foreign | Unexpected: L4",
        "No active sessions found on server.",
    ],
)
def test_refuses_mixed_valid_and_unparsed_inventory(fixture, tail):
    f = fixture
    f["inventory"]({"owned": "L4"}, [active("owned", 1.54)])
    f["env"]["INVENTORY_RAW"] = "[?] owned | Hardware: L4 | Shape: Standard\n" + tail
    result = f["invoke"]()
    assert result.returncode == 1 and "account session" in result.stderr
    assert not runs(f)


@pytest.mark.parametrize(
    "hardware,record",
    [
        ("L4", active("owned", 2, priced_rate_units_per_hour=2, observed_rate_units_per_hour=None)),
        ("L4", active("owned", 2)),
        ("CPU", active("owned", 0)),
        ("A100", active("owned", True)),
        ("A100", active("owned", float("nan"))),
        ("H100", active("owned", 8)),
    ],
)
def test_refuses_unpriced_or_unregistered_owned_endpoint(fixture, hardware, record):
    f = fixture
    f["inventory"]({"owned": hardware}, [record])
    result = f["invoke"]()
    assert result.returncode == 1
    assert not runs(f)


def test_live_manifest_is_create_only(fixture):
    f = fixture
    # Refuse a preexisting sibling even when its contents would be identical.
    target = f["ops"] / "manifests/lane.live.json"
    target.parent.mkdir()
    target.write_text("immutable evidence")
    result = f["invoke"]()
    assert result.returncode == 1 and "File exists" in result.stderr
    assert target.read_text() == "immutable evidence"
    assert not runs(f)


def test_live_manifest_tampering_after_validation_refuses_launch(fixture):
    f = fixture
    f["env"]["MUTATE_LIVE"] = "1"
    result = f["invoke"]()
    assert result.returncode == 1 and "refreshed manifest changed" in result.stderr
    assert not runs(f)


def test_exact_retry_refreshes_inventory_again(fixture):
    f = fixture
    f["change"](test_mode="retry_success")
    f["inventory"]({}, [active("arriving", 1.54)])
    f["env"]["NEXT_INVENTORY"] = json.dumps({"arriving": "L4"})
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert [row["other"] for row in runs(f)] == [{}, {"arriving": 1.54}]
    proof = f["ops"] / "dispatch/lane/lane-retry-1.accounting.json"
    assert json.loads(proof.read_text())["other_assignments"] == {"arriving": 1.54}


def test_lock_held_after_early_assignment_until_calibration(fixture):
    f = fixture
    f["env"]["ALLOC_DELAY"] = ".3"
    process = f["invoke"](capture=False)
    wait_for(lambda: (f["ops"] / "state/lane/job/assignment.json").exists())
    with (f["ops"] / "alloc.lock").open("a") as lock:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stderr
    assert "allocation=success" in stdout
    with (f["ops"] / "alloc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert "OPS_ALLOCATION_SUCCESS" in (f["ops"] / "logs/lane.log").read_text()


def test_one_exact_retry_fresh_outputs_preserves_original_evidence(fixture):
    f = fixture
    f["change"](test_mode="retry_success")
    before = f["manifest"].read_bytes()
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    assert [r["job"] for r in runs(f)] == ["job", "job-retry-1"]
    retry = json.loads((f["ops"] / "manifests/lane-retry-1.json").read_text())
    assert retry["output_dir"] == str(f["tmp"] / "output-retry-1")
    assert retry["resume"] == json.loads(before)["resume"]
    assert f["manifest"].read_bytes() == before
    first = (f["ops"] / "logs/lane.log").read_text()
    assert "original failure evidence" in first and "EXIT=1" in first
    assert "OPS_ALLOCATION_RETRY" in first
    assert "OPS_ALLOCATION_SUCCESS" in (f["ops"] / "logs/lane-retry-1.log").read_text()
    assert len((f["ops"] / "watch-logs.txt").read_text().splitlines()) == 2


@pytest.mark.parametrize(
    "values",
    [
        {"test_error": EXACT + " extra"},
        {"test_error": "Refused('assignment endpoints changed during allocation')"},
        {"test_stop": False},
        {"test_cleanup": "unverified stop"},
        {"test_mode": "early_exit"},
    ],
)
def test_other_failures_never_retry(fixture, values):
    f = fixture
    f["change"](test_mode="retry_success")
    f["change"](**values)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert len(runs(f)) == 1


def test_second_exact_failure_never_gets_third_attempt(fixture):
    f = fixture
    f["change"](test_mode="always_exact_failure")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert len(runs(f)) == 2


def test_refuses_held_lock_and_validation_failure(fixture):
    f = fixture
    with (f["ops"] / "alloc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = f["invoke"]()
    assert result.returncode != 0 and "lock is held" in result.stderr
    assert not runs(f)
    f["env"]["VALIDATE_FAIL"] = "1"
    assert f["invoke"]("invalid").returncode != 0
    assert not runs(f)


def test_refuses_dirty_or_unqualified_transport(fixture):
    f = fixture
    path = f["broker"] / "research/colab_cli/new.py"
    path.write_text("change")
    assert f["invoke"]().returncode != 0
    f["git"]("add", ".")
    f["git"]("commit", "-qm", "new transport")
    assert f["invoke"]("changed").returncode != 0
    assert not runs(f)


def test_timeout_is_bounded_and_sigkill_does_not_strand_lock(fixture):
    f = fixture
    f["env"]["ALLOC_DELAY"] = "1.8"
    f["env"]["OPS_ALLOC_LOCK_TIMEOUT_SECONDS"] = "1"
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=timeout" in result.stdout
    f["env"]["OPS_ALLOC_LOCK_TIMEOUT_SECONDS"] = "3"
    f["change"](output_dir=str(f["tmp"] / "killed-output"))
    process = f["invoke"]("killed", capture=False)
    wait_for(lambda: (f["ops"] / "state/killed/job/assignment.json").exists())
    process.kill()
    process.communicate(timeout=5)
    with (f["ops"] / "alloc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_waits_for_terminal_exit_before_retry(fixture):
    f = fixture
    f["change"](test_mode="retry_success")
    f["env"]["EXIT_DELAY"] = ".3"
    process = f["invoke"](capture=False)
    wait_for(lambda: (f["tmp"] / "output/receipt.json").exists())
    time.sleep(0.05)
    assert len(runs(f)) == 1
    assert process.communicate(timeout=10)[1] == ""
    assert process.returncode == 0
    assert len(runs(f)) == 2


def test_refuses_unsafe_names_and_blocked_instructions(fixture):
    f = fixture
    assert f["invoke"]("../unsafe").returncode != 0
    (f["tmp"] / "DISPATCH.txt").write_text("HOLD: operator review\n")
    assert f["invoke"]().returncode != 0
    assert not runs(f)


def test_completed_job_receipt_does_not_turn_allocation_success_into_failure(fixture):
    f = fixture
    f["env"]["OPS_ALLOC_POLL_INTERVAL_SECONDS"] = ".2"
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    assert json.loads((f["tmp"] / "output/receipt.json").read_text())["status"] == "PASS"
    assert "OPS_ALLOCATION_FAILURE" not in (f["ops"] / "logs/lane.log").read_text()


def test_refuses_unqualified_transport_pinned_by_manifest_even_with_qualified_broker(fixture):
    f = fixture
    launch = f["broker"] / "research/colab_cli/launch.sh"
    original = launch.read_text()
    launch.write_text(original + "\n# unqualified runtime change\n")
    f["git"]("add", ".")
    f["git"]("commit", "-qm", "unqualified runtime")
    bad = f["git"]("rev-parse", "HEAD")
    launch.write_text(original)
    f["git"]("add", ".")
    f["git"]("commit", "-qm", "restore qualified transport")
    f["change"](commit=bad)
    result = f["invoke"]()
    assert result.returncode != 0
    assert "COLAB_CLI_UNQUALIFIED" in result.stderr
    assert not runs(f)


def test_long_job_id_still_gets_one_valid_distinct_retry(fixture):
    f = fixture
    f["change"](job_id="j" * 64, test_mode="retry_success")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    job_ids = [row["job"] for row in runs(f)]
    assert len(job_ids) == 2 and len(job_ids[1]) <= 64 and job_ids[1] != job_ids[0]


def test_snapshot_is_used_even_if_source_manifest_changes_during_validation(fixture):
    f = fixture
    original = f["manifest"].read_bytes()
    f["env"]["MUTATE_SOURCE"] = str(f["manifest"])
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    assert f["manifest"].read_bytes() != original
    assert (f["ops"] / "manifests/lane.json").read_bytes() == original
    assert runs(f)[0]["output"] == str(f["tmp"] / "output")


def test_retry_directory_is_sibling_when_original_has_trailing_slash(fixture):
    f = fixture
    f["change"](test_mode="retry_success", output_dir=str(f["tmp"] / "output") + "/")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert [row["output"] for row in runs(f)] == [
        str(f["tmp"] / "output"),
        str(f["tmp"] / "output-retry-1"),
    ]
    assert not (f["tmp"] / "output/-retry-1").exists()


@pytest.mark.parametrize(
    "resume",
    [
        None,
        False,
        [],
        "episode",
        {},
        {"unit_kind": "episode", "ledger_path": "ledger.json", "skip_verified": True},
        {
            "unit_kind": "episode",
            "ledger_path": "ledger.json",
            "max_outcome_blind_retries": 2,
        },
    ],
)
def test_resume_contract_required_before_any_operational_side_effect(fixture, resume):
    f = fixture
    f["change"](resume=resume)
    result = f["invoke"]()
    assert result.returncode == 1 and "resume" in result.stderr
    assert not (f["tmp"] / "trace").exists()
    assert not (f["ops"] / "dispatch").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit_kind", ""),
        ("unit_kind", "  "),
        ("unit_kind", ["episode"]),
        ("ledger_path", None),
        ("ledger_path", "\t"),
        ("skip_verified", False),
        ("skip_verified", "true"),
        ("skip_verified", 1),
        ("max_outcome_blind_retries", True),
        ("max_outcome_blind_retries", 2.0),
        ("max_outcome_blind_retries", "2"),
        ("max_outcome_blind_retries", 0),
        ("max_outcome_blind_retries", 1),
        ("max_outcome_blind_retries", 3),
    ],
)
def test_refuses_invalid_resume_fields(fixture, field, value):
    f = fixture
    resume = json.loads(f["manifest"].read_text())["resume"]
    resume[field] = value
    f["change"](resume=resume)
    result = f["invoke"]()
    assert result.returncode == 1 and f"resume.{field}" in result.stderr
    assert not (f["tmp"] / "trace").exists()


@pytest.mark.parametrize("field", ["unit_kind", "ledger_path"])
def test_refuses_missing_resume_identity_fields(fixture, field):
    f = fixture
    resume = json.loads(f["manifest"].read_text())["resume"]
    del resume[field]
    f["change"](resume=resume)
    result = f["invoke"]()
    assert result.returncode == 1 and f"resume.{field}" in result.stderr
    assert not runs(f)


def without_resume(f):
    value = json.loads(f["manifest"].read_text())
    del value["resume"]
    f["manifest"].write_text(json.dumps(value))


@pytest.mark.parametrize("kind,minutes", [("short-canary", 40), ("acceptance", 60)])
def test_explicit_operational_exemptions_pass_and_are_preserved(fixture, kind, minutes):
    f = fixture
    without_resume(f)
    exemption = {"kind": kind, "reason": "Bounded transport check; no scoring look"}
    f["change"](resume_exemption=exemption, timeout_minutes=minutes)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=success" in result.stdout
    saved = json.loads((f["ops"] / "manifests/lane.json").read_text())
    assert saved["resume_exemption"] == exemption
    assert "resume" not in saved


@pytest.mark.parametrize(
    "exemption,timeout",
    [
        (None, 10),
        ({"kind": "production", "reason": "test"}, 10),
        ({"kind": "short-canary"}, 10),
        ({"kind": "acceptance", "reason": " "}, 10),
        ({"kind": "short-canary", "reason": "test"}, 40.01),
        ({"kind": "acceptance", "reason": "test"}, 60.01),
        ({"kind": "short-canary", "reason": "test"}, None),
        ({"kind": "acceptance", "reason": "test"}, True),
        ({"kind": "acceptance", "reason": "test"}, "20"),
        ({"kind": "acceptance", "reason": "test"}, -1),
        ({"kind": "acceptance", "reason": "test"}, 0),
        ({"kind": "acceptance", "reason": "test"}, float("inf")),
        ({"kind": "acceptance", "reason": "test"}, float("nan")),
    ],
)
def test_invalid_exemption_cannot_bypass_resume_gate(fixture, exemption, timeout):
    f = fixture
    without_resume(f)
    f["change"](resume_exemption=exemption, timeout_minutes=timeout)
    result = f["invoke"]()
    assert result.returncode == 1 and "resume" in result.stderr
    assert not (f["tmp"] / "trace").exists()


def test_exemption_cannot_hide_malformed_resume_contract(fixture):
    f = fixture
    f["change"](
        resume=None,
        resume_exemption={"kind": "short-canary", "reason": "test"},
        timeout_minutes=10,
    )
    result = f["invoke"]()
    assert result.returncode == 1 and "never both" in result.stderr
    assert not runs(f)


def test_canary_name_and_env_do_not_implicitly_exempt_manifest(fixture):
    f = fixture
    without_resume(f)
    f["change"](job_id="acceptance-canary", env={"COLAB_ACCEPTANCE": "1"}, timeout_minutes=10)
    result = f["invoke"]("short-canary")
    assert result.returncode == 1 and "requires a resume block" in result.stderr
    assert not runs(f)


@pytest.mark.parametrize(
    "requirements", [{"gpu": "A100"}, {"gpu": "L4", "allowed_gpus": ["L4", "A100"]}]
)
def test_a100_gpu_canary_missing_compile_pass_refuses_before_side_effects(fixture, requirements):
    fixture["change"](
        requirements=requirements,
        script="research/gpu_sim_spike/colab.sh",
        canary={"mode": "probe"},
    )
    result = fixture["invoke"]()
    assert result.returncode != 0
    assert "requires native CUDA compile PASS" in result.stderr
    assert not (fixture["ops"] / "dispatch").exists()
    assert not (fixture["tmp"] / "trace").exists()
