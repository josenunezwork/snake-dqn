"""Prepare, execute once, and independently reduce O1 retention acceptance.

Preparation is offline. Execution is an explicit operator action; it never
publishes a qualification registry. Production still needs independent review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

AGGREGATE_HOURS = 4.8
AGGREGATE_UNITS = 48.0
RESERVE_UNITS = 9.6
MANIFEST_MINUTES = 20
A100_RATE_UNITS_PER_HOUR = 10.0
REVIEW_TOKEN = "A100_RETENTION_ACCEPTANCE_REVIEWED"
ACCEPTANCE_AUTHORITY = "round-36-a100-retention-20261023"
RULING_SHA256 = "156c80b412064defd88281424e4c84e6a732d9cb8bc6e163561ef352ffb5fe3b"
REQUIRED_FAULTS = (
    "mac_disappearance_lease_expiry",
    "stale_generation",
    "child_failure",
    "cap_during_handoff",
    "accounting_refusal",
    "export_failure",
    "interrupted_second_job_launch",
    "recovery",
)
SCHEMA = "snake-dqn-a100-retention-acceptance-plan/v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create_only(path: Path, value: Any) -> None:
    """Create immutable review material; never overwrite historical evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def bound_json(value: dict[str, str]) -> dict[str, Any]:
    path = Path(value.get("path", value.get("manifest", "")))
    if not path.is_absolute() or sha256_file(path) != value["sha256"]:
        raise ValueError("absent or changed SHA-bound acceptance evidence")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("acceptance evidence must be an object")
    return result


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo, timeout=30)


def executable_inventory(repo: Path) -> dict[str, str]:
    """Bind every maintained transport/ops executable, including added files."""
    return {
        str(path.relative_to(repo)): sha256_file(path)
        for folder in ("research/colab_cli", "tools/ops")
        for path in sorted((repo / folder).rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"}
    }


def verify_inventory(repo: Path, commit: str, expected: dict[str, str]) -> None:
    if not expected or executable_inventory(repo) != expected:
        raise ValueError("retention acceptance executable inventory changed")
    for relative, digest in expected.items():
        if sha256_bytes(git(repo, "show", f"{commit}:{relative}")) != digest:
            raise ValueError(f"acceptance source differs from pinned commit: {relative}")


def _native_manifest(repo: Path, run: Path, commit: str, packet: str) -> dict[str, Any]:
    inputs = run / "inputs"
    return {
        "job_id": f"o1-{sha256_bytes(str(run).encode())[:12]}-{packet}",
        "commit": commit,
        "script": "research/colab_cli/retention_acceptance_probe.sh",
        "source_paths": ["src", "research/colab_cli", "pyproject.toml"],
        "env": {
            "PYTHONUNBUFFERED": "1",
            "SNAKE_DQN_DEVICE": "cuda",
            "COLAB_CLI_REQUIREMENTS": "research/colab_cli/retention_requirements.txt",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "COLAB_RETENTION_FAULT": "none",
            "COLAB_RETENTION_PACKET": packet,
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 1,
            "est_minutes": 10,
        },
        "timeout_minutes": MANIFEST_MINUTES,
        "output_dir": str(run / "outputs" / packet),
        "inputs": [
            {**binding(inputs / "probe-input.json"), "dest": "probe-input.json"},
            {**binding(inputs / "verified-ledger.json"), "dest": "verified-ledger.json"},
        ],
        "budget": {
            "unit_cap": A100_RATE_UNITS_PER_HOUR * MANIFEST_MINUTES / 60,
            "rate_units_per_hour": A100_RATE_UNITS_PER_HOUR,
            "reserve_units": RESERVE_UNITS,
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 64 * 1024 * 1024,
        "required_exports": ["probe.json", "units/unit-000.json"],
        "resume_exemption": {
            "kind": "acceptance",
            "reason": "O1 bounded synthetic native transport proof",
            "authority_sha256": RULING_SHA256,
        },
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "main",
            "family": "research/colab_cli/retention_acceptance_probe.sh",
        },
    }


def prepare(
    repo: Path, output_root: Path, other_assignments: dict[str, float] | None = None
) -> Path:
    """Freeze every native packet and final-only regression before any data."""
    from research.colab_cli import acceptance

    repo, run = repo.resolve(), output_root.resolve()
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    inventory = executable_inventory(repo)
    verify_inventory(repo, commit, inventory)
    if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("prepare requires a clean committed candidate")
    run.mkdir(parents=True, exist_ok=False)
    (run / "inputs").mkdir()
    write_create_only(
        run / "inputs/probe-input.json",
        {
            "schema": "o1-probe/v1",
            "namespace": "o1-frozen-native-proof",
            "allocation_index": 0,
            "units": [0],
            "seed": 3601,
            "extension_source_sha256": sha256_file(repo / "research/colab_cli/retention_probe.cu"),
        },
    )
    write_create_only(
        run / "inputs/verified-ledger.json",
        {
            "schema": "o1-verified-ledger/v1",
            "namespace": "o1-frozen-native-proof",
            "allocation_index": 0,
            "verified_units": [],
        },
    )
    other_assignments = other_assignments or {}
    regression = acceptance.build_plan(
        repo, run / "regression", profile="final-only", other_assignments=other_assignments
    )
    # The ordinary observer/gates and unmodified acceptance execution are used.
    # Only the explicit bounded resume declaration is added before freezing.
    for row in regression["plan"]["jobs"]:
        path = Path(row["manifest"])
        value = json.loads(path.read_text())
        value["resume_exemption"] = {
            "kind": "acceptance",
            "reason": "O1 source-bound final-only regression",
        }
        # build_plan has not dispatched; publish a new reviewed manifest rather
        # than changing the initial generator's artifact.
        reviewed = path.with_name(path.stem + "-reviewed.json")
        write_create_only(reviewed, value)
        row["manifest"] = str(reviewed)
    regression_plan = run / "regression-plan.json"
    write_create_only(regression_plan, regression["plan"])
    templates = []
    packets = ["native-canary", "cold-1", "cold-2", "retained-1", "retained-2"]
    packets += ["fault-" + name.replace("_", "-") for name in REQUIRED_FAULTS]
    for packet in packets:
        m = _native_manifest(repo, run, commit, packet)
        m["budget"]["other_assignments"] = other_assignments
        if packet == "native-canary":
            m["canary"]["mode"] = "probe"
            m["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/colab_cli")
        if packet.startswith("fault-"):
            m["timeout_minutes"] = 8
            m["requirements"]["est_minutes"] = 4
            m["budget"]["unit_cap"] = 8 * A100_RATE_UNITS_PER_HOUR / 60
            m["env"]["COLAB_RETENTION_FAULT"] = packet[6:].replace("-", "_")
        path = run / "templates" / f"{packet}.json"
        write_create_only(path, m)
        templates.append({"packet": packet, "manifest": str(path), "sha256": sha256_file(path)})
    plan = {
        "schema": SCHEMA,
        "status": "NOT_RUN",
        "qualification": "INCOMPLETE_UNTIL_INDEPENDENT_PASS",
        "repo": str(repo),
        "commit": commit,
        "transport_tree": git(repo, "rev-parse", f"{commit}:research/colab_cli").decode().strip(),
        "source_inventory": inventory,
        "authority_sha256": RULING_SHA256,
        "regression_plan": binding(regression_plan),
        "regression_manifests": [
            binding(Path(row["manifest"])) for row in regression["plan"]["jobs"]
        ],
        "jobs": templates,
        "chain": {
            "chain_id": "o1-" + sha256_bytes(str(run).encode())[:24],
            "retention_minutes": 40,
            "retention_units": 40 * 10 / 60,
            "hardware": "A100",
        },
        "caps": {
            "aggregate_hours": 4.8,
            "aggregate_minutes": 288,
            "aggregate_units": 48.0,
            "reserve_units": RESERVE_UNITS,
            "manifest_timeout_minutes_max": 60,
            "first_cap_wins": True,
            "all_attempts_and_reserve_included": True,
        },
        "envelopes": {
            "forecast_a100_hours": 2.0,
            "base_hours": 2.4,
            "base_units": 24,
            "effective_hours": 4.8,
            "effective_units": 48,
            "frozen_max_minutes": 214,
            "frozen_job_unit_liability": 28.5,
        },
        "required_faults": list(REQUIRED_FAULTS),
        "acceptance_bars": {
            "second_setup_reduction_percent": 50,
            "allocated_wall_saving_percent": 25,
        },
        "mechanical_binding_only_after_canary": ["canary.receipt", "retention_chain"],
        "automatic_retries": 0,
    }
    path = run / "acceptance-plan.json"
    write_create_only(path, plan)
    write_create_only(
        run / "boundary-admission.template.json",
        {
            "schema": "o1-boundary-admission/v1",
            "plan": binding(path),
            "original_sealed_admission": None,
            "unencumbered_units": None,
            "unencumbered_runtime_minutes": None,
            "t0_epoch": None,
            "t0_json_pointer": None,
            "safe_a100_boundary": False,
            "required_work_not_displaced": False,
            "prior_attempt_receipts": [],
            "other_assignments": other_assignments,
            "note": "Operator creates a separate current admission; template is not authorization.",
        },
    )
    command = shlex.join(
        [
            str(repo / "tools/ops/dispatch.sh"),
            "--retention-acceptance",
            str(path),
            "--boundary-admission",
            str(run / "boundary-admission.json"),
            "--review-token",
            REVIEW_TOKEN,
            "--broker",
            str(repo),
            "--cwd",
            str(repo),
        ]
    )
    (run / "DISPATCH.txt").write_text(command + "\n")
    return run


def verify_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA or plan.get("authority_sha256") != RULING_SHA256:
        raise ValueError("unrecognized retention acceptance plan")
    repo = Path(plan["repo"])
    verify_inventory(repo, plan["commit"], plan["source_inventory"])
    if git(repo, "rev-parse", "HEAD").decode().strip() != plan["commit"]:
        raise ValueError("candidate HEAD changed after freeze")
    if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("candidate must remain clean")
    bound_json(plan["regression_plan"])
    for row in plan["regression_manifests"] + plan["jobs"]:
        m = bound_json(row)
        if m["commit"] != plan["commit"] or Path(m["output_dir"]).exists():
            raise ValueError("acceptance pin changed or output already used")
        for item in m.get("inputs", []):
            if "path" in item:
                bound_json(item) if Path(item["path"]).suffix == ".json" else _check_input(item)
    if plan["caps"]["aggregate_hours"] != 4.8 or plan["caps"]["aggregate_units"] != 48:
        raise ValueError("acceptance envelope changed")
    return plan


def _check_input(item: dict[str, str]) -> None:
    if sha256_file(Path(item["path"])) != item["sha256"]:
        raise ValueError("acceptance input changed")


def _positive(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("expected finite positive acceptance measurement")
    return float(value)


def check_boundary(plan_path: Path, plan: dict[str, Any], path: Path) -> dict[str, Any]:
    """Validate a separate current, hash-bound operator reservation."""
    value = json.loads(path.read_text())
    if value.get("schema") != "o1-boundary-admission/v1" or value.get("plan") != binding(plan_path):
        raise ValueError("boundary admission does not bind this acceptance plan")
    sealed = bound_json(value["original_sealed_admission"])
    pointer = value.get("t0_json_pointer")
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("original sealed T0 requires an explicit JSON pointer")
    original_t0: Any = sealed
    try:
        for part in pointer[1:].split("/"):
            if re.search(r"~(?![01])", part):
                raise ValueError("invalid JSON pointer escape")
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(original_t0, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", key):
                    raise ValueError("invalid JSON pointer array index")
                original_t0 = original_t0[int(key)]
            else:
                original_t0 = original_t0[key]
        if isinstance(original_t0, str):
            instant = datetime.fromisoformat(original_t0.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                raise ValueError("sealed T0 must include its timezone")
            original_t0 = instant.timestamp()
        original_t0 = _positive(original_t0)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("original sealed T0 is absent or malformed") from exc
    if _positive(value["t0_epoch"]) != original_t0 or original_t0 > time.time():
        raise ValueError("boundary T0 differs from original sealed T0 or is in the future")
    if (
        value.get("safe_a100_boundary") is not True
        or value.get("required_work_not_displaced") is not True
    ):
        raise ValueError(
            "operator must record a safe sole-A100 boundary without displacing required work"
        )
    if (
        _positive(value["unencumbered_units"]) < 48
        or _positive(value["unencumbered_runtime_minutes"]) < 288
    ):
        raise ValueError("O1 full effective reservation is unavailable")
    if time.time() > _positive(value["t0_epoch"]) + 120 * 3600:
        raise ValueError("original round-28 utility deadline elapsed")
    if time.time() + AGGREGATE_HOURS * 3600 > original_t0 + 120 * 3600:
        raise ValueError("full O1 effective envelope does not fit original utility deadline")
    if value.get("prior_attempt_receipts"):
        raise ValueError("completion needs a separately reviewed remaining-envelope package")
    rates = value.get("other_assignments")
    if not isinstance(rates, dict) or any(_positive(rate) <= 0 for rate in rates.values()):
        raise ValueError("reviewed other assignment rates required")
    return value


def _receipt(output: Path, *, success: bool = True, retained: bool = False) -> dict[str, Any]:
    """Read an actual immutable broker receipt and rehash every exported payload."""
    from research.colab_cli import supervisor

    value = json.loads((output / "receipt.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if value.get("manifest_sha256") != sha256_file(output / "manifest.json"):
        raise ValueError("receipt manifest digest differs")
    if value.get("commit") != manifest["commit"] or value.get("job_id") != manifest["job_id"]:
        raise ValueError("receipt source/job differs")
    if value.get("cleanup_error"):
        raise ValueError("receipt cleanup was not verified")
    if success and (
        value.get("error")
        or value.get("status") != ("COMPLETED_SESSION_RETAINED" if retained else "PASS")
    ):
        raise ValueError("packet did not complete as registered")
    if retained:
        if value.get("stop_verified") is not False:
            raise ValueError("retained packet falsely claims between-job release")
        release = json.loads((output / "session-release.json").read_text())
        if release.get("stop_verified") is not True or release.get("error"):
            raise ValueError("chain lacks final verified release")
    elif value.get("stop_verified") is not True:
        raise ValueError("standalone packet lacks verified release")
    accounting = value.get("accounting") or {}
    _positive(accounting.get("own_rate_units_per_hour"))
    _positive(value.get("elapsed_seconds"))
    for row in value.get("exports", []):
        target = output / "exports" / row["path"]
        if target.stat().st_size != row["size"] or supervisor.sha256(target) != row["sha256"]:
            raise ValueError("verified native export changed")
        if not Path(str(target) + ".complete").is_file():
            raise ValueError("native export completion marker absent")
    return value


def account_boundary(cli: Any, target: Path, *, own_endpoint: str | None = None) -> None:
    """Prove the native window contains only this acceptance A100."""
    raw = cli.call("sessions")
    rows = re.findall(r"^\[([^\]]+)\]\s+(\S+)\s+\| Hardware:\s*(\S+)", raw, re.M)
    if not rows and "No active sessions found on server." not in raw:
        raise ValueError("native boundary has unrecognized backend sessions")
    a100 = {endpoint for _, endpoint, hardware in rows if hardware == "A100"}
    expected = {own_endpoint} if own_endpoint else set()
    if a100 != expected or len(rows) > 3:
        raise ValueError("native acceptance requires the sole sprint A100 boundary")
    write_create_only(
        target, {"sessions_stdout": raw, "a100_endpoints": sorted(a100), "observed_at": time.time()}
    )


class FaultObserver:
    """Inject failures only into this package's owned synthetic assignment."""

    def __init__(self, fault: str, output: Path) -> None:
        self.fault, self.output = fault, output
        self.record: dict[str, Any] = {}

    def __call__(self, stage: str, context: dict[str, Any]) -> None:
        from research.colab_cli import supervisor

        if stage == "runtime_started":
            account_boundary(
                context["cli"],
                self.output / "native-boundary.json",
                own_endpoint=context["cli"].accounting.endpoint,
            )
        if stage != "ready":
            return
        cli, root, generation = context["cli"], context["root"], context["generation"]
        if self.fault in {"mac_disappearance_lease_expiry", "stale_generation"}:
            endpoint = cli.accounting.endpoint
            cli.stop_lease(wait=True)
            code = (
                "import importlib.util,json,time\nfrom pathlib import Path\n"
                f"s=importlib.util.spec_from_file_location('o1_fault',{root + '/runtime.py'!r})\n"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                f"m.read_state({root!r},{generation!r},renew=True,disable_contents=True)\n"
            )
            if self.fault == "stale_generation":
                code += (
                    f"p=Path({root!r})/'lease.json'\n"
                    "v=json.loads(p.read_text());v['generation']='o1-stale-generation'\n"
                    "p.write_text(json.dumps(v))\n"
                )
            began = time.monotonic()
            cli.poll_rpc(code + "result=True")
            # No kernel work or lease renewal follows this injection. Read-only
            # backend evidence must precede Broker.finally fallback cleanup.
            cli.renew = lambda: None
            expiry = began + (
                0 if self.fault == "stale_generation" else context["manifest"]["lease_seconds"]
            )
            deadline = min(expiry + 300, cli.deadline, cli.priced_deadline)
            readings = []
            while time.monotonic() < deadline:
                sessions = cli.sessions(timeout=min(30, deadline - time.monotonic()))
                readings.append({"at": time.monotonic(), "sessions": sessions})
                if endpoint not in sessions:
                    self.record = {
                        "fault": self.fault,
                        "endpoint": endpoint,
                        "generation": generation,
                        "expiry_monotonic": expiry,
                        "observed_monotonic": time.monotonic(),
                        "readings": readings,
                        "before_fallback_cleanup": True,
                    }
                    write_create_only(self.output / "fault-observation.json", self.record)
                    raise supervisor.Refused("O1 expected watchdog fault observed")
                time.sleep(min(5, max(0, deadline - time.monotonic())))
            raise supervisor.Refused("O1 watchdog disappearance bound FAILED")
        if self.fault == "accounting_refusal":
            original = cli.usage()
            altered = dict(original, rate=original["rate"] + 100)
            try:
                cli.accounting.check(altered, sessions=cli.sessions())
            except (ValueError, supervisor.Refused) as exc:
                self.record = {
                    "fault": self.fault,
                    "actual_usage": original,
                    "injected_usage": altered,
                    "error": repr(exc),
                }
                write_create_only(self.output / "fault-observation.json", self.record)
                raise supervisor.Refused("O1 expected accounting refusal") from exc
            raise supervisor.Refused("O1 accounting fault was not refused")
        if self.fault == "export_failure":
            from research.colab_cli.acceptance_faults import AcceptanceObserver

            observer = AcceptanceObserver(cli.oauth_python)
            observer._corrupt_export_rejection(cli, root, self.output)
            self.record = {"fault": self.fault, "observation": observer.records}
            write_create_only(self.output / "fault-observation.json", self.record)
            raise supervisor.Refused("O1 expected export rejection")
        # Chain-handoff, uncertain-launch and durable recovery are independently
        # executed as offline fault tests by Codex-V, on this exact inventory.


def _native_admission(
    plan_path: Path, plan: dict[str, Any], manifests: list[Path], final: Path, target: Path
) -> Path:
    value = {
        "schema": SCHEMA,
        "parent_plan": binding(plan_path),
        "commit": plan["commit"],
        "source_inventory": plan["source_inventory"],
        "final_only_result": binding(final),
        "retention_minutes": plan["chain"]["retention_minutes"],
        "retention_units": plan["chain"]["retention_units"],
        "caps": plan["caps"],
        "manifests": [
            {
                "path": str(path),
                "sha256": sha256_bytes(
                    json.dumps(
                        json.loads(path.read_text()), sort_keys=True, separators=(",", ":")
                    ).encode()
                ),
            }
            for path in manifests
        ],
    }
    write_create_only(target, value)
    return target


def verify_native_admission(path: Path, manifests: list[dict[str, Any]]) -> None:
    """Tie the native-only exception to the pre-data packet and released canary."""
    value = json.loads(path.read_text())
    parent_path = Path(value["parent_plan"]["path"])
    plan = bound_json(value["parent_plan"])
    run, repo = parent_path.parent, Path(plan["repo"])
    if plan.get("schema") != SCHEMA or plan.get("authority_sha256") != RULING_SHA256:
        raise ValueError("native admission lacks the frozen funded parent plan")
    if value["source_inventory"] != plan["source_inventory"]:
        raise ValueError("native admission inventory differs from frozen parent")
    verify_inventory(repo, plan["commit"], plan["source_inventory"])
    regression = bound_json(plan["regression_plan"])
    final = Path(regression["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json"
    if value["final_only_result"] != binding(final):
        raise ValueError("native regression differs from the frozen final-only packet")
    canary_row = next(row for row in plan["jobs"] if row["packet"] == "native-canary")
    canary_output = Path(bound_json(canary_row)["output_dir"])
    _receipt(canary_output)
    pins = json.loads((canary_output / "exports/probe.json").read_text())["bootstrap"]["pins"]
    if len(manifests) != 2:
        raise ValueError("O1 native exception admits exactly the frozen two main packets")
    for packet, actual in zip(("retained-1", "retained-2"), manifests):
        row = next(row for row in plan["jobs"] if row["packet"] == packet)
        expected = bound_json(row)
        expected["canary"]["receipt"] = binding(canary_output / "receipt.json")
        expected["retention_chain"] = {
            "enabled": True,
            "qualification": {},
            "cache": pins,
            "shared": {
                "chain_id": plan["chain"].get("chain_id", run.name),
                **{key: plan["chain"][key] for key in ("retention_minutes", "retention_units")},
            },
        }
        if expected != actual:
            raise ValueError("native exception differs from frozen main packet")


def execute(
    plan_path: Path, boundary_path: Path, review_token: str, *, colab: str, oauth_python: str
) -> Path:
    """Execute the exact reviewed series once; never adopt or retry on failure."""
    from research.colab_cli import acceptance, admission, supervisor

    if review_token != REVIEW_TOKEN:
        raise ValueError("exact retention acceptance review token required")
    plan = verify_plan(plan_path)
    boundary = check_boundary(plan_path, plan, boundary_path)
    run, repo = plan_path.parent, Path(plan["repo"])
    # Account maps are pre-data inputs. A changed live map requires fresh
    # preparation; it is never patched into a running immutable manifest.
    regression = bound_json(plan["regression_plan"])
    for row in regression["jobs"]:
        if (
            json.loads(Path(row["manifest"]).read_text())["budget"]["other_assignments"]
            != boundary["other_assignments"]
        ):
            raise ValueError("prepare a fresh package with the reviewed account map")
    write_create_only(
        run / "execute-started.json",
        {
            "plan": binding(plan_path),
            "boundary": binding(boundary_path),
            "started_at": time.time(),
        },
    )
    started = time.monotonic()
    absolute_deadline = boundary["t0_epoch"] + 120 * 3600
    events = []
    try:
        _series_remaining(run, started, 50, absolute_deadline)
        if acceptance.execute(
            regression,
            state_dir=run / "regression-state",
            colab=colab,
            review_token=acceptance.REVIEW_TOKEN,
            oauth_python=oauth_python,
        ):
            raise ValueError("final-only regression failed; no native allocation permitted")
        final = Path(regression["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json"
        admission.require_final_only(json.loads(final.read_text()), repo)
        rows = {row["packet"]: row for row in plan["jobs"]}
        canary = bound_json(rows["native-canary"])
        _series_remaining(run, started, 20, absolute_deadline)
        boundary_cli = supervisor.CLI(colab, run / "native-boundary-sessions.json")
        account_boundary(boundary_cli, run / "native-boundary-before.json")

        def observe_native(stage: str, context: dict[str, Any]) -> None:
            if stage == "runtime_started":
                account_boundary(
                    context["cli"],
                    context["output"] / "native-boundary.json",
                    own_endpoint=context["cli"].accounting.endpoint,
                )

        broker = supervisor.Broker(run / "native-state", colab, observer=observe_native)
        broker.run(canary)
        canary_output = Path(canary["output_dir"])
        _receipt(canary_output)
        canary_binding = binding(canary_output / "receipt.json")
        cache_pins = json.loads((canary_output / "exports/probe.json").read_text())["bootstrap"][
            "pins"
        ]
        # No retained canary promotion: its entire assignment was independently
        # released before any main packet gets admitted.
        native: dict[str, Path] = {}
        for packet, row in rows.items():
            if packet == "native-canary":
                continue
            m = bound_json(row)
            m["canary"]["receipt"] = canary_binding
            m["budget"]["other_assignments"] = boundary["other_assignments"]
            if packet.startswith("retained-"):
                m["retention_chain"] = {
                    "enabled": True,
                    "qualification": {},
                    "shared": {
                        "chain_id": plan["chain"].get("chain_id", run.name),
                        **{
                            key: plan["chain"][key]
                            for key in ("retention_minutes", "retention_units")
                        },
                    },
                    "cache": cache_pins,
                }
            target = run / "admitted" / f"{packet}.json"
            m = supervisor.validate(m, repo)
            write_create_only(target, m)
            native[packet] = target
        for packet in ("cold-1", "cold-2"):
            _series_remaining(run, started, 20, absolute_deadline)
            m = json.loads(native[packet].read_text())
            broker.run(m)
            _receipt(Path(m["output_dir"]))
        retained = [native["retained-1"], native["retained-2"]]
        native_plan = _native_admission(
            plan_path, plan, retained, final, run / "native-admission.json"
        )
        _series_remaining(run, started, 40, absolute_deadline)
        broker.chain(
            [json.loads(path.read_text()) for path in retained],
            plan["chain"]["retention_minutes"],
            plan["chain"]["retention_units"],
            acceptance_plan=native_plan,
        )
        for packet in ("retained-1", "retained-2"):
            _receipt(Path(json.loads(native[packet].read_text())["output_dir"]), retained=True)
        for fault in REQUIRED_FAULTS:
            if fault in {"cap_during_handoff", "interrupted_second_job_launch", "recovery"}:
                continue  # independent current-inventory offline fault proof required by reduction
            _series_remaining(run, started, 8, absolute_deadline)
            packet = "fault-" + fault.replace("_", "-")
            m = json.loads(native[packet].read_text())
            observer = FaultObserver(fault, Path(m["output_dir"]))
            try:
                supervisor.Broker(run / "fault-state", colab, observer=observer).run(m)
            except supervisor.Refused:
                pass  # Receipt and observation must prove the expected injected failure.
            receipt = _receipt(Path(m["output_dir"]), success=False)
            if receipt.get("status") != "INCOMPLETE":
                raise ValueError("fault did not cause the expected refusal")
            events.append(
                {"fault": fault, "receipt": binding(Path(m["output_dir"]) / "receipt.json")}
            )
        observer_cli = supervisor.CLI(colab, run / "final-sessions.json")
        write_create_only(
            run / "sessions-after.json",
            {
                "sessions": observer_cli.sessions(),
                "usage": observer_cli.usage(),
                "observed_at": time.time(),
                "independent_backend_read": True,
            },
        )
        result = reduce_plan(plan_path)
        write_create_only(run / "acceptance-result.json", result)
        return run / "acceptance-result.json"
    finally:
        write_create_only(
            run / "execution-finished.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "events": events,
                "no_registry_update": True,
            },
        )


def _series_remaining(
    run: Path, started: float, next_minutes: float, absolute_deadline: float
) -> None:
    """Reserve each next maximum before spending; charge retained assignment once."""
    elapsed = time.monotonic() - started
    charged_units = 0.0
    chain_sessions: set[str] = set()
    for path in run.rglob("receipt.json"):
        value = json.loads(path.read_text())
        rate = (value.get("accounting") or {}).get("own_rate_units_per_hour")
        if not rate:
            continue
        if value.get("status") == "COMPLETED_SESSION_RETAINED":
            if value["session"] in chain_sessions:
                continue
            release = json.loads(path.with_name("session-release.json").read_text())
            charged_units += _positive(release["elapsed_seconds"]) * _positive(rate) / 3600
            chain_sessions.add(value["session"])
        else:
            charged_units += _positive(value["elapsed_seconds"]) * _positive(rate) / 3600
    if (
        time.time() + next_minutes * 60 > absolute_deadline
        or elapsed + next_minutes * 60 > 288 * 60
        or charged_units + next_minutes * 10 / 60 > 48 - RESERVE_UNITS
    ):
        raise ValueError("O1 cumulative first-cap reservation reached; no next allocation")


def _offline_faults(plan: dict[str, Any], run: Path) -> dict[str, Any]:
    """Verify independent current-inventory fault tests and their immutable log."""
    value = json.loads((run / "codex-v-fault-review.json").read_text())
    if (
        value.get("schema") != "o1-codex-v-fault-review/v1"
        or value.get("independent_reviewer") != "Codex-V"
    ):
        raise ValueError("independent Codex-V fault review absent")
    if value.get("source_inventory") != plan["source_inventory"]:
        raise ValueError("independent fault tests bind different executable bytes")
    report = bound_json(value["test_report"])
    if (
        report.get("exit_code") != 0
        or report.get("failed") != 0
        or _positive(report.get("passed")) < 8
    ):
        raise ValueError("independent offline fault tests did not pass")
    _check_input(report["log"])
    cases = value.get("fault_tests", {})
    if not isinstance(cases, dict) or set(cases) != set(REQUIRED_FAULTS):
        raise ValueError("independent review requires an executable test for each fault")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("nodeid"), str)
        or "::test_" not in row["nodeid"]
        or "source" not in row
        for row in cases.values()
    ):
        raise ValueError("SHA-bound executable test for each fault required")
    _check_input(report["junit_xml"])
    xml = ET.parse(report["junit_xml"]["path"])
    successful = {
        item.get("classname", "").replace(".", "/") + ".py::" + item.get("name", "")
        for item in xml.iter("testcase")
        if not any(item.find(tag) is not None for tag in ("failure", "error", "skipped"))
    }
    if len({row["nodeid"] for row in cases.values()}) != len(REQUIRED_FAULTS):
        raise ValueError("each fault needs its own executed test")
    for row in cases.values():
        _check_input(row["source"])
        relative = row["nodeid"].split("::")[0]
        if (
            row["nodeid"] not in successful
            or sha256_bytes(git(Path(plan["repo"]), "show", f"{plan['commit']}:{relative}"))
            != row["source"]["sha256"]
        ):
            raise ValueError("fault test lacks current-pin passing JUnit evidence")
    return value


def _native_record(
    plan: dict[str, Any], packet: str, run: Path, cache_pins: dict[str, Any]
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Recheck only the prospectively allowed mechanical admission fields."""
    from research.colab_cli import canary

    row = next(row for row in plan["jobs"] if row["packet"] == packet)
    template = bound_json(row)
    m = json.loads((run / "admitted" / f"{packet}.json").read_text())
    canary_row = next(row for row in plan["jobs"] if row["packet"] == "native-canary")
    canary_output = Path(bound_json(canary_row)["output_dir"])
    template["canary"]["receipt"] = binding(canary_output / "receipt.json")
    if packet.startswith("retained-"):
        template["retention_chain"] = {
            "enabled": True,
            "qualification": {},
            "cache": cache_pins,
            "shared": {
                "chain_id": plan["chain"].get("chain_id", run.name),
                **{key: plan["chain"][key] for key in ("retention_minutes", "retention_units")},
            },
        }
    if m != template:
        raise ValueError("admitted native packet changed outside frozen mechanical bindings")
    canary.check(m, Path(plan["repo"]))
    output = Path(m["output_dir"])
    if json.loads((output / "manifest.json").read_text()) != m:
        # Broker.validate adds defaults. Compare its normalized frozen value.
        normalized = json.loads(json.dumps(m))
        # Validation refuses used output paths, so use the committed receipt's
        # source/job/input identities below rather than inventing a fresh path.
        actual = json.loads((output / "manifest.json").read_text())
        for key, value in normalized.items():
            if actual.get(key) != value:
                raise ValueError("executed native manifest differs from admitted packet")
    receipt = _receipt(
        output, success=not packet.startswith("fault-"), retained=packet.startswith("retained-")
    )
    hardware = json.loads((output / "hardware.json").read_text())
    if hardware["hardware"]["kind"] != "A100":
        raise ValueError("native packet was not actual A100 hardware")
    boundary = json.loads((output / "native-boundary.json").read_text())
    if boundary["a100_endpoints"] != [receipt["endpoint"]]:
        raise ValueError("native packet lacks sole A100 assignment evidence")
    return m, output, receipt


def reduce_plan(plan_path: Path) -> dict[str, Any]:
    """Derive a verdict from fixed receipts and bytes; missing proof stays incomplete."""
    from research.colab_cli import admission, retention

    plan = json.loads(plan_path.read_text())
    run, repo = plan_path.parent, Path(plan["repo"])
    reasons = []
    measurements: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    evidence_files = []
    try:
        verify_inventory(repo, plan["commit"], plan["source_inventory"])
        regression = bound_json(plan["regression_plan"])
        final_path = (
            Path(regression["jobs"][0]["manifest"]).parent.parent / "acceptance-result.json"
        )
        admission.require_final_only(json.loads(final_path.read_text()), repo)
        evidence_files.append(binding(final_path))
        _offline_faults(plan, run)
        evidence_files.append(binding(run / "codex-v-fault-review.json"))
        canary_row = next(row for row in plan["jobs"] if row["packet"] == "native-canary")
        canary_output = Path(bound_json(canary_row)["output_dir"])
        canary_receipt = _receipt(canary_output)
        from research.colab_cli import canary

        canary_manifest = json.loads((canary_output / "manifest.json").read_text())
        if canary.evidence(
            canary_manifest, canary_receipt, canary_output, repo
        ) != canary_receipt.get("canary_evidence"):
            raise ValueError("native canary evidence was not independently reproduced")
        cache_pins = json.loads((canary_output / "exports/probe.json").read_text())["bootstrap"][
            "pins"
        ]
        caches = {}
        extensions = {}
        for packet in ("cold-1", "cold-2", "retained-1", "retained-2"):
            m, output, receipt = _native_record(plan, packet, run, cache_pins)
            receipts[packet] = receipt
            evidence_files.append(binding(output / "receipt.json"))
            probe = json.loads((output / "exports/probe.json").read_text())
            if (
                probe["device"] != "cuda"
                or probe["root"] != "/content/snake-cli/" + receipt["generation"]
            ):
                raise ValueError("native packet lacks independent root and actual CUDA proof")
            cache = probe.get("bootstrap")
            if not isinstance(cache, dict):
                raise ValueError("native setup timing/cache evidence absent")
            expected_inputs = {row["dest"]: row["sha256"] for row in m["inputs"]}
            if (
                probe["input_sha256"] != expected_inputs["probe-input.json"]
                or probe["ledger_sha256"] != expected_inputs["verified-ledger.json"]
            ):
                raise ValueError("native probe inputs or resume ledger differ")
            if cache["pins"] != cache_pins:
                raise ValueError("native setup toolchain/source/config pins differ")
            caches[packet] = cache
            extensions[packet] = probe.get("extension_cache", [])
            measurements[packet] = _positive(cache["setup_seconds"]) + _positive(
                probe["compile_load_seconds"]
            )
        if (
            caches["retained-1"]["cache_status"] != "MISS"
            or caches["retained-2"]["cache_status"] != "HIT"
            or caches["retained-1"]["cache_key"] != caches["retained-2"]["cache_key"]
        ):
            raise ValueError("retention did not reuse the exact immutable cache")
        first_extension, second_extension = extensions["retained-1"], extensions["retained-2"]
        if len(first_extension) != 1 or len(second_extension) != 1:
            raise ValueError("native extension compile/import observations incomplete")
        first_extension, second_extension = first_extension[0], second_extension[0]
        if (
            first_extension.get("status") != "MISS_COMPILED"
            or second_extension.get("status") != "HIT_VERIFIED_IMPORT"
            or any(
                first_extension.get(key) != second_extension.get(key)
                for key in ("key", "sha256", "module_name", "size")
            )
        ):
            raise ValueError("native warm extension did not import the same immutable binary")

        if len({value["generation"] for value in receipts.values()}) != 4:
            raise ValueError("cold/retained roots are not all distinct")
        if receipts["cold-1"]["endpoint"] == receipts["cold-2"]["endpoint"]:
            raise ValueError("cold comparator did not use two independent assignments")
        if receipts["retained-1"]["endpoint"] != receipts["retained-2"]["endpoint"]:
            raise ValueError("retained comparison changed assignment")
        out = Path(json.loads((run / "admitted/retained-2.json").read_text())["output_dir"])
        release = json.loads((out / "session-release.json").read_text())
        evidence_files.append(binding(out / "session-release.json"))
        cold_wall = sum(
            _positive(receipts[key]["assignment_allocated_elapsed_seconds"])
            for key in ("cold-1", "cold-2")
        )
        retained_wall = _positive(release["assignment_allocated_elapsed_seconds"])
        if release.get("chain_error") or not isinstance(release.get("continuous_accounting"), dict):
            raise ValueError("retained chain lacks uninterrupted assignment accounting")
        setup_reduction = 100 * (1 - measurements["retained-2"] / measurements["cold-2"])
        wall_saving = 100 * (1 - retained_wall / cold_wall)
        measurements.update(
            second_setup_reduction_percent=setup_reduction,
            allocated_wall_saving_percent=wall_saving,
            cold_allocated_seconds=cold_wall,
            retained_allocated_seconds=retained_wall,
        )
        if setup_reduction < 50 or wall_saving < 25:
            reasons.append(
                "measured reuse missed >=50% second setup or >=25% allocated wall bar; PARK"
            )
        for fault in (
            "mac_disappearance_lease_expiry",
            "stale_generation",
            "child_failure",
            "accounting_refusal",
            "export_failure",
        ):
            packet = "fault-" + fault.replace("_", "-")
            _m, output, receipt = _native_record(plan, packet, run, cache_pins)
            evidence_files.append(binding(output / "receipt.json"))
            if receipt.get("status") != "INCOMPLETE":
                raise ValueError("fault expected a failed native packet")
            if fault == "child_failure":
                if (receipt.get("terminal") or {}).get("returncode") != 23:
                    raise ValueError("child failure was not independently observed")
                continue
            observation = json.loads((output / "fault-observation.json").read_text())
            evidence_files.append(binding(output / "fault-observation.json"))
            if observation.get("fault") != fault:
                raise ValueError("native fault observation has wrong identity")
            if fault == "accounting_refusal" and (
                observation.get("injected_usage", {}).get("rate", 0)
                != observation.get("actual_usage", {}).get("rate", 0) + 100
                or "rate" not in observation.get("error", "").lower()
            ):
                raise ValueError("native accounting drift refusal was not observed")
            if (
                fault == "export_failure"
                and observation.get("observation", {})
                .get("failure-and-corrupt-export", {})
                .get("corrupt_export_rejected")
                is not True
            ):
                raise ValueError("native corrupt export rejection was not observed")
            if fault in {"mac_disappearance_lease_expiry", "stale_generation"}:
                delta = observation["observed_monotonic"] - observation["expiry_monotonic"]
                if not 0 <= delta <= 300 or observation.get("before_fallback_cleanup") is not True:
                    raise ValueError("native watchdog disappearance bound failed")
                if (
                    not observation.get("readings")
                    or observation["endpoint"] in observation["readings"][-1]["sessions"]
                ):
                    raise ValueError("native watchdog lacks backend disappearance")
        # Independent session observation happens after all Broker cleanup. It
        # is create-only and cannot be inferred from a retained receipt flag.
        sessions = json.loads((run / "sessions-after.json").read_text())
        own_endpoints = {value["endpoint"] for value in receipts.values()}
        if own_endpoints & set(sessions["sessions"]):
            raise ValueError("owned acceptance assignment remains listed")
        evidence_files.append(binding(run / "sessions-after.json"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        reasons.append(str(exc))
    return {
        "schema": "snake-dqn-a100-retention-acceptance-result/v1",
        "status": "PASS" if not reasons else "INCOMPLETE",
        "plan": binding(plan_path),
        "source_inventory": retention.source_inventory(repo),
        "complete_source_inventory": plan["source_inventory"],
        "native_a100_two_packet": not reasons,
        "independent_reviewer": None,
        "review_required_before_adoption": True,
        "metrics": measurements,
        "evidence": evidence_files,
        "reasons": reasons,
        "no_registry_update": True,
    }


def reduce_evidence(_evidence: dict[str, Any]) -> dict[str, Any]:
    """Historical scaffold's supplied booleans can never confer qualification."""
    return {
        "status": "INCOMPLETE",
        "adoption": "REFUSED_UNTIL_PASS",
        "reasons": ["use reduce --plan: qualification is derived from immutable receipts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--repo", type=Path, default=Path.cwd())
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--other-assignments", type=Path)
    run = actions.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--boundary-admission", type=Path, required=True)
    run.add_argument("--review-token", required=True)
    run.add_argument("--colab", default=str(Path.home() / ".local/bin/colab"))
    run.add_argument("--oauth-python", default=str(Path.home() / ".venvs/colab-cli/bin/python"))
    reduce = actions.add_parser("reduce")
    reduce.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            from research.colab_cli.acceptance import load_other_assignments

            print(
                prepare(args.repo, args.output_root, load_other_assignments(args.other_assignments))
            )
        elif args.action == "execute":
            print(
                execute(
                    args.plan.resolve(),
                    args.boundary_admission.resolve(),
                    args.review_token,
                    colab=args.colab,
                    oauth_python=args.oauth_python,
                )
            )
        else:
            result = reduce_plan(args.plan.resolve())
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"REFUSE: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
