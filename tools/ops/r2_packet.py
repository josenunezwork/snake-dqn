"""R2 operator companion: L4 compile prerequisite, A100 canary, then A100 main.

Mirrors the I0 companion (``integration_compile.py``): every runtime/transport
check runs from the frozen broker checkout of the R2 runtime pin, while the
host dispatcher (``tools/ops/dispatch.py`` of the ``--host`` checkout, which
has ``--dry-run``) performs governed validation and the single allocation.
Every lane first runs the real dispatcher with ``--dry-run`` (allocations=0)
and publishes a bound manifest only after it passes.  Nothing here dispatches
unless the operator omits ``--dry-run``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.colab_cli.accounting import budget_warning, measurement_rate

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/Users/josenunez/Projects/ml")
ARTIFACT = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-r2-padding-20261024"
BROKER = BASE / "snake-dqn-gpur2-qualified.noindex"
HOST = BASE / "snake-dqn"
PY = BASE / "snake-dqn/venv/bin/python"
PACKAGE = "research/gpu_sim_spike"
CANARY_CAP_UNITS = 2.0  # ruling step 4 / R2 protocol: whole canary allowance
MAX_CANARY_ATTEMPTS = 2


@dataclass(frozen=True)
class Family:
    """One packet family served by this companion (R2 by default; R6 via r6_packet.py)."""

    tag: str
    artifact: Path
    broker: Path
    producer_key: str
    tool: str


R2 = Family(
    "r2",
    ARTIFACT,
    BROKER,
    "R2_BENCH_SECONDS",
    "r2_packet.py",
)
FAMILY = R2


def _suffix(attempt: int) -> str:
    return "" if attempt == 1 else f"-{attempt:02d}"


def attempt_manifest(
    first: dict[str, Any], prior_units: float, output_dir: Path, attempt: int = 2
) -> dict[str, Any]:
    """Derive one fresh retry while retaining the original wall-clock envelope.

    Remaining unit allowance is evidence and warnings only. Producer settings,
    sample counts and the one-retry identity rule remain unchanged.
    """
    if attempt != 2:
        raise ValueError("the canary allows exactly one fresh attempt (attempt 2)")
    if not math.isfinite(prior_units) or prior_units < 0:
        raise ValueError("prior canary charge must be finite and nonnegative")
    rate = measurement_rate(first.get("budget", {}).get("rate_units_per_hour", 0), "canary rate")
    cap = math.floor((CANARY_CAP_UNITS - prior_units) * 100) / 100
    timeout = first["timeout_minutes"]
    producer_minutes = int(first["env"][FAMILY.producer_key]) / 60
    if cap <= 0 or (rate > 0 and cap * 60 / rate < producer_minutes + 3):
        budget_warning("CANARY_UNIT_THRESHOLD", prior_units=prior_units, remaining_units=cap)
    cap = max(0.0, cap)
    job = copy.deepcopy(first)
    if not job["job_id"].endswith("-01"):
        raise ValueError("first canary job id does not end in -01")
    job["job_id"] = job["job_id"][:-3] + f"-{attempt:02d}"
    job["output_dir"] = str(output_dir)
    job.setdefault("budget", {})["unit_cap"] = cap
    job["budget"]["reserve_units"] = cap / 6
    job["timeout_minutes"] = timeout
    job["requirements"]["est_minutes"] = timeout / 2
    job["canary_attempt"] = {
        "attempt": attempt,
        "allowance_units": CANARY_CAP_UNITS,
        "prior_attempt_charged_units": prior_units,
        "remaining_unit_cap": cap,
        "unit_threshold_warning_only": True,
        "changed_fields": [
            "job_id",
            "output_dir",
            "budget.unit_cap",
            "budget.reserve_units",
            "timeout_minutes",
            "requirements.est_minutes",
            "canary_attempt",
        ],
    }
    return job


def prior_failure(artifact: Path) -> dict[str, Any]:
    """Prove attempt 1 was a released pre-work kernel-creation failure and charge it."""
    sys.path.insert(0, str(ROOT))
    from tools.ops.redispatch_fresh import verify_failure

    packet = artifact / "packet"
    proof = verify_failure(packet, packet / f"{FAMILY.tag}-canary-compiled.json")
    receipt = json.loads((packet / "canary-output/receipt.json").read_text())
    before = receipt.get("usage_before", {}).get("balance")
    after = receipt.get("usage_after", {}).get("balance")
    drop = (
        float(before) - float(after)
        if isinstance(before, (int, float)) and isinstance(after, (int, float))
        else 0.0
    )
    proof["observed_balance_drop_units"] = drop
    proof["charged_units"] = max(float(proof["prior_elapsed_rate_units"]), drop, 0.0)
    return proof


STAGING_ERROR = "CanaryError('staging must complete within 300 seconds')"


def _charge(receipt_path: Path) -> dict[str, Any]:
    """Charge one released receipt: max(elapsed-rate units, observed balance drop)."""
    sys.path.insert(0, str(ROOT))
    from tools.ops.spend import read_spend

    receipt = json.loads(receipt_path.read_text())
    spend = read_spend(receipt_path)
    before = receipt.get("usage_before", {}).get("balance")
    after = receipt.get("usage_after", {}).get("balance")
    drop = (
        float(before) - float(after)
        if isinstance(before, (int, float)) and isinstance(after, (int, float))
        else 0.0
    )
    return {
        "receipt": {"path": str(receipt_path), "sha256": spend.sha256},
        "prior_allocation_elapsed_seconds": spend.elapsed_seconds,
        "prior_elapsed_rate_units": spend.units,
        "observed_balance_drop_units": drop,
        "charged_units": max(float(spend.units), drop, 0.0),
    }


def compile_staging_failure(artifact: Path) -> dict[str, Any]:
    """Prove compile attempt 1 finished its work but failed only the 300 s staging rule.

    Accepted shape: released receipt (stop verified, no cleanup error), terminal
    exit 0, the executed manifest equal to the published one apart from the
    live assignment inventory, a staging measurement above 300 s, and exactly
    the qualified canary-evidence refusal as the error.  Compilation is
    idempotent and produces no outcome data, so a fresh attempt is mechanical.
    """
    root = artifact / "packet/native-compile"
    source = root / "native-compile.json"
    output = root / "output"
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    saved = output / "manifest.json"
    published = _verified(source)
    executed = json.loads(saved.read_text())
    executed.setdefault("budget", {})["other_assignments"] = published.get("budget", {}).get(
        "other_assignments", {}
    )
    staging = json.loads((output / "upload-measurement.json").read_text())
    terminal = receipt.get("terminal") or {}
    if (
        receipt.get("status") != "INCOMPLETE"
        or receipt.get("failure_class") != "SUPERVISOR"
        or receipt.get("error") != STAGING_ERROR
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or receipt.get("manifest_sha256") != hashlib.sha256(saved.read_bytes()).hexdigest()
        or executed != published
        or receipt.get("job_id") != published["job_id"]
        or terminal.get("returncode") != 0
        or not float(staging.get("elapsed_seconds", 0)) > 300
    ):
        raise ValueError("compile attempt 1 is not a released staging-time-limit failure")
    return {
        **_charge(receipt_path),
        "source_manifest": {"path": str(source), "sha256": _sha(source)},
        "failure": STAGING_ERROR,
        "staging_seconds": staging,
        "terminal_returncode": 0,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_attempt_manifest(
    first: dict[str, Any], proof: dict[str, Any], output_dir: Path, attempt: int = 2
) -> dict[str, Any]:
    """Fresh compile: identical frozen manifest except job id and output directory."""
    if attempt != 2:
        raise ValueError("the compile allows exactly one fresh attempt (attempt 2)")
    job = copy.deepcopy(first)
    job["job_id"] = f"{first['job_id']}-{attempt:02d}"
    job["output_dir"] = str(output_dir)
    job["compile_attempt"] = {
        "attempt": attempt,
        "prior_attempt_charged_units": proof["charged_units"],
        "prior_failure": proof["failure"],
        "changed_fields": ["job_id", "output_dir", "compile_attempt"],
    }
    return job


def frozen(broker: Path, code: str, *args: str) -> Any:
    """Run pinned package/transport code from the broker without authoring imports."""
    result = subprocess.run(
        [
            str(PY),
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv.pop(1)); " + code,
            str(broker),
            *args,
        ],
        cwd=broker,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode:
        raise RuntimeError(f"broker helper failed: {result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, timeout=30).strip()


def upload(job: dict[str, Any], broker: Path, stage: Path) -> dict[str, Any]:
    """Real supervisor validation plus the exact staged-upload byte bound."""
    with tempfile.TemporaryDirectory(prefix="r2-upload-manifest-") as temporary:
        path = Path(temporary) / "job.json"
        path.write_text(json.dumps(job))
        return frozen(
            broker,
            "import json; from pathlib import Path; "
            "from tools.ops.repack_gpu_upload import check_upload; "
            "print(json.dumps(check_upload(json.loads(Path(sys.argv[1]).read_text()), "
            "Path(sys.path[0]), Path(sys.argv[2]))))",
            str(path),
            str(stage),
        )


def _verified(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from tools.ops.repack_gpu_upload import verified_json

    return verified_json(path)


def _publish(path: Path, value: Any) -> None:
    sys.path.insert(0, str(ROOT))
    from tools.ops.repack_gpu_upload import publish

    publish(path, value)


def controls(artifact: Path, broker: Path) -> tuple[dict[str, Any], str]:
    """The prepared canary, its pin, and an unchanged qualified broker at that pin."""
    sys.path.insert(0, str(ROOT))
    from tools.ops.qualification import OPS, check

    job = _verified(artifact / f"packet/{FAMILY.tag}-canary.json")
    pin = job["commit"]
    if git(broker, "rev-parse", "HEAD") != pin:
        raise ValueError(f"broker checkout is not at the {FAMILY.tag} runtime pin")
    check(broker, OPS / "colab-cli-qualified.json")
    check(broker, OPS / "colab-cli-qualified.json", ref=pin)
    return job, pin


def dispatch_command(
    host: Path, artifact: Path, broker: Path, name: str, path: Path, lane: str, attempt: int = 1
) -> list[str]:
    state = {
        "compile": "packet/native-compile/state" + _suffix(attempt),
        "canary": "packet/state-canary" + _suffix(attempt),
        "main": "packet/state-main",
    }[lane]
    return [
        str(PY),
        str(host / "tools/ops/dispatch.py"),
        name,
        str(path),
        "--broker",
        str(broker),
        "--cwd",
        str(broker),
        "--state-dir",
        str(artifact / state),
        "--dispatch-txt",
        str(artifact / "packet/DISPATCH.txt"),
    ]


def prepare(artifact: Path, broker: Path, host: Path = HOST) -> dict[str, Any]:
    """Create the same-pin L4 compile manifest and the operator wrappers."""
    _canary, pin = controls(artifact, broker)
    compile_root = artifact / "packet/native-compile"
    job = frozen(
        broker,
        "import json; from pathlib import Path; "
        "from research.gpu_sim_spike.native_compile_pack import manifest; "
        "from tools.ops.repack_gpu_upload import compile_sources; "
        "j=manifest(sys.argv[1], Path(sys.argv[2])); "
        "j['source_paths']=compile_sources(Path(sys.path[0]), sys.argv[1]); "
        "print(json.dumps(j))",
        pin,
        str(compile_root),
    )
    job["job_id"] += f"-{FAMILY.tag}"
    path = compile_root / "native-compile.json"
    if path.exists():
        if _verified(path) != job:
            raise ValueError("immutable compiler manifest differs")
    else:
        report = upload(job, broker, compile_root / "upload-audit")
        _publish(compile_root / "UPLOAD_CHECK.json", report)
        _publish(path, job)
    for lane in ("compile", "canary", "main"):
        script = artifact / f"dispatch-{lane}.sh"
        command = shlex.join(
            [
                str(PY),
                "-I",
                str(host / "tools/ops" / FAMILY.tool),
                lane,
                "--artifact-dir",
                str(artifact),
                "--broker",
                str(broker),
                "--host",
                str(host),
            ]
        )
        content = "#!/usr/bin/env bash\nset -euo pipefail\nexec " + command + ' "$@"\n'
        if script.exists() and script.read_text() != content:
            raise ValueError(f"existing wrapper differs: {script}")
        script.write_text(content)
        script.chmod(0o755)
    return {"status": "PREPARED_NOT_DISPATCHED", "runtime_pin": pin, "compile_manifest": str(path)}


def _main_manifest(artifact: Path, broker: Path, pin: str, receipt: Path) -> Path:
    """Prepare (once) the canary-bound main through the broker's frozen packer.

    Only the main manifest is built.  ``r2_pack.prepare`` would also regenerate
    the already executed canary manifest, whose recorded local path for the
    public-input catalog depends on the checkout that packed it (identical
    bytes/destination, different absolute path) and so cannot be re-derived
    from the broker byte-for-byte.  The same packer functions, source-pin and
    preflight checks, supervisor validation and create-only write are used.
    """
    path = artifact / f"packet/{FAMILY.tag}-main.json"
    if not path.exists() and FAMILY.tag != "r2":
        # Families other than R2 expose the same main-only builder in their packer.
        frozen(
            broker,
            "import json; from pathlib import Path; "
            f"from research.gpu_sim_spike.{FAMILY.tag}_pack import prepare_main; "
            "print(json.dumps({'main': str(prepare_main(Path(sys.argv[1]), sys.argv[2], "
            "Path(sys.argv[3]), Path(sys.argv[4])))}))",
            str(artifact / "packet"),
            pin,
            str(artifact / "mac-preflight/preflight.json"),
            str(receipt),
        )
    elif not path.exists():
        frozen(
            broker,
            "import json; from pathlib import Path; "
            "from research.gpu_sim_spike import r2_pack as p; "
            "from research.colab_cli.supervisor import validate; "
            "packet, commit = Path(sys.argv[1]), sys.argv[2]; "
            "p.require_source_pin(commit); "
            "pre = p.preflight_binding(Path(sys.argv[3]), commit); "
            "m = p.manifest(commit=commit, output_dir=packet / 'main-output', "
            "ledger=packet / 'verified-ledger.json', "
            "canary_ledger=packet / 'canary-ledger/verified-ledger.json', canary=False, "
            "preflight=pre, canary_receipt=Path(sys.argv[4])); "
            "validate(m, repo=p.ROOT); "
            "p._write(packet / 'r2-main.json', json.dumps(m, indent=2, sort_keys=True) + '\\n'); "
            "print(json.dumps({'main': str(packet / 'r2-main.json')}))",
            str(artifact / "packet"),
            pin,
            str(artifact / "mac-preflight/preflight.json"),
            str(receipt),
        )
    return path


def run(
    lane: str,
    artifact: Path,
    broker: Path,
    host: Path,
    dry: bool,
    receipt: Path | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    if attempt not in range(1, MAX_CANARY_ATTEMPTS + 1) or (attempt > 1 and lane == "main"):
        raise ValueError("--attempt 2 applies only to the compile or canary lane")
    original, pin = controls(artifact, broker)
    accounting = None
    name = {
        "canary": f"gpu-{FAMILY.tag}-canary-{pin[:8]}",
        "main": f"gpu-{FAMILY.tag}-main-{pin[:8]}",
    }.get(lane)
    if lane == "compile":
        if receipt is not None:
            raise ValueError("a receipt is only valid for canary or main")
        path = artifact / "packet/native-compile/native-compile.json"
        job = _verified(path)
        if job["commit"] != pin or job["script"] != f"{PACKAGE}/native_compile.sh":
            raise ValueError("compiler identity differs")
        if attempt > 1:
            accounting = compile_staging_failure(artifact)
            job = compile_attempt_manifest(
                job, accounting, artifact / f"packet/native-compile/output{_suffix(attempt)}"
            )
            path = artifact / f"packet/native-compile/native-compile{_suffix(attempt)}.json"
        name = job["job_id"]
    elif lane == "canary":
        path = artifact / f"packet/{FAMILY.tag}-canary-compiled{_suffix(attempt)}.json"
        compiled = artifact / "packet/native-compile/output-02/receipt.json"
        if not compiled.is_file():
            compiled = artifact / "packet/native-compile/output/receipt.json"
        receipt = receipt or Path(os.environ.get("NATIVE_COMPILE_RECEIPT") or compiled)
        job = dict(original)
        # The host reader (as used by dispatch.py's gate) carries the corrected
        # payload-only export inventory; the pinned runtime reader predates it.
        job["native_cuda_compile"] = {
            "receipt": frozen(
                host,
                "import json; from pathlib import Path; "
                "from research.gpu_sim_spike import native_compile as n; "
                "b=n.binding(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])); "
                "n.require_for_a100_canary({**json.loads(sys.argv[4]), "
                "'native_cuda_compile': {'receipt': b}}, Path(sys.argv[3])); "
                "print(json.dumps(b))",
                str(receipt),
                pin,
                str(broker),
                json.dumps(original),
            )
        }
        if attempt > 1:
            # Same compile receipt binding as attempt 1; identical content otherwise
            # except fresh identity and the remaining canary allowance.
            first = _verified(artifact / f"packet/{FAMILY.tag}-canary-compiled.json")
            if first["native_cuda_compile"] != job["native_cuda_compile"]:
                raise ValueError("attempt 2 must bind attempt 1's native compile receipt")
            accounting = prior_failure(artifact)
            job = attempt_manifest(
                first,
                accounting["charged_units"],
                artifact / f"packet/canary-output{_suffix(attempt)}",
                attempt,
            )
            name = f"{name}-a{attempt}"
    else:
        default = artifact / "packet/canary-output-02/receipt.json"
        if not default.is_file():
            default = artifact / "packet/canary-output/receipt.json"
        receipt = receipt or default
        if dry and not receipt.is_file():
            raise ValueError(
                "main dry-run requires the canary receipt (the qualified supervisor binds it)"
            )
        path = _main_manifest(artifact, broker, pin, receipt)
        job = _verified(path)
        if job["commit"] != pin or job.get("scientific_admission") is not True:
            raise ValueError("main manifest identity differs")
    if path.exists() and _verified(path) != job:
        raise ValueError("immutable bound manifest differs")
    with tempfile.TemporaryDirectory(prefix="r2-validation-") as temporary:
        temporary_root = Path(temporary)
        report = upload(job, broker, temporary_root / "upload")
        validation_manifest = temporary_root / path.name
        validation_manifest.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        command = dispatch_command(host, artifact, broker, name, validation_manifest, lane, attempt)
        subprocess.run([*command, "--dry-run"], cwd=host, check=True)
        if not dry:
            if accounting is not None:
                record = artifact / (
                    f"packet/native-compile/COMPILE_ATTEMPT{_suffix(attempt)}.json"
                    if lane == "compile"
                    else f"packet/CANARY_ATTEMPT{_suffix(attempt)}.json"
                )
                if not record.exists():
                    _publish(record, accounting)
            if not path.exists():
                _publish(path, job)
            subprocess.run(
                dispatch_command(host, artifact, broker, name, path, lane, attempt),
                cwd=host,
                check=True,
            )
    return {
        "status": "PASS_DRY_RUN" if dry else "DISPATCHED",
        "runtime_pin": pin,
        "lane": lane,
        "dispatch_name": name,
        "largest_upload_bytes": report.get("largest_upload_bytes"),
        "source_files": report.get("source_files"),
        "source_archive_bytes": report.get("source_archive_bytes"),
        "input_rows": len(job["inputs"]),
        "live_accounting": "NOT_CHECKED" if dry else "CHECKED",
        "attempt": attempt,
        "canary_attempt": job.get("canary_attempt"),
        "compile_attempt": job.get("compile_attempt"),
        "prior_attempt_accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compile", "canary", "main"))
    parser.add_argument("--artifact-dir", type=Path, default=FAMILY.artifact)
    parser.add_argument("--broker", type=Path, default=FAMILY.broker)
    parser.add_argument("--host", type=Path, default=HOST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--attempt", type=int, default=1, help="canary only: 2 = fresh attempt")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.action == "prepare":
        if args.dry_run or args.receipt:
            parser.error("prepare accepts neither --dry-run nor --receipt")
        result = prepare(args.artifact_dir.resolve(), args.broker.resolve(), args.host.resolve())
    else:
        result = run(
            args.action,
            args.artifact_dir.resolve(),
            args.broker.resolve(),
            args.host.resolve(),
            args.dry_run,
            args.receipt,
            args.attempt,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
