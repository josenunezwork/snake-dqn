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
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/Users/josenunez/Projects/ml")
ARTIFACT = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-r2-padding-20261024"
BROKER = BASE / "snake-dqn-gpur2-qualified.noindex"
HOST = BASE / "snake-dqn"
PY = BASE / "snake-dqn/venv/bin/python"
PACKAGE = "research/gpu_sim_spike"
CANARY_CAP_UNITS = 2.0  # ruling step 4 / R2 protocol: whole canary allowance
MAX_CANARY_ATTEMPTS = 2


def _suffix(attempt: int) -> str:
    return "" if attempt == 1 else f"-{attempt:02d}"


def attempt_manifest(
    first: dict[str, Any], prior_units: float, output_dir: Path, attempt: int = 2
) -> dict[str, Any]:
    """Fresh canary attempt: same science/inputs/env; only identity and remaining cap change.

    The failed attempt's charge is deducted from the 2 u canary allowance. The
    whole-session timeout is the largest value that the remaining cap prices
    (qualified rule ``timeout * rate / 60 <= cap``) and the declared estimate
    stays at the qualified half-timeout (``timeout >= 2 * est``). The producer
    deadline (``R2_BENCH_SECONDS``) is unchanged.
    """
    if attempt != 2:
        raise ValueError("R2 canary allows exactly one fresh attempt (attempt 2)")
    if not math.isfinite(prior_units) or prior_units < 0:
        raise ValueError("prior canary charge must be finite and nonnegative")
    rate = float(first["budget"]["rate_units_per_hour"])
    cap = math.floor((CANARY_CAP_UNITS - prior_units) * 100) / 100
    timeout = math.floor(cap * 60 / rate * 10) / 10
    producer_minutes = int(first["env"]["R2_BENCH_SECONDS"]) / 60
    if cap <= 0 or timeout < producer_minutes + 3:
        raise ValueError("remaining canary allowance cannot fit the unchanged producer deadline")
    job = copy.deepcopy(first)
    if not job["job_id"].endswith("-01"):
        raise ValueError("first canary job id does not end in -01")
    job["job_id"] = job["job_id"][:-3] + f"-{attempt:02d}"
    job["output_dir"] = str(output_dir)
    job["budget"]["unit_cap"] = cap
    job["budget"]["reserve_units"] = cap / 6
    job["timeout_minutes"] = timeout
    job["requirements"]["est_minutes"] = timeout / 2
    job["canary_attempt"] = {
        "attempt": attempt,
        "allowance_units": CANARY_CAP_UNITS,
        "prior_attempt_charged_units": prior_units,
        "remaining_unit_cap": cap,
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
    proof = verify_failure(packet, packet / "r2-canary-compiled.json")
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

    job = _verified(artifact / "packet/r2-canary.json")
    pin = job["commit"]
    if git(broker, "rev-parse", "HEAD") != pin:
        raise ValueError("broker checkout is not at the R2 runtime pin")
    check(broker, OPS / "colab-cli-qualified.json")
    check(broker, OPS / "colab-cli-qualified.json", ref=pin)
    return job, pin


def dispatch_command(
    host: Path, artifact: Path, broker: Path, name: str, path: Path, lane: str, attempt: int = 1
) -> list[str]:
    state = {
        "compile": "packet/native-compile/state",
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
    job["job_id"] += "-r2"
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
                str(host / "tools/ops/r2_packet.py"),
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
    """Prepare (once) the canary-bound main through the broker's frozen packer."""
    path = artifact / "packet/r2-main.json"
    if not path.exists():
        frozen(
            broker,
            "import json; from pathlib import Path; "
            "from research.gpu_sim_spike.r2_pack import prepare; "
            "print(json.dumps(prepare(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), "
            "canary_receipt=Path(sys.argv[4]), validate=True)))",
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
    if attempt not in range(1, MAX_CANARY_ATTEMPTS + 1) or (attempt > 1 and lane != "canary"):
        raise ValueError("--attempt 2 applies only to the canary lane")
    original, pin = controls(artifact, broker)
    accounting = None
    name = {
        "canary": f"gpu-r2-canary-{pin[:8]}",
        "main": f"gpu-r2-main-{pin[:8]}",
    }.get(lane)
    if lane == "compile":
        if receipt is not None:
            raise ValueError("a receipt is only valid for canary or main")
        path = artifact / "packet/native-compile/native-compile.json"
        job = _verified(path)
        if job["commit"] != pin or job["script"] != f"{PACKAGE}/native_compile.sh":
            raise ValueError("compiler identity differs")
        name = job["job_id"]
    elif lane == "canary":
        path = artifact / f"packet/r2-canary-compiled{_suffix(attempt)}.json"
        receipt = receipt or Path(
            os.environ.get("NATIVE_COMPILE_RECEIPT")
            or artifact / "packet/native-compile/output/receipt.json"
        )
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
            first = _verified(artifact / "packet/r2-canary-compiled.json")
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
                record = artifact / f"packet/CANARY_ATTEMPT{_suffix(attempt)}.json"
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
        "prior_attempt_accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compile", "canary", "main"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    parser.add_argument("--broker", type=Path, default=BROKER)
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
