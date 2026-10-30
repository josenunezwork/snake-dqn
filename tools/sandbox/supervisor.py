"""Host-only sandbox admission adapter for the unchanged qualified supervisor.

Only the family canary receipt decision is replaced. Ordinary jobs always call
the original checker. Allocation, runtime, accounting and cleanup stay qualified.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def require_source_delivery(manifest: dict[str, Any]) -> None:
    """Check the execution policy during offline validation as well as live use."""
    from tools.sandbox.source_delivery import OPT_IN

    if manifest.get("sandbox_source_delivery") != OPT_IN:
        return
    if manifest.get("script") == "research/sandbox_a_resident/ct_cuda_canary_job.sh":
        from research.sandbox_a_resident.ct_canary_admission import require_smoke

        require_smoke(manifest)
        return
    if manifest.get("script") not in {
        "research/sandbox_a_resident/train_job.sh",
        "research/sandbox_a_resident/ct_job.sh",
        "research/sandbox_a_resident/ct_h5000/job.sh",
        "research/sandbox_a_resident/ct_stage2/job.sh",
    }:
        raise ValueError(
            "background sparse source delivery is exclusive to TRAIN-01 and t34-cf-1028"
        )


@contextmanager
def sandbox_admission_route() -> Iterator[None]:
    """Install a process-local policy adapter, restoring it even after refusal."""
    from research.colab_cli import canary, supervisor
    from tools.ops.dispatch import require_sandbox_admission
    from tools.sandbox.source_delivery import OPT_IN, background_sparse_route

    original = canary.check
    original_validate = supervisor.validate
    original_execute = supervisor.Broker.execute
    original_run = supervisor.Broker.run
    original_call_once = supervisor.CLI._call_once
    original_calibrate = supervisor.Broker.calibrate_rate
    from tools.sandbox.admission import LEASE_ENV, AllocationLease

    lease: AllocationLease | None = None

    def run(self: Any, manifest: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal lease
        if "sandbox_admission" not in manifest:
            return original_run(self, manifest, *args, **kwargs)
        path = os.environ.get(LEASE_ENV)
        if not path and manifest.get("script") != "research/sandbox_a_resident/train_job.sh":
            return original_run(self, manifest, *args, **kwargs)
        if not path:
            raise ValueError("sandbox run requires an ops allocation lease")
        import sys

        lease = AllocationLease(Path(path), Path(sys.argv[-1]).resolve())
        try:
            return original_run(self, manifest, *args, **kwargs)
        finally:
            lease.close()
            lease = None

    def call_once(self: Any, *args: str, timeout: float) -> str:
        if lease is not None:
            if args[0] == "new":
                lease.acquire()
            if lease.handle is not None and args[0] != "stop" and not self.cleanup_mode:
                timeout = lease.remaining(timeout)
        return original_call_once(self, *args, timeout=timeout)

    def calibrate(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_calibrate(self, *args, **kwargs)
        if lease is not None:
            lease.close()
        return result

    def check(manifest: dict[str, Any], repo: Path) -> None:
        if "sandbox_admission" in manifest:
            require_sandbox_admission(manifest, repo)
        else:
            original(manifest, repo)

    def validate(manifest: dict[str, Any], repo: Path | None = None) -> dict[str, Any]:
        # Refuse protected/foreign sandbox inputs BEFORE the qualified validator
        # resolves or hashes their bytes. Its later canary hook rechecks policy.
        if "sandbox_admission" in manifest:
            require_sandbox_admission(manifest, supervisor.REPO if repo is None else repo)
        require_source_delivery(manifest)
        return original_validate(manifest, repo)

    def execute(self: Any, cli: Any, manifest: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Wrap only the qualified execution phase that owns source delivery."""
        if manifest.get("sandbox_source_delivery") != OPT_IN:
            return original_execute(self, cli, manifest, *args, **kwargs)
        require_source_delivery(manifest)
        with background_sparse_route(cli):
            return original_execute(self, cli, manifest, *args, **kwargs)

    canary.check = check
    supervisor.validate = validate
    supervisor.Broker.execute = execute
    supervisor.Broker.run = run
    supervisor.CLI._call_once = call_once
    supervisor.Broker.calibrate_rate = calibrate
    try:
        yield
    finally:
        canary.check = original
        supervisor.validate = original_validate
        supervisor.Broker.execute = original_execute
        supervisor.Broker.run = original_run
        supervisor.CLI._call_once = original_call_once
        supervisor.Broker.calibrate_rate = original_calibrate


def main() -> None:
    """Run the qualified CLI with explicit sandbox policy and no probe mode."""
    from research.colab_cli import supervisor

    with sandbox_admission_route():
        supervisor.main()


if __name__ == "__main__":
    main()
