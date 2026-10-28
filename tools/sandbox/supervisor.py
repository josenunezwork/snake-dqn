"""Host-only sandbox admission adapter for the unchanged qualified supervisor.

Only the family canary receipt decision is replaced. Ordinary jobs always call
the original checker. Allocation, runtime, accounting and cleanup stay qualified.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def sandbox_admission_route() -> Iterator[None]:
    """Install a process-local policy adapter, restoring it even after refusal."""
    from research.colab_cli import canary, supervisor
    from tools.ops.dispatch import require_sandbox_admission
    from tools.sandbox.source_delivery import OPT_IN, background_sparse_route

    original = canary.check
    original_validate = supervisor.validate
    original_execute = supervisor.Broker.execute

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
        return original_validate(manifest, repo)

    def execute(self: Any, cli: Any, manifest: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Wrap only the qualified execution phase that owns source delivery."""
        if manifest.get("sandbox_source_delivery") != OPT_IN:
            return original_execute(self, cli, manifest, *args, **kwargs)
        if manifest.get("script") not in {
            "research/sandbox_a_resident/train_job.sh",
            "research/watch_lead_p/run_job.sh",
        }:
            raise ValueError("background sparse source delivery is TRAIN-01 only")
        with background_sparse_route(cli):
            return original_execute(self, cli, manifest, *args, **kwargs)

    canary.check = check
    supervisor.validate = validate
    supervisor.Broker.execute = execute
    try:
        yield
    finally:
        canary.check = original
        supervisor.validate = original_validate
        supervisor.Broker.execute = original_execute


def main() -> None:
    """Run the qualified CLI with explicit sandbox policy and no probe mode."""
    from research.colab_cli import supervisor

    with sandbox_admission_route():
        supervisor.main()


if __name__ == "__main__":
    main()
