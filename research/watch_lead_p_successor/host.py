"""Narrow host-only source delivery adapter; native canary admission is unchanged."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCRIPT = "research/watch_lead_p_successor/profile_job.sh"


def require_profile(manifest: dict, broker: Path) -> None:
    """Admit background sparse delivery only for the exact bounded CPU probe."""
    if (
        manifest.get("p_profile_source_delivery") != "background-sparse-v1"
        or manifest.get("script") != SCRIPT
        or manifest.get("device") != "cpu"
        or manifest.get("canary")
        != {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
        or manifest.get("requirements", {}).get("allowed_gpus") not in (["L4"], ["A100"])
        or [manifest.get("requirements", {}).get("gpu")]
        != manifest.get("requirements", {}).get("allowed_gpus")
        or manifest.get("timeout_minutes") != 30
        or manifest.get("resume_exemption", {}).get("kind") != "short-canary"
        or manifest.get("env", {}).get("SNAKE_COLAB_CANARY") != "1"
        or "sandbox_admission" in manifest
        or broker.resolve() != Path(__file__).resolve().parents[2]
    ):
        raise ValueError("source adapter requires exact 30-minute P CPU profile probe")


@contextmanager
def profile_source_route() -> Iterator[None]:
    """Scope the existing sparse-source adapter to qualified execution only."""
    from research.colab_cli import supervisor
    from tools.sandbox.source_delivery import background_sparse_route

    original = supervisor.Broker.execute

    def execute(self: Any, cli: Any, manifest: dict, *args: Any, **kwargs: Any) -> Any:
        require_profile(manifest, supervisor.REPO)
        with background_sparse_route(cli):
            return original(self, cli, manifest, *args, **kwargs)

    supervisor.Broker.execute = execute
    try:
        yield
    finally:
        supervisor.Broker.execute = original


def main() -> None:
    from research.colab_cli import supervisor

    with profile_source_route():
        supervisor.main()


if __name__ == "__main__":
    main()
