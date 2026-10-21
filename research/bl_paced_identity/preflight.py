"""Bounded, unpaced Mac fixture entrypoint. This module cannot launch a study."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from .controller import PacedController
from .events import sha256
from .fixtures import (
    FIXTURE_CHECKPOINT,
    fixture_identity,
    fixture_request,
    run_adversarial_fixtures,
)
from .receipt import file_hashes, make_receipt
from .record import Journal, export_json
from .replay import replay_journal
from .semantic import check_semantics
from .semantic_fixture import tiny_semantic_fixtures
from .transport import THREAD_ENV, ProcessTransport


def tiny_worker(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Real spawned JSON roundtrip; action choice is an injected fixture only."""
    return {"request": request, **payload["result"]}


class TinyGame:
    """Real GridBatchSim plus reacting opponents for an eight-tick fixture replay."""

    def __init__(self, header: Mapping[str, Any] | None = None) -> None:
        self.reset({} if header is None else header["initial_state"])

    def reset(self, initial_state: Mapping[str, Any]) -> None:
        self.runtime = tiny_semantic_fixtures()[0].candidate_factory()
        self.epoch = 0
        self.mass_sum = 0
        self.alive_frames = 0
        self.deaths = 0
        self.terminal = False
        self.cutoff_base_ns: int | None = None
        self.expected_submissions: list[Mapping[str, Any]] = []
        if (
            initial_state.get("snapshot_base64")
            and base64.b64decode(initial_state["snapshot_base64"]) != self.runtime.snapshot()
        ):
            raise ValueError("fixture initial game/opponent/RNG bytes changed")

    def context(self) -> dict[str, Any]:
        return {
            "session": "tiny-real-game",
            "life": 0,
            "frame": int(self.runtime.sim.frame[0]),
            "checkpoint_sha": FIXTURE_CHECKPOINT,
            "shadow_epoch": self.epoch,
            "root_digest": hashlib.sha256(self.runtime.snapshot()).hexdigest(),
        }

    def mask(self) -> list[bool]:
        return self.runtime.sim.get_legal_action_mask()[0, 0].astype(bool).tolist()

    def bare(self) -> int:
        mask = self.mask()
        return 1 if mask[1] else next((index for index, legal in enumerate(mask) if legal), 0)

    def observe(self, kind: str, time_ns: int, data: Mapping[str, Any]) -> None:
        if kind == "clock_read" and data.get("label") == "request_cutoff_base":
            self.cutoff_base_ns = data["value_ns"]
            self.expected_submissions.append(self.submission(time_ns))

    def finish_expectations(self) -> Mapping[str, Any]:
        return {"submissions": self.expected_submissions, "dispatches": ["real-spawn"]}

    def submission(self, now_ns: int) -> Mapping[str, Any]:
        """Independently reconstruct the fixture root, payload and deadline."""
        context = self.context()
        bare, mask = self.bare(), self.mask()
        action = next((index for index, legal in enumerate(mask) if legal and index != bare), bare)
        payload = {"result": {"action": action, "bare": bare, "mask": mask}}
        if self.cutoff_base_ns is None:
            raise ValueError("missing root-target clock observation")
        request = replace(
            fixture_request("real-spawn"),
            session=context["session"],
            source_frame=context["frame"],
            target_frame=context["frame"],
            root_digest=context["root_digest"],
            payload_sha=sha256(payload),
            cutoff_ns=self.cutoff_base_ns + 15_000_000_000,
        )
        return {"request": asdict(request), "payload": payload}

    def validate(self, payload: Mapping[str, Any]) -> bool:
        request = payload.get("request", {})
        context = self.context()
        return (
            all(
                request.get(key) == context["frame" if key == "target_frame" else key]
                for key in context
                if key != "frame"
            )
            and request.get("target_frame") == context["frame"]
            and payload.get("bare") == self.bare()
            and payload.get("mask") == self.mask()
        )

    def step(self, action: int) -> dict[str, Any]:
        bare = self.bare()
        row = self.runtime.step(action)
        if action != bare:
            self.epoch += 1
        if not self.terminal:
            self.alive_frames += int(row["hero_alive"])
            self.mass_sum += row["integer_mass_sum"]
            if row["terminal"]:
                self.terminal = True
                self.deaths += 1
        return {
            "state_digest": hashlib.sha256(self.runtime.snapshot()).hexdigest(),
            "digest_schema": "partial-fixture-state/v1",
            "mass_sum": self.mass_sum,
            "alive_frames": self.alive_frames,
            "deaths": self.deaths,
            "frame": self.context()["frame"],
            "opponent_actions": row["opponent_actions"],
        }


def run_preflight(directory: Path) -> dict[str, Any]:
    """Run only tiny synthetic checks, one child and one math thread each."""
    if platform.system() != "Darwin":
        raise RuntimeError("Mac-only preflight")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("run preflight under nice -n 10")
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise RuntimeError("all numerical thread env limits must be 1")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic_ns()
    semantic = check_semantics(tiny_semantic_fixtures())
    export_json(directory / "semantic.json", semantic)
    if semantic["verdict"] != "PASS":
        raise RuntimeError(f"semantic fixture failed: {semantic['first_divergence']}")
    fixtures = run_adversarial_fixtures(directory / "interleavings")
    identity = fixture_identity()
    game = TinyGame()
    initial_state = {
        "synthetic": True,
        "fixture_id": "tiny-grid-seed741-H8",
        "shadow_epoch": 0,
        "snapshot_base64": base64.b64encode(game.runtime.snapshot()).decode("ascii"),
    }
    journal = Journal(directory / "real-game-trace", asdict(identity), initial_state, capacity=512)
    controller = PacedController(identity, journal)
    transport = ProcessTransport(controller, tiny_worker)
    context = game.context()
    bare, mask = game.bare(), game.mask()
    action = next((index for index, legal in enumerate(mask) if legal and index != bare), bare)
    payload = {"result": {"action": action, "bare": bare, "mask": mask}}
    cutoff_base = controller.read_clock("request_cutoff_base")
    request = replace(
        fixture_request("real-spawn"),
        session=context["session"],
        source_frame=context["frame"],
        target_frame=context["frame"],
        root_digest=context["root_digest"],
        payload_sha=sha256(payload),
        cutoff_ns=cutoff_base + 15_000_000_000,
    )
    if not transport.submit(request, payload):
        raise RuntimeError("tiny transport admission refused")
    transport.dispatch()
    # An explicit off-tick join is allowed only in this UNPACED fixture.
    transport.close()
    for _ in range(8):
        context = game.context()
        chosen = controller.poll(context, game.bare(), game.mask(), game.validate)
        controller.game_state(game.step(chosen))
        journal.drain()
    seal = journal.seal()
    replay = replay_journal(journal.directory, TinyGame)
    export_json(directory / "replay.json", replay)
    if replay["verdict"] != "PASS":
        raise RuntimeError(f"tiny game replay failed: {replay}")
    elapsed = (time.monotonic_ns() - start) / 1e9
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    root = Path(__file__).resolve().parents[2]
    # Bind the imported local closure without walking external artifacts.
    own_sources = {path.resolve() for path in source_paths}
    reference_paths = sorted(
        {
            Path(module.__file__).resolve()
            for module in list(sys.modules.values())
            if getattr(module, "__file__", None)
            and Path(module.__file__).suffix == ".py"
            and Path(module.__file__).is_file()
            and Path(module.__file__).resolve().is_relative_to(root)
            and Path(module.__file__).resolve() not in own_sources
        }
    )
    forecast = {
        "scope": "BUILD_ONLY_NO_STUDY_DISPATCH",
        "colab_units": 0,
        "preflight_elapsed_seconds_contended_mac": elapsed,
        "quiet_latency_or_throughput_measurement": False,
        "reserved_future_semantic_mac_hours": 2,
        "reserved_future_unpaced_replay_mac_hours": 8,
        "l3_paced_cap_mac_hours": 9,
        "l4_paced_cap_mac_hours": 32,
        "illustrative_640_world_candidate_full_horizon_paced_hours": 640 * 5000 / 24 / 3600,
        "strict_bank_size": None,
        "strict_bare_comparator_hours": None,
        "production_semantic_forecast_hours": None,
        "production_replay_forecast_hours": None,
        "note": "tiny fixture cannot price production replay, semantic or paced serving",
    }
    receipt = make_receipt(
        identity=asdict(identity),
        semantic=semantic,
        replay=replay,
        paced={"verdict": "NOT_RUN", "reason": "owner requested no paced study runs"},
        source_hashes=file_hashes(source_paths),
        input_hashes={"initial_state": sha256(initial_state)},
        reference_hashes=file_hashes(reference_paths),
        commands=[" ".join(sys.argv)],
        journal_seals=[seal, *[row["seal"] for row in fixtures["fixtures"]]],
        forecast=forecast,
    )
    receipt["preflight"] = {
        "verdict": "PASS",
        "synthetic_fixture_count": fixtures["fixture_count"],
        "real_game_ticks": 8,
        "real_worker_processes": 1,
        "math_threads_per_process": 1,
        "phase0_running": True,
        "update_p95_ns": None,
        "overruns": None,
        "timing_qualification": "NOT_MEASURED_CONTENDED_FIXTURE",
        "paced_study_runs": 0,
        "semantic_production_bank_run": False,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    receipt["libraries"].update(
        {name: importlib.metadata.version(name) for name in ("numpy", "torch", "numba")}
    )
    export_json(directory / "receipt.json", receipt)
    manifest = {
        "schema": "bl-paced-identity-build-manifest/v1",
        "status": "BUILD_VERIFIED_NOT_QUALIFIED",
        "dispatchable": False,
        "receipt_sha256": sha256(receipt),
        "source_sha256": receipt["source_sha256"],
        "configurations": [
            {"workers": 1, "max_pending": 1, "owner_multi_worker": False},
            {"workers": 4, "max_pending": 8, "owner_multi_worker": "PENDING"},
        ],
        "forecast": forecast,
        "family_policy": receipt["family_policy"],
        "missing_admission": [
            "production_18_plus_96_semantic_receipt",
            "frozen_candidate_sources_roots_controller_and_load_envelope",
            "quiet_mac_recorder_and_L1_admission",
            "L2_FUND_SYSTEMS_and_frozen_L3_authorization",
            "fresh_paced_runs_each_own_full_replay",
        ],
    }
    export_json(directory / "manifest.json", manifest)
    return {"receipt": str(directory / "receipt.json"), "preflight": "PASS", "elapsed": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh absolute artifact directory")
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error("--out must be absolute")
    import json

    print(json.dumps(run_preflight(args.out), sort_keys=True))


if __name__ == "__main__":
    main()
