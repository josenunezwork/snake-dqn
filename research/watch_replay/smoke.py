"""Tiny Mac serial/parallel byte proof and actual WebSocket playback/fallback smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from research.watch_replay.run import allocation, identity
from tools.ops.package_spec import digest, empty_ledger, publish_json
from web.backend.watch_replay import Recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    contract = {
        **allocation("t34-watch-replay-parity-fixture-v1", 3, "exact-stop-l2-prime", 12),
        "commit": commit,
    }
    publish_json(args.out / "contract.json", contract)
    publish_json(
        args.out / "ledger.json",
        empty_ledger(contract["namespace"], "sessions", identity(contract)),
    )
    for label, workers in (("serial", 1), ("parallel", 2)):
        with (args.out / f"{label}.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.watch_replay.run",
                    "--contract",
                    str(args.out / "contract.json"),
                    "--ledger",
                    str(args.out / "ledger.json"),
                    "--out",
                    str(args.out / label),
                    "--workers",
                    str(workers),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    checksums = []
    for index in range(3):
        first = args.out / "serial" / f"{index:06d}.watch.gz"
        second = args.out / "parallel" / first.name
        if digest(first) != digest(second):
            raise ValueError(f"serial/parallel replay bytes diverged at {index}")
        checksums.append(digest(first))

    from fastapi.testclient import TestClient

    from research.watch_replay.product import open_watch
    from web.backend import app as app_module

    bridge = open_watch({"seed": 91842001}, 24, "bare")
    session = bridge.live
    record = Recording(args.out / "serial/000000.watch.gz", product=False)
    expected = record.frame
    session._watch_replay = record
    session.playing = False
    live_frame = session.game.frame
    try:
        # Only this explicit offline test injects an excluded short fixture.
        # ReplayBank always refuses fixtures in product operation.
        with (
            patch.dict("os.environ", {"SNAKE_SCORES_DB": ":memory:"}),
            patch.object(app_module, "GameSession", return_value=session),
            TestClient(app_module.app) as client,
        ):
            with client.websocket_connect("/ws/stream") as ws:
                got = ws.receive_json()
                assert got["session"]["watch_source"] == "replay"
                for key in ("snakes", "food", "stats", "inspector", "netviz", "events"):
                    assert got[key] == json.loads(json.dumps(expected[key]))
                assert session.game.frame == live_frame
                session._watch_replay = None
                session.playing = True
                session.step()
                fallback = session.snapshot()
                assert session.game.frame == live_frame + 1
                assert fallback["checkpoint_name"] == "b8_s3_t7_20261016.pth"
                assert "replay" not in fallback["session"]
                assert session.pathb is None
                session.set_mode("play")
                play = session.snapshot()
                assert play["play"] is not None and session.mode == "play"
                session.human_input("up")
                session.step()
                assert session.run_frames > 0
    finally:
        bridge.close()
    publish_json(
        args.out / "smoke.json",
        {
            "status": "PASS",
            "commit": commit,
            "horizon": 12,
            "fixture_units": 3,
            "serial_equals_parallel_sha256": checksums,
            "websocket_recorded_dto_parity": True,
            "live_b8_fallback_tick": True,
            "play_live_input_tick": True,
        },
    )
    print("PASS: 3 serial==parallel recordings; WebSocket DTO parity; live B8 fallback; live Play")


if __name__ == "__main__":
    main()
