#!/usr/bin/env python3
"""Offline identity, credential and idle helper; never contacts a provider."""

import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if "-c" in sys.argv:
    code = sys.argv[sys.argv.index("-c") + 1]
    if "SNAKE_ACCEPTANCE_KERNEL_METADATA_" in code:
        path, session = Path(sys.argv[-2]), sys.argv[-1]
        sessions = json.loads(path.read_text())
        saved = sessions[session]
        result = {key: saved[key] for key in ("kernel_id", "endpoint", "url")}
        result["token_sha256"] = hashlib.sha256(saved["token"].encode()).hexdigest()
        if "SNAKE_ACCEPTANCE_KERNEL_METADATA_CLEAR" in code:
            saved["kernel_id"] = None
            path.write_text(json.dumps(sessions, sort_keys=True))
        print(json.dumps(result))
    elif "REFRESH_OK" in code:
        print("REFRESH_OK")
    else:
        print(
            json.dumps(
                {
                    "access_token": "fake-downscoped-access-token",
                    "expires_at": time.time() + 3600,
                    "scopes": json.loads(sys.argv[-1]),
                }
            )
        )
elif "--kernel-idle" in sys.argv:
    idle = os.environ.get("FAKE_COLAB_MODE") != "busy_exec"
    if idle:
        path, session = Path(sys.argv[-2]), sys.argv[-1]
        if path.exists():
            sessions = json.loads(path.read_text())
            if session in sessions and not sessions[session].get("kernel_id"):
                sessions[session]["kernel_id"] = "fake-kernel-id"
                path.write_text(json.dumps(sessions, sort_keys=True))
    print(json.dumps({"idle": idle}))
elif "--runtime-ready" in sys.argv:
    print(json.dumps({"ready": True}))
elif "--recover-assignment" in sys.argv:
    state = json.loads(Path(os.environ["FAKE_COLAB_STATE"]).read_text())
    key = sys.argv[-1].replace("-", "")
    endpoint = next(
        (e for e, r in state["sessions"].items() if r.get("notebook_hash") == key), None
    )
    print(json.dumps({"endpoint": endpoint}))
elif "--unassign-endpoint" in sys.argv:
    path = Path(os.environ["FAKE_COLAB_STATE"])
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        if os.environ.get("FAKE_COLAB_MODE") != "stop_lie":
            removed = state["sessions"].pop(sys.argv[-1], None)
            if removed:
                state["rate"] = max(
                    0, state["rate"] - float(os.environ.get("FAKE_COLAB_OWN_RATE", "6.84"))
                )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state))
            os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(json.dumps({"unassigned": True}))

else:
    raise SystemExit("unsupported fake helper protocol: " + repr(sys.argv[1:]))
