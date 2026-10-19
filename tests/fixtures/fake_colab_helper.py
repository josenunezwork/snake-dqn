#!/usr/bin/env python3
"""Offline identity, credential and idle helper; never contacts a provider."""

import json
import os
import sys
import time
from pathlib import Path

if "-c" in sys.argv:
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
    print(json.dumps({"idle": os.environ.get("FAKE_COLAB_MODE") != "busy_exec"}))
elif "--recover-assignment" in sys.argv:
    state = json.loads(Path(os.environ["FAKE_COLAB_STATE"]).read_text())
    key = sys.argv[-1].replace("-", "")
    endpoint = next(
        (e for e, r in state["sessions"].items() if r.get("notebook_hash") == key), None
    )
    print(json.dumps({"endpoint": endpoint}))
elif "--unassign-endpoint" in sys.argv:
    path = Path(os.environ["FAKE_COLAB_STATE"])
    state = json.loads(path.read_text())
    if os.environ.get("FAKE_COLAB_MODE") != "stop_lie":
        state["sessions"].pop(sys.argv[-1], None)
        path.write_text(json.dumps(state))
    print(json.dumps({"unassigned": True}))
else:
    raise SystemExit("unsupported fake helper protocol: " + repr(sys.argv[1:]))
