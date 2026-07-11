"""Run the snake-dqn web app (FastAPI backend + built frontend) on one port.

    ./venv/bin/python web/serve.py            # http://localhost:8000
    PORT=9000 ./venv/bin/python web/serve.py

Serves the API, the WebSocket frame stream, and the built React frontend
(web/frontend/dist) from a single origin. Build the frontend once first:
    cd web/frontend && npm install && npm run build
"""

import os
import sys

# Make the repo root importable (src.*, web.*) regardless of CWD.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import uvicorn  # noqa: E402

# Pin torch to a single intra-op thread. Inference for this small model runs
# inline in the async game-loop coroutine; letting torch spin up a thread per
# core there oversubscribes the CPU (measured ~3x slowdown) against the event
# loop and the frame serializer. One thread is faster and more predictable.
torch.set_num_threads(1)

if __name__ == "__main__":
    uvicorn.run(
        "web.backend.app:app",
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        log_level="warning",
    )
