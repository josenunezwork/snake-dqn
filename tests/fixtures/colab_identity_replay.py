"""Network-free CLI boundary for assignment-to-first-exec identity regression.

Run under ``python -S`` so the real supervisor-generated sitecustomize can be
loaded after installing these small stand-ins for the pinned client modules.
Only provider assignment/HTTP and the upstream StateStore are replaced.
"""

import json
import os
import runpy
import sys
import types
import uuid
from pathlib import Path

config = Path(os.environ["SNAKE_COLAB_CONFIG"])
session = os.environ["SNAKE_COLAB_SESSION"]
events = config.with_suffix(".events.jsonl")


def event(value: dict) -> None:
    with events.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


class Store:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def get(self, name: str) -> object:
        assert name == session
        return types.SimpleNamespace(**json.loads(self.path.read_text()))

    def add(self, state: object) -> None:
        self.path.write_text(json.dumps(vars(state)))


class Client:
    def assign(self, notebook_hash: uuid.UUID) -> object:
        event({"assign_uuid": str(notebook_hash)})
        return types.SimpleNamespace(endpoint="owned")

    def _issue_request(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("network forbidden")


class Session:
    def request(self, method: str, url: str, **kwargs: object) -> object:
        assert method == "POST"
        assert url == "https://owned.example//api/kernels"
        event({"post": url})
        return types.SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"id": "created-by-us"}
        )


sys.modules["colab_cli"] = types.ModuleType("colab_cli")
sys.modules["colab_cli.client"] = types.SimpleNamespace(Client=Client)
sys.modules["colab_cli.state"] = types.SimpleNamespace(StateStore=Store)
sys.modules["requests"] = types.SimpleNamespace(sessions=types.SimpleNamespace(Session=Session))
# Use exactly the hook file and environment exported by CLI._call_once.
runpy.run_path(str(Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / "sitecustomize.py"))
if "new" in sys.argv:
    Client().assign(uuid.uuid4())
    Store(str(config)).add(
        types.SimpleNamespace(
            endpoint="owned", url="https://owned.example/", token="private", kernel_id=None
        )
    )
else:
    assert "exec" in sys.argv
    Session().request(
        "POST",
        "https://owned.example//api/kernels",
        headers={"X-Colab-Runtime-Proxy-Token": "private"},
    )
