"""Self-contained fire-and-forget kernel requests with durable at-most-once claims."""

from __future__ import annotations

import hashlib
import re


def request_code(code: str, request_id: str, namespace: str, nonce: str) -> str:
    """Build a launch/poll envelope; exclusive started claims are never removed.

    A worker dying between claim and completion leaves an ambiguous request.
    Polling may observe it, but must never execute that command again. Results,
    including command errors, are atomically published before acknowledgement.
    """
    if not all(re.fullmatch(r"[0-9a-f]{32}", value) for value in (request_id, namespace)):
        raise ValueError("request identity must be a 32-character lowercase hex UUID")
    digest = hashlib.sha256(code.encode()).hexdigest()
    header = f"# SNAKE_REQUEST_ID {request_id}\n# SNAKE_REQUEST_SHA256 {digest}\n"
    body = (
        "def _snake_request():\n"
        "    import json, os, threading\n"
        "    from pathlib import Path\n"
        f"    root = Path('/content/.snake-rpc-{namespace}')\n"
        "    root.mkdir(parents=True, exist_ok=True)\n"
        f"    started = root / '{request_id}.started'\n"
        f"    done = root / '{request_id}.done'\n"
        f"    identity = {{'request_id': {request_id!r}, 'sha256': {digest!r}}}\n"
        "    def command():\n"
        + "\n".join("        " + line for line in code.splitlines())
        + "\n        return result\n"
        "    def worker():\n"
        "        try:\n"
        "            reply = {'ok': True, 'value': command()}\n"
        "            payload = json.dumps({**identity, **reply})\n"
        "        except BaseException as exc:\n"
        "            payload = json.dumps({**identity, 'ok': False, 'error': repr(exc)})\n"
        "        temporary = done.with_suffix('.tmp')\n"
        "        with temporary.open('x') as handle:\n"
        "            handle.write(payload)\n"
        "            handle.flush()\n"
        "            os.fsync(handle.fileno())\n"
        "        os.replace(temporary, done)\n"
        "    try:\n"
        "        handle = started.open('x')\n"
        "    except FileExistsError:\n"
        "        # Even a partial claim forbids replay; no command can run twice.\n"
        "        if started.read_text() != json.dumps(identity, sort_keys=True):\n"
        "            raise RuntimeError('request claim identity mismatch; refusing replay')\n"
        "    else:\n"
        "        with handle:\n"
        "            handle.write(json.dumps(identity, sort_keys=True))\n"
        "            handle.flush()\n"
        "            os.fsync(handle.fileno())\n"
        "        threading.Thread(target=worker, daemon=True).start()\n"
        "    if done.exists():\n"
        "        reply = json.loads(done.read_text())\n"
        "        if any(reply.get(k) != v for k, v in identity.items()):\n"
        "            raise RuntimeError('request completion identity mismatch')\n"
        "        return reply\n"
        "    return {**identity, 'ok': True, 'pending': True}\n"
        "try:\n"
        "    _snake_reply = _snake_request()\n"
        "except BaseException as _snake_exc:\n"
        "    _snake_reply = {'ok': False, 'error': repr(_snake_exc)}\n"
        "import json as _snake_json\n"
        f"print({nonce!r} + _snake_json.dumps(_snake_reply))\n"
    )
    compile(header + body, "<colab-cli-request>", "exec")
    return header + body
