"""Mutable demonstration-stream state for M3-C full checkpoints."""

from __future__ import annotations

import queue
from typing import Any


def _key(path) -> str:
    return f"{path.parent.name}/{path.name}"


def capture_demo_state(demos: Any, timeout: float = 30.0) -> dict | None:
    """Finish the one pending prefetch and expose state for caller-owned detachment.

    The ready queue is copied without consuming it, so live continuation and resume
    start with the same pending shard. The caller must subsequently capture_state
    this result before continuing training. Private buffers can add several GB.
    """
    if demos is None:
        return None
    thread = demos._thread
    if thread is not None:
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("demonstration prefetch did not reach checkpoint boundary")
    with demos._q.mutex:
        pending = list(demos._q.queue)
    result = {
        "order": [_key(path) for path in demos.order],
        "cap": demos.cap,
        "size": demos.size,
        "write": demos.write,
        "next_shard": demos.next_shard,
        "loaded": demos.loaded,
        "distinct": demos.distinct,
        "pending": pending,
        "store_header": None if demos.store is None else demos.store.header,
        "pack": None if demos.pack is None else demos.pack.to_dict(),
    }
    if demos.store is None:
        result["buf"] = demos.buf
    else:
        result["index"] = demos.index
    return result


def restore_demo_state(demos: Any, state: dict | None) -> None:
    """Restore into a freshly constructed stream, retaining its local data paths."""
    if demos is None or state is None:
        if demos is not None or state is not None:
            raise ValueError("full-state demonstration configuration differs")
        return
    if demos._thread is not None and demos._thread.is_alive():
        raise ValueError("restore requires an idle demonstration stream")
    if demos.cap != state["cap"]:
        raise ValueError("full-state demonstration capacity differs")
    header = None if demos.store is None else demos.store.header
    pack = None if demos.pack is None else demos.pack.to_dict()
    if header != state["store_header"] or pack != state["pack"]:
        raise ValueError("full-state demonstration store/layout differs")
    paths = {_key(path): path for path in demos.order}
    if set(paths) != set(state["order"]):
        raise ValueError("full-state demonstration shards differ")
    demos.order = [paths[key] for key in state["order"]]
    if demos.store is None:
        for name, value in state["buf"].items():
            if demos.buf[name].shape != value.shape or demos.buf[name].dtype != value.dtype:
                raise ValueError("full-state demonstration buffer shape/dtype differs")
            demos.buf[name][...] = value
    else:
        demos.index[...] = state["index"]
    for name in ("size", "write", "next_shard", "loaded", "distinct"):
        setattr(demos, name, state[name])
    demos._q = queue.Queue(maxsize=max(2, len(state["pending"])))
    for part in state["pending"]:
        demos._q.put_nowait(part)
    demos._thread = None
