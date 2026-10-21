"""Development-only six-thread branch-window transport; original math is reused."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class BranchPool:
    """Use the calling controller worker plus N-1 helpers, with one search job."""

    def __init__(self, workers: int, evaluator: Callable[..., Any]) -> None:
        if workers not in (1, 6):
            raise ValueError("smoke supports exactly one or six worker threads")
        self.workers = workers
        self.evaluator = evaluator
        self.observer: Any = None
        self.pool = (
            ThreadPoolExecutor(workers - 1, thread_name_prefix="smoke-branch")
            if workers > 1
            else None
        )

    def evaluate(
        self, decisions: Any, horizons: Any, *, cancel_event: Any = None, on_complete: Any = None
    ) -> list[dict]:
        original_callback = on_complete

        def observed(index: int, outcomes: dict) -> bool:
            if self.observer is not None:
                self.observer(decisions[index], outcomes)
            return bool(original_callback and original_callback(index, outcomes))

        on_complete = observed
        if self.pool is None:
            return self.evaluator(
                decisions, horizons, cancel_event=cancel_event, on_complete=on_complete
            )
        from research.t34_pathb.engine_v2 import _thread_cap

        # Contiguous stable chunks, no extra outstanding controller request.
        chunks = [
            list(
                range(i * len(decisions) // self.workers, (i + 1) * len(decisions) // self.workers)
            )
            for i in range(self.workers)
        ]
        chunks = [indices for indices in chunks if indices]
        lock = threading.Lock()
        done = threading.Event()
        results: list[dict] = [{} for _ in decisions]

        class Cancel:
            def is_set(self) -> bool:
                return done.is_set() or bool(cancel_event and cancel_event.is_set())

        def run(indices: list[int]) -> None:
            _thread_cap()

            def complete(local: int, outcomes: dict) -> bool:
                index = indices[local]
                with lock:
                    results[index] = outcomes
                    if not done.is_set() and on_complete and on_complete(index, outcomes):
                        done.set()
                return done.is_set()

            # A sibling success stops at the next fully evaluated source, so
            # no partial evaluation can escape as an improving action.
            try:
                self.evaluator(
                    [decisions[i] for i in indices],
                    [horizons[i] for i in indices],
                    cancel_event=Cancel(),
                    on_complete=complete,
                )
            except TimeoutError:
                if not done.is_set():
                    raise

        futures = [self.pool.submit(run, chunk) for chunk in chunks[1:]]
        try:
            run(chunks[0])
        finally:
            for future in futures:
                future.result()
        return results

    def close(self) -> None:
        """Drain only our helper threads; never signal another process."""
        if self.pool is not None:
            self.pool.shutdown(wait=True)
