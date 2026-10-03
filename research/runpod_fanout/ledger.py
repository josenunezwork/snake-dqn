"""Account-level spend ledger shared by every concurrent fan-out run (fcntl-guarded).

File: ``<artifacts>/runpod-fanout/ledger-v2.json`` (lock: ``ledger-v2.lock``). Every run
registers itself, reserves each pod's worst case BEFORE creating it, binds the pod id
after the create, and releases the pod (with its estimated cost) when it is deleted.

Global rules checked atomically at every reservation, across ALL live runs:

* remaining worst case of every live pod reservation (``rate x (until - now + slack)``)
  plus the new pod's ``<= balance_now - global_floor_usd``;
* the same ``<= project_cap_usd - project_spent`` (prior spend + settled/elapsed costs);
* live pods (all runs) ``< max_live_pods``.

A run whose process died keeps its reservations until each pod's ``until`` (its watchdog
deletes them by then). Legacy ``ledger.jsonl`` rows (the single-lock runner) still count:
settled costs as spend, unsettled budgets as full reservations.

Idempotency: :meth:`register_run` refuses a job id that already has a live run (pid
alive) or a successfully completed run, here or in the legacy ledger.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

SCHEMA = "runpod-fanout-ledger/v2"
SLACK_SECONDS = 300.0  # balance lags billing; every live reservation carries 5 min extra
SAFETY = 1.02


class DuplicateRun(RuntimeError):
    """The job id already has a live or successfully completed run."""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False
    return True


class SharedLedger:
    def __init__(
        self,
        policy: Mapping[str, Any],
        clock: Callable[[], float] = time.time,
        alive: Callable[[int], bool] = pid_alive,
    ):
        self.policy = policy
        self.root = Path(policy["artifacts_root"]) / "runpod-fanout"
        self.path = self.root / "ledger-v2.json"
        self.lock_path = self.root / "ledger-v2.lock"
        self.legacy_path = self.root / "ledger.jsonl"
        self.clock = clock
        self.alive = alive

    # ------------------------------------------------------------ storage
    @contextmanager
    def locked(self) -> Iterator[Dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                state = (
                    json.loads(self.path.read_text())
                    if self.path.exists()
                    else {"schema": SCHEMA, "runs": {}}
                )
                yield state
                tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
                os.replace(tmp, self.path)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def legacy(self) -> Dict[str, Any]:
        reserved: Dict[str, Dict[str, Any]] = {}
        settled: Dict[str, float] = {}
        if self.legacy_path.exists():
            for line in self.legacy_path.read_text().splitlines():
                row = json.loads(line)
                if row["kind"] == "reserve":
                    reserved[row["run"]] = row
                elif row["kind"] == "settle":
                    settled[row["run"]] = float(row["cost_usd"])
        open_runs = {r: row for r, row in reserved.items() if r not in settled}
        return {"settled": settled, "open": open_runs}

    # ------------------------------------------------------------ accounting
    def _pod_remaining(self, pod: Mapping[str, Any], now: float) -> float:
        if pod.get("deleted") is not None:
            return 0.0
        left = max(0.0, float(pod["until"]) - now) + SLACK_SECONDS
        return SAFETY * float(pod["rate"]) * left / 3600.0

    def _pod_spent(self, pod: Mapping[str, Any], now: float) -> float:
        if pod.get("deleted") is not None:
            return float(pod.get("cost") or 0.0)
        created = pod.get("created") or now
        return float(pod["rate"]) * max(0.0, min(now, float(pod["until"])) - created) / 3600.0

    def totals(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        now = self.clock()
        legacy = self.legacy()
        reserved = sum(float(r["budget_usd"]) for r in legacy["open"].values())
        spent = float(self.policy.get("prior_spend_usd", 0.0)) + sum(legacy["settled"].values())
        live_pods = 0
        for run in state["runs"].values():
            for pod in run["pods"].values():
                reserved += self._pod_remaining(pod, now)
                spent += self._pod_spent(pod, now)
                if pod.get("deleted") is None and float(pod["until"]) + SLACK_SECONDS > now:
                    live_pods += 1
        return {
            "reserved_usd": round(reserved, 5),
            "spent_usd": round(spent, 5),
            "live_pods": live_pods,
            "legacy_open_runs": sorted(legacy["open"]),
        }

    # ------------------------------------------------------------ runs
    def register_run(
        self, run_id: str, job_id: str, run_dir: str, budget: float, pid: int
    ) -> Dict[str, Any]:
        with self.locked() as state:
            for rid, run in state["runs"].items():
                if run["job_id"] != job_id:
                    continue
                if run["status"] == "live" and self.alive(run["pid"]):
                    raise DuplicateRun(f"job {job_id} already has a live run {rid}")
                if run["status"] == "finished" and run.get("success"):
                    raise DuplicateRun(f"job {job_id} already completed successfully ({rid})")
            for row in self.legacy()["open"].values():
                if row.get("job_id") == job_id:
                    raise DuplicateRun(f"job {job_id} has an unsettled legacy run {row['run']}")
            for receipt in sorted(self.root.glob(f"{job_id}/*/receipt.json")):
                try:
                    data = json.loads(receipt.read_text())
                except (OSError, ValueError):
                    continue
                if data.get("exit_code") == 0 and data.get("backend") != "local":
                    raise DuplicateRun(f"job {job_id} already completed ({receipt.parent})")
            state["runs"][run_id] = {
                "job_id": job_id,
                "run_dir": run_dir,
                "pid": pid,
                "budget": float(budget),
                "status": "live",
                "started": self.clock(),
                "pods": {},
            }
            return self.totals(state)

    def finish_run(self, run_id: str, success: bool) -> None:
        with self.locked() as state:
            run = state["runs"].get(run_id)
            if run is not None:
                run["status"] = "finished"
                run["success"] = bool(success)
                run["finished"] = self.clock()

    # ------------------------------------------------------------ pods
    def reserve_pod(
        self, run_id: str, key: str, rate: float, until: float, balance: float
    ) -> Optional[str]:
        """Atomically reserve a pod's worst case; returns a refusal reason or None."""
        now = self.clock()
        with self.locked() as state:
            totals = self.totals(state)
            new = SAFETY * rate * (max(0.0, until - now) + SLACK_SECONDS) / 3600.0
            floor = float(self.policy["global_floor_usd"])
            cap = float(self.policy["project_cap_usd"])
            if totals["live_pods"] + 1 > int(self.policy["max_live_pods"]):
                return f"global max_live_pods {self.policy['max_live_pods']} reached"
            if totals["reserved_usd"] + new > balance - floor:
                return (
                    f"account floor: reserved {totals['reserved_usd']:.3f} + new {new:.3f} > "
                    f"balance {balance:.2f} - floor {floor}"
                )
            if totals["reserved_usd"] + new > cap - totals["spent_usd"]:
                return (
                    f"project cap: reserved {totals['reserved_usd']:.3f} + new {new:.3f} > "
                    f"cap {cap} - spent {totals['spent_usd']:.3f}"
                )
            state["runs"][run_id]["pods"][key] = {
                "pod_id": None,
                "rate": float(rate),
                "created": now,
                "until": float(until),
                "deleted": None,
            }
            return None

    def bind_pod(self, run_id: str, key: str, pod_id: str, rate: float) -> None:
        with self.locked() as state:
            pod = state["runs"][run_id]["pods"][key]
            pod["pod_id"] = pod_id
            pod["rate"] = max(float(rate), float(pod["rate"]))

    def release_pod(self, run_id: str, key: str, cost: float) -> None:
        with self.locked() as state:
            pod = state["runs"].get(run_id, {}).get("pods", {}).get(key)
            if pod is not None and pod.get("deleted") is None:
                pod["deleted"] = self.clock()
                pod["cost"] = float(cost)

    def snapshot(self) -> Dict[str, Any]:
        with self.locked() as state:
            out = self.totals(state)
            out["live_runs"] = sorted(
                rid
                for rid, r in state["runs"].items()
                if r["status"] == "live" and self.alive(r["pid"])
            )
            return out
