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


def pid_start(pid: int) -> Optional[float]:
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:  # noqa: BLE001 - no psutil / no such process
        return None


def pid_alive(pid: int, started: Optional[float] = None) -> bool:
    """Is ``pid`` alive (and, if ``started`` is known, the same process, not a reuse)?"""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (TypeError, ValueError):
        return False
    if started is not None:
        now_start = pid_start(pid)
        if now_start is not None and abs(now_start - float(started)) > 1.0:
            return False
    return True


def _alive(alive: Callable[..., bool], run: Mapping[str, Any]) -> bool:
    try:
        return alive(run["pid"], run.get("pid_start"))
    except TypeError:  # simple one-argument test doubles
        return alive(run["pid"])


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
    def locked(self, write: bool = True) -> Iterator[Dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            try:
                state = (
                    json.loads(self.path.read_text())
                    if self.path.exists()
                    else {"schema": SCHEMA, "runs": {}}
                )
                yield state
                if write:
                    tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
                    with tmp.open("w") as fh:
                        fh.write(json.dumps(state, indent=1, sort_keys=True))
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, self.path)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _mirror(self, row: Mapping[str, Any]) -> None:
        """Mirror into the legacy ledger.jsonl (tagged v2) so an old-format runner counts
        v2 runs: an unsettled v2 run is a full-budget reservation to it."""
        with self.legacy_path.open("a") as fh:
            fh.write(json.dumps({**row, "v2": True}, sort_keys=True) + "\n")

    def legacy(self) -> Dict[str, Any]:
        reserved: Dict[str, Dict[str, Any]] = {}
        settled: Dict[str, float] = {}
        if self.legacy_path.exists():
            for line in self.legacy_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:  # a legacy writer's partial last line
                    continue
                if row.get("v2"):
                    continue
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
            if run_id in state["runs"]:
                raise DuplicateRun(f"run id {run_id} is already registered")
            for rid, run in state["runs"].items():
                if run["job_id"] != job_id:
                    continue
                if run["status"] == "live" and _alive(self.alive, run):
                    raise DuplicateRun(f"job {job_id} already has a live run {rid}")
                if run["status"] == "finished" and run.get("episodes_complete"):
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
                "pid_start": pid_start(pid),
                "budget": float(budget),
                "status": "live",
                "started": self.clock(),
                "pods": {},
            }
            self._mirror(
                {
                    "kind": "reserve",
                    "run": run_id,
                    "job_id": job_id,
                    "budget_usd": float(budget),
                    "utc": time.time(),
                }
            )
            return self.totals(state)

    def finish_run(self, run_id: str, success: bool, episodes_complete: bool) -> None:
        with self.locked() as state:
            run = state["runs"].get(run_id)
            if run is not None:
                run["status"] = "finished"
                run["success"] = bool(success)
                run["episodes_complete"] = bool(episodes_complete)
                run["finished"] = self.clock()
                cost = sum(self._pod_spent(p, self.clock()) for p in run["pods"].values())
                self._mirror(
                    {
                        "kind": "settle",
                        "run": run_id,
                        "cost_usd": round(cost, 5),
                        "utc": time.time(),
                    }
                )

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
                if r["status"] == "live" and _alive(self.alive, r)
            )
            return out

    def peek(self) -> Dict[str, Any]:
        """Read-only snapshot (shared lock, never writes)."""
        if not self.path.exists():
            return {**self.totals({"runs": {}}), "live_runs": [], "live_jobs": []}
        with self.locked(write=False) as state:
            out = self.totals(state)
            out["live_runs"] = sorted(
                rid
                for rid, r in state["runs"].items()
                if r["status"] == "live" and _alive(self.alive, r)
            )
            out["live_jobs"] = sorted(
                {
                    r["job_id"]
                    for r in state["runs"].values()
                    if r["status"] == "live" and _alive(self.alive, r)
                }
            )
            return out

    # ------------------------------------------------------------ reconciliation
    def reconcile(self, rp: Any, delete_dead: bool = True) -> Dict[str, Any]:
        """Square the ledger with RunPod (race-safe across concurrent runs).

        * Pods created within 300 s of the listing are skipped (a fresh pod may be missing
          from GET /pods, or be created between the listing and the lock).
        * A bound pod is released only after it is absent from two listings >= 60 s apart.
        * A pending reservation (unresolved failed create) is matched by NAME: if listed it is
          bound and treated as a pod; it is released only when absent and past its horizon.
        * A listed pod of a DEAD run past its horizon is deleted.
        """
        t_list = self.clock()
        rows = rp.list_pods()
        by_id = {str(p.get("id")): p for p in rows}
        by_name = {str(p.get("name")): p for p in rows}
        released, deleted = [], []
        with self.locked() as state:
            now = self.clock()
            for rid, run in state["runs"].items():
                live_run = run["status"] == "live" and _alive(self.alive, run)
                for key, pod in run["pods"].items():
                    if pod.get("deleted") is not None:
                        continue
                    if float(pod.get("created") or now) >= t_list - 300:
                        continue
                    past = now > float(pod["until"]) + SLACK_SECONDS
                    if pod.get("pod_id") is None:
                        if key in by_name:
                            pod["pod_id"] = str(by_name[key]["id"])
                        else:
                            if past:
                                pod["deleted"], pod["cost"] = now, 0.0
                                released.append(key)
                            continue
                    pid = pod["pod_id"]
                    if pid in by_id:
                        pod.pop("missing_since", None)
                        if past and not live_run and delete_dead:
                            deleted.append((rid, key, pid))
                        continue
                    first = pod.get("missing_since")
                    if first is None:
                        pod["missing_since"] = t_list
                    elif t_list - float(first) >= 60:
                        end = min(float(first), float(pod["until"]))
                        pod["deleted"] = now
                        pod["cost"] = float(pod["rate"]) * max(0.0, end - pod["created"]) / 3600
                        released.append(key)
        for rid, key, pid in deleted:
            try:
                rp.delete_pod(pid, confirm=True)
            except Exception:  # noqa: BLE001 - next reconcile retries
                continue
        return {"released": released, "deleted_overdue_dead_run_pods": [d[2] for d in deleted]}
