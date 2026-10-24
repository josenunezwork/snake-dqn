"""Small Darwin ``proc_pid_rusage`` binding used by factory companions.

The V2 child counters survive a reaped short-lived descendant.  On this ARM
Darwin host they are Mach ticks, converted through ``mach_timebase_info``;
they are not assumed to be nanoseconds.  Callers retain the raw start tick as
the PID identity and fail closed when it changes.
"""

from __future__ import annotations

import ctypes
import errno
import platform
from dataclasses import dataclass
from functools import lru_cache


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


@lru_cache(maxsize=1)
def _tick_seconds() -> float:
    """Convert Darwin rusage CPU ticks to seconds without changing start ticks."""
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = library.mach_timebase_info
    function.argtypes = [ctypes.POINTER(_MachTimebaseInfo)]
    function.restype = ctypes.c_int
    info = _MachTimebaseInfo()
    if function(ctypes.byref(info)) != 0 or not info.numer or not info.denom:
        raise RuntimeError("cannot obtain Darwin mach timebase")
    return (info.numer / info.denom) / 1_000_000_000


@dataclass(frozen=True)
class NativeUsage:
    """Cumulative native accounting attached to one kernel process identity."""

    pid: int
    start_tick: int
    self_cpu_seconds: float
    child_cpu_seconds: float
    resident_bytes: int


def usage(pid: int) -> NativeUsage:
    """Read Darwin RUSAGE_INFO_V2 or refuse unsupported/vanished processes."""
    if platform.system() != "Darwin":
        raise RuntimeError("proc_pid_rusage is only available on Darwin")
    if ctypes.sizeof(_RusageInfoV2) != 160:
        raise RuntimeError("unexpected Darwin rusage_info_v2 layout")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    result = _RusageInfoV2()
    if function(pid, 2, ctypes.byref(result)) != 0:
        code = ctypes.get_errno()
        raise ProcessLookupError(code, errno.errorcode.get(code, "proc_pid_rusage"))
    return NativeUsage(
        pid=pid,
        start_tick=int(result.proc_start_abstime),
        self_cpu_seconds=(result.user_time + result.system_time) * _tick_seconds(),
        child_cpu_seconds=(result.child_user_time + result.child_system_time) * _tick_seconds(),
        resident_bytes=int(result.resident_size),
    )
