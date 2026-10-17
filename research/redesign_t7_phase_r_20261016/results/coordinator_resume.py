"""Operational-only recovery for macOS killpg permission race; frozen units unchanged.

The original coordinator repeatedly SIGCONT'ed even running process groups. macOS
returned EPERM during a completed-worker transition and again during cleanup.
Route signals only to tracked, unreaped Popen children. Single-process evaluation
workers need no group signal, and a running worker needs no SIGCONT.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path('/Users/josenunez/Projects/ml/snake-dqn-t7')
sys.path.insert(0, str(REPO))
from research.redesign_t7_phase_r_20261016 import phase_r_t7 as runner

ORIGINAL_POPEN = subprocess.Popen
OWNED = {}
PAUSED = set()


def tracked_popen(*args, **kwargs):
    proc = ORIGINAL_POPEN(*args, **kwargs)
    argv = args[0] if args else kwargs.get('args', [])
    if isinstance(argv, list) and 'unit' in argv and str(Path(runner.__file__).resolve()) in argv:
        OWNED[proc.pid] = proc
        print(f'OWNED_WORKER pid={proc.pid}', flush=True)
    return proc


def owned_signal(pid, sig):
    proc = OWNED.get(pid)
    if proc is None:
        raise RuntimeError(f'refusing signal to unowned PID {pid}')
    if proc.poll() is not None:
        PAUSED.discard(pid)
        return
    if sig == signal.SIGCONT and pid not in PAUSED:
        return
    proc.send_signal(sig)
    if sig == signal.SIGSTOP:
        PAUSED.add(pid)
    elif sig == signal.SIGCONT:
        PAUSED.discard(pid)


def selftest():
    proc = ORIGINAL_POPEN([sys.executable, '-c', 'import time; time.sleep(60)'])
    OWNED[proc.pid] = proc
    try:
        owned_signal(proc.pid, signal.SIGCONT)
        owned_signal(proc.pid, signal.SIGSTOP)
        assert proc.pid in PAUSED
        owned_signal(proc.pid, signal.SIGCONT)
        assert proc.pid not in PAUSED
        owned_signal(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
        owned_signal(proc.pid, signal.SIGCONT)
        try:
            owned_signal(-1, signal.SIGTERM)
        except RuntimeError:
            pass
        else:
            raise AssertionError('unowned PID accepted')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    print('OWNED_SIGNAL_SELFTEST_PASS', flush=True)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        subprocess.Popen = tracked_popen
        os.killpg = owned_signal
        raise SystemExit(runner.run(runner.ART_ROOT, 6))
