"""Focused priority checks for persistent operational launcher shells."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

from tools.ops.dispatch import NICE_FLOOR, persistent_shell


def test_persistent_shell_floors_outer_shell_and_descendant() -> None:
    """The pane's exec-replaced shell and its child both retain nice >= 15."""
    inherited_nice = os.getpriority(os.PRIO_PROCESS, 0)
    probe = """
import json
import os
import subprocess
import sys

child = subprocess.check_output(
    [sys.executable, '-c', 'import os; print(os.getpriority(os.PRIO_PROCESS, 0))'], text=True
)
shell = subprocess.check_output(['/bin/ps', '-o', 'nice=', '-p', str(os.getppid())], text=True)
print(json.dumps(dict(shell=int(shell), self=os.getpriority(os.PRIO_PROCESS, 0),
                      child=int(child))))
"""
    command = persistent_shell(f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}; :")
    result = subprocess.run(
        ["/bin/bash", "-c", command], capture_output=True, check=True, text=True, timeout=10
    )

    observed = json.loads(result.stdout)
    expected_nice = max(inherited_nice, NICE_FLOOR)
    assert observed == {"shell": expected_nice, "self": expected_nice, "child": expected_nice}


def test_real_colab_launcher_with_stub_supervisor_only(tmp_path) -> None:
    """Exercise the actual launcher, substituting a stdlib priority-only client."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    probe = tmp_path / "supervisor-probe"
    probe.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,subprocess,sys\n"
        "code='import os; print(os.getpriority(os.PRIO_PROCESS,0))'\n"
        "child=subprocess.check_output([sys.executable,'-c',code],text=True)\n"
        "print(json.dumps(dict(nice=os.getpriority(os.PRIO_PROCESS,0),\n"
        "                     child=int(child),args=sys.argv[1:])))\n"
    )
    probe.chmod(0o700)
    command = persistent_shell(
        shlex.join([str(root / "research/colab_cli/launch.sh"), "status"]) + "; :"
    )
    env = {**os.environ, "SNAKE_DQN_PYTHON": str(probe)}
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    observed = json.loads(result.stdout)
    assert observed["nice"] == observed["child"] == max(15, os.getpriority(os.PRIO_PROCESS, 0))
    assert observed["args"] == ["-m", "research.colab_cli.supervisor", "status"]


def test_persistent_shell_preserves_an_already_lower_priority() -> None:
    """A shell inherited at nice 19 is never promoted to the floor value."""
    command = persistent_shell(
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import os; print(os.getpriority(os.PRIO_PROCESS, 0))")
    )
    inherited_nice = max(os.getpriority(os.PRIO_PROCESS, 0), 19)
    result = subprocess.run(
        [
            "/usr/bin/nice",
            "-n",
            str(inherited_nice - os.getpriority(os.PRIO_PROCESS, 0)),
            "/bin/bash",
            "-c",
            command,
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )

    assert int(result.stdout) == inherited_nice
