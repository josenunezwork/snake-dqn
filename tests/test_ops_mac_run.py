"""Hermetic contracts for the pinned local-job dispatcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ops" / "mac_run.sh"


def run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def make_repo(tmp_path: Path, name: str = "repo") -> tuple[Path, str]:
    repo = tmp_path / name
    repo.mkdir()
    for args in (
        ("init",),
        ("config", "user.email", "ops@example.test"),
        ("config", "user.name", "Ops Test"),
    ):
        assert run(["git", *args], cwd=repo, env=os.environ.copy()).returncode == 0
    (repo / "README").write_text("fixture\n")
    assert run(["git", "add", "README"], cwd=repo, env=os.environ.copy()).returncode == 0
    assert run(["git", "commit", "-m", "fixture"], cwd=repo, env=os.environ.copy()).returncode == 0
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo, env=os.environ.copy()).stdout.strip()
    # mac_run starts this repository's watcher after registering the log.  The
    # fake tmux recognizes the path and deliberately does not execute it.
    watcher = repo / "tools" / "ops" / "watchd.sh"
    watcher.parent.mkdir(parents=True)
    watcher.write_text("#!/bin/bash\nexit 0\n")
    watcher.chmod(0o755)
    return repo, commit


def fake_mac_commands(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pmset").write_text("#!/bin/bash\necho 'Now drawing from AC Power'\n")
    (bin_dir / "ioreg").write_text("#!/bin/bash\necho '\"AppleClamshellState\" = No'\n")
    (bin_dir / "caffeinate").write_text(
        "#!/bin/bash\n"
        "while [ $# -gt 0 ] && [[ $1 == -* ]]; do shift; done\n"
        'if [ "${1:-}" = env ]; then shift; fi\n'
        'while [ $# -gt 0 ] && [[ $1 == *=* ]]; do export "$1"; shift; done\n'
        'exec "$@"\n'
    )
    (bin_dir / "tmux").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = has-session ]; then\n'
        '  [ "${OPS_FAKE_TMUX_TERM:-0}" = 1 ] && kill -TERM "$PPID"\n'
        "  exit 1\n"
        "fi\n"
        '[ "$1" = new-session ] || exit 9\n'
        'if [ "${4:-}" = watchd ]; then\n'
        '  printf "%s\\n" "${!#}" > "$OPS_DIR/watcher-command"; exit 0; fi\n'
        "shift\n"
        '[ "$1" = -d ] && shift\n'
        '[ "$1" = -s ] && shift 2\n'
        'if [ "${1:-}" = "exec $OPS_FAKE_WATCHER" ]; then exit 0; fi\n'
        '[ "${OPS_FAKE_TMUX_FAIL:-0}" = 1 ] && exit 12\n'
        'if [ "${OPS_FAKE_TMUX_TERM_SUBMIT:-0}" = 1 ]; then kill -TERM "$PPID"; exit 0; fi\n'
        'bash -c "$1"\n'
        "exit 0\n"
    )
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    return bin_dir


def test_mac_run_creates_pinned_detached_worktree_runs_and_registers_log(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path, "repo with spaces")
    bin_dir = fake_mac_commands(tmp_path)
    ops = tmp_path / "artifacts with spaces" / "ops"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", OPS_DIR=str(ops))
    # The fake tmux recognizes the runner's watcher command only after the job's
    # own runner has executed, so it exercises the real lock cleanup path.
    env["OPS_FAKE_WATCHER"] = str(repo / "tools" / "ops" / "watchd.sh")

    result = run(
        [
            str(SCRIPT),
            "--repo",
            str(repo),
            "quick",
            commit,
            "--",
            "sh",
            "-c",
            'test -z "$(git status --porcelain)" && echo completed',
        ],
        cwd=repo,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    worktree = tmp_path / "snake-dqn-run-quick.noindex"
    assert run(["git", "rev-parse", "HEAD"], cwd=worktree, env=env).stdout.strip() == commit
    assert run(["git", "symbolic-ref", "-q", "HEAD"], cwd=worktree, env=env).returncode != 0
    assert run(["git", "status", "--porcelain"], cwd=worktree, env=env).stdout == ""
    assert not (worktree / ".IN_USE_LOCK").exists()
    log = ops / "logs" / f"mac-quick-{commit[:12]}.log"
    assert "completed" in log.read_text()
    assert "EXIT=0" in log.read_text()
    assert f"{log} mac-quick" in (ops / "watch-logs.txt").read_text()
    watcher_command = (ops / "watcher-command").read_text()
    assert "OPS_DIR=" in watcher_command and "OPS_REPO=" in watcher_command


def test_mac_run_records_a_nonzero_job_exit_and_releases_its_own_lock(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    bin_dir = fake_mac_commands(tmp_path)
    ops = tmp_path / "artifacts" / "ops"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", OPS_DIR=str(ops))
    env["OPS_FAKE_WATCHER"] = str(repo / "tools" / "ops" / "watchd.sh")

    result = run(
        [str(SCRIPT), "--repo", str(repo), "fails", commit, "--", "sh", "-c", "exit 7"],
        cwd=repo,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    worktree = tmp_path / "snake-dqn-run-fails.noindex"
    assert not (worktree / ".IN_USE_LOCK").exists()
    assert "EXIT=7" in (ops / "logs" / f"mac-fails-{commit[:12]}.log").read_text()


def test_mac_run_refuses_existing_in_use_lock_without_replacing_it(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    target = tmp_path / "snake-dqn-run-busy.noindex"
    assert (
        run(
            ["git", "worktree", "add", "--detach", str(target), commit],
            cwd=repo,
            env=os.environ.copy(),
        ).returncode
        == 0
    )
    lock = target / ".IN_USE_LOCK"
    lock.write_text("another-owner\n")
    bin_dir = fake_mac_commands(tmp_path)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", OPS_DIR=str(tmp_path / "ops"))

    result = run(
        [str(SCRIPT), "--repo", str(repo), "busy", commit, "--", "true"], cwd=repo, env=env
    )

    assert result.returncode == 3
    assert "in use" in result.stderr
    assert lock.read_text() == "another-owner\n"


def test_mac_run_refuses_reuse_when_worktree_is_dirty_wrong_or_attached(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    target = tmp_path / "snake-dqn-run-reuse.noindex"
    assert (
        run(
            ["git", "worktree", "add", "--detach", str(target), commit],
            cwd=repo,
            env=os.environ.copy(),
        ).returncode
        == 0
    )
    bin_dir = fake_mac_commands(tmp_path)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", OPS_DIR=str(tmp_path / "ops"))

    (target / "untracked").write_text("dirty\n")
    dirty = run(
        [str(SCRIPT), "--repo", str(repo), "reuse", commit, "--", "true"], cwd=repo, env=env
    )
    assert dirty.returncode == 3
    assert "clean" in dirty.stderr
    (target / "untracked").unlink()

    (repo / "later").write_text("new commit\n")
    assert run(["git", "add", "later"], cwd=repo, env=env).returncode == 0
    assert run(["git", "commit", "-m", "later"], cwd=repo, env=env).returncode == 0
    later = run(["git", "rev-parse", "HEAD"], cwd=repo, env=env).stdout.strip()
    wrong = run([str(SCRIPT), "--repo", str(repo), "reuse", later, "--", "true"], cwd=repo, env=env)
    assert wrong.returncode == 3
    assert "pinned" in wrong.stderr

    assert run(["git", "checkout", "-b", "fixture-branch"], cwd=target, env=env).returncode == 0
    attached = run(
        [str(SCRIPT), "--repo", str(repo), "reuse", commit, "--", "true"], cwd=repo, env=env
    )
    assert attached.returncode == 3
    assert "detached" in attached.stderr


def test_mac_run_retains_its_lock_if_tmux_submission_is_uncertain(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    bin_dir = fake_mac_commands(tmp_path)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        OPS_DIR=str(tmp_path / "ops"),
        OPS_FAKE_TMUX_FAIL="1",
    )

    result = run(
        [str(SCRIPT), "--repo", str(repo), "tmuxfail", commit, "--", "true"], cwd=repo, env=env
    )

    assert result.returncode == 12
    assert (tmp_path / "snake-dqn-run-tmuxfail.noindex" / ".IN_USE_LOCK").exists()
    assert "submission uncertain" in result.stderr


def test_mac_run_signal_before_launch_cleans_lock_and_does_not_start_job(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    bin_dir = fake_mac_commands(tmp_path)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        OPS_DIR=str(tmp_path / "ops"),
        OPS_FAKE_TMUX_TERM="1",
    )

    result = run(
        [str(SCRIPT), "--repo", str(repo), "terminated", commit, "--", "true"], cwd=repo, env=env
    )

    assert result.returncode == 143
    assert not (tmp_path / "snake-dqn-run-terminated.noindex" / ".IN_USE_LOCK").exists()


def test_mac_run_signal_during_submission_retains_lock(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    bin_dir = fake_mac_commands(tmp_path)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        OPS_DIR=str(tmp_path / "ops"),
        OPS_FAKE_TMUX_TERM_SUBMIT="1",
    )
    result = run(
        [str(SCRIPT), "--repo", str(repo), "submitting", commit, "--", "true"], cwd=repo, env=env
    )
    assert result.returncode == 143
    assert (tmp_path / "snake-dqn-run-submitting.noindex" / ".IN_USE_LOCK").exists()


def test_mac_run_preserves_global_ignore_without_final_newline(tmp_path: Path) -> None:
    repo, commit = make_repo(tmp_path)
    ignore = tmp_path / "global-ignore"
    ignore.write_text("ignored-output")
    assert (
        run(
            ["git", "config", "core.excludesFile", str(ignore)], cwd=repo, env=os.environ.copy()
        ).returncode
        == 0
    )
    bin_dir = fake_mac_commands(tmp_path)
    ops = tmp_path / "ops"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", OPS_DIR=str(ops))
    env["OPS_FAKE_WATCHER"] = str(repo / "tools/ops/watchd.sh")
    result = run(
        [
            str(SCRIPT),
            "--repo",
            str(repo),
            "ignore",
            commit,
            "--",
            "sh",
            "-c",
            "touch ignored-output; test -f .IN_USE_LOCK && "
            'test -z "$(git status --porcelain)" && echo clean',
        ],
        cwd=repo,
        env=env,
    )
    assert result.returncode == 0
    assert "clean" in (ops / "logs" / f"mac-ignore-{commit[:12]}.log").read_text()
