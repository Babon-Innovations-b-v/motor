#!/usr/bin/env python3
"""Reap orphaned git worktrees left behind by crashed Claude sessions.

Each Claude worktree is locked while its session is live, and the lock
reason carries the session pid. A crash leaves the lock behind, and every later session skips the
worktree as "in use by another live session" forever.

A worktree is reaped only when its lock pid is dead AND it is clean (no uncommitted changes, HEAD
already on origin/main). Dirty orphans are reported, never deleted, so no work is lost. An unlocked
worktree (a finished agent's, or one made by hand) is reaped only when it is clean and merged AND
untouched for IDLE_HOURS, because finished agents leave many of them behind: they fill the disk, and
they sit inside the repository tree, where a check that walks the tree finds files from other
branches. A dirty or unmerged unlocked one is never touched.

Every worktree git knows of is swept, wherever it sits: one made beside the repository, in the home folder or in
/tmp is still this repository's, and a misplaced one that nothing ever removed was how ~45 copies of the repositories
piled up beside them (2026-11-25). The repositories named in `.claude/project.json` under `worktree_repos` (paths
from this checkout) are swept by the same rules, for the sister repositories whose worktrees this one's sessions make.

Run standalone (`python3 .claude/scripts/worktree-reap.py`) or via the SessionStart hook. Exit
status is always 0 so it never blocks a session.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

LOCK_PID_RE = re.compile(r"pid (\d+)")
IDLE_HOURS = 6
THIS_PC_HOME = Path(os.environ.get("SCORE_THIS_PC_HOME", Path.home() / ".score-this-pc"))
PROC = Path("/proc")


def git_out(args, cwd=None):
    """Run git, returning (returncode, stripped stdout)."""
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def git_common_dir():
    """The shared .git directory, asked of git because in a linked worktree `.git` is a file."""
    code, common = git_out(["rev-parse", "--path-format=absolute", "--git-common-dir"],
                           cwd=Path(__file__).resolve().parent)
    return Path(common) if code == 0 else None


def pid_alive(pid):
    """True if a process with this pid currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists; we just can't signal it
        pass
    return True


def lock_pid(admin):
    """The pid recorded in this worktree's lock file (in its git admin folder), or None if unlocked."""
    lock_file = admin / "locked"
    if not lock_file.is_file():
        return None
    match = LOCK_PID_RE.search(lock_file.read_text())
    return int(match.group(1)) if match else None


def idle_hours(admin):
    """Hours since this worktree's HEAD or index last changed, from git's own admin files (not its reflog, which
    repository-wide housekeeping rewrites)."""
    stamps = [path.stat().st_mtime for path in (admin / "HEAD", admin / "index") if path.exists()]
    return (time.time() - max(stamps)) / 3600 if stamps else 0.0


def process_paths(pid_dir):
    """Every path one process holds: its working folder, program, root, open and mapped files, command-line words."""
    paths = []
    for link in ("cwd", "exe", "root"):
        try:
            paths.append(os.readlink(pid_dir / link))
        except OSError:  # gone, or another user's: it holds nothing we can see
            pass
    try:
        paths.extend(os.readlink(fd) for fd in (pid_dir / "fd").iterdir())
    except OSError:
        pass
    try:
        paths.extend(line.split(None, 5)[5] for line in (pid_dir / "maps").read_text().splitlines()
                     if len(line.split(None, 5)) == 6)
    except OSError:
        pass
    try:
        paths.extend((pid_dir / "cmdline").read_bytes().decode(errors="replace").split("\0"))
    except OSError:
        pass
    return paths


def roots_in(paths, roots):
    """The worktree roots that these paths fall in (any text containing such a root, then a slash, a space or its
    end, counts)."""
    if not roots:
        return set()
    pattern = re.compile("(" + "|".join(re.escape(str(root)) for root in roots) + r")(?=/|\s|$)")
    return {Path(found) for path in paths for found in pattern.findall(path)}


def held_by_processes(roots):
    """The worktree roots some live process uses (see process_paths)."""
    paths = []
    for pid_dir in PROC.glob("[0-9]*"):
        paths.extend(process_paths(pid_dir))
    return roots_in(paths, roots)


def strings_in(value):
    """Every string inside a JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings_in(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings_in(item)]
    return []


def read_json(path):
    """The JSON in path, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def this_pc_records():
    """The this-PC run records that name code to keep: every running registry entry, and the paused record or None."""
    entries = [read_json(path) for path in sorted((THIS_PC_HOME / "runs").glob("*.json"))]
    running = [entry for entry in entries if isinstance(entry, dict) and entry.get("state") == "running"]
    paused = read_json(THIS_PC_HOME / "paused.json")
    return running, paused if isinstance(paused, dict) else None


def named_by_runs(roots):
    """The worktree roots a this-PC run names by path, or whose HEAD is the paused run's commit."""
    running, paused = this_pc_records()
    records = running + ([paused] if paused else [])
    named = roots_in([text for record in records for text in strings_in(record)], roots)
    commit = (paused or {}).get("commit")
    for root in roots:
        if commit and git_out(["rev-parse", "HEAD"], cwd=root)[1] == commit:
            named.add(root)
    return named


def is_clean(worktree_path):
    """True when no uncommitted changes (untracked files included) and HEAD is on origin/main.

    `main` is the deploy interface here, so once a worktree's commits are on origin/main its branch
    is disposable. The local origin/main is used, no fetch: if it is stale, the worst case is that a
    merged worktree is kept, never that unmerged work is deleted. A missing ref keeps the worktree.
    """
    code, dirty = git_out(["status", "--porcelain"], cwd=worktree_path)
    # A virtual environment left untracked is rebuilt from the lock file, not work.
    if code != 0 or [line for line in dirty.splitlines() if line not in ("?? .venv", "?? .venv/")]:
        return False
    code, _ = git_out(["merge-base", "--is-ancestor", "HEAD", "origin/main"], cwd=worktree_path)
    return code == 0


def remove_worktree(entry, checkout):
    """Remove one orphan worktree and its auto-named branch; False when git refuses."""
    _, branch = git_out(["symbolic-ref", "--short", "HEAD"], cwd=entry)
    git_out(["worktree", "unlock", str(entry)], cwd=checkout)
    code, _ = git_out(["worktree", "remove", "--force", str(entry)], cwd=checkout)
    if code != 0:
        return False
    # Only the EnterWorktree "worktree-*" naming, so a meaningfully named branch is never touched.
    if branch.startswith("worktree-"):
        git_out(["branch", "-D", branch], cwd=checkout)
    return True


def linked_worktrees(checkout):
    """Every linked worktree of the repository at checkout that is on disk: (its root, its git admin folder)."""
    code, listing = git_out(["worktree", "list", "--porcelain"], cwd=checkout)
    if code != 0:
        return []
    roots = [Path(line[len("worktree "):]) for line in listing.splitlines() if line.startswith("worktree ")]
    found = []
    for root in roots[1:]:  # the first is the main checkout
        code, admin = git_out(["rev-parse", "--path-format=absolute", "--git-dir"], cwd=root) \
            if root.is_dir() else (1, "")
        if code == 0:
            found.append((root.resolve(), Path(admin)))
    return found


def sister_checkouts(checkout):
    """The checkouts project.json names under worktree_repos (paths from this checkout) that are git repositories."""
    config = read_json(checkout / ".claude" / "project.json")
    named = config.get("worktree_repos", []) if isinstance(config, dict) else []
    found = []
    for name in named:
        other = (checkout / name).resolve()
        code, top = git_out(["rev-parse", "--show-toplevel"], cwd=other) if other.is_dir() else (1, "")
        if code == 0 and Path(top).resolve() == other:
            found.append(other)
    return found


def reap():
    """Names of the worktrees reaped, and of the stale-locked ones kept because of their work."""
    common_dir = git_common_dir()
    if common_dir is None:
        return [], []
    checkout = common_dir.parent
    self_root = Path(__file__).resolve().parents[2]
    trees = [(checkout, root, admin) for root, admin in linked_worktrees(checkout)]
    for other in sister_checkouts(checkout):
        trees += [(other, root, admin) for root, admin in linked_worktrees(other)]
    roots = [root for _, root, _ in trees]
    in_use = held_by_processes(roots) | named_by_runs(roots)
    reaped, kept_dirty = [], []
    for owner, entry, admin in sorted(trees, key=lambda tree: tree[1].name):
        if entry == self_root or entry in in_use:
            continue
        pid = lock_pid(admin)
        if pid is None:
            if idle_hours(admin) >= IDLE_HOURS and is_clean(entry) and remove_worktree(entry, owner):
                reaped.append(entry.name)
            continue
        if pid_alive(pid):
            continue
        if is_clean(entry) and remove_worktree(entry, owner):
            reaped.append(entry.name)
        else:
            kept_dirty.append(entry.name)
    for owner in {owner for owner, _, _ in trees}:
        git_out(["worktree", "prune"], cwd=owner)
    return reaped, kept_dirty


def report(reaped, kept_dirty):
    """One line for the session; empty when there is nothing worth saying."""
    lines = []
    if reaped:
        lines.append(f"Reaped {len(reaped)} orphaned worktree(s): {', '.join(reaped)}.")
    if kept_dirty:
        lines.append(
            f"Left {len(kept_dirty)} stale-locked worktree(s) with uncommitted work: "
            f"{', '.join(kept_dirty)}. Recover or discard by hand."
        )
    return " ".join(lines)


def main():
    context = report(*reap())
    if context and os.environ.get("CLAUDE_HOOK") == "SessionStart":
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                                 "additionalContext": context}}))
    elif context:
        print(context)


if __name__ == "__main__":
    main()
