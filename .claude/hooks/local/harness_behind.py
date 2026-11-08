#!/usr/bin/env python3
"""SessionStart hook: one line when the harness lock is behind score main.

Score's shared harness is copied here by `.claude/scripts/harness-sync.py`
(and an edit made here is carried back by the commit that makes it), which
records the score commit in `.claude/harness.lock`. This compares that commit with score main and, when
they differ, says so in one line with the command that catches up.

Cheap: one `git ls-remote` with a short timeout, its answer cached for an hour
(ten minutes when score could not be reached). Fails open, always: offline or
no lock means one short note or nothing, never a blocked session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / ".claude" / "harness.lock"
CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "motor" / "harness-behind.json"
FRESH_SECONDS = 3600
UNREACHABLE_SECONDS = 600
TIMEOUT_SECONDS = 6


def _remote_main(source: str) -> str | None:
    """score main's sha, from the cache when it is fresh; None when score cannot be reached."""
    try:
        cached = json.loads(CACHE.read_text())
        age = time.time() - cached["at"]
        if cached["source"] == source and age < (FRESH_SECONDS if cached["sha"] else UNREACHABLE_SECONDS):
            return cached["sha"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        listed = subprocess.run(["git", "ls-remote", source, "refs/heads/main"], capture_output=True,
                                text=True, timeout=TIMEOUT_SECONDS,
                                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        sha = listed.stdout.split()[0] if listed.returncode == 0 and listed.stdout else None
    except (OSError, subprocess.TimeoutExpired):
        sha = None
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"source": source, "sha": sha, "at": time.time()}))
    except OSError:
        pass
    return sha


def notice() -> str | None:
    """The line to show, or None when the lock is current or there is no lock."""
    try:
        lock = json.loads(LOCK.read_text())
    except (OSError, ValueError):
        return None
    remote = _remote_main(lock.get("source", ""))
    if remote is None:
        return "harness: score could not be reached, so the harness lock was not compared with score main."
    if remote == lock.get("commit"):
        return None
    return (f"harness: the lock is at score {lock.get('commit', '')[:12]}, score main is at {remote[:12]}; "
            "run `python3 .claude/scripts/harness-sync.py` in a worktree to catch up.")


def main() -> int:
    try:
        line = notice()
    except Exception:  # a courtesy notice never breaks a session
        return 0
    if line:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
