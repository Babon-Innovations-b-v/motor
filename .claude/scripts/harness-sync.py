#!/usr/bin/env python3
"""Keep this repository's copy of the shared harness locked to score's, both ways.

score holds the shared harness (hooks, settings, scripts and skills under
`.claude/`). This repository keeps a copy, and `.claude/harness.lock` says
which score commit it came from and the sha256 of every shared file. A shared
file is edited here like any other file: the commit that changes it carries
the change to score main in the same step and lands with a lock that names the
score commit holding it. Score's changes arrive through a sync.

    python3 .claude/scripts/harness-sync.py               copy score main in, rewrite the lock
    python3 .claude/scripts/harness-sync.py --check       exit 1 listing files that differ from the lock
    python3 .claude/scripts/harness-sync.py --staged      pre-commit: check that a staged edit can go to score
    python3 .claude/scripts/harness-sync.py --contribute M  commit-msg: commit the edit to score with message M
    python3 .claude/scripts/harness-sync.py --seal        post-commit: put the new lock into the commit
    python3 .claude/scripts/harness-sync.py --add P       start sharing a path score already has

A sync copies only the paths the lock lists; a path score no longer has is
deleted here and dropped from the lock. A locked file edited here and not yet
committed is merged with score's copy rather than overwritten. `--source`
takes a local checkout or another url instead of the one the lock names.

HOW A COMMIT CARRIES AN EDIT. git builds the commit's tree before it knows the
final message, so the work is split over three hooks. pre-commit runs
`--staged`: score must be reachable, its main must still hold exactly what the
lock says, and the edit may add no fact of this repository. commit-msg runs
`--contribute`: the edit is committed onto score main in a temporary worktree
of a local score clone, with this commit's message, score's own checks run,
and it is pushed. Any refusal up to here aborts this commit, so nothing lands
anywhere. post-commit runs `--seal`: the new lock is added to the commit just
made (a plumbing rewrite, so no hook runs twice). The score clone is
`$HARNESS_SCORE_CLONE`, else `~/projects/score` when its origin is the lock's
source, else a clone made once under `~/.cache/harness/`.

FAILURE POLICY: a sync that cannot fetch score exits 2 and changes nothing.
`--staged` fails open (exit 0) when there is no lock to read. A commit that
edits a locked file while score cannot be reached is refused, with no
override: this repository and score never drift.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LOCK = ".claude/harness.lock"
MARKER = "harness-contributed"
SYNC = "python3 .claude/scripts/harness-sync.py"
NETWORK_SECONDS = 120
CHECK_SECONDS = 600

#: Lines a contribution may not add: score names no fact of a repository that
#: locks it. Each entry is (pattern, what it is).
DENY = (
    (re.compile(r"babon", re.IGNORECASE), "a company name (the shared files name no business)"),
    (re.compile(r"Babon-Innovations-b-v/motor|\bMyoFullBody\b|\bLabValidation\b"), "a fact of this repository"),
    (re.compile(r"\bfr-par\b|scw\.cloud|\bscw\s"), "a cloud location or id"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "a uuid (an account or project id)"),
    (re.compile(r"\bJoey\b"), "a person's name (only the owner handle is allowed)"),
    (re.compile(r"(?<!Joey)Kardolus"), "a person's name (only the owner handle is allowed)"),
    (re.compile(r"^(?:<{7}|>{7})(?: |$)"), "an unresolved merge marker"),
)
#: This repository's own references, dropped from a message before it goes to score.
ISSUE_REF = re.compile(r"\s*\((?:#\d+[,\s]*)+\)|#\d+")
TRAILER_DROPPED = re.compile(r"^(?:Worktree|Closes|Fixes|Resolves|Refs?):", re.IGNORECASE)
#: Git variables that would point a child git at the repository being committed to.
KEEP_GIT = ("GIT_AUTHOR_", "GIT_COMMITTER_", "GIT_SSH", "GIT_ASKPASS")


class _Refused(Exception):
    """A contribution that must not go ahead; the message says why and what to do."""


def _digest(content: bytes | None) -> str | None:
    """sha256 of a file's bytes, or None for a file that is not there."""
    return None if content is None else hashlib.sha256(content).hexdigest()


def _differing(locked: dict[str, str], read) -> list[str]:
    """Locked paths whose content, as `read(path)` returns it, does not hash to the lock."""
    return sorted(path for path, expected in locked.items() if _digest(read(path)) != expected)


def _git(repo: Path, *args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=check, **kwargs)


def _score_env() -> dict[str, str]:
    """The environment for a git aimed at score: none of this commit's GIT_DIR or index."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_") or key.startswith(KEEP_GIT)}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=15")
    return env


def _score_git(repo: Path, *args: str, check: bool = True,
               timeout: int = NETWORK_SECONDS) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=check,
                          env=_score_env(), timeout=timeout)


def _read_worktree(repo: Path):
    def read(path: str) -> bytes | None:
        file = repo / path
        return file.read_bytes() if file.is_file() else None
    return read


def _read_blob(repo: Path, rev: str, git=_git):
    """Reader for a tree-ish (`HEAD`) or the index (rev ``""``)."""
    def read(path: str) -> bytes | None:
        shown = git(repo, "show", f"{rev}:{path}", check=False)
        return shown.stdout if shown.returncode == 0 else None
    return read


def _load_lock(content: bytes | None) -> dict:
    return json.loads(content) if content else {}


def _lock_text(lock: dict) -> str:
    return json.dumps(lock, indent=2, sort_keys=True) + "\n"


def _check(repo: Path) -> int:
    """Exit 1 listing every locked file in the working tree that differs from the lock."""
    lock = _load_lock(_read_worktree(repo)(LOCK))
    if not lock:
        print(f"no {LOCK} here, nothing to check", file=sys.stderr)
        return 2
    drifted = _differing(lock["files"], _read_worktree(repo))
    for path in drifted:
        print(f"differs from the lock: {path}")
    if drifted:
        print(f"\ncommit them (the commit carries them to score), or take score's copy: {SYNC}",
              file=sys.stderr)
    return 1 if drifted else 0


# ── Where score is ─────────────────────────────────────────────────────────


def _slug(url: str) -> str | None:
    """`owner/name` of a GitHub url (https or ssh), else None."""
    found = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url or "")
    return found.group(1).lower() if found else None


def _push_url(source: str) -> str:
    """The url a fresh clone is made from: ssh for GitHub, so it can push."""
    slug = _slug(source)
    return f"git@github.com:{slug}.git" if slug else source


def _same_repository(clone: Path, source: str) -> bool:
    origin = _score_git(clone, "remote", "get-url", "origin", check=False, timeout=10)
    url = origin.stdout.decode().strip()
    if origin.returncode != 0:
        return False
    return (_slug(url) or url) == (_slug(source) or source)


def _score_clone(source: str) -> Path:
    """A local clone of score whose origin is `source`; cloned once when there is none."""
    override = os.environ.get("HARNESS_SCORE_CLONE")
    if override:
        clone = Path(override)
    else:
        projects = Path.home() / "projects" / "score"
        if (projects / ".git").exists() and _same_repository(projects, source):
            return projects
        clone = Path.home() / ".cache" / "harness" / hashlib.sha256(source.encode()).hexdigest()[:12]
    if (clone / ".git").exists():
        return clone
    clone.parent.mkdir(parents=True, exist_ok=True)
    try:
        _score_git(clone.parent, "clone", "-q", "--filter=blob:none", _push_url(source), str(clone))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as failed:
        shutil.rmtree(clone, ignore_errors=True)
        raise _Refused(_offline(failed)) from failed
    # A clone made here runs score's own git hooks when it commits.
    _score_git(clone, "config", "core.hooksPath", ".claude/hooks", check=False, timeout=10)
    return clone


def _offline(failed: Exception) -> str:
    why = getattr(failed, "stderr", b"") or b""
    why = why.decode(errors="replace").strip().splitlines()[-1:] if isinstance(why, bytes) else [str(why)]
    return ("score cannot be reached" + (f" ({why[0]})" if why else "") + ".\n"
            "A commit that changes a shared harness file lands in score and here together, or in "
            "neither. Commit again when online; there is no override.")


def _score_main(clone: Path) -> str:
    """Fetch score main into `clone`; its sha."""
    try:
        _score_git(clone, "fetch", "-q", "origin", "main")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as failed:
        raise _Refused(_offline(failed)) from failed
    return _score_git(clone, "rev-parse", "FETCH_HEAD", timeout=10).stdout.decode().strip()


# ── What a commit carries ──────────────────────────────────────────────────


def _added_lines(old: bytes | None, new: bytes | None) -> list[str]:
    before = (old or b"").decode(errors="replace").splitlines()
    after = (new or b"").decode(errors="replace").splitlines()
    return [line[1:] for line in difflib.unified_diff(before, after, lineterm="", n=0)
            if line.startswith("+") and not line.startswith("+++")]


def _leaks(lines: list[str]) -> list[str]:
    """One finding per line that names a fact score must not hold."""
    return [f"{what}: {line.strip()[:120]}" for line in lines for pattern, what in DENY
            if pattern.search(line)]


def _plan(repo: Path, lock: dict, read_new) -> tuple[Path, str, list[str]]:
    """The clone, score main's sha and the paths to carry, or _Refused with the reason."""
    touched = _differing(lock["files"], read_new)
    clone = _score_clone(lock.get("source", ""))
    sha = _score_main(clone)
    read_score = _read_blob(clone, sha, git=_score_git)
    moved = _differing(lock["files"], read_score)
    if moved:
        raise _Refused(
            "score main has changed shared files since this lock:\n"
            + "".join(f"  {path}\n" for path in moved)
            + f"Take score's latest first (your edits are merged, not overwritten):\n  {SYNC}\n"
            "then stage the result and commit again.")
    found = [f"{path}: {finding}" for path in touched
             for finding in _leaks(_added_lines(read_score(path), read_new(path)))]
    if found:
        raise _Refused("score holds no fact of this repository, and this edit adds some:\n"
                      + "".join(f"  {line}\n" for line in found)
                      + "Write it generically, or keep it in a file this repository does not share.")
    return clone, sha, touched


def _scrub(message: str) -> str:
    """This commit's message as score gets it: no comment, trailer or issue of this repository."""
    lines = [line for line in message.splitlines()
             if not line.startswith("#") and not TRAILER_DROPPED.match(line)]
    text = ISSUE_REF.sub("", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _staged(repo: Path) -> int:
    """Exit 1 when a commit's change to locked files cannot land in score and here together."""
    before_lock = _load_lock(_read_blob(repo, "HEAD")(LOCK))
    after_lock = _load_lock(_read_blob(repo, "")(LOCK))
    before = before_lock.get("files", {})
    after = after_lock.get("files", {})
    changed = set(_git(repo, "diff", "--cached", "--name-only", "--no-renames", "-z")
                  .stdout.decode().split("\0")) - {""}
    touched = sorted(changed & (set(before) | set(after)))
    if not touched:
        return 0
    if LOCK not in changed:
        return _preflight(repo, before_lock)
    wrong = [path for path in touched if path not in after]
    wrong += _differing({path: after[path] for path in touched if path in after},
                       _read_blob(repo, ""))
    # A sync from the same score commit writes the same bytes, so a locked
    # file whose hash moved while the score commit did not was edited by hand.
    if after_lock.get("commit") == before_lock.get("commit"):
        wrong += [path for path in touched if path in before and before[path] != after.get(path)]
    if not wrong:
        return 0
    print("BLOCKED: this commit changes the harness lock and shared files it does not match:",
          file=sys.stderr)
    for path in sorted(set(wrong)):
        print(f"  {path}", file=sys.stderr)
    print("\nThe lock is written by harness-sync, never by hand. Unstage it and commit the edit "
          f"alone (the commit carries it to score), or take score's copy:\n  {SYNC}", file=sys.stderr)
    return 1


def _preflight(repo: Path, lock: dict) -> int:
    try:
        _clone, _sha, touched = _plan(repo, lock, _read_blob(repo, ""))
    except _Refused as refused:
        print(f"BLOCKED: this commit edits the shared harness and cannot carry it to score.\n{refused}",
              file=sys.stderr)
        return 1
    print(f"harness: {len(touched)} shared file(s) go to score main with this commit's message",
          file=sys.stderr)
    return 0


def _score_checks(worktree: Path, clone: Path, touched: list[str]) -> list[list[str]]:
    """score's own checks for the files carried: its test rule, and its hook tests when hooks change."""
    checks = []
    if (worktree / "tools" / "test_rule" / "check.py").is_file():
        checks.append([sys.executable, "tools/test_rule/check.py"])
    if any(path.startswith(".claude/hooks/") for path in touched) and (worktree / ".claude/hooks/tests").is_dir():
        venv = clone / ".venv" / "bin" / "python"
        python = str(venv) if venv.is_file() else sys.executable
        checks.append([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", ".claude/hooks/tests"])
    return checks


def _run_checks(worktree: Path, checks: list[list[str]]) -> dict[str, tuple[bool, str]]:
    """Each check's result in `worktree`: passed, and the tail of its output."""
    results = {}
    with tempfile.TemporaryDirectory() as home:
        env = {**_score_env(), "PROPS_HOME": home}
        for command in checks:
            ran = subprocess.run(command, cwd=worktree, capture_output=True, text=True, env=env,
                                 timeout=CHECK_SECONDS)
            tail = "\n".join((ran.stdout + ran.stderr).strip().splitlines()[-15:])
            results[" ".join(command[1:])] = (ran.returncode == 0, tail)
    return results


def _refuse_new_failures(before: dict[str, tuple[bool, str]], after: dict[str, tuple[bool, str]]) -> None:
    """Refuse only a check this edit turned red: score's own red on main is not this commit's to carry."""
    for name, (passed, tail) in after.items():
        if not passed and before.get(name, (True, ""))[0]:
            raise _Refused(f"score's check `{name}` failed on this edit:\n{tail}")


def _commit_to_score(repo: Path, clone: Path, sha: str, touched: list[str], message: str) -> str:
    """Commit `touched`, as this commit's index holds them, onto score `sha` and push; the new sha."""
    read_new = _read_blob(repo, "")
    with tempfile.TemporaryDirectory(prefix="harness-contribute-") as tmp:
        worktree = Path(tmp) / "score"
        _score_git(clone, "worktree", "add", "-q", "--detach", str(worktree), sha, timeout=60)
        try:
            checks = _score_checks(worktree, clone, touched)
            before = _run_checks(worktree, checks)
            for path in touched:
                content, target = read_new(path), worktree / path
                if content is None:
                    _score_git(worktree, "rm", "-q", "--", path, timeout=30)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                mode = _git(repo, "ls-files", "-s", "--", path).stdout.decode().split()[:1]
                os.chmod(target, 0o755 if mode == ["100755"] else 0o644)
                _score_git(worktree, "add", "--", path, timeout=30)
            committed = subprocess.run(["git", "-C", str(worktree), "commit", "-q", "-F", "-"],
                                       input=message.encode(), capture_output=True, env=_score_env(),
                                       timeout=CHECK_SECONDS)
            if committed.returncode != 0:
                raise _Refused("score's commit hooks refused this edit:\n"
                              + (committed.stdout + committed.stderr).decode(errors="replace").strip())
            _refuse_new_failures(before, _run_checks(worktree, checks))
            new = _score_git(worktree, "rev-parse", "HEAD", timeout=10).stdout.decode().strip()
            try:
                _score_git(worktree, "push", "-q", "origin", "HEAD:main")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as failed:
                why = (getattr(failed, "stderr", b"") or b"").decode(errors="replace").strip()
                raise _Refused(f"score refused the push ({why or 'timed out'}). score main may have "
                              "moved: commit again, and run the sync first if it says so.") from failed
            return new
        finally:
            _score_git(clone, "worktree", "remove", "--force", str(worktree), check=False, timeout=60)


def _contribute(repo: Path, message_file: Path) -> int:
    """commit-msg: carry the commit's edit of locked files to score main; exit 1 aborts the commit."""
    head_lock = _load_lock(_read_blob(repo, "HEAD")(LOCK))
    index_lock = _load_lock(_read_blob(repo, "")(LOCK))
    if not head_lock or head_lock != index_lock:
        return 0  # no lock, or a sync commit: --staged already judged it
    if not _differing(head_lock["files"], _read_blob(repo, "")):
        return 0  # the commit leaves every shared file alone: no network
    try:
        clone, sha, touched = _plan(repo, head_lock, _read_blob(repo, ""))
        message = _scrub(message_file.read_text())
        found = _leaks(message.splitlines())
        if found:
            raise _Refused("this commit's message names facts score must not hold:\n"
                          + "".join(f"  {line}\n" for line in found) + "Reword it and commit again.")
        new = _commit_to_score(repo, clone, sha, touched, message)
    except _Refused as refused:
        print(f"BLOCKED: the shared harness edit did not reach score, so this commit did not land "
              f"either.\n{refused}", file=sys.stderr)
        return 1
    read_new = _read_blob(repo, "")
    files = dict(head_lock["files"])
    for path in touched:
        content = read_new(path)
        if content is None:
            files.pop(path, None)
        else:
            files[path] = _digest(content)
    lock = {**head_lock, "commit": new, "files": files}
    marker = Path(_git(repo, "rev-parse", "--git-path", MARKER).stdout.decode().strip())
    if not marker.is_absolute():
        marker = repo / marker
    marker.write_text(_lock_text(lock))
    (repo / LOCK).write_text(_lock_text(lock))
    print(f"harness: carried to score main {new[:12]}: {', '.join(touched)}", file=sys.stderr)
    return 0


def _seal(repo: Path) -> int:
    """post-commit: add the lock --contribute wrote to the commit just made."""
    marker = Path(_git(repo, "rev-parse", "--git-path", MARKER).stdout.decode().strip())
    if not marker.is_absolute():
        marker = repo / marker
    if not marker.is_file():
        return 0
    text = marker.read_text()
    marker.unlink()
    lock = json.loads(text)
    head_files = _read_blob(repo, "HEAD")
    if _differing(lock["files"], head_files):
        print(f"harness: the commit does not hold what was carried to score; {LOCK} is left "
              "in the working tree for the next commit", file=sys.stderr)
        return 1
    blob = _git(repo, "hash-object", "-w", "--stdin", input=text.encode()).stdout.decode().strip()
    head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        subprocess.run(["git", "-C", str(repo), "read-tree", head], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "update-index", "--add", "--cacheinfo",
                        f"100644,{blob},{LOCK}"], check=True, env=env)
        tree = subprocess.run(["git", "-C", str(repo), "write-tree"], check=True, env=env,
                              capture_output=True).stdout.decode().strip()
    raw = _git(repo, "cat-file", "commit", head).stdout.decode()
    message = raw.split("\n\n", 1)[1]
    fields = _git(repo, "log", "-1", "--date=raw",
                  "--format=%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd%x00%P", head).stdout.decode()
    names = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE",
             "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE")
    values = fields.rstrip("\n").split("\0")
    env = {**os.environ, **dict(zip(names, values[:6]))}
    parents = [arg for parent in values[6].split() for arg in ("-p", parent)]
    sealed = subprocess.run(["git", "-C", str(repo), "commit-tree", tree, *parents],
                            input=message.encode(), capture_output=True, check=True,
                            env=env).stdout.decode().strip()
    _git(repo, "update-ref", "-m", f"harness-sync: lock at score {lock['commit'][:12]}",
         "HEAD", sealed, head)
    (repo / LOCK).write_text(text)
    _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{LOCK}")
    print(f"harness: the lock at score {lock['commit'][:12]} is in the commit, now {sealed[:12]}",
          file=sys.stderr)
    return 0


# ── Sync ───────────────────────────────────────────────────────────────────


def _fetch(source: str, into: Path) -> str:
    """Shallow, blob-less fetch of `source` main into `into`; returns the commit sha."""
    _git(into, "init", "-q")
    _git(into, "fetch", "-q", "--depth", "1", "--filter=blob:none", source, "main")
    return _git(into, "rev-parse", "FETCH_HEAD").stdout.decode().strip()


def _executable(clone: Path, commit: str, path: str) -> bool:
    entry = _git(clone, "ls-tree", commit, "--", path).stdout.decode().split()
    return bool(entry) and entry[0] == "100755"


def _merge3(yours: bytes, base: bytes, theirs: bytes) -> tuple[bytes, bool]:
    """git merge-file of the three; the result and whether it merged without conflict."""
    with tempfile.TemporaryDirectory() as tmp:
        names = []
        for name, content in (("yours", yours), ("lock", base), ("score", theirs)):
            (Path(tmp) / name).write_bytes(content)
            names.append(str(Path(tmp) / name))
        merged = subprocess.run(["git", "merge-file", "-p", "-L", "yours", "-L", "lock", "-L", "score",
                                 *names], capture_output=True)
    return merged.stdout, merged.returncode == 0


def _unmergeable(repo: Path, path: str, content: bytes | None, locked: str | None) -> bool:
    """True when `path` has a local edit, score's copy differs, and HEAD holds no base to merge from."""
    target = repo / path
    local = target.read_bytes() if target.is_file() else None
    if local is None or local == content or _digest(local) == locked:
        return False
    return content is None or _digest(_read_blob(repo, "HEAD")(path)) != locked


def _take(repo: Path, path: str, content: bytes, locked: str | None) -> str | None:
    """Write score's `content` for `path`, merging an uncommitted local edit; a conflict note or None."""
    target = repo / path
    local = target.read_bytes() if target.is_file() else None
    if local is not None and local != content and _digest(local) != locked:
        merged, clean = _merge3(local, _read_blob(repo, "HEAD")(path), content)
        target.write_bytes(merged)
        print(f"merged your edit with score's {path}")
        return None if clean else f"{path}: conflict markers to resolve"
    if local != content:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f"synced {path}")
    return None


def _sync(repo: Path, source: str | None, adding: list[str]) -> int:
    """Copy every locked path from score main into `repo` and rewrite the lock."""
    lock = _load_lock(_read_worktree(repo)(LOCK))
    source = source or lock.get("source")
    if not source:
        print("no source: give --source", file=sys.stderr)
        return 2
    locked = lock.get("files", {})
    paths = sorted(set(locked) | set(adding))
    conflicts = []
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp)
        try:
            commit = _fetch(source, clone)
        except subprocess.CalledProcessError as failed:
            print(f"could not fetch {source}: {failed.stderr.decode().strip()}", file=sys.stderr)
            return 2
        read = _read_blob(clone, commit)
        stuck = [path for path in paths if _unmergeable(repo, path, read(path), locked.get(path))]
        if stuck:
            print("these shared files have edits here that cannot be merged with score's copy "
                  "(no committed base); `git stash`, sync again, then `git stash pop`. Nothing changed:",
                  file=sys.stderr)
            for path in stuck:
                print(f"  {path}", file=sys.stderr)
            return 2
        files: dict[str, str] = {}
        for path in paths:
            content = read(path)
            if content is None:
                print(f"score no longer has {path}: removed here", file=sys.stderr)
                (repo / path).unlink(missing_ok=True)
                continue
            note = _take(repo, path, content, locked.get(path))
            if note:
                conflicts.append(note)
            os.chmod(repo / path, 0o755 if _executable(clone, commit, path) else 0o644)
            files[path] = _digest(content)
    # A --source override (a local checkout, say) never replaces the url the lock names.
    lock = {"source": lock.get("source") or source, "commit": commit, "files": files}
    (repo / LOCK).write_text(_lock_text(lock))
    print(f"lock at score {commit[:12]}, {len(files)} files")
    for note in conflicts:
        print(f"  {note}", file=sys.stderr)
    return 1 if conflicts else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=None, help="repository root (default: this checkout)")
    parser.add_argument("--check", action="store_true", help="list files that differ from the lock")
    parser.add_argument("--staged", action="store_true", help="pre-commit check of the index")
    parser.add_argument("--contribute", type=Path, metavar="MESSAGE_FILE",
                        help="commit-msg: carry the edit to score")
    parser.add_argument("--seal", action="store_true", help="post-commit: put the new lock in the commit")
    parser.add_argument("--source", help="score checkout or url (default: the lock's)")
    parser.add_argument("--add", nargs="+", default=[], metavar="PATH", help="paths to start sharing")
    args = parser.parse_args(argv)
    repo = args.repo or Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.decode().strip())
    if args.check:
        return _check(repo)
    if args.staged:
        return _staged(repo)
    if args.contribute:
        return _contribute(repo, args.contribute)
    if args.seal:
        return _seal(repo)
    return _sync(repo, args.source, args.add)


if __name__ == "__main__":
    sys.exit(main())
