#!/usr/bin/env bash
# Get a fresh machine working in this repo.
#
#   bash .claude/scripts/bootstrap.sh
#
# Link Claude's memory folder into the repo (git-ignored: the repo is public), use the repo's git hooks, build MOTOR's
# Python from uv.lock, and report what else the loop needs. Safe to re-run: every step skips what is already done.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_MEM="$REPO/.claude/memory"

say() { printf '%s\n' "$*"; }

# Claude Code keeps memory at ~/.claude/projects/<repo path with / as ->/memory: point it at .claude/memory.
link_memory() {
  local enc target
  enc="$(printf '%s' "$REPO" | sed 's#/#-#g')"
  target="$HOME/.claude/projects/$enc/memory"
  mkdir -p "$REPO_MEM" "$(dirname "$target")"
  if [ -L "$target" ] && [ "$(readlink -f "$target")" = "$REPO_MEM" ]; then
    say "memory: already linked"
    return
  fi
  if [ -L "$target" ]; then
    rm "$target"
  elif [ -d "$target" ]; then
    cp -a "$target/." "$REPO_MEM/"
    mv "$target" "$target.bak.$$"
    say "memory: existing folder copied into the repo and saved to $target.bak.$$"
  fi
  ln -s "$REPO_MEM" "$target"
  say "memory: linked $target -> $REPO_MEM"
}

# .claude/hooks holds the shared git hooks; each runs its .claude/hooks/local/ counterpart when there is one.
use_repo_hooks() {
  git -C "$REPO" config core.hooksPath .claude/hooks
  say "git hooks: .claude/hooks"
}

build_python() {
  if ! command -v uv >/dev/null 2>&1; then
    say "uv: NOT INSTALLED. MOTOR's Python needs it: https://docs.astral.sh/uv/"
    return
  fi
  (cd "$REPO" && uv sync -q --frozen)
  say "python: .venv from uv.lock"
}

check_gh() {
  if ! command -v gh >/dev/null 2>&1; then
    say "gh: NOT INSTALLED. to-prd, triage and close go through it: https://cli.github.com"
  elif ! gh auth status >/dev/null 2>&1; then
    say "gh: installed but not logged in. Run: gh auth login"
  else
    say "gh: ready"
  fi
}

check_harness() {
  python3 "$REPO/.claude/scripts/harness-sync.py" --check --repo "$REPO" >/dev/null 2>&1 \
    && say "harness: matches its lock" \
    || say "harness: differs from its lock. Run: python3 .claude/scripts/harness-sync.py --check"
}

link_memory
use_repo_hooks
bash "$REPO/.claude/scripts/ssh-agent.sh"
build_python
check_gh
check_harness
