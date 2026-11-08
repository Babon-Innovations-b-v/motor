#!/usr/bin/env bash
# Keep one ssh-agent on a fixed socket for every shell, with ~/.ssh/id_ed25519 loaded in it.
#
#   . .claude/scripts/ssh-agent.sh      a bootstrap sources it, so its own later ssh calls use the agent
#   bash .claude/scripts/ssh-agent.sh   or run it on its own
#
# An agent started with a bare `ssh-agent -s` dies with the script that started it, and no other
# shell ever has the key. So the agent lives on ~/.ssh/agent/agent.sock and a ~/.bashrc block
# points every shell at it, restarting it when its socket is dead (a reboot, a WSL restart);
# `AddKeysToAgent yes` in ~/.ssh/config loads the key again on its first use.
#
# Some networks refuse GitHub's ssh on port 22 while port 443 works with the same key. When a
# probe of port 22 fails and one of ssh.github.com:443 succeeds, GitHub's documented fallback
# (Host github.com -> ssh.github.com, port 443) is added to ~/.ssh/config, once; the probe's
# accept-new puts ssh.github.com's host key in known_hosts (a changed key is still refused).
#
# Does nothing without ~/.ssh/id_ed25519 or ssh-add. Safe to re-run and safe under
# `set -euo pipefail`: every step skips what is already done.

ssh_agent_up() {
  local key="$HOME/.ssh/id_ed25519"
  if [ ! -f "$key" ] || ! command -v ssh-add >/dev/null 2>&1; then
    echo "ssh-agent: no ~/.ssh/id_ed25519 or no ssh-add, skipped"
    return 0
  fi
  local start='export SSH_AUTH_SOCK="$HOME/.ssh/agent/agent.sock"
if ssh-add -l >/dev/null 2>&1; [ $? = 2 ]; then mkdir -p -m 700 "$HOME/.ssh/agent"; rm -f "$SSH_AUTH_SOCK"; ssh-agent -a "$SSH_AUTH_SOCK" >/dev/null; fi'
  grep -qF 'ssh/agent/agent.sock' "$HOME/.bashrc" 2>/dev/null \
    || printf '\n# harness ssh-agent: one ssh-agent for every shell\n%s\n' "$start" >> "$HOME/.bashrc"
  grep -qiF 'AddKeysToAgent' "$HOME/.ssh/config" 2>/dev/null \
    || printf 'Host *\n  AddKeysToAgent yes\n' >> "$HOME/.ssh/config"
  chmod 600 "$HOME/.ssh/config"
  eval "$start"
  local print
  print="$(ssh-keygen -lf "$key.pub" 2>/dev/null | cut -d' ' -f2 || true)"
  if [ -n "$print" ] && ssh-add -l 2>/dev/null | grep -qF "$print"; then
    echo "ssh-agent: key already loaded ($SSH_AUTH_SOCK)"
  else
    ssh-add "$key" || true
    echo "ssh-agent: key loaded ($SSH_AUTH_SOCK)"
  fi
  github_port_443
}

# GitHub answers a working key with "successfully authenticated" and exit 1 (it gives no shell),
# so the answer is read, not the exit code (a pipe into grep would fail under pipefail).
github_answers() {
  local answer
  answer="$(ssh -T -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "$@" 2>&1 || true)"
  case "$answer" in
    *"successfully authenticated"*) return 0 ;;
    *) return 1 ;;
  esac
}

github_port_443() {
  if grep -qiF 'ssh.github.com' "$HOME/.ssh/config" 2>/dev/null; then
    echo "ssh github: port 443 fallback already in ~/.ssh/config"
  elif github_answers git@github.com; then
    echo "ssh github: port 22 works"
  elif github_answers -p 443 git@ssh.github.com; then
    printf '\nHost github.com\n  HostName ssh.github.com\n  Port 443\n  User git\n' >> "$HOME/.ssh/config"
    echo "ssh github: port 22 refused, 443 works: fallback added to ~/.ssh/config"
  else
    echo "ssh github: neither port 22 nor 443 takes the key (is it on the GitHub account?)"
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
fi
ssh_agent_up
