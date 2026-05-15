#!/usr/bin/env bash
# Claude Code PreToolUse hook for BidMate-DocAgent — Bash matcher.
#
# Registered in `.claude/settings.json` with matcher `Bash`. Fires before
# Claude runs any Bash command. Two guards in this single dispatcher:
#
#   1. `gh pr merge --delete-branch` with open stacked dependents
#      (auto-enforces CLAUDE.md `## Prohibited` after PR #423 → #431 and
#      PR #470 stacked-PR auto-close incidents).
#   2. `gh pr create` without `--base` when the current branch's fork
#      point is a non-main feature branch (issue #826 — caught after
#      repeated PR-A0/A1/A2/A3 manual base re-edits).
#
# Behavior:
#   - exit 0  : safe / not applicable / fail-open
#   - exit 2  : refuse the command, print rationale to stderr
#
# Fail-open philosophy: a buggy hook silently letting one bad merge
# through is recoverable (re-open the dependent PR — see #423→#431).
# A buggy hook silently blocking every Bash command is not.
#
# Hook input (stdin, JSON):
#   { "tool_name": "Bash",
#     "tool_input": { "command": "..." }, ... }

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

input=$(cat)

cmd=$(printf '%s' "$input" | python3 -c 'import json,sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get("tool_input", {}).get("command", ""))
except Exception:
    pass' 2>/dev/null)

# Fast path: empty command.
if [[ -z "$cmd" ]]; then
  exit 0
fi

# Identify which gh pr subcommand (if any) the command runs. Honors
# command chaining (`;`, `&&`, `||`, `|`, `&`, newline) and parses each
# segment with shlex so quoted args don't fool us.
gh_subcmd=$(printf '%s' "$cmd" | python3 -c '
import sys, shlex, re
for part in re.split(r"[;&|\n]", sys.stdin.read()):
    part = part.strip().lstrip("(")
    try:
        tokens = shlex.split(part)
    except ValueError:
        continue
    if len(tokens) >= 3 and tokens[0] == "gh" and tokens[1] == "pr" and tokens[2] in ("merge", "create"):
        print(tokens[2]); break
' 2>/dev/null)

case "$gh_subcmd" in
  merge)
    if ! grep -qE -- '--delete-branch' <<<"$cmd"; then
      exit 0
    fi
    ;;
  create)
    # `gh pr create` guard. Skip if user already specified `--base <X>`
    # (any of: `--base main`, `--base=main`, `-B main`).
    if grep -qE -- '(--base([= ])|[^[:alnum:]_-]-B[ ])' <<<"$cmd"; then
      exit 0
    fi

    head_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
    if [[ -z "$head_branch" || "$head_branch" == "HEAD" ]]; then
      exit 0  # detached HEAD — fail-open
    fi

    # Resolve main reference (origin/main preferred, fall back to main).
    main_ref="origin/main"
    if ! git rev-parse --verify --quiet "$main_ref" >/dev/null 2>&1; then
      main_ref="main"
      if ! git rev-parse --verify --quiet "$main_ref" >/dev/null 2>&1; then
        exit 0  # no main ref — fail-open
      fi
    fi

    # Find feature branches (excluding main, HEAD, current) that contain
    # HEAD's parent — i.e. branches HEAD was forked from. If any such
    # branch is a closer ancestor than main, this is a stacked PR.
    # Heuristic: a non-main local/remote branch that is an ancestor of
    # HEAD AND not an ancestor of main.
    candidates=$(git for-each-ref --format='%(refname:short)' \
                    refs/heads refs/remotes/origin 2>/dev/null \
                  | grep -vE '^(origin/HEAD|origin/main|main)$' \
                  | grep -v "^${head_branch}$" \
                  | grep -v "^origin/${head_branch}$" \
                  || true)

    stacked_parent=""
    while IFS= read -r ref; do
      [[ -z "$ref" ]] && continue
      # ref must be ancestor of HEAD AND not ancestor of main_ref AND
      # not equal to HEAD.
      if git merge-base --is-ancestor "$ref" HEAD 2>/dev/null \
         && ! git merge-base --is-ancestor "$ref" "$main_ref" 2>/dev/null \
         && [[ "$(git rev-parse "$ref" 2>/dev/null)" != "$(git rev-parse HEAD 2>/dev/null)" ]]; then
        stacked_parent="$ref"
        break
      fi
    done <<< "$candidates"

    if [[ -z "$stacked_parent" ]]; then
      exit 0  # fork point is main — `gh pr create` defaults are correct
    fi

    {
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      printf '%s|blocked|gh-pr-create-missing-base|%s\n' "$ts" "$head_branch"
    } >> "$REPO_ROOT/.claude/.hook-fires.log" 2>/dev/null || true

    cat >&2 <<EOF
⛔ Refusing \`gh pr create\` without \`--base\`: \`$head_branch\` looks
   stacked on \`$stacked_parent\` (which is not main).

   Without an explicit \`--base\`, gh would target main and the diff
   would include $stacked_parent's commits too — confusing reviewers
   and inflating CI cost. Re-issue with one of:

       gh pr create --base $stacked_parent  ...   # keep the stack
       gh pr create --base main             ...   # collapse to main

   To override (e.g. \`$stacked_parent\` was just rebased onto main and
   the heuristic is wrong), pass \`--base main\` explicitly.
EOF
    exit 2
    ;;
  *)
    exit 0
    ;;
esac

# Resolve the head branch whose PR is being merged.
#   `gh pr merge <N>` → look up PR N's head branch
#   `gh pr merge`     → current branch is the implicit target
head_branch=""
pr_number=$(grep -oE 'gh[[:space:]]+pr[[:space:]]+merge[[:space:]]+([0-9]+)' <<<"$cmd" \
            | grep -oE '[0-9]+$' || true)

if [[ -n "$pr_number" ]]; then
  head_branch=$(gh pr view "$pr_number" --json headRefName --jq .headRefName 2>/dev/null || true)
else
  head_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
fi

if [[ -z "$head_branch" ]]; then
  # Could not resolve — fail-open with a soft warning.
  cat >&2 <<EOF
⚠️  Bash guard: could not resolve head branch for \`gh pr merge --delete-branch\`.
    Skipping stacked-dependent audit. Verify manually:
        gh pr list --base <branch> --state open
EOF
  exit 0
fi

# Query open PRs targeting head_branch as base.
dependents=$(gh pr list --base "$head_branch" --state open \
               --json number,title,headRefName 2>/dev/null || true)

if [[ -z "$dependents" || "$dependents" == "[]" ]]; then
  # No open dependents — `--delete-branch` is safe.
  exit 0
fi

# Render the dependent list and refuse.
listing=$(printf '%s' "$dependents" \
            | python3 -c 'import json,sys
try:
    for p in json.loads(sys.stdin.read()):
        print(f"      PR #{p[\"number\"]} — {p[\"title\"]} (head: {p[\"headRefName\"]})")
except Exception:
    pass' 2>/dev/null)

printf '%s|blocked|gh-merge-delete-branch|%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$head_branch" \
  >> "$REPO_ROOT/.claude/.hook-fires.log" 2>/dev/null || true

cat >&2 <<EOF
⛔ Refusing \`gh pr merge --delete-branch\`: stacked dependents exist on \`$head_branch\`.

$listing

    Two recovery options:
      (a) Drop \`--delete-branch\` from the merge command (dependents survive,
          the base branch lingers — fine for a short-lived stack).
      (b) Rebase each dependent onto main first, then re-run:
              gh pr edit <M> --base main
              gh pr edit <K> --base main

    Policy: CLAUDE.md \`## Prohibited\` — verify
            \`gh pr list --base $head_branch --state open\` is empty
            before \`--delete-branch\`.
    Precedent: PR #423 → #431 recovery after the stacked dependent was
               auto-closed by this exact pattern.
EOF
exit 2
