#!/usr/bin/env bash
# Claude Code PreToolUse hook for BidMate-DocAgent — Write matcher (issue #826).
#
# Registered in `.claude/settings.json` with matcher `Write|Edit|MultiEdit`.
# Fires before Claude writes a *new* ADR file (`docs/adr/NNNN-slug.md`)
# and refuses if the candidate content has no `## Verification` H2
# section.
#
# The existing `.githooks/pre-commit` lint already rejects ADR files
# without the section at commit time (issue #793). This hook catches
# the same problem one step earlier — at file-write time — so Claude
# can re-attempt with the marker in the very next turn instead of
# discovering it three commits later.
#
# Why not auto-inject the section? Claude Code PreToolUse hooks can
# block or pass but cannot rewrite tool_input. Stderr message includes
# the canonical template so the retry can paste it verbatim.
#
# Scope deliberately narrow:
#   - Only `docs/adr/NNNN-slug.md` paths (NNNN = 4 digits).
#   - Only when the target file does NOT yet exist (new ADR, not edit).
#   - For Edit / MultiEdit, the hook is a no-op — pre-commit lint
#     still catches retrofits without verification.
#
# Behavior:
#   - exit 0  : safe / not applicable / fail-open
#   - exit 2  : refuse the write, print template to stderr
#
# Hook input (stdin, JSON):
#   { "tool_name": "Write",
#     "tool_input": { "file_path": "...", "content": "..." }, ... }

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

input=$(cat)

tool_name=$(printf '%s' "$input" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get("tool_name", ""))
except Exception:
    pass' 2>/dev/null)

# Only operate on Write (new file). Edit/MultiEdit pass through —
# pre-commit lint is the safety net for retrofits.
if [[ "$tool_name" != "Write" ]]; then
  exit 0
fi

file_path=$(printf '%s' "$input" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get("tool_input", {}).get("file_path", ""))
except Exception:
    pass' 2>/dev/null)

if [[ -z "$file_path" ]]; then
  exit 0
fi

# Strip absolute-path prefix to repo-relative form for matching.
rel_path="$file_path"
if [[ "$rel_path" == "$REPO_ROOT"/* ]]; then
  rel_path="${rel_path#$REPO_ROOT/}"
fi

# Match `docs/adr/NNNN-slug.md` (case-insensitive on extension).
if ! [[ "$rel_path" =~ ^docs/adr/[0-9]{4}-[a-z0-9][a-z0-9-]*\.md$ ]]; then
  exit 0
fi

# Skip the template file itself and README.
case "$(basename "$rel_path")" in
  _template.md|README.md) exit 0 ;;
esac

# If the file already exists, this is effectively an overwrite —
# pre-commit will catch it on the next commit. Don't block here.
abs_path="$file_path"
if [[ "$abs_path" != /* ]]; then
  abs_path="$REPO_ROOT/$rel_path"
fi
if [[ -f "$abs_path" ]]; then
  exit 0
fi

content=$(printf '%s' "$input" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get("tool_input", {}).get("content", ""))
except Exception:
    pass' 2>/dev/null)

# Detect `## Verification` H2 (anywhere in the file, leading whitespace
# allowed only for the line itself). Mirrors the regex in
# scripts/_governance.py::ADR_VERIFICATION_HEADER_RE.
if printf '%s' "$content" | grep -qE '^##[[:space:]]+Verification[[:space:]]*$'; then
  exit 0
fi

{
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%s|blocked|adr-template-missing-verification|%s\n' "$ts" "$rel_path"
} >> "$REPO_ROOT/.claude/.hook-fires.log" 2>/dev/null || true

cat >&2 <<EOF
⛔ Refusing to Write new ADR \`$rel_path\` — missing \`## Verification\` section.

   The pre-commit hook rejects ADRs without this section anyway
   (issue #793). Catching it at write time saves a round-trip.

   Append this section before \`## Decision\` / \`## Consequences\`:

       ## Verification

       <!-- verifies-key: <relative-path>:<key-substring> -->

       Describe what changes if this ADR's commitment regresses. Examples:

       - \`<!-- verifies-key: reports/eval_summary.json:naive_baseline -->\`
       - \`<!-- verifies-key: tests/test_<name>_regression.py:test_<case> -->\`

   The marker is checked by \`scripts/_governance.py --lint-adr-consequences\`:
   if the target file exists, the key substring must appear in it.

   See docs/adr/_template.md for the full canonical layout.
EOF
exit 2
