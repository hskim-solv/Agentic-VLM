#!/usr/bin/env python3
"""Single source of truth for the load-bearing path list (CLAUDE.md).

A "load-bearing" path is one whose change requires PR template item 5b
(real-data eval delta) per CLAUDE.md and the PR #69 lesson — synthetic
CI delta alone missed an intended-abstention regression there.

Three call sites that previously hardcoded their own copy now read this
module:

- `.githooks/pre-push` (soft-warn reminder)
- `scripts/claude-hooks/pretooluse-loadbearing.sh` (Claude awareness)
- `.github/workflows/branch-and-issue-check.yml` via
  `scripts/check_branch_and_issue.py --check-5b` (hard-fail CI gate)

Exit codes:
    0  match (CLI succeeded — for --is-load-bearing / --any-match
       this means "path is load-bearing")
    1  no match
    2  internal / usage error
"""

from __future__ import annotations

import argparse
import sys


# Canonical load-bearing path list. The order is not significant; add
# new entries here and the three consumers above pick them up
# automatically. Entries ending in "/" are treated as directories
# (prefix match); others as files (exact name or "/<name>" suffix match).
LOAD_BEARING_PATHS: list[str] = [
    "rag_core.py",
    "rag_retrieval.py",
    "rag_verifier.py",
    "rag_answer.py",
    "rag_query.py",
    "ingestion.py",
    "visual_ingestion.py",
    "eval/",
    "api/",
    "docs/adr/",
    "scripts/build_index.py",
]


# Numeric thresholds shared across the governance surface. Single source
# of truth for hook scripts, the self-review collector, and any future
# CI gate. SKILL.md (`.claude/skills/self-review-quarterly/SKILL.md`) is
# the parallel SoT for *grading bands* (✓/△/✗); this dict only holds the
# *raw values* that bash hooks + python collectors need to agree on.
THRESHOLDS: dict[str, int] = {
    # PR #747 PreToolUse MEMORY.md line-count matcher
    "MEMORY_LINE_AWARE": 20,
    "MEMORY_LINE_BLOCK": 30,
    # PR #745 axis #2 (Agent delegation) non-trivial-PR LOC cut-off
    "AXIS_2_LOC": 50,
}


def _normalize(path: str) -> str:
    if path.startswith("./"):
        return path[2:]
    return path


def is_load_bearing(path: str) -> bool:
    """Return True if `path` matches any canonical load-bearing entry.

    Accepts both repo-relative paths (`rag_core.py`, `eval/config.yaml`)
    and absolute paths (`/Users/.../rag_core.py`). Matching is anchored
    so that `myapi/main.py` does NOT match the `api/` directory entry.
    """
    if not path:
        return False
    p = _normalize(path)
    for entry in LOAD_BEARING_PATHS:
        if entry.endswith("/"):
            stripped = entry.rstrip("/")
            if p == stripped or p.startswith(entry) or f"/{entry}" in p:
                return True
        else:
            if p == entry or p.endswith("/" + entry):
                return True
    return False


def _cmd_is_load_bearing(path: str) -> int:
    return 0 if is_load_bearing(path) else 1


def _cmd_any_match() -> int:
    first_hit: str | None = None
    for line in sys.stdin:
        candidate = line.strip()
        if not candidate:
            continue
        if is_load_bearing(candidate):
            first_hit = candidate
            break
    if first_hit is None:
        return 1
    sys.stdout.write(first_hit + "\n")
    return 0


def _cmd_list() -> int:
    sys.stdout.write("\n".join(LOAD_BEARING_PATHS) + "\n")
    return 0


def _cmd_threshold(key: str) -> int:
    val = THRESHOLDS.get(key)
    if val is None:
        sys.stderr.write(
            f"unknown threshold key: {key!r}; available: "
            f"{sorted(THRESHOLDS)}\n"
        )
        return 1
    sys.stdout.write(f"{val}\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Load-bearing paths + numeric thresholds SSoT "
                    "(CLAUDE.md, PR #69 / #747 / #745 lessons).",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--is-load-bearing", metavar="PATH",
        help="Exit 0 if PATH is load-bearing, 1 otherwise.",
    )
    g.add_argument(
        "--any-match", action="store_true",
        help="Read newline-delimited paths from stdin; exit 0 if any "
             "match (printing the first match to stdout), else 1.",
    )
    g.add_argument(
        "--list", action="store_true",
        help="Print the canonical load-bearing list, one per line.",
    )
    g.add_argument(
        "--threshold", metavar="KEY",
        help="Print the numeric THRESHOLDS[KEY] to stdout (exit 0). "
             "Exit 1 if KEY is unknown.",
    )
    args = p.parse_args()

    if args.is_load_bearing is not None:
        return _cmd_is_load_bearing(args.is_load_bearing)
    if args.any_match:
        return _cmd_any_match()
    if args.list:
        return _cmd_list()
    if args.threshold is not None:
        return _cmd_threshold(args.threshold)
    return 2


if __name__ == "__main__":
    sys.exit(main())
