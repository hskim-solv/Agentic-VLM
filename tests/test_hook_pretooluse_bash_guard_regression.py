"""Regression: PreToolUse Bash guard — gh pr create stacked-base check (issue #826).

Pins the new branch of the dispatcher: when the current branch's fork
point is a non-main feature branch and the `gh pr create` invocation has
no explicit `--base`, refuse with exit 2. Existing `gh pr merge
--delete-branch` guard branch is unchanged.

Tests build a tiny throwaway git repo (no network, no `gh` calls), copy
the hook in, then drive it with synthetic JSON payloads.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).parents[1]
HOOK = REPO / "scripts" / "claude-hooks" / "pretooluse-bash-guard.sh"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def _commit(cwd: Path, path: str, body: str, message: str) -> None:
    (cwd / path).write_text(body)
    _git(cwd, "add", path)
    _git(cwd, "commit", "--quiet", "-m", message)


class TestBashGuardPRCreate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self._repo = self._tmp / "repo"
        (self._repo / "scripts" / "claude-hooks").mkdir(parents=True)
        (self._repo / ".claude").mkdir()
        shutil.copy(HOOK, self._repo / "scripts" / "claude-hooks" / HOOK.name)
        self._hook = self._repo / "scripts" / "claude-hooks" / HOOK.name
        self._fires = self._repo / ".claude" / ".hook-fires.log"

        # Build: main has one commit; feature-parent forks from main with
        # one extra commit; HEAD branch forks from feature-parent with
        # one more commit (stacked situation).
        _git(self._repo, "init", "--quiet", "--initial-branch=main")
        _commit(self._repo, "a.txt", "a\n", "main: a")
        _git(self._repo, "checkout", "--quiet", "-b", "feat/issue-100-parent")
        _commit(self._repo, "b.txt", "b\n", "parent: b")
        _git(self._repo, "checkout", "--quiet", "-b", "feat/issue-101-child")
        _commit(self._repo, "c.txt", "c\n", "child: c")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, command: str, tool_name: str = "Bash") -> subprocess.CompletedProcess:
        payload = {"tool_name": tool_name, "tool_input": {"command": command}}
        return subprocess.run(
            ["bash", str(self._hook)],
            cwd=self._repo,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )

    # --- Pass-through cases ------------------------------------------------

    def test_non_gh_command_is_noop(self) -> None:
        r = self._run("ls -la")
        self.assertEqual(r.returncode, 0)
        self.assertFalse(self._fires.exists())

    def test_gh_pr_create_with_explicit_base_passes(self) -> None:
        r = self._run("gh pr create --base feat/issue-100-parent --fill")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_gh_pr_create_with_dash_b_passes(self) -> None:
        r = self._run("gh pr create -B main --fill")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_gh_pr_create_with_base_equals_passes(self) -> None:
        r = self._run("gh pr create --base=main --fill")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_gh_pr_create_from_main_branch_passes(self) -> None:
        _git(self._repo, "checkout", "--quiet", "main")
        r = self._run("gh pr create --fill")
        self.assertEqual(r.returncode, 0, r.stderr)

    # --- Block case --------------------------------------------------------

    def test_stacked_pr_create_without_base_blocks(self) -> None:
        # HEAD is feat/issue-101-child, parent is feat/issue-100-parent.
        r = self._run("gh pr create --fill")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("feat/issue-100-parent", r.stderr)
        self.assertIn("--base", r.stderr)
        self.assertTrue(self._fires.exists())
        self.assertIn("|blocked|gh-pr-create-missing-base|",
                      self._fires.read_text())

    # --- Existing merge guard regression (unchanged behavior) --------------

    def test_gh_pr_merge_without_delete_branch_passes(self) -> None:
        r = self._run("gh pr merge 123 --squash")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
