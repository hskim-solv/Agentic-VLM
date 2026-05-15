"""Regression: .githooks/commit-msg hook (issue #826, axis #3 automation ROI).

Pins the contract: when the current branch matches ADR 0007's
`<type>/issue-<N>` form, the commit message must reference `#N` or
the hook exits 1. Exempt branches (bot / revert / dependabot) and
non-conforming branches pass through (separate gates own those).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).parents[1]
HOOK = REPO / ".githooks" / "commit-msg"
SCRIPTS_DIR = REPO / "scripts"


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


class TestCommitMsgHook(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self._repo = self._tmp / "repo"
        self._repo.mkdir()
        _git(self._repo, "init", "--quiet", "--initial-branch=main")
        # Mirror the scripts/ dir so the hook can import
        # check_branch_and_issue.py.
        shutil.copytree(SCRIPTS_DIR, self._repo / "scripts",
                        dirs_exist_ok=True)
        (self._repo / ".claude").mkdir()
        # Copy hook to a known location and chmod +x.
        shutil.copy(HOOK, self._repo / "commit-msg")
        (self._repo / "commit-msg").chmod(0o755)
        self._hook = self._repo / "commit-msg"
        self._msg = self._repo / "msg.txt"
        self._fires = self._repo / ".claude" / ".hook-fires.log"

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _branch(self, name: str) -> None:
        _git(self._repo, "checkout", "--quiet", "-B", name)

    def _run(self, message: str) -> subprocess.CompletedProcess:
        self._msg.write_text(message)
        return subprocess.run(
            ["bash", str(self._hook), str(self._msg)],
            cwd=self._repo,
            capture_output=True,
            text=True,
            check=False,
        )

    # --- Branch parsing ----------------------------------------------------

    def test_non_conforming_branch_passes_through(self) -> None:
        # ADR 0007 pre-push gate owns this case — commit-msg stays out of it.
        self._branch("zen-jackson-673ae6")
        r = self._run("any old message")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_exempt_branch_passes_through(self) -> None:
        self._branch("dependabot/npm/foo-1.2.3")
        r = self._run("chore(deps): bump foo")
        self.assertEqual(r.returncode, 0, r.stderr)

    # --- Issue-conforming branch -------------------------------------------

    def test_conforming_branch_with_issue_ref_passes(self) -> None:
        self._branch("feat/issue-999-something")
        r = self._run("feat: do thing (closes #999)")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_conforming_branch_bare_hash_ref_passes(self) -> None:
        self._branch("fix/issue-42")
        r = self._run("fix: something\n\nrelated #42\n")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_conforming_branch_no_issue_ref_blocks(self) -> None:
        self._branch("feat/issue-999-something")
        r = self._run("feat: do thing without ref")
        self.assertEqual(r.returncode, 1)
        self.assertIn("#999", r.stderr)
        # Telemetry written.
        self.assertTrue(self._fires.exists())
        self.assertIn("|blocked|commit-msg-no-issue|", self._fires.read_text())

    def test_wrong_issue_number_blocks(self) -> None:
        # Hook checks exact match — `#100` does NOT satisfy a branch
        # that references issue #1.
        self._branch("feat/issue-1-tiny")
        r = self._run("feat: tiny (closes #100)")
        self.assertEqual(r.returncode, 1)

    def test_comment_lines_do_not_satisfy(self) -> None:
        # Git scissor comments start with `#`; the hook must strip them.
        self._branch("feat/issue-7-x")
        r = self._run("feat: x\n\n# On branch feat/issue-7-x\n# Closes #7\n")
        self.assertEqual(r.returncode, 1, r.stderr)

    def test_merge_commit_passes(self) -> None:
        self._branch("feat/issue-77-merge-test")
        r = self._run("Merge branch 'foo' into bar")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
