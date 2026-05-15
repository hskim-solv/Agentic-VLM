"""Regression: PreToolUse Write — ADR Verification boilerplate guard (issue #826).

Pins: new `docs/adr/NNNN-slug.md` Write payloads without `## Verification`
H2 are refused (exit 2 + boilerplate to stderr). Existing files, non-ADR
paths, Edit/MultiEdit tools, and ADRs that already include the section
all pass through.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).parents[1]
HOOK = REPO / "scripts" / "claude-hooks" / "pretooluse-adr-template.sh"


class TestADRTemplateHook(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self._repo = self._tmp / "repo"
        (self._repo / "scripts" / "claude-hooks").mkdir(parents=True)
        (self._repo / "docs" / "adr").mkdir(parents=True)
        (self._repo / ".claude").mkdir()
        shutil.copy(HOOK, self._repo / "scripts" / "claude-hooks" / HOOK.name)
        self._hook = self._repo / "scripts" / "claude-hooks" / HOOK.name
        self._fires = self._repo / ".claude" / ".hook-fires.log"

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, payload: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self._hook)],
            cwd=self._repo,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )

    # --- Pass-through ------------------------------------------------------

    def test_non_adr_path_is_noop(self) -> None:
        r = self._run({
            "tool_name": "Write",
            "tool_input": {"file_path": "docs/notes.md", "content": "# X"},
        })
        self.assertEqual(r.returncode, 0)

    def test_edit_tool_is_noop(self) -> None:
        r = self._run({
            "tool_name": "Edit",
            "tool_input": {"file_path": "docs/adr/0099-x.md",
                           "old_string": "a", "new_string": "b"},
        })
        self.assertEqual(r.returncode, 0)

    def test_template_file_is_noop(self) -> None:
        # _template.md is a known exception (not a real ADR).
        r = self._run({
            "tool_name": "Write",
            "tool_input": {"file_path": "docs/adr/_template.md",
                           "content": "# Template\n"},
        })
        self.assertEqual(r.returncode, 0)

    def test_existing_file_is_noop(self) -> None:
        existing = self._repo / "docs/adr/0042-existing.md"
        existing.write_text("# Existing\n")
        r = self._run({
            "tool_name": "Write",
            "tool_input": {"file_path": str(existing),
                           "content": "# Existing\nfresh content\n"},
        })
        self.assertEqual(r.returncode, 0)

    def test_adr_with_verification_section_passes(self) -> None:
        content = (
            "# ADR 0099: Test\n\n"
            "## Status\nProposed\n\n"
            "## Decision\nDo thing.\n\n"
            "## Verification\n\n"
            "<!-- verifies-key: reports/eval_summary.json:naive_baseline -->\n"
        )
        r = self._run({
            "tool_name": "Write",
            "tool_input": {"file_path": "docs/adr/0099-test.md",
                           "content": content},
        })
        self.assertEqual(r.returncode, 0, r.stderr)

    # --- Block -------------------------------------------------------------

    def test_adr_missing_verification_blocks(self) -> None:
        r = self._run({
            "tool_name": "Write",
            "tool_input": {"file_path": "docs/adr/0099-test.md",
                           "content": "# Test\n## Status\nProposed\n"},
        })
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("Verification", r.stderr)
        self.assertIn("verifies-key", r.stderr)
        self.assertTrue(self._fires.exists())
        self.assertIn("|blocked|adr-template-missing-verification|",
                      self._fires.read_text())

    def test_malformed_adr_filename_is_noop(self) -> None:
        # `docs/adr/foo.md` — pre-commit hook owns this case.
        r = self._run({
            "tool_name": "Write",
            "tool_input": {"file_path": "docs/adr/foo.md",
                           "content": "# Foo\n"},
        })
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
