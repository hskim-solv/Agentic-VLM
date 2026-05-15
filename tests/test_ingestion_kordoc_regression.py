"""HwpKordocLoader regression suite (ADR 0049, issue #890).

Pins the load-bearing surfaces from ADR 0049's Verification section:

* ``HwpKordocLoader`` invocation shape (npx command + flags).
* Node-missing → ``data_list_csv_text`` fallback (CI without Node).
* Subprocess failure → ``data_list_csv_text`` fallback (offline / kordoc error).
* Telemetry-key stability (``last_text_source`` / ``last_fallback_reason``).
* Batch priming reads cache; per-row ``load_text`` consumes it.
* NFC normalization for Korean filenames (macOS HFS+ NFD round-trip).
* ``_resolve_loader`` honors ``BIDMATE_HWP_LOADER=csv_text`` opt-out.

The subprocess is mocked end-to-end so the suite runs on CI without Node
installed — the Node-missing case explicitly checks the graceful path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unicodedata
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ingestion import (
    HwpCsvTextLoader,
    HwpKordocLoader,
    _kordoc_output_stem,
    _read_kordoc_version_spec,
    _reset_hwp_kordoc_loader,
    _resolve_loader,
)


class HwpKordocLoaderRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = os.environ.get("BIDMATE_HWP_LOADER")
        os.environ.pop("BIDMATE_HWP_LOADER", None)
        _reset_hwp_kordoc_loader()

    def tearDown(self) -> None:
        if self._env_backup is None:
            os.environ.pop("BIDMATE_HWP_LOADER", None)
        else:
            os.environ["BIDMATE_HWP_LOADER"] = self._env_backup
        _reset_hwp_kordoc_loader()

    def test_default_loader_is_kordoc(self) -> None:
        loader = _resolve_loader("hwp")
        self.assertIsInstance(loader, HwpKordocLoader)

    def test_csv_text_opt_out(self) -> None:
        os.environ["BIDMATE_HWP_LOADER"] = "csv_text"
        loader = _resolve_loader("hwp")
        self.assertIsInstance(loader, HwpCsvTextLoader)
        self.assertNotIsInstance(loader, HwpKordocLoader)

    def test_legacy_native_aliased_to_csv_with_deprecation(self) -> None:
        for legacy in ("native", "native_tables"):
            with self.subTest(legacy=legacy):
                os.environ["BIDMATE_HWP_LOADER"] = legacy
                with self.assertWarns(DeprecationWarning):
                    loader = _resolve_loader("hwp")
                self.assertIsInstance(loader, HwpCsvTextLoader)

    def test_load_text_falls_back_when_cache_empty(self) -> None:
        loader = HwpKordocLoader()
        row = {"텍스트": "csv body text"}
        result = loader.load_text(row, Path("/no/such/file.hwp"))
        self.assertEqual(result, "csv body text")
        self.assertEqual(loader.last_text_source, "data_list_csv_text")

    def test_load_text_empty_text_raises(self) -> None:
        loader = HwpKordocLoader()
        with self.assertRaises(ValueError) as ctx:
            loader.load_text({"텍스트": ""}, Path("/no/such/file.hwp"))
        self.assertEqual(str(ctx.exception), "empty_text")

    def test_load_text_reads_primed_cache_with_nfc_stem(self) -> None:
        loader = HwpKordocLoader()
        nfd_stem = unicodedata.normalize("NFD", "한국어공고")
        source_path = Path(f"/files/{nfd_stem}.hwp")
        loader._batch_cache[str(source_path)] = "kordoc markdown body"
        result = loader.load_text({"텍스트": "csv fallback"}, source_path)
        self.assertEqual(result, "kordoc markdown body")
        self.assertEqual(loader.last_text_source, "kordoc")

    def test_kordoc_version_spec_reads_pinned_version(self) -> None:
        spec = _read_kordoc_version_spec()
        self.assertTrue(
            spec == "kordoc" or spec.startswith("kordoc@"),
            f"unexpected spec: {spec!r}",
        )

    def test_kordoc_output_stem_normalizes_to_nfc(self) -> None:
        nfd_stem = unicodedata.normalize("NFD", "한국어공고")
        self.assertNotEqual(nfd_stem, "한국어공고")
        normalized = _kordoc_output_stem(Path(f"/files/{nfd_stem}.hwp"))
        self.assertEqual(normalized, "한국어공고")

    def test_prime_batch_node_missing_falls_back_gracefully(self) -> None:
        loader = HwpKordocLoader()
        with mock.patch.object(shutil, "which", return_value=None):
            with self.assertWarns(RuntimeWarning):
                loader.prime_batch([Path("/files/a.hwp")])
        self.assertEqual(loader._batch_cache, {})
        self.assertIn("node/npx", loader.last_fallback_reason or "")
        result = loader.load_text(
            {"텍스트": "csv body"}, Path("/files/a.hwp")
        )
        self.assertEqual(result, "csv body")
        self.assertEqual(loader.last_text_source, "data_list_csv_text")

    def test_prime_batch_subprocess_error_falls_back(self) -> None:
        loader = HwpKordocLoader()
        fake_error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["npx"],
            output="",
            stderr="kordoc: parse error",
        )
        with mock.patch.object(shutil, "which", return_value="/usr/bin/npx"):
            with mock.patch.object(subprocess, "run", side_effect=fake_error):
                with self.assertWarns(RuntimeWarning):
                    loader.prime_batch([Path("/files/a.hwp")])
        self.assertEqual(loader._batch_cache, {})
        self.assertIn("npx exit 1", loader.last_fallback_reason or "")

    def test_prime_batch_success_populates_cache(self) -> None:
        loader = HwpKordocLoader()
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "doc.hwp"
            source.write_bytes(b"\x00" * 4)
            captured_cmd: list[list[str]] = []

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                captured_cmd.append(list(cmd))
                out_dir = Path(cmd[cmd.index("-d") + 1])
                (out_dir / "doc.md").write_text(
                    "# Heading\n\nkordoc body\n", encoding="utf-8"
                )
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(shutil, "which", return_value="/usr/bin/npx"):
                with mock.patch.object(subprocess, "run", side_effect=fake_run):
                    loader.prime_batch([source])

        self.assertEqual(len(captured_cmd), 1)
        cmd = captured_cmd[0]
        self.assertEqual(cmd[0], "npx")
        self.assertIn("kordoc", cmd)
        self.assertIn(_read_kordoc_version_spec(), cmd)
        self.assertIn("pdfjs-dist", cmd)
        self.assertIn("--silent", cmd)
        self.assertIn(str(source), cmd)
        self.assertIsNone(loader.last_fallback_reason)
        self.assertIn(str(source), loader._batch_cache)
        result = loader.load_text({"텍스트": "csv"}, source)
        self.assertIn("kordoc body", result)
        self.assertEqual(loader.last_text_source, "kordoc")


if __name__ == "__main__":
    unittest.main()
