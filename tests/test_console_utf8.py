"""Offline regression tests for Windows-safe UTF-8 console output."""

from __future__ import annotations

import io
import json
import sys
import unittest
from unittest.mock import patch

from console_utf8 import configure_utf8_stdio, safe_console_print


class _CP1252TextIO(io.TextIOBase):
    """Simulates a Windows cp1252 console that rejects emoji on write."""

    encoding = "cp1252"

    def __init__(self) -> None:
        self._buf = io.BytesIO()
        self._encoding = "cp1252"
        self._errors = "strict"
        self.reconfigure_calls = 0

    def reconfigure(self, *, encoding=None, errors=None, **kwargs):
        self.reconfigure_calls += 1
        if encoding is not None:
            self._encoding = encoding
            self.encoding = encoding
        if errors is not None:
            self._errors = errors

    def write(self, s: str) -> int:
        data = s.encode(self._encoding, errors=self._errors)
        return self._buf.write(data)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return self._buf.getvalue().decode(self._encoding, errors="replace")


class ConfigureUtf8StdioTests(unittest.TestCase):
    def test_reconfigure_allows_checkmark_on_former_cp1252_stream(self):
        fake = _CP1252TextIO()
        with self.assertRaises(UnicodeEncodeError):
            fake.write("\u2705 wiki ok\n")

        with patch.object(sys, "stdout", fake), patch.object(sys, "stderr", fake):
            configure_utf8_stdio()
            # Must not raise after reconfigure to utf-8 + replace.
            print("\u2705 **Wikipedia Page:** Mercedes Sosa")

        self.assertGreaterEqual(fake.reconfigure_calls, 1)
        self.assertEqual(fake.encoding, "utf-8")
        self.assertIn("\u2705", fake.getvalue())

    def test_safe_console_print_never_raises_on_strict_cp1252(self):
        class StrictCP1252:
            encoding = "cp1252"

            def __init__(self):
                self.chunks = []

            def write(self, s):
                self.chunks.append(s.encode("cp1252"))  # raises on emoji

            def flush(self):
                return None

        stream = StrictCP1252()
        # Without buffer, safe_console_print falls back to replaced text reprint.
        # Attach a buffer so the helper can write replaced bytes.
        stream.buffer = io.BytesIO()

        def write(s):
            raise UnicodeEncodeError("cp1252", s, 0, 1, "test")

        stream.write = write  # type: ignore[method-assign]
        safe_console_print("prefix", "\u2705", file=stream, flush=True)
        raw = stream.buffer.getvalue()
        self.assertTrue(raw)
        # Replacement character or ascii-safe payload; must not raise.
        self.assertIsInstance(raw, (bytes, bytearray))

    def test_json_preserves_unicode_with_utf8(self):
        payload = [{"note": "\u2705 preserved", "answer": "3"}]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.assertIn("\u2705", text)
        encoded = text.encode("utf-8")
        self.assertIn("\u2705".encode("utf-8"), encoded)


class EvalJsonEncodingTests(unittest.TestCase):
    def test_run_local_eval_writes_utf8_json(self):
        import inspect
        import run_local_eval

        source = inspect.getsource(run_local_eval)
        self.assertIn('ensure_ascii=False', source)
        self.assertIn('encoding="utf-8"', source)
        self.assertIn("configure_utf8_stdio", source)


if __name__ == "__main__":
    unittest.main()
