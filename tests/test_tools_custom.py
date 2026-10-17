"""Unit tests for custom tools (no live network / no gated Hub access)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import requests

from tools_custom import (
    DownloadGaiaFileTool,
    SafeVisitWebpageTool,
    SafeWikipediaSearchTool,
    hub_fallback_allowed,
    is_wikipedia_url,
    normalize_url_key,
)


class UrlHelperTests(unittest.TestCase):
    def test_wikipedia_urls(self):
        self.assertTrue(
            is_wikipedia_url(
                "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles_promoted_in_2016"
            )
        )
        self.assertTrue(is_wikipedia_url("https://www.wikipedia.org/wiki/Foo"))
        self.assertFalse(is_wikipedia_url("https://example.com/wiki/Foo"))

    def test_normalize_url_key(self):
        a = normalize_url_key("https://Example.com/path/#frag")
        b = normalize_url_key("https://example.com/path")
        self.assertEqual(a, b)


class SafeVisitWebpageTests(unittest.TestCase):
    def test_blocks_wikipedia_without_calling_inner(self):
        tool = SafeVisitWebpageTool(max_output_length=3500)
        tool._inner = MagicMock()
        out = tool.forward(
            "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles_promoted_in_2016"
        )
        self.assertIn("Do not fetch Wikipedia", out)
        tool._inner.forward.assert_not_called()
        self.assertEqual(tool.failed_urls, set())

    def test_records_and_blocks_failed_url_retry(self):
        tool = SafeVisitWebpageTool(max_output_length=3500)
        tool._inner = MagicMock(
            forward=MagicMock(
                return_value="Error fetching the webpage: 403 Client Error: Forbidden"
            )
        )
        url = "https://example.com/page"
        first = tool.forward(url)
        self.assertIn("403", first)
        self.assertIn(normalize_url_key(url), tool.failed_urls)
        self.assertIn(normalize_url_key(url), tool.visited_urls)

        tool._inner.forward.reset_mock()
        second = tool.forward(url)
        self.assertIn("already visited", second)
        tool._inner.forward.assert_not_called()

    def test_blocks_duplicate_successful_visit(self):
        tool = SafeVisitWebpageTool(max_output_length=3500)
        tool._inner = MagicMock(forward=MagicMock(return_value="hello world content"))
        url = "https://example.com/ok"
        first = tool.forward(url)
        self.assertEqual(first, "hello world content")
        tool._inner.forward.reset_mock()
        second = tool.forward(url)
        self.assertIn("already visited", second)
        tool._inner.forward.assert_not_called()

    def test_success_pass_through(self):
        tool = SafeVisitWebpageTool(max_output_length=3500)
        tool._inner = MagicMock(forward=MagicMock(return_value="hello world content"))
        out = tool.forward("https://example.com/ok")
        self.assertEqual(out, "hello world content")
        self.assertEqual(tool.failed_urls, set())

    def test_output_length_cap_includes_errors(self):
        tool = SafeVisitWebpageTool(max_output_length=80)
        tool._inner = MagicMock(forward=MagicMock(return_value="A" * 500))
        out = tool.forward("https://example.com/long")
        self.assertLessEqual(len(out), 80)
        self.assertIn("truncated", out)

    def test_reset_clears_per_question_state(self):
        tool = SafeVisitWebpageTool(max_output_length=3500)
        tool._inner = MagicMock(forward=MagicMock(return_value="ok"))
        url = "https://example.com/once"
        tool.forward(url)
        self.assertTrue(tool.visited_urls)
        tool.reset()
        self.assertEqual(tool.visited_urls, set())
        self.assertEqual(tool.failed_urls, set())
        tool._inner.forward.reset_mock()
        tool.forward(url)
        tool._inner.forward.assert_called_once()


class SafeWikipediaTests(unittest.TestCase):
    def test_output_cap(self):
        tool = SafeWikipediaSearchTool.__new__(SafeWikipediaSearchTool)
        tool.max_output_length = 60
        tool._inner = MagicMock(forward=MagicMock(return_value="W" * 400))
        out = tool.forward("Olympics")
        self.assertLessEqual(len(out), 60)
        self.assertIn("truncated", out)


class DownloadGaiaFileTests(unittest.TestCase):
    def test_missing_attachment_blocks_without_network(self):
        tool = DownloadGaiaFileTool(token=None)
        tool.configure_for_question(None)
        with patch("tools_custom.requests.get") as mock_get:
            out = tool.forward("some-task-id", "ignored.png")
        self.assertIn("No attachment is available", out)
        mock_get.assert_not_called()

    def test_empty_task_id(self):
        tool = DownloadGaiaFileTool(token=None)
        tool.configure_for_question("x.py")
        self.assertIn("task_id is required", tool.forward(""))

    def test_hub_fallback_disabled_by_default(self):
        with patch.dict("os.environ", {"GAIA_ALLOW_HUB_FALLBACK": "0"}, clear=False):
            self.assertFalse(hub_fallback_allowed())
            tool = DownloadGaiaFileTool(token="hf_test_not_real")
            tool.configure_for_question("x.png")
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.json.return_value = {"detail": "No file path associated"}
            mock_resp.text = "No file path associated"
            err = requests.HTTPError(response=mock_resp)
            mock_resp.raise_for_status.side_effect = err

            with patch("tools_custom.requests.get", return_value=mock_resp):
                # Simulate raise_for_status raising HTTPError
                def raise_http():
                    raise requests.HTTPError(response=mock_resp)

                mock_resp.raise_for_status.side_effect = raise_http
                out = tool.forward("cca530fc-4052-43b2-b130-b30968d8aa44", "x.png")
            self.assertIn("Hub fallback is disabled", out)
            self.assertNotIn("downloaded via authorized", out.lower())

    def test_scoring_api_success_writes_file(self):
        tool = DownloadGaiaFileTool(token=None)
        tool.configure_for_question("f918266a-b3e0-4914-865d-4faa564f1aef.py")
        content = b"print(1)\n"
        mock_resp = MagicMock()
        mock_resp.content = content
        mock_resp.headers = {"content-type": "text/x-python"}
        mock_resp.raise_for_status = MagicMock()

        with patch("tools_custom.requests.get", return_value=mock_resp):
            with patch("tools_custom.DOWNLOAD_DIR") as fake_dir:
                from pathlib import Path
                import tempfile

                with tempfile.TemporaryDirectory() as tmp:
                    fake_path = Path(tmp)
                    # Re-bind DOWNLOAD_DIR used inside forward via mkdir/write
                    with patch("tools_custom.DOWNLOAD_DIR", fake_path):
                        out = tool.forward(
                            "f918266a-b3e0-4914-865d-4faa564f1aef",
                            "f918266a-b3e0-4914-865d-4faa564f1aef.py",
                        )
                        self.assertIn("Downloaded file saved", out)
                        self.assertIn("print(1)", out)


if __name__ == "__main__":
    unittest.main()
