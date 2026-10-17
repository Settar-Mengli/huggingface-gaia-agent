"""Mocked tests for DescribeImageTool path handoff and vision requests."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools_custom import DescribeImageTool, DownloadGaiaFileTool, is_image_filename


class ImageFilenameTests(unittest.TestCase):
    def test_image_suffixes(self):
        self.assertTrue(is_image_filename("board.png"))
        self.assertTrue(is_image_filename("x.JPG"))
        self.assertFalse(is_image_filename("sheet.xlsx"))
        self.assertFalse(is_image_filename(None))


class DescribeImageToolTests(unittest.TestCase):
    def _png_bytes(self) -> bytes:
        # Minimal valid PNG signature + tiny payload (not a full PNG parse needed).
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def test_successful_request_uses_provider_model_and_data_url(self):
        download = DownloadGaiaFileTool(token=None)
        tool = DescribeImageTool(
            model_id="google/gemma-3-4b-it",
            provider="deepinfra",
            token="hf_test_not_real",
            download_tool=download,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            path.write_bytes(self._png_bytes())
            download.last_saved_path = str(path.resolve())

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Rd8#"))]
            )
            with patch(
                "huggingface_hub.InferenceClient", return_value=mock_client
            ) as client_cls:
                out = tool.forward(file_path="not-a-real-path", question="best move?")
            self.assertEqual(out, "Rd8#")
            self.assertTrue(tool.vision_succeeded)
            self.assertFalse(tool.vision_failed)
            client_cls.assert_called_once_with(
                token="hf_test_not_real", provider="deepinfra"
            )
            kwargs = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "google/gemma-3-4b-it")
            content = kwargs["messages"][0]["content"]
            url = content[1]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            self.assertNotIn("file://", url)
            encoded = url.split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded)[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_direct_saved_path_handoff_without_llm_path(self):
        download = DownloadGaiaFileTool(token=None)
        tool = DescribeImageTool(
            model_id="google/gemma-3-4b-it",
            provider="deepinfra",
            token="hf_test_not_real",
            download_tool=download,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            path.write_bytes(self._png_bytes())
            download.last_saved_path = str(path.resolve())
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))]
            )
            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(file_path=None, question="describe")
            self.assertEqual(out, "ok")
            self.assertTrue(tool.vision_succeeded)

    def test_state_reset_between_questions(self):
        download = DownloadGaiaFileTool(token=None)
        tool = DescribeImageTool(download_tool=download, token=None)
        download.last_saved_path = "/tmp/x.png"
        tool.vision_succeeded = True
        tool.vision_failed = True
        download.configure_for_question("y.png")
        tool.reset_for_question()
        self.assertIsNone(download.last_saved_path)
        self.assertFalse(tool.vision_succeeded)
        self.assertFalse(tool.vision_failed)

    def test_provider_failure_does_not_auto_retry(self):
        download = DownloadGaiaFileTool(token=None)
        tool = DescribeImageTool(
            model_id="google/gemma-3-4b-it",
            provider="deepinfra",
            token="hf_test_not_real",
            download_tool=download,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            path.write_bytes(self._png_bytes())
            download.last_saved_path = str(path.resolve())
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError(
                "model_not_supported"
            )
            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(file_path=str(path), question="move?")
            self.assertIn("ERROR describing image", out)
            self.assertIn("model_not_supported", out)
            self.assertTrue(tool.vision_failed)
            self.assertFalse(tool.vision_succeeded)
            self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_download_preview_records_last_saved_path(self):
        tool = DownloadGaiaFileTool(token=None)
        tool.configure_for_question("x.png")
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "x.png"
            content = self._png_bytes()
            out_path.write_bytes(content)
            preview = tool._preview_for(out_path, content, "x.png")
            self.assertTrue(preview.startswith("LOCAL_PATH: "))
            self.assertEqual(tool.last_saved_path, str(out_path.resolve()))


class VisionFailClosedAgentTests(unittest.TestCase):
    def _make_agent(self):
        from agent import GaiaAgent
        from context_budget import TokenBudgetTracker

        ga = GaiaAgent.__new__(GaiaAgent)
        ga._max_retries = 1
        ga.agent = MagicMock()
        ga.budget_tracker = TokenBudgetTracker()
        ga.visit_tool = MagicMock()
        ga.download_tool = MagicMock()
        ga.describe_tool = MagicMock()
        ga.describe_tool.vision_succeeded = False
        ga.describe_tool.reset_for_question = MagicMock()
        return ga

    def test_image_task_failed_vision_returns_empty(self):
        ga = self._make_agent()
        ga.describe_tool.vision_succeeded = False
        ga.agent.run.return_value = "g8Q"
        out = ga("chess?", task_id="t1", file_name="board.png")
        self.assertEqual(out, "")

    def test_image_task_later_success_not_rejected(self):
        ga = self._make_agent()
        ga.describe_tool.vision_succeeded = True
        ga.agent.run.return_value = "Rd8#"
        out = ga("chess?", task_id="t1", file_name="board.png")
        self.assertEqual(out, "Rd8#")

    def test_non_image_task_unaffected(self):
        ga = self._make_agent()
        ga.describe_tool.vision_succeeded = False
        ga.agent.run.return_value = "Paris"
        out = ga("capital?", task_id="t2", file_name=None)
        self.assertEqual(out, "Paris")


if __name__ == "__main__":
    unittest.main()
