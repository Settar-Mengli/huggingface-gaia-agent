"""Offline tests for TranscribeAudioTool (mocked InferenceClient)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools_custom import DownloadGaiaFileTool, TranscribeAudioTool


class TranscribeAudioToolTests(unittest.TestCase):
    def _write_fake_mp3(self, directory: Path, name: str = "sample.mp3") -> Path:
        path = directory / name
        path.write_bytes(b"ID3fake-mp3-bytes")
        return path.resolve()

    def _download_response(self, audio_path: Path) -> str:
        return (
            f"LOCAL_PATH: {audio_path}\n"
            f"Downloaded file saved to: {audio_path}\n"
            f"Bytes: {audio_path.stat().st_size}\n"
            f"Suggested name: {audio_path.name}\n"
            "After download, use describe_image (images) or transcribe_audio (audio); "
            "the saved path is shared automatically."
        )

    def test_valid_path_passes_raw_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_fake_mp3(Path(tmp))
            tool = TranscribeAudioTool(token="hf_test_token_not_real")
            mock_client = MagicMock()
            mock_client.automatic_speech_recognition.return_value = {"text": "hello world"}

            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(str(audio_path))

            self.assertEqual(out, "hello world")
            args, kwargs = mock_client.automatic_speech_recognition.call_args
            self.assertIsInstance(args[0], (bytes, bytearray))
            self.assertEqual(args[0], b"ID3fake-mp3-bytes")
            self.assertEqual(kwargs.get("model"), tool.model_id)

    def test_download_tool_response_string_resolves_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_fake_mp3(Path(tmp))
            download = DownloadGaiaFileTool(token=None)
            download.last_saved_path = str(audio_path)
            tool = TranscribeAudioTool(
                token="hf_test_token_not_real", download_tool=download
            )
            mock_client = MagicMock()
            mock_client.automatic_speech_recognition.return_value = {"text": "from blob"}

            blob = self._download_response(audio_path)
            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(blob)

            self.assertEqual(out, "from blob")
            args, _kwargs = mock_client.automatic_speech_recognition.call_args
            self.assertEqual(args[0], b"ID3fake-mp3-bytes")

    def test_uses_last_saved_path_when_file_path_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_fake_mp3(Path(tmp))
            download = DownloadGaiaFileTool(token=None)
            download.last_saved_path = str(audio_path)
            tool = TranscribeAudioTool(
                token="hf_test_token_not_real", download_tool=download
            )
            mock_client = MagicMock()
            mock_client.automatic_speech_recognition.return_value = MagicMock(
                text="  spoken  "
            )

            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(file_path=None)

            self.assertEqual(out, "spoken")

    def test_uses_last_saved_path_when_invalid_path_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_fake_mp3(Path(tmp))
            download = DownloadGaiaFileTool(token=None)
            download.last_saved_path = str(audio_path)
            tool = TranscribeAudioTool(
                token="hf_test_token_not_real", download_tool=download
            )
            mock_client = MagicMock()
            mock_client.automatic_speech_recognition.return_value = {"text": "ok"}

            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward("not-a-real-path.mp3")

            self.assertEqual(out, "ok")

    def test_missing_file_returns_error_without_inference(self):
        tool = TranscribeAudioTool(token="hf_test_token_not_real")
        with patch("huggingface_hub.InferenceClient") as client_cls:
            out = tool.forward("C:/definitely/missing/audio.mp3")
        self.assertTrue(out.startswith("ERROR: audio file not found"))
        client_cls.assert_not_called()

    def test_invalid_inputs_return_error(self):
        tool = TranscribeAudioTool(token="hf_test_token_not_real")
        with patch("huggingface_hub.InferenceClient") as client_cls:
            for bad in (None, "", "   ", "LOCAL_PATH: /no/such/file.mp3\nBytes: 1"):
                out = tool.forward(bad)  # type: ignore[arg-type]
                self.assertTrue(
                    out.startswith("ERROR: audio file not found"),
                    msg=repr(bad),
                )
        client_cls.assert_not_called()

    def test_provider_typeerror_is_surfaced_as_error_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_fake_mp3(Path(tmp))
            tool = TranscribeAudioTool(token="hf_test_token_not_real")
            mock_client = MagicMock()
            mock_client.automatic_speech_recognition.side_effect = TypeError(
                "a bytes-like object is required, not '_io.BufferedReader'"
            )

            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(str(audio_path))

            self.assertTrue(out.startswith("ERROR transcribing"))
            self.assertIn("bytes-like object is required", out)

    def test_empty_transcription_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._write_fake_mp3(Path(tmp))
            tool = TranscribeAudioTool(token="hf_test_token_not_real")
            mock_client = MagicMock()
            mock_client.automatic_speech_recognition.return_value = {"text": "   "}

            with patch("huggingface_hub.InferenceClient", return_value=mock_client):
                out = tool.forward(str(audio_path))

            self.assertEqual(out, "ERROR: empty transcription.")


if __name__ == "__main__":
    unittest.main()
