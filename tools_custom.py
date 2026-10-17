"""Custom tools for GAIA Level-1 evaluation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

import requests
from smolagents import Tool, VisitWebpageTool

from answer_utils import format_spoken_page_numbers

DEFAULT_API_URL = os.getenv(
    "GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"
)
DOWNLOAD_DIR = Path(os.getenv("GAIA_DOWNLOAD_DIR", "downloads"))

_WIKI_HOST_RE = re.compile(
    r"(?:^|\.)(?:wikipedia\.org|wikimedia\.org|mediawiki\.org)$", re.I
)


def normalize_url_key(url: str) -> str:
    """Canonical key for failed-URL tracking (scheme/host/path, no fragment)."""
    raw = (url or "").strip()
    try:
        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower()
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        return urlunparse((parsed.scheme.lower(), host, path, "", "", ""))
    except Exception:
        return raw.rstrip("/").lower()


def is_wikipedia_url(url: str) -> bool:
    try:
        host = urlparse(url.strip()).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return bool(_WIKI_HOST_RE.search(host))
    except Exception:
        return "wikipedia.org" in (url or "").lower()


def hub_fallback_allowed() -> bool:
    """
    Hub fallback hits gated gaia-benchmark/GAIA and must be explicitly opted in
    after the user accepts dataset terms on the Hub.
    """
    return os.getenv("GAIA_ALLOW_HUB_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _truncate_tool_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)].rstrip() + "\n...[truncated]"


class SafeVisitWebpageTool(Tool):
    """
    Visit webpage with Wikipedia blocked and per-question visit memory.

    Prefer wikipedia_search / web_search for encyclopedic facts; Wikipedia
    often returns HTTP 403 to generic scrapers.
    """

    name = "visit_webpage"
    description = (
        "Visit a webpage URL and return a short text excerpt. "
        "Do NOT use this for Wikipedia/Wikimedia URLs (they often return 403); "
        "use wikipedia_search or web_search instead. "
        "Never retry a URL already visited (success or failure) for this question. "
        "Do not paste large tables into Python — extract only needed values."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL to visit (non-Wikipedia).",
        },
    }
    output_type = "string"

    def __init__(self, max_output_length: Optional[int] = None):
        super().__init__()
        if max_output_length is None:
            max_output_length = int(os.getenv("GAIA_VISIT_MAX_CHARS", "3500"))
        self.max_output_length = max_output_length
        # Inner tool gets the same cap so fetch path stays bounded.
        self._inner = VisitWebpageTool(max_output_length=max_output_length)
        self.failed_urls: set[str] = set()
        self.visited_urls: set[str] = set()

    def reset(self) -> None:
        """Clear per-question visit history (call between tasks)."""
        self.failed_urls.clear()
        self.visited_urls.clear()

    def forward(self, url: str) -> str:
        target = (url or "").strip()
        if not target:
            return _truncate_tool_text("ERROR: empty URL.", self.max_output_length)
        if is_wikipedia_url(target):
            return _truncate_tool_text(
                "ERROR: Do not fetch Wikipedia/Wikimedia via visit_webpage "
                "(commonly HTTP 403). Use wikipedia_search for article text, "
                "or web_search for titles/snippets, then answer.",
                self.max_output_length,
            )
        key = normalize_url_key(target)
        if key in self.visited_urls:
            return _truncate_tool_text(
                f"ERROR: URL already visited this question; do not retry. "
                f"Use a different source or answer from prior observations. url={target}",
                self.max_output_length,
            )
        if key in self.failed_urls:
            return _truncate_tool_text(
                f"ERROR: URL previously failed; do not retry. "
                f"Use a different source. url={target}",
                self.max_output_length,
            )
        try:
            result = self._inner.forward(target)
        except Exception as exc:
            self.failed_urls.add(key)
            self.visited_urls.add(key)
            return _truncate_tool_text(
                f"ERROR visiting {target}: {exc}", self.max_output_length
            )

        text = result if isinstance(result, str) else str(result)
        lower = text.lower()
        self.visited_urls.add(key)
        if (
            "error fetching" in lower
            or "403" in text
            or "404" in text
            or "forbidden" in lower
            or text.startswith("ERROR")
        ):
            self.failed_urls.add(key)
        return _truncate_tool_text(text, self.max_output_length)


class DownloadGaiaFileTool(Tool):
    """Download an attachment for a GAIA task_id from the scoring API."""

    name = "download_gaia_file"
    description = (
        "Download the file attached to a GAIA question using its task_id. "
        "Returns the local file path and a short text preview when possible. "
        "Only use when the question context explicitly provides an attachment file_name."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "The GAIA task_id for the current question.",
        },
        "file_name": {
            "type": "string",
            "description": "Optional original file name if known (helps choose extension).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, api_url: str = DEFAULT_API_URL, token: Optional[str] = None):
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.token = token or os.getenv("HF_TOKEN") or os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )
        self._attachment_available = False
        self._expected_file_name: Optional[str] = None
        # Absolute path of the last successful download in the current question.
        self.last_saved_path: Optional[str] = None

    def configure_for_question(self, file_name: Optional[str] = None) -> None:
        """Enable downloads only when the current question has an attachment."""
        name = (file_name or "").strip() or None
        self._expected_file_name = name
        self._attachment_available = bool(name)
        self.last_saved_path = None

    def _record_saved_path(self, out_path: Path) -> Path:
        resolved = out_path.resolve()
        self.last_saved_path = str(resolved)
        return resolved

    def _preview_for(self, out_path: Path, content: bytes, file_name: Optional[str]) -> str:
        resolved = self._record_saved_path(out_path)
        suffix = resolved.suffix.lower()
        preview = ""
        if suffix in {".py", ".txt", ".md", ".csv", ".json", ".xml"}:
            try:
                preview = content.decode("utf-8", errors="replace")[:4000]
            except Exception:
                preview = ""
        elif suffix in {".xlsx", ".xls"}:
            preview = (
                "Excel workbook downloaded. Use Python (openpyxl/pandas) to open "
                f"the file at: {resolved}"
            )
        parts = [
            f"LOCAL_PATH: {resolved}",
            f"Downloaded file saved to: {resolved}",
            f"Bytes: {len(content)}",
            f"Suggested name: {file_name or resolved.name}",
            "After download, use describe_image (images) or transcribe_audio (audio); "
            "the saved path is shared automatically.",
        ]
        if preview:
            parts.append("Preview / guidance:\n" + preview)
        return "\n".join(parts)

    def _download_via_scoring_api(
        self, task_id: str, file_name: Optional[str]
    ) -> Optional[str]:
        url = f"{self.api_url}/files/{task_id}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        suffix = Path(file_name).suffix if file_name else ""
        if not suffix:
            content_disp = response.headers.get("content-disposition", "")
            match = re.search(r'filename="?([^";]+)"?', content_disp)
            if match:
                suffix = Path(match.group(1)).suffix
            else:
                content_type = response.headers.get("content-type", "")
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "audio/mpeg": ".mp3",
                    "audio/wav": ".wav",
                    "text/x-python": ".py",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                }.get(content_type.split(";")[0].strip(), ".bin")

        out_path = DOWNLOAD_DIR / f"{task_id}{suffix}"
        out_path.write_bytes(response.content)
        return self._preview_for(out_path, response.content, file_name)

    def _download_via_gaia_hub(
        self, task_id: str, file_name: Optional[str]
    ) -> str:
        """
        Authorized fallback when the course /files endpoint is broken.

        Uses the same gated dataset the scoring Space loads:
        https://huggingface.co/datasets/gaia-benchmark/GAIA

        Requires ALL of:
        - GAIA_ALLOW_HUB_FALLBACK=1 (explicit opt-in after accepting Hub terms)
        - HF_TOKEN with access to that gated dataset
        """
        if not hub_fallback_allowed():
            return (
                "ERROR: Official /files/{task_id} failed and Hub fallback is disabled. "
                "To enable the authorized fallback: (1) accept terms at "
                "https://huggingface.co/datasets/gaia-benchmark/GAIA while logged in, "
                "(2) set GAIA_ALLOW_HUB_FALLBACK=1 in .env / Space secrets. "
                f"task_id={task_id} file_name={file_name or ''}."
            )

        from huggingface_hub import hf_hub_download, list_repo_files

        if not self.token:
            return (
                "ERROR: Hub fallback enabled but HF_TOKEN is missing. "
                "Accept terms at https://huggingface.co/datasets/gaia-benchmark/GAIA "
                "and set HF_TOKEN."
            )

        candidates: list[str] = []
        if file_name:
            candidates.append(f"2023/validation/{file_name}")
            candidates.append(file_name)
        try:
            repo_files = list_repo_files(
                "gaia-benchmark/GAIA", repo_type="dataset", token=self.token
            )
            candidates.extend(
                [f for f in repo_files if task_id in f and not f.endswith(".jsonl")]
            )
        except Exception as exc:
            return (
                f"ERROR listing gaia-benchmark/GAIA (check token + dataset access): {exc}"
            )

        seen: set[str] = set()
        last_error = None
        for filename in candidates:
            if not filename or filename in seen:
                continue
            seen.add(filename)
            try:
                cached = hf_hub_download(
                    repo_id="gaia-benchmark/GAIA",
                    repo_type="dataset",
                    filename=filename,
                    token=self.token,
                )
                content = Path(cached).read_bytes()
                suffix = Path(filename).suffix or (
                    Path(file_name).suffix if file_name else ".bin"
                )
                out_path = DOWNLOAD_DIR / f"{task_id}{suffix}"
                out_path.write_bytes(content)
                note = (
                    "NOTE: Official /files/{task_id} was unavailable; "
                    "downloaded via authorized gaia-benchmark/GAIA Hub fallback.\n"
                )
                return note + self._preview_for(
                    out_path, content, file_name or Path(filename).name
                )
            except Exception as exc:
                last_error = exc
                continue
        return (
            f"ERROR: could not download attachment for task_id={task_id}. "
            f"Last Hub error: {last_error}"
        )

    def forward(self, task_id: str, file_name: Optional[str] = None) -> str:
        if not self._attachment_available:
            return (
                "ERROR: No attachment is available for this question. "
                "Do not attempt to download any attachment file."
            )
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        task_id = (task_id or "").strip()
        if not task_id:
            return "ERROR: task_id is required to download a GAIA attachment."
        file_name = (file_name or "").strip() or self._expected_file_name

        try:
            return self._download_via_scoring_api(task_id, file_name)
        except requests.HTTPError as api_exc:
            status = getattr(getattr(api_exc, "response", None), "status_code", "?")
            detail = ""
            try:
                detail = api_exc.response.json().get("detail", "")  # type: ignore[union-attr]
            except Exception:
                detail = (getattr(api_exc.response, "text", None) or "")[:200]
            print(
                f"Official /files endpoint HTTP {status} for {task_id}: {detail or api_exc}"
            )
            return self._download_via_gaia_hub(task_id, file_name)
        except requests.RequestException as api_exc:
            print(f"Official /files endpoint network error for {task_id}: {api_exc}")
            return self._download_via_gaia_hub(task_id, file_name)
        except Exception as exc:
            return f"ERROR downloading file for task_id={task_id}: {exc}"


class YouTubeTranscriptTool(Tool):
    """Fetch a YouTube video transcript when available."""

    name = "youtube_transcript"
    description = (
        "Fetch the transcript/captions for a YouTube video URL or video id. "
        "Use this for questions about spoken content in YouTube videos."
    )
    inputs = {
        "video": {
            "type": "string",
            "description": "YouTube URL or 11-character video id.",
        },
    }
    output_type = "string"

    def forward(self, video: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return "ERROR: youtube-transcript-api is not installed."

        video_id = video.strip()
        match = re.search(
            r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", video_id
        )
        if match:
            video_id = match.group(1)
        elif len(video_id) != 11:
            return f"ERROR: could not parse YouTube video id from: {video}"

        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)
            lines = [getattr(item, "text", str(item)) for item in transcript]
            text = " ".join(lines).strip()
            if not text:
                return "ERROR: empty transcript returned."
            # Cap length to keep model context manageable.
            return text[:12000]
        except Exception as exc:
            return f"ERROR fetching transcript for {video_id}: {exc}"


_LOCAL_PATH_RE = re.compile(
    r"(?im)^\s*(?:LOCAL_PATH|Downloaded file saved to):\s*(.+?)\s*$"
)


def _extract_local_path_from_tool_text(text: str) -> Optional[str]:
    """Pull LOCAL_PATH from a download_gaia_file response blob, if present."""
    if not text:
        return None
    match = _LOCAL_PATH_RE.search(str(text))
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


class ExtractPageNumbersTool(Tool):
    """Extract textbook page numbers from ASR / spoken transcript text."""

    name = "extract_page_numbers"
    description = (
        "Extract textbook page numbers mentioned in transcribed speech or prose. "
        "Handles 'page 245', lists like 'pages 132, 133 and 134', and ranges "
        "like 'pages 132-134'. Returns an ascending comma-delimited list. "
        "Use after transcribe_audio when the question asks for page numbers. "
        "Do not invent pages; ignores problem/homework numbers not tied to 'page'."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": "Transcript or prose that may mention page numbers.",
        },
    }
    output_type = "string"

    def forward(self, text: str) -> str:
        result = format_spoken_page_numbers(text or "")
        if not result:
            return "ERROR: no page numbers found in text."
        return result


class TranscribeAudioTool(Tool):
    """Transcribe a local audio file via Hugging Face Inference API (Whisper)."""

    name = "transcribe_audio"
    description = (
        "Transcribe a local audio file (mp3/wav/etc.) to text using Whisper on "
        "Hugging Face Inference. After download_gaia_file succeeds, the saved audio "
        "path is used automatically if file_path is omitted or is the download tool "
        "response text."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": (
                "Path to a local audio file previously downloaded. "
                "Optional when download_gaia_file already saved audio this question; "
                "may also be the full download_gaia_file response containing LOCAL_PATH."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        download_tool: Optional["DownloadGaiaFileTool"] = None,
    ):
        super().__init__()
        self.model_id = model_id or os.getenv(
            "GAIA_ASR_MODEL", "openai/whisper-large-v3"
        )
        self.token = token or os.getenv("HF_TOKEN") or os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )
        self.download_tool = download_tool

    def _resolve_audio_path(self, file_path: Optional[str]) -> Optional[Path]:
        """Resolve a real audio file from path, download response text, or shared path."""
        candidates: list[Path] = []
        raw = (file_path or "").strip()
        if raw:
            direct = Path(raw)
            # Only treat as a direct path when it looks like one line / path-like.
            if "\n" not in raw and len(raw) < 512:
                candidates.append(direct)
            extracted = _extract_local_path_from_tool_text(raw)
            if extracted:
                candidates.append(Path(extracted))
        shared = getattr(self.download_tool, "last_saved_path", None)
        if shared:
            candidates.append(Path(shared))
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def forward(self, file_path: Optional[str] = None) -> str:
        path = self._resolve_audio_path(file_path)
        if path is None:
            return (
                "ERROR: audio file not found. Call download_gaia_file first, "
                "or pass a valid local file_path "
                f"(got {file_path!r})."
            )
        try:
            from huggingface_hub import InferenceClient

            # Providers (e.g. FalAI) base64-encode the payload and require raw
            # bytes — a file object raises: "a bytes-like object is required,
            # not '_io.BufferedReader'".
            audio_bytes = path.read_bytes()
            client = InferenceClient(token=self.token)
            result = client.automatic_speech_recognition(
                audio_bytes, model=self.model_id
            )
            if isinstance(result, dict):
                text = result.get("text") or str(result)
            else:
                text = getattr(result, "text", None) or str(result)
            return text.strip() or "ERROR: empty transcription."
        except Exception as exc:
            return f"ERROR transcribing {path}: {exc}"


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def is_image_filename(file_name: Optional[str]) -> bool:
    if not file_name:
        return False
    return Path(str(file_name)).suffix.lower() in _IMAGE_SUFFIXES


class DescribeImageTool(Tool):
    """Describe / answer questions about a local image via a vision LLM."""

    name = "describe_image"
    description = (
        "Analyze a local image file with a vision model. Use for chess boards, "
        "diagrams, screenshots, and other visual attachments. Provide a clear question. "
        "After download_gaia_file succeeds, the saved image path is used automatically "
        "if file_path is omitted or invalid."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": (
                "Path to a local image file previously downloaded. "
                "Optional when download_gaia_file already saved an image this question."
            ),
            "nullable": True,
        },
        "question": {
            "type": "string",
            "description": "What to determine from the image.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        provider: Optional[str] = None,
        download_tool: Optional["DownloadGaiaFileTool"] = None,
    ):
        super().__init__()
        self.model_id = model_id or os.getenv(
            "GAIA_VISION_MODEL", "google/gemma-3-4b-it"
        )
        self.provider = (
            provider
            if provider is not None
            else os.getenv("GAIA_VISION_PROVIDER", "deepinfra")
        )
        self.token = token or os.getenv("HF_TOKEN") or os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )
        self.download_tool = download_tool
        self.vision_succeeded = False
        self.vision_failed = False

    def reset_for_question(self) -> None:
        """Clear per-question vision outcome flags (call between tasks/retries)."""
        self.vision_succeeded = False
        self.vision_failed = False

    def _resolve_image_path(self, file_path: Optional[str]) -> Optional[Path]:
        raw = (file_path or "").strip()
        if raw:
            candidate = Path(raw)
            if candidate.exists() and candidate.is_file():
                return candidate
        shared = getattr(self.download_tool, "last_saved_path", None)
        if shared:
            candidate = Path(shared)
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def forward(self, question: str, file_path: Optional[str] = None) -> str:
        path = self._resolve_image_path(file_path)
        if path is None:
            self.vision_failed = True
            return (
                "ERROR: image file not found. Call download_gaia_file first, "
                "or pass a valid local file_path."
            )
        try:
            import base64
            from huggingface_hub import InferenceClient

            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
            }.get(path.suffix.lower(), "image/png")
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
            client_kwargs = {"token": self.token}
            if self.provider:
                client_kwargs["provider"] = self.provider
            client = InferenceClient(**client_kwargs)
            prompt = (
                "Answer precisely and concisely. If the user asks for a chess move, "
                "reply only in algebraic notation.\n\nQuestion: "
                + question
            )
            result = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                max_tokens=512,
            )
            content = (result.choices[0].message.content or "").strip()
            if not content:
                self.vision_failed = True
                return f"ERROR describing image {path}: empty vision model response."
            self.vision_succeeded = True
            self.vision_failed = False
            return content
        except Exception as exc:
            self.vision_failed = True
            return f"ERROR describing image {path}: {exc}"


class SafeWikipediaSearchTool(Tool):
    """Wikipedia search using summary content with a hard output cap."""

    name = "wikipedia_search"
    description = (
        "Searches Wikipedia and returns a concise summary of the topic, "
        "along with the page URL. Prefer this over visit_webpage for Wikipedia."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smengli-GAIA-Agent (huggingface-agents-course)",
        language: str = "en",
        max_output_length: Optional[int] = None,
    ):
        super().__init__()
        from smolagents import WikipediaSearchTool

        if max_output_length is None:
            max_output_length = int(os.getenv("GAIA_WIKI_MAX_CHARS", "4000"))
        self.max_output_length = max_output_length
        self._inner = WikipediaSearchTool(
            user_agent=user_agent,
            language=language,
            content_type="summary",
            extract_format="WIKI",
        )

    def forward(self, query: str) -> str:
        try:
            result = self._inner.forward(query)
        except Exception as exc:
            return _truncate_tool_text(
                f"ERROR fetching Wikipedia summary: {exc}", self.max_output_length
            )
        text = result if isinstance(result, str) else str(result)
        return _truncate_tool_text(text, self.max_output_length)
