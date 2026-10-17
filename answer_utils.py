"""Answer normalization and retry helpers (no network / no secrets)."""

from __future__ import annotations

import re
from typing import Any


_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:final\s*answer|answer|the\s+answer\s+is|the\s+final\s+answer\s+is)\s*[:\-–—]?\s*"
)
_USER_PREFIX_RE = re.compile(r"(?is)^\s*user:\s*")
# Single-line prose that embeds "was X" / "is X" — only strip when clearly a full sentence.
_SENTENCE_WAS_RE = re.compile(
    r"(?is)^(?:the\s+)?(?:user\s+who\s+.+?\s+was|nominator\s+was|answer\s+was)\s+(.+)$"
)
_SENTENCE_IS_RE = re.compile(
    r"(?is)^(?:the\s+)?(?:correct\s+answer|final\s+answer)\s+is\s+(.+)$"
)
_ATOMIC_TOKEN = (
    r"\$?-?\d+(?:\.\d+)?"
    r"|[A-Za-z][\w.'-]{0,39}"
    r"|[O0]-[O0](?:-[O0])?"
    r"|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
)
# Trailing cue + atomic value (e.g. "is **3**", "would be **two**", "code **AND**").
_TRAILING_CUE_ATOMIC_RE = re.compile(
    rf"(?is)\b(?:is|be|was|code|number)\s+(?:\*\*|__)?({_ATOMIC_TOKEN})(?:\*\*|__)?\s*\.?\s*$"
)
# Trailing Markdown-wrapped atomic at end of long prose.
_TRAILING_MD_ATOMIC_RE = re.compile(
    rf"(?is)(?:\*\*|__)({_ATOMIC_TOKEN})(?:\*\*|__)\s*\.?\s*$"
)

# Provider / transport failures that must never be scored as answers.
_ERROR_LIKE_RE = re.compile(
    r"(?is)("
    r"error\s+in\s+generating\s+final\s+llm\s+output"
    r"|http\s*[:=]?\s*500"
    r"|internal\s+server\s+error"
    r"|bad\s+gateway"
    r"|service\s+unavailable"
    r"|gateway\s+timeout"
    r"|http\s*[:=]?\s*502"
    r"|http\s*[:=]?\s*503"
    r"|http\s*[:=]?\s*504"
    r"|status[_\s]?code\s*[:=]?\s*50[0-4]"
    r"|rate\s*limit(?:ed|ing)?"
    r"|too\s+many\s+requests"
    r"|invalid\s+user\s+token"
    r"|unauthorized"
    r"|payment\s+required"
    r"|no\s+remaining\s+credits"
    r"|context\s+budget\s+exceeded"
    r"|huggingfacehubHTTPError"
    r"|server\s+error"
    r")"
)

_TOOL_DUMP_RE = re.compile(
    r"(?is)("
    r"python_interpreter"
    r"|\[\s*\{\s*['\"]id['\"]"
    r"|['\"]type['\"]\s*:\s*['\"]function['\"]"
    r"|tool_calls?\b"
    r"|function['\"]\s*:\s*\{"
    r")"
)

_HEDGE_RE = re.compile(
    r"(?is)\b("
    r"please note"
    r"|best to verify"
    r"|would be best to verify"
    r"|cannot (?:be )?(?:sure|determined|confirm)"
    r"|unable to (?:determine|confirm|verify)"
    r"|not (?:enough|sufficient) (?:information|data|evidence)"
    r"|plausible answer based on"
    r"|I (?:think|believe|guess|am not sure)"
    r")\b"
)


def extract_answer_text(raw: object) -> str:
    """Flatten agent/model outputs (str, ChatMessage, content blocks) to text."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    content = getattr(raw, "content", None)
    if content is not None:
        return _content_to_text(content)
    return str(raw)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def is_error_like_answer(raw: object) -> bool:
    """
    True when output looks like a provider/transport/finalization error payload.

    Conservative: short legitimate answers (lists, dates, numbers, names) pass.
    """
    text = extract_answer_text(raw).strip()
    if not text:
        return False
    # Very short non-prose answers are almost never error payloads.
    if len(text) <= 64 and not _ERROR_LIKE_RE.search(text):
        return False
    if _ERROR_LIKE_RE.search(text):
        return True
    lower = text.lower()
    if lower.startswith("error") and any(
        key in lower
        for key in ("http", "500", "502", "503", "504", "token", "unauthorized", "timeout")
    ):
        return True
    return False


def is_unusable_answer(raw: object) -> bool:
    """
    True for tool-call dumps / hedge-only non-answers that must not be scored.

    Does not reject legitimate short answers, lists, names, dates, or numbers.
    """
    if is_error_like_answer(raw):
        return True
    text = extract_answer_text(raw).strip()
    if not text:
        return False
    if _TOOL_DUMP_RE.search(text):
        return True
    # Hedge-only: long text with verification hedges and no short atomic payload.
    if len(text) >= 80 and _HEDGE_RE.search(text):
        if _looks_like_prose_sentence(text):
            extracted = _extract_from_prose(text)
            if not extracted:
                return True
    return False


def clean_answer(raw: object) -> str:
    """
    Conservatively normalize agent output for GAIA exact-match scoring.

    Preserves multiword answers, numbers, dates, and comma-separated lists.
    Does not hardcode benchmark answers or aggressively truncate content.
    Never converts provider/error/tool-dump/hedge payloads into scored answers.
    """
    if is_unusable_answer(raw):
        # Still try to salvage a clear trailing atomic from hedge prose.
        text0 = extract_answer_text(raw).strip()
        if text0 and _looks_like_prose_sentence(text0) and not _TOOL_DUMP_RE.search(text0):
            salvaged = _extract_from_prose(text0)
            if salvaged and _is_atomic_answer_token(salvaged):
                return _strip_markdown_wrap(salvaged)
        return ""

    text = extract_answer_text(raw).strip()
    if not text:
        return ""

    # Prefer the last non-empty line if the model explained then answered.
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[-1]

    if is_unusable_answer(text):
        return ""

    text = _PREFIX_RE.sub("", text).strip()
    text = _USER_PREFIX_RE.sub("", text).strip()

    # Strip matching surrounding quotes only.
    if len(text) >= 2 and (
        (text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")
    ):
        text = text[1:-1].strip()

    text = _strip_markdown_wrap(text)

    # Unwrap a clear full-sentence wrapper without touching short phrases/lists.
    if _looks_like_prose_sentence(text):
        extracted = _extract_from_prose(text)
        if extracted:
            text = extracted

    text = _strip_markdown_wrap(text)

    # Wikipedia-style "User:Name" → Name when the whole answer is a user link.
    user_only = re.match(r"(?i)^User:(\S+)$", text)
    if user_only:
        text = user_only.group(1)

    # Light trailing punctuation trim for names/numbers (not lists).
    if "," not in text and not re.search(r"\d+\.\d+", text):
        text = text.rstrip(".").strip()

    if is_unusable_answer(text):
        return ""

    return text


def _strip_markdown_wrap(text: str) -> str:
    """Unwrap a single layer (or few) of bold/italic markers around the whole value."""
    t = text.strip()
    for _ in range(3):
        bold = re.match(r"^(?:\*\*|__)(.+?)(?:\*\*|__)$", t, flags=re.DOTALL)
        if bold:
            t = bold.group(1).strip()
            continue
        italic = re.match(r"^\*(.+?)\*$", t, flags=re.DOTALL)
        if italic and not t.startswith("**"):
            t = italic.group(1).strip()
            continue
        break
    return t


def _extract_from_prose(text: str) -> str:
    """
    Pull a clearly marked final value out of long prose.

    Conservative: trailing cue+atomic / markdown-wrapped atomic / explicit
    final-answer wrappers. Does not scan for the last number in arbitrary text.
    """
    # Prefer the trailing "was <Name>" (avoids matching mid-sentence "that was").
    was_match = re.search(
        r"(?is)\bwas\s+(User:\S+|[A-Za-z][\w.'-]*)\s*\.?\s*$", text
    )
    if was_match and text.count(" ") >= 8:
        return was_match.group(1).strip().rstrip(".")

    for pattern in (_SENTENCE_IS_RE, _SENTENCE_WAS_RE):
        match = pattern.match(text)
        if match:
            candidate = match.group(1).strip().rstrip(".")
            candidate = _USER_PREFIX_RE.sub("", candidate).strip()
            candidate = _strip_markdown_wrap(candidate)
            if candidate:
                return candidate

    if text.count(" ") >= 6:
        for pattern in (_TRAILING_CUE_ATOMIC_RE, _TRAILING_MD_ATOMIC_RE):
            trailing = pattern.search(text)
            if trailing:
                candidate = _strip_markdown_wrap(trailing.group(1).strip())
                if candidate and _is_atomic_answer_token(candidate):
                    return candidate
    return ""


def _is_atomic_answer_token(text: str) -> bool:
    """True for short exact-match style tokens (not residual prose)."""
    t = text.strip()
    if not t or len(t) > 40:
        return False
    if "\n" in t:
        return False
    # Allow at most a short multiword phrase (e.g. New York); reject sentences.
    if t.count(" ") > 3:
        return False
    return True


def _looks_like_prose_sentence(text: str) -> bool:
    if "," in text and text.count(",") >= 2 and len(text) < 90:
        return False  # likely an ingredients / list answer
    if len(text) < 40:
        return False
    if not text[0].isupper():
        return False
    lower = text.lower()
    if any(
        key in lower
        for key in (" was ", " is ", " be ", "would be", "therefore", "based on")
    ):
        return True
    # Long multi-sentence max-steps prose without those cues.
    return len(text) >= 80 and "." in text


def is_transient_error(exc: BaseException | str) -> bool:
    """Return True only for errors worth a bounded retry."""
    message = str(exc).lower()
    # Permanent auth / billing / budget — never retry.
    if any(
        key in message
        for key in (
            "401",
            "403",
            "402",
            "invalid user token",
            "unauthorized",
            "payment required",
            "no remaining credits",
            "invalid username or password",
            "context budget exceeded",
        )
    ):
        return False
    return any(
        key in message
        for key in (
            "429",
            "rate limit",
            "ratelimit",
            "timeout",
            "timed out",
            "temporar",
            "502",
            "503",
            "504",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "gateway",
        )
    )


def should_retry(attempt: int, max_retries: int, exc: BaseException | str) -> bool:
    """Whether another attempt should run after a failure."""
    if attempt >= max_retries:
        return False
    return is_transient_error(exc)


# Spoken textbook page mentions (ASR transcripts). Ignore problem/homework numbers.
_PAGE_ANCHOR_RE = re.compile(r"(?i)\bpages?\b")
_PAGE_STOP_RE = re.compile(
    r"(?i)(?:[.;:\n]|problems?\b|homework\b|quiz\b|midterm\b|test\b|exam\b|"
    r"chapter\b|section\b(?!\s*,?\s*page))"
)
_PAGE_RANGE_DASH_RE = re.compile(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})")
_PAGE_RANGE_TO_RE = re.compile(r"(?i)(\d{1,4})\s*(?:to|through)\s*(\d{1,4})")
_PAGE_NUM_RE = re.compile(r"\b(\d{1,4})\b")
_MAX_PAGE = 9999
_MAX_RANGE_SPAN = 40


def extract_spoken_page_numbers(text: str) -> list[int]:
    """
    Extract textbook page numbers from natural-speech / ASR text.

    Handles singular/plural cues and lists/ranges, e.g.:
    - "page 245"
    - "pages 132, 133 and 134"
    - "pages 132-134" / "pages 132 to 134"

    Does not treat bare problem numbers, dates, or other digits as pages unless
    they appear in a short window after a "page"/"pages" mention.
    """
    if not text or not str(text).strip():
        return []
    found: list[int] = []
    for match in _PAGE_ANCHOR_RE.finditer(str(text)):
        window = str(text)[match.end() : match.end() + 96]
        window = re.sub(
            r"(?i)^\s*(?:numbers?|nos?\.?|no\.?|#|:)?\s*",
            "",
            window,
        )
        stop = _PAGE_STOP_RE.search(window)
        segment = window[: stop.start()] if stop else window
        found.extend(_parse_page_number_segment(segment))
    return sorted(set(found))


def format_spoken_page_numbers(text: str) -> str:
    """Comma-delimited ascending page list for exact-match style answers."""
    return ",".join(str(n) for n in extract_spoken_page_numbers(text))


def _parse_page_number_segment(segment: str) -> list[int]:
    pages: list[int] = []
    for a, b in _PAGE_RANGE_DASH_RE.findall(segment):
        pages.extend(_expand_page_range(int(a), int(b)))
    for a, b in _PAGE_RANGE_TO_RE.findall(segment):
        pages.extend(_expand_page_range(int(a), int(b)))
    cleaned = _PAGE_RANGE_DASH_RE.sub(" ", segment)
    cleaned = _PAGE_RANGE_TO_RE.sub(" ", cleaned)
    for num in _PAGE_NUM_RE.findall(cleaned):
        value = int(num)
        if 1 <= value <= _MAX_PAGE:
            pages.append(value)
    return pages


def _expand_page_range(lo: int, hi: int) -> list[int]:
    if lo > hi:
        lo, hi = hi, lo
    if lo < 1 or hi > _MAX_PAGE or (hi - lo) > _MAX_RANGE_SPAN:
        return []
    return list(range(lo, hi + 1))
