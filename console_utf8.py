"""UTF-8-safe stdout/stderr configuration for Windows consoles."""

from __future__ import annotations

import io
import os
import sys
from typing import TextIO, Optional


_CONFIGURED = False


def configure_utf8_stdio() -> None:
    """
    Reconfigure process stdout/stderr to UTF-8 with replacement on encode errors.

    Call once at process startup (agent CLI, Gradio app, or tests). Safe to call
    repeatedly. Does not alter in-memory Unicode used by tools/models/JSON.
    """
    global _CONFIGURED
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _configure_stream("stdout")
    _configure_stream("stderr")
    _CONFIGURED = True


def _configure_stream(name: str) -> None:
    stream: Optional[TextIO] = getattr(sys, name, None)
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
            return
        except Exception:
            pass
    # Fallback for streams without reconfigure or when reconfigure is rejected.
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        return
    try:
        wrapped = io.TextIOWrapper(
            buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=bool(getattr(stream, "line_buffering", True)),
            write_through=bool(getattr(stream, "write_through", False)),
        )
        setattr(sys, name, wrapped)
    except Exception:
        return


def safe_console_print(*args, file: Optional[TextIO] = None, **kwargs) -> None:
    """
    Print helper that never raises UnicodeEncodeError on narrow consoles.

    Prefer configure_utf8_stdio() at startup; this is a belt-and-suspenders path
    for tests and any remaining legacy streams.
    """
    target = file if file is not None else sys.stdout
    try:
        print(*args, file=target, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = getattr(target, "encoding", None) or "utf-8"
        payload = (text + kwargs.get("end", "\n")).encode(encoding, errors="replace")
        buffer = getattr(target, "buffer", None)
        if buffer is not None:
            buffer.write(payload)
            if kwargs.get("flush"):
                buffer.flush()
        else:
            # Last resort: replace and retry via str path.
            print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), file=target, **{k: v for k, v in kwargs.items() if k != "file"})
