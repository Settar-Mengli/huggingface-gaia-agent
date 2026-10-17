"""GAIA prompt-template customization for smolagents (no site-packages edits)."""

from __future__ import annotations

import copy
import importlib.resources
import re
from typing import Any

import yaml

# Neutral reverse-string few-shot (replaces Einstein/Ulam exemplar).
_REVERSE_STRING_EXAMPLE = """---
Task:
The string "dlrow" is reversed English. What is the original word?

Thought: Reverse the characters with Python, then return only the word.
{{code_block_opening_tag}}
print("dlrow"[::-1])
final_answer("world")
{{code_block_closing_tag}}

"""

_EINSTEIN_TASK_RE = re.compile(
    r"(?ms)^---\nTask:\nIn a 1979 interview, Stanislaus Ulam.*?"
    r"^final_answer\(\"diminished\"\)\n\{\{code_block_closing_tag\}\}\n\n"
)


def load_gaia_prompt_templates() -> dict[str, Any]:
    """
    Load smolagents CodeAgent templates, remove Einstein/Ulam few-shot contamination,
    and tighten max-steps final-answer instructions.
    """
    raw = (
        importlib.resources.files("smolagents.prompts")
        .joinpath("code_agent.yaml")
        .read_text(encoding="utf-8")
    )
    templates = yaml.safe_load(raw)
    templates = copy.deepcopy(templates)
    templates["system_prompt"] = sanitize_system_prompt(templates["system_prompt"])
    templates["final_answer"] = {
        "pre_messages": (
            "An agent attempted a user task but reached its step limit. "
            "Using ONLY the observations already present in the agent memory below, "
            "produce the concise final answer for exact-match scoring."
        ),
        "post_messages": (
            "User task:\n{{task}}\n\n"
            "Return ONLY the exact answer value supported by those observations. "
            "Do not perform new research, speculate, mention tools, serialize tool calls, "
            "use Markdown, or write explanations. "
            "If observations are insufficient, return an empty string."
        ),
    }
    return templates


def sanitize_system_prompt(system_prompt: str) -> str:
    """Remove Einstein/Ulam few-shot and insert a neutral reverse-string example."""
    text = system_prompt
    if "Stanislaus Ulam" in text or "Einstein learning too much math" in text:
        replaced, n = _EINSTEIN_TASK_RE.subn(_REVERSE_STRING_EXAMPLE, text, count=1)
        if n:
            text = replaced
        else:
            # Fallback: drop any paragraph that names the contaminated exemplar.
            text = re.sub(
                r"(?is)Stanislaus Ulam.*?final_answer\(\"diminished\"\).*?"
                r"\{\{code_block_closing_tag\}\}\s*",
                "",
                text,
                count=1,
            )
            if "---\nTask:\nThe string \"dlrow\"" not in text:
                # Insert before the "Above examples were using notional tools" section.
                marker = "Above examples were using notional tools"
                if marker in text:
                    text = text.replace(marker, _REVERSE_STRING_EXAMPLE + marker, 1)
    return text


def effective_system_prompt_text() -> str:
    """Return the sanitized system prompt string (for offline tests)."""
    return load_gaia_prompt_templates()["system_prompt"]
