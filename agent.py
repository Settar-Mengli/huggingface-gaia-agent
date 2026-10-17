"""GAIA Level-1 agent built with smolagents + Hugging Face Inference."""

from __future__ import annotations

import os
import time
from typing import Optional

from console_utf8 import configure_utf8_stdio

configure_utf8_stdio()

from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
)
from smolagents.agents import populate_template
from smolagents.models import ChatMessage, MessageRole

from answer_utils import (
    clean_answer,
    is_error_like_answer,
    is_unusable_answer,
    should_retry,
)
from context_budget import (
    BudgetGuardedModel,
    ContextBudgetExceeded,
    TokenBudgetTracker,
)
from prompt_config import load_gaia_prompt_templates
from tools_custom import (
    DescribeImageTool,
    DownloadGaiaFileTool,
    ExtractPageNumbersTool,
    SafeVisitWebpageTool,
    SafeWikipediaSearchTool,
    TranscribeAudioTool,
    YouTubeTranscriptTool,
    is_image_filename,
)

load_dotenv()

GAIA_SYSTEM_PROMPT = """You are a careful general AI assistant solving GAIA Level-1 questions.

OUTPUT RULES (critical — exact match scoring):
- When you know the answer, call final_answer with ONLY the exact answer string immediately.
- Do NOT include explanations, reasoning, markdown, or labels like "FINAL ANSWER".
- No full sentences. Return the value the question asks for (name, number, short phrase, list).
- Match the format requested (number, name, short phrase, algebraic chess move, comma-separated list, etc.).
- If the question asks for a city name without abbreviations, return the full city name only.
- If the question asks for a first name only, return only that first name.
- If the question asks for a Wikipedia username / nominator, return only the username (no "User:" prose).

TOOL STRATEGY (efficiency):
- Prefer wikipedia_search for English Wikipedia facts. Do NOT use visit_webpage on wikipedia.org / wikimedia.org (often HTTP 403).
- Prefer web_search snippets when a page visit is blocked or unnecessary.
- Never repeat the same search query. Never revisit a URL already fetched (success or ERROR / 403 / 404).
- After one failed visit_webpage, switch strategy (different source or answer from snippets).
- Do NOT copy long webpage tables or large HTML dumps into generated Python code. Extract only the specific cells/values needed.
- Stop as soon as you have a high-confidence answer — do not keep exploring.

ATTACHMENTS & MEDIA:
- If text looks reversed / mirrored, reverse the string in Python first, then answer.
- Only call download_gaia_file when the question context explicitly provides an attachment file_name.
- If the context says no attachment is available, do not call download_gaia_file.
- For attached files: call download_gaia_file with the provided task_id and file_name, then analyze the file.
- For .py attachments: read/run the code with authorized Python imports and report the final numeric output.
- For .xlsx attachments: use pandas/openpyxl to compute the requested total.
- For .mp3 attachments: download, then transcribe_audio. If the question asks for page
  numbers, call extract_page_numbers on the transcript (handles "pages 132, 133 and 134"
  and ranges). Do not use a naive regex like page\\s+(\\d+) that misses plural lists.
- For image attachments (e.g. chess): download_gaia_file, then describe_image with a precise question.
  The downloaded image path is shared automatically — do not invent answers if describe_image returns ERROR.
- For YouTube URLs: prefer youtube_transcript; if that fails, search the web for known quotes/captions.
"""


def _get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Public alias for tests / callers that imported the old name.
_clean_answer = clean_answer


class BudgetAwareCodeAgent(CodeAgent):
    """CodeAgent that pre-checks token budgets before max-steps finalization."""

    def provide_final_answer(self, task: str) -> ChatMessage:
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"],
                            variables={"task": task},
                        ),
                    }
                ],
            )
        )
        model = self.model
        tracker = getattr(model, "tracker", None)
        if tracker is not None:
            tracker.check_can_generate(messages)
        # Call generate directly so parent exception-swallowing cannot turn a
        # budget failure into a scored "Error in generating..." answer.
        return model.generate(messages)


class GaiaAgent:
    """Multi-tool CodeAgent wrapper used by the Gradio evaluation app."""

    def __init__(self):
        token = _get_hf_token()
        if not token:
            print(
                "WARNING: HF_TOKEN / HUGGINGFACEHUB_API_TOKEN not set. "
                "Inference calls will fail until you add a token to .env or Space secrets."
            )

        model_id = os.getenv("GAIA_TEXT_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
        provider = os.getenv("HF_INFERENCE_PROVIDER")  # None => auto

        model_kwargs = {
            "model_id": model_id,
            "token": token,
            "temperature": float(os.getenv("GAIA_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("GAIA_MAX_TOKENS", "2048")),
        }
        if provider:
            model_kwargs["provider"] = provider

        print(f"Initializing GaiaAgent with model={model_id}, provider={provider or 'auto'}")
        self.budget_tracker = TokenBudgetTracker()
        inner_model = InferenceClientModel(**model_kwargs)
        self.model = BudgetGuardedModel(inner_model, self.budget_tracker)

        self.visit_tool = SafeVisitWebpageTool()
        self.download_tool = DownloadGaiaFileTool(token=token)
        self.describe_tool = DescribeImageTool(
            token=token, download_tool=self.download_tool
        )
        self.transcribe_tool = TranscribeAudioTool(
            token=token, download_tool=self.download_tool
        )

        tools = [
            DuckDuckGoSearchTool(max_results=8),
            self.visit_tool,
            SafeWikipediaSearchTool(
                user_agent="Smengli-GAIA-Agent (huggingface-agents-course)",
                language="en",
            ),
            self.download_tool,
            YouTubeTranscriptTool(),
            self.transcribe_tool,
            ExtractPageNumbersTool(),
            self.describe_tool,
        ]

        self.max_steps = int(os.getenv("GAIA_MAX_STEPS", "8"))
        self.prompt_templates = load_gaia_prompt_templates()
        self.agent = BudgetAwareCodeAgent(
            tools=tools,
            model=self.model,
            max_steps=self.max_steps,
            prompt_templates=self.prompt_templates,
            additional_authorized_imports=[
                "pandas",
                "numpy",
                "openpyxl",
                "math",
                "statistics",
                "json",
                "re",
                "csv",
                "io",
                "datetime",
                "collections",
                "itertools",
                "pathlib",
            ],
            instructions=GAIA_SYSTEM_PROMPT,
            step_callbacks=[self._on_action_step],
        )
        # Default 1 attempt; only transient errors may retry when max_retries > 1.
        self._max_retries = int(os.getenv("GAIA_MAX_RETRIES", "1"))
        print(f"GaiaAgent ready (max_steps={self.max_steps}, max_retries={self._max_retries}).")

    def _on_action_step(self, memory_step, agent=None) -> None:
        """Optional early exit after a step once cumulative budget is exhausted."""
        if self.budget_tracker.cumulative_input_tokens >= (
            self.budget_tracker.max_cumulative_input_tokens
        ):
            target = agent if agent is not None else self.agent
            if hasattr(target, "interrupt"):
                target.interrupt()

    def _reset_executor_state(self) -> None:
        """Clear Python executor locals between questions; keep tools bound."""
        agent = self.agent
        agent.state = {}
        executor = getattr(agent, "python_executor", None)
        if executor is None or not hasattr(executor, "state"):
            return
        tools = {**getattr(agent, "tools", {}), **getattr(agent, "managed_agents", {})}
        executor.state.clear()
        executor.state["__name__"] = "__main__"
        if hasattr(executor, "custom_tools"):
            executor.custom_tools.clear()
        if hasattr(executor, "send_tools"):
            executor.send_tools(tools)

    def _reset_question_state(self, file_name: Optional[str]) -> None:
        self.budget_tracker.reset()
        self.visit_tool.reset()
        self.download_tool.configure_for_question(file_name)
        self.describe_tool.reset_for_question()
        self._reset_executor_state()

    def __call__(
        self,
        question: str,
        task_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        print(f"Agent received question (first 80 chars): {question[:80]!r}...")
        self._reset_question_state(file_name)
        image_required = is_image_filename(file_name)

        prompt_parts = [question.strip()]
        if file_name:
            if task_id:
                prompt_parts.append(
                    f"\n\n[Context] task_id={task_id}. "
                    f"Attachment file_name={file_name}. "
                    "Pass task_id and file_name to download_gaia_file when downloading."
                )
            else:
                prompt_parts.append(
                    f"\n\n[Context] Attachment file_name={file_name}. "
                    "Pass file_name to download_gaia_file when downloading."
                )
            if image_required:
                prompt_parts.append(
                    " This is an image attachment: after download_gaia_file, call "
                    "describe_image before answering. Do not guess from memory."
                )
        else:
            ctx = "\n\n[Context] "
            if task_id:
                ctx += f"task_id={task_id}. "
            ctx += (
                "No attachment is available for this question. "
                "Do not attempt to download any attachment file."
            )
            prompt_parts.append(ctx)
        prompt = "".join(prompt_parts)

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                self._reset_question_state(file_name)
                result = self.agent.run(prompt, reset=True)
                if image_required and not self.describe_tool.vision_succeeded:
                    print(
                        f"Image task without successful vision on attempt {attempt}; "
                        "rejecting invented answer."
                    )
                    return ""
                if is_unusable_answer(result) and not clean_answer(result):
                    print(
                        f"Agent produced unusable result on attempt {attempt}; rejecting."
                    )
                    return ""
                if is_error_like_answer(result):
                    print(
                        f"Agent produced error-like result on attempt {attempt}; rejecting."
                    )
                    return ""
                answer = clean_answer(result)
                if not answer or is_error_like_answer(answer) or is_unusable_answer(answer):
                    print(
                        f"Agent produced empty/invalid cleaned answer on attempt {attempt}; rejecting."
                    )
                    return ""
                print(f"Agent answer (attempt {attempt}): {answer!r}")
                return answer
            except ContextBudgetExceeded as exc:
                print(f"Agent context budget exceeded on attempt {attempt}: {exc}")
                return ""
            except Exception as exc:
                last_error = exc
                print(f"Agent error on attempt {attempt}/{self._max_retries}: {exc}")
                if not should_retry(attempt, self._max_retries, exc):
                    break
                time.sleep(min(2 ** attempt, 20))

        print(f"Agent failed after retries: {last_error}")
        return ""


# Backward-compatible name used by some template snippets.
BasicAgent = GaiaAgent
