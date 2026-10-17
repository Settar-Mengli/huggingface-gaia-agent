"""Offline tests for prompt sanitization and executor state isolation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from smolagents.local_python_executor import LocalPythonExecutor

from agent import GAIA_SYSTEM_PROMPT, GaiaAgent
from prompt_config import effective_system_prompt_text, load_gaia_prompt_templates


class PromptSanitizationTests(unittest.TestCase):
    def test_einstein_ulam_exemplar_removed(self):
        text = effective_system_prompt_text()
        self.assertNotIn("Stanislaus Ulam", text)
        self.assertNotIn("Einstein learning too much math", text)
        self.assertNotIn('final_answer("diminished")', text)

    def test_neutral_reverse_example_present(self):
        text = effective_system_prompt_text()
        self.assertIn('final_answer("world")', text)
        self.assertIn("dlrow", text)

    def test_required_codeagent_instructions_remain(self):
        text = effective_system_prompt_text()
        for required in (
            "Thought:",
            "final_answer",
            "{{code_block_opening_tag}}",
            "{{code_block_closing_tag}}",
            "You are an expert assistant",
            "{{custom_instructions}}",
            "Observation:",
        ):
            self.assertIn(required, text, msg=f"missing {required!r}")

    def test_final_answer_prompt_is_concise_only(self):
        templates = load_gaia_prompt_templates()
        post = templates["final_answer"]["post_messages"]
        pre = templates["final_answer"]["pre_messages"]
        self.assertIn("exact answer", post.lower())
        self.assertIn("Do not perform new research", post)
        self.assertIn("serialize tool calls", post)
        self.assertIn("Markdown", post)
        self.assertIn("observations already present", pre.lower())

    def test_gaia_instructions_constant_intact(self):
        self.assertIn("exact match scoring", GAIA_SYSTEM_PROMPT)
        self.assertIn("final_answer", GAIA_SYSTEM_PROMPT)
        self.assertIn("wikipedia_search", GAIA_SYSTEM_PROMPT)


class ExecutorIsolationTests(unittest.TestCase):
    def test_reset_clears_variables_preserves_tools(self):
        ga = GaiaAgent.__new__(GaiaAgent)
        executor = LocalPythonExecutor(additional_authorized_imports=["math"])

        def _final_answer(x):
            return x

        tools = {"final_answer": _final_answer}
        executor.send_tools(tools)
        executor.state["leaked_var"] = 99
        executor.custom_tools["helper"] = lambda: 1

        agent = MagicMock()
        agent.state = {"leaked_var": 99}
        agent.tools = tools
        agent.managed_agents = {}
        agent.python_executor = executor
        ga.agent = agent

        ga._reset_executor_state()

        self.assertEqual(agent.state, {})
        self.assertNotIn("leaked_var", executor.state)
        self.assertEqual(executor.state.get("__name__"), "__main__")
        self.assertNotIn("helper", executor.custom_tools)
        self.assertIsNotNone(executor.static_tools)
        self.assertIn("final_answer", executor.static_tools)


if __name__ == "__main__":
    unittest.main()
