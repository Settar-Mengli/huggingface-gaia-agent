"""Retry handling unit tests for GaiaAgent.__call__ (no LLM calls)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agent import GaiaAgent
from context_budget import TokenBudgetTracker


class GaiaAgentRetryTests(unittest.TestCase):
    def _make_agent(self, max_retries: int = 1) -> GaiaAgent:
        agent = GaiaAgent.__new__(GaiaAgent)
        agent._max_retries = max_retries
        agent.agent = MagicMock()
        agent.budget_tracker = TokenBudgetTracker()
        agent.visit_tool = MagicMock()
        agent.download_tool = MagicMock()
        agent.describe_tool = MagicMock()
        agent.describe_tool.vision_succeeded = True
        agent.describe_tool.reset_for_question = MagicMock()
        return agent

    def test_success_first_attempt(self):
        ga = self._make_agent(max_retries=1)
        ga.agent.run.return_value = "Answer: 42"
        out = ga("What is six*seven?")
        self.assertEqual(out, "42")
        self.assertEqual(ga.agent.run.call_count, 1)

    def test_permanent_error_does_not_retry_even_if_budget(self):
        ga = self._make_agent(max_retries=3)
        ga.agent.run.side_effect = RuntimeError("401 Unauthorized")
        with patch("agent.time.sleep") as sleep:
            out = ga("q")
            sleep.assert_not_called()
        self.assertEqual(out, "")
        self.assertEqual(ga.agent.run.call_count, 1)

    def test_transient_error_retries_when_allowed(self):
        ga = self._make_agent(max_retries=2)
        ga.agent.run.side_effect = [
            RuntimeError("429 rate limit"),
            "Paris",
        ]
        with patch("agent.time.sleep"):
            out = ga("capital?")
        self.assertEqual(out, "Paris")
        self.assertEqual(ga.agent.run.call_count, 2)

    def test_default_one_attempt_skips_transient_retry(self):
        ga = self._make_agent(max_retries=1)
        ga.agent.run.side_effect = RuntimeError("503 service unavailable")
        with patch("agent.time.sleep") as sleep:
            out = ga("q")
            sleep.assert_not_called()
        self.assertEqual(out, "")
        self.assertEqual(ga.agent.run.call_count, 1)

    def test_rejects_http_500_final_result(self):
        ga = self._make_agent(max_retries=1)
        ga.agent.run.return_value = (
            "Error in generating final LLM output: 500 Internal Server Error"
        )
        out = ga("olympics?")
        self.assertEqual(out, "")

    def test_missing_attachment_prompt_and_tool_config(self):
        ga = self._make_agent(max_retries=1)
        ga.agent.run.return_value = "42"
        out = ga("plain question", task_id="tid-1", file_name=None)
        self.assertEqual(out, "42")
        prompt = ga.agent.run.call_args.args[0]
        self.assertIn("No attachment is available", prompt)
        self.assertNotIn("download_gaia_file", prompt)
        ga.download_tool.configure_for_question.assert_called_with(None)
        ga.visit_tool.reset.assert_called()

    def test_attachment_prompt_when_file_present(self):
        ga = self._make_agent(max_retries=1)
        ga.agent.run.return_value = "1"
        out = ga("with file", task_id="tid-2", file_name="sheet.xlsx")
        self.assertEqual(out, "1")
        prompt = ga.agent.run.call_args.args[0]
        self.assertIn("file_name=sheet.xlsx", prompt)
        self.assertIn("download_gaia_file", prompt)
        ga.download_tool.configure_for_question.assert_called_with("sheet.xlsx")


if __name__ == "__main__":
    unittest.main()
