"""Token budget guards (mocked; no LLM / no network)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from context_budget import (
    BudgetGuardedModel,
    ContextBudgetExceeded,
    TokenBudgetTracker,
    estimate_messages_tokens,
)
from smolagents.models import ChatMessage, MessageRole, TokenUsage


class EstimateTests(unittest.TestCase):
    def test_estimate_from_string_content_blocks(self):
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "abcd" * 100}],
            )
        ]
        est = estimate_messages_tokens(messages)
        self.assertGreaterEqual(est, 100)


class BudgetGuardedModelTests(unittest.TestCase):
    def test_blocks_oversized_request_before_generate(self):
        tracker = TokenBudgetTracker(
            max_cumulative_input_tokens=100000,
            max_request_input_tokens=50,
        )
        inner = MagicMock()
        model = BudgetGuardedModel(inner, tracker)
        huge = "x" * 1000
        messages = [ChatMessage(role=MessageRole.USER, content=huge)]
        with self.assertRaises(ContextBudgetExceeded):
            model.generate(messages)
        inner.generate.assert_not_called()

    def test_blocks_when_cumulative_would_exceed(self):
        tracker = TokenBudgetTracker(
            max_cumulative_input_tokens=100,
            max_request_input_tokens=1000,
        )
        tracker.cumulative_input_tokens = 90
        inner = MagicMock()
        model = BudgetGuardedModel(inner, tracker)
        messages = [ChatMessage(role=MessageRole.USER, content="y" * 200)]
        with self.assertRaises(ContextBudgetExceeded):
            model.generate(messages)
        inner.generate.assert_not_called()

    def test_records_actual_token_usage(self):
        tracker = TokenBudgetTracker(
            max_cumulative_input_tokens=100000,
            max_request_input_tokens=100000,
        )
        inner = MagicMock()
        inner.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="ok",
            token_usage=TokenUsage(input_tokens=12, output_tokens=3),
        )
        model = BudgetGuardedModel(inner, tracker)
        messages = [ChatMessage(role=MessageRole.USER, content="hi")]
        model.generate(messages)
        self.assertEqual(tracker.cumulative_input_tokens, 12)


class FinalizationBudgetTests(unittest.TestCase):
    def test_provide_final_answer_blocked_before_generate(self):
        from agent import BudgetAwareCodeAgent

        tracker = TokenBudgetTracker(
            max_cumulative_input_tokens=100000,
            max_request_input_tokens=40,
        )
        inner = MagicMock()
        guarded = BudgetGuardedModel(inner, tracker)

        agent = BudgetAwareCodeAgent.__new__(BudgetAwareCodeAgent)
        agent.model = guarded
        agent.prompt_templates = {
            "final_answer": {
                "pre_messages": "System final.",
                "post_messages": "Task: {{task}}",
            }
        }
        agent.write_memory_to_messages = MagicMock(
            return_value=[
                ChatMessage(role=MessageRole.SYSTEM, content="sys"),
                ChatMessage(
                    role=MessageRole.USER,
                    content="memory " + ("z" * 500),
                ),
            ]
        )

        with self.assertRaises(ContextBudgetExceeded):
            agent.provide_final_answer("solve this")
        inner.generate.assert_not_called()

    def test_gaia_agent_returns_empty_on_budget_exceeded(self):
        from agent import GaiaAgent

        ga = GaiaAgent.__new__(GaiaAgent)
        ga._max_retries = 2
        ga.budget_tracker = TokenBudgetTracker()
        ga.visit_tool = MagicMock()
        ga.download_tool = MagicMock()
        ga.describe_tool = MagicMock()
        ga.describe_tool.vision_succeeded = True
        ga.describe_tool.reset_for_question = MagicMock()
        ga.agent = MagicMock()
        ga.agent.run.side_effect = ContextBudgetExceeded("Context budget exceeded")

        out = ga("question?", task_id="t1", file_name=None)
        self.assertEqual(out, "")
        # Budget errors are non-transient: single attempt.
        self.assertEqual(ga.agent.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
