"""Ensure local evaluation paths cannot submit to the leaderboard."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import MagicMock, patch

import app as app_module


class SubmissionSafetyTests(unittest.TestCase):
    def test_local_sample_source_has_no_submit_call(self):
        source = inspect.getsource(app_module.run_local_sample)
        # Docstring may mention /submit; executable body must not POST.
        self.assertNotIn("requests.post", source)
        self.assertIn("Does NOT call /submit", source)
        body = source.split('"""', 2)[-1]
        self.assertNotIn("/submit", body)
        self.assertNotIn("submit_url", body)

    def test_local_eval_script_documents_no_submit(self):
        import run_local_eval

        source = inspect.getsource(run_local_eval)
        self.assertIn("Does NOT call /submit", source)
        # POST must not appear outside comments for scoring submit.
        self.assertNotIn('"/submit"', source)
        self.assertNotIn("'/submit'", source)

    def test_run_local_sample_never_posts(self):
        import gradio as gr

        profile = gr.OAuthProfile(
            {
                "name": "Smengli",
                "preferred_username": "Smengli",
                "profile": "https://huggingface.co/Smengli",
                "picture": "",
            }
        )
        fake_question = {
            "task_id": "00000000-0000-0000-0000-000000000001",
            "question": "What is 2+2?",
            "file_name": None,
        }
        mock_agent = MagicMock(return_value="4")
        mock_get = MagicMock()
        mock_get.return_value.json.return_value = fake_question
        mock_get.return_value.raise_for_status = MagicMock()

        with patch("app.GaiaAgent", return_value=mock_agent):
            with patch("app.requests.get", mock_get):
                with patch("app.requests.post") as mock_post:
                    status, df = app_module.run_local_sample(1, profile=profile)
                    mock_post.assert_not_called()
                    self.assertIn("LOCAL TEST ONLY", status)
                    self.assertIn("nothing submitted", status.lower())
                    mock_agent.assert_called_once()
                    self.assertFalse(df.empty)

    def test_submit_path_uses_agent_answers(self):
        source = inspect.getsource(app_module.run_and_submit_all)
        self.assertIn("/submit", source)
        self.assertIn("_run_agent_on_questions", source)
        self.assertIn("answers_payload", source)


class AgentRetryDefaultTests(unittest.TestCase):
    def test_max_steps_and_retries_are_env_configurable(self):
        # Import without constructing full agent (would need model).
        from agent import GaiaAgent

        init_source = inspect.getsource(GaiaAgent.__init__)
        self.assertIn('os.getenv("GAIA_MAX_RETRIES", "1")', init_source)
        # Default comes from env fallback string only — not a hardcoded runtime constant.
        self.assertIn('os.getenv("GAIA_MAX_STEPS", "8")', init_source)

        import context_budget

        budget_source = inspect.getsource(context_budget.TokenBudgetTracker.__init__)
        self.assertIn('os.getenv("GAIA_MAX_CUMULATIVE_INPUT_TOKENS", "100000")', budget_source)
        self.assertIn('os.getenv("GAIA_MAX_REQUEST_INPUT_TOKENS", "24000")', budget_source)



if __name__ == "__main__":
    unittest.main()
