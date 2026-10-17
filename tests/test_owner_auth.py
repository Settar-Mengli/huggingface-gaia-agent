"""Owner-only OAuth authorization for Gradio Local Test and submission."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

import app as app_module


def _oauth_profile(username: str) -> gr.OAuthProfile:
    return gr.OAuthProfile(
        {
            "name": username,
            "preferred_username": username,
            "profile": f"https://huggingface.co/{username}",
            "picture": "",
        }
    )


class AuthorizeOwnerHelperTests(unittest.TestCase):
    def test_anonymous_denied(self):
        ok, msg = app_module.authorize_owner(None)
        self.assertFalse(ok)
        self.assertIn("Access denied", msg)

    def test_other_user_denied(self):
        ok, msg = app_module.authorize_owner(_oauth_profile("someone-else"))
        self.assertFalse(ok)
        self.assertIn("Access denied", msg)

    def test_owner_authorized(self):
        ok, msg = app_module.authorize_owner(_oauth_profile("Smengli"))
        self.assertTrue(ok)
        self.assertEqual(msg, "Smengli")

    def test_plain_username_string_rejected(self):
        ok, msg = app_module.authorize_owner("Smengli")
        self.assertFalse(ok)
        self.assertIn("invalid oauth", msg.lower())

    def test_dict_spoof_rejected(self):
        ok, msg = app_module.authorize_owner(
            {"preferred_username": "Smengli", "username": "Smengli"}
        )
        self.assertFalse(ok)
        self.assertIn("invalid oauth", msg.lower())


class LocalTestAuthTests(unittest.TestCase):
    def test_anonymous_local_test_denied_no_agent(self):
        with patch("app.GaiaAgent") as mock_agent:
            with patch("app.requests.get") as mock_get:
                status, df = app_module.run_local_sample(1, profile=None)
        self.assertIn("Access denied", status)
        self.assertIsNone(df)
        mock_agent.assert_not_called()
        mock_get.assert_not_called()

    def test_other_user_local_test_denied_no_agent(self):
        with patch("app.GaiaAgent") as mock_agent:
            with patch("app.requests.get") as mock_get:
                status, df = app_module.run_local_sample(
                    1, profile=_oauth_profile("intruder")
                )
        self.assertIn("Access denied", status)
        self.assertIsNone(df)
        mock_agent.assert_not_called()
        mock_get.assert_not_called()

    def test_owner_local_test_authorized(self):
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
                    status, df = app_module.run_local_sample(
                        1, profile=_oauth_profile("Smengli")
                    )
                    mock_post.assert_not_called()
        self.assertIn("LOCAL TEST ONLY", status)
        self.assertIn("Smengli", status)
        mock_agent.assert_called_once()
        self.assertFalse(df.empty)

    def test_username_kwarg_cannot_bypass_auth(self):
        with self.assertRaises(TypeError):
            app_module.run_local_sample(1, username="Smengli")  # type: ignore[call-arg]

    def test_string_profile_arg_cannot_bypass(self):
        with patch("app.GaiaAgent") as mock_agent:
            with patch("app.requests.get") as mock_get:
                status, df = app_module.run_local_sample(1, profile="Smengli")  # type: ignore[arg-type]
        self.assertIn("Access denied", status)
        self.assertIsNone(df)
        mock_agent.assert_not_called()
        mock_get.assert_not_called()

    def test_space_local_test_count_capped_to_one(self):
        fake_q = {
            "task_id": "t1",
            "question": "q?",
            "file_name": None,
        }
        mock_agent = MagicMock(return_value="a")
        mock_get = MagicMock()
        mock_get.return_value.json.return_value = fake_q
        mock_get.return_value.raise_for_status = MagicMock()

        with patch.object(app_module, "_ON_SPACE", True):
            with patch("app.GaiaAgent", return_value=mock_agent):
                with patch("app.requests.get", mock_get) as get:
                    status, _df = app_module.run_local_sample(
                        5, profile=_oauth_profile("Smengli")
                    )
        self.assertIn("Ran agent on 1 question", status)
        get.assert_called_once()
        self.assertIn("/random-question", get.call_args.args[0])


class SubmitAuthTests(unittest.TestCase):
    def test_anonymous_submission_denied_no_agent(self):
        with patch("app.GaiaAgent") as mock_agent:
            with patch("app.requests.get") as mock_get:
                with patch("app.requests.post") as mock_post:
                    status, df = app_module.run_and_submit_all(profile=None)
        self.assertIn("Access denied", status)
        self.assertIsNone(df)
        mock_agent.assert_not_called()
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_other_user_submission_denied_no_agent(self):
        with patch("app.GaiaAgent") as mock_agent:
            with patch("app.requests.get") as mock_get:
                with patch("app.requests.post") as mock_post:
                    status, df = app_module.run_and_submit_all(
                        profile=_oauth_profile("not-owner")
                    )
        self.assertIn("Access denied", status)
        self.assertIsNone(df)
        mock_agent.assert_not_called()
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_owner_submission_authorized(self):
        questions = [
            {
                "task_id": "00000000-0000-0000-0000-000000000001",
                "question": "What is 2+2?",
                "file_name": None,
            }
        ]
        mock_agent = MagicMock(return_value="4")
        mock_get = MagicMock()
        mock_get.return_value.json.return_value = questions
        mock_get.return_value.raise_for_status = MagicMock()
        mock_post = MagicMock()
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "username": "Smengli",
            "score": 100,
            "correct_count": 1,
            "total_attempted": 1,
            "message": "ok",
        }

        with patch("app.GaiaAgent", return_value=mock_agent):
            with patch("app.requests.get", mock_get):
                with patch("app.requests.post", mock_post):
                    status, df = app_module.run_and_submit_all(
                        profile=_oauth_profile("Smengli")
                    )
        self.assertIn("Submission Successful", status)
        mock_agent.assert_called_once()
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.kwargs["json"]["username"], "Smengli")
        self.assertFalse(df.empty)

    def test_string_profile_cannot_authorize_submit(self):
        with patch("app.GaiaAgent") as mock_agent:
            with patch("app.requests.post") as mock_post:
                status, df = app_module.run_and_submit_all(profile="Smengli")  # type: ignore[arg-type]
        self.assertIn("Access denied", status)
        self.assertIsNone(df)
        mock_agent.assert_not_called()
        mock_post.assert_not_called()


class LocalTestCountHelperTests(unittest.TestCase):
    def test_helper_caps_on_space(self):
        with patch.object(app_module, "_ON_SPACE", True):
            self.assertEqual(app_module._local_test_question_count(20), 1)
            self.assertEqual(app_module._local_test_question_count("3"), 1)

    def test_helper_allows_multi_off_space(self):
        with patch.object(app_module, "_ON_SPACE", False):
            self.assertEqual(app_module._local_test_question_count(3), 3)


if __name__ == "__main__":
    unittest.main()
