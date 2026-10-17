"""Unit tests for answer normalization and retry helpers."""

from __future__ import annotations

import unittest

from answer_utils import (
    clean_answer,
    extract_answer_text,
    is_error_like_answer,
    is_transient_error,
    is_unusable_answer,
    should_retry,
)


class CleanAnswerTests(unittest.TestCase):
    def test_strips_final_answer_prefix(self):
        self.assertEqual(clean_answer("FINAL ANSWER: Paris"), "Paris")
        self.assertEqual(clean_answer("Answer: 42"), "42")

    def test_preserves_multiword_and_lists(self):
        self.assertEqual(
            clean_answer("ripe strawberries, sugar, salt"),
            "ripe strawberries, sugar, salt",
        )
        self.assertEqual(clean_answer("New York"), "New York")

    def test_preserves_numbers_and_decimals(self):
        self.assertEqual(clean_answer("123.45"), "123.45")
        self.assertEqual(clean_answer("$89.67"), "$89.67")

    def test_preserves_dates(self):
        self.assertEqual(clean_answer("June 6, 2023"), "June 6, 2023")

    def test_unwraps_verbose_prose_nominator(self):
        verbose = (
            "The user who nominated the only Featured Article on English Wikipedia "
            "about a dinosaur (Allosaurus) that was promoted in November 2016 was User:KSmrq."
        )
        self.assertEqual(clean_answer(verbose), "KSmrq")

    def test_takes_last_line_of_explanation(self):
        raw = "I think it is Paris.\nParis"
        self.assertEqual(clean_answer(raw), "Paris")

    def test_none_and_empty(self):
        self.assertEqual(clean_answer(None), "")
        self.assertEqual(clean_answer(""), "")
        self.assertEqual(clean_answer("   "), "")

    def test_rejects_http_500_string(self):
        payload = (
            "Error in generating final LLM output: "
            "HfHubHTTPError: 500 Server Error: Internal Server Error "
            "for url: https://router.huggingface.co/v1/chat/completions"
        )
        self.assertTrue(is_error_like_answer(payload))
        self.assertEqual(clean_answer(payload), "")

    def test_rejects_chatmessage_error_blocks(self):
        class FakeMsg:
            content = [
                {
                    "type": "text",
                    "text": "Error in generating final LLM output: HTTP 500 Internal Server Error",
                }
            ]

        self.assertTrue(is_error_like_answer(FakeMsg()))
        self.assertEqual(clean_answer(FakeMsg()), "")
        self.assertIn("HTTP 500", extract_answer_text(FakeMsg()))

    def test_legitimate_short_answers_not_error_like(self):
        for value in ("42", "Paris", "June 6, 2023", "a, b, c", "500"):
            self.assertFalse(is_error_like_answer(value), value)
            self.assertEqual(clean_answer(value), value)

    def test_observed_batch_outputs(self):
        # Mercedes Sosa max-steps prose / markdown finalization.
        self.assertEqual(
            clean_answer(
                "Therefore, the number of studio albums published by Mercedes "
                "Sosa between 2000 and 2009 is **3**."
            ),
            "3",
        )
        # YouTube bird count — already atomic.
        self.assertEqual(clean_answer("3"), "3")
        # Reversed-text puzzle.
        self.assertEqual(clean_answer("right"), "right")

    def test_markdown_wrapped_atomic(self):
        self.assertEqual(clean_answer("**3**"), "3")
        self.assertEqual(clean_answer("**Paris**"), "Paris")

    def test_does_not_blindly_take_last_number(self):
        ambiguous = "There were 3 albums in 2000 and 4 albums in 2009."
        # No clear trailing "is <atomic>" — leave text intact (after light trim).
        self.assertEqual(clean_answer(ambiguous), ambiguous.rstrip("."))

    def test_preserves_coordinates_and_lists(self):
        self.assertEqual(clean_answer("12.3, 45.6"), "12.3, 45.6")
        self.assertEqual(
            clean_answer("ripe strawberries, sugar, salt"),
            "ripe strawberries, sugar, salt",
        )

    def test_trailing_is_name_still_works(self):
        self.assertEqual(
            clean_answer(
                "After reviewing the sources carefully, the capital city is Paris."
            ),
            "Paris",
        )

    def test_extracts_verbose_two_nnx_and(self):
        self.assertEqual(
            clean_answer(
                "Therefore, based on the available information from the English "
                "Wikipedia page, Mercedes Sosa published **at least one** studio "
                "album between 2000 and 2009. If we assume both 2003 and 2006 "
                "releases are studio albums, the count would be **two**."
            ),
            "two",
        )
        self.assertEqual(
            clean_answer(
                "Therefore, the work performed by R. G. Arendt was supported by "
                "NASA under award number **NNX17AI56G**."
            ),
            "NNX17AI56G",
        )
        self.assertEqual(
            clean_answer(
                "Therefore, the country with the least number of athletes at the "
                "1928 Summer Olympics, based on the provided data, is Andorra "
                "with the IOC country code **AND**."
            ),
            "AND",
        )

    def test_rejects_tool_call_dump(self):
        dump = (
            "[{'id': 'call_9', 'type': 'function', 'function': "
            "{'name': 'python_interpreter', 'arguments': "
            "'dinosaur_fa_title_query = \"exact name\"\\n"
            "dinosaur_fa_title_info = web_search(query=dinosaur_fa_title_query)"
            "\\nprint(dinosaur_fa_title_info)'}}]"
        )
        self.assertTrue(is_unusable_answer(dump))
        self.assertEqual(clean_answer(dump), "")

    def test_rejects_hedge_only_non_answer(self):
        hedge = (
            "Please note that this would be best to verify with additional "
            "sources. I am not sure and cannot be sure without more evidence."
        )
        self.assertTrue(is_unusable_answer(hedge))
        self.assertEqual(clean_answer(hedge), "")

    def test_preserves_currency_and_multi_item_lists(self):
        self.assertEqual(clean_answer("$89.67"), "$89.67")
        self.assertEqual(
            clean_answer("apples, oranges, bananas"),
            "apples, oranges, bananas",
        )


class RetryTests(unittest.TestCase):
    def test_transient_detected(self):
        self.assertTrue(is_transient_error("429 rate limit"))
        self.assertTrue(is_transient_error("Gateway Timeout 504"))
        self.assertTrue(is_transient_error(TimeoutError("timed out")))

    def test_permanent_not_retried(self):
        self.assertFalse(is_transient_error("401 Unauthorized"))
        self.assertFalse(is_transient_error("402 Payment Required"))
        self.assertFalse(is_transient_error("Invalid user token"))
        self.assertFalse(is_transient_error("no remaining credits"))
        self.assertFalse(is_transient_error("Context budget exceeded: per-request"))

    def test_should_retry_respects_bounds(self):
        self.assertFalse(should_retry(1, 1, "429 rate limit"))
        self.assertTrue(should_retry(1, 2, "429 rate limit"))
        self.assertFalse(should_retry(1, 3, "401 Unauthorized"))
        self.assertFalse(should_retry(2, 2, "503 service unavailable"))


if __name__ == "__main__":
    unittest.main()
