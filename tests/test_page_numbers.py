"""Offline tests for spoken page-number extraction (ASR transcripts)."""

from __future__ import annotations

import unittest

from answer_utils import extract_spoken_page_numbers, format_spoken_page_numbers
from tools_custom import ExtractPageNumbersTool


# Pattern observed in the Homework.mp3 e2e run (no reference labels).
_OBSERVED_TRANSCRIPT = (
    "Before you all go, I want to remind you that the midterm is next week. "
    "Here's a little hint. You should be familiar with the differential equations "
    "on page 245. Problems that are very similar to problems 32, 33 and 44 from "
    "that page might be on the test. And also some of you might want to brush up "
    "on the last page in the integration section, page 197. I know some of you "
    "struggled on last week's quiz. I foresee problem 22 from page 197 being on "
    "your midterm. Oh, and don't forget to brush up on the section on related "
    "rates on pages 132, 133 and 134."
)


class ExtractSpokenPageNumbersTests(unittest.TestCase):
    def test_observed_transcript_includes_plural_list(self):
        pages = extract_spoken_page_numbers(_OBSERVED_TRANSCRIPT)
        self.assertEqual(pages, [132, 133, 134, 197, 245])
        self.assertEqual(format_spoken_page_numbers(_OBSERVED_TRANSCRIPT), "132,133,134,197,245")

    def test_naive_page_regex_would_miss_list_but_helper_does_not(self):
        import re

        naive = sorted({int(x) for x in re.findall(r"page\s+(\d+)", _OBSERVED_TRANSCRIPT)})
        self.assertEqual(naive, [197, 245])
        self.assertEqual(extract_spoken_page_numbers(_OBSERVED_TRANSCRIPT), [132, 133, 134, 197, 245])

    def test_page_lists_and_and(self):
        text = "Please review pages 10, 11 and 12 before class."
        self.assertEqual(extract_spoken_page_numbers(text), [10, 11, 12])

    def test_page_ranges_dash_and_to(self):
        self.assertEqual(
            extract_spoken_page_numbers("Skim pages 132-134 tonight."),
            [132, 133, 134],
        )
        self.assertEqual(
            extract_spoken_page_numbers("Skim pages 132 to 134 tonight."),
            [132, 133, 134],
        )

    def test_ignores_unrelated_problem_and_date_numbers(self):
        text = (
            "Homework 5 is due on March 15, 2023. Problems 32, 33 and 44 are hard. "
            "Course MATH 101 meets in room 204. No page mentions here."
        )
        self.assertEqual(extract_spoken_page_numbers(text), [])

    def test_ignores_problem_numbers_near_page_sentence(self):
        text = (
            "Study page 245. Problems that are similar to problems 32, 33 and 44 "
            "from that page might appear."
        )
        self.assertEqual(extract_spoken_page_numbers(text), [245])

    def test_empty_and_none_like(self):
        self.assertEqual(extract_spoken_page_numbers(""), [])
        self.assertEqual(format_spoken_page_numbers(""), "")


class ExtractPageNumbersToolTests(unittest.TestCase):
    def test_tool_formats_observed_transcript(self):
        tool = ExtractPageNumbersTool()
        self.assertEqual(
            tool.forward(_OBSERVED_TRANSCRIPT),
            "132,133,134,197,245",
        )

    def test_tool_error_when_none_found(self):
        tool = ExtractPageNumbersTool()
        out = tool.forward("Problems 32 and 33 are optional.")
        self.assertTrue(out.startswith("ERROR:"))


if __name__ == "__main__":
    unittest.main()
