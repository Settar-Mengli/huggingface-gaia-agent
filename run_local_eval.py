"""
Local dry-run for GAIA Unit 4 questions.

Fetches questions from the official scoring API, runs GaiaAgent, writes answers
to local_eval_answers.json. Does NOT call /submit.

Usage:
  .\\.venv\\Scripts\\python.exe run_local_eval.py              # all 20
  .\\.venv\\Scripts\\python.exe run_local_eval.py --limit 1    # one random
  .\\.venv\\Scripts\\python.exe run_local_eval.py --limit 3    # first 3 from /questions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from console_utf8 import configure_utf8_stdio

configure_utf8_stdio()

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = os.getenv(
    "GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local GAIA agent dry-run (no submit)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all questions; 1 = one random; N>1 = first N from /questions",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="local_eval_answers.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print(
            "ERROR: Set HF_TOKEN in .env (Hugging Face access token with Inference permission)."
        )
        return 1
    token = token.strip().strip('"').strip("'")
    if not token.startswith("hf_"):
        print("ERROR: HF_TOKEN should start with 'hf_'. Check .env formatting.")
        return 1

    # Fail fast on bad tokens (never print the token value).
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=token).whoami()
        print(f"HF auth OK for user: {who.get('name', '?')}")
    except Exception as exc:
        print(
            "ERROR: HF_TOKEN was loaded from .env but Hugging Face rejected it "
            "(invalid / revoked / truncated). Create a new token at "
            "https://huggingface.co/settings/tokens with at least Inference "
            "permission, paste it into .env as HF_TOKEN=..., save, and retry."
        )
        print(f"Auth detail: {type(exc).__name__}")
        return 1

    from agent import GaiaAgent

    api = DEFAULT_API_URL.rstrip("/")
    if args.limit == 1:
        resp = requests.get(f"{api}/random-question", timeout=30)
        resp.raise_for_status()
        questions = [resp.json()]
    else:
        resp = requests.get(f"{api}/questions", timeout=30)
        resp.raise_for_status()
        questions = resp.json()
        if args.limit and args.limit > 1:
            questions = questions[: args.limit]

    print(f"Running GaiaAgent on {len(questions)} question(s). No submission will be made.")
    agent = GaiaAgent()
    answers = []
    for i, item in enumerate(questions, start=1):
        task_id = item["task_id"]
        question = item["question"]
        file_name = item.get("file_name") or None
        print(f"\n=== [{i}/{len(questions)}] task_id={task_id} file={file_name} ===")
        print(question[:200].replace("\n", " "))
        try:
            answer = agent(question, task_id=task_id, file_name=file_name)
        except Exception as exc:
            answer = ""
            print(f"ERROR: {exc}")
        print(f"ANSWER: {answer!r}")
        answers.append(
            {
                "task_id": task_id,
                "file_name": file_name,
                "question": question,
                "submitted_answer": answer,
            }
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(answers)} answers to {out_path.resolve()}")
    print("LOCAL TEST ONLY — leaderboard was NOT updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
