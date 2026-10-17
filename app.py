import os

from console_utf8 import configure_utf8_stdio

configure_utf8_stdio()

import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv

from agent import GaiaAgent

load_dotenv()

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = os.getenv(
    "GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"
)
ALLOWED_HF_USER = os.getenv("ALLOWED_HF_USER", "Smengli").strip() or "Smengli"
_ON_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def authorize_owner(profile: object | None) -> tuple[bool, str]:
    """
    Server-side owner gate for paid inference / official submission.

    Requires a real gr.OAuthProfile whose username matches ALLOWED_HF_USER.
    Fails closed for missing OAuth, wrong accounts, and non-profile spoof values
    (e.g. a plain username string). Returns (ok, message_or_username).
    """
    if profile is None:
        return False, (
            f"Access denied: log in to Hugging Face as '{ALLOWED_HF_USER}' "
            "before running Local Test or submission."
        )
    if not isinstance(profile, gr.OAuthProfile):
        return False, "Access denied: invalid OAuth session."
    username = getattr(profile, "username", None)
    if not isinstance(username, str) or not username.strip():
        return False, "Access denied: missing OAuth username."
    username = username.strip()
    if username != ALLOWED_HF_USER:
        return False, (
            f"Access denied: only Hugging Face user '{ALLOWED_HF_USER}' "
            "may run inference or submit from this Space."
        )
    return True, username


def _run_agent_on_questions(agent: GaiaAgent, questions_data: list) -> tuple[list, list]:
    """Run the agent on a list of question dicts. Does not submit."""
    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or None
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(
                question_text, task_id=task_id, file_name=file_name
            )
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "File": file_name or "",
                    "Submitted Answer": submitted_answer,
                }
            )
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "File": file_name or "",
                    "Submitted Answer": f"AGENT ERROR: {e}",
                }
            )
    return answers_payload, results_log


def _local_test_question_count(n: object) -> int:
    """Normalize local-test count; on Spaces hard-cap to one question."""
    try:
        count = max(1, int(n))
    except (TypeError, ValueError):
        count = 1
    if _ON_SPACE:
        return 1
    return count


def run_local_sample(
    n: int = 1,
    profile: gr.OAuthProfile | None = None,
):
    """
    Safe local test mode: fetches question(s) and runs the agent.
    Does NOT call /submit or update the leaderboard.
    Owner OAuth required before any agent construction or inference.
    """
    ok, auth_msg = authorize_owner(profile)
    if not ok:
        return auth_msg, None

    n = _local_test_question_count(n)
    api_url = DEFAULT_API_URL

    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    questions_data = []
    try:
        if n == 1:
            response = requests.get(f"{api_url}/random-question", timeout=30)
            response.raise_for_status()
            questions_data = [response.json()]
        else:
            response = requests.get(f"{api_url}/questions", timeout=30)
            response.raise_for_status()
            all_q = response.json()
            questions_data = all_q[:n]
    except Exception as e:
        return f"Error fetching question(s): {e}", None

    answers_payload, results_log = _run_agent_on_questions(agent, questions_data)
    status = (
        f"LOCAL TEST ONLY — nothing submitted.\n"
        f"Authorized user: {auth_msg}\n"
        f"Ran agent on {len(answers_payload)} question(s)."
    )
    return status, pd.DataFrame(results_log)


def run_and_submit_all(profile: gr.OAuthProfile | None = None):
    """
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
    and displays the results. Owner OAuth required before agent / inference.
    """
    ok, auth_msg = authorize_owner(profile)
    if not ok:
        return auth_msg, None
    username = auth_msg

    space_id = os.getenv("SPACE_ID")
    print(f"User authorized: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (only after owner auth)
    try:
        agent = GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    print(f"Running agent on {len(questions_data)} questions...")
    answers_payload, results_log = _run_agent_on_questions(agent, questions_data)

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = (
        f"Agent finished. Submitting {len(answers_payload)} answers "
        f"for user '{username}'..."
    )
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
# LoginButton enables OAuth middleware. Outside Spaces, Gradio requires HF_TOKEN /
# huggingface-cli login just to *start* the app with OAuth UI.
_HAS_HF_TOKEN = bool(
    os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
_ENABLE_OAUTH_UI = _ON_SPACE or _HAS_HF_TOKEN

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        f"""
**Instructions:**

1. Clone / duplicate this Space, then set secrets (`HF_TOKEN`, optional model overrides).
2. Log in to Hugging Face as **{ALLOWED_HF_USER}** (owner-only inference and submission).
3. Use **Local Test (no submit)** first to verify the agent (owner login required).
4. Click **Run Evaluation & Submit All Answers** only when you are ready to post a leaderboard score.

---
**Disclaimers:**
Only the authorized Hugging Face account may trigger paid inference or `/submit`.
Submission can take a long time while the agent answers all 20 questions.
Keep the Space **public** so your `agent_code` link can be reviewed.
        """
    )

    if _ENABLE_OAUTH_UI:
        gr.LoginButton()
    else:
        gr.Markdown(
            "_Local mode: HF login button hidden until `HF_TOKEN` is set "
            "(or the app runs on a Space). Owner OAuth is still required for "
            "Local Test and submission once the UI is available._"
        )

    with gr.Row():
        local_n = gr.Number(
            value=1,
            precision=0,
            minimum=1,
            maximum=1 if _ON_SPACE else 20,
            label=(
                "Local test question count (Space: always 1)"
                if _ON_SPACE
                else "Local test question count (1 = random-question)"
            ),
        )
        local_button = gr.Button("Local Test (no submit)")
        run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    local_button.click(
        fn=run_local_sample,
        inputs=[local_n],
        outputs=[status_output, results_table],
        api_name=False,
    )
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
        api_name=False,
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️ SPACE_ID environment variable not found (running locally?). "
            "Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    # Public deployment must not enable Gradio debug mode.
    demo.launch(debug=False, share=False)
