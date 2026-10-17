---
title: GAIA Certificate Agent
emoji: 🕵️‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Certificate Agent

Tool-using GAIA Level-1 agent built for the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) Unit 4 final assignment.

## Official results

| Item | Value |
|------|-------|
| Benchmark | GAIA Level 1 (official course scoring API) |
| Score | **45%** (**9 / 20**) |
| Certificate threshold | 30% |
| Offline unit tests at certification | **106** passing |

Companion Hugging Face Space (currently **paused**, not serving traffic):

- [Smengli/gaia-certificate-agent](https://huggingface.co/spaces/Smengli/gaia-certificate-agent)

Official evaluation requires authorized Hugging Face access and **can incur inference costs**. Do not run a full evaluation unless you intend to spend credits.

## What this project is (and is not)

| This repo contains | This repo does **not** contain |
|--------------------|--------------------------------|
| Original application code (agent, tools, Gradio app, budgets, tests) | GAIA questions, attachments, reference answers, or grading labels |
| Configuration templates (`.env.example`) | Live API tokens or Space secrets |
| Offline unit tests | A redistributable copy of the GAIA benchmark dataset |

Third-party libraries (for example **smolagents**, **Gradio**, **Hugging Face Hub**) remain under their own licenses. GAIA benchmark materials remain under the terms of the [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) dataset and the Agents Course scoring service. This project's original code is released under the [MIT License](LICENSE).

## Architecture

```text
Gradio UI (app.py)
  └─ GaiaAgent (agent.py)
       ├─ smolagents CodeAgent + BudgetAwareCodeAgent
       ├─ BudgetGuardedModel → HF Inference (default: Qwen/Qwen2.5-Coder-32B-Instruct)
       ├─ Tools (tools_custom.py)
       │    web search · safe visit/wiki · download attachments
       │    YouTube transcript · Whisper ASR · vision describe
       │    spoken page-number extraction
       ├─ Prompt templates (prompt_config.py) — sanitized few-shots
       ├─ Answer normalization (answer_utils.py)
       └─ Dual token budgets (context_budget.py)
```

Highlights:

- **smolagents `CodeAgent`** plans in short Thought/Code/Observation loops and calls tools from Python.
- **Default text model:** `Qwen/Qwen2.5-Coder-32B-Instruct` via Hugging Face Inference Providers (provider auto unless overridden).
- **Attachments:** scoring-API download with optional authorized Hub fallback; shared `last_saved_path` for vision/ASR.
- **Answer hygiene:** strip prefixes, reject tool-call dumps / hedge-only text, extract clear trailing atomics.
- **Budgets:** per-request and per-question cumulative input-token caps; env-configurable `max_steps` / retries.

## Security design

- **Owner-only OAuth:** Local Test and official `/submit` require a valid `gr.OAuthProfile` whose username matches `ALLOWED_HF_USER` (default `Smengli`). Anonymous and other accounts are denied before `GaiaAgent` is constructed.
- **Secrets:** tokens live in local `.env` or Space secrets — never in source. `.env` is gitignored.
- **Inference limits:** dual token budgets, default single attempt (`GAIA_MAX_RETRIES=1`), Space Local Test hard-capped to one question.
- **API surface:** Gradio event endpoints use `api_name=False`; authorization still applies if called directly.
- **Debug:** `demo.launch(debug=False)` for deployment.

See [DEPLOY.md](DEPLOY.md) for the Space allowlist, deny list, and upload checklist.

## Local setup

Clone the repository, then create a virtual environment and install dependencies.

### Windows (PowerShell)

```powershell
cd huggingface-gaia-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cd huggingface-gaia-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set a Hugging Face token with Inference permission:

```env
HF_TOKEN=hf_your_token_here
```

Optional variables are documented in `.env.example` (vision model/provider, Hub fallback, steps, budgets). Defaults match the certificate run unless you change them.

### Offline unit tests (no network / no paid inference)

```bash
python -m unittest discover -s tests -v
```

### Local dry-run (uses inference credits; never submits)

```bash
python run_local_eval.py --limit 1
python app.py
```

In Gradio, prefer **Local Test (no submit)** before any official evaluation. Owner OAuth login is required for Local Test and submission when OAuth UI is enabled.

### Pre-push security check (read-only)

```bash
python scripts/prepush_security_check.py
```

Reports PASS/FAIL and filenames only. Does not stage, commit, push, or modify files.

## Repository layout

| Path | Role |
|------|------|
| `app.py` | Gradio UI: local dry-run + official evaluate/submit |
| `agent.py` | `GaiaAgent` / budget-aware CodeAgent |
| `tools_custom.py` | Custom tools (download, ASR, vision, wiki/visit guards, page extract) |
| `answer_utils.py` | Answer cleaning / unusable-answer rejection / page extract helpers |
| `prompt_config.py` | Sanitized CodeAgent prompt templates |
| `context_budget.py` | Dual token-budget guards |
| `console_utf8.py` | UTF-8 console helpers |
| `run_local_eval.py` | CLI dry-run (never calls `/submit`) |
| `scripts/prepush_security_check.py` | Offline secret / artifact scan |
| `tests/` | Offline unit tests |
| `requirements.txt` | Python dependencies |
| `DEPLOY.md` | Hugging Face Space deployment checklist |
| `.env.example` | Safe env template (placeholder token only) |
| `LICENSE` | MIT license for original project code |

## Limitations

- **Imperfect accuracy:** 45% on Level 1 means many tasks remain unsolved; exact-match scoring is strict about format.
- **Inference cost:** Local dry-runs, Space Local Test, and full evaluation call paid or credit-limited inference providers.
- **Model-generated code risk:** The CodeAgent generates and executes Python. This introduces untrusted-code execution risks, including unintended file operations, network requests, or other actions permitted by its runtime. Run the agent only in an appropriately restricted, isolated environment with limited credentials and permissions. Generated code can also produce incorrect answers or waste inference resources.
- **GAIA data restrictions:** This repository does not redistribute benchmark questions, attachments, or reference answers. Access to gated dataset fallback requires accepting Hub terms separately.

## Official APIs (course scoring)

- Questions: `GET https://agents-course-unit4-scoring.hf.space/questions`
- Random: `GET …/random-question`
- Files: `GET …/files/{task_id}`
- Submit: `POST …/submit` (exact-match scoring; certificate threshold ≥30%)

Running full evaluation or Space inference costs credits and should only be done with an account you control.

## Acknowledgments

- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
- [smolagents](https://github.com/huggingface/smolagents)
- [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) (dataset terms apply; not redistributed here)
