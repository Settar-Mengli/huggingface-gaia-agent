# Deploy to a public Hugging Face Space

Do **not** upload, push, enable paid hardware, or submit the leaderboard until you explicitly approve those steps.

Target Space (already created): **Smengli/gaia-certificate-agent**

## 1. Space settings (pre-upload)

1. SDK: **Gradio** (matches `README.md` front matter)
2. Hardware: **CPU basic** unless you explicitly approve paid hardware
3. Visibility: **Public** (required so `agent_code` / `…/tree/main` is reviewable)
4. Secrets (Settings → Variables and secrets) — set **before** first useful run:

| Secret / variable | Notes |
|-------------------|--------|
| `HF_TOKEN` | Required. Inference access; also GAIA Hub access if fallback enabled |
| `GAIA_ALLOW_HUB_FALLBACK` | Optional; `1` only after accepting `gaia-benchmark/GAIA` terms |
| `GAIA_VISION_MODEL` / `GAIA_VISION_PROVIDER` | Optional; defaults already in code |
| `GAIA_MAX_STEPS` / `GAIA_MAX_RETRIES` / budgets | Optional; leave unset to keep local defaults |

Do **not** put tokens in uploaded files.

## 2. Deployment allowlist (upload these only)

### Required runtime (Space must run)

| File | Role |
|------|------|
| `app.py` | Gradio UI + local dry-run + official submit entrypoint |
| `agent.py` | `GaiaAgent` / `BudgetAwareCodeAgent` |
| `tools_custom.py` | Download, visit/wiki guards, ASR, YouTube, vision |
| `answer_utils.py` | Answer clean / unusable-answer rejection |
| `prompt_config.py` | Sanitized smolagents prompt templates |
| `context_budget.py` | Dual token-budget guards |
| `console_utf8.py` | UTF-8 stdio helper (imported by `app.py` / `agent.py`) |
| `requirements.txt` | Dependencies |
| `README.md` | Gradio Space metadata (`sdk`, `app_file`, `hf_oauth`) |

### Optional (safe, not required to boot)

| File | Role |
|------|------|
| `.env.example` | Documents env vars (placeholder token only) |
| `DEPLOY.md` | This checklist |
| `run_local_eval.py` | Local CLI dry-run only (unused by Space UI) |
| `tests/` | Offline unit tests (not imported at runtime) |
| `.gitattributes` | HF LFS defaults (harmless) |
| `.gitignore` | Keeps local secrets/artifacts out of git |

## 3. Deny list (never upload)

- `.env` and any real tokens / credentials
- `.venv/`, `venv/`, `__pycache__/`
- `downloads/`, `attachments/`
- `*.log` (e.g. `local_eval_20_run.log`, `local_eval_regress3_run.log`)
- Local eval answer artifacts: `local_eval_answers.json`, `local_eval_regress3_answers.json`, `local_eval*.json`, `*_eval_answers.json`, `gaia_answers_cache.json`, `questions_sample.json`
- Any file containing a live `HF_TOKEN` value

## 4. Safe upload plan (execute only after approval)

1. Confirm Space `Smengli/gaia-certificate-agent` exists and is Gradio + public.
2. Set Space secret `HF_TOKEN` (do not commit it).
3. Upload **only** the allowlist files (web UI or `git push` to the Space remote) — start with the **Required runtime** set.
4. Wait for Space build; open the Space URL.
5. Use **Local Test (no submit)** once to confirm boot.
6. Official **Run Evaluation & Submit** only after separate explicit approval.

Suggested CLI (do **not** run until approved):

```powershell
# After adding the Space as a git remote named `space`:
# git add <allowlist files only>
# git commit -m "Deploy GAIA certificate agent runtime"
# git push space main
```

Or use the Hugging Face Space **Files** UI to upload the required runtime files only.

## 5. Verify after upload (no submit until approved)

1. Space builds without missing-module errors (`prompt_config`, `answer_utils`, `context_budget`, `console_utf8`).
2. Click **Local Test (no submit)**.
3. Confirm no leaderboard change.
4. Only after your approval: log in and **Run Evaluation & Submit All Answers**.

## 6. Certificate target

Need ≥ **30%** (6/20) on:
`https://agents-course-unit4-scoring.hf.space`
