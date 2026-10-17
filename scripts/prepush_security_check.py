#!/usr/bin/env python3
"""
Read-only pre-push / pre-commit security scan for portfolio safety.

- Inspects intended Git staging candidates (tracked + untracked, respecting .gitignore).
- Flags secret files, eval artifacts, and likely credential patterns.
- Prints PASS/FAIL and filenames only — never prints secret values.
- Does not stage, commit, push, delete, or modify repository files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Basename / path patterns that must never be committed.
FORBIDDEN_NAME_RES = [
    re.compile(r"(?i)^\.env$"),
    re.compile(r"(?i)^\.env\..+$"),
    re.compile(r"(?i)^.*credentials.*$"),
    re.compile(r"(?i)^.*secrets.*$"),
    re.compile(r"(?i)^.*\.pem$"),
    re.compile(r"(?i)^.*\.key$"),
    re.compile(r"(?i)^gaia_answers_cache\.json$"),
    re.compile(r"(?i)^questions_sample\.json$"),
    re.compile(r"(?i)^local_eval.*\.json$"),
    re.compile(r"(?i)^.*_eval_answers\.json$"),
    re.compile(r"(?i)^.*answers_cache.*$"),
    re.compile(r"(?i)^.*\.log$"),
    re.compile(r"(?i)^metadata\.jsonl$"),
]

FORBIDDEN_PATH_RES = [
    re.compile(r"(?i)(^|/)downloads(/|$)"),
    re.compile(r"(?i)(^|/)attachments(/|$)"),
    re.compile(r"(?i)(^|/)_deploy_stage"),
    re.compile(r"(?i)(^|/)\.venv(/|$)"),
    re.compile(r"(?i)(^|/)venv(/|$)"),
    re.compile(r"(?i)gaia-benchmark"),
    re.compile(r"(?i)(^|/)GAIA(/|$)"),
]

# Content patterns (values never printed).
CREDENTIAL_CONTENT_RES = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)api[_-]?key\s*=\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"(?i)HF_TOKEN\s*=\s*['\"]?hf_[A-Za-z0-9]{10,}"),
]

# Allowlisted exception: placeholder only in .env.example
PLACEHOLDER_TOKEN = "hf_your_token_here"

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".example",
    ".gitignore",
    ".gitattributes",
    "",
}


def _git_ls_candidates() -> list[str]:
    """
    Intended staging candidates: tracked files + untracked not ignored.
    Read-only: uses `git ls-files` and `git ls-files --others --exclude-standard`.
    """
    paths: set[str] = set()
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0 and tracked.stdout:
            paths.update(p for p in tracked.stdout.decode("utf-8", "replace").split("\0") if p)
        others = subprocess.run(
            ["git", "ls-files", "-z", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if others.returncode == 0 and others.stdout:
            paths.update(p for p in others.stdout.decode("utf-8", "replace").split("\0") if p)
    except FileNotFoundError:
        print("FAIL: git executable not found")
        return []
    # Fallback if repo has no commits yet and ls-files tracked is empty:
    # still scan untracked allowlisted-by-gitignore set (already in others).
    return sorted(paths)


def _is_forbidden_name(rel: str) -> bool:
    name = Path(rel).name
    posix = rel.replace("\\", "/")
    if name == ".env.example":
        return False
    for cre in FORBIDDEN_NAME_RES:
        if cre.search(name):
            return True
    for cre in FORBIDDEN_PATH_RES:
        if cre.search(posix):
            return True
    return False


def _scan_content(rel: str) -> bool:
    path = ROOT / rel
    if not path.is_file():
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
        "LICENSE",
        "Dockerfile",
        "Makefile",
    }:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for cre in CREDENTIAL_CONTENT_RES:
        for match in cre.finditer(text):
            snippet = match.group(0)
            if PLACEHOLDER_TOKEN in snippet or snippet.endswith(PLACEHOLDER_TOKEN):
                continue
            if "hf_your_token" in snippet.lower():
                continue
            return True
    return False


def main() -> int:
    print(f"Pre-push security check (read-only) — root: {ROOT.name}")
    candidates = _git_ls_candidates()
    if not candidates:
        # Empty candidate set with a git repo that has no files listed is unusual;
        # treat as FAIL so a misconfigured scan is not silently green.
        print("FAIL: no staging candidates discovered via git ls-files")
        return 1

    print(f"Candidates scanned: {len(candidates)}")
    forbidden_files: list[str] = []
    credential_files: list[str] = []

    for rel in candidates:
        if _is_forbidden_name(rel):
            forbidden_files.append(rel)
            continue
        if _scan_content(rel):
            credential_files.append(rel)

    if forbidden_files:
        print("FAIL: forbidden artifact / secret path candidates:")
        for name in forbidden_files:
            print(f"  - {name}")
    if credential_files:
        print("FAIL: likely credential pattern in file content (values not shown):")
        for name in credential_files:
            print(f"  - {name}")

    if forbidden_files or credential_files:
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
