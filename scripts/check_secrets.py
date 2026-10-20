#!/usr/bin/env python3
"""Refuse commits that contain credentials or personal identifiers.

Install: ln -sf ../../scripts/check_secrets.py .git/hooks/pre-commit
Run on everything: python scripts/check_secrets.py --all
"""
import re
import subprocess
import sys

PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned secret": re.compile(r"(?i)(api[_-]?key|api[_-]?secret|secret[_-]?key|password|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
    "long hex/base62 token": re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])"),
    "webhook url": re.compile(r"https://open\.feishu\.cn/open-apis/bot/v2/hook/[A-Za-z0-9-]+|hooks\.slack\.com/services/\S+"),
    "corporate email": re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9-]+\.(com|cn|net|org)\b"),
    "home path": re.compile(r"/(data/)?home/[a-z][a-z0-9_-]+/|/Users/[A-Za-z0-9_-]+/"),
}
ALLOW = re.compile(r"example\.com|noreply")


def files(all_files: bool) -> list[str]:
    cmd = ["git", "ls-files"] if all_files else ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    return [f for f in subprocess.check_output(cmd, text=True).splitlines() if f]


def main() -> int:
    bad = []
    for f in files("--all" in sys.argv):
        if f == "scripts/check_secrets.py":
            continue
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except (IsADirectoryError, FileNotFoundError):
            continue
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                if not ALLOW.search(m.group(0)):
                    line = text.count("\n", 0, m.start()) + 1
                    bad.append(f"{f}:{line}: {name}: {m.group(0)[:60]}")
    if bad:
        print("possible secret / personal data, commit blocked:\n  " + "\n  ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
