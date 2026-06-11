#!/usr/bin/env python3
"""Build the recent-updates feed shown on the home page.

Derives per-page "added" and "updated" dates from git history (committer
dates, day granularity). Commits that touch many pages at once — retagging,
formatting sweeps — are treated as infrastructure, not content updates, so
the feed reflects real writing activity. Pages staged right now count as
updated today, which lets the pre-commit hook fold the commit being made
into the feed it stages.

Output: _data/recent_updates.json, rendered by index.md at build time.
Re-run manually anytime:

    python3 _scripts/build_recent_updates.py
"""

import json
import os
import subprocess
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_search_index as bsi

ROOT = bsi.ROOT
OUTPUT = os.path.join(ROOT, "_data", "recent_updates.json")
FEED_SIZE = 30
# Commits touching more published markdown files than this are sweeps
# (retagging, mass formatting), not content updates.
BULK_COMMIT_THRESHOLD = 15


def published_pages():
    """relpath -> {title, url} for every tracked, published markdown page."""
    pages = {}
    for relpath, scalars, _tags, _body in bsi.iter_published_sources():
        url = scalars.get("permalink") or bsi.url_for(relpath)
        if url in bsi.EXCLUDED_PERMALINKS:
            continue
        pages[relpath] = {"title": scalars["title"], "url": url}
    return pages


def git_dates(pages):
    """Per page: most recent non-bulk commit date ("updated") and first-ever
    commit date ("added"). git log is newest-first, so the first qualifying
    hit per file is its latest update and the last hit is its creation."""
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", "log", "--no-merges",
         "--name-only", "--date=short", "--pretty=format:%x00%cd",
         "--", *bsi.SOURCES],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8", errors="replace")

    updated = {}
    added = {}
    for block in out.split("\x00"):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        commit_date = lines[0]
        md_files = [f for f in lines[1:] if f.endswith((".md", ".markdown"))]
        relevant = [f for f in md_files if f in pages]
        for relpath in relevant:
            added[relpath] = commit_date
        if len(md_files) > BULK_COMMIT_THRESHOLD:
            continue
        for relpath in relevant:
            updated.setdefault(relpath, commit_date)
    return updated, added


def staged_pages():
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", "diff", "--cached",
             "--name-only", "--diff-filter=ACMR", "--", *bsi.SOURCES],
            cwd=ROOT, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {
        name for name in out.stdout.decode("utf-8", errors="replace").splitlines()
        if name.endswith((".md", ".markdown"))
    }


def main():
    quiet = "--quiet" in sys.argv
    pages = published_pages()
    try:
        updated, added = git_dates(pages)
    except (OSError, subprocess.CalledProcessError):
        print("recent updates: git history unavailable; skipping", file=sys.stderr)
        return

    today = date.today().isoformat()
    for relpath in staged_pages() & set(pages):
        updated[relpath] = today
        added.setdefault(relpath, today)

    entries = []
    for relpath, info in pages.items():
        last = updated.get(relpath) or added.get(relpath)
        if not last:
            continue
        entries.append({
            "title": info["title"],
            "url": info["url"],
            "date": last,
            "added": added.get(relpath) == last,
        })
    entries.sort(key=lambda e: e["title"])
    entries.sort(key=lambda e: e["date"], reverse=True)
    entries = entries[:FEED_SIZE]

    try:
        with open(OUTPUT, encoding="utf-8") as handle:
            if json.load(handle).get("entries") == entries:
                if not quiet:
                    print(f"recent updates: unchanged ({len(entries)} entries)")
                return
    except (OSError, ValueError):
        pass

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    if not quiet:
        print(f"recent updates: {len(entries)} entries -> {os.path.relpath(OUTPUT, ROOT)}")


if __name__ == "__main__":
    main()
