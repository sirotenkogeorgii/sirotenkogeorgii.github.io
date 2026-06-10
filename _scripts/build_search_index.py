#!/usr/bin/env python3
"""Build the full-text search index for /search/.

Walks every published page (same set Jekyll renders), strips front matter,
math, HTML and markdown syntax, and stores each page's unique lowercase words.
Deduplication keeps the index small enough to ship to the browser while still
answering "which pages contain this word" exactly.

Output: assets/data/search-index.json (committed; GitHub Pages serves it
as-is). Re-run after editing notes — the pre-commit hook does this
automatically for commits that touch markdown files:

    python3 _scripts/build_search_index.py
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "assets", "data", "search-index.json")

# Directories that hold publishable pages. _notes is the Jekyll collection
# (permalink /notes/:path/); the rest are plain pages.
SOURCES = ["_notes", "subpages", "paper-notes", "notes"]

# Pages that should not be searchable (utility pages / the search page itself).
EXCLUDED_PERMALINKS = {"/tags/", "/graph/", "/search/", "/assets/data/tag-graph.json"}

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>\n]+>")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Word-like runs only. Math/markdown syntax never matches, so LaTeX-heavy
# text degrades into a handful of deduplicated command words (frac, mathbb,
# ...) instead of breaking extraction — deliberately no math stripping, since
# unbalanced $$ in source files would make any pairing regex eat prose.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*[a-z0-9]|[a-z0-9]")
HAS_LETTER_RE = re.compile(r"[a-z]")


def parse_front_matter(raw):
    """Return (scalars, tags, body) for the tiny YAML subset used here."""
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return None, [], raw
    body = raw[match.end():]
    scalars = {}
    tags = []
    in_tags_list = False
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        list_item = re.match(r"-\s*(.+)$", stripped)
        if list_item and in_tags_list:
            tags.append(list_item.group(1).strip().strip("'\""))
            continue
        kv = re.match(r"([A-Za-z_][\w-]*):\s*(.*)$", stripped)
        if kv:
            key, value = kv.group(1), kv.group(2).strip()
            in_tags_list = key == "tags" and value == ""
            if value == "":
                continue
            if key == "tags":
                # inline form: tags: [a, b] or tags: a
                tags.extend(t.strip().strip("'\"") for t in value.strip("[]").split(",") if t.strip())
            else:
                scalars[key] = value.strip("'\"")
    return scalars, tags, body


def extract_words(body, extra_text):
    text = body
    text = MD_IMAGE_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r" \1 ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower() + " " + extra_text.lower()
    words = set()
    for token in TOKEN_RE.findall(text):
        if len(token) > 40 or not HAS_LETTER_RE.search(token):
            continue
        words.add(token)
    return " ".join(sorted(words))


def url_for(relpath):
    """Mirror Jekyll's URL mapping (including percent-encoding) for this site."""
    no_ext = re.sub(r"\.(md|markdown)$", "", relpath)
    if relpath.startswith("_notes/"):
        url = "/notes/" + no_ext[len("_notes/"):] + "/"
    elif no_ext.endswith("/index"):
        url = "/" + no_ext[: -len("index")]
    else:
        url = "/" + no_ext + ".html"
    return quote(url, safe="/")


def collect_pages():
    pages = []
    for source in SOURCES:
        base = os.path.join(ROOT, source)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for name in sorted(filenames):
                if not name.endswith((".md", ".markdown")) or name.startswith("_"):
                    continue
                path = os.path.join(dirpath, name)
                relpath = os.path.relpath(path, ROOT)
                try:
                    raw = open(path, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                scalars, tags, body = parse_front_matter(raw)
                if scalars is None or "title" not in scalars:
                    continue  # no front matter -> Jekyll treats it as a static file
                url = scalars.get("permalink") or url_for(relpath)
                if url in EXCLUDED_PERMALINKS:
                    continue
                title = scalars["title"]
                words = extract_words(body, title + " " + " ".join(tags))
                pages.append({
                    "title": title,
                    "url": url,
                    "tags": tags,
                    "words": words,
                })
    pages.sort(key=lambda p: p["url"])
    return pages


def main():
    quiet = "--quiet" in sys.argv
    pages = collect_pages()
    index = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pageCount": len(pages),
        "pages": pages,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    if not quiet:
        size_mb = os.path.getsize(OUTPUT) / 1e6
        print(f"search index: {len(pages)} pages, {size_mb:.1f} MB -> {os.path.relpath(OUTPUT, ROOT)}")


if __name__ == "__main__":
    main()
