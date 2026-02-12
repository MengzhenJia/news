#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Medium AI News Bot
- Fetches Medium RSS for AI/ML tags
- Filters last N hours
- Fetches per-article claps/comments via __APOLLO_STATE__
- Outputs Markdown report
"""

import argparse
import datetime as dt
import json
import re
import sys
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup

DEFAULT_TAGS = [
    "artificial-intelligence",
    "machine-learning",
]
DEFAULT_TOP_N = 10
DEFAULT_HOURS = 24
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def fetch_rss(tag: str, session: requests.Session) -> feedparser.FeedParserDict:
    url = f"https://medium.com/feed/tag/{tag}"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def parse_published(entry: feedparser.FeedParserDict) -> Optional[dt.datetime]:
    for key in ("published", "updated", "pubDate"):
        value = entry.get(key)
        if value:
            try:
                dt_val = parsedate_to_datetime(value)
                if dt_val.tzinfo is None:
                    dt_val = dt_val.replace(tzinfo=dt.timezone.utc)
                return dt_val.astimezone(dt.timezone.utc)
            except Exception:
                continue
    return None


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def simple_cn_summary(text: str, max_len: int = 120) -> str:
    base = strip_html(text)
    if not base:
        return "文章要点：暂无摘要。"
    # Split by common sentence punctuation
    parts = re.split(r"[。！？.!?]", base)
    first = next((p.strip() for p in parts if p.strip()), "")
    if not first:
        first = base.strip()
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "…"
    if not first.endswith("。"):
        first = first + "。"
    return f"文章要点：{first}"


def find_apollo_state(html: str) -> Optional[Dict]:
    # Try to locate the JSON blob assigned to __APOLLO_STATE__
    idx = html.find("__APOLLO_STATE__")
    if idx == -1:
        return None
    # Find the first '{' after the assignment
    brace_start = html.find("{", idx)
    if brace_start == -1:
        return None
    # Extract a balanced JSON object
    depth = 0
    for i in range(brace_start, len(html)):
        ch = html[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = html[brace_start : i + 1]
                try:
                    return json.loads(json_str)
                except Exception:
                    return None
    return None


def normalize_url(url: str) -> str:
    return url.split("?")[0].rstrip("/")


def resolve_ref(state: Dict, value):
    if isinstance(value, dict) and "__ref" in value:
        return state.get(value["__ref"], {})
    return value if isinstance(value, dict) else {}


def extract_metrics(state: Dict, target_url: str) -> Tuple[int, int]:
    target = normalize_url(target_url)
    post_candidates = []
    for key, val in state.items():
        if not isinstance(val, dict):
            continue
        if key.startswith("Post:"):
            post_candidates.append(val)
    chosen = None
    for post in post_candidates:
        url = post.get("mediumUrl") or post.get("canonicalUrl") or post.get("url")
        if url and normalize_url(url) == target:
            chosen = post
            break
    if not chosen and post_candidates:
        chosen = post_candidates[0]
    if not chosen:
        return 0, 0

    # Clap count
    claps = 0
    if "clapCount" in chosen:
        claps = chosen.get("clapCount") or 0
    else:
        virtuals = chosen.get("virtuals", {}) if isinstance(chosen.get("virtuals"), dict) else {}
        claps = virtuals.get("clapCount", 0)

    # Comments/responses count
    responses = 0
    post_responses = chosen.get("postResponses")
    resolved = resolve_ref(state, post_responses)
    if isinstance(resolved, dict):
        responses = resolved.get("count", 0) or 0

    return int(claps), int(responses)


def fetch_article_metrics(url: str, session: requests.Session) -> Tuple[int, int]:
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text
        state = find_apollo_state(html)
        if not state:
            return 0, 0
        return extract_metrics(state, url)
    except Exception:
        return 0, 0


def collect_entries(tags: Iterable[str], session: requests.Session) -> List[Dict]:
    entries = []
    seen = set()
    for tag in tags:
        feed = fetch_rss(tag, session)
        for entry in feed.entries:
            link = entry.get("link")
            if not link:
                continue
            norm = normalize_url(link)
            if norm in seen:
                continue
            seen.add(norm)
            published = parse_published(entry)
            entries.append(
                {
                    "title": entry.get("title", ""),
                    "author": entry.get("author", ""),
                    "link": link,
                    "published": published,
                    "summary": entry.get("summary", ""),
                }
            )
    return entries


def filter_recent(entries: List[Dict], hours: int) -> List[Dict]:
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=hours)
    recent = []
    for e in entries:
        pub = e.get("published")
        if pub and pub >= cutoff:
            recent.append(e)
    return recent


def render_markdown(items: List[Dict], report_date: dt.date) -> str:
    lines = [f"# {report_date.isoformat()} Medium AI News", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"## {i}. {item['title']}")
        lines.append(f"- 作者：{item['author'] or 'Unknown'}")
        lines.append(f"- 链接：{item['link']}")
        lines.append(f"- Claps：{item['claps']}")
        lines.append(f"- 评论数：{item['responses']}")
        lines.append(f"- 摘要：{item['cn_summary']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Medium AI News Bot")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument("--tags", type=str, default=",".join(DEFAULT_TAGS))
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        print("No tags provided.", file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    entries = collect_entries(tags, session)
    recent = filter_recent(entries, args.hours)

    enriched = []
    for e in recent:
        claps, responses = fetch_article_metrics(e["link"], session)
        enriched.append(
            {
                **e,
                "claps": claps,
                "responses": responses,
                "cn_summary": simple_cn_summary(e.get("summary", "")),
            }
        )

    enriched.sort(key=lambda x: (x["claps"], x["responses"]), reverse=True)
    top_items = enriched[: args.top_n]

    report_date = dt.date.today()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{report_date.isoformat()}-AI-News.md"
    outfile.write_text(render_markdown(top_items, report_date), encoding="utf-8")

    print(f"Saved: {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
