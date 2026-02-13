#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily technical intelligence digest pipeline.

Features:
- Fetch OPML source list from a Gist API endpoint.
- Parse RSS/Atom feeds and keep only posts in the last N hours.
- Fetch article full text (fallback to feed summary/content).
- Summarize each item in Chinese with OpenAI-compatible API (MiniMax by default).
- Render a Reuters/Bloomberg-style HTML digest.
- Deliver digest via SMTP (Gmail / SendGrid switchable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import logging
import math
import os
import smtplib
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import feedparser
import requests
from bs4 import BeautifulSoup
from jinja2 import Template
from openai import OpenAI
from readability import Document
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

DEFAULT_GIST_API_URL = "https://api.github.com/gists/e6d2bf860ccc367fe37ff953ba6de66b"
DEFAULT_OPML_FILENAME = "hn-popular-blogs-2025.opml"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_HOURS = 24
DEFAULT_MAX_ITEMS = 40
DEFAULT_OUTPUT_FILE = "daily_digest.html"
DEFAULT_USER_AGENT = "daily-digest-bot/1.0"


@dataclass
class FeedSource:
    name: str
    feed_url: str
    site_url: str


@dataclass
class ArticleItem:
    source: str
    title: str
    link: str
    published_utc: dt.datetime
    content_text: str
    summary_zh: str
    read_minutes: int
    score: float
    published_local: str = ""


@dataclass
class DigestReport:
    date_local: dt.date
    headline: Optional[ArticleItem]
    grouped_items: Dict[str, List[ArticleItem]]
    stats: Dict[str, Any]


def get_logger() -> logging.Logger:
    logger = logging.getLogger("daily_digest")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def parse_bool_env(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_recipients(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def to_utc(datetime_obj: dt.datetime) -> dt.datetime:
    if datetime_obj.tzinfo is None:
        datetime_obj = datetime_obj.replace(tzinfo=dt.timezone.utc)
    return datetime_obj.astimezone(dt.timezone.utc)


def parse_entry_published(entry: Dict[str, Any]) -> Optional[dt.datetime]:
    for key in ("published_parsed", "updated_parsed"):
        ts = entry.get(key)
        if ts:
            try:
                return dt.datetime.fromtimestamp(time.mktime(ts), tz=dt.timezone.utc)
            except Exception:
                pass

    for key in ("published", "updated", "pubDate"):
        val = entry.get(key)
        if not val:
            continue
        try:
            parsed = email.utils.parsedate_to_datetime(val)
            if parsed is None:
                continue
            return to_utc(parsed)
        except Exception:
            pass

    for key in ("published", "updated"):
        val = entry.get(key)
        if not val:
            continue
        try:
            parsed = dt.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            return to_utc(parsed)
        except Exception:
            pass

    return None


def estimate_read_minutes(text: str) -> int:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 1
    cjk_chars = sum(1 for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
    non_cjk_words = len([w for w in cleaned.split(" ") if w])
    by_cjk = cjk_chars / 300.0
    by_words = non_cjk_words / 200.0
    return max(1, math.ceil(max(by_cjk, by_words)))


def strip_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())


def compress_text_for_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    return f"{head}\n...\n{tail}"


def request_with_retry(
    fn,
    attempts: int,
    retry_exceptions: Tuple[type[BaseException], ...],
):
    retrier = Retrying(
        stop=stop_after_attempt(max(1, attempts)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True,
    )
    return retrier(fn)


def fetch_opml_from_gist(
    session: requests.Session,
    gist_api_url: str,
    opml_filename: str,
    timeout_sec: int,
    attempts: int,
) -> str:
    def _fetch_meta() -> requests.Response:
        resp = session.get(gist_api_url, timeout=timeout_sec)
        resp.raise_for_status()
        return resp

    meta_resp = request_with_retry(_fetch_meta, attempts, (requests.RequestException,))
    meta = meta_resp.json()
    files = meta.get("files", {})
    file_meta = files.get(opml_filename)
    if not file_meta:
        raise RuntimeError(f"Cannot find OPML file '{opml_filename}' in gist metadata")
    raw_url = file_meta.get("raw_url")
    if not raw_url:
        raise RuntimeError("OPML file has no raw_url")

    def _fetch_opml() -> requests.Response:
        resp = session.get(raw_url, timeout=timeout_sec)
        resp.raise_for_status()
        return resp

    opml_resp = request_with_retry(_fetch_opml, attempts, (requests.RequestException,))
    return opml_resp.text


def parse_opml_feeds(opml_xml: str) -> List[FeedSource]:
    root = ET.fromstring(opml_xml)
    seen = set()
    sources: List[FeedSource] = []
    for node in root.findall(".//outline"):
        xml_url = node.attrib.get("xmlUrl", "").strip()
        if not xml_url:
            continue
        if xml_url in seen:
            continue
        seen.add(xml_url)
        name = node.attrib.get("title") or node.attrib.get("text") or xml_url
        site_url = node.attrib.get("htmlUrl", "")
        sources.append(FeedSource(name=name.strip(), feed_url=xml_url, site_url=site_url.strip()))
    return sources


def fetch_feed_entries(
    session: requests.Session,
    source: FeedSource,
    timeout_sec: int,
    attempts: int,
) -> List[Dict[str, Any]]:
    def _fetch() -> requests.Response:
        resp = session.get(source.feed_url, timeout=timeout_sec)
        resp.raise_for_status()
        return resp

    resp = request_with_retry(_fetch, attempts, (requests.RequestException,))
    parsed = feedparser.parse(resp.content)
    entries: List[Dict[str, Any]] = []
    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        published = parse_entry_published(entry)
        if not published:
            continue
        feed_summary = ""
        if entry.get("summary"):
            feed_summary = strip_html(entry.get("summary", ""))
        elif entry.get("content") and isinstance(entry["content"], list) and entry["content"]:
            content_value = entry["content"][0].get("value", "")
            feed_summary = strip_html(content_value)

        entries.append(
            {
                "source": source.name,
                "title": (entry.get("title") or "Untitled").strip(),
                "link": link.strip(),
                "published_utc": to_utc(published),
                "feed_summary": feed_summary,
            }
        )
    return entries


def filter_recent_entries(entries: Sequence[Dict[str, Any]], hours: int, now_utc: dt.datetime) -> List[Dict[str, Any]]:
    cutoff = now_utc - dt.timedelta(hours=hours)
    out = []
    for entry in entries:
        published = entry.get("published_utc")
        if isinstance(published, dt.datetime) and published >= cutoff:
            out.append(entry)
    return out


def fetch_article_full_text(
    session: requests.Session,
    article_url: str,
    timeout_sec: int,
    attempts: int,
) -> str:
    def _fetch() -> requests.Response:
        resp = session.get(article_url, timeout=timeout_sec)
        resp.raise_for_status()
        return resp

    resp = request_with_retry(_fetch, attempts, (requests.RequestException,))
    html = resp.text
    doc = Document(html)
    summary_html = doc.summary(html_partial=True)
    text = strip_html(summary_html)
    if len(text) < 200:
        text = strip_html(html)
    return text


def build_llm_client(api_key: str, base_url: str) -> OpenAI:
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def summarize_article(
    client: OpenAI,
    model: str,
    title: str,
    link: str,
    content_text: str,
    temperature: float,
    max_input_chars: int,
    attempts: int,
) -> str:
    prompt_text = compress_text_for_prompt(content_text, max_input_chars)
    system_prompt = (
        "你是科技新闻编辑。请用客观、专业、简洁的中文输出3-5句摘要，"
        "风格接近路透/彭博早报。必须覆盖：发生了什么、为何重要、潜在影响。"
        "禁止夸张和主观评价，不要编造事实。"
    )
    user_prompt = (
        f"标题：{title}\n"
        f"链接：{link}\n"
        f"正文：\n{prompt_text}\n\n"
        "请输出：\n"
        "1) 一段3-5句中文摘要；\n"
        "2) 最后单独一行以“关键信号：”开头，列出不超过3个短语。"
    )

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        text = text.strip()
        if not text:
            raise RuntimeError("Empty summary from LLM")
        return text

    return request_with_retry(_call, attempts, (Exception,))


def score_article(item: Dict[str, Any]) -> float:
    published = item["published_utc"]
    age_hours = max(0.0, (dt.datetime.now(dt.timezone.utc) - published).total_seconds() / 3600.0)
    recency = max(0.0, 24.0 - age_hours)
    read_minutes = estimate_read_minutes(item.get("content_text", ""))
    return recency + read_minutes * 0.1


def resolve_content_text(primary: str, fallback: str, title: str) -> str:
    if primary and primary.strip():
        return primary.strip()
    if fallback and fallback.strip():
        return fallback.strip()
    return (title or "").strip()


def choose_headline(items: Sequence[ArticleItem]) -> Optional[ArticleItem]:
    if not items:
        return None
    return max(items, key=lambda x: (x.read_minutes, x.published_utc))


def group_by_source(items: Sequence[ArticleItem]) -> Dict[str, List[ArticleItem]]:
    grouped: Dict[str, List[ArticleItem]] = defaultdict(list)
    for item in items:
        grouped[item.source].append(item)
    for source in grouped:
        grouped[source].sort(key=lambda x: x.published_utc, reverse=True)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0].lower()))


def render_html(report: DigestReport, timezone_name: str) -> str:
    template = Template(
        """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{ title }}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background: #f7f8fa; color: #1f2937; margin: 0; padding: 24px; }
    .wrap { max-width: 960px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 16px rgba(15,23,42,.08); }
    h1, h2, h3 { margin: 0 0 12px; }
    .meta { color: #4b5563; margin-bottom: 20px; }
    .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
    .title a { color: #0f766e; text-decoration: none; }
    .title a:hover { text-decoration: underline; }
    .small { color: #6b7280; font-size: 13px; margin-top: 4px; }
    .summary { margin-top: 8px; line-height: 1.6; white-space: pre-wrap; }
    .section { margin-top: 24px; }
    .divider { height: 1px; background: #e5e7eb; margin: 20px 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{{ title }}</h1>
    <div class="meta">时区：{{ timezone_name }} | 统计：源 {{ stats.source_count }}，成功 {{ stats.feed_ok }}，失败 {{ stats.feed_fail }}，新增 {{ stats.article_new }}，摘要成功 {{ stats.summary_ok }}，摘要失败 {{ stats.summary_fail }}</div>

    <div class="section">
      <h2>今日头条</h2>
      {% if headline %}
      <div class="card">
        <div class="title"><a href="{{ headline.link }}" target="_blank" rel="noopener noreferrer">{{ headline.title }}</a></div>
        <div class="small">来源：{{ headline.source }} | 发布时间：{{ headline.published_local }} | 预计阅读：{{ headline.read_minutes }} 分钟</div>
        <div class="summary">{{ headline.summary_zh }}</div>
      </div>
      {% else %}
      <div class="card">今日无新增重点文章。</div>
      {% endif %}
    </div>

    <div class="divider"></div>

    <div class="section">
      <h2>分类资讯（按博客源）</h2>
      {% for source, items in grouped_items.items() %}
      <h3>{{ source }}</h3>
      {% for item in items %}
      <div class="card">
        <div class="title"><a href="{{ item.link }}" target="_blank" rel="noopener noreferrer">{{ item.title }}</a></div>
        <div class="small">发布时间：{{ item.published_local }} | 预计阅读：{{ item.read_minutes }} 分钟</div>
        <div class="summary">{{ item.summary_zh }}</div>
      </div>
      {% endfor %}
      {% endfor %}
    </div>

    <div class="divider"></div>
    <div class="small">生成时间：{{ generated_at }} | 失败源：{{ failed_sources_text }}</div>
  </div>
</body>
</html>
        """
    )
    return template.render(
        title=f"每日技术情报简报 - {report.date_local.isoformat()}",
        timezone_name=timezone_name,
        headline=report.headline,
        grouped_items=report.grouped_items,
        stats=report.stats,
        generated_at=report.stats.get("generated_at"),
        failed_sources_text=", ".join(report.stats.get("failed_sources", [])) or "无",
    )


def get_smtp_config() -> Dict[str, Any]:
    provider = os.getenv("SMTP_PROVIDER", "sendgrid").strip().lower()
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "0") or 0)
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    use_tls = parse_bool_env(os.getenv("SMTP_USE_TLS", "true"), True)

    if provider == "gmail":
        host = host or "smtp.gmail.com"
        port = port or 587
    elif provider == "sendgrid":
        host = host or "smtp.sendgrid.net"
        port = port or 587
        username = username or "apikey"
    else:
        raise ValueError("SMTP_PROVIDER must be gmail or sendgrid")

    if not host or not port or not username or not password:
        raise ValueError("SMTP credentials are incomplete")

    return {
        "provider": provider,
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "use_tls": use_tls,
    }


def send_html_email(
    subject: str,
    html_content: str,
    sender: str,
    recipients: Sequence[str],
    smtp_config: Dict[str, Any],
    attempts: int,
) -> None:
    if not recipients:
        raise ValueError("No recipients provided")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    def _send() -> None:
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"], timeout=30) as server:
            if smtp_config["use_tls"]:
                server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            server.sendmail(sender, list(recipients), msg.as_string())

    request_with_retry(_send, attempts, (smtplib.SMTPException, OSError, TimeoutError))


def run_pipeline(args: argparse.Namespace) -> int:
    logger = get_logger()

    request_timeout = parse_int_env("REQUEST_TIMEOUT_SEC", 20)
    http_retries = parse_int_env("HTTP_RETRY_COUNT", 3)
    llm_retries = parse_int_env("LLM_RETRY_COUNT", 3)
    llm_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.minimax.chat/v1")
    llm_model = os.getenv("LLM_MODEL", "MiniMax-Text-01")
    llm_temperature = parse_float_env("LLM_TEMPERATURE", 0.2)
    llm_max_input_chars = parse_int_env("LLM_MAX_INPUT_CHARS", 12000)
    tz = ZoneInfo(args.timezone)

    if not llm_key:
        raise ValueError("LLM_API_KEY is required")

    recipients = parse_recipients(os.getenv("DIGEST_RECIPIENTS", ""))
    mail_from = os.getenv("MAIL_FROM", "")
    subject_prefix = os.getenv("MAIL_SUBJECT_PREFIX", "[Daily Digest]")

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    client = build_llm_client(api_key=llm_key, base_url=llm_base_url)

    now_utc = dt.datetime.now(dt.timezone.utc)
    today_local = now_utc.astimezone(tz).date()

    opml_xml = fetch_opml_from_gist(
        session=session,
        gist_api_url=args.opml_url,
        opml_filename=DEFAULT_OPML_FILENAME,
        timeout_sec=request_timeout,
        attempts=http_retries,
    )
    sources = parse_opml_feeds(opml_xml)

    stats: Dict[str, Any] = {
        "source_count": len(sources),
        "feed_ok": 0,
        "feed_fail": 0,
        "article_new": 0,
        "summary_ok": 0,
        "summary_fail": 0,
        "failed_sources": [],
        "generated_at": now_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    log_event(logger, "source_loaded", source_count=stats["source_count"])

    all_recent: List[Dict[str, Any]] = []
    seen_links = set()

    for source in sources:
        try:
            entries = fetch_feed_entries(
                session=session,
                source=source,
                timeout_sec=request_timeout,
                attempts=http_retries,
            )
            recent = filter_recent_entries(entries, args.hours, now_utc=now_utc)
            stats["feed_ok"] += 1
            for item in recent:
                norm_link = item["link"].split("#")[0]
                if norm_link in seen_links:
                    continue
                seen_links.add(norm_link)
                all_recent.append(item)
        except Exception as exc:
            stats["feed_fail"] += 1
            stats["failed_sources"].append(source.name)
            log_event(logger, "feed_error", source=source.name, error=str(exc))

    all_recent.sort(key=lambda x: x["published_utc"], reverse=True)
    all_recent = all_recent[: args.max_items]
    stats["article_new"] = len(all_recent)
    log_event(
        logger,
        "entries_collected",
        feed_ok=stats["feed_ok"],
        feed_fail=stats["feed_fail"],
        article_new=stats["article_new"],
    )

    articles: List[ArticleItem] = []

    for entry in all_recent:
        content_text = ""
        try:
            content_text = fetch_article_full_text(
                session=session,
                article_url=entry["link"],
                timeout_sec=request_timeout,
                attempts=http_retries,
            )
        except Exception:
            content_text = entry.get("feed_summary", "")

        content_text = resolve_content_text(
            primary=content_text,
            fallback=entry.get("feed_summary", ""),
            title=entry.get("title", ""),
        )

        try:
            summary_zh = summarize_article(
                client=client,
                model=llm_model,
                title=entry["title"],
                link=entry["link"],
                content_text=content_text,
                temperature=llm_temperature,
                max_input_chars=llm_max_input_chars,
                attempts=llm_retries,
            )
            stats["summary_ok"] += 1
        except Exception as exc:
            stats["summary_fail"] += 1
            summary_zh = f"摘要生成失败：{exc.__class__.__name__}。"

        read_minutes = estimate_read_minutes(content_text)
        article = ArticleItem(
            source=entry["source"],
            title=entry["title"],
            link=entry["link"],
            published_utc=entry["published_utc"],
            content_text=content_text,
            summary_zh=summary_zh,
            read_minutes=read_minutes,
            score=score_article({**entry, "content_text": content_text}),
        )
        articles.append(article)

    headline = choose_headline(articles)
    grouped = group_by_source(articles)

    for collection in grouped.values():
        for item in collection:
            item.published_local = item.published_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")

    if headline:
        headline.published_local = headline.published_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")

    report = DigestReport(
        date_local=today_local,
        headline=headline,
        grouped_items=grouped,
        stats=stats,
    )

    html_content = render_html(report, timezone_name=args.timezone)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    log_event(logger, "digest_rendered", output_file=args.output_file, bytes=len(html_content.encode("utf-8")))

    if args.dry_run:
        log_event(logger, "dry_run_done", mail_sent=False)
        return 0

    if not mail_from:
        raise ValueError("MAIL_FROM is required when dry-run is disabled")
    if not recipients:
        raise ValueError("DIGEST_RECIPIENTS is required when dry-run is disabled")

    smtp_config = get_smtp_config()
    subject = f"{subject_prefix} 每日技术情报简报 {today_local.isoformat()}"
    send_html_email(
        subject=subject,
        html_content=html_content,
        sender=mail_from,
        recipients=recipients,
        smtp_config=smtp_config,
        attempts=http_retries,
    )
    log_event(logger, "mail_sent", recipients=len(recipients), provider=smtp_config["provider"])
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily technical intelligence digest")
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS, help="Lookback window in hours")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS, help="Maximum number of digest items")
    parser.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE, help="Timezone for digest display")
    parser.add_argument("--dry-run", action="store_true", help="Generate HTML only; do not send email")
    parser.add_argument("--opml-url", type=str, default=DEFAULT_GIST_API_URL, help="Gist API URL")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output HTML file path")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
