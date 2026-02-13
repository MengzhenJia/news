import datetime as dt
import unittest
from unittest.mock import patch

from daily_digest import (
    ArticleItem,
    build_fallback_cn_summary,
    choose_headline,
    estimate_read_minutes,
    filter_recent_entries,
    is_chinese_summary,
    parse_str_env,
    parse_entry_published,
    parse_opml_feeds,
    resolve_content_text,
)


class DailyDigestTests(unittest.TestCase):
    def test_parse_opml_feeds_dedup_and_skip_empty(self):
        opml = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='2.0'><body><outline text='Blogs'>
  <outline type='rss' text='A' title='A' xmlUrl='https://a.com/feed.xml' htmlUrl='https://a.com'/>
  <outline type='rss' text='A2' title='A2' xmlUrl='https://a.com/feed.xml' htmlUrl='https://a.com'/>
  <outline type='rss' text='B' title='B'/>
</outline></body></opml>"""
        feeds = parse_opml_feeds(opml)
        self.assertEqual(1, len(feeds))
        self.assertEqual("A", feeds[0].name)
        self.assertEqual("https://a.com/feed.xml", feeds[0].feed_url)

    def test_parse_entry_published_rfc822_and_iso(self):
        e1 = {"published": "Wed, 12 Feb 2026 01:30:00 GMT"}
        dt1 = parse_entry_published(e1)
        self.assertIsNotNone(dt1)
        self.assertEqual(dt.timezone.utc, dt1.tzinfo)

        e2 = {"updated": "2026-02-12T09:30:00+08:00"}
        dt2 = parse_entry_published(e2)
        self.assertIsNotNone(dt2)
        self.assertEqual(1, dt2.hour)

    def test_filter_recent_entries_includes_cutoff_boundary(self):
        now = dt.datetime(2026, 2, 12, 12, 0, tzinfo=dt.timezone.utc)
        cutoff = now - dt.timedelta(hours=24)
        entries = [
            {"published_utc": cutoff, "id": "in"},
            {"published_utc": cutoff - dt.timedelta(seconds=1), "id": "out"},
        ]
        out = filter_recent_entries(entries, hours=24, now_utc=now)
        self.assertEqual(1, len(out))
        self.assertEqual("in", out[0]["id"])

    def test_estimate_read_minutes(self):
        words = "word " * 400
        self.assertEqual(2, estimate_read_minutes(words))
        chinese = "测" * 600
        self.assertEqual(2, estimate_read_minutes(chinese))

    def test_choose_headline_prefers_longest_then_latest(self):
        base = dt.datetime(2026, 2, 12, 0, 0, tzinfo=dt.timezone.utc)
        a = ArticleItem("S", "A", "https://a", base, "x", "s", 5, 1.0)
        b = ArticleItem("S", "B", "https://b", base + dt.timedelta(hours=1), "x", "s", 5, 1.0)
        c = ArticleItem("S", "C", "https://c", base, "x", "s", 6, 1.0)
        h = choose_headline([a, b, c])
        self.assertEqual("C", h.title)

    def test_resolve_content_text_with_fallback(self):
        self.assertEqual("primary", resolve_content_text(" primary ", "fallback", "title"))
        self.assertEqual("fallback", resolve_content_text("", " fallback ", "title"))
        self.assertEqual("title", resolve_content_text("", "", " title "))

    def test_build_fallback_cn_summary(self):
        text = "第一段讲了新模型发布。第二段讲了性能提升。第三段讲了产业影响。第四段补充细节。"
        summary = build_fallback_cn_summary(text, max_sentences=3)
        self.assertTrue(summary.startswith("要点："))
        self.assertIn("第一段讲了新模型发布", summary)
        self.assertIn("第三段讲了产业影响", summary)
        self.assertNotIn("第四段补充细节", summary)

    def test_build_fallback_cn_summary_for_english_content(self):
        text = "New AI benchmark shows improvement in coding and reasoning with lower latency."
        summary = build_fallback_cn_summary(text, title="Test Article")
        self.assertTrue(summary.startswith("要点："))
        self.assertIn("《Test Article》", summary)

    def test_parse_str_env_uses_default_when_blank(self):
        with patch.dict("os.environ", {"LLM_MODEL": "   "}):
            self.assertEqual("MiniMax-Text-01", parse_str_env("LLM_MODEL", "MiniMax-Text-01"))

    def test_is_chinese_summary(self):
        self.assertTrue(is_chinese_summary("这是一个中文摘要，包含足够的中文字符用于判断。"))
        self.assertFalse(is_chinese_summary("This is an English summary only."))


if __name__ == "__main__":
    unittest.main()
