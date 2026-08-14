"""Collect-plan pure functions: tier, hard exclude, dedupe, priority."""

from __future__ import annotations

import re
from typing import Any, Mapping

# Spec-narrow hard exclude only (not full batch_download.EXCLUDED_BVIDS).
HARD_EXCLUDED_BVIDS = frozenset(
    {
        "BV1Hh9XBnEPF",  # non-huayin
        "BV18ro9BVEQJ",  # full song stream
        "BV1ReodBdEM1",  # same song stream
        "BV1e6oZB8EeD",  # song-cut
        "BV1yGzyB8EVn",  # 3D concert
    }
)

PREFERRED_UPLOADERS = {
    "xun隐者不遇": 5,
    "小电视录播姬": 5,
    "主播代录mklubo": 5,
    "HatsuneMiku9358": 4,
    "大前神": 3,
    "朝乃芳Channel": 3,
    "SilverCreeper": 3,
}

CHAT_POSITIVE = (
    "学中文",
    "中文学习",
    "中文",
    "杂谈",
    "聊天",
    "白菜来啦",
    "白菜来了",
    "今天怎么样",
    "做饭",
    "料理",
    "菜鸟直播",
    "后日谈",
)

WATCH_KEYWORDS = (
    "一起看",
    "一起看看",
    "看看",
    "白菜看",
    "视频鉴赏",
    "看最美的夜",
)

DATE_RE = re.compile(
    r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})"
)


def is_hard_excluded(bvid: str, title: str) -> bool:
    if bvid in HARD_EXCLUDED_BVIDS:
        return True
    title = title or ""
    if "3D演唱会" in title:
        return True
    if "演唱会" in title and "看" not in title:
        return True
    if "歌回" in title and any(k in title for k in ("全程", "整场", "全收录")):
        return True
    return False


def classify_tier(title: str) -> str:
    title = title or ""
    if any(k in title for k in ("切片", "经典老番")) or (
        "剪辑" in title and "歌回" not in title
    ):
        return "A"
    if any(k in title for k in CHAT_POSITIVE) and not any(
        k in title for k in WATCH_KEYWORDS
    ) and "歌回" not in title and "演唱会" not in title:
        return "B"
    return "C"


def infer_lang_hint(title: str) -> str:
    title = title or ""
    keys = ("中文", "学中文", "杂谈", "白菜来啦", "今天怎么样")
    return "zh_bias" if any(k in title for k in keys) else "mixed"


def infer_risks(title: str) -> list[str]:
    title = title or ""
    risks: list[str] = []
    if any(k in title for k in WATCH_KEYWORDS):
        risks.append("watch")
    if any(k in title for k in ("歌", "唱")) and "歌回" not in title:
        if "bgm" not in risks:
            risks.append("bgm")
    return risks


def extract_stream_date(title: str, pubdate: int = 0) -> str | None:
    m = DATE_RE.search(title or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def priority_score(item: Mapping[str, Any]) -> float:
    tier = str(item.get("tier") or "C")
    tier_base = {"B": 300.0, "A": 200.0, "C": 100.0}.get(tier, 100.0)
    score = tier_base
    if item.get("lang_hint") == "zh_bias":
        score += 10.0
    risks = item.get("risk") or item.get("risks") or []
    if "watch" in risks:
        score -= 20.0
    play = float(item.get("play") or 0)
    duration = float(item.get("duration") or 0)
    score += min(play, 1_000_000) / 1_000_000
    score += min(duration, 20_000) / 20_000
    return score


def dedupe_candidates(
    items: list[dict[str, Any]],
    *,
    min_duration_by_tier: Mapping[str, int],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for raw in items:
        bvid = str(raw.get("bvid") or "")
        title = str(raw.get("title") or "")
        if not bvid or is_hard_excluded(bvid, title):
            continue
        tier = classify_tier(title)
        min_dur = int(min_duration_by_tier.get(tier, 0))
        duration = int(raw.get("duration") or 0)
        if duration < min_dur:
            continue
        row = {
            **raw,
            "tier": tier,
            "lang_hint": infer_lang_hint(title),
            "risk": infer_risks(title),
            "stream_date": extract_stream_date(title, int(raw.get("pubdate") or 0)),
        }
        prepared.append(row)

    clips = [r for r in prepared if r["tier"] == "A"]
    streams = [r for r in prepared if r["tier"] != "A"]

    by_bvid: dict[str, dict[str, Any]] = {}
    for r in clips:
        prev = by_bvid.get(r["bvid"])
        if prev is None or int(r.get("duration") or 0) > int(prev.get("duration") or 0):
            by_bvid[r["bvid"]] = r

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in streams:
        key = r.get("stream_date") or f"bvid:{r['bvid']}"
        groups.setdefault(str(key), []).append(r)

    best_streams: list[dict[str, Any]] = []
    for _key, vids in groups.items():
        def score(v: dict[str, Any]) -> tuple:
            return (
                PREFERRED_UPLOADERS.get(str(v.get("author") or ""), 0),
                int(v.get("duration") or 0),
            )

        best_streams.append(max(vids, key=score))

    return list(by_bvid.values()) + best_streams


def build_plan_entries(
    raw: list[dict[str, Any]],
    *,
    min_duration_by_tier: Mapping[str, int] | None = None,
    hard_excluded_bvids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build sorted plan items with status=planned."""
    del hard_excluded_bvids  # HARD_EXCLUDED_BVIDS is authoritative
    mins = min_duration_by_tier or {"B": 600, "A": 30, "C": 1800}
    deduped = dedupe_candidates(raw, min_duration_by_tier=mins)
    entries: list[dict[str, Any]] = []
    for r in deduped:
        entry = {
            "bvid": r["bvid"],
            "title": r.get("title", ""),
            "author": r.get("author", ""),
            "duration": int(r.get("duration") or 0),
            "pubdate": int(r.get("pubdate") or 0),
            "play": int(r.get("play") or 0),
            "tier": r["tier"],
            "lang_hint": r.get("lang_hint", "mixed"),
            "risk": list(r.get("risk") or []),
            "stream_date": r.get("stream_date"),
            "status": "planned",
        }
        entry["priority_score"] = priority_score(entry)
        entries.append(entry)
    entries.sort(key=lambda x: (-float(x["priority_score"]), str(x["bvid"])))
    return entries
