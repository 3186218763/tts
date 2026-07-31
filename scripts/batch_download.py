"""批量搜索 + 下载真白花音录播音频。

流程：
  1. 调用 Bilibili 搜索 API，多关键词、多页翻页
  2. 过滤录播候选（时长 > 阈值、近 N 年）
  3. 按直播日期去重（同一场直播只保留最长版本）
  4. 批量下载音频轨（复用 bili_download 的 API 逻辑）
  5. ffmpeg 转 48kHz 单声道 WAV（build_dataset.py 的输入格式）

幂等：已下载的 BVID 自动跳过。支持断点续传。

用法：
  python scripts/batch_download.py                    # 默认下载全部候选
  python scripts/batch_download.py --max-hours 30     # 只下载约 30 小时
  python scripts/batch_download.py --search-only      # 只搜索不下载
  python scripts/batch_download.py --skip-existing    # 跳过已下载
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import httpx

# 强制无缓冲输出——后台运行时 stdout 默认全缓冲，会导致看不到进度
print = partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"       # m4a 原始下载
WAV_DIR = PROJECT_ROOT / "data" / "wav"        # 48kHz mono WAV
SEARCH_CACHE = PROJECT_ROOT / "data" / "search_results.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
}

# 录播优先上传者（按可靠性和完整性排序）
PREFERRED_UPLOADERS = {
    "xun隐者不遇": 5,
    "小电视录播姬": 5,
    "主播代录mklubo": 5,
    "HatsuneMiku9358": 4,
    "大前神": 3,
    "朝乃芳Channel": 3,
    "SilverCreeper": 3,
}

# ─────────────────────────── 搜索 ───────────────────────────

def init_client() -> httpx.Client:
    """创建已激活 cookie 的 client。"""
    client = httpx.Client(headers=HEADERS, timeout=60.0)
    resp = client.get("https://api.bilibili.com/x/frontend/finger/spi")
    data = resp.json()
    client.cookies.set("buvid3", data["data"]["b_3"], domain=".bilibili.com")
    client.cookies.set("buvid4", data["data"]["b_4"], domain=".bilibili.com")
    client.get("https://www.bilibili.com/")
    time.sleep(0.5)
    return client


def parse_duration(dur) -> int:
    """B 站 duration 可能是 'HH:MM:SS' / 'MM:SS' 或整数秒。"""
    if isinstance(dur, int):
        return dur
    if isinstance(dur, str):
        parts = dur.split(":")
        return sum(int(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    return 0


def search_videos(client: httpx.Client, keyword: str, max_pages: int = 5) -> list[dict]:
    """搜索 Bilibili 视频，返回去重后的结果列表。"""
    results = []
    for page in range(1, max_pages + 1):
        resp = client.get(
            "https://api.bilibili.com/x/web-interface/search/type",
            params={
                "keyword": keyword,
                "search_type": "video",
                "page": page,
                "page_size": 20,
                "order": "pubdate",
            },
        )
        d = resp.json()
        if d.get("code") != 0:
            print(f"  [{keyword}] page {page}: code={d.get('code')} {d.get('message', '')}")
            break
        items = d.get("data", {}).get("result", [])
        if not items:
            break
        for r in items:
            title = re.sub(r"<[^>]+>", "", r.get("title", ""))
            results.append({
                "bvid": r.get("bvid"),
                "title": title,
                "duration": parse_duration(r.get("duration", 0)),
                "pubdate": r.get("pubdate", 0),
                "play": r.get("play", 0),
                "author": r.get("author", ""),
            })
        time.sleep(1)
    return results


def deduplicate(results: list[dict], min_duration: int = 1800) -> list[dict]:
    """去重：同一直播日期只保留最佳版本。

    评分 = 上传者优先级 + 时长（越长越好）。
    """
    # 按近似日期分组（从 pubdate 提取日期）
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r["duration"] < min_duration:
            continue
        if r["pubdate"]:
            dt = datetime.fromtimestamp(r["pubdate"], tz=timezone.utc)
            date_key = dt.strftime("%Y-%m-%d")
        else:
            date_key = "unknown"
        groups.setdefault(date_key, []).append(r)

    best_per_day = []
    for date_key, vids in sorted(groups.items()):
        def score(v):
            uploader_bonus = PREFERRED_UPLOADERS.get(v["author"], 0)
            return (uploader_bonus, v["duration"])
        best = max(vids, key=score)
        best_per_day.append(best)

    # 按日期降序（最新优先）
    best_per_day.sort(key=lambda x: x.get("pubdate", 0), reverse=True)
    return best_per_day


# ─────────────────────────── 下载 ───────────────────────────

def get_video_info(client: httpx.Client, bvid: str) -> dict:
    resp = client.get(
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
    )
    data = resp.json()
    if data["code"] != 0:
        raise RuntimeError(f"获取视频信息失败: {data.get('message', data)}")
    return data["data"]


def get_play_url(client: httpx.Client, bvid: str, cid: int) -> dict:
    resp = client.get(
        "https://api.bilibili.com/x/player/wbi/playurl",
        params={
            "bvid": bvid,
            "cid": cid,
            "qn": 64,
            "fnval": 16,
            "fnver": 0,
            "fourk": 0,
        },
    )
    data = resp.json()
    if data["code"] != 0:
        raise RuntimeError(f"获取播放地址失败: {data.get('message', data)}")
    return data["data"]


def download_audio(client: httpx.Client, play_data: dict, output_path: Path) -> Path:
    """流式下载音频——分块写入磁盘 + HTTP Range 断点续传。

    B 站 CDN 会在传输约 10-15MB 后主动断连，必须用 Range header 续传。
    """
    audio_streams = play_data.get("dash", {}).get("audio", [])
    if not audio_streams:
        raise RuntimeError("没有找到音频流")
    best = max(audio_streams, key=lambda x: x.get("bandwidth", 0))
    audio_url = best["baseUrl"]
    backup_url = best.get("backupUrl", [None])[0]

    headers = {**HEADERS, "Referer": "https://www.bilibili.com/"}
    dl_timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    urls = [u for u in [audio_url, backup_url] if u]

    # 先发 HEAD 请求获取文件总大小
    total_size = 0
    for url in urls:
        try:
            resp = client.head(url, headers=headers, follow_redirects=True, timeout=dl_timeout)
            total_size = int(resp.headers.get("content-length", 0))
            if total_size > 0:
                break
        except Exception:
            continue

    max_retries = 15  # 每个文件的最多重试次数
    for attempt in range(max_retries):
        url = urls[attempt % len(urls)]  # 轮换主/备链接
        offset = output_path.stat().st_size if output_path.exists() else 0
        if total_size > 0 and offset >= total_size:
            print(f"    ✓ 已完整下载 ({offset / 1024 / 1024:.1f} MB)")
            return output_path

        range_headers = dict(headers)
        if offset > 0:
            range_headers["Range"] = f"bytes={offset}-"

        try:
            with client.stream("GET", url, headers=range_headers, follow_redirects=True, timeout=dl_timeout) as resp:
                if resp.status_code not in (200, 206):
                    print(f"\n    HTTP {resp.status_code}, 重试...")
                    time.sleep(2)
                    continue
                # 206Partial Content 时 content-length 是剩余大小
                chunk_total = int(resp.headers.get("content-length", 0))
                expected = total_size if total_size > 0 else offset + chunk_total
                with open(output_path, "ab") as f:
                    for chunk in resp.iter_bytes(chunk_size=256 * 1024):
                        f.write(chunk)
                        offset += len(chunk)
                        pct = offset / expected * 100 if expected > 0 else 0
                        print(f"\r    下载: {offset / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="")
                print()
                if total_size > 0 and offset >= total_size:
                    return output_path
                elif total_size == 0 and chunk_total > 0 and offset >= offset - chunk_total + chunk_total:
                    # 没有 HEAD 信息但流结束了
                    return output_path
        except Exception as e:
            print(f"\n    断连 (已传 {offset / 1024 / 1024:.1f} MB): {e}")
            time.sleep(1)
            continue

    raise RuntimeError(f"下载失败：重试 {max_retries} 次仍未完成")


def convert_to_wav(m4a_path: Path) -> Path:
    """ffmpeg 转 48kHz 单声道 WAV（build_dataset.py 期望的格式）。"""
    wav_path = WAV_DIR / (m4a_path.stem + ".wav")
    if wav_path.exists():
        return wav_path
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", str(m4a_path),
            "-ac", "1", "-ar", "48000",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        capture_output=True, check=True,
    )
    return wav_path


def download_one(client: httpx.Client, bvid: str) -> Path | None:
    """下载单个视频的音频并转 WAV。返回 WAV 路径或 None。"""
    # 检查是否已完成
    existing = list(RAW_DIR.glob(f"{bvid}_*.m4a"))
    if existing:
        m4a = existing[0]
        print(f"  跳过（已存在）: {m4a.name}")
        wav = convert_to_wav(m4a)
        return wav

    try:
        info = get_video_info(client, bvid)
        title = re.sub(r'[\\/:*?"<>|]', "_", info["title"])
        cid = info["cid"]
        duration = info.get("duration", 0)
        hours = duration / 3600
        print(f"  标题: {info['title'][:60]}")
        print(f"  时长: {hours:.1f}h")

        play_data = get_play_url(client, bvid, cid)
        m4a_path = RAW_DIR / f"{bvid}_{title}.m4a"
        download_audio(client, play_data, m4a_path)

        size_mb = m4a_path.stat().st_size / 1024 / 1024
        print(f"  ✓ 音频: {size_mb:.1f} MB")

        wav = convert_to_wav(m4a_path)
        wav_mb = wav.stat().st_size / 1024 / 1024
        print(f"  ✓ WAV: {wav.name} ({wav_mb:.1f} MB)")
        return wav

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return None


# ─────────────────────────── 主入口 ───────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="批量下载真白花音录播音频")
    p.add_argument("--search-only", action="store_true", help="只搜索不下载")
    p.add_argument("--max-hours", type=float, default=0, help="最多下载多少小时（0=全部）")
    p.add_argument("--min-duration", type=int, default=1800, help="最短时长秒（默认 1800=30min）")
    p.add_argument("--keywords", nargs="*", default=None, help="自定义搜索关键词")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 真白花音录播批量下载 ===\n")

    # ── 搜索 ──
    keywords = args.keywords or [
        "真白花音 录播",
        "真白花音 直播录像",
        "眞白花音 录播",
        "真白花音 直播回放",
    ]

    print("[1/3] 搜索 Bilibili ...")
    client = init_client()

    all_results: dict[str, dict] = {}
    for kw in keywords:
        print(f"  搜索: {kw}")
        results = search_videos(client, kw, max_pages=5)
        print(f"    → {len(results)} 条")
        for r in results:
            all_results[r["bvid"]] = r

    results_list = list(all_results.values())
    SEARCH_CACHE.write_text(
        json.dumps(results_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 去重筛选 ──
    # 黑名单：非录播内容（毕业后的杂谈、非花音本人等）
    BLACKLIST = {"BV1Hh9XBnEPF"}  # 白菜毕业之后（非花音录播，仅 4MB）
    results_list = [r for r in results_list if r["bvid"] not in BLACKLIST]

    print(f"\n[2/3] 去重筛选（>{args.min_duration}s, 按日期去重）...")
    candidates = deduplicate(results_list, min_duration=args.min_duration)
    total_hours = sum(c["duration"] for c in candidates) / 3600
    print(f"  候选: {len(candidates)} 个录播, 共 {total_hours:.1f} 小时")

    for c in candidates:
        dt = datetime.fromtimestamp(c["pubdate"], tz=timezone.utc).strftime("%Y-%m-%d") if c["pubdate"] else "?"
        hours = c["duration"] / 3600
        print(f"    {c['bvid']} | {hours:5.1f}h | {dt} | {c['author'][:12]:12s} | {c['title'][:50]}")

    if args.search_only:
        print(f"\n搜索结果已保存到 {SEARCH_CACHE}")
        return

    # ── 下载 ──
    print(f"\n[3/3] 下载音频 ...")
    download_list = candidates
    if args.max_hours > 0:
        acc = 0
        download_list = []
        for c in candidates:
            if acc >= args.max_hours:
                break
            download_list.append(c)
            acc += c["duration"] / 3600

    print(f"  将下载 {len(download_list)} 个视频")
    accumulated_hours = 0
    success = 0
    for i, c in enumerate(download_list, 1):
        hours = c["duration"] / 3600
        print(f"\n--- [{i}/{len(download_list)}] {c['bvid']} ({hours:.1f}h) ---")
        wav = download_one(client, c["bvid"])
        if wav:
            success += 1
            accumulated_hours += hours
        time.sleep(3)  # 礼貌延迟

    print(f"\n=== 完成 ===")
    print(f"  成功: {success}/{len(download_list)}")
    print(f"  总时长: {accumulated_hours:.1f} 小时")
    print(f"  音频目录: {RAW_DIR}")
    print(f"  WAV 目录: {WAV_DIR}")


if __name__ == "__main__":
    main()
