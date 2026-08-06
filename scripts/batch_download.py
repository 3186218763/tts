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
  python scripts/batch_download.py --new-only         # 只下载 data/raw 中没有的候选
  python scripts/batch_download.py --bvids BV1xxx     # 精确下载指定 BVID
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import httpx

# 强制无缓冲输出——后台运行时 stdout 默认全缓冲，会导致看不到进度
print = partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"       # m4a 原始下载
WAV_DIR = PROJECT_ROOT / "data" / "wav"        # 48kHz mono WAV
SEARCH_CACHE = PROJECT_ROOT / "data" / "search_results.json"
CHAT_CANDIDATES = PROJECT_ROOT / "data" / "chat_candidates.json"
PARTS_DIR = RAW_DIR / ".parts"

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

# 纯杂谈候选采用保守规则：必须命中正向词，并且不能命中歌回、观看、游戏或联动词。
CHAT_KEYWORDS = (
    "白菜来啦",
    "白菜来了",
    "今天怎么样",
    "杂谈",
    "聊天",
    "谢谢",
    "粉丝",
    "做饭",
    "料理",
    "学中文",
    "中文学习",
    "中文",
    "回顾",
    "后日谈",
    "菜鸟直播",
)
NON_CHAT_KEYWORDS = (
    "歌回",
    "唱歌",
    "演唱会",
    "音乐会",
    "一起看",
    "一起看看",
    "看看",
    "白菜看",
    "视频鉴赏",
    "看最美的夜",
    "看三国",
    "看动画",
    "看jojo",
    "bml",
    "联动",
    "repo",
    "连麦",
    "玩破防",
    "只狼",
    "我的世界",
    "逃离鸭科夫",
    "盛世天下",
    "小游戏",
    "麻将",
    "pvz",
    "切片",
    "剪辑",
)

EXCLUDED_BVIDS = {
    "BV1Hh9XBnEPF",  # 非花音本人录播
    "BV18ro9BVEQJ",  # 2026-04-25 歌回整场
    "BV1ReodBdEM1",  # 同一场 2026-04-25 歌回
    "BV1e6oZB8EeD",  # 歌回剪辑
    "BV1yGzyB8EVn",  # 3D 演唱会
    "BV1B6SEBrEXF",  # 观看周杰伦演唱会
    "BV1BMtzztEGw",  # 白菜看 BML
    "BV1Ck1FBXEX3",  # 一起看看
    "BV1Kh4y1K7Rb",  # 反应切片
    "BV1TPerzhEh8",  # 白菜看凭实力单身
    "BV1bRqPBQEDD",  # 看最美的夜
    "BV1kj4uz2E68",  # 白菜看哈基米
    "BV1mHi1BaEeF",  # 看最美的夜
}

CHINA_TZ = timezone(timedelta(hours=8))

# ─────────────────────────── 搜索 ───────────────────────────

def init_client() -> httpx.Client:
    """创建已激活 cookie 的 client。

    使用 trust_env=False，避免本机 HTTP(S)_PROXY 在直连 B 站时触发
    SSL UNEXPECTED_EOF（常见于本地代理只适合部分域名的情况）。
    """
    client = httpx.Client(headers=HEADERS, timeout=60.0, trust_env=False)
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


def is_chat_candidate(title: str) -> bool:
    normalized = title.lower().replace(" ", "")
    if any(keyword in normalized for keyword in NON_CHAT_KEYWORDS):
        return False
    return any(keyword in normalized for keyword in CHAT_KEYWORDS)


def extract_stream_date(title: str, pubdate: int = 0) -> str | None:
    """尽量从标题提取直播日期，避免按上传日期错误去重。"""
    inferred_year = (
        datetime.fromtimestamp(pubdate, tz=CHINA_TZ).year
        if pubdate
        else datetime.now(tz=CHINA_TZ).year
    )
    patterns = [
        r"(?P<y>20\d{2})[-_/.年](?P<m>\d{1,2})[-_/.月](?P<d>\d{1,2})",
        r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})",
        r"(?<!\d)(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})(?!\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, flags=re.IGNORECASE)
        if not match:
            continue
        year = int(match.group("y"))
        if year < 100:
            year += 2000
        try:
            return datetime(year, int(match.group("m")), int(match.group("d"))).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            continue

    match = re.search(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]", title)
    if match:
        try:
            return datetime(
                inferred_year, int(match.group("m")), int(match.group("d"))
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def infer_record_date(record: dict) -> date | None:
    """Resolve the likely stream date, falling back to upload date."""
    stream_date = extract_stream_date(
        record.get("title", ""), record.get("pubdate", 0)
    )
    if stream_date:
        return date.fromisoformat(stream_date)
    pubdate = int(record.get("pubdate", 0) or 0)
    if pubdate > 0:
        return datetime.fromtimestamp(pubdate, tz=CHINA_TZ).date()
    return None


def search_videos(client: httpx.Client, keyword: str, max_pages: int = 5) -> list[dict]:
    """搜索 Bilibili 视频，返回去重后的结果列表。"""
    results = []
    for page in range(1, max_pages + 1):
        d = None
        for attempt in range(3):
            try:
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
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                d = resp.json()
                break
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                print(
                    f"  [{keyword}] page {page} attempt {attempt + 1}/3: {exc}"
                )
                time.sleep(2 * (attempt + 1))
        if not isinstance(d, dict):
            print(f"  [{keyword}] page {page}: skipped after retries")
            continue
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
    # 优先按标题中的直播日期分组。标题无日期时不强行合并，避免误删同日不同场。
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r["duration"] < min_duration:
            continue
        stream_date = extract_stream_date(r.get("title", ""), r.get("pubdate", 0))
        r = {**r, "stream_date": stream_date}
        date_key = stream_date or f"bvid:{r['bvid']}"
        groups.setdefault(date_key, []).append(r)

    best_per_day = []
    for date_key, vids in sorted(groups.items()):
        def score(v):
            uploader_bonus = PREFERRED_UPLOADERS.get(v["author"], 0)
            return (uploader_bonus, v["duration"])
        best = max(vids, key=score)
        best_per_day.append(best)

    # 按直播日期降序（标题无日期时再退回上传时间）
    best_per_day.sort(
        key=lambda x: x.get("stream_date")
        or datetime.fromtimestamp(x.get("pubdate", 0), tz=CHINA_TZ).strftime(
            "%Y-%m-%d"
        ),
        reverse=True,
    )
    return best_per_day


# ─────────────────────────── 下载 ───────────────────────────


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def probe_decodable_duration(path: Path) -> float:
    """实际解码整条音频，避免被损坏文件的容器时长元数据误导。"""
    result, duration = _decode_probe(path)
    if result.returncode != 0 and duration <= 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return duration


def _decode_probe(path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    """Decode an audio file and return both progress and decoder diagnostics."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
            "-progress",
            "pipe:1",
            "-nostats",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    duration = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("out_time_us="):
            try:
                duration = max(
                    duration, int(line.split("=", 1)[1]) / 1_000_000
                )
            except ValueError:
                continue
    return result, duration


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


def _part_state(
    path: Path, expected_duration: float, total_size: int
) -> tuple[str, str]:
    """Classify a completed or resumable part without trusting container metadata."""
    if not path.exists():
        return "missing", "文件不存在"

    size = path.stat().st_size
    if size <= 0:
        return "invalid", "空文件"
    if total_size > 0 and size > total_size:
        return "invalid", f"文件过量 {size}/{total_size} bytes"

    result, decoded_duration = _decode_probe(path)
    decoder_clean = result.returncode == 0 and not result.stderr.strip()
    if expected_duration > 0:
        duration_ratio = decoded_duration / expected_duration
        if duration_ratio > 1.02:
            return (
                "invalid",
                f"解码时长过量 {decoded_duration:.1f}/{expected_duration:.1f}s",
            )
        if decoder_clean and duration_ratio >= 0.98:
            return "complete", f"可解码 {decoded_duration:.1f}s"
    else:
        duration_ratio = 0.0
        if decoder_clean and total_size > 0 and size == total_size:
            return "complete", f"完整文件 {size} bytes"

    if total_size > 0:
        if size == total_size:
            return (
                "invalid",
                f"字节完整但解码不完整 {decoded_duration:.1f}/{expected_duration:.1f}s",
            )
        if expected_duration > 0:
            byte_ratio = size / total_size
            # Audio bitrate is nearly constant. A large gap means earlier Range
            # responses were appended at the wrong byte offset.
            if byte_ratio > 0.05 and duration_ratio + 0.08 < byte_ratio:
                return (
                    "invalid",
                    f"字节/解码进度不一致 {byte_ratio:.1%}/{duration_ratio:.1%}",
                )
    elif not decoder_clean:
        return "invalid", "无法确认大小且解码器报告错误"

    if not decoder_clean and decoded_duration <= 0:
        return "invalid", "无法解码"
    return "resumable", f"有效前缀 {size / 1024 / 1024:.1f} MB"


def _quarantine_part(path: Path, reason: str) -> Path:
    """Atomically preserve an unusable part outside the active resume path."""
    quarantine_dir = path.parent / ".quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{path.name}.{time.time_ns()}.invalid"
    path.replace(target)
    print(f"    隔离异常分片: {path.name} ({reason})")
    return target


def _prepare_part_paths(
    output_path: Path, expected_duration: float, total_size: int
) -> tuple[Path, bool]:
    """Adopt legacy partials into a .part work file and preserve bad files."""
    work_path = output_path.with_name(f"{output_path.name}.part")
    resumable: list[Path] = []

    for path in (output_path, work_path):
        state, reason = _part_state(path, expected_duration, total_size)
        if state == "complete":
            if path == work_path:
                if output_path.exists():
                    _quarantine_part(output_path, "被完整临时分片替代")
                work_path.replace(output_path)
            elif work_path.exists():
                _quarantine_part(work_path, "已有完整成品分片")
            print(f"    ✓ 已完整下载 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return output_path, True
        if state == "invalid":
            _quarantine_part(path, reason)
        elif state == "resumable":
            resumable.append(path)

    if resumable:
        chosen = max(resumable, key=lambda path: path.stat().st_size)
        for path in resumable:
            if path != chosen:
                _quarantine_part(path, "存在更完整的可续传分片")
        if chosen == output_path:
            output_path.replace(work_path)
        print(f"    续传已有分片: {work_path.stat().st_size / 1024 / 1024:.1f} MB")
    return work_path, False


def _parse_content_range(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(
        r"bytes\s+(\d+)-(\d+)/(\d+|\*)",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    total = 0 if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def download_audio(
    client: httpx.Client,
    play_data: dict,
    output_path: Path,
    expected_duration: float = 0.0,
) -> Path:
    """流式下载音频——分块写入磁盘 + HTTP Range 断点续传。

    B 站 CDN 会在传输约 10-15MB 后主动断连，必须用 Range header 续传。
    """
    audio_streams = play_data.get("dash", {}).get("audio", [])
    if not audio_streams:
        raise RuntimeError("没有找到音频流")
    best = max(audio_streams, key=lambda x: x.get("bandwidth", 0))
    audio_url = best["baseUrl"]
    backup_urls = best.get("backupUrl") or []
    backup_url = backup_urls[0] if backup_urls else None

    headers = {
        **HEADERS,
        "Referer": "https://www.bilibili.com/",
        "Accept-Encoding": "identity",
    }
    dl_timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    urls = [u for u in [audio_url, backup_url] if u]
    if not urls:
        raise RuntimeError("播放地址为空")

    # 先发 HEAD 请求获取文件总大小
    total_size = 0
    for url in urls:
        try:
            resp = client.head(url, headers=headers, follow_redirects=True, timeout=dl_timeout)
            if resp.status_code not in (200, 206):
                continue
            total_size = int(resp.headers.get("content-length", 0))
            if total_size > 0:
                break
        except Exception:
            continue

    work_path, complete = _prepare_part_paths(
        output_path, expected_duration, total_size
    )
    if complete:
        return output_path

    max_retries = 15  # 每个文件的最多重试次数
    for attempt in range(max_retries):
        url = urls[attempt % len(urls)]  # 轮换主/备链接
        state, reason = _part_state(work_path, expected_duration, total_size)
        if state == "complete":
            work_path.replace(output_path)
            print(f"    ✓ 已完整下载 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return output_path
        if state == "invalid":
            _quarantine_part(work_path, reason)
        offset = work_path.stat().st_size if work_path.exists() else 0

        range_headers = dict(headers)
        if offset > 0:
            range_headers["Range"] = f"bytes={offset}-"

        try:
            with client.stream(
                "GET",
                url,
                headers=range_headers,
                follow_redirects=True,
                timeout=dl_timeout,
            ) as resp:
                if resp.status_code not in (200, 206):
                    print(f"\n    HTTP {resp.status_code}, 重试...")
                    time.sleep(2)
                    continue

                range_info = _parse_content_range(
                    resp.headers.get("content-range", "")
                )
                # 206 Partial Content 时 content-length 是剩余大小。
                chunk_total = int(resp.headers.get("content-length", 0))
                if resp.status_code == 206:
                    if range_info is None or range_info[0] != offset:
                        if work_path.exists():
                            _quarantine_part(
                                work_path,
                                "CDN 返回的 Content-Range 与续传偏移不一致",
                            )
                        print("\n    CDN Range 响应无效，改用下一地址重试")
                        continue
                    _, range_end, range_total = range_info
                    if chunk_total and chunk_total != range_end - offset + 1:
                        if work_path.exists():
                            _quarantine_part(
                                work_path,
                                "CDN Content-Length 与 Content-Range 不一致",
                            )
                        print("\n    CDN Range 长度无效，改用下一地址重试")
                        continue
                    if range_total > 0:
                        total_size = range_total
                    if range_end < offset:
                        print("\n    CDN Range 响应区间无效，重试")
                        continue
                mode = "ab"
                if offset > 0 and resp.status_code == 200:
                    _quarantine_part(work_path, "CDN 不支持当前分片续传")
                    print("\n    CDN 未接受 Range，保留旧分片并从头下载")
                    offset = 0
                    mode = "wb"
                elif offset == 0:
                    mode = "wb"
                if resp.status_code == 200 and chunk_total > 0:
                    total_size = chunk_total
                expected = total_size if total_size > 0 else offset + chunk_total
                with open(work_path, mode) as f:
                    for chunk in resp.iter_bytes(chunk_size=256 * 1024):
                        f.write(chunk)
                        offset += len(chunk)
                        pct = offset / expected * 100 if expected > 0 else 0
                        print(f"\r    下载: {offset / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="")
                    f.flush()
                    os.fsync(f.fileno())
                print()

                state, reason = _part_state(
                    work_path, expected_duration, total_size
                )
                if state == "complete":
                    work_path.replace(output_path)
                    return output_path
                if state == "invalid":
                    _quarantine_part(work_path, reason)
                else:
                    print(f"    分片尚未完整，继续续传 ({reason})")
        except Exception as e:
            print(f"\n    断连 (已传 {offset / 1024 / 1024:.1f} MB): {e}")
            time.sleep(1)
            continue

    raise RuntimeError(f"下载失败：重试 {max_retries} 次仍未完成")


def _safe_title(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|']", "_", title)


def _video_pages(info: dict) -> list[dict]:
    pages = info.get("pages") or []
    if pages:
        return pages
    return [
        {
            "page": 1,
            "cid": info["cid"],
            "duration": info.get("duration", 0),
            "part": info.get("title", "P1"),
        }
    ]


def _concat_audio_parts(parts: list[Path], output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.m4a")
    if len(parts) == 1:
        shutil.copy2(parts[0], tmp_path)
    else:
        concat_file = parts[0].parent / "concat.txt"
        lines = []
        for part in parts:
            escaped = str(part.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
        )
    tmp_path.replace(output_path)


def _download_all_pages(
    client: httpx.Client, bvid: str, info: dict, output_path: Path
) -> Path:
    pages = _video_pages(info)
    part_dir = PARTS_DIR / bvid
    part_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for index, page in enumerate(pages, 1):
        part_name = _safe_title(page.get("part") or f"P{index}")[:80]
        part_path = part_dir / f"p{index:03d}_{part_name}.m4a"
        print(
            f"    分 P {index}/{len(pages)}: "
            f"{page.get('duration', 0) / 60:.1f}min {part_name}"
        )
        play_data = get_play_url(client, bvid, int(page["cid"]))
        expected = float(page.get("duration") or 0)
        download_audio(
            client,
            play_data,
            part_path,
            expected_duration=expected,
        )
        if expected and probe_decodable_duration(part_path) < expected * 0.98:
            raise RuntimeError(f"分 P 下载不完整：{part_path.name}")
        downloaded.append(part_path)

    _concat_audio_parts(downloaded, output_path)
    expected_total = sum(float(page.get("duration") or 0) for page in pages)
    if expected_total and probe_decodable_duration(output_path) < expected_total * 0.98:
        raise RuntimeError(f"合并后的音频不完整：{output_path.name}")
    return output_path


def convert_to_wav(m4a_path: Path) -> Path:
    """ffmpeg 转 48kHz 单声道 WAV（build_dataset.py 期望的格式）。"""
    wav_path = WAV_DIR / (m4a_path.stem + ".wav")
    source_duration = probe_decodable_duration(m4a_path)
    metadata_duration = probe_duration(m4a_path)
    if metadata_duration - source_duration > 5.0:
        print(
            f"  ⚠ 原始音频尾部损坏/截短：可解码 {source_duration / 60:.1f}min，"
            f"容器标称 {metadata_duration / 60:.1f}min"
        )
    if wav_path.exists():
        try:
            if abs(source_duration - probe_duration(wav_path)) <= 2.0:
                return wav_path
            print(f"  WAV 时长不完整，重新转换: {wav_path.name}")
        except (OSError, ValueError, subprocess.CalledProcessError):
            print(f"  WAV 完整性无法确认，重新转换: {wav_path.name}")
    tmp_path = wav_path.with_name(f"{wav_path.name}.tmp.wav")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", str(m4a_path),
            "-ac", "1", "-ar", "48000",
            "-acodec", "pcm_s16le",
            str(tmp_path),
        ],
        capture_output=True, check=True,
    )
    tmp_path.replace(wav_path)
    if abs(source_duration - probe_duration(wav_path)) > 2.0:
        raise RuntimeError(f"WAV 转换后时长异常：{wav_path}")
    return wav_path


def download_one(client: httpx.Client, bvid: str) -> Path | None:
    """下载单个视频的音频并转 WAV。返回 WAV 路径或 None。"""
    try:
        info = get_video_info(client, bvid)
        title = _safe_title(info["title"])
        pages = _video_pages(info)
        duration = sum(float(page.get("duration") or 0) for page in pages)
        hours = duration / 3600
        print(f"  标题: {info['title'][:60]}")
        print(f"  时长: {hours:.1f}h，分 P: {len(pages)}")

        existing = list(RAW_DIR.glob(f"{bvid}_*.m4a"))
        m4a_path = existing[0] if existing else RAW_DIR / f"{bvid}_{title}.m4a"
        needs_download = not m4a_path.exists()
        if m4a_path.exists():
            existing_duration: float | None = None
            try:
                existing_duration = probe_decodable_duration(m4a_path)
            except (ValueError, subprocess.CalledProcessError):
                print(f"  现有音频无法完整解码，重新下载: {m4a_path.name}")
                needs_download = True
            else:
                needs_download = existing_duration <= 0 or (
                    duration > 0 and existing_duration < duration * 0.98
                )
            if not needs_download:
                print(f"  跳过下载（完整文件已存在）: {m4a_path.name}")
            elif (
                existing_duration is not None
                and existing_duration > 0
                and duration > 0
            ):
                print(
                    f"  现有音频不完整（{existing_duration / 60:.1f}/"
                    f"{duration / 60:.1f}min），补齐全部分 P"
                )
        if needs_download:
            _download_all_pages(client, bvid, info, m4a_path)

        size_mb = m4a_path.stat().st_size / 1024 / 1024
        print(f"  ✓ 音频: {size_mb:.1f} MB")

        wav = convert_to_wav(m4a_path)
        wav_mb = wav.stat().st_size / 1024 / 1024
        print(f"  ✓ WAV: {wav.name} ({wav_mb:.1f} MB)")
        return wav

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return None


def download_one_with_client(candidate: dict) -> tuple[dict, Path | None]:
    """为并行任务创建独立 HTTP client，避免跨线程共享连接状态。"""
    client = init_client()
    try:
        return candidate, download_one(client, candidate["bvid"])
    finally:
        client.close()


def _normalize_bvid(value: str) -> str:
    bvid = value.strip()
    if bvid[:2].lower() == "bv":
        bvid = "BV" + bvid[2:]
    else:
        bvid = "BV" + bvid
    if not re.fullmatch(r"BV[0-9A-Za-z]{10}", bvid):
        raise ValueError(f"无效 BVID: {value}")
    return bvid


def _select_bvid_candidates(values: list[str]) -> list[dict]:
    """Resolve exact BVIDs from existing caches without modifying either cache."""
    cached: dict[str, dict] = {}
    for path in (CHAT_CANDIDATES, SEARCH_CACHE):
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            bvid = str(record.get("bvid", ""))
            if bvid:
                cached.setdefault(bvid, record)

    selected: list[dict] = []
    seen: set[str] = set()
    for value in values:
        bvid = _normalize_bvid(value)
        if bvid in seen:
            continue
        seen.add(bvid)
        record = dict(cached.get(bvid, {}))
        record.update(
            {
                "bvid": bvid,
                "title": record.get("title", "(下载时获取标题)"),
                "duration": int(record.get("duration", 0) or 0),
                "pubdate": int(record.get("pubdate", 0) or 0),
                "play": int(record.get("play", 0) or 0),
                "author": str(record.get("author", "")),
                "stream_date": record.get("stream_date"),
            }
        )
        selected.append(record)
    return selected


# ─────────────────────────── 主入口 ───────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="批量下载真白花音录播音频")
    p.add_argument("--search-only", action="store_true", help="只搜索不下载")
    p.add_argument(
        "--from-cache",
        action="store_true",
        help="直接使用 data/search_results.json，不访问搜索 API",
    )
    p.add_argument(
        "--chat-only",
        action="store_true",
        help="只保留标题明确为纯杂谈倾向、且不含观看/歌回/游戏风险的候选",
    )
    p.add_argument(
        "--new-only",
        action="store_true",
        help="只保留 data/raw 中尚未出现的 BVID",
    )
    p.add_argument(
        "--bvids",
        nargs="+",
        metavar="BVID",
        help="精确下载指定 BVID；直接读取现有候选缓存且不改写缓存",
    )
    p.add_argument("--max-hours", type=float, default=0, help="最多下载多少小时（0=全部）")
    p.add_argument("--min-duration", type=int, default=1800, help="最短时长秒（默认 1800=30min）")
    p.add_argument("--max-pages", type=int, default=5, help="每个关键词最多搜索页数")
    p.add_argument(
        "--since-date",
        help="只保留该日期（含）之后的直播，格式 YYYY-MM-DD",
    )
    p.add_argument(
        "--until-date",
        help="只保留该日期（含）之前的直播，格式 YYYY-MM-DD",
    )
    p.add_argument("--keywords", nargs="*", default=None, help="自定义搜索关键词")
    p.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="并行下载/转换任务数（建议 2-3，默认 1）",
    )
    args = p.parse_args()
    if args.download_workers < 1:
        p.error("--download-workers 必须至少为 1")

    try:
        since_date = date.fromisoformat(args.since_date) if args.since_date else None
        until_date = date.fromisoformat(args.until_date) if args.until_date else None
    except ValueError as exc:
        p.error(f"日期格式必须是 YYYY-MM-DD: {exc}")
    if since_date and until_date and since_date > until_date:
        p.error("--since-date 不能晚于 --until-date")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 真白花音录播批量下载 ===\n")

    # ── 搜索 ──
    keywords = args.keywords or [
        "真白花音 录播",
        "真白花音 直播录像",
        "眞白花音 录播",
        "真白花音 直播回放",
        "真白花音 白菜来啦 直播回放",
        "眞白花音 杂谈 录播",
    ]

    client: httpx.Client | None = None
    if args.bvids:
        try:
            candidates = _select_bvid_candidates(args.bvids)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            p.error(str(exc))
        print(f"[1/3] 精确选择 {len(candidates)} 个 BVID（候选缓存保持不变）")
    elif args.from_cache:
        if not SEARCH_CACHE.exists():
            raise SystemExit(f"搜索缓存不存在：{SEARCH_CACHE}")
        results_list = json.loads(SEARCH_CACHE.read_text(encoding="utf-8"))
        print(f"[1/3] 从缓存载入 {len(results_list)} 条搜索结果")
    else:
        print("[1/3] 搜索 Bilibili ...")
        client = init_client()
        all_results: dict[str, dict] = {}
        for kw in keywords:
            print(f"  搜索: {kw}")
            results = search_videos(client, kw, max_pages=args.max_pages)
            print(f"    → {len(results)} 条")
            for r in results:
                all_results[r["bvid"]] = r

        results_list = list(all_results.values())
        SEARCH_CACHE.write_text(
            json.dumps(results_list, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 去重筛选 ──
    if args.bvids:
        print("\n[2/3] 使用精确 BVID 选择，跳过搜索去重")
    else:
        results_list = [r for r in results_list if r["bvid"] not in EXCLUDED_BVIDS]
        if since_date or until_date:
            before = len(results_list)
            dated_results = []
            for result in results_list:
                record_date = infer_record_date(result)
                if record_date is None:
                    continue
                if since_date and record_date < since_date:
                    continue
                if until_date and record_date > until_date:
                    continue
                dated_results.append(result)
            results_list = dated_results
            print(f"  日期范围过滤: {before} → {len(results_list)}")
        if args.chat_only:
            before = len(results_list)
            results_list = [
                r for r in results_list if is_chat_candidate(r.get("title", ""))
            ]
            print(f"  纯杂谈标题过滤: {before} → {len(results_list)}")

        print(f"\n[2/3] 去重筛选（>{args.min_duration}s, 按直播日期去重）...")
        candidates = deduplicate(results_list, min_duration=args.min_duration)
        # Keep the cache as the complete chat candidate set. ``--new-only`` is
        # a download-time filter and must not erase downloaded candidates.
        if args.chat_only:
            CHAT_CANDIDATES.write_text(
                json.dumps(candidates, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    if args.new_only:
        existing_bvids = {
            path.name.split("_", 1)[0] for path in RAW_DIR.glob("*.m4a")
        }
        candidates = [c for c in candidates if c["bvid"] not in existing_bvids]
        print(f"  仅新增素材: {len(candidates)} 个")
    total_hours = sum(c["duration"] for c in candidates) / 3600
    print(f"  候选: {len(candidates)} 个录播, 共 {total_hours:.1f} 小时")

    for c in candidates:
        dt = c.get("stream_date") or (
            datetime.fromtimestamp(c["pubdate"], tz=CHINA_TZ).strftime("%Y-%m-%d")
            if c["pubdate"]
            else "?"
        )
        hours = c["duration"] / 3600
        print(f"    {c['bvid']} | {hours:5.1f}h | {dt} | {c['author'][:12]:12s} | {c['title'][:50]}")

    if args.search_only:
        if args.bvids:
            print("\n精确 BVID 选择完成；未修改候选缓存")
        else:
            result_path = CHAT_CANDIDATES if args.chat_only else SEARCH_CACHE
            print(f"\n候选结果已保存到 {result_path}")
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
    if not download_list:
        print("\n=== 完成：没有待下载视频 ===")
        return
    accumulated_hours = 0
    success = 0
    if args.download_workers == 1:
        if client is None:
            client = init_client()
        for i, c in enumerate(download_list, 1):
            hours = c["duration"] / 3600
            print(
                f"\n--- [{i}/{len(download_list)}] "
                f"{c['bvid']} ({hours:.1f}h) ---"
            )
            wav = download_one(client, c["bvid"])
            if wav:
                success += 1
                accumulated_hours += hours
            time.sleep(3)  # 礼貌延迟
    else:
        worker_count = min(args.download_workers, len(download_list))
        print(f"  使用 {worker_count} 个并行下载/转换任务")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(download_one_with_client, candidate): candidate
                for candidate in download_list
            }
            for completed, future in enumerate(as_completed(futures), 1):
                candidate = futures[future]
                try:
                    _, wav = future.result()
                except Exception as exc:
                    print(f"  ✗ {candidate['bvid']} 任务失败: {exc}")
                    wav = None
                if wav:
                    success += 1
                    accumulated_hours += candidate["duration"] / 3600
                print(
                    f"  并行进度: {completed}/{len(download_list)}，"
                    f"成功 {success}"
                )

    print(f"\n=== 完成 ===")
    print(f"  成功: {success}/{len(download_list)}")
    print(f"  总时长: {accumulated_hours:.1f} 小时")
    print(f"  音频目录: {RAW_DIR}")
    print(f"  WAV 目录: {WAV_DIR}")


if __name__ == "__main__":
    main()
