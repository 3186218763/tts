"""Bilibili 音频下载工具——通过 API 获取视频流并下载音频。

Bilibili 对 yt-dlp 做了反爬（412），此脚本通过 API 获取 cookie 后直接下载。
"""

import re
import sys
import time
from pathlib import Path

import httpx

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
}


def get_cookies(client: httpx.Client) -> dict:
    """从 Bilibili 获取 buvid3/buvid4 cookie。"""
    # 1. 获取指纹 cookie
    resp = client.get("https://api.bilibili.com/x/frontend/finger/spi")
    data = resp.json()
    b3 = data["data"]["b_3"]
    b4 = data["data"]["b_4"]

    # 2. 设置 cookie 并激活
    client.cookies.set("buvid3", b3, domain=".bilibili.com")
    client.cookies.set("buvid4", b4, domain=".bilibili.com")

    # 3. 访问主页激活 session
    client.get("https://www.bilibili.com/")
    time.sleep(0.5)

    return {"buvid3": b3, "buvid4": b4}


def get_video_info(client: httpx.Client, bvid: str) -> dict:
    """获取视频信息（标题、cid 等）。"""
    resp = client.get(
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
    )
    data = resp.json()
    if data["code"] != 0:
        raise RuntimeError(f"获取视频信息失败: {data.get('message', data)}")
    return data["data"]


def get_play_url(client: httpx.Client, bvid: str, cid: int) -> dict:
    """获取视频播放地址。fnval=16 获取 dash 格式。"""
    resp = client.get(
        "https://api.bilibili.com/x/player/wbi/playurl",
        params={
            "bvid": bvid,
            "cid": cid,
            "qn": 64,       # 720P 足够提取音频
            "fnval": 16,    # DASH 格式
            "fnver": 0,
            "fourk": 0,
        },
    )
    data = resp.json()
    if data["code"] != 0:
        raise RuntimeError(f"获取播放地址失败: {data.get('message', data)}")
    return data["data"]


def download_audio(client: httpx.Client, play_data: dict, output_path: Path) -> Path:
    """从 DASH 数据中提取音频流并下载。"""
    audio_streams = play_data.get("dash", {}).get("audio", [])
    if not audio_streams:
        raise RuntimeError("没有找到音频流")

    # 选最高质量的音频
    best = max(audio_streams, key=lambda x: x.get("bandwidth", 0))
    audio_url = best["baseUrl"]
    backup_url = best.get("backupUrl", [None])[0]

    print(f"  音频质量: {best.get('codecs', '?')} @ {best.get('bandwidth', 0) // 1000}kbps")

    # 下载
    headers = {**HEADERS, "Referer": "https://www.bilibili.com/"}
    try:
        resp = client.get(audio_url, headers=headers, follow_redirects=True)
        if resp.status_code != 200:
            resp = client.get(backup_url, headers=headers, follow_redirects=True)
    except Exception:
        resp = client.get(backup_url, headers=headers, follow_redirects=True)

    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def download_bilibili_audio(bvid: str, output_dir: str = "data/raw") -> Path:
    """下载 Bilibili 视频的音频轨。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        print(f"获取 cookie...")
        get_cookies(client)

        print(f"获取视频信息: {bvid}")
        info = get_video_info(client, bvid)
        title = re.sub(r'[\\/:*?"<>|]', "_", info["title"])
        cid = info["cid"]
        duration = info.get("duration", 0)
        print(f"  标题: {title}")
        print(f"  时长: {duration}s")

        print(f"获取播放地址...")
        play_data = get_play_url(client, bvid, cid)

        output_path = output_dir / f"{bvid}_{title}.m4a"
        print(f"下载音频...")
        download_audio(client, play_data, output_path)

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ 已保存: {output_path} ({size_mb:.1f} MB)")
        return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认下载花音的热门切片
        bvids = [
            "BV1Kh4y1K7Rb",  # 【切片】一看到这个
            "BV1TPerzhEh8",  # 白菜看凭实力单身
            "BV1BMtzztEGw",  # 白菜看BML雑踏
            "BV1kj4uz2E68",  # 白菜看让世界感受哈基米
        ]
        for bvid in bvids:
            try:
                download_bilibili_audio(bvid)
                time.sleep(2)  # 礼貌延迟
            except Exception as e:
                print(f"✗ {bvid} 失败: {e}")
    else:
        bvid = sys.argv[1].replace("BV", "").strip()
        if not bvid.startswith("BV"):
            bvid = "BV" + bvid
        download_bilibili_audio(bvid)
