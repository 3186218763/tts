"""将人审通过的 asr-review-queue 条目合并进 fewshot-lines.json 的 asr_pool。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 启用 asr 池的最低门槛：总数 + 关键场景覆盖
_ENABLE_MIN_TOTAL = 40
_ENABLE_SCENE_MIN = 3
_ENABLE_REQUIRED_SCENES = ("问候", "夸赞", "自嘲", "食物", "游戏")
_OK_STATUS = frozenset({"approved", "edited"})


def _resolve_text(item: dict) -> str:
    """edited 优先用非空 final_text，否则用 text。"""
    status = (item.get("status") or "").strip()
    final = (item.get("final_text") or "").strip()
    if status == "edited" and final:
        return final
    # approved 也可能有 final_text 覆盖；空则回落 text
    if final:
        return final
    return (item.get("text") or "").strip()


def _ready_to_enable(pool: list[dict]) -> bool:
    if len(pool) < _ENABLE_MIN_TOTAL:
        return False
    for scene in _ENABLE_REQUIRED_SCENES:
        count = sum(1 for item in pool if scene in (item.get("scenes") or []))
        if count < _ENABLE_SCENE_MIN:
            return False
    return True


def merge_fewshot(
    fewshot_path: Path,
    queue_path: Path,
    out_path: Path,
    *,
    enable_if_ready: bool = False,
) -> dict:
    """合并 approved/edited 队列条目到 asr_pool，写 out_path，返回统计。"""
    few = json.loads(fewshot_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))

    asr_pool: list[dict] = []
    approved_count = 0
    skipped_no_scene = 0
    skipped_empty_text = 0

    for item in queue.get("items") or []:
        status = (item.get("status") or "").strip()
        if status not in _OK_STATUS:
            continue
        approved_count += 1
        scenes = item.get("scenes") or []
        if not scenes:
            skipped_no_scene += 1
            continue
        text = _resolve_text(item)
        if not text:
            skipped_empty_text += 1
            continue
        asr_pool.append(
            {
                "text": text,
                "lang": item.get("lang") or "zh",
                "scenes": list(scenes),
                "source": item.get("source") or "",
                "era_tag": item.get("era_tag") or "unknown",
                "high_conf": True,
            }
        )

    enabled = bool(few.get("asr_pool_enabled", False))
    if enable_if_ready:
        enabled = _ready_to_enable(asr_pool)
    elif not enable_if_ready:
        # 显式 False 时不自动开启；保留调用方意图（默认不启用）
        enabled = False

    out = {
        "schema_version": few.get("schema_version", 1),
        "note": few.get("note", ""),
        "injection_strategy": few.get("injection_strategy", ""),
        "asr_pool_enabled": enabled,
        "official_comments": few.get("official_comments") or [],
        "fan_pool": few.get("fan_pool") or [],
        "asr_pool": asr_pool,
    }
    # 透传其余顶层字段（若有），但不覆盖已写字段
    for key, value in few.items():
        if key not in out:
            out[key] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return {
        "approved_count": approved_count,
        "pool_size": len(asr_pool),
        "skipped_no_scene": skipped_no_scene,
        "skipped_empty_text": skipped_empty_text,
        "asr_pool_enabled": enabled,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fewshot",
        type=Path,
        default=Path("docs/persona/fewshot-lines.json"),
    )
    p.add_argument(
        "--queue",
        type=Path,
        default=Path("docs/persona/asr-review-queue.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/persona/fewshot-lines.json"),
    )
    p.add_argument(
        "--enable-if-ready",
        action="store_true",
        help="达到门槛时将 asr_pool_enabled 设为 true",
    )
    args = p.parse_args(argv)
    stats = merge_fewshot(
        args.fewshot,
        args.queue,
        args.out,
        enable_if_ready=args.enable_if_ready,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
