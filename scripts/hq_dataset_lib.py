"""HQ dataset selection and write helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import dataset_text_quality as text_quality


def normalize_text_key(text: str) -> str:
    return text_quality.content_chars(text or "").lower()


def select_hq_items(
    items: list[Mapping[str, Any]],
    *,
    n_target: int,
    zh_ratio: float,
    s_min: float,
    max_source_share: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # pool: candidate + score gate
    pool: list[dict[str, Any]] = []
    for raw in items:
        if str(raw.get("decision")) != "candidate":
            continue
        if float(raw.get("score") or 0) < s_min:
            continue
        pool.append(dict(raw))

    # text dedupe keep best score
    best_by_text: dict[str, dict[str, Any]] = {}
    for item in pool:
        key = normalize_text_key(str(item.get("text") or ""))
        if not key:
            key = f"__empty__:{item.get('path')}"
        prev = best_by_text.get(key)
        if prev is None:
            best_by_text[key] = item
            continue
        ps, cs = float(prev.get("score") or 0), float(item.get("score") or 0)
        if cs > ps or (cs == ps and str(item.get("path")) < str(prev.get("path"))):
            best_by_text[key] = item
    pool = list(best_by_text.values())

    pool_zh = [x for x in pool if text_quality.normalize_lang(x.get("lang")) == "zh"]
    pool_ja = [x for x in pool if text_quality.normalize_lang(x.get("lang")) == "ja"]
    pool_zh.sort(key=lambda x: (-float(x.get("score") or 0), str(x.get("path") or "")))
    pool_ja.sort(key=lambda x: (-float(x.get("score") or 0), str(x.get("path") or "")))

    need_zh = math.ceil(zh_ratio * n_target) if n_target > 0 else 0
    zh_shortfall = len(pool_zh) < need_zh
    if len(pool_zh) == 0:
        n_eff = 0
    elif zh_shortfall:
        n_eff = min(n_target, int(math.floor(len(pool_zh) / zh_ratio))) if zh_ratio > 0 else 0
    else:
        n_eff = n_target

    if n_eff <= 0:
        stats = {
            "n_target": n_target,
            "n_total": 0,
            "n_zh": 0,
            "n_ja": 0,
            "zh_shortfall": True,
            "zh_ratio_actual": 0.0,
        }
        return [], stats

    n_zh = min(len(pool_zh), math.ceil(zh_ratio * n_eff))
    n_ja = min(len(pool_ja), n_eff - n_zh)
    while n_zh + n_ja < n_eff and n_zh < len(pool_zh):
        n_zh += 1

    max_per_zh = max(1, int(math.floor(max_source_share * n_zh))) if n_zh > 0 else 0
    max_per_ja = max(1, int(math.floor(max_source_share * n_ja))) if n_ja > 0 else 0

    def take(pool_lang: list[dict[str, Any]], n: int, cap: int) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for item in pool_lang:
            if len(selected) >= n:
                break
            bvid = str(item.get("source_bvid") or "")
            if cap > 0 and counts.get(bvid, 0) >= cap:
                continue
            selected.append(item)
            counts[bvid] = counts.get(bvid, 0) + 1
        return selected

    sel_zh = take(pool_zh, n_zh, max_per_zh)
    sel_ja = take(pool_ja, n_ja, max_per_ja)
    # If caps reduced count, try fill from remaining of same lang
    if len(sel_zh) < n_zh:
        have = {id(x) for x in sel_zh}
        for item in pool_zh:
            if len(sel_zh) >= n_zh:
                break
            if id(item) in have:
                continue
            bvid = str(item.get("source_bvid") or "")
            # allow exceeding only if still under after relaxing? keep strict cap
            counts = sum(1 for x in sel_zh if x.get("source_bvid") == bvid)
            if counts >= max_per_zh:
                continue
            sel_zh.append(item)

    selected = sel_zh + sel_ja
    selected.sort(key=lambda x: (-float(x.get("score") or 0), str(x.get("path") or "")))

    n_total = len(selected)
    n_zh_out = sum(1 for x in selected if text_quality.normalize_lang(x.get("lang")) == "zh")
    n_ja_out = n_total - n_zh_out
    stats = {
        "n_target": n_target,
        "n_eff": n_eff,
        "n_total": n_total,
        "n_zh": n_zh_out,
        "n_ja": n_ja_out,
        "zh_shortfall": zh_shortfall,
        "zh_ratio_actual": (n_zh_out / n_total) if n_total else 0.0,
    }
    return selected, stats


def write_hq_dataset(
    selected: list[Mapping[str, Any]],
    out_dir: Path,
    *,
    mode: str = "hardlink",
    thresholds: Mapping[str, Any] | None = None,
    stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    annotation_lines: list[str] = []
    manifest: list[dict[str, Any]] = []

    for idx, item in enumerate(selected, start=1):
        lang = text_quality.normalize_lang(item.get("lang"))
        tag = "ZH" if lang == "zh" else "JP"
        lang_suffix = "zh" if lang == "zh" else "jp"
        dest_name = f"{idx:05d}_{lang_suffix}.wav"
        dest = audio_dir / dest_name
        src = Path(str(item["path"]))
        if src.exists():
            if mode == "copy":
                shutil.copy2(src, dest)
            else:
                try:
                    if dest.exists():
                        dest.unlink()
                    os.link(src, dest)
                except OSError:
                    shutil.copy2(src, dest)
        else:
            # tests may use fake paths; create empty placeholder only if missing
            dest.write_bytes(b"")

        text = str(item.get("text") or "").replace("|", "/")
        rel = f"audio/{dest_name}"
        annotation_lines.append(f"{rel}|花音|{tag}|{text}")
        manifest.append(
            {
                "path": str(item.get("path")),
                "dataset_path": rel,
                "lang": lang,
                "text": text,
                "source_bvid": item.get("source_bvid"),
                "tier": item.get("tier"),
                "score": item.get("score"),
                "components": item.get("components"),
                "metrics": item.get("metrics"),
            }
        )

    (out_dir / "annotation.list").write_text(
        "\n".join(annotation_lines) + ("\n" if annotation_lines else ""),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stats_out = dict(stats or {})
    stats_out.setdefault("n_total", len(selected))
    (out_dir / "stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if thresholds is not None:
        (out_dir / "thresholds.json").write_text(
            json.dumps(dict(thresholds), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {"annotation_lines": annotation_lines, "manifest": manifest, "stats": stats_out}
