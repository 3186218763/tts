"""从 ASR 与现有 fewshot asr_pool 生成人设审阅队列（不注入运行时）。"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# 允许 `python scripts/build_persona_review_queue.py` 直接跑
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_text_quality import content_chars, evaluate_text_quality  # noqa: E402


# high_conf 与 tests/test_persona_context.py 历史门槛对齐
def meets_asr_conf(record: dict) -> bool:
    return (
        record.get("segment_count") in (1, None)
        and (record.get("avg_logprob") or -99) >= -0.2
        and (record.get("language_probability") or 0) >= 0.9
        and (record.get("max_no_speech_probability") or 0) <= 0.3
        and (record.get("max_compression_ratio") or 99) <= 1.2
        and evaluate_text_quality(record.get("text") or "", record.get("lang")).keep
    )


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", "", t)


def in_length_range(text: str, lo: int = 4, hi: int = 40) -> bool:
    n = len(content_chars(text))
    return lo <= n <= hi


def guess_era_tag(source_name: str) -> str:
    """从文件名猜时代；无法判断则 unknown。

    注意：仓库内大量素材为 2025–2026，peak 可能偏少——队列仍应收 late，
    由人审决定是否 approved 及是否标 late。
    """
    years = [int(y) for y in re.findall(r"(20[0-2][0-9])", source_name or "")]
    # 过滤明显非活动年份噪声时可再收紧；优先看 2019-2026
    years = [y for y in years if 2019 <= y <= 2026]
    if not years:
        return "unknown"
    y = years[0]
    if 2020 <= y <= 2023:
        return "peak"
    if y >= 2024:
        return "late"
    return "unknown"


def build_queue(
    *,
    asr_path: Path,
    fewshot_path: Path,
    out_path: Path,
    max_items: int = 180,
    prefer_zh: bool = True,
) -> dict:
    asr = json.loads(asr_path.read_text(encoding="utf-8"))
    few = json.loads(fewshot_path.read_text(encoding="utf-8")) if fewshot_path.exists() else {}
    seen: set[str] = set()
    items: list[dict] = []

    def try_add(
        text: str,
        lang: str,
        source: str,
        avg_logprob,
        suggested_scenes=None,
        from_seed=False,
        allow_ja: bool = False,
    ):
        if len(items) >= max_items:
            return
        text = (text or "").strip()
        if not text or not in_length_range(text):
            return
        key = normalize_text(text)
        if key in seen:
            return
        if not evaluate_text_quality(text, lang).keep:
            return
        lang_n = (lang or "zh").lower()
        if prefer_zh and lang_n == "ja" and not from_seed and not allow_ja:
            # 日语降权：先装满中文后再考虑（二阶段 allow_ja=True）
            return
        seen.add(key)
        items.append(
            {
                "id": f"Q{len(items)+1:04d}",
                "text": text,
                "lang": lang or "zh",
                "source": source,
                "avg_logprob": avg_logprob,
                "suggested_scenes": suggested_scenes or [],
                "status": "pending",
                "final_text": "",
                "scenes": [],
                "era_tag": guess_era_tag(source),
                "reviewer_note": "",
            }
        )

    # 1) 现有 fewshot asr_pool 种子优先
    for row in few.get("asr_pool") or []:
        try_add(
            row.get("text", ""),
            row.get("lang", "zh"),
            row.get("source", "fewshot-seed"),
            row.get("avg_logprob"),
            row.get("scenes") or [],
            from_seed=True,
        )

    # 2) ASR high_conf：peak 优先排序；prefer_zh 时先只收非 ja
    candidates = []
    for rec in asr:
        if not meets_asr_conf(rec):
            continue
        source = Path(rec.get("path") or "").name
        candidates.append((guess_era_tag(source), rec, source))
    rank = {"peak": 0, "unknown": 1, "late": 2}
    candidates.sort(key=lambda x: (rank.get(x[0], 9), x[1].get("lang") != "zh"))

    for _era, rec, source in candidates:
        if len(items) >= max_items:
            break
        try_add(
            rec.get("text", ""),
            rec.get("lang", "zh"),
            source,
            rec.get("avg_logprob"),
            allow_ja=False,
        )

    # 3) 若未满 max_items，二阶段允许 ja（中文优先已尽量装满）
    if len(items) < max_items:
        for _era, rec, source in candidates:
            if len(items) >= max_items:
                break
            lang = (rec.get("lang") or "zh").lower()
            if lang != "ja":
                continue
            try_add(
                rec.get("text", ""),
                rec.get("lang", "ja"),
                source,
                rec.get("avg_logprob"),
                allow_ja=True,
            )

    out = {
        "schema_version": 1,
        "note": "待人审；不得被 build_persona_context 读取",
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"written": len(items)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asr", type=Path, default=Path("data/asr_results.json"))
    p.add_argument("--fewshot", type=Path, default=Path("docs/persona/fewshot-lines.json"))
    p.add_argument("--out", type=Path, default=Path("docs/persona/asr-review-queue.json"))
    p.add_argument("--max-items", type=int, default=180)
    args = p.parse_args(argv)
    stats = build_queue(
        asr_path=args.asr,
        fewshot_path=args.fewshot,
        out_path=args.out,
        max_items=args.max_items,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
