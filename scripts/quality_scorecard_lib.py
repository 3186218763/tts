"""HQ quality scorecard: hard gates + soft scores (pure functions)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import dataset_text_quality as text_quality
from dataset_hq_config import DatasetHQConfig

_BVID_RE = re.compile(r"^(BV[0-9A-Za-z]+)")


_OPENCC_T2S = None


def _to_simplified(text: str) -> str:
    """Normalize zh text to Simplified Chinese (t2s). Lazy; no-op if missing."""
    global _OPENCC_T2S
    if _OPENCC_T2S is None:
        try:
            from opencc import OpenCC

            _OPENCC_T2S = OpenCC("t2s")
        except ImportError:
            _OPENCC_T2S = False
    if _OPENCC_T2S:
        return _OPENCC_T2S.convert(text)
    return text




def extract_bvid(path: str | Path) -> str:
    name = Path(path).name
    m = _BVID_RE.match(name)
    return m.group(1) if m else ""


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _is_singing(filter_row: Mapping[str, Any] | None, cfg: DatasetHQConfig) -> bool:
    if not filter_row:
        return False
    vr = _finite(filter_row.get("voiced_ratio"))
    f0 = _finite(filter_row.get("f0_cv"))
    lv = _finite(filter_row.get("longest_voiced"))
    if vr is not None and f0 is not None and lv is not None:
        if (
            vr > cfg.voiced_ratio_threshold
            and f0 < cfg.f0_cv_threshold
            and lv > cfg.longest_voiced_threshold
        ):
            return True
    if str(filter_row.get("reason") or "") == "singing":
        # Prefer metrics when present; reason alone still flags singing if metrics missing
        if vr is None or f0 is None or lv is None:
            return True
    return False


def _speaker_allowed(decision: str, policy: str) -> bool:
    allowed = {
        "accept": {"accept"},
        "accept-or-uncertain": {"accept", "uncertain"},
    }.get(policy, {"accept"})
    return decision in allowed


def _soft_components(
    *,
    similarity: float | None,
    lang_prob: float | None,
    avg_logprob: float | None,
    speaker_score: float | None,
    text: str,
    lang: str,
    tier: str,
    risks: list[str],
    duration: float | None,
    cfg: DatasetHQConfig,
) -> dict[str, float]:
    s_align = _clamp(similarity if similarity is not None else 0.0)

    if lang_prob is None and avg_logprob is None:
        s_asr = 0.0
    else:
        lp = _clamp(lang_prob if lang_prob is not None else 0.0)
        # map avg_logprob from [min_avg_logprob, 0] → [0, 1]
        span = 0.0 - cfg.min_avg_logprob
        if avg_logprob is None or span <= 0:
            lp_part = 0.0
        else:
            lp_part = _clamp((avg_logprob - cfg.min_avg_logprob) / span)
        s_asr = 0.5 * lp + 0.5 * lp_part

    s_spk = _clamp(speaker_score if speaker_score is not None else 0.0)

    script = text_quality.script_ratio(text, lang) if text and lang else 0.0
    content_len = len(text_quality.content_chars(text or ""))
    if content_len < 2:
        content_factor = 0.0
    elif content_len <= 40:
        content_factor = 1.0
    else:
        content_factor = max(0.5, 40.0 / content_len)
    s_text = _clamp(script * content_factor)

    s_src = float(cfg.tier_src_score.get(tier, 0.6))
    if any(r in risks for r in ("watch", "bgm")):
        s_src *= 0.85
    s_src = _clamp(s_src)

    if duration is None:
        s_dur = 0.0
    elif 2.0 <= duration <= 12.0:
        s_dur = 1.0
    elif cfg.duration_min <= duration < 2.0:
        # linear 0.5 → 1.0
        span = 2.0 - cfg.duration_min
        s_dur = 0.5 + 0.5 * ((duration - cfg.duration_min) / span if span > 0 else 1.0)
    elif 12.0 < duration <= cfg.duration_max:
        span = cfg.duration_max - 12.0
        s_dur = 1.0 - 0.5 * ((duration - 12.0) / span if span > 0 else 0.0)
    else:
        s_dur = 0.0
    s_dur = _clamp(s_dur)

    return {
        "align": s_align,
        "asr": s_asr,
        "spk": s_spk,
        "text": s_text,
        "src": s_src,
        "dur": s_dur,
    }


def _weighted_score(components: Mapping[str, float], cfg: DatasetHQConfig) -> float:
    w = cfg.weights
    total_w = w.align + w.asr + w.spk + w.text + w.src + w.dur
    if total_w <= 0:
        return 0.0
    raw = (
        w.align * components["align"]
        + w.asr * components["asr"]
        + w.spk * components["spk"]
        + w.text * components["text"]
        + w.src * components["src"]
        + w.dur * components["dur"]
    )
    return 100.0 * raw / total_w


def score_one(
    *,
    path: str,
    filter_row: Mapping[str, Any] | None,
    speaker_row: Mapping[str, Any] | None,
    asr_row: Mapping[str, Any] | None,
    align_row: Mapping[str, Any] | None,
    tier: str,
    risks: list[str],
    cfg: DatasetHQConfig,
) -> dict[str, Any]:
    reject_reasons: list[str] = []
    tier = tier or "C"
    risks = list(risks or [])

    duration = None
    if filter_row is not None:
        duration = _finite(filter_row.get("duration"))
    if duration is None and asr_row is not None:
        duration = _finite(asr_row.get("audio_duration"))

    # H1 duration
    if duration is None or not (cfg.duration_min <= duration <= cfg.duration_max):
        reject_reasons.append("duration")

    # H2 energy
    rms = _finite(filter_row.get("rms_db")) if filter_row else None
    if rms is None or rms < cfg.min_rms_db:
        reject_reasons.append("low_energy")

    # H3 singing
    if _is_singing(filter_row, cfg):
        reject_reasons.append("singing")

    # H4 speaker
    decision = str((speaker_row or {}).get("decision") or "")
    score_lower = _finite((speaker_row or {}).get("score_lower"))
    speaker_score = _finite((speaker_row or {}).get("score"))
    if speaker_row is None or not _speaker_allowed(decision, cfg.speaker_policy):
        reject_reasons.append("speaker")
    elif tier == "A" and (
        score_lower is None or score_lower < cfg.clip_min_score_lower
    ):
        reject_reasons.append("speaker")

    # H5/H6 ASR
    lang = ""
    text = ""
    lang_prob = None
    avg_logprob = None
    if asr_row is None:
        reject_reasons.append("missing_asr")
        reject_reasons.append("asr_quality")
    else:
        lang = text_quality.normalize_lang(asr_row.get("lang"))
        text = str(asr_row.get("text") or "").strip()
        lang_prob = _finite(asr_row.get("language_probability"))
        avg_logprob = _finite(asr_row.get("avg_logprob"))
        no_speech = _finite(asr_row.get("max_no_speech_probability"))
        compression = _finite(asr_row.get("max_compression_ratio"))
        if lang not in {"zh", "ja"}:
            reject_reasons.append("lang")
        asr_bad = False
        if lang_prob is None or lang_prob < cfg.min_language_probability:
            asr_bad = True
        if avg_logprob is None or avg_logprob < cfg.min_avg_logprob:
            asr_bad = True
        if no_speech is not None and no_speech > cfg.max_no_speech_probability:
            asr_bad = True
        if compression is not None and compression > cfg.max_compression_ratio:
            asr_bad = True
        if asr_bad:
            reject_reasons.append("asr_quality")

    # text from alignment preferred later
    similarity = None
    if align_row is None:
        reject_reasons.append("missing_alignment")
        reject_reasons.append("alignment")
    else:
        similarity = _finite(align_row.get("similarity"))
        t_align = cfg.t_align_clip if tier == "A" else cfg.t_align
        if similarity is None or similarity < t_align:
            reject_reasons.append("alignment")
        forced = str(align_row.get("forced_text") or "").strip()
        if (
            align_row.get("keep") is True
            and similarity is not None
            and similarity >= t_align
            and forced
        ):
            text = forced

    # H6.5 Chinese text normalized to simplified Chinese
    if text and lang == "zh":
        text = _to_simplified(text)

    # H7 text quality
    if text:
        tq = text_quality.evaluate_text_quality(text, lang or "zh", allow_en=False)
        if not tq.keep:
            reject_reasons.append("text_quality")
    else:
        reject_reasons.append("text_quality")

    # de-dupe reasons preserve order
    seen: set[str] = set()
    uniq_reasons: list[str] = []
    for r in reject_reasons:
        if r not in seen:
            seen.add(r)
            uniq_reasons.append(r)

    components = _soft_components(
        similarity=similarity,
        lang_prob=lang_prob,
        avg_logprob=avg_logprob,
        speaker_score=speaker_score,
        text=text,
        lang=lang or "zh",
        tier=tier,
        risks=risks,
        duration=duration,
        cfg=cfg,
    )
    score = _weighted_score(components, cfg)

    if uniq_reasons:
        decision_out = "reject"
    elif score < cfg.s_min:
        decision_out = "borderline"
    else:
        decision_out = "candidate"

    return {
        "path": path,
        "source_bvid": extract_bvid(path),
        "tier": tier,
        "lang": lang,
        "text": text,
        "decision": decision_out,
        "score": round(score, 4),
        "components": components,
        "reject_reasons": uniq_reasons,
        "metrics": {
            "duration": duration,
            "similarity": similarity,
            "speaker_score": speaker_score,
            "score_lower": score_lower,
            "language_probability": lang_prob,
            "avg_logprob": avg_logprob,
            "rms_db": rms,
        },
        "risks": risks,
    }


def build_scorecard_items(
    *,
    filter_rows: list[Mapping[str, Any]],
    speaker_rows: list[Mapping[str, Any]],
    asr_rows: list[Mapping[str, Any]],
    align_rows: list[Mapping[str, Any]],
    bvid_to_meta: Mapping[str, Mapping[str, Any]],
    cfg: DatasetHQConfig,
    canonical_path=None,
) -> list[dict[str, Any]]:
    def key(p: str) -> str:
        if canonical_path is not None:
            return str(canonical_path(p))
        return str(Path(p).resolve()) if p else p

    def index_by_path(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        out: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            p = row.get("path")
            if p:
                out[key(str(p))] = row
        return out

    f_map = index_by_path(list(filter_rows))
    s_map = index_by_path(list(speaker_rows))
    a_map = index_by_path(list(asr_rows))
    al_map = index_by_path(list(align_rows))
    paths = set(f_map) | set(s_map) | set(a_map) | set(al_map)

    items: list[dict[str, Any]] = []
    for p in sorted(paths):
        # prefer original path string from any row
        orig = (
            str((a_map.get(p) or al_map.get(p) or f_map.get(p) or s_map.get(p) or {}).get("path") or p)
        )
        bvid = extract_bvid(orig)
        meta = bvid_to_meta.get(bvid) or {}
        tier = str(meta.get("tier") or "C")
        risks = list(meta.get("risk") or meta.get("risks") or [])
        items.append(
            score_one(
                path=orig,
                filter_row=f_map.get(p),
                speaker_row=s_map.get(p),
                asr_row=a_map.get(p),
                align_row=al_map.get(p),
                tier=tier,
                risks=risks,
                cfg=cfg,
            )
        )
    return items
