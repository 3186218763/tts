"""No-human high-precision transcript consensus (multi-ASR agreement).

Goal: approximate research-grade label accuracy (~98%) without human listen
by **precision filtering**: keep only clips where independent ASR systems
strongly agree; drop the rest.

Rationale
---------
A single ASR on colloquial live speech cannot guarantee 98% CER on all clips.
When two *heterogeneous* systems (e.g. Whisper + FunASR Paraformer) agree
after normalization, residual error rate on the kept set is typically far
lower than either system alone (correlated errors still possible, but much
rarer). Same-family multi-decode (Whisper temp 0 vs 0.6) is weaker evidence
and is treated as a secondary signal only.

Decisions (default thresholds)
------------------------------
- keep_strict: sim(A,B) >= 0.92  -> keep, text = pick_text(A,B)
- keep_soft:   0.85 <= sim < 0.92 and a third vote agrees (>=0.92 with A or B)
- drop:        otherwise

sim is SequenceMatcher ratio on compact text (zh simplified, punctuation
stripped). This is a *precision proxy*, not a human MOS score.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

_PUNCT_RE = re.compile(
    r"[\s"
    r"，。！？、；：…,.!?;:·~～\-_/\\|@#$%^&*+=<>"
    r"()（）「」『』【】\[\]{}\"'“”‘’"
    r"]+"
)


def compact_text(text: str, *, lang: str = "zh") -> str:
    """Normalize for agreement comparison."""
    raw = text or ""
    lang = (lang or "zh").lower()
    if lang in {"zh", "cn", "zho", "chinese"}:
        try:
            from zhconv import convert

            raw = convert(raw, "zh-cn")
        except Exception:
            pass
    return _PUNCT_RE.sub("", raw)


def similarity(a: str, b: str, *, lang: str = "zh") -> float:
    x, y = compact_text(a, lang=lang), compact_text(b, lang=lang)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return SequenceMatcher(None, x, y).ratio()


def char_error_rate(ref: str, hyp: str, *, lang: str = "zh") -> float:
    """CER on compact text (0 = identical)."""
    r, h = compact_text(ref, lang=lang), compact_text(hyp, lang=lang)
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    # Levenshtein via SequenceMatcher opcodes
    sm = SequenceMatcher(None, r, h)
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            edits += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            edits += i2 - i1
        elif tag == "insert":
            edits += j2 - j1
    return edits / max(len(r), 1)


@dataclass(frozen=True)
class ConsensusThresholds:
    """High-precision defaults: prefer fewer clean clips over more dirty ones."""

    keep_strict: float = 0.92
    keep_soft: float = 0.85
    third_agree: float = 0.92
    min_chars: int = 2
    # Prefer primary engine when sims equal; "primary" is usually Whisper path.
    prefer: str = "primary"


@dataclass(frozen=True)
class ConsensusDecision:
    decision: str  # keep | drop
    tier: str  # strict | soft_third | drop_low_sim | drop_empty | drop_no_pair
    text: str | None
    sim_ab: float | None
    sim_ac: float | None
    sim_bc: float | None
    cer_ab: float | None
    source: str | None  # which engine text was chosen

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pick_text(
    primary: str,
    secondary: str,
    *,
    lang: str = "zh",
    prefer: str = "primary",
    primary_logprob: float | None = None,
    secondary_logprob: float | None = None,
) -> tuple[str, str]:
    """Choose final text when A and B agree enough.

    Prefer longer compact content when lengths differ a lot (avoids truncated
    ASR); otherwise prefer primary, or better avg_logprob if both provided.
    """
    a, b = primary or "", secondary or ""
    ca, cb = compact_text(a, lang=lang), compact_text(b, lang=lang)
    if not ca and cb:
        return b, "secondary"
    if not cb and ca:
        return a, "primary"
    if not ca and not cb:
        return a, "primary"

    # large length gap: keep the longer one (truncated partner)
    if len(ca) >= len(cb) * 1.25 and len(ca) - len(cb) >= 3:
        return a, "primary_longer"
    if len(cb) >= len(ca) * 1.25 and len(cb) - len(ca) >= 3:
        return b, "secondary_longer"

    if primary_logprob is not None and secondary_logprob is not None:
        # whisper: higher (less negative) logprob is better
        if primary_logprob > secondary_logprob + 0.05:
            return a, "primary_logprob"
        if secondary_logprob > primary_logprob + 0.05:
            return b, "secondary_logprob"

    if prefer == "secondary":
        return b, "secondary"
    return a, "primary"


def adjudicate(
    primary: str | None,
    secondary: str | None,
    *,
    third: str | None = None,
    lang: str = "zh",
    thresholds: ConsensusThresholds | None = None,
    primary_logprob: float | None = None,
    secondary_logprob: float | None = None,
) -> ConsensusDecision:
    """Adjudicate multi-ASR outputs into keep/drop + final text."""
    thr = thresholds or ConsensusThresholds()
    p = (primary or "").strip()
    s = (secondary or "").strip()
    t = (third or "").strip() if third is not None else None

    if not compact_text(p, lang=lang) and not compact_text(s, lang=lang):
        return ConsensusDecision(
            decision="drop",
            tier="drop_empty",
            text=None,
            sim_ab=None,
            sim_ac=None,
            sim_bc=None,
            cer_ab=None,
            source=None,
        )

    if not compact_text(p, lang=lang) or not compact_text(s, lang=lang):
        # only one engine produced text -> not enough evidence for 98% proxy
        return ConsensusDecision(
            decision="drop",
            tier="drop_no_pair",
            text=None,
            sim_ab=0.0,
            sim_ac=None,
            sim_bc=None,
            cer_ab=None,
            source=None,
        )

    sim_ab = similarity(p, s, lang=lang)
    cer_ab = char_error_rate(p, s, lang=lang)
    sim_ac = similarity(p, t, lang=lang) if t is not None else None
    sim_bc = similarity(s, t, lang=lang) if t is not None else None

    final_ca = compact_text(p, lang=lang)
    if len(final_ca) < thr.min_chars and len(compact_text(s, lang=lang)) < thr.min_chars:
        return ConsensusDecision(
            decision="drop",
            tier="drop_empty",
            text=None,
            sim_ab=sim_ab,
            sim_ac=sim_ac,
            sim_bc=sim_bc,
            cer_ab=cer_ab,
            source=None,
        )

    if sim_ab >= thr.keep_strict:
        text, src = pick_text(
            p,
            s,
            lang=lang,
            prefer=thr.prefer,
            primary_logprob=primary_logprob,
            secondary_logprob=secondary_logprob,
        )
        return ConsensusDecision(
            decision="keep",
            tier="strict",
            text=text,
            sim_ab=sim_ab,
            sim_ac=sim_ac,
            sim_bc=sim_bc,
            cer_ab=cer_ab,
            source=src,
        )

    if thr.keep_soft <= sim_ab < thr.keep_strict and t is not None:
        # third vote breaks the soft band
        if sim_ac is not None and sim_ac >= thr.third_agree:
            return ConsensusDecision(
                decision="keep",
                tier="soft_third",
                text=p,
                sim_ab=sim_ab,
                sim_ac=sim_ac,
                sim_bc=sim_bc,
                cer_ab=cer_ab,
                source="primary_third",
            )
        if sim_bc is not None and sim_bc >= thr.third_agree:
            return ConsensusDecision(
                decision="keep",
                tier="soft_third",
                text=s,
                sim_ab=sim_ab,
                sim_ac=sim_ac,
                sim_bc=sim_bc,
                cer_ab=cer_ab,
                source="secondary_third",
            )

    return ConsensusDecision(
        decision="drop",
        tier="drop_low_sim",
        text=None,
        sim_ab=sim_ab,
        sim_ac=sim_ac,
        sim_bc=sim_bc,
        cer_ab=cer_ab,
        source=None,
    )


def summarize_decisions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    keep = [r for r in rows if r.get("decision") == "keep"]
    drop = [r for r in rows if r.get("decision") != "keep"]
    tiers: dict[str, int] = {}
    for r in rows:
        t = str(r.get("tier") or "unknown")
        tiers[t] = tiers.get(t, 0) + 1
    sims = [float(r["sim_ab"]) for r in keep if r.get("sim_ab") is not None]
    identical = sum(
        1
        for r in keep
        if r.get("sim_ab") is not None and float(r["sim_ab"]) >= 0.999
    )
    return {
        "n_total": n,
        "n_keep": len(keep),
        "n_drop": len(drop),
        "keep_rate": (len(keep) / n) if n else 0.0,
        "tiers": tiers,
        "keep_identical_rate": (identical / len(keep)) if keep else 0.0,
        "keep_sim_mean": (sum(sims) / len(sims)) if sims else None,
        "keep_sim_min": min(sims) if sims else None,
        # Proxy: fraction of keeps with sim>=0.95 (stricter than gate)
        "keep_sim_ge_0_95_rate": (
            sum(1 for s in sims if s >= 0.95) / len(sims) if sims else None
        ),
        "note": (
            "keep_identical_rate / keep_sim_* are precision *proxies* from "
            "inter-ASR agreement, not human-measured CER."
        ),
    }
