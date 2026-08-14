"""HQ dataset auto validation V1–V8 and stratified listen sampling."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import dataset_text_quality as text_quality
from dataset_hq_config import DatasetHQConfig
from hq_dataset_lib import normalize_text_key


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _median(values: list[float]) -> float | None:
    return _percentile(values, 50)


def run_auto_validation(
    manifest: list[Mapping[str, Any]],
    stats: Mapping[str, Any],
    scorecard_by_path: Mapping[str, Mapping[str, Any]],
    annotation_lines: list[str],
    wav_names: list[str],
    cfg: DatasetHQConfig,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    # V1 shortfall: quantity gate. When fail_on_zh_shortfall is false this is
    # informational only (dataset size is accepted), and quality gates V2-V8 +
    # listen review carry the verdict.
    shortfall = bool(stats.get("zh_shortfall"))
    v1_pass = not (shortfall and cfg.validation.fail_on_zh_shortfall)
    if shortfall and not cfg.validation.fail_on_zh_shortfall:
        warnings.append(
            "V1: zh_shortfall=True tolerated (fail_on_zh_shortfall=False); "
            f"n_total={stats.get('n_total')} < n_target={stats.get('n_target')}"
        )
    checks.append(
        {
            "id": "V1",
            "pass": v1_pass,
            "detail": f"n_total={stats.get('n_total')} zh_shortfall={shortfall}",
        }
    )

    # V2 ratio
    n_total = int(stats.get("n_total") or 0)
    n_zh = int(stats.get("n_zh") or 0)
    ratio = (n_zh / n_total) if n_total else 0.0
    v2_pass = n_total > 0 and ratio >= cfg.zh_ratio - 1e-9
    checks.append(
        {
            "id": "V2",
            "pass": v2_pass,
            "detail": f"zh_ratio={ratio:.4f} need>={cfg.zh_ratio}",
        }
    )

    # V3 scorecard consistency
    v3_bad = []
    for m in manifest:
        p = str(m.get("path") or "")
        sc = scorecard_by_path.get(p)
        if sc is None:
            v3_bad.append(p)
            continue
        if sc.get("decision") != "candidate" or float(sc.get("score") or 0) < cfg.s_min:
            v3_bad.append(p)
    checks.append(
        {
            "id": "V3",
            "pass": len(v3_bad) == 0,
            "detail": f"bad={len(v3_bad)}",
        }
    )

    # V4 alignment stats
    sims = []
    for m in manifest:
        sc = scorecard_by_path.get(str(m.get("path") or ""), {})
        metrics = sc.get("metrics") or {}
        sim = metrics.get("similarity")
        if sim is not None:
            sims.append(float(sim))
    med = _median(sims)
    p10 = _percentile(sims, 10)
    v4_pass = (
        med is not None
        and p10 is not None
        and med >= cfg.validation.align_median_min
        and p10 >= cfg.validation.align_p10_min
    )
    if not sims:
        v4_pass = True
        warnings.append("V4 skipped: no similarity metrics")
    checks.append(
        {
            "id": "V4",
            "pass": v4_pass,
            "detail": f"median={med} p10={p10}",
        }
    )

    # V5 speaker p5 margin
    sel_sl = []
    cand_sl = []
    for sc in scorecard_by_path.values():
        sl = (sc.get("metrics") or {}).get("score_lower")
        if sl is None:
            continue
        if sc.get("decision") == "candidate":
            cand_sl.append(float(sl))
    for m in manifest:
        sc = scorecard_by_path.get(str(m.get("path") or ""), {})
        sl = (sc.get("metrics") or {}).get("score_lower")
        if sl is not None:
            sel_sl.append(float(sl))
    if sel_sl and cand_sl:
        sel_p5 = _percentile(sel_sl, 5)
        cand_p5 = _percentile(cand_sl, 5)
        v5_pass = sel_p5 is not None and cand_p5 is not None and (
            sel_p5 >= cand_p5 - cfg.validation.speaker_p5_margin
        )
        detail = f"sel_p5={sel_p5} cand_p5={cand_p5}"
    else:
        v5_pass = True
        warnings.append("V5 skipped: missing score_lower")
        detail = "skipped"
    checks.append({"id": "V5", "pass": v5_pass, "detail": detail})

    # V6 source cap
    by_lang: dict[str, list[str]] = {"zh": [], "ja": []}
    for m in manifest:
        lang = text_quality.normalize_lang(m.get("lang"))
        if lang in by_lang:
            by_lang[lang].append(str(m.get("source_bvid") or ""))
    v6_ok = True
    for lang, bvids in by_lang.items():
        n_lang = len(bvids)
        if n_lang == 0:
            continue
        cap = max(1, int(math.floor(cfg.max_source_share * n_lang)))
        counts = Counter(bvids)
        if any(c > cap for c in counts.values()):
            v6_ok = False
    checks.append({"id": "V6", "pass": v6_ok, "detail": "source share caps"})

    # V7 text quality + dupes
    v7_ok = True
    keys = []
    for m in manifest:
        text = str(m.get("text") or "")
        lang = text_quality.normalize_lang(m.get("lang"))
        tq = text_quality.evaluate_text_quality(text, lang, allow_en=False)
        if not tq.keep:
            v7_ok = False
        keys.append(normalize_text_key(text))
    if len(keys) != len(set(keys)):
        v7_ok = False
    checks.append({"id": "V7", "pass": v7_ok, "detail": "text_quality+dedupe"})

    # V8 format
    tags_ok = True
    for line in annotation_lines:
        parts = line.split("|")
        if len(parts) < 4 or parts[2] not in {"ZH", "JP"}:
            tags_ok = False
            break
    v8_pass = (
        len(annotation_lines) == len(wav_names) == len(manifest) and tags_ok
    )
    checks.append(
        {
            "id": "V8",
            "pass": v8_pass,
            "detail": f"ann={len(annotation_lines)} wav={len(wav_names)} man={len(manifest)}",
        }
    )

    auto_pass = all(c["pass"] for c in checks)
    failed = [c for c in checks if not c["pass"]]
    verdict = "pending_listen" if auto_pass else "fail"
    return {
        "checks": checks,
        "failed_checks": failed,
        "auto_pass": auto_pass,
        "verdict": verdict,
        "ready_for_train": False,
        "warnings": warnings,
    }


def build_listen_sample(
    manifest: list[Mapping[str, Any]],
    scorecard_by_path: Mapping[str, Mapping[str, Any]],
    *,
    n: int,
    seed: int,
    cfg: DatasetHQConfig,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    layers_quota = {
        "L1": int(n * 0.25),
        "L2": int(n * 0.25),
        "L3": int(n * 0.20),
        "L4": int(n * 0.15),
        "L5": int(n * 0.15),
    }
    assigned = sum(layers_quota.values())
    layers_quota["L1"] += max(0, n - assigned)

    def meta(m):
        sc = scorecard_by_path.get(str(m.get("path") or ""), {})
        return sc

    l1 = [m for m in manifest if m.get("tier") == "B" and text_quality.normalize_lang(m.get("lang")) == "zh"]
    l2 = [m for m in manifest if m.get("tier") == "A"]
    l3 = []
    for m in manifest:
        sc = meta(m)
        sim = (sc.get("metrics") or {}).get("similarity")
        if sim is None:
            continue
        t0 = cfg.t_align
        if t0 <= float(sim) <= t0 + 0.1:
            l3.append(m)
    scores = sorted(
        manifest,
        key=lambda m: (
            float(m.get("score") or meta(m).get("score") or 0),
            str(m.get("path") or ""),
        ),
    )
    cutoff = max(1, int(math.ceil(0.10 * len(scores)))) if scores else 0
    l4 = scores[:cutoff]
    l5 = [m for m in manifest if text_quality.normalize_lang(m.get("lang")) == "ja"]

    pools = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5}
    sample: list[dict[str, Any]] = []
    used: set[str] = set()
    for layer, quota in layers_quota.items():
        pool = [m for m in pools[layer] if str(m.get("path")) not in used]
        rng.shuffle(pool)
        for m in pool[:quota]:
            p = str(m.get("path"))
            used.add(p)
            sc = meta(m)
            sample.append(
                {
                    "id": f"{layer}_{len(sample):03d}",
                    "layer": layer,
                    "path": p,
                    "text": m.get("text") or sc.get("text"),
                    "lang": m.get("lang"),
                    "tier": m.get("tier"),
                    "score": m.get("score") or sc.get("score"),
                    "label": "",
                }
            )
    return sample


def evaluate_listen_results(
    rows: list[Mapping[str, Any]],
    cfg: DatasetHQConfig,
) -> dict[str, Any]:
    labeled = [r for r in rows if str(r.get("label") or "").strip()]
    n = len(labeled)
    if n == 0:
        return {"pass": False, "ok_rate": 0.0, "detail": "no labels"}

    ok = sum(1 for r in labeled if str(r.get("label")).strip() == "OK")
    ok_rate = ok / n

    def layer_ok(layer: str) -> float | None:
        xs = [r for r in labeled if r.get("layer") == layer]
        if not xs:
            return None
        return sum(1 for r in xs if str(r.get("label")).strip() == "OK") / len(xs)

    def critical_rate(layer: str | None = None) -> float:
        xs = labeled if layer is None else [r for r in labeled if r.get("layer") == layer]
        if not xs:
            return 0.0
        crit = {"非本人", "文不对题"}
        return sum(1 for r in xs if str(r.get("label")).strip() in crit) / len(xs)

    l2 = layer_ok("L2")
    l3 = layer_ok("L3")
    coverage = n / max(len(rows), 1)
    passed = coverage >= 0.95 and ok_rate >= cfg.validation.listen_ok_rate
    if l2 is not None and l2 < cfg.validation.listen_l2_ok_rate:
        passed = False
    if l3 is not None and l3 < cfg.validation.listen_l3_ok_rate:
        passed = False
    # any layer critical >= threshold
    for layer in {str(r.get("layer")) for r in labeled}:
        if critical_rate(layer) >= cfg.validation.listen_critical_fail_rate:
            passed = False

    return {
        "pass": passed,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "l2_ok_rate": l2,
        "l3_ok_rate": l3,
        "critical_rates": {
            layer: critical_rate(layer)
            for layer in sorted({str(r.get("layer")) for r in labeled})
        },
    }


def export_listen_pack(
    sample: list[Mapping[str, Any]],
    out_dir: Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in sample:
        lines.append(
            "\t".join(
                [
                    str(row.get("id")),
                    str(row.get("layer")),
                    str(row.get("path")),
                    str(row.get("text") or "").replace("\t", " "),
                    str(row.get("label") or ""),
                ]
            )
        )
        src = Path(str(row.get("path") or ""))
        dest = out_dir / f"{row.get('id')}.wav"
        if src.exists() and not dest.exists():
            try:
                dest.write_bytes(src.read_bytes())
            except OSError:
                pass
    manifest = out_dir / "manifest.tsv"
    manifest.write_text(
        "id\tlayer\tpath\ttext\tlabel\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return manifest
