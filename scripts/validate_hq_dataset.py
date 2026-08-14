"""CLI: validate HQ dataset + export stratified listen sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dataset_hq_config import load_dataset_hq_config  # noqa: E402
from validate_hq_lib import (  # noqa: E402
    build_listen_sample,
    evaluate_listen_results,
    export_listen_pack,
    run_auto_validation,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate HQ dataset")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset_hq.yaml")
    p.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "dataset_hq")
    p.add_argument(
        "--scorecard",
        type=Path,
        default=PROJECT_ROOT / "data" / "quality_scorecard.json",
    )
    p.add_argument("--export-listen", action="store_true")
    p.add_argument("--with-listen", action="store_true")
    p.add_argument(
        "--listen-results",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/listen_results.json",
    )
    args = p.parse_args(argv)

    cfg = load_dataset_hq_config(args.config)
    ds = Path(args.dataset_dir)
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((ds / "stats.json").read_text(encoding="utf-8"))
    annotation_lines = [
        ln for ln in (ds / "annotation.list").read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    wav_names = sorted(p.name for p in (ds / "audio").glob("*.wav")) if (ds / "audio").exists() else []

    sc_doc = {}
    if args.scorecard.exists():
        sc_doc = json.loads(args.scorecard.read_text(encoding="utf-8"))
    sc_items = sc_doc.get("items", sc_doc if isinstance(sc_doc, list) else [])
    scorecard_by_path = {str(it.get("path")): it for it in sc_items if it.get("path")}

    report = run_auto_validation(
        manifest, stats, scorecard_by_path, annotation_lines, wav_names, cfg
    )

    if args.export_listen or args.with_listen:
        sample = build_listen_sample(
            manifest,
            scorecard_by_path,
            n=cfg.listen_sample_size,
            seed=42,
            cfg=cfg,
        )
        export_listen_pack(sample, ds / "listen_sample")
        (ds / "listen_sample" / "sample.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if args.with_listen:
        listen_path = args.listen_results or (ds / "listen_results.json")
        if not listen_path.exists():
            report["verdict"] = "pending_listen"
            report["ready_for_train"] = False
            report["listen"] = {"pass": False, "detail": f"missing {listen_path}"}
        else:
            rows = json.loads(listen_path.read_text(encoding="utf-8"))
            listen = evaluate_listen_results(rows, cfg)
            report["listen"] = listen
            if report["auto_pass"] and listen.get("pass"):
                report["verdict"] = "pass"
                report["ready_for_train"] = True
            elif report["auto_pass"]:
                report["verdict"] = "fail"
                report["ready_for_train"] = False
            else:
                report["ready_for_train"] = False

    out = ds / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ds / "validation_report.md"
    lines = [
        f"# HQ validation",
        f"",
        f"- verdict: **{report['verdict']}**",
        f"- auto_pass: {report['auto_pass']}",
        f"- ready_for_train: {report.get('ready_for_train')}",
        f"",
        f"| check | pass | detail |",
        f"|-------|------|--------|",
    ]
    for c in report["checks"]:
        lines.append(f"| {c['id']} | {c['pass']} | {c['detail']} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # update stats ready flag
    stats = dict(stats)
    stats["ready_for_train"] = bool(report.get("ready_for_train"))
    (ds / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"verdict={report['verdict']} auto_pass={report['auto_pass']} → {out}")
    return 0 if report["verdict"] in {"pass", "pending_listen"} and report["auto_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
