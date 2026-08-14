import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_download import (
    load_plan_items,
    filter_plan_for_download,
    mark_plan_status,
    append_training_assets,
)


def test_load_plan_items_supports_wrapped_doc(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(
        '{"version":1,"items":[{"bvid":"BV1","duration":10,"priority_score":1,"status":"planned"}]}',
        encoding="utf-8",
    )
    items = load_plan_items(p)
    assert items[0]["bvid"] == "BV1"


def test_filter_plan_respects_priority_and_max_hours():
    items = [
        {
            "bvid": "BV1",
            "tier": "B",
            "duration": 3600,
            "priority_score": 100,
            "status": "planned",
        },
        {
            "bvid": "BV2",
            "tier": "A",
            "duration": 120,
            "priority_score": 50,
            "status": "planned",
        },
        {
            "bvid": "BV3",
            "tier": "C",
            "duration": 7200,
            "priority_score": 10,
            "status": "planned",
        },
    ]
    out = filter_plan_for_download(items, max_hours=1.1, existing_bvids=set())
    assert [x["bvid"] for x in out] == ["BV1", "BV2"]


def test_filter_skips_downloaded_and_respects_new_only():
    items = [
        {
            "bvid": "BV1",
            "duration": 100,
            "priority_score": 10,
            "status": "downloaded",
        },
        {
            "bvid": "BV2",
            "duration": 100,
            "priority_score": 9,
            "status": "planned",
        },
    ]
    out = filter_plan_for_download(items, max_hours=0, existing_bvids=set())
    assert [x["bvid"] for x in out] == ["BV2"]


def test_mark_plan_status():
    items = [{"bvid": "BV1", "status": "planned"}]
    out = mark_plan_status(items, "BV1", "failed")
    assert out[0]["status"] == "failed"


def test_append_training_assets(tmp_path):
    assets = tmp_path / "training_assets.txt"
    assets.write_text("old.wav\n", encoding="utf-8")
    append_training_assets(assets, ["new.wav", "old.wav"])
    lines = assets.read_text(encoding="utf-8").splitlines()
    assert lines == ["old.wav", "new.wav"]
