import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import merge_persona_fewshot as merge  # noqa: E402


def test_merge_only_approved_and_edited(tmp_path: Path):
    few = {
        "schema_version": 1,
        "note": "x",
        "injection_strategy": "y",
        "asr_pool_enabled": False,
        "official_comments": [{"text": "阿里嘎多！红豆泥阿里嘎多", "lang": "zh", "scenes": ["感谢"]}],
        "fan_pool": [],
        "asr_pool": [{"text": "旧的未审", "lang": "zh", "scenes": [], "source": "old.wav"}],
    }
    queue = {
        "schema_version": 1,
        "items": [
            {
                "id": "Q1",
                "text": "坏的",
                "final_text": "尼们好呀",
                "lang": "zh",
                "source": "a.wav",
                "status": "edited",
                "scenes": ["问候"],
                "era_tag": "peak",
            },
            {
                "id": "Q2",
                "text": "找到蟑螂了",
                "final_text": "",
                "lang": "zh",
                "source": "b.wav",
                "status": "approved",
                "scenes": ["屑"],
                "era_tag": "peak",
            },
            {
                "id": "Q3",
                "text": "不要",
                "final_text": "",
                "lang": "zh",
                "source": "c.wav",
                "status": "rejected",
                "scenes": ["问候"],
                "era_tag": "peak",
            },
            {
                "id": "Q4",
                "text": "毕业了哈哈",
                "final_text": "",
                "lang": "zh",
                "source": "d.wav",
                "status": "approved",
                "scenes": ["告别"],
                "era_tag": "late",
            },
        ],
    }
    few_path = tmp_path / "few.json"
    q_path = tmp_path / "q.json"
    few_path.write_text(json.dumps(few, ensure_ascii=False), encoding="utf-8")
    q_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.json"
    stats = merge.merge_fewshot(few_path, q_path, out, enable_if_ready=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    texts = [x["text"] for x in data["asr_pool"]]
    assert "尼们好呀" in texts
    assert "找到蟑螂了" in texts
    assert "坏的" not in texts
    assert "不要" not in texts
    assert "旧的未审" not in texts
    assert data["asr_pool_enabled"] is False
    late = [x for x in data["asr_pool"] if x.get("era_tag") == "late"]
    assert len(late) == 1
    assert stats["approved_count"] == 3


def test_merge_preserves_asr_pool_enabled_when_not_enable_if_ready(tmp_path: Path):
    """enable_if_ready=False 时保留 fewshot 原有 asr_pool_enabled，不强制 false。"""
    few = {
        "schema_version": 1,
        "note": "x",
        "injection_strategy": "y",
        "asr_pool_enabled": True,
        "official_comments": [],
        "fan_pool": [],
        "asr_pool": [],
    }
    queue = {
        "schema_version": 1,
        "items": [
            {
                "id": "Q1",
                "text": "尼们好呀",
                "final_text": "",
                "lang": "zh",
                "source": "a.wav",
                "status": "approved",
                "scenes": ["问候"],
                "era_tag": "peak",
            },
        ],
    }
    few_path = tmp_path / "few.json"
    q_path = tmp_path / "q.json"
    few_path.write_text(json.dumps(few, ensure_ascii=False), encoding="utf-8")
    q_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.json"
    stats = merge.merge_fewshot(few_path, q_path, out, enable_if_ready=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["asr_pool_enabled"] is True
    assert stats["asr_pool_enabled"] is True


def test_merge_enable_if_ready_true_when_pool_meets_threshold(tmp_path: Path):
    """enable_if_ready=True 且池达标 → asr_pool_enabled True。"""
    few = {
        "schema_version": 1,
        "note": "x",
        "injection_strategy": "y",
        "asr_pool_enabled": False,
        "official_comments": [],
        "fan_pool": [],
        "asr_pool": [],
    }
    # 总数 ≥40，且 问候/夸赞/自嘲/食物/游戏 各 ≥3
    required = ("问候", "夸赞", "自嘲", "食物", "游戏")
    items = []
    n = 0
    for scene in required:
        for i in range(3):
            n += 1
            items.append(
                {
                    "id": f"R{n}",
                    "text": f"台词{n}",
                    "final_text": "",
                    "lang": "zh",
                    "source": f"r{n}.wav",
                    "status": "approved",
                    "scenes": [scene],
                    "era_tag": "peak",
                }
            )
    # 补到 ≥40
    while n < 40:
        n += 1
        items.append(
            {
                "id": f"R{n}",
                "text": f"台词{n}",
                "final_text": "",
                "lang": "zh",
                "source": f"r{n}.wav",
                "status": "approved",
                "scenes": ["屑"],
                "era_tag": "peak",
            }
        )
    queue = {"schema_version": 1, "items": items}
    few_path = tmp_path / "few.json"
    q_path = tmp_path / "q.json"
    few_path.write_text(json.dumps(few, ensure_ascii=False), encoding="utf-8")
    q_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.json"
    stats = merge.merge_fewshot(few_path, q_path, out, enable_if_ready=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["asr_pool_enabled"] is True
    assert stats["asr_pool_enabled"] is True
    assert stats["pool_size"] == 40
