from pathlib import Path
import sys

import pytest

from scripts import train_gpt_sovits as train


def _write_training_inputs(exp_dir: Path, list_path: Path, *, include_sv: bool) -> None:
    names = ["001_zh.wav", "002_jp.wav"]
    list_path.write_text(
        "\n".join(f"audio/{name}|speaker|ZH|text" for name in names) + "\n",
        encoding="utf-8",
    )
    (exp_dir / "2-name2text.txt").write_text(
        "\n".join(f"{name}\tphones\t[1]\ttext" for name in names) + "\n",
        encoding="utf-8",
    )
    for dirname in ("4-cnhubert", "5-wav32k", "7-sv_cn"):
        (exp_dir / dirname).mkdir()
    for name in names:
        (exp_dir / "4-cnhubert" / f"{name}.pt").touch()
        (exp_dir / "5-wav32k" / name).touch()
        if include_sv:
            (exp_dir / "7-sv_cn" / f"{name}.pt").touch()


def test_validate_training_features_rejects_missing_v2pro_speaker_embeddings(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    list_path = tmp_path / "annotation.list"
    _write_training_inputs(exp_dir, list_path, include_sv=False)

    with pytest.raises(RuntimeError, match=r"7-sv_cn.*expected=2.*ready=0"):
        train.validate_training_features(exp_dir, list_path, "v2Pro")


def test_validate_training_features_accepts_complete_v2pro_inputs(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    list_path = tmp_path / "annotation.list"
    _write_training_inputs(exp_dir, list_path, include_sv=True)

    summary = train.validate_training_features(exp_dir, list_path, "v2Pro")

    assert summary["expected"] == 2
    assert summary["ready"] == 2
    assert summary["sv"] == 2


def test_default_dataset_paths_use_project_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_gpt_sovits.py"])

    args = train.parse_args()

    assert args.list_path == train.DEFAULT_DATASET_DIR / "annotation.list"
    assert args.wav_dir == train.DEFAULT_DATASET_DIR / "audio"
    assert args.batch_size_s2 == 12
    assert args.batch_size_s1 == 8
