# Huayin 交付权重

**配方：** [`configs/huayin_precision.yaml`](../configs/huayin_precision.yaml)

| 文件 | 角色 | 说明 |
|------|------|------|
| `huayin-gpt.ckpt` | S1 语义模型 | 验证最优（原 e5，val top3 acc 0.250） |
| `huayin-sovits.pth` | S2 声学模型 | 音色（原 e12 终盘） |
| `huayin-ref.wav` | 参考音频 | 默认「日常」说话语气主 ref |
| `refs/` | 多说话语气参考音库 | 见 `refs/speaking_style_refs.json`（日常/元气/温柔/俏皮/倔强/惊讶） |

```bash
python scripts/run_huayin_api.py --python /home/mtr/miniconda3/envs/gptsovits/bin/python
python scripts/test_huayin_tts.py "你好，今天也要加油。"
```

运行时仍依赖 `/home/mtr/tt/GPT-SoVITS` 与官方 `pretrained_models/`。
