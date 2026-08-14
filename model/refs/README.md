# 说话语气参考音库

闭集六类：`日常`（默认）/ `元气` / `温柔` / `俏皮` / `倔强` / `惊讶`。

| 标签 | 听感 | 主 ref |
|------|------|--------|
| 日常 | 一般闲聊、平叙 | `model/huayin-ref.wav`（交付听感锁定） |
| 元气 | 更轻快、上扬 | `model/refs/yuanqi_p.wav` |
| 温柔 | 软、短、真 | `model/refs/wenrou_p.wav` |
| 俏皮 | 欠揍、互怼 | `model/refs/qiaopi_p.wav` |
| 倔强 | 硬气反驳 | `model/refs/juejiang_p.wav` |
| 惊讶 | 短促愣/上扬 | `model/refs/jingya_p.wav` |

- 清单：`speaking_style_refs.json`
- 每类 1 主 + ≤2 备选；运行时取 primary
- 更换某类主 ref：改 wav + 同步 json 的 `text`/`source`，并做听感验收
